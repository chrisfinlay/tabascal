from tabascal.telescope import Telescope
from tabascal.coordinates import (
    radec_to_lmn,
    radec_to_XYZ,
    ENU_to_UVW,
    GEO_to_XYZ,
    orbit,
)
from tabascal.interferometry import rfi_vis, astro_vis
from scipy.special import jv
import jax.numpy as jnp
from jax import vmap, random
from jax.interpreters.xla import _DeviceArray
import sys
from jax.config import config

config.update("jax_enable_x64", True)


class Observation(Telescope):
    """
    Construct an Observation object defining a radio interferometry
    observation.

    Parameters
    ----------
    latitude: float
        Latitude of the telescope.
    longitude: float
        Longitude of the telescope.
    elevation: float
        Elevation of the telescope.
    ra: float
        Right Ascension of the phase centre.
    dec: float
        Declination of the phase centre.
    times: ndarray (n_time,)
        Time centroids of each data point.
    freqs: ndarray (n_freq,)
        Frequency centroids for each observation channel.
    ENU_path: str
        Path to a txt file containing the ENU coordinates of each antenna.
    ENU_array: ndarray (n_ant, 3)
        ENU coordinates of each antenna.
    dish_d: float
        Diameter of each antenna dish.
    random_seed: int
        Random seed to use for random number generator.
    auto_corrs: bool
        Flag to include autocorrelations in simulation.
    n_int_samples: int
        Number of samples per time step which are then averaged. Must be
        large enough to capture time-smearing of RFI sources on longest
        baseline.
    name: str
        Name of the telescope.
    """

    def __init__(
        self,
        latitude: float,
        longitude: float,
        elevation: float,
        ra: float,
        dec: float,
        times: jnp.ndarray,
        freqs: jnp.ndarray,
        ENU_path=None,
        ENU_array=None,
        dish_d=13.965,
        random_seed=0,
        auto_corrs=False,
        n_int_samples=4,
        name="MeerKAT",
    ):
        self.ra = ra
        self.dec = dec
        self.times = times
        self.int_time = jnp.abs(jnp.diff(times)[0])
        self.n_int_samples = n_int_samples
        if n_int_samples > 1:
            self.times_fine = self.int_time / (2 * n_int_samples) + jnp.arange(
                times[0] - self.int_time / 2,
                times[-1] + self.int_time / 2,
                self.int_time / n_int_samples,
            )
        else:
            self.times_fine = times
        self.n_time = len(times)
        self.n_time_fine = len(self.times_fine)
        self.freqs = freqs
        self.n_freq = len(freqs)
        self.dish_d = dish_d
        self.auto_corrs = auto_corrs
        super().__init__(
            latitude,
            longitude,
            elevation,
            ENU_array=ENU_array,
            ENU_path=ENU_path,
            name=name,
        )
        self.n_ant = len(self.ENU)
        self.ants_uvw = ENU_to_UVW(
            self.ENU,
            latitude=self.latitude,
            longitude=self.longitude,
            ra=self.ra,
            dec=self.dec,
            times=self.times_fine,
        )
        self.a1, self.a2 = jnp.triu_indices(self.n_ants, 0 if auto_corrs else 1)
        self.bl_uvw = self.ants_uvw[:, self.a1] - self.ants_uvw[:, self.a2]
        self.n_bl = len(self.a1)
        self.ants_xyz = vmap(GEO_to_XYZ, in_axes=(1, None), out_axes=1)(
            self.GEO_ants[None, ...], self.times_fine
        )
        self.n_ast = 0
        self.n_rfi = 0
        self.n_ter_rfi = 0
        self.vis_ast = jnp.zeros(
            (self.n_time_fine, self.n_bl, self.n_freq), dtype=jnp.complex128
        )
        self.vis_rfi = jnp.zeros(
            (self.n_time_fine, self.n_bl, self.n_freq), dtype=jnp.complex128
        )
        self.gains_bl = jnp.ones(
            (self.n_time_fine, self.n_bl, self.n_freq), dtype=jnp.complex128
        )
        self.key = random.PRNGKey(random_seed)
        self.create_source_dicts()

    def create_source_dicts(self):
        self.ast_I = {}
        self.ast_lmn = {}
        self.ast_radec = {}

        self.rfi_I = {}
        self.rfi_xyz = {}
        self.rfi_orbit = {}
        self.rfi_geo = {}
        self.rfi_ang_sep = {}
        self.rfi_A_app = {}

    def addAstro(self, I: jnp.ndarray, ra: jnp.ndarray, dec: jnp.ndarray):
        """
        Add a set of astronomical sources to the observation.

        Parameters
        ----------
        I: ndarray (n_src, n_freq)
            Intensity of the sources in Jy.
        ra: array (n_src,)
            Right ascension of the sources in degrees.
        dec: array (n_src,)
            Declination of the sources in degrees.
        """
        lmn = radec_to_lmn(ra, dec, jnp.array([self.ra, self.dec]))
        theta = jnp.arcsin(jnp.linalg.norm(lmn[:, :-1], axis=-1))
        I_app = I[:, None, None, :] * self.beam(theta[:, None, None, None]) ** 2
        vis_ast = astro_vis(I_app[:, 0, 0, :].T, self.bl_uvw, lmn, self.freqs)

        n_src = I.shape[0]
        for i in range(n_src):
            self.ast_I.update({self.n_ast: I[i]})
            self.ast_lmn.update({self.n_ast: lmn})
            self.ast_radec.update({self.n_ast: jnp.array([ra, dec])})
            self.n_ast += 1
        self.vis_ast += vis_ast

    def addSatelliteRFI(
        self,
        Pv: jnp.ndarray,
        elevation: jnp.ndarray,
        inclination: jnp.ndarray,
        lon_asc_node: jnp.ndarray,
        periapsis: jnp.ndarray,
        Ga=0.0,
    ):
        """
        Add a satellite-based source of RFI to the observation.

        Parameters
        ----------
        Pv: ndarray (n_src, n_freq)
            Specific Emission Power in W/Hz
        elevation: ndarray (n_src,)
            Elevation/Altitude of the orbit in metres.
        inclination: ndarray (n_src,)
            Inclination angle of the orbit relative to the equatorial plane.
        lon_asc_node: ndarray (n_src,)
            Longitude of the ascending node of the orbit. This is the longitude of
            when the orbit crosses the equator from the south to the north.
        periapsis: ndarray (n_src,)
            Perisapsis of the orbit. This is the angular starting point of the orbit
            at t = 0.
        Ga:
            Emission antenna gain relative to an isotropic radiator in dBi.
        """
        orbit_vmap = vmap(orbit, in_axes=(None, 0, 0, 0, 0))
        rfi_xyz = orbit_vmap(
            self.times_fine, elevation, inclination, lon_asc_node, periapsis
        )
        # rfi_xyz is shape (n_src,n_time_fine,3)
        # self.ants_xyz is shape (n_time_fine,n_ant,3)
        distances = jnp.linalg.norm(
            self.ants_xyz[None, :, :, :] - rfi_xyz[:, :, None, :], axis=-1
        )
        # distances is shape (n_src,n_time_fine,n_ant)
        I = vmap(self.Pv_to_Sv, in_axes=(0, 0, None))(Pv, distances, self.db_to_lin(Ga))
        # I is shape (n_src,n_time_fine,n_ant,n_freq)
        rfi_orbit = jnp.array([elevation, inclination, lon_asc_node, periapsis]).T

        angular_seps = self.angular_separation(rfi_xyz)
        # angular_seps is shape (n_src,n_time_fine,n_ant)
        rfi_A_app = jnp.sqrt(I) * self.beam(angular_seps[:, :, :, None])

        n_src = Pv.shape[0]
        for i in range(n_src):
            self.rfi_I.update({self.n_rfi: I[i]})
            self.rfi_xyz.update({self.n_rfi: rfi_xyz[i]})
            self.rfi_orbit.update({self.n_rfi: rfi_orbit[i]})
            self.rfi_ang_sep.update({self.n_rfi: angular_seps[i]})
            self.rfi_A_app.update({self.n_rfi: rfi_A_app[i]})
            self.n_rfi += 1

        # self.rfi_A_app is shape (n_src,n_time_fine,n_ant,n_freqs)
        # distances is shape (n_src,n_time_fine,n_ant)
        # self.ants_uvw is shape (n_time_fine,n_ant,3)
        vis_rfi = rfi_vis(
            jnp.transpose(rfi_A_app, (1, 2, 3, 0)),
            (jnp.transpose(distances, (1, 2, 0)) - self.ants_uvw[:, :, None, -1]),
            self.freqs,
        )
        self.vis_rfi += vis_rfi

    def addStationaryRFI(
        self,
        Pv: jnp.ndarray,
        latitude: jnp.ndarray,
        longitude: jnp.ndarray,
        elevation: jnp.ndarray,
        Ga=0.0,
    ):
        """
        Add a stationary source of RFI to the observation.

        Parameters
        ----------
        Pv: ndarray (n_src, n_freq)
            Specific Emission Power in W/Hz
        latitude: ndarray (n_src,)
            Geopgraphic latitude of the source in degrees.
        longitude: ndarray (n_src,)
            Geographic longitude of the source in degrees.
        elevation: ndarray (n_src,)
            Elevation/Altitude of the source above sea level in metres.
        Ga:
            Emission antenna gain relative to an isotropic radiator in dBi.
        """
        rfi_geo = jnp.array([latitude, longitude, elevation]).T[:, None, :]
        # rfi_ter_geo is shape (n_src,n_time,3)
        rfi_xyz = vmap(GEO_to_XYZ, in_axes=(0, None))(rfi_geo, self.times_fine)
        # rfi_ter_xyz is shape (n_src,n_time_fine,3)
        # self.ants_xyz is shape (n_time_fine,n_ant,3)
        distances = jnp.linalg.norm(
            self.ants_xyz[None, :, :, :] - rfi_xyz[:, :, None, :], axis=-1
        )
        # distances is shape (n_src,n_time_fine,n_ant)
        I = vmap(self.Pv_to_Sv, in_axes=(0, 0, None))(Pv, distances, self.db_to_lin(Ga))
        # I is shape (n_src,n_time,n_ant,n_freq)

        angular_seps = self.angular_separation(rfi_xyz)
        rfi_A_app = jnp.sqrt(I) * self.beam(angular_seps[:, :, :, None])

        n_src = Pv.shape[0]
        for i in range(n_src):
            self.rfi_I.update({self.n_rfi: I[i]})
            self.rfi_xyz.update({self.n_rfi: rfi_xyz[i]})
            self.rfi_geo.update({self.n_rfi: rfi_geo[i]})
            self.rfi_ang_sep.update({self.n_rfi: angular_seps[i]})
            self.rfi_A_app.update({self.n_rfi: rfi_A_app[i]})
            self.n_rfi += 1

        # self.rfi_A_app is shape (n_src,n_time_fine,n_ant,n_freqs)
        # self.ants_uvw is shape (n_time_fine,n_ant,3)
        vis_rfi = rfi_vis(
            jnp.transpose(rfi_A_app, (1, 2, 3, 0)),
            (jnp.transpose(distances, (1, 2, 0)) - self.ants_uvw[:, :, None, -1]),
            self.freqs,
        )
        self.vis_rfi += vis_rfi

    def addGains(
        self,
        G0_mean: complex,
        G0_std: float,
        Gt_std_amp: float,
        Gt_std_phase: float,
        key=None,
    ):
        """
        Add complex antenna gains to the simulation. Gain amplitudes and phases
        are modelled as linear time-variates. Gains for all antennas at t = 0
        are randomly sampled from a Gaussian described by the G0 parameters.
        The rate of change of both ampltudes and phases are sampled from a zero
        mean Gaussian with standard deviation as provided.

        Parameters
        ----------
        G0_mean: complex
            Mean of Gaussian at t = 0.
        G0_std: float
            Standard deviation of Gaussian at t = 0.
        Gt_std_amp: float
            Standard deviation of Gaussian describing the rate of change in the
            gain amplitudes in 1/seconds.
        Gt_std_phase: float
            Standard deviation of Gaussian describing the rate of change in the
            gain phases in rad/seconds.
        key: jax.random.PRNGKey
            Random number generator key.
        """
        if not isinstance(key, _DeviceArray):
            key = self.key
        G0 = G0_mean * jnp.exp(
            1.0j * jnp.pi * (random.uniform(key, (1, self.n_ants, self.n_freq)) - 0.5)
        )
        key, subkey = random.split(key)
        gains_noise = G0_std * random.normal(key, (2, self.n_ants, self.n_freq))
        key, subkey = random.split(key)
        G0 = G0 + gains_noise[0] + 1.0j * gains_noise[1]

        gains_amp = (
            Gt_std_amp
            * random.normal(key, (1, self.n_ants, self.n_freq))
            * (self.times_fine)[:, None, None]
        )
        key, subkey = random.split(key)
        gains_phase = (
            Gt_std_phase
            * random.normal(key, (1, self.n_ants, self.n_freq))
            * (self.times_fine)[:, None, None]
        )
        self.key, subkey = random.split(key)
        self.gains_ants = G0 + gains_amp * jnp.exp(1.0j * gains_phase)
        self.gains_ants = self.gains_ants.at[:, -1, :].set(
            jnp.abs(self.gains_ants[:, -1, :])
        )
        self.gains_bl = (
            self.gains_ants[:, self.a1, :] * self.gains_ants[:, self.a2, :].conjugate()
        )

    def addNoise(self, noise: float, key=None):
        """
        Add complex gaussian noise to the integrated visibilities. The real and
        imaginary components will each get this level of noise.

        Parameters
        ----------
        noise: float
            Standard deviation of the complex noise.
        key: jax.random.PRNGKey
            Random number generator key.
        """
        if not isinstance(key, _DeviceArray):
            key = self.key
            self_key = True
        else:
            self_key = False
        self.noise_data = noise * random.normal(
            key, shape=(self.n_time, self.n_bl, self.n_freq), dtype=jnp.complex128
        )
        if self_key:
            self.key = random.split(key)[0]
        self.noise = noise
        self.vis_obs += self.noise_data

    def angular_separation(self, rfi_xyz: jnp.ndarray):
        """
        Calculate the angular separation between the pointing direction of each
        antenna and the satellite source.

        Parameters
        ----------
        rfi_xyz: ndarray (n_src, n_time, 3)
            Position of the RFI sources in ECEF reference frame over time. This
            is the same frame as self.ants_xyz.

        Returns
        -------
        angles: ndarray (n_src, n_time, n_ant)
        """
        src_xyz = radec_to_XYZ(self.ra, self.dec)
        ant_to_sat_xyz = rfi_xyz[:, :, None, :] - self.ants_xyz[None, :, :, :]
        costheta = jnp.einsum("i,ljki->ljk", src_xyz, ant_to_sat_xyz) / jnp.linalg.norm(
            ant_to_sat_xyz, axis=-1
        )
        angles = jnp.rad2deg(jnp.arccos(costheta))
        return angles

    def beam(self, theta: jnp.ndarray):
        """
        Calculate the primary beam voltage at a given angular distance from the
        pointing direction. The beam intensity model is the Airy disk as
        defined by the dish diameter. This is the same a the CASA default.

        Parameters
        ----------
        theta: (n_src, n_time, n_ant, n_freq)
            The angular separation between the pointing directiona and the
            source.

        Returns
        -------
        E: ndarray (n_src, n_time, n_ant, n_freq)
        """
        c = 299792458.0
        mask = jnp.where(theta > 90.0, 0, 1)
        theta = jnp.deg2rad(theta)
        x = jnp.where(
            theta == 0.0,
            sys.float_info.epsilon,
            jnp.pi * self.freqs[None, None, None, :] * self.dish_d * jnp.sin(theta) / c,
        )
        return (2 * jv(1, x) / x) * mask

    def Pv_to_Sv(self, Pv: jnp.ndarray, d: jnp.ndarray, Ga: float):
        """
        Convert emission power to received intensity in Jy. Assumes constant
        power across the bandwidth. Calculated from

        .. math::
            Jy = \frac{P G_a}{FSPL \Delta\nu A_e}
               = \frac{P G_a \lambda^2 4\pi}{(4\pi d)^2 \Delta\nu \lambda^2}
               = \frac{P G_a}{4\pi d^2 \Delta \nu}

        Parameters
        ----------
        Pv: ndarray (n_freq,)
            Specific emission power in W/Hz.
        d: ndarray (n_time, n_ant)
            Distances from source to receiving antennas in m.
        G: float
            Emission antenna gain

        Returns
        -------
        Sv: ndarray (n_time, n_ant, n_freq)
            Spectral flux density at the receiving antennas in Jy
        """
        return Pv[None, None, :] * Ga / (4 * jnp.pi * d[:, :, None] ** 2) * 1e26

    def db_to_lin(self, dB: float):
        """
        Convert deciBels to linear units.

        Parameters
        ----------
        dB: float, ndarray
            deciBel value to convert.
        """
        return 10.0 ** (dB / 10.0)

    def calculate_vis(self):
        """
        Calculate the total gain amplified visibilities and average down to the
        originally defined sampling rate.
        """
        self.vis = self.gains_bl * (self.vis_ast + self.vis_rfi)
        self.vis_obs = self.vis.reshape(
            self.n_time, self.n_int_samples, self.n_bl, self.n_freq
        ).mean(axis=1)
