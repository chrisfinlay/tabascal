import jax.numpy as jnp
from jax import random
from jax.config import config
from jax.interpreters.xla import _DeviceArray

import numpy as np

from tabascal.jax.coordinates import (
    ENU_to_UVW,
    ENU_to_GEO,
    GEO_to_XYZ_vmap0,
    GEO_to_XYZ_vmap1,
    orbit_vmap,
    radec_to_lmn,
    angular_separation,
)
from tabascal.jax.interferometry import (
    astro_vis,
    rfi_vis,
    add_noise,
    airy_beam,
    Pv_to_Sv,
    SEFD_to_noise_std,
    int_sample_times,
    generate_gains,
    apply_gains,
    time_avg,
)
from tabascal.utils.tools import beam_size, construct_observation_ds

config.update("jax_enable_x64", True)


class Telescope(object):
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
    ENU_path: str
        Path to a txt file containing the ENU coordinates of each antenna.
    ENU_array: ndarray (n_ant, 3)
        ENU coordinates of each antenna.
    name: str
        Name of the telescope.
    """

    def __init__(
        self,
        latitude: float,
        longitude: float,
        elevation: float,
        ENU_array=None,
        ENU_path=None,
        name=None,
    ):
        self.name = name
        self.latitude = latitude
        self.longitude = longitude
        self.elevation = elevation
        self.GEO = np.array([latitude, longitude, elevation])
        self.ENU_path = None
        self.createArrayENU(ENU_array=ENU_array, ENU_path=ENU_path)
        self.n_ants = len(self.ENU)

    def __str__(self):
        msg = f"""\nTelescope Location
------------------
Latitude : {self.latitude}
Longitude : {self.longitude}
Elevation : {self.elevation}\n"""
        return msg

    def createArrayENU(self, ENU_array=None, ENU_path=None):
        if ENU_array is not None:
            self.ENU = ENU_array
        elif ENU_path is not None:
            self.ENU = np.loadtxt(ENU_path)
        else:
            self.ENU = None
            msg = """Error : East-North-Up coordinates are needed either in an 
                     array or as a csv like file."""
            print(msg)
            return

        self.ENU_path = ENU_path
        self.GEO_ants = ENU_to_GEO(self.GEO, self.ENU)


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
    SEFD: ndarray (n_freq,)
        System Equivalent Flux Density of the telescope over frequency.
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
        SEFD: jnp.ndarray,
        ENU_path=None,
        ENU_array=None,
        dish_d=13.965,
        random_seed=0,
        auto_corrs=False,
        n_int_samples=4,
        name="MeerKAT",
        time_chunk=None,
        freq_chunk=None,
        bl_chunk=None,
    ):
        self.backend = "jax"
        self.ra = ra
        self.dec = dec
        self.times = times
        self.int_time = jnp.abs(jnp.diff(times)[0])
        self.n_int_samples = n_int_samples
        self.times_fine = int_sample_times(times, n_int_samples)
        self.n_time = len(times)
        self.n_time_fine = len(self.times_fine)
        self.freqs = freqs
        self.SEFD = SEFD
        self.chan_width = jnp.diff(freqs)[0] if len(freqs) > 1 else 250e3
        self.n_freq = len(freqs)
        self.noise_std = SEFD_to_noise_std(self.SEFD, self.chan_width, self.int_time)
        self.dish_d = dish_d
        self.fov = beam_size(self.dish_d, self.freqs.max())
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
        self.bl_uvw = self.ants_uvw[:, self.a1, :] - self.ants_uvw[:, self.a2, :]
        self.mag_uvw = jnp.linalg.norm(self.bl_uvw[0], axis=-1)
        self.n_bl = len(self.a1)
        self.ants_xyz = GEO_to_XYZ_vmap1(self.GEO_ants[None, ...], self.times_fine)
        self.syn_bw = beam_size(self.mag_uvw.max(), self.freqs[-1])
        self.n_ast = 0
        self.n_rfi = 0
        self.vis_ast = jnp.zeros(
            (self.n_time_fine, self.n_bl, self.n_freq), dtype=jnp.complex128
        )
        self.vis_rfi = jnp.zeros(
            (self.n_time_fine, self.n_bl, self.n_freq), dtype=jnp.complex128
        )
        self.gains_ants = jnp.ones(
            (self.n_time_fine, self.n_ant, self.n_freq), dtype=jnp.complex128
        )
        self.key = random.PRNGKey(random_seed)

        self.n_ast = 0
        self.n_rfi_satellite = 0
        self.n_rfi_stationary = 0

        self.create_source_dicts()

    def __str__(self):
        msg = """
Observation Details
-------------------
Phase Centre (ra, dec) :  ({ra:.1f}, {dec:.1f}) deg.
Number of antennas :       {n_ant}
Number of baselines :      {n_bl}
Autocorrelations :         {auto_corrs}

Frequency range :         ({freq_min:.0f} - {freq_max:.0f}) MHz
Channel width :            {chan_width:.0f} kHz
Number of channels :       {n_freq}

Observation time :        ({time_min:.0f} - {time_max:.0f}) s
Integration time :         {int_time:.0f} s
Sampling rate :            {sampling_rate:.1f} Hz
Number of time steps :     {n_time}

Source Details
--------------
Number of ast. sources:    {n_ast}
Number of RFI sources:     {n_rfi}
Number of satellite RFI :  {n_sat}
Number of stationary RFI : {n_stat}"""

        params = {
            "ra": self.ra,
            "dec": self.dec,
            "n_ant": self.n_ant,
            "n_bl": self.n_bl,
            "auto_corrs": self.auto_corrs,
            "freq_min": self.freqs.min() / 1e6,
            "freq_max": self.freqs.max() / 1e6,
            "time_min": self.times.min(),
            "time_max": self.times.max(),
            "chan_width": self.chan_width / 1e3,
            "n_freq": self.n_freq,
            "times": self.times,
            "int_time": self.int_time,
            "n_time": self.n_time,
            "sampling_rate": self.n_int_samples / self.int_time,
            "n_ast": self.n_ast,
            "n_rfi": self.n_rfi_satellite + self.n_rfi_stationary,
            "n_sat": self.n_rfi_satellite,
            "n_stat": self.n_rfi_stationary,
        }

        return super().__str__() + msg.format(**params)

    def create_source_dicts(self):
        self.ast_I = []
        self.ast_lmn = []
        self.ast_radec = []

        self.rfi_satellite_xyz = []
        self.rfi_satellite_orbit = []
        self.rfi_satellite_ang_sep = []
        self.rfi_satellite_A_app = []

        self.rfi_stationary_xyz = []
        self.rfi_stationary_geo = []
        self.rfi_stationary_ang_sep = []
        self.rfi_stationary_A_app = []

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
        lmn = radec_to_lmn(ra, dec, [self.ra, self.dec])
        theta = jnp.arcsin(jnp.linalg.norm(lmn[:, :-1], axis=-1))
        I_app = (
            I
            * airy_beam(theta[:, None, None], self.freqs, self.dish_d)[:, 0, 0, :] ** 2
        )

        self.vis_ast += astro_vis(I_app, self.bl_uvw, lmn, self.freqs)

        self.ast_I.append(I)
        self.ast_lmn.append(lmn)
        self.ast_radec.append(jnp.array([ra, dec]))
        self.n_ast += len(I)

    def addSatelliteRFI(
        self,
        Pv: jnp.ndarray,
        elevation: jnp.ndarray,
        inclination: jnp.ndarray,
        lon_asc_node: jnp.ndarray,
        periapsis: jnp.ndarray,
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
        """
        rfi_xyz = orbit_vmap(
            self.times_fine, elevation, inclination, lon_asc_node, periapsis
        )
        # rfi_xyz is shape (n_src,n_time_fine,3)
        # self.ants_xyz is shape (n_time_fine,n_ant,3)
        distances = jnp.linalg.norm(
            self.ants_xyz[None, :, :, :] - rfi_xyz[:, :, None, :], axis=-1
        )
        # distances is shape (n_src,n_time_fine,n_ant)
        I = Pv_to_Sv(Pv, distances)
        # I is shape (n_src,n_time_fine,n_ant,n_freq)

        angular_seps = angular_separation(rfi_xyz, self.ants_xyz, self.ra, self.dec)
        # angular_seps is shape (n_src,n_time_fine,n_ant)
        rfi_A_app = jnp.sqrt(I) * airy_beam(
            angular_seps[:, :, :], self.freqs, self.dish_d
        )

        orbits = jnp.stack([elevation, inclination, lon_asc_node, periapsis], axis=1)

        self.rfi_satellite_xyz.append(rfi_xyz)
        self.rfi_satellite_orbit.append(orbits)
        self.rfi_satellite_ang_sep.append(angular_seps)
        self.rfi_satellite_A_app.append(rfi_A_app)
        self.n_rfi_satellite += len(I)

        # self.rfi_A_app is shape (n_src,n_time_fine,n_ant,n_freqs)
        # distances is shape (n_src,n_time_fine,n_ant)
        # self.ants_uvw is shape (n_time_fine,n_ant,3)
        vis_rfi = rfi_vis(
            rfi_A_app,
            distances - self.ants_uvw[None, :, :, -1],
            self.freqs,
            self.a1,
            self.a2,
        )
        self.vis_rfi += vis_rfi

    def addStationaryRFI(
        self,
        Pv: jnp.ndarray,
        latitude: jnp.ndarray,
        longitude: jnp.ndarray,
        elevation: jnp.ndarray,
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
        """
        rfi_geo = jnp.array([latitude, longitude, elevation]).T[:, None, :]
        # rfi_geo is shape (n_src,n_time,3)
        rfi_xyz = GEO_to_XYZ_vmap0(rfi_geo, self.times_fine)
        # rfi_xyz is shape (n_src,n_time_fine,3)
        # self.ants_xyz is shape (n_time_fine,n_ant,3)
        distances = jnp.linalg.norm(
            self.ants_xyz[None, :, :, :] - rfi_xyz[:, :, None, :], axis=-1
        )
        # distances is shape (n_src,n_time_fine,n_ant)
        I = Pv_to_Sv(Pv, distances)
        # I is shape (n_src,n_time,n_ant,n_freq)

        angular_seps = angular_separation(rfi_xyz, self.ants_xyz, self.ra, self.dec)
        rfi_A_app = jnp.sqrt(I) * airy_beam(
            angular_seps[:, :, :], self.freqs, self.dish_d
        )

        # self.rfi_A_app is shape (n_src,n_time_fine,n_ant,n_freqs)
        # self.ants_uvw is shape (n_time_fine,n_ant,3)
        vis_rfi = rfi_vis(
            rfi_A_app,
            distances - self.ants_uvw[None, :, :, -1],
            self.freqs,
            self.a1,
            self.a2,
        )

        self.vis_rfi += vis_rfi

        positions = jnp.stack([latitude, longitude, elevation], axis=1)

        self.rfi_stationary_xyz.append(rfi_xyz)
        self.rfi_stationary_geo.append(positions)
        self.rfi_stationary_ang_sep.append(angular_seps)
        self.rfi_stationary_A_app.append(rfi_A_app)
        self.n_rfi_stationary += len(I)

    def addGains(
        self,
        G0_mean: complex,
        G0_std: float,
        Gt_std_amp: float,
        Gt_std_phase: float,
        key=None,
    ):
        """Add complex antenna gains to the simulation. Gain amplitudes and phases
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
        self.gains_ants = generate_gains(
            G0_mean,
            G0_std,
            Gt_std_amp,
            Gt_std_phase,
            self.times_fine,
            self.n_ant,
            self.n_freq,
            key,
        )

    def calculate_vis(self):
        """
        Calculate the total gain amplified visibilities and average down to the
        originally defined sampling rate.
        """
        self.vis = apply_gains(
            self.vis_ast, self.vis_rfi, self.gains_ants, self.a1, self.a2
        )
        self.vis_avg = time_avg(self.vis, self.n_int_samples)
        self.vis_obs, self.noise_data = add_noise(
            self.vis_avg, self.noise_std, self.key
        )

    def write_to_disk(self, path: str = "Observation", overwrite: bool = False):
        """
        Write the visibilities to disk.
        """
        mode = "w" if overwrite else "w-"
        ds = construct_observation_ds(self)
        ds.to_zarr(path, mode=mode)
        return ds
