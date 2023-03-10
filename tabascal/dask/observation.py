import jax.numpy as jnp
from jax.config import config

import dask.array as da
import numpy as np

from tabascal.dask.coordinates import (
    ENU_to_UVW,
    ENU_to_GEO,
    GEO_to_XYZ_vmap0,
    GEO_to_XYZ_vmap1,
    orbit_vmap,
    radec_to_lmn,
    angular_separation,
)
from tabascal.dask.interferometry import (
    astro_vis,
    rfi_vis,
    add_noise,
    airy_beam,
    Pv_to_Sv,
    SEFD_to_noise_std,
    int_sample_times,
    generate_gains,
    ants_to_bl,
    time_avg,
)
from tabascal.utils.tools import beam_size

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
        self.latitude = da.from_array(latitude)
        self.longitude = da.from_array(longitude)
        self.elevation = da.from_array(elevation)
        self.GEO = da.from_array([latitude, longitude, elevation])
        self.ENU_path = None
        self.createArrayENU(ENU_array=ENU_array, ENU_path=ENU_path)
        self.n_ant = len(self.ENU)

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
        self.ENU = da.from_array(self.ENU)
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
        super().__init__(
            latitude,
            longitude,
            elevation,
            ENU_array=ENU_array,
            ENU_path=ENU_path,
            name=name,
        )

        self.auto_corrs = auto_corrs

        a1, a2 = jnp.triu_indices(self.n_ant, 0 if auto_corrs else 1)
        self.n_bl = len(a1)

        self.time_chunk = time_chunk if time_chunk else len(times)
        self.freq_chunk = freq_chunk if freq_chunk else len(freqs)
        self.ant_chunk = self.n_ant
        self.bl_chunk = bl_chunk if bl_chunk else self.n_bl

        self.a1 = da.from_array(a1, chunks=(self.bl_chunk,))
        self.a2 = da.from_array(a2, chunks=(self.bl_chunk,))

        self.ra = da.from_array(ra)
        self.dec = da.from_array(dec)

        self.times = da.from_array(times, chunks=(self.time_chunk,))
        self.int_time = da.abs(da.diff(times)[0])
        self.n_int_samples = n_int_samples
        self.times_fine = int_sample_times(self.times, n_int_samples)
        self.n_time = len(times)
        self.n_time_fine = len(self.times_fine)

        self.freqs = da.from_array(freqs, chunks=(self.freq_chunk,))
        self.chan_width = da.diff(freqs)[0] if len(freqs) > 1 else 250e3
        self.n_freq = len(freqs)

        self.SEFD = da.from_array(SEFD)
        self.noise_std = SEFD_to_noise_std(self.SEFD, self.chan_width, self.int_time)

        self.dish_d = da.from_array(dish_d)
        self.fov = beam_size(dish_d, freqs.max())

        self.ants_uvw = ENU_to_UVW(
            self.ENU,
            self.latitude,
            self.longitude,
            self.ra,
            self.dec,
            self.times_fine,
        )

        self.bl_uvw = self.ants_uvw[:, self.a1, :] - self.ants_uvw[:, self.a2, :]
        self.mag_uvw = da.linalg.norm(self.bl_uvw[0], axis=-1)
        self.syn_bw = beam_size(self.mag_uvw.max().compute(), freqs.max())

        self.ants_xyz = GEO_to_XYZ_vmap1(
            self.GEO_ants[None, ...]
            * da.ones(
                shape=(self.n_time_fine, self.n_ant, 3),
                chunks=(self.time_chunk, self.ant_chunk, 3),
            ),
            self.times_fine,
        )
        self.n_ast = 0
        self.n_rfi = 0
        self.vis_ast = da.zeros(
            shape=(self.n_time_fine, self.n_bl, self.n_freq),
            chunks=(self.time_chunk, self.bl_chunk, self.freq_chunk),
            dtype=jnp.complex128,
        )
        self.vis_rfi = da.zeros(
            shape=(self.n_time_fine, self.n_bl, self.n_freq),
            chunks=(self.time_chunk, self.bl_chunk, self.freq_chunk),
            dtype=jnp.complex128,
        )
        self.gains_bl = da.ones(
            shape=(self.n_time_fine, self.n_bl, self.n_freq),
            chunks=(self.time_chunk, self.bl_chunk, self.freq_chunk),
            dtype=jnp.complex128,
        )
        self.random_seed = np.random.default_rng(random_seed)
        self.create_source_dicts()

    def __str__(self):
        msg = f"""
Observation Details
-------------------
Phase Centre (ra, dec) :  ({self.ra:.1f}, {self.dec:.1f}) deg.
Number of antennas :       {self.n_ant}
Number of baselines :      {self.n_bl}
Autocorrelations :         {self.auto_corrs}

Frequency range :         ({self.freqs.min()/1e6:.0f} - {self.freqs.max()/1e6:.0f}) MHz
Channel width :            {self.chan_width/1e3:.0f} kHz
Number of channels :       {self.n_freq}

Observation time :        ({self.times[0]:.0f} - {self.times[-1]:.0f}) s
Integration time :         {self.int_time:.0f} s
Sampling rate :            {self.n_int_samples/self.int_time:.1f} Hz
Number of time steps :     {self.n_time}

Source Details
--------------
Number of ast. sources:    {self.n_ast}
Number of RFI sources:     {self.n_rfi}
Number of satellite RFI :  {len(self.rfi_orbit.keys())}
Number of stationary RFI : {len(self.rfi_geo.keys())}"""
        return super().__str__() + msg

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
        lmn = radec_to_lmn(ra, dec, [self.ra, self.dec])
        theta = da.arcsin(da.linalg.norm(lmn[:, :-1], axis=-1))
        I_app = (
            I[:, None, None, :]
            * airy_beam(theta[:, None, None], self.freqs, self.dish_d) ** 2
        )
        vis_ast = astro_vis(I_app[:, 0, 0, :], self.bl_uvw, lmn, self.freqs)

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
        n_src = Pv.shape[0]
        Pv = da.from_array(Pv, chunks=(n_src, self.freq_chunk))
        elevation = da.from_array(elevation, chunks=(n_src,))
        inclination = da.from_array(inclination, chunks=(n_src,))
        lon_asc_node = da.from_array(lon_asc_node, chunks=(n_src,))
        periapsis = da.from_array(periapsis, chunks=(n_src,))

        rfi_xyz = orbit_vmap(
            self.times_fine, elevation, inclination, lon_asc_node, periapsis
        )
        # rfi_xyz is shape (n_src,n_time_fine,3)
        # self.ants_xyz is shape (n_time_fine,n_ant,3)
        distances = da.linalg.norm(
            self.ants_xyz[None, :, :, :] - rfi_xyz[:, :, None, :], axis=-1
        )
        # distances is shape (n_src,n_time_fine,n_ant)
        I = Pv_to_Sv(Pv, distances)
        # I is shape (n_src,n_time_fine,n_ant,n_freq)

        angular_seps = angular_separation(rfi_xyz, self.ants_xyz, self.ra, self.dec)

        # angular_seps is shape (n_src,n_time_fine,n_ant)
        rfi_A_app = da.sqrt(I) * airy_beam(angular_seps, self.freqs, self.dish_d)

        orbits = da.from_array([elevation, inclination, lon_asc_node, periapsis]).T
        n_src = Pv.shape[0]
        for i in range(n_src):
            self.rfi_I.update({self.n_rfi: I[i]})
            self.rfi_xyz.update({self.n_rfi: rfi_xyz[i]})
            self.rfi_orbit.update({self.n_rfi: orbits[i]})
            self.rfi_ang_sep.update({self.n_rfi: angular_seps[i]})
            self.rfi_A_app.update({self.n_rfi: rfi_A_app[i]})
            self.n_rfi += 1

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
        n_src = Pv.shape[0]
        Pv = da.from_array(Pv, chunks=(n_src, self.freq_chunk))
        latitude = da.from_array(latitude, chunks=(n_src,))
        longitude = da.from_array(longitude, chunks=(n_src,))
        elevation = da.from_array(elevation, chunks=(n_src,))

        rfi_geo = da.from_array([latitude, longitude, elevation]).T[
            :, None, :
        ] * da.ones(
            shape=(1, self.n_time_fine, 1),
            chunks=(1, self.time_chunk, 1),
        )
        # rfi_geo is shape (n_src,n_time,3)
        rfi_xyz = GEO_to_XYZ_vmap0(rfi_geo, self.times_fine)
        # rfi_xyz is shape (n_src,n_time_fine,3)
        # self.ants_xyz is shape (n_time_fine,n_ant,3)
        distances = da.linalg.norm(
            self.ants_xyz[None, :, :, :] - rfi_xyz[:, :, None, :], axis=-1
        )
        # distances is shape (n_src,n_time_fine,n_ant)
        I = Pv_to_Sv(Pv, distances)
        # I is shape (n_src,n_time,n_ant,n_freq)

        angular_seps = angular_separation(rfi_xyz, self.ants_xyz, self.ra, self.dec)
        rfi_A_app = da.sqrt(I) * airy_beam(angular_seps, self.freqs, self.dish_d)

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
            rfi_A_app,
            distances - self.ants_uvw[None, :, :, -1],
            self.freqs,
            self.a1,
            self.a2,
        )
        self.vis_rfi += vis_rfi

    def addGains(
        self,
        G0_mean: float,
        G0_std: float,
        Gt_std_amp: float,
        Gt_std_phase: float,
        random_seed=None,
    ):
        """Add complex antenna gains to the simulation. Gain amplitudes and phases
        are modelled as linear time-variates. Gains for all antennas at t = 0
        are randomly sampled from a Gaussian described by the G0 parameters.
        The rate of change of both ampltudes and phases are sampled from a zero
        mean Gaussian with standard deviation as provided.

        Parameters
        ----------
        G0_mean: float
            Mean of Gaussian at t = 0.
        G0_std: float
            Standard deviation of Gaussian at t = 0.
        Gt_std_amp: float
            Standard deviation of Gaussian describing the rate of change in the
            gain amplitudes in 1/seconds.
        Gt_std_phase: float
            Standard deviation of Gaussian describing the rate of change in the
            gain phases in rad/seconds.
        random_seed: int, optional
            Random number generator key.
        """
        self.gains_ants = generate_gains(
            G0_mean,
            G0_std,
            Gt_std_amp,
            Gt_std_phase,
            self.times_fine,
            self.n_ant,
            self.n_freq,
            random_seed if random_seed else self.random_seed,
        )
        self.gains_bl = ants_to_bl(self.gains_ants, self.a1, self.a2)

    def calculate_vis(self, random_seed=None):
        """
        Calculate the total gain amplified visibilities,  average down to the
        originally defined sampling rate and add noise.
        """
        self.vis = self.gains_bl * (self.vis_ast + self.vis_rfi)
        self.vis_avg = time_avg(self.vis, self.n_int_samples)
        self.vis_obs, self.noise_data = add_noise(
            self.vis_avg,
            self.noise_std,
            random_seed if random_seed else self.random_seed,
        )
