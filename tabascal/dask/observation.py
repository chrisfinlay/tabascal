import jax.numpy as jnp
from jax import config, Array

import dask.array as da
import numpy as np
import xarray as xr

from tabascal.dask.coordinates import (
    ENU_to_ITRF,
    ITRF_to_UVW,
    ENU_to_GEO,
    GEO_to_XYZ_vmap0,
    ITRF_to_XYZ,
    orbit_vmap,
    radec_to_lmn,
    angular_separation,
)
from tabascal.dask.interferometry import (
    astro_vis,
    astro_vis_gauss,
    astro_vis_exp,
    rfi_vis,
    add_noise,
    airy_beam,
    Pv_to_Sv,
    SEFD_to_noise_std,
    int_sample_times,
    generate_gains,
    generate_fourier_gains,
    apply_gains,
)

from tabascal.jax.interferometry import int_sample_times

from tabascal.jax.coordinates import (
    itrf_to_geo,
    alt_az_of_source,
    gmsa_from_jd,
    mjd_to_jd,
    secs_to_days,
)
from tabascal.utils.tools import beam_size
from tabascal.utils.write import construct_observation_ds, write_ms
from tabascal.utils.dask_extras import get_chunksizes
from tabascal.utils.tle import get_satellite_positions, ants_pos, sat_distance

config.update("jax_enable_x64", True)

from astropy.time import Time


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
        elevation: float = 0.0,
        ENU_array: Array = None,
        ENU_path: str = None,
        ITRF_array: Array = None,
        ITRF_path: str = None,
        tel_name: str = None,
        n_ant: int = None,
    ):
        self.tel_name = tel_name
        self.latitude = da.asarray(latitude)
        self.longitude = da.asarray(longitude)
        self.elevation = da.asarray(elevation)
        self.GEO = da.asarray([latitude, longitude, elevation])
        self.ITRF = None
        self.ENU = None
        self.n_ant = n_ant

        if ENU_array is not None or ENU_path is not None:
            self.createArrayENU(ENU_array, ENU_path)
        if ITRF_array is not None or ITRF_path is not None:
            self.createArrayITRF(ITRF_array, ITRF_path)
        if self.ITRF is None and self.ENU is None:
            raise ValueError(
                "One of ('ENU_array', 'ENU_path', 'ITRF_array', 'ITRF_path') must be provided to create a Telescope object."
            )
        self.n_ant = len(self.ITRF)

    def __str__(self):
        msg = """\nTelescope Location
------------------
Latitude : {latitude}
Longitude : {longitude}
Elevation : {elevation}\n"""
        params = {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "elevation": self.elevation,
        }
        params = {k: v.compute() for k, v in params.items()}
        return msg.format(**params)

    def createArrayENU(self, ENU_array=None, ENU_path=None):
        if ENU_array is not None:
            self.ENU = ENU_array
        elif ENU_path is not None:
            self.ENU = np.loadtxt(ENU_path, usecols=(0, 1, 2), max_rows=self.n_ant)
        else:
            self.ENU = None
            msg = """Error : East-North-Up coordinates are needed either in an 
                     array or as a csv like file."""
            print(msg)
            return
        self.ENU = da.asarray(self.ENU)
        self.ENU_path = ENU_path
        self.GEO_ants = ENU_to_GEO(self.GEO, self.ENU)
        self.ITRF = ENU_to_ITRF(self.ENU, self.latitude, self.longitude, self.elevation)

    def createArrayITRF(self, ITRF_array, ITRF_path):
        if ITRF_array is not None:
            self.ITRF = ITRF_array
        elif ITRF_path is not None:
            self.ITRF = np.loadtxt(ITRF_path, usecols=(0, 1, 2), max_rows=self.n_ant)
        else:
            self.ITRF = None
            msg = """Error : ITRF antenna coordinates are needed either in an 
                     array or as a csv like file."""
            print(msg)
            return
        self.ITRF = da.asarray(self.ITRF)
        self.GEO_ants = da.asarray(itrf_to_geo(self.ITRF.compute()))


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
        Time centroids of each data point in seconds as a Greenwich Mean Sidereal Time (GMST).
    freqs: ndarray (n_freq,)
        Frequency centroids for each observation channel in Hz.
    SEFD: ndarray (n_freq,)
        System Equivalent Flux Density of the telescope over frequency.
    chan_width: float
        Frequency channel width in Hz. Only used if `n_freq=1`, else calculated from `freqs`.
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
    no_w: bool
        Whether to zero out the w-component of the baselines.
    n_int_samples: int
        Number of samples per time step which are then averaged. Must be
        large enough to capture time-smearing of RFI sources on longest
        baseline.
    tel_name: str
        Name of the telescope.
    target_name: str
        Name fo the target field.
    """

    def __init__(
        self,
        latitude: float,
        longitude: float,
        elevation: float,
        ra: float,
        dec: float,
        freqs: Array,
        SEFD: Array,
        times_mjd: Array,
        int_time: float = 2.0,
        chan_width: float = 209e3,
        ENU_array: Array = None,
        ENU_path: str = None,
        ITRF_array: Array = None,
        ITRF_path: str = None,
        n_ant: int = None,
        dish_d: float = 13.5,
        random_seed: int = 0,
        auto_corrs: bool = False,
        no_w: bool = False,
        n_int_samples: int = 4,
        tel_name: str = "MeerKAT",
        target_name: str = "unknown",
        max_chunk_MB: float = 100.0,
    ):
        super().__init__(
            latitude,
            longitude,
            elevation,
            ENU_array,
            ENU_path,
            ITRF_array,
            ITRF_path,
            tel_name,
            n_ant,
        )

        n_time = len(times_mjd)
        start_mjd = times_mjd[0]
        # times = (times_mjd - times_mjd[0]) * 24 * 3600
        times = da.linspace(0, int_time * n_time, n_time, endpoint=False)
        times_mjd = start_mjd + secs_to_days(times)

        n_int_samples = n_int_samples + 1 if n_int_samples % 2 == 0 else n_int_samples

        self.target_name = target_name
        self.auto_corrs = auto_corrs

        a1, a2 = jnp.triu_indices(self.n_ant, 0 if auto_corrs else 1)
        self.n_bl = len(a1)

        self.ant_chunk = self.n_ant
        self.bl_chunk = self.n_bl

        self.a1 = da.asarray(a1, chunks=(self.bl_chunk,))
        self.a2 = da.asarray(a2, chunks=(self.bl_chunk,))

        self.ra = da.asarray(ra)
        self.dec = da.asarray(dec)

        chunksize = get_chunksizes(
            len(times), len(freqs), n_int_samples, self.n_bl, max_chunk_MB
        )
        self.time_chunk = chunksize["time"]
        self.time_fine_chunk = self.time_chunk * n_int_samples
        self.freq_chunk = chunksize["freq"]

        self.times = da.asarray(times).rechunk(self.time_chunk)
        self.times_mjd = da.asarray(times_mjd).rechunk(self.time_chunk)
        self.int_time = int_time
        self.n_int_samples = n_int_samples

        self.times_fine = da.asarray(
            int_sample_times(self.times.compute(), n_int_samples)
        ).rechunk(self.time_fine_chunk)

        self.times_mjd_fine = start_mjd + secs_to_days(self.times_fine)

        #######################################
        # dt = da.diff(self.times_fine)[:1]
        # dt_jd = da.diff(self.times_mjd_fine)[:1]

        # self.times_fine = da.concatenate([self.times_fine, dt]).rechunk(
        #     self.time_fine_chunk + 1
        # )
        # self.times_mjd_fine = da.concatenate([self.times_mjd_fine, dt_jd]).rechunk(
        #     self.time_fine_chunk + 1
        # )
        #######################################

        self.n_time = len(times)
        self.n_time_fine = len(self.times_fine)
        self.t_idx = da.arange(
            self.n_int_samples // 2, self.n_time_fine, self.n_int_samples
        ).rechunk(self.time_chunk)

        self.gsa = da.asarray(
            Time(self.times_mjd_fine.compute(), format="mjd")
            .sidereal_time("mean", "greenwich")
            .hour
            * 15,
            chunks=(self.time_fine_chunk,),
        )
        self.gha = (self.gsa - self.ra) % 360
        self.lsa = (self.gsa + longitude) % 360
        self.lha = (((self.gha + longitude) % 360 - 180) % 360) - 180
        self.altaz = da.asarray(alt_az_of_source(self.lsa.compute(), latitude, ra, dec))

        self.freqs = da.asarray(freqs).rechunk((self.freq_chunk,))
        self.chan_width = da.diff(freqs)[0] if len(freqs) > 1 else chan_width
        self.n_freq = len(freqs)
        self.lamda = 299792458.0 / self.freqs

        self.SEFD = da.asarray(SEFD) * da.ones(self.n_freq, chunks=(self.freq_chunk,))
        self.noise_std = SEFD_to_noise_std(self.SEFD, self.chan_width, self.int_time)

        self.dish_d = da.asarray(dish_d)
        self.fov = beam_size(dish_d, freqs.max(), fwhp=False)

        self.ants_uvw = ITRF_to_UVW(self.ITRF, self.gha, self.dec)

        if no_w:
            self.ants_uvw[:, :, -1] = 0.0

        self.bl_uvw = self.ants_uvw[:, self.a1, :] - self.ants_uvw[:, self.a2, :]
        self.mag_uvw = da.linalg.norm(self.bl_uvw[0], axis=-1)
        self.syn_bw = beam_size(self.mag_uvw.max().compute(), freqs.max())

        self.ants_xyz = ITRF_to_XYZ(self.ITRF, self.gsa)
        self.vis_rfi = da.zeros(
            shape=(self.n_time, self.n_bl, self.n_freq),
            chunks=(self.time_chunk, self.bl_chunk, self.freq_chunk),
            dtype=jnp.complex128,
        )
        self.vis_ast = da.zeros(
            shape=(self.n_time, self.n_bl, self.n_freq),
            chunks=(self.time_chunk, self.bl_chunk, self.freq_chunk),
            dtype=jnp.complex128,
        )
        self.gains_ants = da.ones(
            shape=(self.n_time, self.n_ant, self.n_freq),
            chunks=(self.time_chunk, self.ant_chunk, self.freq_chunk),
            dtype=jnp.complex128,
        )
        self.random_seed = random_seed

        self.n_ast = 0
        self.n_p_ast = 0
        self.n_g_ast = 0
        self.n_e_ast = 0
        self.n_rfi = 0
        self.n_rfi_satellite = 0
        self.n_rfi_stationary = 0
        self.n_rfi_tle_satellite = 0

        self.create_source_dicts()

    def __str__(self):
        msg = """
Observation Details
-------------------
Phase Centre (ra, dec) :  ({ra:.1f}, {dec:.1f}) deg.
Local Hour Angle range :  ({lha_min:.1f}, {lha_max:.1f}) deg.
Source Altitude range  :  ({alt_min:.1f}, {alt_max:.1f}) deg.
Number of antennas     :   {n_ant}
Number of baselines    :   {n_bl}
Autocorrelations       :   {auto_corrs}

Frequency range        :   ({freq_min:.0f} - {freq_max:.0f}) MHz
Channel width          :    {chan_width:.0f} kHz
Number of channels     :    {n_freq}

Observation time       :   ({time_min} - {time_max})
Integration time       :    {int_time:.0f} s
Sampling rate          :    {sampling_rate:.1f} Hz
Number of time steps   :    {n_time}

Source Details
--------------
Number of ast. sources   :  {n_ast}
Number of RFI sources    :  {n_rfi}
Number of satellite RFI  :  {n_sat}
Number of stationary RFI :  {n_stat}"""

        params = {
            "ra": self.ra,
            "dec": self.dec,
            "n_ant": self.n_ant,
            "n_bl": self.n_bl,
            "auto_corrs": self.auto_corrs,
            "freq_min": self.freqs.min() / 1e6,
            "freq_max": self.freqs.max() / 1e6,
            "time_min": Time(self.times_mjd.min(), format="mjd").strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            "time_max": Time(self.times_mjd.max(), format="mjd").strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            "lha_min": self.lha.min(),
            "lha_max": self.lha.max(),
            "alt_min": self.altaz[:, 0].min(),
            "alt_max": self.altaz[:, 0].max(),
            "chan_width": self.chan_width / 1e3,
            "n_freq": self.n_freq,
            "int_time": self.int_time,
            "n_time": self.n_time,
            "sampling_rate": self.n_int_samples / self.int_time,
        }
        params = {
            k: v.compute() if isinstance(v, da.Array) else v for k, v in params.items()
        }
        params.update(
            {
                "n_ast": self.n_ast,
                "n_rfi": self.n_rfi,
                "n_sat": self.n_rfi_satellite,
                "n_tle_sat": self.n_rfi_tle_satellite,
                "n_stat": self.n_rfi_stationary,
            }
        )

        return super().__str__() + msg.format(**params)

    def create_source_dicts(self):
        self.ast_p_I = []
        self.ast_p_lmn = []
        self.ast_p_radec = []

        self.ast_g_I = []
        self.ast_g_lmn = []
        self.ast_g_radec = []
        self.ast_g_major = []
        self.ast_g_minor = []
        self.ast_g_pos_angle = []

        self.ast_e_I = []
        self.ast_e_lmn = []
        self.ast_e_radec = []
        self.ast_e_major = []

        self.rfi_satellite_xyz = []
        self.rfi_satellite_orbit = []
        self.rfi_satellite_ang_sep = []
        self.rfi_satellite_A_app = []

        self.rfi_tle_satellite_xyz = []
        self.rfi_tle_satellite_orbit = []
        self.rfi_tle_satellite_ang_sep = []
        self.rfi_tle_satellite_A_app = []
        self.norad_ids = []

        self.rfi_stationary_xyz = []
        self.rfi_stationary_geo = []
        self.rfi_stationary_ang_sep = []
        self.rfi_stationary_A_app = []

    def addAstro(self, I: Array, ra: Array, dec: Array):
        """
        Add a set of astronomical sources to the observation.

        Parameters
        ----------
        I: ndarray (n_src, n_time, n_freq) or
            Intensity of the sources in Jy. If I.ndim==2, then this is assumed
            to the spectrogram (n_time, n_freq) of a single source. If
            I.ndim==1, then this is assumed to be the spectral profile of a
            single source.
        ra: array (n_src,)
            Right ascension of the sources in degrees.
        dec: array (n_src,)
            Declination of the sources in degrees.
        """
        I = da.atleast_2d(I)
        if I.ndim == 2:
            I = da.expand_dims(I, axis=0)
        I = I * da.ones(
            shape=(I.shape[0], self.n_time, self.n_freq),
            chunks=(I.shape[0], self.time_chunk, self.freq_chunk),
        )
        ra = da.atleast_1d(ra)
        dec = da.atleast_1d(dec)
        lmn = radec_to_lmn(ra, dec, [self.ra, self.dec])
        theta = da.rad2deg(da.arcsin(da.linalg.norm(lmn[:, :-1], axis=-1)))
        I_app = (
            I
            * (airy_beam(theta[:, None, None], self.freqs, self.dish_d)[:, :, 0, :])
            ** 2
        )
        vis_ast = astro_vis(I_app, self.bl_uvw[self.t_idx], lmn, self.freqs)

        self.ast_p_I.append(I)
        self.ast_p_lmn.append(lmn)
        self.ast_p_radec.append(jnp.array([ra, dec]))
        self.n_p_ast += len(I)
        self.n_ast += len(I)

        self.vis_ast += vis_ast

    def addAstroGauss(
        self,
        I: Array,
        major: Array,
        minor: Array,
        pos_angle: Array,
        ra: Array,
        dec: Array,
    ):
        """
        Add a set of astronomical sources to the observation.

        Parameters
        ----------
        I: Array (n_src, n_time, n_freq) or
            Intensity of the sources in Jy. If I.ndim==2, then this is assumed
            to the spectrogram (n_time, n_freq) of a single source. If
            I.ndim==1, then this is assumed to be the spectral profile of a
            single source.
        major: Array (n_src,)
            FWHM of major axis of sources in arcseconds.
        major: Array (n_src,)
            FWHM of minor axis of sources in arcseconds.
        pos_angle: Array (n_src,)
            Position angle of sources in degrees west of north for the major axis.
        ra: Array (n_src,)
            Right ascension of the sources in degrees.
        dec: Array (n_src,)
            Declination of the sources in degrees.
        """
        I = da.atleast_2d(I)
        if I.ndim == 2:
            I = da.expand_dims(I, axis=0)
        I = I * da.ones(
            shape=(I.shape[0], self.n_time, self.n_freq),
            chunks=(I.shape[0], self.time_chunk, self.freq_chunk),
        )
        major = da.atleast_1d(major)
        minor = da.atleast_1d(minor)
        pos_angle = da.atleast_1d(pos_angle)
        ra = da.atleast_1d(ra)
        dec = da.atleast_1d(dec)
        lmn = radec_to_lmn(ra, dec, [self.ra, self.dec])
        theta = da.rad2deg(da.arcsin(da.linalg.norm(lmn[:, :-1], axis=-1)))
        I_app = (
            I
            * (airy_beam(theta[:, None, None], self.freqs, self.dish_d)[:, :, 0, :])
            ** 2
        )
        vis_ast = astro_vis_gauss(
            I_app, major, minor, pos_angle, self.bl_uvw[self.t_idx], lmn, self.freqs
        )

        self.ast_g_major.append(major)
        self.ast_g_minor.append(minor)
        self.ast_g_pos_angle.append(pos_angle)
        self.ast_g_I.append(I)
        self.ast_g_lmn.append(lmn)
        self.ast_g_radec.append(jnp.array([ra, dec]))
        self.n_g_ast += len(I)
        self.n_ast += len(I)

        self.vis_ast += vis_ast

    def addAstroExp(self, I: Array, shape: Array, ra: Array, dec: Array):
        """
        Add a set of astronomical sources to the observation.

        Parameters
        ----------
        I: Array (n_src, n_time, n_freq) or
            Intensity of the sources in Jy. If I.ndim==2, then this is assumed
            to the spectrogram (n_time, n_freq) of a single source. If
            I.ndim==1, then this is assumed to be the spectral profile of a
            single source.
        shape: array (n_src,)
            Shape of gaussian sources. Only circular gaussians accepted for now.
        ra: array (n_src,)
            Right ascension of the sources in degrees.
        dec: array (n_src,)
            Declination of the sources in degrees.
        """
        I = da.atleast_2d(I)
        if I.ndim == 2:
            I = da.expand_dims(I, axis=0)
        I = I * da.ones(
            shape=(I.shape[0], self.n_time, self.n_freq),
            chunks=(I.shape[0], self.time_chunk, self.freq_chunk),
        )
        shape = da.atleast_1d(shape)
        ra = da.atleast_1d(ra)
        dec = da.atleast_1d(dec)
        lmn = radec_to_lmn(ra, dec, [self.ra, self.dec])
        theta = da.rad2deg(da.arcsin(da.linalg.norm(lmn[:, :-1], axis=-1)))
        I_app = (
            I
            * (airy_beam(theta[:, None, None], self.freqs, self.dish_d)[:, :, 0, :])
            ** 2
        )
        vis_ast = astro_vis_exp(I_app, shape, self.bl_uvw[self.t_idx], lmn, self.freqs)

        self.ast_e_major.append(shape)
        self.ast_e_I.append(I)
        self.ast_e_lmn.append(lmn)
        self.ast_e_radec.append(jnp.array([ra, dec]))
        self.n_e_ast += len(I)
        self.n_ast += len(I)

        self.vis_ast += vis_ast

    def addSatelliteRFI(
        self,
        Pv: Array,
        elevation: Array,
        inclination: Array,
        lon_asc_node: Array,
        periapsis: Array,
    ):
        """
        Add a satellite-based source of RFI to the observation.

        Parameters
        ----------
        Pv: ndarray (n_src, n_time_fine, n_freq)
            Specific Emission Power in W/Hz. If Pv.ndim==1, it is assumed to be
            of shape (n_freq,) and is the spectrum of a single RFI source. If
            Pv.ndim==2, it is assumed to be of shape (n_time_fine, n_freq) and
            is the spectrogram of a single RFI source.
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
        Pv = da.atleast_2d(Pv)
        if Pv.ndim == 2:
            Pv = da.expand_dims(Pv, axis=0)
        Pv = (
            Pv
            * da.ones(
                shape=(1, self.n_time_fine, self.n_freq),
            )
        ).rechunk((-1, self.time_fine_chunk, self.freq_chunk))
        elevation = da.asarray(da.atleast_1d(elevation), chunks=(-1,))
        inclination = da.asarray(da.atleast_1d(inclination), chunks=(-1,))
        lon_asc_node = da.asarray(da.atleast_1d(lon_asc_node), chunks=(-1,))
        periapsis = da.asarray(da.atleast_1d(periapsis), chunks=(-1,))

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
        rfi_A_app = da.sqrt(da.abs(I)) * airy_beam(
            angular_seps, self.freqs, self.dish_d
        )
        # self.rfi_A_app is shape (n_src,n_time_fine,n_ant,n_freqs)
        # distances is shape (n_src,n_time_fine,n_ant)
        # self.ants_uvw is shape (n_time_fine,n_ant,3)

        vis_rfi = rfi_vis(
            rfi_A_app.reshape(
                (-1, self.n_time, self.n_int_samples, self.n_ant, self.n_freq)
            ),
            (distances + self.ants_uvw[None, :, :, -1]).reshape(
                (-1, self.n_time, self.n_int_samples, self.n_ant)
            ),
            self.freqs,
            self.a1,
            self.a2,
        )
        self.vis_rfi += vis_rfi

        orbits = da.stack([elevation, inclination, lon_asc_node, periapsis], axis=1)

        self.rfi_satellite_xyz.append(rfi_xyz)
        self.rfi_satellite_orbit.append(orbits)
        self.rfi_satellite_ang_sep.append(angular_seps)
        self.rfi_satellite_A_app.append(rfi_A_app)
        self.n_rfi_satellite += len(I)
        self.n_rfi += len(I)

    def addTLESatelliteRFI(
        self,
        Pv: Array,
        norad_ids: list[int],
        tles: Array,
    ):
        """
        Add a satellite-based source of RFI to the observation.

        Parameters
        ----------
        Pv: ndarray (n_src, n_time_fine, n_freq)
            Specific Emission Power in W/Hz. If Pv.ndim==1, it is assumed to be
            of shape (n_freq,) and is the spectrum of a single RFI source. If
            Pv.ndim==2, it is assumed to be of shape (n_time_fine, n_freq) and
            is the spectrogram of a single RFI source.
        norad_ids: list[int] (n_src,)
            NORAD IDs for the satellites to include.
        tles: Array (n_src, 2)
            TLEs of the satellites corresponding to the NORAD IDs.
        """
        Pv = da.atleast_2d(Pv)
        if Pv.ndim == 2:
            Pv = da.expand_dims(Pv, axis=0)
        Pv = (
            Pv
            * da.ones(
                shape=(1, self.n_time_fine, self.n_freq),
            )
        ).rechunk((-1, self.time_fine_chunk, self.freq_chunk))
        norad_ids = da.asarray(da.atleast_1d(norad_ids), chunks=(-1,))

        rfi_xyz = da.asarray(
            get_satellite_positions(tles, mjd_to_jd(self.times_mjd_fine.compute())),
            chunks=(-1, self.time_fine_chunk, 3),
        )
        tles = da.asarray(da.atleast_2d(tles), chunks=(-1,))
        # rfi_xyz is shape (n_src,n_time_fine,3)
        # self.ants_xyz is shape (n_time_fine,n_ant,3)
        distances = da.linalg.norm(
            self.ants_xyz[None, :, :, :] - rfi_xyz[:, :, None, :], axis=-1
        )
        # distances = da.asarray(da.linalg.norm(
        #     ants_pos(self.ITRF.compute(), mjd_to_jd(self.times_mjd_fine.compute()))[None, :, :, :] - rfi_xyz[:, :, None, :], axis=-1
        # ), chunks=(-1, self.time_fine_chunk, self.n_ant))
        # distances is shape (n_src,n_time_fine,n_ant)
        I = Pv_to_Sv(Pv, distances)
        # I is shape (n_src,n_time_fine,n_ant,n_freq)

        angular_seps = angular_separation(rfi_xyz, self.ants_xyz, self.ra, self.dec)

        # angular_seps is shape (n_src,n_time_fine,n_ant)
        rfi_A_app = da.sqrt(da.abs(I)) * airy_beam(
            angular_seps, self.freqs, self.dish_d
        )
        # rfi_A_app = da.ones((Pv.shape[0], self.n_time_fine, self.n_ant, self.n_freq), chunks=(-1, self.time_fine_chunk, self.n_ant, self.freq_chunk))
        # self.rfi_A_app is shape (n_src,n_time_fine,n_ant,n_freqs)
        # distances is shape (n_src,n_time_fine,n_ant)
        # self.ants_uvw is shape (n_time_fine,n_ant,3)

        vis_rfi = rfi_vis(
            rfi_A_app.reshape(
                (-1, self.n_time, self.n_int_samples, self.n_ant, self.n_freq)
            ),
            (distances + self.ants_uvw[None, :, :, -1]).reshape(
                (-1, self.n_time, self.n_int_samples, self.n_ant)
            ),
            self.freqs,
            self.a1,
            self.a2,
        )
        self.vis_rfi += vis_rfi

        self.rfi_tle_satellite_xyz.append(rfi_xyz)
        self.rfi_tle_satellite_orbit.append(tles)
        self.rfi_tle_satellite_ang_sep.append(angular_seps)
        self.rfi_tle_satellite_A_app.append(rfi_A_app)
        self.norad_ids.append(norad_ids)
        self.n_rfi_tle_satellite += len(I)
        self.n_rfi += len(I)

    def addStationaryRFI(
        self,
        Pv: Array,
        latitude: Array,
        longitude: Array,
        elevation: Array,
    ):
        """
        Add a stationary source of RFI to the observation.

        Parameters
        ----------
        Pv: ndarray (n_src, n_time_fine, n_freq)
            Specific Emission Power in W/Hz. If Pv.ndim==1, it is assumed to be
            of shape (n_freq,) and is the spectrum of a single RFI source. If
            Pv.ndim==2, it is assumed to be of shape (n_time_fine, n_freq) and
            is the spectrogram of a single RFI source.
        latitude: ndarray (n_src,)
            Geopgraphic latitude of the source in degrees.
        longitude: ndarray (n_src,)
            Geographic longitude of the source in degrees.
        elevation: ndarray (n_src,)
            Elevation/Altitude of the source above sea level in metres.
        """
        Pv = da.atleast_2d(Pv)
        if Pv.ndim == 2:
            Pv = da.expand_dims(Pv, axis=0)
        Pv = (
            Pv
            * da.ones(
                shape=(1, self.n_time_fine, self.n_freq),
            )
        ).rechunk((-1, self.time_fine_chunk, self.freq_chunk))
        latitude = da.asarray(da.atleast_1d(latitude), chunks=(-1,))
        longitude = da.asarray(da.atleast_1d(longitude), chunks=(-1,))
        elevation = da.asarray(da.atleast_1d(elevation), chunks=(-1,))

        rfi_geo = (
            da.stack([latitude, longitude, elevation], axis=1)[:, None, :]
            * da.ones(shape=(1, self.n_time_fine, 3))
        ).rechunk(
            (-1, self.time_fine_chunk, 3),
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
        rfi_A_app = da.sqrt(da.abs(I)) * airy_beam(
            angular_seps, self.freqs, self.dish_d
        )

        # self.rfi_A_app is shape (n_src,n_time_fine,n_ant,n_freqs)
        # self.ants_uvw is shape (n_time_fine,n_ant,3)
        vis_rfi = rfi_vis(
            rfi_A_app.reshape(
                (-1, self.n_time, self.n_int_samples, self.n_ant, self.n_freq)
            ),
            (distances + self.ants_uvw[None, :, :, -1]).reshape(
                (-1, self.n_time, self.n_int_samples, self.n_ant)
            ),
            self.freqs,
            self.a1,
            self.a2,
        )

        self.vis_rfi += vis_rfi

        positions = da.stack([latitude, longitude, elevation], axis=1)

        self.rfi_stationary_xyz.append(rfi_xyz)
        self.rfi_stationary_geo.append(positions)
        self.rfi_stationary_ang_sep.append(angular_seps)
        self.rfi_stationary_A_app.append(rfi_A_app)
        self.n_rfi_stationary += len(I)
        self.n_rfi += len(I)

    def addGains(
        self,
        G0_mean: float,
        G0_std: float,
        Gt_std_amp: float,
        Gt_std_phase: float,
        Gt_corr_amp: float,
        Gt_corr_phase: float,
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
        # self.gains_ants = generate_gains(
        #     G0_mean,
        #     G0_std,
        #     Gt_std_amp,
        #     Gt_std_phase,
        #     self.times,
        #     self.n_ant,
        #     self.n_freq,
        #     random_seed if random_seed else self.random_seed,
        # ).rechunk((self.time_chunk, self.ant_chunk, self.freq_chunk))
        self.gains_ants = generate_fourier_gains(
            G0_mean,
            G0_std,
            Gt_std_amp,
            Gt_std_phase,
            Gt_corr_amp,
            Gt_corr_phase,
            self.times_mjd,
            self.n_ant,
            self.n_freq,
            random_seed if random_seed else self.random_seed,
        ).rechunk((self.time_chunk, self.ant_chunk, self.freq_chunk))

    def calculate_vis(self, flags: bool = True, random_seed=None):
        """
        Calculate the total gain amplified visibilities,  average down to the
        originally defined sampling rate and add noise.
        """
        self.vis_uncal = apply_gains(
            self.vis_ast, self.vis_rfi, self.gains_ants, self.a1, self.a2
        ).rechunk((self.time_chunk, self.bl_chunk, self.freq_chunk))
        self.vis_obs, self.noise_data = add_noise(
            self.vis_uncal,
            self.noise_std,
            random_seed if random_seed else self.random_seed,
        )
        self.vis_cal = apply_gains(
            self.vis_obs,
            da.zeros_like(self.vis_obs),
            1.0 / self.gains_ants,
            self.a1,
            self.a2,
        )
        if flags:
            if self.noise_std.mean() > 0:
                self.flags = (
                    da.abs(self.vis_cal - self.vis_ast)
                    > 3.0 * self.noise_std[None, None, :]
                )  # * da.sqrt(2)
            else:
                self.flags = (
                    da.abs(self.vis_cal - self.vis_ast)
                    > 3.0 * da.std(self.vis_model, axis=0)[None, ...]
                )
        else:
            self.flags = da.zeros(shape=self.vis_cal.shape, dtype=bool)

        self.dataset = construct_observation_ds(self)
        return self.dataset

    def write_to_zarr(self, path: str = "Observation", overwrite: bool = False):
        """
        Write the visibilities to disk using zarr format.
        """
        mode = "w" if overwrite else "w-"
        self.dataset.to_zarr(path, mode=mode)
        self.dataset = xr.open_zarr(path)
        return self.dataset

    def write_to_ms(
        self,
        path: str = "Observation.ms",
        overwrite: bool = False,
        ds: xr.Dataset = None,
    ):
        """
        Write the visibilities to disk using Measurement Set format.
        """
        if ds is None:
            ds = self.dataset
        write_ms(ds, path, overwrite=overwrite)
