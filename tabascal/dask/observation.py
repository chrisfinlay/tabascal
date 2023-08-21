import jax.numpy as jnp
from jax.config import config

import dask.array as da
import numpy as np

from jax import Array
from numpy.typing import ArrayLike

from typing import List, MutableMapping, Literal
from os import PathLike

from xarray import Dataset

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
    apply_gains,
    time_avg,
)
from tabascal.utils.tools import beam_size
from tabascal.utils.write import construct_observation_ds, write_ms
from tabascal.utils.dask import get_chunksizes

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
        ENU_array: ArrayLike | None = None,
        ENU_path: str | None = None,
        name: str | None = None,
    ):
        self.name = name
        self.latitude = da.asarray(latitude)
        self.longitude = da.asarray(longitude)
        self.elevation = da.asarray(elevation)
        self.GEO = da.asarray([latitude, longitude, elevation])
        self.createArrayENU(ENU_array=ENU_array, ENU_path=ENU_path)

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
            self.ENU = np.loadtxt(ENU_path)
        else:
            self.ENU = None
            msg = """Error : East-North-Up coordinates are needed either in an 
                     array or as a csv like file."""
            print(msg)
            return
        self.ENU = da.asarray(self.ENU)
        self.ENU_path = ENU_path
        self.GEO_ants = ENU_to_GEO(self.GEO, self.ENU)
        self.n_ant = len(self.ENU)


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
        times: Array,
        freqs: Array,
        SEFD: Array,
        ENU_path: str | None = None,
        ENU_array: Array | None = None,
        dish_d: float = 13.965,
        random_seed: int = 0,
        auto_corrs: bool = False,
        n_int_samples: int = 4,
        name: str = "MeerKAT",
        max_chunk_MB: float = 100.0,
    ):
        super().__init__(
            latitude,
            longitude,
            elevation,
            ENU_array=ENU_array,
            ENU_path=ENU_path,
            name=name,
        )

        self.backend = "dask"
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
        self.int_time = da.abs(da.diff(times)[0]) if len(times) > 1 else 2.0
        self.n_int_samples = n_int_samples
        self.times_fine = int_sample_times(self.times, n_int_samples).rechunk(
            self.time_fine_chunk
        )
        self.n_time = len(times)
        self.n_time_fine = len(self.times_fine)

        self.freqs = da.asarray(freqs, chunks=(self.freq_chunk,))
        self.chan_width = da.diff(freqs)[0] if len(freqs) > 1 else 209e3
        self.n_freq = len(freqs)

        self.SEFD = da.asarray(SEFD) * da.ones(self.n_freq, chunks=(self.freq_chunk,))
        self.noise_std = SEFD_to_noise_std(self.SEFD, self.chan_width, self.int_time)

        self.dish_d = da.asarray(dish_d)
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
                chunks=(self.time_fine_chunk, self.ant_chunk, 3),
            ),
            self.times_fine,
        )
        self.vis_ast = da.zeros(
            shape=(self.n_time_fine, self.n_bl, self.n_freq),
            chunks=(self.time_fine_chunk, self.bl_chunk, self.freq_chunk),
            dtype=jnp.complex128,
        )
        self.vis_rfi = da.zeros(
            shape=(self.n_time_fine, self.n_bl, self.n_freq),
            chunks=(self.time_fine_chunk, self.bl_chunk, self.freq_chunk),
            dtype=jnp.complex128,
        )
        self.gains_ants = da.ones(
            shape=(self.n_time_fine, self.n_ant, self.n_freq),
            chunks=(self.time_fine_chunk, self.ant_chunk, self.freq_chunk),
            dtype=jnp.complex128,
        )
        self.random_seed = np.random.default_rng(random_seed)

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
                "n_rfi": self.n_rfi_satellite + self.n_rfi_stationary,
                "n_sat": self.n_rfi_satellite,
                "n_stat": self.n_rfi_stationary,
            }
        )

        return super().__str__() + msg.format(**params)

    def create_source_dicts(self) -> None:
        self.ast_I: List[ArrayLike] = []
        self.ast_lmn: List[ArrayLike] = []
        self.ast_radec: List[ArrayLike] = []

        self.rfi_satellite_xyz: List[ArrayLike] = []
        self.rfi_satellite_orbit: List[ArrayLike] = []
        self.rfi_satellite_ang_sep: List[ArrayLike] = []
        self.rfi_satellite_A_app: List[ArrayLike] = []

        self.rfi_stationary_xyz: List[ArrayLike] = []
        self.rfi_stationary_geo: List[ArrayLike] = []
        self.rfi_stationary_ang_sep: List[ArrayLike] = []
        self.rfi_stationary_A_app: List[ArrayLike] = []

    def addAstro(self, I: Array, ra: Array, dec: Array) -> None:
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
        ra = da.atleast_1d(ra)
        dec = da.atleast_1d(dec)
        lmn = radec_to_lmn(ra, dec, [self.ra, self.dec])
        theta = da.arcsin(da.linalg.norm(lmn[:, :-1], axis=-1))
        I_app = (
            I
            * (airy_beam(theta[:, None, None], self.freqs, self.dish_d)[:, 0, 0, :])
            ** 2
        )
        vis_ast = astro_vis(I_app, self.bl_uvw, lmn, self.freqs)

        self.ast_I.append(I)
        self.ast_lmn.append(lmn)
        self.ast_radec.append(jnp.array([ra, dec]))
        self.n_ast += len(I)

        self.vis_ast += vis_ast

    def addSatelliteRFI(
        self,
        Pv: Array,
        elevation: Array,
        inclination: Array,
        lon_asc_node: Array,
        periapsis: Array,
    ) -> None:
        """
        Add a satellite-based source of RFI to the observation.

        Parameters
        ----------
        Pv: ndarray (n_src, n_time_fine, n_freq)
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
        Pv = da.atleast_2d(Pv)
        if Pv.ndim == 2:
            Pv = da.expand_dims(Pv, axis=0)
        Pv = da.rechunk(Pv, (-1, self.time_fine_chunk, self.freq_chunk))
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
            rfi_A_app,
            distances - self.ants_uvw[None, :, :, -1],
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

    def addStationaryRFI(
        self,
        Pv: Array,
        latitude: Array,
        longitude: Array,
        elevation: Array,
    ) -> None:
        """
        Add a stationary source of RFI to the observation.

        Parameters
        ----------
        Pv: ndarray (n_src, n_time_fine, n_freq)
            Specific Emission Power in W/Hz
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
        Pv = da.rechunk(Pv, (-1, self.time_fine_chunk, self.freq_chunk))
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
            rfi_A_app,
            distances - self.ants_uvw[None, :, :, -1],
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

    def addGains(
        self,
        G0_mean: float,
        G0_std: float,
        Gt_std_amp: float,
        Gt_std_phase: float,
        random_seed=None,
    ) -> None:
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
        ).rechunk((self.time_fine_chunk, self.ant_chunk, self.freq_chunk))

    def calculate_vis(self, random_seed=None) -> Dataset:
        """
        Calculate the total gain amplified visibilities,  average down to the
        originally defined sampling rate and add noise.
        """
        self.vis = apply_gains(
            self.vis_ast, self.vis_rfi, self.gains_ants, self.a1, self.a2
        ).rechunk((self.time_fine_chunk, self.bl_chunk, self.freq_chunk))
        self.vis_avg = time_avg(self.vis, self.n_int_samples).rechunk(
            ((self.time_chunk, self.bl_chunk, self.freq_chunk))
        )
        self.vis_obs, self.noise_data = add_noise(
            self.vis_avg,
            self.noise_std,
            random_seed if random_seed else self.random_seed,
        )
        self.dataset = construct_observation_ds(self)
        return self.dataset

    def write_to_zarr(
        self,
        path: MutableMapping | str | PathLike[str] | None = "Observation",
        overwrite: bool = False,
    ):
        """
        Write the visibilities to disk using zarr format.
        """
        mode: Literal["w", "w-", "a", "r+"] = "w" if overwrite else "w-"
        self.dataset.to_zarr(path, mode=mode)

    def write_to_ms(self, path: str = "Observation.ms", overwrite: bool = False):
        """
        Write the visibilities to disk using Measurement Set format.
        """
        write_ms(self.dataset, path, overwrite=overwrite)
