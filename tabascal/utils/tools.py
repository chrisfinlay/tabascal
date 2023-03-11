import argparse
import os
from pathlib import Path

import dask.array as da
import h5py
import jax.numpy as jnp
import numpy as np
import xarray as xr

pkg_dir = Path(__file__).parent.absolute()


def uniform_points_disk(radius: float, n_src: int, seed=None):
    """
    Generate uniformly distributed random points on a disk.

    Parameters:
    -----------
    radius: float
        Radius of the disk.
    n_src: int
        The number of sources/points to generate.
    seed: int
        Random number generator seed/key.

    Returns:
    --------
    points: array_like (2, n_src)
        The coordinate positions of the random points centred on (0,0).
    """
    rng = np.random.default_rng(seed)
    r = jnp.sqrt(rng.uniform(low=0.0, high=radius, size=(n_src,)))
    theta = rng.uniform(low=0.0, high=2.0 * jnp.pi, size=(n_src,))

    return r * jnp.array([jnp.cos(theta), jnp.sin(theta)])


def beam_size(diameter: float, frequency: float, fwhp=True):
    """
    Calculate the beam size of an antenna or an array. For an array use
    fwhp = True. This assumes an Airy disk primary beam pattern.

    Parameters:
    -----------
    diameter: float
        Diameter of the dish or baseline in metres.
    frequency: float
        Observation frequency in Hertz.
    fwhp: bool
        True if you want the field of view to be the full width at half power
        and False if you want the first null.

    Returns:
    --------
    beam_width: float
        beam_width in degrees.
    """
    diameter = jnp.asarray(diameter)
    frequency = jnp.asarray(frequency)

    c = 299792458.0
    lamda = c / frequency
    beam_width = 1.02 * lamda / diameter if fwhp else 1.22 * lamda / diameter

    return jnp.rad2deg(beam_width)


def generate_random_sky(
    n_src: int,
    mean_I,
    freqs: jnp.ndarray,
    spec_idx_mean: float = 0.7,
    spec_idx_std: float = 0.2,
    fov: float = 1.0,
    beam_width=0.0,
    random_seed: int = None,
):
    """
    Generate uniformly distributed point sources inside the field of view with
    an exponential intensity distribution. Setting the beam width will make
    sure souces are separated by 5 beam widths apart.

    Parameters:
    -----------
    n_src: int
        Number of sources to generate.
    mean_I: float
        Mean intensity of the sources.
    freqs: array_like (n_freq,)
        Frequencies to generate the sources at.
    spec_idx_mean: float
        Mean spectral index of the sources.
    spec_idx_std: float
        Standard deviation of the spectral index of the sources.
    fov: float
        Field of view to generate positions within. Same units as beam_width.
    beam_width: float
        Width of the resolving beam to ensure sources do not overlap. Sources will be separated by >5*beam_width. Same units as fov.
    random_seed: int
        Random number generator seed/key.

    Returns:
    --------
    I: array_like (n_src, n_freq)
        The sources intensities.
    delta_ra: array_like
        The sources right ascensions relative to (0,0).
    delta_dec: array_like
        The sources declinations relative to (0,0).
    """
    rng = np.random.default_rng(random_seed)

    I = rng.exponential(scale=mean_I, size=(n_src,))
    positions = uniform_points_disk(fov / 2.0, 1, rng)
    while positions.shape[1] < n_src:
        n_sample = 2 * (n_src - positions.shape[1])
        new_positions = uniform_points_disk(fov / 2.0, n_sample, rng)
        positions = jnp.concatenate([positions, new_positions], axis=1)
        s1, s2 = jnp.triu_indices(positions.shape[1], 1)
        d = jnp.linalg.norm(positions[:, s1] - positions[:, s2], axis=0)
        idx = jnp.where(d < 5 * beam_width)[0]
        remove_source_idx = jnp.unique(jnp.concatenate([s1[idx], s2[idx]]))
        positions = jnp.delete(positions, remove_source_idx, axis=1)

    d_ra, d_dec = positions[:, :n_src]

    spectral_indices = rng.normal(loc=spec_idx_mean, scale=spec_idx_std, size=(n_src,))
    I = I[:, None] * ((freqs / freqs[0]) ** (-spectral_indices))[:, None]

    return I, d_ra, d_dec


def save_observations(file_path: str, observations: list):
    """
    Save a list of observations to HDF5 file format.

    Parameters:
    -----------
    file_path: str
        File path (including file name) location to save the HDF5 file.
    observations: list
        List of observation class instances.
    """
    with h5py.File(file_path + ".h5", "w") as fp:
        for i, obs in enumerate(observations):
            fp[f"track{i}/n_ant"] = obs.n_ant
            fp[f"track{i}/n_bl"] = obs.n_bl
            fp[f"track{i}/n_time"] = obs.n_time
            fp[f"track{i}/n_freq"] = obs.n_freq
            fp[f"track{i}/n_int_samples"] = obs.n_int_samples
            fp[f"track{i}/target"] = [obs.ra, obs.dec]
            fp[f"track{i}/int_time"] = obs.int_time
            fp[f"track{i}/times"] = obs.times
            fp[f"track{i}/times_fine"] = obs.times_fine
            fp[f"track{i}/freqs"] = obs.freqs
            fp[f"track{i}/ants_ENU"] = obs.ENU
            fp[f"track{i}/ants_XYZ"] = obs.ants_xyz
            fp[f"track{i}/ants_UVW"] = obs.ants_uvw
            fp[f"track{i}/bl_UVW"] = obs.bl_uvw

            fp[f"track{i}/latitude"] = obs.latitude
            fp[f"track{i}/longitude"] = obs.longitude
            fp[f"track{i}/elevation"] = obs.elevation
            fp[f"track{i}/antenna1"] = obs.a1
            fp[f"track{i}/antenna2"] = obs.a2
            fp[f"track{i}/vis_ast"] = obs.vis_ast
            fp[f"track{i}/vis_rfi"] = obs.vis_rfi
            fp[f"track{i}/vis_obs"] = obs.vis_obs
            fp[f"track{i}/noise_std"] = obs.noise_std
            fp[f"track{i}/noise_data"] = obs.noise_data

            fp[f"track{i}/gains"] = obs.gains_ants

            fp[f"track{i}/ast_I"] = jnp.array([x for x in obs.ast_I.values()])
            fp[f"track{i}/ast_radec"] = jnp.array([x for x in obs.ast_radec.values()])

            fp[f"track{i}/rfi_XYZ"] = jnp.array([x for x in obs.rfi_xyz.values()])
            fp[f"track{i}/rfi_A"] = jnp.array([x for x in obs.rfi_A_app.values()])
            fp[f"track{i}/rfi_orbit"] = jnp.array([x for x in obs.rfi_orbit.values()])
            fp[f"track{i}/rfi_geo"] = jnp.array([x for x in obs.rfi_geo.values()])
            fp[f"track{i}/rfi_sat_idx"] = jnp.array([x for x in obs.rfi_orbit.keys()])
            fp[f"track{i}/rfi_stat_idx"] = jnp.array([x for x in obs.rfi_geo.keys()])


def load_antennas(telescope="MeerKAT"):
    """
    Load the ENU coordinates for a telescope. Currently only MeerKAT is
    included.

    Parameters:
    -----------
    telescope: str
        The name of the telescope.

    Returns:
    --------
    enu: array_like (n_ant, 3)
        The East, North, Up coordinates of each antenna relative to a reference
        position.
    """
    if telescope == "MeerKAT":
        enu = np.loadtxt(os.path.join(pkg_dir, "../data/Meerkat.enu.txt"))
    else:
        print("Only MeerKAT antennas are currentyl available.")
        enu = None
    return enu


def str2bool(v: str):
    """
    Convert string to boolean.

    Parameters:
    -----------
    v: str
        String to convert to boolean.

    Raises:
    -------
        argparse.ArgumentTypeError: If the string is not a boolean.

    Returns:
    --------
    bool: bool
        The boolean value of the string.
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def construct_observation_ds(obs):
    """Construct a dataset for a single observation."""

    visibility_data = get_visibility_data(obs)
    optional_data = get_optional_data(obs)

    astromonical_source_data = get_astromonical_source_data(obs)
    rfi_satellite_data = get_satellite_rfi_data(obs)
    rfi_stationary_data = get_stationary_rfi_data(obs)

    coordinates = get_coordinates(obs)
    attributes = get_observation_attributes(obs)

    data = {
        **visibility_data,
        **optional_data,
        **astromonical_source_data,
        **rfi_satellite_data,
        **rfi_stationary_data,
    }

    ds = xr.Dataset(data_vars=data, coords=coordinates, attrs=attributes)

    return ds


def get_visibility_data(obs):
    vis_data = {
        "vis_obs": (["time", "bl", "freq"], obs.vis_obs),
        "vis_ast": (["time_fine", "bl", "freq"], obs.vis_ast),
        "vis_rfi": (["time_fine", "bl", "freq"], obs.vis_rfi),
    }
    if obs.backend == "jax":
        vis_data = {k: (v[0], da.from_array(v[1])) for k, v in vis_data.items()}
    return vis_data


def get_optional_data(obs):
    opt_data = {
        # Antenna indices
        "antenna1": (["bl"], obs.a1),
        "antenna2": (["bl"], obs.a2),
        # Antenna positions
        "ants_enu": (["ant", "enu"], obs.ENU),
        "ants_xyz": (["time_fine", "ant", "xyz"], obs.ants_xyz),
        "ants_uvw": (["time_fine", "ant", "uvw"], obs.ants_uvw),
        "bl_uvw": (["time_fine", "bl", "uvw"], obs.bl_uvw),
        # Noise parameters
        "SEFD": (["freq"], obs.SEFD),
        "noise_std": (["freq"], obs.noise_std),
        "noise_data": (["time", "bl", "freq"], obs.noise_data),
        # Gain parameters
        "gains_ants": (["time_fine", "ant", "freq"], obs.gains_ants),
    }
    if obs.backend == "jax":
        opt_data = {k: (v[0], da.from_array(v[1])) for k, v in opt_data.items()}
    return opt_data


def get_observation_attributes(obs):
    attrs = {
        "tel_latitude": obs.latitude,
        "tel_longitude": obs.longitude,
        "tel_elevation": obs.elevation,
        "target_ra": obs.ra,
        "target_dec": obs.dec,
        "n_ant": obs.n_ant,
        "n_freq": obs.n_freq,
        "n_bl": obs.n_bl,
        "n_time": obs.n_time,
        "n_time_fine": obs.n_time_fine,
        "n_int_samples": obs.n_int_samples,
        "n_sat_src": obs.n_rfi_satellite,
        "n_stat_src": obs.n_rfi_stationary,
        "n_ast_src": obs.n_ast,
    }
    if obs.backend == "dask":
        attrs = {
            k: v.compute() if isinstance(v, da.Array) else v for k, v in attrs.items()
        }
    return attrs


def get_coordinates(obs):
    coords = {
        "time": obs.times,
        "time_fine": obs.times_fine,
        "freq": obs.freqs,
        "bl": np.arange(obs.n_bl),
        "ant": np.arange(obs.n_ant),
        "uvw": np.array(["u", "v", "w"]),
        "lmn": np.array(["l", "m", "n"]),
        "xyz": np.array(["x", "y", "z"]),
        "enu": np.array(["east", "north", "up"]),
        "radec": np.array(["ra", "dec"]),
        "geo": np.array(["latitude", "longitude", "elevation"]),
    }
    if obs.backend == "jax":
        coords = {k: da.from_array(v) for k, v in coords.items()}
    return coords


def get_astromonical_source_data(obs):
    if obs.n_ast > 0:
        ast_data = {
            # Astronomical source parameters
            "ast_I": (["ast_src", "freq"], da.concatenate(obs.ast_I, axis=0)),
            "ast_lmn": (["ast_src", "lmn"], da.concatenate(obs.ast_lmn, axis=0)),
            "ast_radec": (
                ["ast_src", "radec"],
                da.concatenate(obs.ast_radec, axis=0).T,
            ),
        }
    else:
        ast_data = {}
    return ast_data


def get_satellite_rfi_data(obs):
    if obs.n_rfi_satellite > 0:
        rfi_sat = {
            # Satellite RFI parameters
            "rfi_sat_A": (
                ["sat_src", "time_fine", "ant", "freq"],
                da.concatenate(obs.rfi_satellite_A_app, axis=0),
            ),
            "rfi_sat_xyz": (
                ["sat_src", "time_fine", "xyz"],
                da.concatenate(obs.rfi_satellite_xyz, axis=0),
            ),
            "rfi_sat_ang_sep": (
                ["sat_src", "time_fine", "ant"],
                da.concatenate(obs.rfi_satellite_ang_sep, axis=0),
            ),
            "rfi_sat_orbit": (
                ["sat_src", "orbit"],
                da.concatenate(obs.rfi_satellite_orbit, axis=0),
            ),
        }
    else:
        rfi_sat = {}
    return rfi_sat


def get_stationary_rfi_data(obs):
    if obs.n_rfi_stationary > 0:
        rfi_stat = {
            # Stationary RFI parameters
            "rfi_stat_A": (
                ["stat_src", "time_fine", "ant", "freq"],
                da.concatenate(obs.rfi_stationary_A_app, axis=0),
            ),
            "rfi_stat_xyz": (
                ["stat_src", "time_fine", "xyz"],
                da.concatenate(obs.rfi_stationary_xyz, axis=0),
            ),
            "rfi_stat_ang_sep": (
                ["stat_src", "time_fine", "ant"],
                da.concatenate(obs.rfi_stationary_ang_sep, axis=0),
            ),
            "rfi_stat_geo": (
                ["stat_src", "geo"],
                da.concatenate(obs.rfi_stationary_geo, axis=0),
            ),
        }
    else:
        rfi_stat = {}
    return rfi_stat


###################################################################################################
# Progress bar for JAX scan
# Credit to https://www.jeremiecoullon.com/2021/01/29/jax_progress_bar/

from jax import lax
from jax.experimental import host_callback
from tqdm import tqdm


def progress_bar_scan(num_samples: int, message=None):
    "Progress bar for a JAX scan"
    if message is None:
        message = f"Running for {num_samples:,} iterations"
    tqdm_bars = {}
    if num_samples > 20:
        print_rate = int(num_samples / 20)
    else:
        print_rate = 1  # if you run the sampler for less than 20 iterations

    remainder = num_samples % print_rate

    def _define_tqdm(arg, transform):
        tqdm_bars[0] = tqdm(range(num_samples))
        tqdm_bars[0].set_description(message, refresh=False)

    def _update_tqdm(arg, transform):
        tqdm_bars[0].update(arg)

    def _update_progress_bar(iter_num):
        "Updates tqdm progress bar of a JAX scan or loop"
        _ = lax.cond(
            iter_num == 0,
            lambda _: host_callback.id_tap(_define_tqdm, None, result=iter_num),
            lambda _: iter_num,
            operand=None,
        )

        _ = lax.cond(
            # update tqdm every multiple of `print_rate` except at the end
            (iter_num % print_rate == 0) & (iter_num != num_samples - remainder),
            lambda _: host_callback.id_tap(_update_tqdm, print_rate, result=iter_num),
            lambda _: iter_num,
            operand=None,
        )

        _ = lax.cond(
            # update tqdm by `remainder`
            iter_num == num_samples - remainder,
            lambda _: host_callback.id_tap(_update_tqdm, remainder, result=iter_num),
            lambda _: iter_num,
            operand=None,
        )

    def _close_tqdm(arg, transform):
        tqdm_bars[0].close()

    def close_tqdm(result, iter_num):
        return lax.cond(
            iter_num == num_samples - 1,
            lambda _: host_callback.id_tap(_close_tqdm, None, result=result),
            lambda _: result,
            operand=None,
        )

    def _progress_bar_scan(func):
        """Decorator that adds a progress bar to `body_fun` used in `lax.scan`.
        Note that `body_fun` must either be looping over `np.arange(num_samples)`,
        or be looping over a tuple who's first element is `np.arange(num_samples)`
        This means that `iter_num` is the current iteration number
        """

        def wrapper_progress_bar(carry, x):
            if type(x) is tuple:
                iter_num, *_ = x
            else:
                iter_num = x
            _update_progress_bar(iter_num)
            result = func(carry, x)
            return close_tqdm(result, iter_num)

        return wrapper_progress_bar

    return _progress_bar_scan
