import dask.array as da
import numpy as np
import xarray as xr
from dask import delayed

from tabascal.jax import interferometry as itf


def astro_vis(
    sources: da.Array, uvw: da.Array, lmn: da.Array, freqs: da.Array
) -> da.Array:
    """Calculate visibilities from sources, uvw, lmn, and freqs.

    Parameters:
    -----------
    sources: da.Array (n_src, n_time, n_freq)
        Array of point source intensities in Jy.
    uvw: da.Array (ntime, n_bl, 3)
        (u,v,w) coordinates of each baseline.
    lmn: da.Array (n_src, 3)
        (l,m,n) coordinate of each source.
    freqs: da.Array (n_freq,)
        Frequencies in Hz.

    Returns:
    --------
    vis: da.Array (n_time, n_bl, n_freq)
    """
    n_time, n_bl = uvw.shape[:2]
    n_freq = freqs.shape[0]

    time_chunk, bl_chunk = uvw.chunksize[:2]
    freq_chunk = freqs.chunksize[0]

    src_n_time = sources.shape[1]
    if src_n_time != n_time and src_n_time != 1:
        ValueError(
            "The size of the time dimension for the astronomical sources must be n_time_fine or 1."
        )

    t = "time" if src_n_time is n_time else "time1"
    I = {"I": (["src", f"{t}", "freq"], sources)}

    input = xr.Dataset(
        {
            **I,
            "uvw": (["time", "bl", "space"], uvw),
            "lmn": (["src", "space"], lmn),
            "freqs": (["freq"], freqs),
        }
    )
    output = xr.Dataset(
        {
            "vis": (
                ["time", "bl", "freq"],
                da.zeros(
                    shape=(n_time, n_bl, n_freq),
                    chunks=(time_chunk, bl_chunk, freq_chunk),
                    dtype=complex,
                ),
            )
        }
    )

    def _astro_vis(ds):
        vis = delayed(itf.astro_vis)(
            ds.I.data, ds.uvw.data, ds.lmn.data, ds.freqs.data
        ).compute()
        ds_out = xr.Dataset({"vis": (["time", "bl", "freq"], vis)})
        return ds_out

    ds = xr.map_blocks(_astro_vis, input, template=output)

    return ds.vis.data


def astro_vis_gauss(
    sources: da.Array,
    major: da.Array,
    minor: da.Array,
    pos_angle: da.Array,
    uvw: da.Array,
    lmn: da.Array,
    freqs: da.Array,
) -> da.Array:
    """Calculate visibilities from sources, uvw, lmn, and freqs.

    Parameters:
    -----------
    sources: da.Array (n_src, n_time, n_freq)
        Array of point source intensities in Jy.
    shapes: array_like (n_src,)
        Array of standard deviations of the gaussian shape sources. These are
        assumed to be circular gaussians for now.
    uvw: da.Array (ntime, n_bl, 3)
        (u,v,w) coordinates of each baseline.
    lmn: da.Array (n_src, 3)
        (l,m,n) coordinate of each source.
    freqs: da.Array (n_freq,)
        Frequencies in Hz.

    Returns:
    --------
    vis: da.Array (n_time, n_bl, n_freq)
    """
    n_time, n_bl = uvw.shape[:2]
    n_freq = freqs.shape[0]

    time_chunk, bl_chunk = uvw.chunksize[:2]
    freq_chunk = freqs.chunksize[0]

    input = xr.Dataset(
        {
            "I": (["src", "time", "freq"], sources),
            "major": (["src"], major),
            "minor": (["src"], minor),
            "pos_angle": (["src"], pos_angle),
            "uvw": (["time", "bl", "space"], uvw),
            "lmn": (["src", "space"], lmn),
            "freqs": (["freq"], freqs),
        }
    )
    output = xr.Dataset(
        {
            "vis": (
                ["time", "bl", "freq"],
                da.zeros(
                    shape=(n_time, n_bl, n_freq),
                    chunks=(time_chunk, bl_chunk, freq_chunk),
                    dtype=complex,
                ),
            )
        }
    )

    def _astro_vis_gauss(ds):
        vis = delayed(itf.astro_vis_gauss)(
            ds.I.data,
            ds.major.data,
            ds.minor.data,
            ds.pos_angle.data,
            ds.uvw.data,
            ds.lmn.data,
            ds.freqs.data,
        ).compute()
        ds_out = xr.Dataset({"vis": (["time", "bl", "freq"], vis)})
        return ds_out

    ds = xr.map_blocks(_astro_vis_gauss, input, template=output)

    return ds.vis.data


def astro_vis_exp(
    sources: da.Array, shapes: da.Array, uvw: da.Array, lmn: da.Array, freqs: da.Array
) -> da.Array:
    """Calculate visibilities from sources, uvw, lmn, and freqs.

    Parameters:
    -----------
    sources: da.Array (n_src, n_time, n_freq)
        Array of point source intensities in Jy.
    shapes: array_like (n_src,)
        Array of shape parameters for the exp sources. These are
        assumed to be circular.
    uvw: da.Array (ntime, n_bl, 3)
        (u,v,w) coordinates of each baseline.
    lmn: da.Array (n_src, 3)
        (l,m,n) coordinate of each source.
    freqs: da.Array (n_freq,)
        Frequencies in Hz.

    Returns:
    --------
    vis: da.Array (n_time, n_bl, n_freq)
    """
    n_time, n_bl = uvw.shape[:2]
    n_freq = freqs.shape[0]

    time_chunk, bl_chunk = uvw.chunksize[:2]
    freq_chunk = freqs.chunksize[0]

    input = xr.Dataset(
        {
            "I": (["src", "time", "freq"], sources),
            "sigmas": (["src"], shapes),
            "uvw": (["time", "bl", "space"], uvw),
            "lmn": (["src", "space"], lmn),
            "freqs": (["freq"], freqs),
        }
    )
    output = xr.Dataset(
        {
            "vis": (
                ["time", "bl", "freq"],
                da.zeros(
                    shape=(n_time, n_bl, n_freq),
                    chunks=(time_chunk, bl_chunk, freq_chunk),
                    dtype=complex,
                ),
            )
        }
    )

    def _astro_vis_exp(ds):
        vis = delayed(itf.astro_vis_exp)(
            ds.I.data, ds.sigmas.data, ds.uvw.data, ds.lmn.data, ds.freqs.data
        ).compute()
        ds_out = xr.Dataset({"vis": (["time", "bl", "freq"], vis)})
        return ds_out

    ds = xr.map_blocks(_astro_vis_exp, input, template=output)

    return ds.vis.data


def rfi_vis(
    app_amplitude: da.Array,
    c_distances: da.Array,
    freqs: da.Array,
    a1: da.Array,
    a2: da.Array,
) -> da.Array:
    """Calculate visibilities from sources, uvw, lmn, and freqs.

    Parameters:
    -----------
    app_amplitude: da.Array (n_src, n_time, n_int, n_ant, n_freq)
        Apparent amplitude of the sources at each antenna.
    c_distances: da.Array (n_src, n_time, n_int, n_ant)
        The phase corrected distances between the rfi sources and the antennas in metres.
    freqs: da.Array (n_freq,)
        Frequencies in Hz.
    a1: da.Array (n_bl,)
        Antenna 1 indexes, between 0 and n_ant-1.
    a2: da.Array (n_bl,)
        Antenna 2 indexes, between 0 and n_ant-1.

    Returns:
    --------
    vis: da.Array (n_time, n_bl, n_freq)
    """
    n_time = app_amplitude.shape[1]
    n_freq = freqs.shape[0]
    n_bl = a1.shape[0]

    time_chunk = app_amplitude.chunksize[1]
    freq_chunk = freqs.chunksize[0]
    bl_chunk = a1.chunksize[0]

    input = xr.Dataset(
        {
            "app_amplitude": (["src", "time", "int", "ant", "freq"], app_amplitude),
            "c_distances": (["src", "time", "int", "ant"], c_distances),
            "freqs": (["freq"], freqs),
            "a1": (["bl"], a1),
            "a2": (["bl"], a2),
        }
    )
    output = xr.Dataset(
        {
            "vis": (
                ["time", "bl", "freq"],
                da.zeros(
                    shape=(n_time, n_bl, n_freq),
                    chunks=(time_chunk, bl_chunk, freq_chunk),
                    dtype=complex,
                ),
            )
        }
    )

    def _rfi_vis(ds):
        vis = delayed(itf.rfi_vis)(
            ds.app_amplitude.data,
            ds.c_distances.data,
            ds.freqs.data,
            ds.a1.data,
            ds.a2.data,
        ).compute()
        ds_out = xr.Dataset({"vis": (["time", "bl", "freq"], vis)})
        return ds_out

    ds = xr.map_blocks(_rfi_vis, input, template=output)

    return ds.vis.data


def ants_to_bl(G: da.Array, a1: da.Array, a2: da.Array) -> da.Array:
    n_time, _, n_freq = G.shape
    n_bl = a1.shape[0]

    time_chunk, _, freq_chunk = G.chunksize
    bl_chunk = a1.chunksize[0]

    input = xr.Dataset(
        {"G": (["time", "ant", "freq"], G), "a1": (["bl"], a1), "a2": (["bl"], a2)}
    )

    output = xr.Dataset(
        {
            "G_bl": (
                ["time", "bl", "freq"],
                da.zeros(
                    (n_time, n_bl, n_freq), chunks=(time_chunk, bl_chunk, freq_chunk)
                ),
            )
        }
    )

    def _ants_to_bl(ds):
        G_bl = delayed(itf.ants_to_bl)(ds.G.data, ds.a1.data, ds.a2.data).compute()
        ds_out = xr.Dataset({"G_bl": (["time", "bl", "freq"], G_bl)})
        return ds_out

    ds = xr.map_blocks(_ants_to_bl, input, template=output)

    return ds.G_bl.data


ants_to_bl.__doc__ = itf.ants_to_bl.__doc__


def airy_beam(theta, freqs, dish_d):
    n_src, n_time, n_ant = theta.shape
    n_freq = freqs.shape[0]

    src_chunk, time_chunk, ant_chunk = theta.chunksize
    freq_chunk = freqs.chunksize[0]

    input = xr.Dataset(
        {
            "theta": (["src", "time", "ant"], theta),
            "freqs": (["freq"], freqs),
            "dish_d": (["space_0"], da.from_array([dish_d])),
        }
    )

    output = xr.Dataset(
        {
            "beam": (
                ["src", "time", "ant", "freq"],
                da.zeros(
                    (n_src, n_time, n_ant, n_freq),
                    chunks=(src_chunk, time_chunk, ant_chunk, freq_chunk),
                ),
            )
        }
    )

    def _airy_beam(ds):
        beam = delayed(itf.airy_beam)(
            ds.theta.data, ds.freqs.data, ds.dish_d.data
        ).compute()
        ds_out = xr.Dataset({"beam": (["src", "time", "ant", "freq"], beam)})
        return ds_out

    ds = xr.map_blocks(_airy_beam, input, template=output)

    return ds.beam.data


airy_beam.__doc__ = itf.airy_beam.__doc__


def Pv_to_Sv(Pv, d):
    n_src, _, n_freq = Pv.shape
    n_time, n_ant = d.shape[1:]

    src_chunk, _, freq_chunk = Pv.chunksize
    time_chunk, ant_chunk = d.chunksize[1:]

    input = xr.Dataset(
        {"Pv": (["src", "time", "freq"], Pv), "d": (["src", "time", "ant"], d)}
    )

    output = xr.Dataset(
        {
            "Sv": (
                ["src", "time", "ant", "freq"],
                da.zeros(
                    (n_src, n_time, n_ant, n_freq),
                    chunks=(src_chunk, time_chunk, ant_chunk, freq_chunk),
                ),
            )
        }
    )

    def _Pv_to_Sv(ds):
        Sv = delayed(itf.Pv_to_Sv)(ds.Pv.data, ds.d.data).compute()
        ds_out = xr.Dataset({"Sv": (["src", "time", "ant", "freq"], Sv)})
        return ds_out

    ds = xr.map_blocks(_Pv_to_Sv, input, template=output)

    return ds.Sv.data


Pv_to_Sv.__doc__ = itf.Pv_to_Sv.__doc__


def add_noise(vis: da.Array, noise_std: float, key: int):
    rng = np.random.default_rng(key)
    noise = rng.normal(0, noise_std, size=vis.shape) + 1.0j * rng.normal(
        0, noise_std, size=vis.shape
    )
    return vis + noise, noise


def SEFD_to_noise_std(SEFD, chan_width, t_int):
    noise_std = SEFD / da.sqrt(chan_width * t_int)
    return noise_std


# def int_sample_times(times, n_int_samples, int_time):
#     times_fine = da.from_array(
#         itf.int_sample_times(times, n_int_samples, int_time), chunks=times.chunksize
#     )
#     return times_fine


def int_sample_times(times, n_int_samples: int, int_time: float = None):

    n_time = len(times)
    time_range = times[-1] - times[0]
    int_time = time_range / (n_time - 1) if not int_time else int_time
    n_time_fine = n_time * n_int_samples
    times_fine = da.linspace(
        -int_time / 2, time_range + int_time / 2, n_time_fine, endpoint=False
    )

    return times[0] + int_time / (2 * n_int_samples) + times_fine


def generate_gains(
    G0_mean: float,
    G0_std: float,
    Gt_std_amp: float,
    Gt_std_phase: float,
    times: np.ndarray,
    n_ant: int,
    n_freq: int,
    random_seed: int,
) -> da.Array:
    rng = np.random.default_rng(random_seed)
    times = times[:, None, None] - times[0]

    # Generate the initial gain values
    G0 = G0_mean * da.exp(
        1.0j * rng.uniform(low=-np.pi / 2, high=np.pi / 2, size=(1, n_ant, n_freq))
    ) + (
        rng.normal(scale=G0_std, size=(1, n_ant, n_freq))
        + 1.0j * rng.normal(scale=G0_std, size=(1, n_ant, n_freq))
    )

    # Generate the gain variations
    gain_amp = rng.normal(scale=Gt_std_amp, size=(1, n_ant, 1)) * times
    gain_phase = rng.normal(scale=Gt_std_phase, size=(1, n_ant, 1)) * times
    # Generate the gain time series
    gain_ants = G0 + gain_amp * da.exp(1.0j * gain_phase)
    # Set the gain on the last antenna to have zero phase (reference antenna)
    gain_ants[:, -1, :] = da.abs(gain_ants[:, -1, :])

    return gain_ants


def generate_fourier_gains(
    G0_mean: float,
    G0_std: float,
    Gt_std_amp: float,
    Gt_std_phase: float,
    Gt_corr_amp: float,
    Gt_corr_phase: float,
    times_jd: np.ndarray,
    n_ant: int,
    n_freq: int,
    random_seed: int,
) -> da.Array:

    rng = np.random.default_rng(random_seed)
    times = times_jd[None, :, None, None] * 24

    # Generate the initial gain values
    G0 = G0_mean * da.exp(
        1.0j * rng.uniform(low=-np.pi / 2, high=np.pi / 2, size=(1, n_ant, n_freq))
    ) + (
        rng.normal(scale=G0_std / 100, size=(1, n_ant, n_freq))
        + 1.0j * rng.normal(scale=G0_std / 100, size=(1, n_ant, n_freq))
    )

    N = 1000
    mode_size = (N, 1, n_ant, 1)

    # Generate the gain variations
    gain_amp = np.mean(
        rng.normal(scale=Gt_std_amp / 100, size=mode_size)
        * np.cos(
            2
            * np.pi
            * rng.normal(scale=1 / (2 * np.pi * Gt_corr_amp), size=mode_size)
            * times
            + rng.uniform(low=0.0, high=2 * np.pi, size=mode_size)
        ),
        axis=0,
    ) * np.sqrt(2 * N)
    gain_phase = np.mean(
        rng.normal(scale=np.deg2rad(Gt_std_phase), size=mode_size)
        * np.cos(
            2
            * np.pi
            * rng.normal(scale=1 / (2 * np.pi * Gt_corr_phase), size=mode_size)
            * times
            + rng.uniform(low=0.0, high=2 * np.pi, size=mode_size)
        ),
        axis=0,
    ) * np.sqrt(2 * N)
    # Generate the gain time series
    gain_ants = G0 + gain_amp * da.exp(1.0j * gain_phase)
    # Set the gain on the last antenna to have zero phase (reference antenna)
    gain_ants[:, -1, :] = da.abs(gain_ants[:, -1, :])

    return gain_ants


def apply_gains(
    vis_ast: da.Array,
    vis_rfi: da.Array,
    gains: da.Array,
    a1: da.Array,
    a2: da.Array,
) -> da.Array:
    n_time, n_bl, n_freq = vis_ast.shape

    time_chunk, bl_chunk, freq_chunk = vis_ast.chunksize

    input = xr.Dataset(
        {
            "vis_ast": (["time", "bl", "freq"], vis_ast),
            "vis_rfi": (["time", "bl", "freq"], vis_rfi),
            "gains": (["time", "ant", "freq"], gains),
            "a1": (["bl"], a1),
            "a2": (["bl"], a2),
        }
    )

    output = xr.Dataset(
        {
            "vis_obs": (
                ["time", "bl", "freq"],
                da.zeros(
                    (n_time, n_bl, n_freq),
                    chunks=(time_chunk, bl_chunk, freq_chunk),
                    dtype=vis_ast.dtype,
                ),
            )
        }
    )

    def _apply_gains(ds):
        vis_obs = delayed(itf.apply_gains)(
            ds.vis_ast.data, ds.vis_rfi.data, ds.gains.data, ds.a1.data, ds.a2.data
        ).compute()
        ds_out = xr.Dataset({"vis_obs": (["time", "bl", "freq"], vis_obs)})
        return ds_out

    ds = xr.map_blocks(_apply_gains, input, template=output)

    return ds.vis_obs.data


def time_avg(vis, n_int_samples):
    return da.mean(
        da.reshape(vis, (-1, n_int_samples, vis.shape[1], vis.shape[2])), axis=1
    )
