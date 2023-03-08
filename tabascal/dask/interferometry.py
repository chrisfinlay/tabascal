import dask.array as da
import xarray as xr
from dask import delayed

from tabascal import interferometry as itf


def astro_vis(sources: da.Array, uvw: da.Array, lmn: da.Array, freqs: da.Array):
    """Calculate visibilities from sources, uvw, lmn, and freqs.

    Parameters:
    -----------
    sources: da.Array (n_freq, n_src)
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
    n_time, n_bl, _ = uvw.shape
    n_freq, _ = sources.shape

    time_chunk = uvw.chunks[0][0]
    bl_chunk = uvw.chunks[1][0]
    freq_chunk = freqs.chunks[0][0]

    sim_data = xr.Dataset(
        {
            "I": (["freq", "src"], sources),
            "uvw": (["time", "bl", "space"], uvw),
            "lmn": (["src", "space"], lmn),
            "freqs": (["freq"], freqs),
        }
    )
    ds_out = xr.Dataset(
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

    def astro_vis_ds(ds):
        vis = delayed(itf.astro_vis)(
            ds.I.data, ds.uvw.data, ds.lmn.data, ds.freqs.data
        ).compute()
        ds_out = xr.Dataset({"vis": (["time", "bl", "freq"], vis)})
        return ds_out

    ds = xr.map_blocks(astro_vis_ds, sim_data, template=ds_out)

    return ds.vis


def rfi_vis(
    app_amplitude: da.Array,
    c_distances: da.Array,
    freqs: da.Array,
    a1: da.Array,
    a2: da.Array,
):
    """Calculate visibilities from sources, uvw, lmn, and freqs.

    Parameters:
    -----------
    app_amplitude: da.Array (n_time, n_ant, n_freq, n_src)
        Apparent amplitude of the sources at each antenna.
    c_distances: da.Array (n_time, n_ant, n_src)
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
    n_time, _, n_freq, _ = app_amplitude.shape
    n_bl = a1.shape[0]

    time_chunk = app_amplitude.chunks[0][0]
    freq_chunk = app_amplitude.chunks[2][0]
    bl_chunk = a1.chunks[0][0]

    sim_data = xr.Dataset(
        {
            "app_amplitude": (["time", "ant", "freq", "src"], app_amplitude),
            "c_distances": (["time", "ant", "src"], c_distances),
            "freqs": (["freq"], freqs),
            "a1": (["bl"], a1),
            "a2": (["bl"], a2),
        }
    )
    ds_out = xr.Dataset(
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

    def rfi_vis_ds(ds):
        vis = delayed(itf.rfi_vis)(
            ds.app_amplitude.data,
            ds.c_distances.data,
            ds.freqs.data,
            ds.a1.data,
            ds.a2.data,
        ).compute()
        ds_out = xr.Dataset({"vis": (["time", "bl", "freq"], vis)})
        return ds_out

    ds = xr.map_blocks(rfi_vis_ds, sim_data, template=ds_out)

    return ds.vis
