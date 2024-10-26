from tabascal.jax import coordinates as coord

import dask.array as da
from dask import delayed
import xarray as xr


def radec_to_lmn(ra: da.Array, dec: da.Array, phase_centre: da.Array) -> da.Array:
    n_src = ra.shape[0]

    src_chunk = ra.chunksize[0]

    input = xr.Dataset(
        {
            "ra": (["src"], ra),
            "dec": (["src"], dec),
            "phase_centre": (["cel_space"], phase_centre),
        }
    )
    output = xr.Dataset(
        {
            "lmn": (
                ["src", "lmn_space"],
                da.zeros(
                    shape=(n_src, 3),
                    chunks=(src_chunk, 3),
                    dtype=float,
                ),
            )
        }
    )

    def _radec_to_lmn(ds):
        lmn = delayed(coord.radec_to_lmn, pure=True)(
            ds.ra.data, ds.dec.data, ds.phase_centre.data
        ).compute()
        ds_out = xr.Dataset({"lmn": (["src", "lmn_space"], lmn)})
        return ds_out

    ds = xr.map_blocks(_radec_to_lmn, input, template=output)

    return ds.lmn.data


radec_to_lmn.__doc__ = coord.radec_to_lmn.__doc__


def radec_to_XYZ(ra: da.Array, dec: da.Array) -> da.Array:
    n_src = ra.shape[0]

    src_chunk = ra.chunksize[0]

    input = xr.Dataset(
        {
            "ra": (["src"], ra),
            "dec": (["src"], dec),
        }
    )
    output = xr.Dataset(
        {
            "XYZ": (
                ["src", "space"],
                da.zeros(
                    shape=(n_src, 3),
                    chunks=(src_chunk, 3),
                    dtype=float,
                ),
            )
        }
    )

    def _radec_to_XYZ(ds):
        XYZ = delayed(coord.radec_to_XYZ, pure=True)(ds.ra.data, ds.dec.data).compute()
        ds_out = xr.Dataset({"XYZ": (["src", "space"], XYZ)})
        return ds_out

    ds = xr.map_blocks(_radec_to_XYZ, input, template=output)

    return ds.XYZ.data


radec_to_XYZ.__doc__ = coord.radec_to_XYZ.__doc__


def ENU_to_GEO(geo_ref: da.Array, ENU: da.Array) -> da.Array:
    n_ant = ENU.shape[0]

    ant_chunk = ENU.chunksize[0]

    input = xr.Dataset(
        {
            "geo_ref": (["geo_space"], geo_ref),
            "ENU": (["ant", "space"], ENU),
        }
    )
    output = xr.Dataset(
        {
            "GEO": (
                ["bl", "space"],
                da.zeros(
                    shape=(n_ant, 3),
                    chunks=(ant_chunk, 3),
                    dtype=float,
                ),
            )
        }
    )

    def _ENU_to_GEO(ds):
        GEO = delayed(coord.ENU_to_GEO, pure=True)(
            ds.geo_ref.data, ds.ENU.data
        ).compute()
        ds_out = xr.Dataset({"GEO": (["ant", "space"], GEO)})
        return ds_out

    ds = xr.map_blocks(_ENU_to_GEO, input, template=output)

    return ds.GEO.data


ENU_to_GEO.__doc__ = coord.ENU_to_GEO.__doc__


def GEO_to_XYZ(geo: da.Array, times: da.Array) -> da.Array:
    n_time = geo.shape[0]

    time_chunk = geo.chunksize[0]

    input = xr.Dataset(
        {
            "geo": (["time", "space"], geo),
            "times": (["time"], times),
        }
    )
    output = xr.Dataset(
        {
            "XYZ": (
                ["time", "space"],
                da.zeros(
                    shape=(n_time, 3),
                    chunks=(time_chunk, 3),
                    dtype=float,
                ),
            )
        }
    )

    def _GEO_to_XYZ(ds):
        XYZ = delayed(coord.GEO_to_XYZ, pure=True)(ds.geo.data, ds.times.data).compute()
        ds_out = xr.Dataset({"XYZ": (["time", "space"], XYZ)})
        return ds_out

    ds = xr.map_blocks(_GEO_to_XYZ, input, template=output)

    return ds.XYZ.data


GEO_to_XYZ.__doc__ = coord.GEO_to_XYZ.__doc__


def GEO_to_XYZ_vmap0(geo: da.Array, times: da.Array) -> da.Array:
    n_src, n_time = geo.shape[:2]

    src_chunk = geo.chunks[0][0]
    time_chunk = geo.chunks[1][0]

    input = xr.Dataset(
        {
            "geo": (["src", "time", "space"], geo),
            "times": (["time"], times),
        }
    )
    output = xr.Dataset(
        {
            "XYZ": (
                ["src", "time", "space"],
                da.zeros(
                    shape=(n_src, n_time, 3),
                    chunks=(src_chunk, time_chunk, 3),
                    dtype=float,
                ),
            )
        }
    )

    def _GEO_to_XYZ(ds):
        XYZ = delayed(coord.GEO_to_XYZ_vmap0, pure=True)(
            ds.geo.data, ds.times.data
        ).compute()
        ds_out = xr.Dataset({"XYZ": (["src", "time", "space"], XYZ)})
        return ds_out

    ds = xr.map_blocks(_GEO_to_XYZ, input, template=output)

    return ds.XYZ.data


GEO_to_XYZ_vmap0.__doc__ = coord.GEO_to_XYZ_vmap0.__doc__


def GEO_to_XYZ_vmap1(geo: da.Array, times: da.Array) -> da.Array:
    n_time, n_ant = geo.shape[:2]

    time_chunk, ant_chunk = geo.chunksize[:2]

    input = xr.Dataset(
        {
            "geo": (["time", "ant", "space"], geo),
            "times": (["time"], times),
        }
    )
    output = xr.Dataset(
        {
            "XYZ": (
                ["time", "ant", "space"],
                da.zeros(
                    shape=(n_time, n_ant, 3),
                    chunks=(time_chunk, ant_chunk, 3),
                    dtype=float,
                ),
            )
        }
    )

    def _GEO_to_XYZ(ds):
        XYZ = delayed(coord.GEO_to_XYZ_vmap1, pure=True)(
            ds.geo.data, ds.times.data
        ).compute()
        ds_out = xr.Dataset({"XYZ": (["time", "ant", "space"], XYZ)})
        return ds_out

    ds = xr.map_blocks(_GEO_to_XYZ, input, template=output)

    return ds.XYZ.data


GEO_to_XYZ_vmap1.__doc__ = coord.GEO_to_XYZ_vmap1.__doc__


def ITRF_to_XYZ(itrf: da.Array, gsa: da.Array) -> da.Array:
    n_time = gsa.shape[0]
    n_ant = itrf.shape[0]

    time_chunk = gsa.chunksize[0]
    ant_chunk = itrf.chunksize[0]

    input = xr.Dataset(
        {
            "itrf": (["ant", "space"], itrf),
            "gsa": (["time"], gsa),
        }
    )
    output = xr.Dataset(
        {
            "xyz": (
                ["time", "ant", "space"],
                da.zeros(
                    shape=(n_time, n_ant, 3),
                    chunks=(time_chunk, ant_chunk, 3),
                    dtype=float,
                ),
            )
        }
    )

    def _ITRF_to_XYZ(ds):
        XYZ = delayed(coord.itrf_to_xyz, pure=True)(
            ds.itrf.data, ds.gsa.data
        ).compute()
        ds_out = xr.Dataset({"xyz": (["time", "ant", "space"], XYZ)})
        return ds_out

    ds = xr.map_blocks(_ITRF_to_XYZ, input, template=output)

    return ds.xyz.data


ITRF_to_XYZ.__doc__ = coord.itrf_to_xyz.__doc__

def ENU_to_ITRF(ENU: da.Array, lat: da.Array, lon: da.Array, el: da.Array) -> da.Array:
    n_ant = ENU.shape[0]

    ant_chunk = ENU.chunksize[0]

    input = xr.Dataset(
        {
            "ENU": (["ant", "space"], ENU),
            "lat": lat,
            "lon": lon,
            "el": el,
            
        }
    )
    output = xr.Dataset(
        {
            "ITRF": (
                ["ant", "space"],
                da.zeros(
                    shape=(n_ant, 3),
                    chunks=(ant_chunk, 3),
                    dtype=float,
                ),
            )
        }
    )

    def _ENU_to_ITRF(ds):
        ITRF = delayed(coord.enu_to_itrf, pure=True)(
            ds.ENU.data, ds.lat.data, ds.lon.data, ds.el.data,
        ).compute()
        ds_out = xr.Dataset({"ITRF": (["ant", "space"], ITRF)})
        return ds_out

    ds = xr.map_blocks(_ENU_to_ITRF, input, template=output)

    return ds.ITRF.data


ENU_to_ITRF.__doc__ = coord.enu_to_itrf.__doc__


def ITRF_to_UVW(
    itrf: da.Array,
    h0: da.Array,
    dec: da.Array,
) -> da.Array:
    n_ant = itrf.shape[0]
    n_time = h0.shape[0]

    ant_chunk = itrf.chunksize[0]
    time_chunk = h0.chunksize[0]

    input = xr.Dataset(
        {
            "itrf": (["ant", "space"], itrf),
            "h0": (["time"], da.atleast_1d(h0)),
            "dec": (["cel_space_1"], da.atleast_1d(dec)),
        }
    )

    output = xr.Dataset(
        {
            "uvw": (
                ["time", "ant", "space"],
                da.zeros(
                    shape=(n_time, n_ant, 3),
                    chunks=(time_chunk, ant_chunk, 3),
                    dtype=float,
                ),
            )
        }
    )

    def _ITRF_to_UVW(ds):
        uvw = delayed(coord.itrf_to_uvw, pure=True)(
            ds.itrf.data,
            ds.h0.data,
            ds.dec.data,
        ).compute()
        ds_out = xr.Dataset({"uvw": (["time", "ant", "space"], uvw)})
        return ds_out

    ds = xr.map_blocks(_ITRF_to_UVW, input, template=output)

    return ds.uvw.data


ITRF_to_UVW.__doc__ = coord.itrf_to_uvw.__doc__

def ENU_to_UVW(
    enu: da.Array,
    latitude: da.Array,
    longitude: da.Array,
    elevation: da.Array,
    ra: da.Array,
    dec: da.Array,
    times: da.Array,
) -> da.Array:
    n_ant = enu.shape[0]
    n_time = times.shape[0]

    ant_chunk = enu.chunksize[0]
    time_chunk = times.chunksize[0]

    input = xr.Dataset(
        {
            "enu": (["ant", "space"], enu),
            "latitude": (["geo_space_0"], da.from_array([latitude])),
            "longitude": (["geo_space_1"], da.from_array([longitude])),
            "elevation": (["geo_space_1"], da.from_array([elevation])),
            "ra": (["cel_space_0"], da.from_array([ra])),
            "dec": (["cel_space_1"], da.from_array([dec])),
            "times": (["time"], times),
        }
    )

    output = xr.Dataset(
        {
            "uvw": (
                ["time", "ant", "space"],
                da.zeros(
                    shape=(n_time, n_ant, 3),
                    chunks=(time_chunk, ant_chunk, 3),
                    dtype=float,
                ),
            )
        }
    )

    def _ENU_to_UVW(ds):
        uvw = delayed(coord.enu_to_uvw, pure=True)(
            ds.enu.data,
            ds.latitude.data,
            ds.longitude.data,
            ds.elevation.data,
            ds.ra.data,
            ds.dec.data,
            ds.times.data,
        ).compute()
        ds_out = xr.Dataset({"uvw": (["time", "ant", "space"], uvw)})
        return ds_out

    ds = xr.map_blocks(_ENU_to_UVW, input, template=output)

    return ds.uvw.data


ENU_to_UVW.__doc__ = coord.enu_to_uvw.__doc__


def angular_separation(
    rfi_xyz: da.Array, ants_xyz: da.Array, ra: da.Array, dec: da.Array
) -> da.Array:
    n_src, n_time = rfi_xyz.shape[:2]
    n_ant = ants_xyz.shape[1]

    src_chunk, time_chunk = rfi_xyz.chunksize[:2]
    ant_chunk = ants_xyz.chunksize[1]

    input = xr.Dataset(
        {
            "rfi_xyz": (["src", "time", "space"], rfi_xyz),
            "ants_xyz": (["time", "ant", "space"], ants_xyz),
            "ra": (["cel_space_0"], da.from_array([ra])),
            "dec": (["cel_space_1"], da.from_array([dec])),
        }
    )

    output = xr.Dataset(
        {
            "sep": (
                ["src", "time", "ant"],
                da.zeros(
                    shape=(n_src, n_time, n_ant),
                    chunks=(src_chunk, time_chunk, ant_chunk),
                    dtype=float,
                ),
            )
        }
    )

    def _angular_separation(ds):
        sep = delayed(coord.angular_separation, pure=True)(
            ds.rfi_xyz.data, ds.ants_xyz.data, ds.ra.data, ds.dec.data
        ).compute()
        ds_out = xr.Dataset({"sep": (["src", "time", "ant"], sep)})
        return ds_out

    ds = xr.map_blocks(_angular_separation, input, template=output)

    return ds.sep.data


angular_separation.__doc__ = coord.angular_separation.__doc__


def orbit_vmap(
    times: da.Array,
    elevation: da.Array,
    inclination: da.Array,
    lon_asc_node: da.Array,
    periapsis,
) -> da.Array:
    n_time = times.shape[0]
    n_src = elevation.shape[0]

    time_chunk = times.chunksize[0]
    src_chunk = elevation.chunksize[0]

    input = xr.Dataset(
        {
            "times": (["time"], times),
            "elevation": (["src"], elevation),
            "inclination": (["src"], inclination),
            "lon_asc_node": (["src"], lon_asc_node),
            "periapsis": (["src"], periapsis),
        }
    )

    output = xr.Dataset(
        {
            "orbit": (
                ["src", "time", "space"],
                da.zeros(
                    shape=(n_src, n_time, 3),
                    chunks=(src_chunk, time_chunk, 3),
                    dtype=float,
                ),
            )
        }
    )

    def _orbit_vmap(ds):
        orbit = delayed(coord.orbit_vmap, pure=True)(
            ds.times.data,
            ds.elevation.data,
            ds.inclination.data,
            ds.lon_asc_node.data,
            ds.periapsis.data,
        ).compute()
        ds_out = xr.Dataset({"orbit": (["src", "time", "space"], orbit)})
        return ds_out

    ds = xr.map_blocks(_orbit_vmap, input, template=output)

    return ds.orbit.data


orbit_vmap.__doc__ = coord.orbit_vmap.__doc__
