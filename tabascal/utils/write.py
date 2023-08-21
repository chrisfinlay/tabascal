# from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from tabascal.jax.observation import Observation as jaxObservation
    from tabascal.dask.observation import Observation as daskObservation

import os
import shutil

import dask
import dask.array as da
import numpy as np
import xarray as xr
from daskms import Dataset, xds_to_table


def construct_observation_ds(obs: jaxObservation | daskObservation) -> xr.Dataset:
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


def get_visibility_data(obs: jaxObservation | daskObservation) -> dict:
    vis_data = {
        "vis_obs": (["time", "bl", "freq"], obs.vis_obs),
        "vis_ast": (["time_fine", "bl", "freq"], obs.vis_ast),
        "vis_rfi": (["time_fine", "bl", "freq"], obs.vis_rfi),
    }
    if obs.backend == "jax":
        vis_data = {k: (v[0], da.asarray(v[1])) for k, v in vis_data.items()}
    return vis_data


def get_optional_data(obs: jaxObservation | daskObservation) -> dict:
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
        opt_data = {k: (v[0], da.asarray(v[1])) for k, v in opt_data.items()}
    return opt_data


def get_observation_attributes(obs: jaxObservation | daskObservation) -> dict:
    attrs = {
        "tel_name": obs.name,
        "tel_latitude": obs.latitude,
        "tel_longitude": obs.longitude,
        "tel_elevation": obs.elevation,
        "target_ra": obs.ra,
        "target_dec": obs.dec,
        "int_time": obs.int_time,
        "chan_width": obs.chan_width,
        "dish_diameter": obs.dish_d,
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


def get_coordinates(obs: jaxObservation | daskObservation) -> dict:
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
        coords = {k: da.asarray(v) for k, v in coords.items()}
    return coords


def get_astromonical_source_data(obs: jaxObservation | daskObservation) -> dict:
    if obs.n_ast > 0:
        ast_data = {
            # Astronomical source parameters
            "ast_I": (["ast_src", "time", "freq"], da.concatenate(obs.ast_I, axis=0)),
            "ast_lmn": (["ast_src", "lmn"], da.concatenate(obs.ast_lmn, axis=0)),
            "ast_radec": (
                ["ast_src", "radec"],
                da.concatenate(obs.ast_radec, axis=0).T,
            ),
        }
    else:
        ast_data = {}
    return ast_data


def get_satellite_rfi_data(obs: jaxObservation | daskObservation) -> dict:
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


def get_stationary_rfi_data(obs: jaxObservation | daskObservation) -> dict:
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


def write_ms(
    ds: Dataset,
    ms_path: str,
    overwrite: bool = False,
    vis_corr: Optional[da.Array] = None,
    flags: Optional[da.Array] = None,
) -> None:
    """Write a dataset to a Measurement Set."""
    if os.path.exists(ms_path):
        if overwrite:
            shutil.rmtree(ms_path)
        else:
            raise FileExistsError(f"File {ms_path} already exists.")
    tables = [
        construct_ms_data_table(ds, ms_path, vis_corr=vis_corr, flags=flags),
        construct_ms_antenna_table(ds, ms_path),
        construct_ms_direction_table(ds, ms_path),
        construct_ms_observation_table(ds, ms_path),
        construct_ms_field_table(ds, ms_path),
        construct_ms_data_desc_table(ds, ms_path),
        construct_ms_spectral_window_table(ds, ms_path),
        construct_ms_feed_table(ds, ms_path),
        construct_ms_polarization_table(ds, ms_path),
    ]
    for table in tables:
        dask.compute(table)


def construct_ms_data_table(ds: Dataset, ms_path: str, vis_corr=None, flags=None):
    """Get the data table for a Measurement Set."""
    n_time = ds.attrs["n_time"]
    n_int_samples = ds.attrs["n_int_samples"]
    n_freq = ds.attrs["n_freq"]
    n_corr = 1
    n_bl = ds.attrs["n_bl"]
    n_row = n_time * n_bl

    centroid_idx = int((n_int_samples) / 2) + n_int_samples * np.arange(n_time)

    vis_obs = ds.vis_obs.data.reshape(-1, n_freq, n_corr)

    if vis_corr is None:
        vis_corr = ds.vis_ast.data[centroid_idx].reshape(-1, n_freq, n_corr)
    else:
        vis_corr = da.asarray(vis_corr).reshape(-1, n_freq, n_corr)

    if flags is None:
        flags = da.zeros(shape=(n_row, n_freq, n_corr)).astype(bool)
    else:
        flags = da.asarray(flags).reshape(-1, n_freq, n_corr)

    row_times = da.asarray(
        (ds.coords["time"].data[:, None] * da.ones(shape=(1, n_bl))).flatten()
    )
    ant1 = (
        (ds.antenna1.data[None, :] * da.ones(shape=(n_time, 1)))
        .flatten()
        .astype(np.int32)
    )
    ant2 = (
        (ds.antenna2.data[None, :] * da.ones(shape=(n_time, 1)))
        .flatten()
        .astype(np.int32)
    )

    uvw = ds.bl_uvw.data[centroid_idx].reshape(-1, 3)

    noise = ds.noise_std.data.mean() * da.ones(shape=(n_row, 1))
    weight = da.ones(shape=(n_row, 1))
    a_id = da.zeros(n_row, dtype=np.int32)

    data_vars = {
        "DATA": (("row", "chan", "corr"), vis_obs),
        "ANTENNA1": (("row"), ant1),
        "ANTENNA2": (("row"), ant2),
        "TIME": (("row"), row_times),
        "TIME_CENTROID": (("row"), row_times),
        "UVW": (("row", "uvw"), uvw),
        "CORRECTED_DATA": (("row", "chan", "corr"), vis_corr),
        "SIGMA": (("row", "corr"), noise),
        "WEIGHT": (("row", "corr"), weight),
        "ARRAY_ID": (("row"), a_id),
        "FLAG": (("row", "chan", "corr"), flags),
    }
    dims = ("row", "chan", "corr")
    chunks = {k: v for k, v in zip(dims, vis_obs.chunksize)}

    return xds_to_table([Dataset(data_vars).chunk(chunks)], ms_path, columns="ALL")

    #####################


def construct_ms_antenna_table(ds: Dataset, ms_path: str):
    n_ant = ds.attrs["n_ant"]

    ants_xyz = ds.ants_xyz.data[0]
    dish_d = ds.attrs["dish_diameter"] * da.ones(n_ant)
    mount = da.asarray(["ALT-AZ" for _ in range(n_ant)])
    dish_type = da.asarray(["GROUND-BASED" for _ in range(n_ant)])
    dish_name = da.asarray([f"m{i:0>3}" for i in range(n_ant)])
    offset = da.zeros((n_ant, 3))
    flag = da.zeros(n_ant)

    data_vars = {
        "POSITION": (("row", "xyz"), ants_xyz),
        "DISH_DIAMETER": (("row"), dish_d),
        "MOUNT": (("row"), mount),
        "TYPE": (("row"), dish_type),
        "NAME": (("row"), dish_name),
        "STATION": (("row"), dish_name),
        "OFFSET": (("row", "xyz"), offset),
        "FLAG_ROW": (("row"), flag),
    }

    return xds_to_table([Dataset(data_vars)], ms_path + "::ANTENNA", columns="ALL")

    #####################


def construct_ms_direction_table(ds: Dataset, ms_path: str):
    target = da.deg2rad(
        da.asarray([ds.attrs["target_ra"], ds.attrs["target_dec"]])
    ).reshape(1, 2)

    data_vars = {"DIRECTION": (("row", "radec"), target)}

    return xds_to_table([Dataset(data_vars)], ms_path + "::SOURCE", columns="ALL")

    #####################


def construct_ms_observation_table(ds: Dataset, ms_path: str):
    flag = da.zeros(1)
    # log = da.zeros(1,1)
    observer = da.array(["Chris"])
    project = da.array(["tabascal"])
    release = da.zeros(1)
    tel_name = da.array([ds.attrs["tel_name"]])
    time_range = da.asarray(ds.coords["time"].data[da.array([0, -1])].reshape(1, 2))

    data_vars = {
        "FLAG_ROW": (("row"), flag),
        # 'LOG':(('row', 'log'), log),
        "OBSERVER": (("row"), observer),
        "PROJECT": (("row"), project),
        "RELEASE_DATE": (("row"), release),
        # 'SCHEDULE': (('row', 'schedule'), schedule),
        # 'SCHEDULE_TYPE': (('row', 'xyz'), schedule_type),
        "TELESCOPE_NAME": (("row"), tel_name),
        "TIME_RANGE": (("row", "obs-exts"), time_range),
    }

    return xds_to_table([Dataset(data_vars)], ms_path + "::OBSERVATION", columns="ALL")

    #####################


def construct_ms_field_table(ds: Dataset, ms_path: str):
    code = da.array(["T"])
    target = da.deg2rad(
        da.asarray([ds.attrs["target_ra"], ds.attrs["target_dec"]])
    ).reshape(1, 1, 2)
    flag = da.zeros(1)
    num_poly = da.zeros(1)
    source_id = da.zeros(1)
    time = da.asarray(ds.coords["time"].data[:1])
    name = da.array(["tabascal_sim"])

    data_vars = {
        "CODE": (("row"), code),
        "DELAY_DIR": (("row", "field-poly", "field-dir"), target),
        "FLAG_ROW": (("row"), flag),
        "NUM_POLY": (("row"), num_poly),
        "PHASE_DIR": (("row", "field-poly", "field-dir"), target),
        "REFERENCE_DIR": (("row", "field-poly", "field-dir"), target),
        "SOURCE_ID": (("row"), source_id),
        "TIME": (("row"), time),
        "NAME": (("row"), name),
    }

    return xds_to_table([Dataset(data_vars)], ms_path + "::FIELD", columns="ALL")

    #####################


def construct_ms_data_desc_table(ds: Dataset, ms_path: str):
    flag = da.zeros(1, dtype=bool)
    pol_id = da.zeros(1)
    spec_id = da.zeros(1)

    data_vars = {
        "FLAG_ROW": (("row"), flag),
        "POLARIZATION_ID": (("row"), pol_id),
        "SPECTRAL_WINDOW_ID": (("row"), spec_id),
    }

    return xds_to_table(
        [Dataset(data_vars)], ms_path + "::DATA_DESCRIPTION", columns="ALL"
    )


def construct_ms_spectral_window_table(ds: Dataset, ms_path: str):
    n_freq = ds.attrs["n_freq"]
    chan_freq = da.asarray(ds.coords["freq"].data.reshape(1, n_freq))
    chan_width = ds.attrs["chan_width"] * da.ones((1, n_freq))
    ref_freq = chan_freq[:, 0]
    total_bw = chan_width.sum(axis=1)
    zero = da.zeros(1)
    one = da.ones(1)

    data_vars = {
        "CHAN_FREQ": (("row", "chan"), chan_freq),
        "CHAN_WIDTH": (("row", "chan"), chan_width),
        "EFFECTIVE_BW": (("row", "chan"), chan_width),
        "FLAG_ROW": (("row"), zero),
        "FREQ_GROUP": (("row"), zero),
        # 'FREQ_GROUP_NAME': (('row'), zero),
        "IF_CONV_CHAIN": (("row"), zero),
        "MEAS_FREQ_REF": (("row"), 5 * one),
        # 'NAME': (('row'), zero),
        "NET_SIDEBAND": (("row"), one),
        "NUM_CHAN": (("row"), one),
        "REF_FREQUENCY": (("row"), ref_freq),
        "RESOLUTION": (("row", "chan"), chan_width),
        "TOTAL_BANDWIDTH": (("row"), total_bw),
    }

    return xds_to_table(
        [Dataset(data_vars)], ms_path + "::SPECTRAL_WINDOW", columns="ALL"
    )


def construct_ms_feed_table(ds: Dataset, ms_path: str):
    n_ant = ds.attrs["n_ant"]
    ant_id = da.arange(n_ant)
    beam_id = da.ones(n_ant)
    beam_offset = da.zeros(shape=(n_ant, 2, 2))
    feed_id = da.zeros(n_ant)
    interval = da.zeros(n_ant)
    n_recep = 2 * da.ones(n_ant)
    pol_type = da.array([["X", "Y"] for _ in range(n_ant)])
    pol_resp = da.array([da.eye(2, dtype=np.complex64) for _ in range(n_ant)]).rechunk(
        n_ant, 2, 2
    )
    pos = da.zeros(shape=(n_ant, 3))
    rec_ang = -0.5 * np.pi * da.ones(shape=(n_ant, 2))
    spec_id = -1 * da.ones(n_ant)
    time = da.zeros(n_ant)

    data_vars = {
        "ANTENNA_ID": (("row"), ant_id),
        "BEAM_ID": (("row"), beam_id),
        "BEAM_OFFSET": (("row", "receptors", "radec"), beam_offset),
        "FEED_ID": (("row"), feed_id),
        "INTERVAL": (("row"), interval),
        "NUM_RECEPTORS": (("row"), n_recep),
        "POLARIZATION_TYPE": (("row", "receptors"), pol_type),
        "POL_RESPONSE": (("row", "receptors", "receptors-2"), pol_resp),
        "POSITION": (("row", "xyz"), pos),
        "RECEPTOR_ANGLE": (("row", "receptors"), rec_ang),
        "SPECTRAL_WINDOW_ID": (("row"), spec_id),
        "TIME": (("row"), time),
    }

    return xds_to_table([Dataset(data_vars)], ms_path + "::FEED", columns="ALL")


def construct_ms_polarization_table(ds: Dataset, ms_path: str):
    corr_prod = da.zeros(shape=(1, 1, 2))
    corr_type = 9 * da.ones(shape=(1, 1))
    flag_row = da.zeros(1)
    num_corr = da.ones(1)

    data_vars = {
        "CORR_PRODUCT": (("row", "corr", "corrprod_idx"), corr_prod),
        "CORR_TYPE": (("row", "corr"), corr_type),
        "FLAG_ROW": (("row"), flag_row),
        "NUM_CORR": (("row"), num_corr),
    }

    return xds_to_table([Dataset(data_vars)], ms_path + "::POLARIZATION", columns="ALL")
