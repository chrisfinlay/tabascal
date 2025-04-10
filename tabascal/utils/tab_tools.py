import numbers
import subprocess
from datetime import datetime
import os

import xarray as xr
from daskms import xds_from_ms, xds_from_table
import dask.array as da
import numpy as np
from astropy.time import Time
import numpy as np
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
from jax import jit, vmap, random
from jax.tree_util import tree_map

from numpyro.infer import MCMC, NUTS, Predictive

from tabascal.jax.coordinates import (
    calculate_sat_corr_time,
    # orbit,
    itrf_to_uvw,
    itrf_to_xyz,
    calculate_fringe_frequency,
    gmsa_from_jd,
    mjd_to_jd,
    secs_to_days,
    angular_separation,
)
from tabascal.utils.tle import get_satellite_positions, get_tles_by_id

# from tabascal.jax.interferometry import int_sample_times
from tabascal.dask.interferometry import int_sample_times
from tabascal.utils.config import yaml_load
from tabascal.utils.plot import plot_predictions


from tab_opt.transform import affine_transform_full_inv, affine_transform_diag_inv
from tab_opt.gp import get_times, resampling_kernel
from tab_opt.vis import get_rfi_phase
from tab_opt.opt import run_svi, svi_predict, f_model_flat, flatten_obs, post_samples

# from tab_opt.data import extract_data


def split_args(args):

    static_args = {}
    array_args = {}
    for key, value in args.items():
        if hasattr(value, "shape"):
            array_args.update({key: value})
        else:
            static_args.update({key: value})

    static_args = frozendict(static_args)

    return static_args, array_args


def get_ast_fringe_rate(uv, freq=1.227e9, D=13.5):

    omega = 2 * np.pi / (24 * 3600)
    lam = 3e8 / freq

    U = jnp.max(jnp.linalg.norm(uv, axis=-1), axis=0)

    bw = 1.22 * lam / D
    max_l = jnp.sin(bw / 2)

    max_fr = omega * U * max_l / lam

    return max_fr


def print_rfi_signal_error(zarr_path, ms_params, true_params, gp_params):

    xds = xr.open_zarr(zarr_path)
    rfi_A_true = jnp.transpose(
        xds.rfi_tle_sat_A.data[:, :, :, 0].compute(), axes=(0, 2, 1)
    )
    rfi_resample = resampling_kernel(
        gp_params["rfi_times"],
        int_sample_times(ms_params["times"], xds.n_int_samples).compute(),
        gp_params["rfi_var"],
        gp_params["rfi_l"],
        1e-8,
    )
    rfi_amp = true_params["rfi_r_induce"] + 1.0j * true_params["rfi_i_induce"]
    rfi_A_pred = vmap(lambda x, y: x @ y.T, in_axes=(0, None))(rfi_amp, rfi_resample)
    print(
        f"RMSE RFI signal : {jnp.sqrt(jnp.mean(jnp.abs(rfi_A_true - rfi_A_pred)**2)):.5f}"
    )


# @jit
def reduced_chi2(pred, true, noise):

    complex_types = [
        complex,
        np.complex64,
        np.complex128,
        jnp.complex64,
        jnp.complex128,
    ]
    is_complex = jnp.any(jnp.array([true.dtype == c_type for c_type in complex_types]))
    if is_complex:
        # print("Complex Data")
        norm = 2 * true.size
    else:
        norm = true.size

    rchi2 = jnp.sum((jnp.abs(pred - true) / noise) ** 2) / norm

    return rchi2


def write_results_xds(vi_pred, args, file_path, overwrite=True):

    # print(vi_pred.keys())
    # print(vi_pred["rfi_vis"].shape)
    # print(vi_pred["rfi_vis"])

    # print(da.asarray(vi_pred["ast_vis"]))
    # print(da.asarray(vi_pred["gains"]))
    # print(da.asarray(vi_pred["rfi_vis"]))
    # print(da.asarray(vi_pred["vis_obs"]))
    # print(da.asarray(vi_pred["rfi_A"]))
    # print(da.asarray(args["rfi_phase"]))

    map_xds = xr.Dataset(
        data_vars={
            "ast_vis": (["sample", "bl", "time"], da.asarray(vi_pred["ast_vis"])),
            "gains": (["sample", "ant", "time"], da.asarray(vi_pred["gains"])),
            "rfi_vis": (["sample", "bl", "time"], da.asarray(vi_pred["rfi_vis"])),
            "vis_obs": (["sample", "bl", "time"], da.asarray(vi_pred["vis_obs"])),
            "rfi_A": (
                ["sample", "src", "ant", "rfi_time"],
                da.asarray(vi_pred["rfi_A"]),
            ),
            "rfi_phase": (
                ["src", "ant", "time_mjd_fine"],
                da.asarray(args["rfi_phase"]),
            ),
        },
        coords={
            "time": da.asarray(args["times"]),
            "rfi_time": da.asarray(args["rfi_times"]),
            "time_mjd_fine": da.asarray(args["times_mjd_fine"]),
        },
    )
    # print(map_xds)

    mode = "w" if overwrite else "w-"

    map_xds.to_zarr(file_path, mode=mode)

    return map_xds


def write_params_xds(vi_params, gp_params, ms_params, file_path, overwrite=True):

    n_ant, n_g_times = vi_params["g_amp_induce_base_auto_loc"].shape
    # print({key: value.shape for key, value in vi_params.items()})
    rfi_amp_base = da.asarray(
        vi_params["rfi_r_induce_base_auto_loc"]
        + 1.0j * vi_params["rfi_i_induce_base_auto_loc"]
    )
    gains_base = da.asarray(
        vi_params["g_amp_induce_base_auto_loc"]
        * np.exp(
            1.0j
            * np.concatenate(
                [
                    vi_params["g_phase_induce_base_auto_loc"],
                    np.zeros((1, n_g_times)),
                ],
                axis=0,
            )
        )
    )
    ast_k_base = da.asarray(
        vi_params["ast_k_r_base_auto_loc"] + 1.0j * vi_params["ast_k_i_base_auto_loc"],
    )
    map_xds = xr.Dataset(
        data_vars={
            "rfi_amp_base": (
                ["src", "ant", "rfi_time"],
                rfi_amp_base,
            ),
            "gains_base": (
                ["ant", "g_time"],
                gains_base,
            ),
            "ast_k_base": (
                ["bl", "ast_time"],
                ast_k_base,
            ),
        },
        coords={
            "rfi_time": da.asarray(gp_params["rfi_times"]),
            "g_time": da.asarray(gp_params["g_times"]),
            # "ast_time": da.asarray(ast_time),
        },
    )

    mode = "w" if overwrite else "w-"

    map_xds.to_zarr(file_path, mode=mode)

    return map_xds


def inv_transform(params, loc, inv_scaling):
    params_trans = {
        "rfi_r_induce_base": vmap(
            vmap(affine_transform_full_inv, (0, None, 0), 0), (1, None, 1), 1
        )(params["rfi_r_induce"], inv_scaling["L_RFI"], loc["mu_rfi_r"]),
        "rfi_i_induce_base": vmap(
            vmap(affine_transform_full_inv, (0, None, 0), 0), (1, None, 1), 1
        )(params["rfi_i_induce"], inv_scaling["L_RFI"], loc["mu_rfi_i"]),
        "g_amp_induce_base": vmap(affine_transform_full_inv, in_axes=(0, None, 0))(
            params["g_amp_induce"], inv_scaling["L_G_amp"], loc["mu_G_amp"]
        ),
        "g_phase_induce_base": vmap(affine_transform_full_inv, in_axes=(0, None, 0))(
            params["g_phase_induce"], inv_scaling["L_G_phase"], loc["mu_G_phase"]
        ),
        "ast_k_r_base": vmap(affine_transform_diag_inv, in_axes=(0, 0, 0))(
            params["ast_k_r"], inv_scaling["sigma_ast_k"], loc["mu_ast_k_r"]
        ),
        "ast_k_i_base": vmap(affine_transform_diag_inv, in_axes=(0, 0, 0))(
            params["ast_k_i"], inv_scaling["sigma_ast_k"], loc["mu_ast_k_i"]
        ),
    }
    return params_trans


def read_ms(ms_path, freq: float = None, corr: str = "xx", data_col: str = "DATA"):

    correlations = {"xx": 0, "xy": 1, "yx": 2, "yy": 3}
    corr = correlations[corr]

    xds = xds_from_ms(ms_path)[0]
    xds_ant = xds_from_table(ms_path + "::ANTENNA")[0]
    xds_spec = xds_from_table(ms_path + "::SPECTRAL_WINDOW")[0]
    xds_src = xds_from_table(ms_path + "::SOURCE")[0]

    ants_itrf = jnp.array(xds_ant.POSITION.data.compute())

    n_ant = ants_itrf.shape[0]
    n_time = len(jnp.unique(xds.TIME.data.compute()))
    n_bl = xds.DATA.data.shape[0] // n_time
    n_freq, n_corr = xds.DATA.data.shape[1:]

    freqs = jnp.array(xds_spec.CHAN_FREQ.data.compute())
    int_time = xds.INTERVAL.data[0].compute()

    times_mjd = jnp.array(xds.TIME.data.reshape(n_time, n_bl)[:, 0].compute())

    times = jnp.linspace(0, n_time * int_time, n_time, endpoint=False)

    if freq:
        chan = jnp.argmin(jnp.abs(freq - freqs))
    else:
        chan = 0

    data = {
        **{
            key: val
            for key, val in zip(
                ["ra", "dec"], jnp.rad2deg(xds_src.DIRECTION.data[0].compute())
            )
        },
        "n_freq": n_freq,
        "n_corr": n_corr,
        "n_time": n_time,
        "n_ant": n_ant,
        "n_bl": n_bl,
        "dish_d": xds_ant.DISH_DIAMETER.data[0].compute(),
        "times_mjd": times_mjd,
        "times": times,
        "int_time": int_time,
        "freqs": freqs[chan],
        "ants_itrf": ants_itrf,
        "uvw": jnp.array(xds.UVW.data.reshape(n_time, n_bl, 3).compute()),
        "vis_obs": jnp.array(
            xds[data_col]
            .data.reshape(n_time, n_bl, n_freq, n_corr)
            .compute()[:, :, chan, corr]
        ),
        "noise": jnp.array(xds.SIGMA.data.mean().compute()),
        "a1": jnp.array(xds.ANTENNA1.data.reshape(n_time, n_bl)[0, :].compute()),
        "a2": jnp.array(xds.ANTENNA2.data.reshape(n_time, n_bl)[0, :].compute()),
    }

    return data


def get_tles(config, ms_params, norad_ids, spacetrack_path, tle_offset=-1):
    if config["satellites"]["norad_ids_path"]:
        norad_ids += [
            int(x) for x in yaml_load(config["satellites"]["norad_ids_path"]).split()
        ]

    if len(config["satellites"]["norad_ids"]) > 0:
        norad_ids += config["satellites"]["norad_ids"]

    norad_ids = list(np.unique(norad_ids))

    n_rfi = len(norad_ids) + len(config["satellites"]["sat_ids"])

    # if len(config["satellites"]["sat_ids"])>0:
    #     import pandas as pd
    #     ole_df = pd.read_csv(config["satellites"]["ole_path"])
    #     oles = ole_df[ole_df["sat_id"].isin(config["satellites"]["sat_ids"])][["elevation", "inclination", "lon_asc_node", "periapsis"]]
    #     rfi_orbit = jnp.atleast_2d(oles.values)

    if len(norad_ids) > 0 and spacetrack_path:
        st_config = yaml_load(spacetrack_path)
        tles_df = get_tles_by_id(
            st_config["username"],
            st_config["password"],
            norad_ids,
            mjd_to_jd(jnp.mean(ms_params["times_mjd"])) + tle_offset,
            window_days=1 + np.abs(tle_offset),
            tle_dir=config["satellites"]["tle_dir"],
        )
        tles = np.atleast_2d(tles_df[["TLE_LINE1", "TLE_LINE2"]].values)
    elif len(norad_ids) > 0:
        raise ValueError("No spacetrack_path has been defined.")

    # return n_rfi, norad_ids, tles, rfi_orbit
    return n_rfi, norad_ids, tles


def estimate_max_rfi_vis(ms_params: dict):

    # Assumes RFI is dominant in the visibilities

    return jnp.max(jnp.abs(ms_params["vis_obs"]))


def estimate_vis_ast(ms_params: dict):

    # Assumes RFI will have fringing-winding loss when averaging
    # visibilities over time to leave astronomical signal only

    vis_ast_est = jnp.mean(ms_params["vis_obs"].T, axis=1, keepdims=True) * jnp.ones(
        (ms_params["n_bl"], ms_params["n_time"])
    )

    return vis_ast_est


def estimate_vis_rfi(ms_params: dict):

    # Assumes accurate astronomical visibility estimate and
    # returns only the magnitude estimate on the 'shortest' baselines

    vis_ast_est = estimate_vis_ast(ms_params)
    vis_rfi_est = jnp.max(jnp.abs(ms_params["vis_obs"] - vis_ast_est.T), axis=1)

    return vis_rfi_est


# def estimate_sampling(config: dict, ms_params: dict, n_rfi: int, norad_ids, tles: list[list[str]], rfi_orbit):
def estimate_sampling(
    config: dict, ms_params: dict, n_rfi: int, norad_ids, tles: list[list[str]]
):
    if config["rfi"]["n_int_samples"]:
        return config["rfi"]["n_int_samples"]

    jd_minute = 1 / (24 * 60)
    times_mjd_coarse = jnp.arange(
        ms_params["times_mjd"][0], ms_params["times_mjd"][-1] + jd_minute, jd_minute
    )
    # times_coarse = times_mjd_coarse * 24 * 3600
    # times_coarse = Time(times_mjd_coarse, format="mjd").sidereal_time("mean", "greenwich").hour*3600 # Convert hours to seconds

    if len(norad_ids) > 0:
        rfi_xyz = get_satellite_positions(tles, np.array(mjd_to_jd(times_mjd_coarse)))
    # if len(config["satellites"]["sat_ids"])>0:
    #     rfi_xyz = jnp.array([orbit(times_coarse, *rfi_orb) for rfi_orb in rfi_orbit])

    gsa = (
        Time(times_mjd_coarse, format="mjd").sidereal_time("mean", "greenwich").hour
        * 15
    )  # Convert hours to degrees
    # gsa = gmsa_from_jd(mjd_to_jd(times_mjd_coarse))
    gh0 = (gsa - ms_params["ra"]) % 360
    ants_u = itrf_to_uvw(ms_params["ants_itrf"], gh0, ms_params["dec"])[
        :, :, 0
    ]  # We want the uvw-coordinates at the coarse sampling rate for the fringe frequency prediction

    max_rfi_vis_est = estimate_max_rfi_vis(ms_params)

    fringe_params = [
        {
            "times_mjd": times_mjd_coarse,
            "freq": jnp.max(ms_params["freqs"]),
            "rfi_xyz": rfi_xyz[i],
            "ants_itrf": ms_params["ants_itrf"],
            "ants_u": ants_u,
            "dec": ms_params["dec"],
        }
        for i in range(n_rfi)
    ]

    fringe_freq = jnp.array(
        [calculate_fringe_frequency(**f_params) for f_params in fringe_params]
    )
    print()
    print(f"Max Fringe Freq: {jnp.max(jnp.abs(fringe_freq)):.2f} Hz")
    print(f"Estimated Max RFI A : {jnp.sqrt(max_rfi_vis_est):.5f} sqrt(Jy)")

    if get_truth_conditional(config):
        xds = xr.open_zarr(config["data"]["zarr_path"])
        print(
            f"True Max RFI A      : {np.max(xds.rfi_tle_sat_A.data).compute():.5f} sqrt(Jy)"
        )

    sample_freq = (
        jnp.pi
        * jnp.max(jnp.abs(fringe_freq))
        * jnp.sqrt(max_rfi_vis_est / (6 * ms_params["noise"]))
    )
    n_int_samples = int(
        jnp.ceil(config["rfi"]["n_int_factor"] * ms_params["int_time"] * sample_freq)
    )

    return n_int_samples


def get_orbit_elevation(rfi_xyz, latitude):

    from tabascal.jax.coordinates import earth_radius

    R_rfi = jnp.max(jnp.linalg.norm(rfi_xyz, axis=-1))
    R_e = earth_radius(latitude)

    return R_rfi - R_e


def get_truth_conditional(config):

    truth_cond = jnp.array(
        [
            bool(config["plots"]["truth"]),
            config["gains"]["amp_mean"] == "truth",
            config["gains"]["phase_mean"] == "truth",
            "sigma" in str(config["gains"]["amp_mean"]),
            "sigma" in str(config["gains"]["phase_mean"]),
            config["ast"]["mean"] == "truth",
            config["rfi"]["mean"] == "truth",
            config["ast"]["mean"] == "truth_mean",
            config["ast"]["init"] == "truth",
            config["rfi"]["init"] == "truth",
            config["ast"]["init"] == "truth_mean",
        ]
    )

    return jnp.any(truth_cond)


def get_observation_data_type(data_col):

    ast = ["DATA", "CAL_DATA", "AST_DATA", "AST_MODEL_DATA"]
    rfi = ["DATA", "CAL_DATA", "RFI_DATA", "RFI_MODEL_DATA"]
    gains = ["DATA"]

    data_type = {
        "ast": data_col in ast,
        "rfi": data_col in rfi,
        "gains": data_col in gains,
    }

    return data_type


def calculate_true_values(
    zarr_path: str,
    config: dict,
    ms_params: dict,
    gp_params: dict,
    n_rfi: int,
    norad_ids: list,
):
    xds = xr.open_zarr(zarr_path)

    data_type = get_observation_data_type(config["data"]["data_col"])

    vis_ast = (
        xds.vis_ast.data[:, :, 0].compute()
        if data_type["ast"]
        else jnp.zeros_like((xds.vis_ast.data[:, :, 0]))
    )
    vis_rfi = (
        xds.vis_rfi.data[:, :, 0].compute()
        if data_type["rfi"]
        else jnp.zeros_like((xds.vis_rfi.data[:, :, 0]))
    )
    gains_ants = (
        xds.gains_ants.data[:, :, 0].compute()
        if data_type["gains"]
        else jnp.ones_like((xds.gains_ants.data[:, :, 0]))
    )
    a1 = xds.antenna1.data.compute()
    a2 = xds.antenna2.data.compute()

    vis = gains_ants[:, a1] * jnp.conjugate(gains_ants[:, a2]) * (vis_ast + vis_rfi)

    rchi2 = reduced_chi2(vis, ms_params["vis_obs"], ms_params["noise"])
    print()
    print(f"Reduced Chi^2 @ truth : {rchi2}")

    # if len(config["satellites"]["sat_ids"]) > 0:
    #     corr_time_params = [
    #         {
    #             "sat_xyz": xds.rfi_sat_xyz.data[i, xds.time_idx].compute(),
    #             "ants_xyz": xds.ants_xyz.data[xds.time_idx].compute(),
    #             "orbit_el": xds.rfi_sat_orbit.data[i, 0].compute(),
    #             "lat": xds.tel_latitude,
    #             "dish_d": xds.dish_diameter,
    #             "freqs": ms_params["freqs"],
    #         }
    #         for i in range(n_rfi)
    #     ]

    #     l = jnp.min(
    #         jnp.array(
    #             [calculate_sat_corr_time(**corr_time_params[i]) for i in range(n_rfi)]
    #         )
    #     )

    #     rfi_induce = jnp.array(
    #         [
    #             vmap(jnp.interp, in_axes=(None, None, 1), out_axes=(0))(
    #                 gp_params["rfi_times"],
    #                 xds.time_fine.data,
    #                 xds.rfi_sat_A[:, :, :, 0].data.compute()[i],
    #             )
    #             for i in range(n_rfi)
    #         ]
    #     )

    if len(norad_ids) > 0:
        corr_time_params = [
            {
                "sat_xyz": xds.rfi_tle_sat_xyz.data[i, xds.time_idx].compute(),
                "ants_xyz": xds.ants_xyz.data[xds.time_idx].compute(),
                "orbit_el": get_orbit_elevation(
                    xds.rfi_tle_sat_xyz.data[i, xds.time_idx].compute(),
                    xds.tel_latitude,
                ),
                "lat": xds.tel_latitude,
                "dish_d": xds.dish_diameter,
                "freqs": ms_params["freqs"],
            }
            for i in range(n_rfi)
        ]

        l = jnp.min(
            jnp.array(
                [calculate_sat_corr_time(**corr_time_params[i]) for i in range(n_rfi)]
            )
        )

        rfi_induce = jnp.array(
            [
                vmap(jnp.interp, in_axes=(None, None, 1), out_axes=(0))(
                    gp_params["rfi_times"],
                    xds.time_fine.data,
                    (
                        xds.rfi_tle_sat_A[:, :, :, 0].data.compute()[i]
                        if data_type["rfi"]
                        else jnp.zeros_like(xds.rfi_tle_sat_A[0, :, :, 0].data)
                    ),
                )
                for i in range(n_rfi)
            ]
        )

    print()
    print(
        f"Minimum expected RFI correlation time : {l:.0f} s ({gp_params['rfi_l']:.0f} s used)"
    )
    print(f"RFI Var used : {gp_params['rfi_var']:.1e} Jy")
    print()
    print(f"Mean RFI Amp. : {jnp.mean(jnp.abs(vis_rfi)):.1f} Jy")
    print(f"Mean AST Amp. : {jnp.mean(jnp.abs(vis_ast)):.1f} Jy")

    if config["model"]["func"] == "fixed_orbit_rfi_full_fft_standard_padded_model":
        vis_ast_padded = vmap(jnp.pad, in_axes=(1, None, None))(
            vis_ast, gp_params["ast_pad"], "linear_ramp"
        )
        ast_k = jnp.fft.fft(vis_ast_padded, axis=1)
    else:
        ast_k = jnp.fft.fft(vis_ast, axis=0).T

    gains_induce = vmap(jnp.interp, in_axes=(None, None, 1), out_axes=(0))(
        gp_params["g_times"], ms_params["times"], gains_ants
    )

    true_params = {
        **{f"g_amp_induce": jnp.abs(gains_induce)},
        **{f"g_phase_induce": jnp.angle(gains_induce[:-1])},
        **{f"rfi_r_induce": rfi_induce.real},
        **{f"rfi_i_induce": rfi_induce.imag},
        **{"ast_k_r": ast_k.real},
        **{"ast_k_i": ast_k.imag},
    }

    truth_args = {
        "vis_ast_true": jnp.array(vis_ast.T),
        "vis_rfi_true": jnp.array(vis_rfi.T),
        "gains_true": jnp.array(gains_ants.T),
    }

    return true_params, truth_args


from tabascal.jax.coordinates import angular_separation
from tabascal.jax.interferometry import airy_beam


def get_rfi_amp_estimate(ms_params, tles):

    rfi_xyz = get_satellite_positions(tles, np.array(mjd_to_jd(ms_params["times_mjd"])))
    ants_xyz = get_antenna_positions(ms_params, 1)
    ants_u = get_antenna_uvw(ms_params, 1)[:, :, 0]

    fringe_freqs = jnp.array(
        [
            calculate_fringe_frequency(
                ms_params["times_mjd"],
                ms_params["freqs"],
                rfi_xyz_,
                ms_params["ants_itrf"],
                ants_u,
                ms_params["dec"],
            )
            for rfi_xyz_ in rfi_xyz
        ]
    )

    # rfi_xyz is shape (n_rfi, n_time, 3)
    # ants_xyz is shape (n_time, n_ant, 3)
    theta = angular_separation(
        rfi_xyz,
        jnp.mean(ants_xyz, axis=1, keepdims=True),
        ms_params["ra"],
        ms_params["dec"],
    )
    # theta is shape (n_rfi, n_time, n_ant)
    B = airy_beam(theta, ms_params["freqs"], ms_params["dish_d"])[:, :, 0, 0]
    # B is shape (n_rfi, n_time, n_ant, n_freq) -> (n_rfi, n_time)
    # fringe_freqs is shape (n_time, n_bl)
    bl = jnp.argmin(jnp.max(jnp.abs(fringe_freqs), axis=0))
    # vis_obs is shape (n_time, n_bl)
    rfi_amp = jnp.sqrt(
        jnp.max(jnp.abs(ms_params["vis_obs"][:, bl])) / jnp.max(jnp.sum(B**2, axis=0))
    )

    return B * rfi_amp


def calculate_estimates(ms_params, config, tles, gp_params):

    vis_ast_est = estimate_vis_ast(ms_params)
    ast_k_est = jnp.fft.fft(vis_ast_est, axis=1)

    if config["model"]["func"] == "fixed_orbit_rfi_full_fft_standard_padded_model":
        vis_ast_est_pad = vmap(jnp.pad, in_axes=(0, None, None))(
            vis_ast_est, gp_params["ast_pad"], "linear_ramp"
        )
        ast_k_est = jnp.fft.fft(vis_ast_est_pad, axis=1)

    rfi_times = gp_params["rfi_times"]
    n_rfi_times = len(rfi_times)
    # n_rfi = len(rfi_amp_ratios)

    rfi_amp = get_rfi_amp_estimate(ms_params, tles)
    n_rfi = len(rfi_amp)

    rfi_induce_est = vmap(
        jnp.interp,
        in_axes=(
            None,
            None,
            0,
        ),
    )(
        rfi_times, ms_params["times"], rfi_amp
    )[:, None, :] * jnp.ones((n_rfi, ms_params["n_ant"], n_rfi_times))

    # vis_rfi_est = estimate_vis_rfi(ms_params)

    # rfi_induce_est = (
    #     jnp.interp(
    #         rfi_times,
    #         ms_params["times"],
    #         jnp.sqrt(vis_rfi_est),
    #     )[
    #         None, None, :
    #     ]  # shape is now (1, 1, n_rfi_times)
    #     * jnp.ones((n_rfi, ms_params["n_ant"], n_rfi_times))
    #     / n_rfi
    # )

    # rfi_induce_est = (
    #     jnp.interp(
    #         rfi_times,
    #         ms_params["times"],
    #         jnp.sqrt(vis_rfi_est),
    #     )[
    #         None, None, :
    #     ]  # shape is now (1, 1, n_rfi_times)
    #     * jnp.ones((n_rfi, ms_params["n_ant"], n_rfi_times))
    #     * rfi_amp_ratios[:, None, ]
    # )

    estimates = {
        "ast_k_est": ast_k_est,
        "rfi_induce_est": rfi_induce_est,
    }

    return estimates


def get_prior_means(config, ms_params, estimates, true_params, n_rfi, gp_params):

    # Set Gain Prior Mean
    if config["gains"]["amp_mean"] == "truth":
        g_amp_prior_mean = true_params["g_amp_induce"]
    elif "sigma" in str(config["gains"]["amp_mean"]):
        n_sig = float(config["gains"]["amp_mean"].replace("sigma", ""))
        seed = int(1e3 * jnp.mean(true_params["g_phase_induce"][0, 0]))
        g_amp_prior_mean = true_params["g_amp_induce"] + n_sig * jnp.sqrt(
            gp_params["g_amp_var"]
        ) * random.normal(random.PRNGKey(1 * seed), (ms_params["n_ant"], 1))
    #     g_amp_prior_mean = true_params["g_amp_induce"] + vmap(
    #         jnp.dot, in_axes=(None, 0)
    #     )(
    #         gp_params["L_G_amp"],
    #         random.normal(
    #             random.PRNGKey(1), (ms_params["n_ant"], gp_params["n_g_times"])
    #         ),
    #     )
    elif isinstance(config["gains"]["amp_mean"], numbers.Number):
        g_amp_prior_mean = config["gains"]["amp_mean"] * jnp.ones(
            (ms_params["n_ant"], gp_params["n_g_times"])
        )
    else:
        ValueError("gains: amp_mean: must be a number, 'truth', or 'Xsigma'.")

    if config["gains"]["phase_mean"] == "truth":
        g_phase_prior_mean = true_params["g_phase_induce"]
    elif "sigma" in str(config["gains"]["phase_mean"]):
        n_sig = float(config["gains"]["phase_mean"].replace("sigma", ""))
        seed = int(1e3 * jnp.mean(true_params["g_phase_induce"][0, 0]))
        g_phase_prior_mean = (
            true_params["g_phase_induce"]
            + n_sig
            * jnp.sqrt(gp_params["g_phase_var"])
            * random.normal(random.PRNGKey(1 * seed), (ms_params["n_ant"], 1))[:-1]
        )
        # g_amp_prior_mean = true_params["g_phase_induce"] + vmap(
        #     jnp.dot, in_axes=(None, 0)
        # )(
        #     gp_params["L_G_phase"],
        #     random.normal(
        #         random.PRNGKey(1), (ms_params["n_ant"] - 1, gp_params["n_g_times"])
        #     ),
        # )
    elif isinstance(config["gains"]["phase_mean"], numbers.Number):
        g_phase_prior_mean = jnp.deg2rad(config["gains"]["phase_mean"]) * jnp.ones(
            (ms_params["n_ant"] - 1, gp_params["n_g_times"])
        )
    else:
        ValueError("gains: phase_mean: must be a number, 'truth', or 'Xsigma'.")

    # Set Astronomical Prior Mean
    if config["ast"]["mean"] == 0:
        if config["model"]["func"] == "fixed_orbit_rfi_full_fft_standard_padded_model":
            ast_k_prior_mean = jnp.zeros(
                (ms_params["n_bl"], ms_params["n_time"] + 2 * gp_params["ast_pad"]),
                dtype=complex,
            )
        else:
            ast_k_prior_mean = jnp.zeros(
                (ms_params["n_bl"], ms_params["n_time"]), dtype=complex
            )
    elif config["ast"]["mean"] == "est":
        ast_k_prior_mean = estimates["ast_k_est"]
    elif config["ast"]["mean"] == "truth":
        ast_k_prior_mean = true_params["ast_k_r"] + 1.0j * true_params["ast_k_i"]
    elif config["ast"]["mean"] == "truth_mean":
        ast_k_prior_mean = jnp.fft.fft(
            (true_params["ast_k_r"] + 1.0j * true_params["ast_k_i"]).mean(axis=0)[
                :, None
            ]
            * jnp.ones((ms_params["n_bl"], ms_params["n_time"])),
            axis=1,
        )
    else:
        ValueError("ast: mean: must be one of (est, prior, truth, truth_mean)")

    # Set RFI Prior Mean
    if config["rfi"]["mean"] == 0:
        rfi_prior_mean = jnp.zeros(
            (n_rfi, ms_params["n_ant"], gp_params["n_rfi_times"]), dtype=complex
        )
    elif config["rfi"]["mean"] == "est":
        rfi_prior_mean = estimates["rfi_induce_est"]
    elif config["rfi"]["mean"] == "truth":
        rfi_prior_mean = (
            true_params["rfi_r_induce"] + 1.0j * true_params["rfi_i_induce"]
        )
    else:
        ValueError("rfi: mean: must be one of (est, prior, truth)")

    prior_means = {
        "g_amp_prior_mean": g_amp_prior_mean,
        "g_phase_prior_mean": g_phase_prior_mean,
        "ast_k_prior_mean": ast_k_prior_mean,
        "rfi_prior_mean": rfi_prior_mean,
    }

    return prior_means


def get_init_params(config, ms_params, prior_means, estimates, true_params):

    # Set RFI parameter initialisation
    if config["rfi"]["init"] == "est":
        rfi_induce_init = estimates["rfi_induce_est"]  # Estimate from data
    elif config["rfi"]["init"] == "prior":
        rfi_induce_init = prior_means["rfi_prior_mean"]  # Prior mean value
    elif config["rfi"]["init"] == "truth":
        rfi_induce_init = (
            true_params["rfi_r_induce"] + 1.0j * true_params["rfi_i_induce"]
        )  # True value
    else:
        ValueError("rfi: init: must be one of (est, prior, truth)")

    # Set Astronomical parameter initialisation
    if config["ast"]["init"] == "est":
        ast_k_init = estimates["ast_k_est"]  # Estimate from data
    elif config["ast"]["init"] == "prior":
        ast_k_init = prior_means["ast_k_prior_mean"]  # Prior mean value
    elif config["ast"]["init"] == "truth":
        ast_k_init = (
            true_params["ast_k_r"] + 1.0j * true_params["ast_k_i"]
        )  # True value
    elif config["ast"]["init"] == "truth_mean":
        ast_k_init = jnp.fft.fft(
            (true_params["ast_k_r"] + 1.0j * true_params["ast_k_i"]).mean(axis=0)[
                :, None
            ]
            * jnp.ones((ms_params["n_bl"], ms_params["n_time"])),
            axis=1,
        )  # Mean of true value
    else:
        ValueError("ast: init: must be one of (est, prior, truth, truth_mean)")

    if config["gains"]["init"] == "truth":
        g_amp_init = true_params["g_amp_induce"]
        g_phase_init = true_params["g_phase_induce"]
    elif config["gains"]["init"] == "prior":
        g_amp_init = prior_means["g_amp_prior_mean"]
        g_phase_init = prior_means["g_phase_prior_mean"]
    else:
        ValueError("gains: init: must be one of (prior, truth)")

    init_params = {
        "g_amp_induce": g_amp_init,
        "g_phase_induce": g_phase_init,
        "ast_k_r": ast_k_init.real,
        "ast_k_i": ast_k_init.imag,
        "rfi_r_induce": rfi_induce_init.real,
        "rfi_i_induce": rfi_induce_init.imag,
    }

    return init_params


def get_gp_params(config, ms_params):
    ### GP Parameters

    # Gain GP Parameters
    g_amp_var = (config["gains"]["amp_std"] / 100) ** 2  # convert % to decimal
    g_phase_var = (
        jnp.deg2rad(config["gains"]["phase_std"]) ** 2
    )  # convert degrees to radians
    g_l = 60.0 * config["gains"]["corr_time"]  # convert minutes to seconds

    # RFI GP Parameters
    if config["rfi"]["var"] == "truth":
        xds = xr.open_zarr(config["data"]["zarr_path"])
        rfi_var = jnp.max(xds.rfi_tle_sat_A.data.compute() ** 2)
    elif config["rfi"]["var"]:
        rfi_var = config["rfi"]["var"]
    else:
        rfi_var = estimate_max_rfi_vis(ms_params)
    # rfi_l can be calculated based on RFI positions. Check True value definitions
    rfi_l = jnp.array(config["rfi"]["corr_time"])

    ### Gain Sampling Times
    g_times = get_times(ms_params["times"], g_l)
    n_g_times = len(g_times)

    ### RFI Sampling Times
    rfi_times = get_times(ms_params["times"], rfi_l)
    n_rfi_times = len(rfi_times)

    gp_params = {
        "g_amp_var": g_amp_var,
        "g_phase_var": g_phase_var,
        "g_l": g_l,
        "g_times": g_times,
        "n_g_times": n_g_times,
        "rfi_var": rfi_var,
        "rfi_l": rfi_l,
        "rfi_times": rfi_times,
        "n_rfi_times": n_rfi_times,
        "ast_pad": int(
            max([config["ast"]["pad_factor"] * ms_params["n_time"] // 2, 1])
        ),
    }

    return gp_params


# Square root of the power spectrum in the time axis for the astronomical visibilities
@jit
def pow_spec_sqrt(k, P0=1e3, k0=1e-3, gamma=1.0):

    k_ = (k / k0) ** 2
    Pk = P0 * 0.5 * (jnp.exp(-0.5 * k_) + 1.0 / ((1.0 + k_) ** (gamma / 2)))

    return Pk


@jit
def pow_spec(k, P0=1e7, k0=1e-3, gamma=1.0):

    k_ = k / k0
    Pk = P0 * 0.5 * (jnp.exp(-(k_**2)) + (1.0 + k_**2) ** -gamma)
    # Pk = P0 / (1.0 + k_**2) ** gamma
    # Pk = P0 * jnp.exp(-(k_**2)) # Leads to NaN values after division

    return Pk


@jit
def get_rfi_phase_from_pos(rfi_xyz, ants_w, ants_xyz, freqs):

    c = 299792458.0
    lam = c / freqs
    c_dist = (
        jnp.linalg.norm(rfi_xyz[:, :, None, :] - ants_xyz[None, :, :, :], axis=-1)
        + ants_w[None, :, :]
    )
    phase = -2.0 * jnp.pi * c_dist[:, :, :, None] / lam[None, None, None, :]

    return phase


def get_antenna_positions(ms_params: dict, n_int_samples: int):

    times_fine = int_sample_times(ms_params["times"], n_int_samples).compute()
    times_mjd_fine = ms_params["times_mjd"][0] + secs_to_days(times_fine)
    # times_mjd_fine = int_sample_times(ms_params["times_mjd"], n_int_samples).compute()

    gsa = (
        Time(times_mjd_fine, format="mjd").sidereal_time("mean", "greenwich").hour * 15
    )  # Convert hours to degrees
    # gsa = gmsa_from_jd(mjd_to_jd(times_mjd_fine)) % 360
    ants_xyz = itrf_to_xyz(ms_params["ants_itrf"], gsa)

    return ants_xyz


def get_sat_positions(ms_params: dict, n_int_samples: int, tles: list):

    times_fine = int_sample_times(ms_params["times"], n_int_samples).compute()
    times_mjd_fine = ms_params["times_mjd"][0] + secs_to_days(times_fine)
    # times_mjd_fine = int_sample_times(ms_params["times_mjd"], n_int_samples).compute()

    rfi_xyz = get_satellite_positions(tles, np.array(mjd_to_jd(times_mjd_fine)))

    return rfi_xyz


def check_antenna_and_satellite_positions(config: dict, ms_params: dict, tles: list):

    xds = xr.open_zarr(config["data"]["zarr_path"])
    n_int_samples = xds.n_int_samples
    ants_xyz_true = xds.ants_xyz.data.compute()
    sat_xyz_true = xds.rfi_tle_sat_xyz.data.compute()

    ants_xyz = get_antenna_positions(ms_params, n_int_samples)
    sat_xyz = get_sat_positions(ms_params, n_int_samples, tles)

    ants_diff = jnp.sum(jnp.linalg.norm(ants_xyz_true - ants_xyz, axis=-1))
    sat_diff = jnp.sum(jnp.linalg.norm(sat_xyz_true - sat_xyz, axis=-1))

    print()
    print(f"Antenna position differences   : {ants_diff:.3f} m")
    print(f"Satellite position differences : {sat_diff:.3f} m")


def get_antenna_uvw(ms_params: dict, n_int_samples: int):

    times_fine = int_sample_times(ms_params["times"], n_int_samples).compute()
    times_mjd_fine = ms_params["times_mjd"][0] + secs_to_days(times_fine)
    # times_mjd_fine = int_sample_times(ms_params["times_mjd"], n_int_samples).compute()

    gsa = (
        Time(times_mjd_fine, format="mjd").sidereal_time("mean", "greenwich").hour * 15
    )  # Convert hours to degrees
    # gsa = gmsa_from_jd(mjd_to_jd(times_mjd_fine)) % 360
    gh0 = (gsa - ms_params["ra"]) % 360

    ants_uvw = itrf_to_uvw(ms_params["ants_itrf"], gh0, ms_params["dec"])

    return ants_uvw


def get_rfi_phase(ms_params: dict, norad_ids: list, tles: list, n_int_samples: int):

    # Beware of time definitions can lead to RFI and antenna position inaccuracies
    times_fine = int_sample_times(ms_params["times"], n_int_samples).compute()
    times_mjd_fine = ms_params["times_mjd"][0] + secs_to_days(times_fine)
    # times_mjd_fine = int_sample_times(ms_params["times_mjd"], n_int_samples).compute()

    dt = np.diff(times_fine)[0]
    dt_jd = np.diff(times_mjd_fine)[0]

    times_fine = np.concatenate([times_fine, times_fine[-1:] + dt])
    times_mjd_fine = np.concatenate([times_mjd_fine, times_mjd_fine[-1:] + dt_jd])

    gsa = (
        Time(times_mjd_fine, format="mjd").sidereal_time("mean", "greenwich").hour * 15
    )  # Convert hours to degrees
    # gsa = gmsa_from_jd(mjd_to_jd(times_mjd_fine)) % 360
    gh0 = (gsa - ms_params["ra"]) % 360

    ants_uvw = itrf_to_uvw(
        ms_params["ants_itrf"], gh0, ms_params["dec"]
    )  # [:,:,2] # We need the uvw-coordinates at the fine sampling rate for the RFI
    ants_xyz = itrf_to_xyz(ms_params["ants_itrf"], gsa)

    if len(norad_ids) > 0:
        rfi_xyz = get_satellite_positions(tles, np.array(mjd_to_jd(times_mjd_fine)))
        n_sat = rfi_xyz.shape[0]
        # rfi_xyz_error = (
        #     config["satellites"]["pos_error"]
        #     / jnp.sqrt(3)
        #     * random.normal(
        #         random.PRNGKey(int(ms_params["ra"])),
        #         (n_sat, 1, 3),
        #     )
        # )
        # rfi_xyz = rfi_xyz + rfi_xyz_error
        rfi_phase = jnp.transpose(
            get_rfi_phase_from_pos(
                rfi_xyz, ants_uvw[..., -1], ants_xyz, ms_params["freqs"]
            )[..., 0],
            axes=(0, 2, 1),
        )

    ang_seps = jnp.min(
        angular_separation(
            rfi_xyz,
            jnp.mean(ants_xyz, axis=1, keepdims=True),
            ms_params["ra"],
            ms_params["dec"],
        ),
        axis=(1, 2),
    )
    rfi_amps = 1 / jnp.deg2rad(ang_seps) ** 2
    norm = jnp.sum(rfi_amps)
    rfi_amp_ratios = rfi_amps / norm

    print()
    print("Minimum Angular Separations")
    for norad_id, ang_sep in zip(norad_ids, ang_seps):
        print(f"{norad_id} : {ang_sep:.1f} degrees")
    print()

    # if len(config["satellites"]["sat_ids"])>0:
    #     ole_df = pd.read_csv(config["satellites"]["ole_path"])
    #     oles = ole_df[ole_df["sat_id"].isin(config["satellites"]["sat_ids"])].values.T
    #     rfi_xyz = vmap(orbit, axes=(None,0,0,0,0))(times_fine, *oles)
    #     rfi_phase = jnp.transpose(
    #     get_rfi_phase_from_pos(rfi_xyz, ants_uvw[...,-1], ants_xyz, ms_params["freqs"])[...,0], axes=(0,2,1)
    #     )
    #
    #     rfi_phase = jnp.array(
    #         [
    #             get_rfi_phase(times_fine, orbit, ants_uvw, ants_xyz, ms_params["freqs"]).T
    #             for orbit in rfi_orbit
    #         ]
    #     )

    return rfi_phase, rfi_amp_ratios, times_fine, times_mjd_fine


def run_mcmc(
    ms_params,
    model,
    model_name,
    subkeys,
    static_args,
    array_args,
    init_params,
    plot_dir,
    mcmc_path,
    num_warmup=500,
    num_samples=1000,
    max_tree_depth=10,
    thin_factor=1,
):

    print(
        f"Running MCMC with {num_warmup:.0f} Warm Up Samples and for {num_samples:.0f} Samples"
    )
    start = datetime.now()

    nuts_kernel = NUTS(
        model, dense_mass=False, max_tree_depth=max_tree_depth
    )  # [('g_phase_0', 'g_phase_1')])
    mcmc = MCMC(
        nuts_kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        thinning=thin_factor,
    )
    mcmc.run(
        subkeys[0],
        static_args=static_args,
        array_args=array_args,
        v_obs=array_args["v_obs_ri"],
        extra_fields=("potential_energy",),
        init_params=init_params,
    )
    print()
    print(f"MCMC Run Time : {datetime.now() - start}")
    print(f"{datetime.now()}")
    start = datetime.now()

    # return mcmc

    param_samples = mcmc.get_samples()
    # pred = Predictive(model, posterior_samples=param_samples)
    # print(param_samples.keys())
    # mcmc_pred = pred(subkeys[1], static_args=static_args, array_args=array_args)
    # print(param_samples)

    # print(param_samples.keys())

    write_results_xds(param_samples, array_args, mcmc_path)
    print(f"Results Written to disk at {mcmc_path}")
    # write_results_xds(mcmc_pred, array_args, mcmc_path)
    # write_params_xds(ms, )

    plot_predictions(
        times=ms_params["times"],
        pred=param_samples,
        args=array_args,
        type="mcmc",
        model_name=model_name,
        max_plots=10,
        save_dir=plot_dir,
    )
    print()
    print(f"MCMC Plot Time : {datetime.now() - start}")
    print(f"{datetime.now()}")


def run_opt(
    config,
    ms_params,
    gp_params,
    model,
    model_name,
    # args,
    static_args,
    array_args,
    subkeys,
    init_params,
    plot_dir,
    ms_path,
    map_path,
    params_path,
):

    guides = {
        "map": "AutoDelta",
    }
    start = datetime.now()
    print()
    print("Running Optimization ...")
    guide_family = guides[config["opt"]["guide"]]
    vi_results, vi_guide = run_svi(
        model=model,
        # args=args,
        static_args=static_args,
        array_args=array_args,
        obs=array_args["v_obs_ri"],
        max_iter=config["opt"]["max_iter"],
        guide_family=guide_family,
        init_params={
            **{k + "_auto_loc": v for k, v in init_params.items()},
        },
        epsilon=config["opt"]["epsilon"],
        key=subkeys[0],
        dual_run=config["opt"]["dual_run"],
    )
    vi_params = vi_results.params
    with jax.profiler.trace(
        "/home/chris/tab_dev/tabascal/log", create_perfetto_link=True
    ):
        vi_pred = svi_predict(
            model=model,
            guide=vi_guide,
            vi_params=vi_params,
            static_args=static_args,
            array_args=array_args,
            # args=args,
            num_samples=1,
            key=subkeys[1],
        )
    print()
    print(f"Optimization Run Time : {datetime.now() - start}")
    print(f"{datetime.now()}")
    start = datetime.now()

    write_results_xds(vi_pred, array_args, map_path)
    # write_params_xds(vi_params, gp_params, ms_params, params_path, overwrite=True)

    plot_predictions(
        ms_params["times"],
        pred=vi_pred,
        args=array_args,
        type=config["opt"]["guide"],
        model_name=model_name,
        max_plots=10,
        save_dir=plot_dir,
    )
    print()
    print(f"Optimize Plot Time : {datetime.now() - start}")
    print(f"{datetime.now()}")

    if get_truth_conditional(config):

        # vi_pred keys are ['ast_vis', 'gains', 'rfi_vis', 'rmse_ast', 'rmse_gains', 'rmse_rfi', 'vis_obs']
        print(f"RMSE Gains      : {jnp.mean(vi_pred['rmse_gains']):.5f}")
        print(f"RMSE RFI Vis    : {jnp.mean(vi_pred['rmse_rfi']):.5f}")
        print(f"RMSE AST Vis    : {jnp.mean(vi_pred['rmse_ast']):.5f}")

    rchi2 = reduced_chi2(
        vi_pred["vis_obs"][0], ms_params["vis_obs"].T, ms_params["noise"]
    )
    print()
    print(f"Reduced Chi^2 @ opt params : {rchi2}")

    plt.semilogy(vi_results.losses)
    plt.savefig(os.path.join(plot_dir, f"{model_name}_opt_loss.pdf"), format="pdf")

    print()
    print(
        "Copying tabascal results to MS file in 'TAB_DATA' and 'TAB_RFI_DATA' columns"
    )
    print(os.path.split(map_path)[1])
    subprocess.run(
        f"tab2MS -m {ms_path} -z {map_path}", shell=True, executable="/bin/bash"
    )

    return vi_params, rchi2


from functools import partial
from frozendict import frozendict


@partial(jit, static_argnums=(0, 2))
def f_model_flat(model, params, static_args, array_args):
    vis_obs = model(params, static_args, array_args)
    vis_obs_flat = flatten_obs(vis_obs)
    return vis_obs_flat


def run_fisher(
    config,
    gp_params,
    ms_params,
    model,
    model_name,
    subkeys,
    vis_model,
    static_args,
    array_args,
    vi_params,
    init_params,
    plot_dir,
    fisher_path,
):

    start = datetime.now()
    n_fisher = config["fisher"]["n_samples"]
    print(f"Calculating {n_fisher:.0f} Fisher Samples ...")

    # f_model = lambda params, args: vis_model(params, args)[0]
    # model_flat = lambda params: f_model_flat(f_model, params, args)

    f_model = lambda params, static_args, array_args: vis_model(
        params, static_args, array_args
    )[0]
    model_flat = lambda params: f_model_flat(f_model, params, static_args, array_args)

    post_mean = (
        {k[:-9]: v for k, v in vi_params.items()}
        if config["inference"]["opt"]
        else init_params
    )

    dtheta = post_samples(
        model_flat,
        post_mean,
        flatten_obs(ms_params["vis_obs"]),
        ms_params["noise"],
        n_fisher,
        subkeys[0],
        config["fisher"]["max_cg_iter"],
    )
    print()
    print(f"Fisher Run Time : {datetime.now() - start}")
    print(f"{datetime.now()}")
    start = datetime.now()

    samples = tree_map(jnp.add, post_mean, dtheta)

    pred = Predictive(model, posterior_samples=samples)
    fisher_pred = pred(subkeys[1], static_args=static_args, array_args=array_args)

    write_results_xds(fisher_pred, array_args, fisher_path)
    # fisher_xds = write_params_xds(fisher_pred, gp_params, ms_params, fisher_path)

    plot_predictions(
        times=ms_params["times"],
        pred=fisher_pred,
        args=array_args,
        type="fisher_opt" if config["inference"]["opt"] else "fisher_true",
        model_name=model_name,
        max_plots=10,
        save_dir=plot_dir,
    )
    print()
    print(f"Fisher Plot Time : {datetime.now() - start}")
    print(f"{datetime.now()}")


# def init_predict(ms_params, model, args, subkey, init_params):

#     pred = Predictive(
#         model=model,
#         posterior_samples=tree_map(lambda x: x[None, :], init_params),
#         batch_ndims=1,
#     )
#     init_pred = pred(subkey, args=args)
#     rchi2 = reduced_chi2(
#         init_pred["vis_obs"][0], ms_params["vis_obs"].T, ms_params["noise"]
#     )
#     print()
#     print(f"Reduced Chi^2 @ init params : {rchi2}")

#     return init_pred


def init_predict(ms_params, model, static_args, array_args, subkey, init_params):

    pred = Predictive(
        model=model,
        posterior_samples=tree_map(lambda x: x[None, :], init_params),
        batch_ndims=1,
    )
    init_pred = pred(subkey, static_args=static_args, array_args=array_args)
    rchi2 = reduced_chi2(
        init_pred["vis_obs"][0], ms_params["vis_obs"].T, ms_params["noise"]
    )
    print()
    print(f"Reduced Chi^2 @ init params : {rchi2}")

    return init_pred


def plot_init(ms_params, config, init_pred, array_args, model_name, plot_dir):

    start = datetime.now()
    print()
    print("Plotting Initial Parameters")
    plot_predictions(
        times=ms_params["times"],
        pred=init_pred,
        args=array_args,
        type="init",
        model_name=model_name,
        max_plots=10,
        save_dir=plot_dir,
    )
    if get_truth_conditional(config):

        # vi_pred keys are ['ast_vis', 'gains', 'rfi_vis', 'rmse_ast', 'rmse_gains', 'rmse_rfi', 'vis_obs']
        print(f"RMSE Gains      : {jnp.mean(init_pred['rmse_gains']):.5f}")
        print(f"RMSE RFI Vis    : {jnp.mean(init_pred['rmse_rfi']):.5f}")
        print(f"RMSE AST Vis    : {jnp.mean(init_pred['rmse_ast']):.5f}")

    print()
    print(f"Initial Plot Time : {datetime.now() - start}")
    print(f"{datetime.now()}")


def plot_truth(
    zarr_path,
    ms_params,
    static_args,
    array_args,
    model,
    model_name,
    subkey,
    true_params,
    gp_params,
    inv_scaling,
    plot_dir,
    true_path,
):

    start = datetime.now()
    print()
    print("Plotting True Parameters")

    xds = xr.open_zarr(zarr_path)
    rfi_A_true = jnp.transpose(
        xds.rfi_tle_sat_A.data[:, :, :, 0].compute(), axes=(0, 2, 1)
    )
    rfi_resample = resampling_kernel(
        gp_params["rfi_times"],
        int_sample_times(ms_params["times"], xds.n_int_samples).compute(),
        gp_params["rfi_var"],
        gp_params["rfi_l"],
        1e-8,
    )
    rfi_amp = true_params["rfi_r_induce"] + 1.0j * true_params["rfi_i_induce"]
    rfi_A_pred = vmap(lambda x, y: x @ y.T, in_axes=(0, None))(rfi_amp, rfi_resample)

    true_params_base = inv_transform(true_params, array_args, inv_scaling)
    pred = Predictive(
        model=model,
        posterior_samples=tree_map(lambda x: x[None, :], true_params_base),
        batch_ndims=1,
    )
    true_pred = pred(subkey, static_args=static_args, array_args=array_args)
    write_results_xds(true_pred, array_args, true_path)

    # true_pred keys are ['ast_vis', 'gains', 'rfi_vis', 'rmse_ast', 'rmse_gains', 'rmse_rfi', 'vis_obs']
    print(f"RMSE Gains      : {jnp.mean(true_pred['rmse_gains']):.5f}")
    print(
        f"RMSE RFI signal : {jnp.sqrt(jnp.mean(jnp.abs(rfi_A_true - rfi_A_pred)**2)):.5f}"
    )
    print(f"RMSE RFI Vis    : {jnp.mean(true_pred['rmse_rfi']):.5f}")
    print(f"RMSE AST Vis    : {jnp.mean(true_pred['rmse_ast']):.5f}")

    rchi2 = reduced_chi2(
        true_pred["vis_obs"][0], ms_params["vis_obs"].T, ms_params["noise"]
    )
    print()
    print(f"Reduced Chi^2 @ true params : {rchi2}")

    plot_predictions(
        times=ms_params["times"],
        pred=true_pred,
        args=array_args,
        type="true",
        model_name=model_name,
        max_plots=10,
        save_dir=plot_dir,
    )
    print()
    print(f"True Plot Time : {datetime.now() - start}")
    print(f"{datetime.now()}")


def plot_prior(
    config, ms_params, model, model_name, static_args, array_args, subkey, plot_dir
):

    start = datetime.now()
    n_prior = config["plots"]["prior_samples"]
    print()
    print(f"Plotting {n_prior:.0f} Prior Parameter Samples")
    pred = Predictive(model, num_samples=n_prior)
    prior_pred = pred(subkey, static_args=static_args, array_args=array_args)
    print("Prior Samples Drawn")
    plot_predictions(
        times=ms_params["times"],
        pred=prior_pred,
        args=array_args,
        type="prior",
        model_name=model_name,
        max_plots=10,
        save_dir=plot_dir,
    )
    print()
    print(f"Prior Plot Time : {datetime.now() - start}")
    print(f"{datetime.now()}")


def save_memory(mem_dir, mem_i):

    mem_i += 1
    jax.profiler.save_device_memory_profile(
        os.path.join(mem_dir, f"memory_{mem_i}.prof")
    )

    return mem_i
