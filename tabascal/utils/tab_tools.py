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
from jax import jit, vmap
from jax.tree_util import tree_map

from numpyro.infer import MCMC, NUTS, Predictive

from tabascal.jax.coordinates import calculate_sat_corr_time, orbit, itrf_to_uvw, itrf_to_xyz, calculate_fringe_frequency, gmsa_from_jd
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

def print_rfi_signal_error(zarr_path, ms_params, true_params, gp_params):

    xds = xr.open_zarr(zarr_path)
    rfi_A_true = jnp.transpose(xds.rfi_tle_sat_A.data[:,:,:,0].compute(), axes=(0,2,1))
    rfi_resample = resampling_kernel(
        gp_params["rfi_times"], 
        int_sample_times(ms_params["times"], xds.n_int_samples).compute(), 
        gp_params["rfi_var"], 
        gp_params["rfi_l"], 
        1e-8
        )
    rfi_amp = true_params["rfi_r_induce"] + 1.0j*true_params["rfi_i_induce"]
    rfi_A_pred = vmap(lambda x, y: x @ y.T, in_axes=(0, None))(rfi_amp, rfi_resample)
    print(f"RMSE RFI signal : {jnp.sqrt(jnp.mean(jnp.abs(rfi_A_true - rfi_A_pred)**2)):.5f}")

# @jit
def reduced_chi2(pred, true, noise):

    if isinstance(true.flatten()[0], np.complex64):
        # print("Complex Data")
        norm = 2 * true.size  
    else:
        norm = true.size

    rchi2 = jnp.sum((jnp.abs(pred - true) / noise) ** 2) / norm

    return rchi2


def write_xds(vi_pred, times, file_path, overwrite=True):

    map_xds = xr.Dataset(
        data_vars={
            "ast_vis": (["sample", "bl", "time"], da.asarray(vi_pred["ast_vis"])),
            "gains": (["sample", "ant", "time"], da.asarray(vi_pred["gains"])),
            "rfi_vis": (["sample", "bl", "time"], da.asarray(vi_pred["rfi_vis"])),
            "vis_obs": (["sample", "bl", "time"], da.asarray(vi_pred["vis_obs"])),
            },
        coords={"time": da.asarray(times)},
        )
    
    mode = "w" if overwrite else "w-"

    map_xds.to_zarr(file_path, mode=mode)
    
    return map_xds


@jit
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


def read_ms(ms_path, freq: float=None, corr: str="xx"):

    correlations = {"xx": 0, "xy": 1, "yx": 2, "yy": 3}
    corr = correlations[corr]

    xds = xds_from_ms(ms_path)[0]
    xds_ant = xds_from_table(ms_path+"::ANTENNA")[0]
    xds_spec = xds_from_table(ms_path+"::SPECTRAL_WINDOW")[0]
    xds_src = xds_from_table(ms_path+"::SOURCE")[0]

    ants_itrf = xds_ant.POSITION.data.compute()

    n_ant = ants_itrf.shape[0]
    n_time = len(jnp.unique(xds.TIME.data.compute()))
    n_bl = xds.DATA.data.shape[0] // n_time
    n_freq, n_corr = xds.DATA.data.shape[1:]

    freqs = xds_spec.CHAN_FREQ.data.compute()

    times_jd = xds.TIME.data.reshape(n_time, n_bl)[:,0].compute()

    if freq:
        chan = jnp.argmin(jnp.abs(freq - freqs))
    else: 
        chan = 0

    data = {
        **{key: val for key, val in zip(["ra", "dec"], jnp.rad2deg(xds_src.DIRECTION.data[0].compute()))},
        "n_freq": n_freq,
        "n_corr": n_corr,
        "n_time": n_time,
        "n_ant": n_ant,
        "n_bl": n_bl,
        "times_jd": times_jd,
        "times": times_jd * 24 * 3600, # Convert Julian date in days to seconds
        "int_time": xds.INTERVAL.data[0].compute(),
        "freqs": freqs[chan],
        "ants_itrf": ants_itrf,
        "vis_obs": xds.DATA.data.reshape(n_time, n_bl, n_freq, n_corr).compute()[:,:,chan, corr],
        "noise": xds.SIGMA.data.mean().compute(),
        "a1": xds.ANTENNA1.data.reshape(n_time, n_bl)[0,:].compute(),
        "a2": xds.ANTENNA2.data.reshape(n_time, n_bl)[0,:].compute(),
    }

    return data

def get_tles(config, ms_params, norad_ids, spacetrack_path):
    if config["satellites"]["norad_ids_path"]:
        norad_ids += [int(x) for x in yaml_load(config["satellites"]["norad_ids_path"]).split()]

    if len(config["satellites"]["norad_ids"])>0:
        norad_ids += config["satellites"]["norad_ids"]

    n_rfi = len(norad_ids) + len(config["satellites"]["sat_ids"])
    
    # if len(config["satellites"]["sat_ids"])>0:
    #     import pandas as pd
    #     ole_df = pd.read_csv(config["satellites"]["ole_path"])
    #     oles = ole_df[ole_df["sat_id"].isin(config["satellites"]["sat_ids"])][["elevation", "inclination", "lon_asc_node", "periapsis"]]
    #     rfi_orbit = jnp.atleast_2d(oles.values)

    if len(norad_ids)>0 and spacetrack_path:
        st_config = yaml_load(spacetrack_path)
        tles_df = get_tles_by_id(
            st_config["username"], 
            st_config["password"], 
            norad_ids,
            jnp.mean(ms_params["times_jd"]),
            tle_dir=config["satellites"]["tle_dir"],
            )
        tles = np.atleast_2d(tles_df[["TLE_LINE1", "TLE_LINE2"]].values)

    # return n_rfi, norad_ids, tles, rfi_orbit
    return n_rfi, norad_ids, tles


# def estimate_sampling(config: dict, ms_params: dict, n_rfi: int, norad_ids, tles: list[list[str]], rfi_orbit):
def estimate_sampling(config: dict, ms_params: dict, n_rfi: int, norad_ids, tles: list[list[str]]):

    jd_minute = 1 / (24*60)
    times_jd_coarse =  jnp.arange(ms_params["times_jd"][0], ms_params["times_jd"][-1]+jd_minute, jd_minute)
    times_coarse = times_jd_coarse * 24 * 3600
    # times_coarse = Time(times_jd_coarse, format="jd").sidereal_time("mean", "greenwich").hour*3600 # Convert hours to seconds

    if len(norad_ids)>0:
        rfi_xyz = get_satellite_positions(tles, np.array(times_jd_coarse))
    # if len(config["satellites"]["sat_ids"])>0:
    #     rfi_xyz = jnp.array([orbit(times_coarse, *rfi_orb) for rfi_orb in rfi_orbit])

    # gsa = Time(times_jd_coarse, format="jd").sidereal_time("mean", "greenwich").hour*15 # Convert hours to degrees
    gsa = gmsa_from_jd(times_jd_coarse)
    gh0 = (gsa - ms_params["ra"]) % 360
    ants_u = itrf_to_uvw(ms_params["ants_itrf"], gh0, ms_params["dec"])[:,:,0] # We want the uvw-coordinates at the coarse sampling rate for the fringe frequency prediction

    fringe_params = [{
        "times": times_coarse,
        "freq": jnp.max(ms_params["freqs"]),
        "rfi_xyz": rfi_xyz[i],
        "ants_itrf": ms_params["ants_itrf"],
        "ants_u": ants_u,
        "dec": ms_params["dec"],
    } for i in range(n_rfi)]

    fringe_freq = jnp.array([calculate_fringe_frequency(**f_params) for f_params in fringe_params])
    print(f"Max Fringe Freq: {jnp.max(jnp.abs(fringe_freq)):.2f} Hz")

    sample_freq = jnp.pi * jnp.max(jnp.abs(fringe_freq)) * jnp.sqrt(jnp.max(jnp.abs(ms_params["vis_obs"])) / (6 * ms_params["noise"]))
    n_int_samples = int(jnp.ceil(config["rfi"]["n_int_factor"] * ms_params["int_time"] * sample_freq))
    # n_int_samples = 65

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
            config["gains"]["amp_mean"]=="truth",
            config["gains"]["phase_mean"]=="truth",
            config["ast"]["mean"]=="truth",
            config["rfi"]["mean"]=="truth",
            config["ast"]["mean"]=="truth_mean",
            config["ast"]["init"]=="truth",
            config["rfi"]["init"]=="truth",
            config["ast"]["init"]=="truth_mean",
        ]
    )

    return jnp.any(truth_cond)


def calculate_true_values(zarr_path: str, config: dict, ms_params: dict, gp_params, n_rfi, norad_ids):
    xds = xr.open_zarr(zarr_path)

    vis_ast = xds.vis_ast.data[:,:,0].compute()
    vis_rfi = xds.vis_rfi.data[:,:,0].compute()
    gains_ants = xds.gains_ants.data[:,:,0].compute()
    a1 = xds.antenna1.data.compute()
    a2 = xds.antenna2.data.compute()

    vis = gains_ants[:,a1] * jnp.conjugate(gains_ants[:,a2]) * (vis_ast + vis_rfi)

    rchi2 = reduced_chi2(vis, ms_params["vis_obs"], ms_params["noise"])
    print()
    print(f"Reduced Chi^2 @ truth : {rchi2}")

    if len(config["satellites"]["sat_ids"])>0:
        corr_time_params = [{
            "sat_xyz": xds.rfi_sat_xyz.data[i,xds.time_idx].compute(),
            "ants_xyz": xds.ants_xyz.data[xds.time_idx].compute(),
            "orbit_el": xds.rfi_sat_orbit.data[i,0].compute(),
            "lat": xds.tel_latitude,
            "dish_d": xds.dish_diameter,
            "freqs": ms_params["freqs"],
        } for i in range(n_rfi)]

        l = jnp.min(jnp.array([calculate_sat_corr_time(**corr_time_params[i]) for i in range(n_rfi)]))

        rfi_induce = jnp.array(
            [
                vmap(jnp.interp, in_axes=(None, None, 1), out_axes=(0))(
                    gp_params["rfi_times"], xds.time_fine.data, xds.rfi_sat_A[:,:,:,0].data.compute()[i]
                )
                for i in range(n_rfi)
            ]
        )

    if len(norad_ids)>0:
        corr_time_params = [{
            "sat_xyz": xds.rfi_tle_sat_xyz.data[i,xds.time_idx].compute(),
            "ants_xyz": xds.ants_xyz.data[xds.time_idx].compute(),
            "orbit_el": get_orbit_elevation(xds.rfi_tle_sat_xyz.data[i,xds.time_idx].compute(), xds.tel_latitude),
            "lat": xds.tel_latitude,
            "dish_d": xds.dish_diameter,
            "freqs": ms_params["freqs"],
        } for i in range(n_rfi)]

        l = jnp.min(jnp.array([calculate_sat_corr_time(**corr_time_params[i]) for i in range(n_rfi)]))

        rfi_induce = jnp.array(
            [
                vmap(jnp.interp, in_axes=(None, None, 1), out_axes=(0))(
                    gp_params["rfi_times"], xds.time_fine.data, xds.rfi_tle_sat_A[:,:,:,0].data.compute()[i]
                )
                for i in range(n_rfi)
            ]
        )

    print()
    print(f"Minimum expected RFI correlation time : {l:.0f} s ({gp_params["rfi_l"]:.0f} s used)")
    print()
    print(f"Mean RFI Amp. : {jnp.mean(jnp.abs(vis_rfi)):.1f} Jy")
    print(f"Mean AST Amp. : {jnp.mean(jnp.abs(vis_ast)):.1f} Jy")

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
        "vis_ast_true": vis_ast.T,
        "vis_rfi_true": vis_rfi.T,
        "gains_true": gains_ants.T,
    }

    return true_params, truth_args


def calculate_estimates(ms_params, n_rfi, rfi_times):

    vis_ast_est = jnp.mean(ms_params["vis_obs"].T, axis=1, keepdims=True) * jnp.ones((ms_params["n_bl"], ms_params["n_time"]))
    ast_k_est = jnp.fft.fft(vis_ast_est, axis=1)

    n_rfi_times = len(rfi_times)

    rfi_induce_est = (
        jnp.interp(rfi_times, 
                   ms_params["times"], 
                   jnp.sqrt(jnp.max(jnp.abs(ms_params["vis_obs"] - vis_ast_est.T), axis=1))
                   )[
            None, None, :
        ] # shape is now (1, 1, n_rfi_times)
        * jnp.ones((n_rfi, ms_params["n_ant"], n_rfi_times))
        / n_rfi
    )

    estimates = {
        "ast_k_est": ast_k_est,
        "rfi_induce_est": rfi_induce_est,
    }

    return estimates

def get_prior_means(config, ms_params, estimates, true_params, n_rfi, gp_params):

    # Set Gain Prior Mean
    if config["gains"]["amp_mean"] == "truth":
        g_amp_prior_mean = true_params["g_amp_induce"]
    elif isinstance(config["gains"]["amp_mean"], numbers.Number):
        g_amp_prior_mean = config["gains"]["amp_mean"] * jnp.ones((ms_params["n_ant"], gp_params["n_g_times"]))
    else:
        ValueError("gains: amp_mean: must be a number or 'truth'.")

    if config["gains"]["phase_mean"] == "truth":
        g_phase_prior_mean = true_params["g_phase_induce"]
    elif isinstance(config["gains"]["phase_mean"], numbers.Number):
        g_phase_prior_mean = jnp.deg2rad(config["gains"]["phase_mean"]) * jnp.ones((ms_params["n_ant"]-1, gp_params["n_g_times"]))
    else:
        ValueError("gains: phase_mean: must be a number or 'truth'.")

    # Set Astronomical Prior Mean
    if config["ast"]["mean"]==0:
        ast_k_prior_mean = jnp.zeros((ms_params["n_bl"], ms_params["n_time"]), dtype=complex)
    elif config["ast"]["mean"]=="est":
        ast_k_prior_mean = estimates["ast_k_est"]
    elif config["ast"]["mean"]=="truth":
        ast_k_prior_mean = true_params["ast_k_r"] + 1.0j * true_params["ast_k_i"]
    elif config["ast"]["mean"]=="truth_mean":
        ast_k_prior_mean = jnp.fft.fft(
        (true_params["ast_k_r"] + 1.0j * true_params["ast_k_i"]).mean(axis=0)[:, None] \
            * jnp.ones((ms_params["n_bl"], ms_params["n_time"])), axis=1
    )
    else:
        ValueError("ast: mean: must be one of (est, prior, truth, truth_mean)")

    # Set RFI Prior Mean
    if config["rfi"]["mean"]==0:
        rfi_prior_mean = jnp.zeros((n_rfi, ms_params["n_ant"], gp_params["n_rfi_times"]), dtype=complex)
    elif config["rfi"]["mean"]=="est":
        rfi_prior_mean = estimates["rfi_induce_est"]
    elif config["rfi"]["mean"]=="truth":
        rfi_prior_mean = true_params["rfi_r_induce"] + 1.0j * true_params["rfi_i_induce"]
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
        rfi_induce_init = estimates["rfi_induce_est"]    # Estimate from data
    elif config["rfi"]["init"] == "prior":
        rfi_induce_init = prior_means["rfi_prior_mean"]    # Prior mean value
    elif config["rfi"]["init"] == "truth":
        rfi_induce_init = true_params["rfi_r_induce"] + 1.0j * true_params["rfi_i_induce"]        # True value
    else:
        ValueError("rfi: init: must be one of (est, prior, truth)")

    # Set Astronomical parameter initialisation
    if config["ast"]["init"] == "est":
        ast_k_init = estimates["ast_k_est"]            # Estimate from data
    elif config["ast"]["init"] == "prior":
        ast_k_init = prior_means["ast_k_prior_mean"]       # Prior mean value
    elif config["ast"]["init"] == "truth":
        ast_k_init = true_params["ast_k_r"] + 1.0j * true_params["ast_k_i"]                  # True value
    elif config["ast"]["init"] == "truth_mean":
        ast_k_init = jnp.fft.fft(
        (true_params["ast_k_r"] + 1.0j * true_params["ast_k_i"]).mean(axis=0)[:, None] \
            * jnp.ones((ms_params["n_bl"], ms_params["n_time"])), axis=1
    )           # Mean of true value
    else:
        ValueError("ast: init: must be one of (est, prior, truth, truth_mean)")

    init_params = {
        "g_amp_induce": prior_means["g_amp_prior_mean"],
        "g_phase_induce": prior_means["g_phase_prior_mean"],
        "ast_k_r": ast_k_init.real,
        "ast_k_i": ast_k_init.imag,
        "rfi_r_induce": rfi_induce_init.real,
        "rfi_i_induce": rfi_induce_init.imag,
    }

    return init_params


def get_gp_params(config, ms_params):
    ### GP Parameters

    # Gain GP Parameters
    g_amp_var = (config["gains"]["amp_std"] / 100)**2 # convert % to decimal
    g_phase_var = jnp.deg2rad(config["gains"]["phase_std"])**2 # convert degrees to radians
    g_l = 60.0 * config["gains"]["corr_time"] # convert minutes to seconds

    # RFI GP Parameters
    if config["rfi"]["var"] is not None:
        rfi_var = config["rfi"]["var"]
    else:
        rfi_var = jnp.max(jnp.abs(ms_params["vis_obs"]))
    # rfi_l can be calculated based on RFI positions. Check True value definitions
    rfi_l = config["rfi"]["corr_time"]

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
    }

    return gp_params


# Square root of the power spectrum in the time axis for the astronomical visibilities
@jit
def pow_spec_sqrt(k, P0=1e3, k0=1e-3, gamma=1.0):

    k_ = (k / k0) ** 2
    Pk = P0 * 0.5 * (jnp.exp(-0.5 * k_) + 1.0 / ((1.0 + k_) ** (gamma / 2)))

    return Pk


@jit
def get_rfi_phase_from_pos(rfi_xyz, ants_w, ants_xyz, freqs):

    c = 299792458.0
    lam = c / freqs
    c_dist = jnp.linalg.norm(rfi_xyz[:,:,None,:] - ants_xyz[None,:,:,:], axis=-1) + ants_w[None,:,:]
    phase = -2.0 * jnp.pi * c_dist[:,:,:,None] / lam[None,None,None,:]

    return phase


def get_rfi_phase(ms_params, norad_ids, tles, n_int_samples):

    # Beware of time definitions can lead to RFI and antenna position inaccuracies
    times_fine = int_sample_times(ms_params["times"], n_int_samples).compute()
    times_jd_fine = int_sample_times(ms_params["times_jd"], n_int_samples).compute()

    # gsa = Time(times_jd_fine, format="jd").sidereal_time("mean", "greenwich").hour*15 # Convert hours to degrees
    gsa = gmsa_from_jd(times_jd_fine) % 360
    gh0 = (gsa - ms_params["ra"]) % 360

    ants_uvw = itrf_to_uvw(ms_params["ants_itrf"], gh0, ms_params["dec"])#[:,:,2] # We need the uvw-coordinates at the fine sampling rate for the RFI
    ants_xyz = itrf_to_xyz(ms_params["ants_itrf"], gsa)
    
    if len(norad_ids)>0:
        rfi_xyz = get_satellite_positions(tles, np.array(times_jd_fine))
        rfi_phase = jnp.transpose(get_rfi_phase_from_pos(rfi_xyz, ants_uvw[...,-1], ants_xyz, ms_params["freqs"])[...,0], axes=(0,2,1))

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

    return rfi_phase, times_fine

def run_mcmc(ms_params, model, model_name, subkeys, args, init_params, plot_dir):

    num_warmup = 500
    num_samples = 1000
    print(f"Running MCMC with {num_warmup:.0f} Warm Up Samples and for {num_samples:.0f} Samples")
    start = datetime.now()

    nuts_kernel = NUTS(model, dense_mass=False)  # [('g_phase_0', 'g_phase_1')])
    mcmc = MCMC(nuts_kernel, num_warmup=num_warmup, num_samples=num_samples)
    mcmc.run(
        subkeys[0],
        args=args,
        v_obs=args["v_obs_ri"],
        extra_fields=("potential_energy",),
        init_params=init_params,
    )
    print()
    print(f"MCMC Run Time : {datetime.now() - start}")
    print(f"{datetime.now()}")
    start = datetime.now()

    pred = Predictive(model, posterior_samples=mcmc.get_samples())
    mcmc_pred = pred(subkeys[1], args=args)
    plot_predictions(
        times=ms_params["times"],
        pred=mcmc_pred,
        args=args,
        type="mcmc",
        model_name=model_name,
        max_plots=10,
        save_dir=plot_dir,
    )
    print()
    print(f"MCMC Plot Time : {datetime.now() - start}")
    print(f"{datetime.now()}")


def run_opt(config, ms_params, model, model_name, args, subkeys, init_params, plot_dir, ms_path, map_path):

    guides = {
        "map": "AutoDelta",
    }
    start = datetime.now()
    print()
    print("Running Optimization ...")
    guide_family = guides[config["opt"]["guide"]]
    vi_results, vi_guide = run_svi(
        model=model,
        args=args,
        obs=args["v_obs_ri"],
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
    vi_pred = svi_predict(
        model=model,
        guide=vi_guide,
        vi_params=vi_params,
        args=args,
        num_samples=1,
        key=subkeys[1],
    )
    print()
    print(f"Optimization Run Time : {datetime.now() - start}")
    print(f"{datetime.now()}")
    start = datetime.now()

    map_xds = write_xds(vi_pred, ms_params["times"], map_path)

    plot_predictions(
        ms_params["times"],
        pred=vi_pred,
        args=args,
        type=config["opt"]["guide"],
        model_name=model_name,
        max_plots=10,
        save_dir=plot_dir,
    )
    print()
    print(f"Optimize Plot Time : {datetime.now() - start}")
    print(f"{datetime.now()}")

    rchi2 = reduced_chi2(vi_pred["vis_obs"][0], ms_params["vis_obs"].T, ms_params["noise"])
    print()
    print(f"Reduced Chi^2 @ opt params : {rchi2}")

    plt.semilogy(vi_results.losses)
    plt.savefig(os.path.join(plot_dir, f"{model_name}_opt_loss.pdf"), format="pdf")

    print()
    print("Copying tabascal results to MS file in TAB_DATA column")
    subprocess.run(f"tab2MS -m {ms_path} -z {map_path}", shell=True, executable="/bin/bash") 
    
    return vi_params, rchi2


def run_fisher(config, ms_params, model, model_name, subkeys, vis_model, args, vi_params, init_params, plot_dir, fisher_path):

    start = datetime.now()
    n_fisher = config["fisher"]["n_samples"]
    print(f"Calculating {n_fisher:.0f} Fisher Samples ...")

    f_model = lambda params, args: vis_model(params, args)[0]
    model_flat = lambda params: f_model_flat(f_model, params, args)

    post_mean = {k[:-9]: v for k, v in vi_params.items()} if config["inference"]["opt"] else init_params

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
    fisher_pred = pred(subkeys[1], args=args)

    fisher_xds = write_xds(fisher_pred, ms_params["times"], fisher_path)

    plot_predictions(
        times=ms_params["times"],
        pred=fisher_pred,
        args=args,
        type="fisher_opt" if config["inference"]["opt"] else "fisher_true",
        model_name=model_name,
        max_plots=10,
        save_dir=plot_dir,
    )
    print()
    print(f"Fisher Plot Time : {datetime.now() - start}")
    print(f"{datetime.now()}")


def init_predict(ms_params, model, args, subkey, init_params):

    pred = Predictive(
        model=model,
        posterior_samples=tree_map(lambda x: x[None, :], init_params),
        batch_ndims=1,
    )
    init_pred = pred(subkey, args=args)
    rchi2 = reduced_chi2(init_pred["vis_obs"][0], ms_params["vis_obs"].T, ms_params["noise"])
    print()
    print(f"Reduced Chi^2 @ init params : {rchi2}")

    return init_pred


def plot_init(ms_params, init_pred, args, model_name, plot_dir):

    start = datetime.now()
    print()
    print("Plotting Initial Parameters")
    plot_predictions(
        times=ms_params["times"],
        pred=init_pred,
        args=args,
        type="init",
        model_name=model_name,
        max_plots=10,
        save_dir=plot_dir,
    )
    print()
    print(f"Initial Plot Time : {datetime.now() - start}")
    print(f"{datetime.now()}")


def plot_truth(zarr_path, ms_params, args, model, model_name, subkey, true_params, gp_params, inv_scaling, plot_dir):
    
    start = datetime.now()
    print()
    print("Plotting True Parameters")

    xds = xr.open_zarr(zarr_path)
    rfi_A_true = jnp.transpose(xds.rfi_tle_sat_A.data[:,:,:,0].compute(), axes=(0,2,1))
    rfi_resample = resampling_kernel(
        gp_params["rfi_times"], 
        int_sample_times(ms_params["times"], xds.n_int_samples).compute(), 
        gp_params["rfi_var"], 
        gp_params["rfi_l"], 
        1e-8
        )
    rfi_amp = true_params["rfi_r_induce"] + 1.0j*true_params["rfi_i_induce"]
    rfi_A_pred = vmap(lambda x, y: x @ y.T, in_axes=(0, None))(rfi_amp, rfi_resample)

    true_params_base = inv_transform(true_params, args, inv_scaling)
    pred = Predictive(
    model=model,
    posterior_samples=tree_map(lambda x: x[None, :], true_params_base),
    batch_ndims=1,
    )
    true_pred = pred(subkey, args=args)
    # true_pred keys are ['ast_vis', 'gains', 'rfi_vis', 'rmse_ast', 'rmse_gains', 'rmse_rfi', 'vis_obs']
    print(f"RMSE Gains      : {jnp.mean(true_pred["rmse_gains"]):.5f}")
    print(f"RMSE RFI        : {jnp.mean(true_pred["rmse_rfi"]):.5f}")
    print(f"RMSE RFI signal : {jnp.sqrt(jnp.mean(jnp.abs(rfi_A_true - rfi_A_pred)**2)):.5f}")
    print(f"RMSE AST        : {jnp.mean(true_pred["rmse_ast"]):.5f}")

    rchi2 = reduced_chi2(true_pred["vis_obs"][0], ms_params["vis_obs"].T, ms_params["noise"])
    print()
    print(f"Reduced Chi^2 @ true params : {rchi2}")

    plot_predictions(
        times=ms_params["times"],
        pred=true_pred,
        args=args,
        type="true",
        model_name=model_name,
        max_plots=10,
        save_dir=plot_dir,
    )
    print()
    print(f"True Plot Time : {datetime.now() - start}")
    print(f"{datetime.now()}")


def plot_prior(config, ms_params, model, model_name, args, subkey, plot_dir):

    start = datetime.now()
    n_prior = config["plots"]["prior_samples"]
    print()
    print(f"Plotting {n_prior:.0f} Prior Parameter Samples")
    pred = Predictive(model, num_samples=n_prior)
    prior_pred = pred(subkey, args=args)
    print("Prior Samples Drawn")
    plot_predictions(
        times=ms_params["times"],
        pred=prior_pred,
        args=args,
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