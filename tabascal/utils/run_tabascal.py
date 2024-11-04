from datetime import datetime

import shutil
import os
import sys
import yaml
import subprocess
import numbers

# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"  # add this
# os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import jax

# jax.config.update("jax_platform_name", "cpu")

import jax.profiler

import jax.numpy as jnp
from jax import random, vmap, jit
from jax.tree_util import tree_map

from numpyro.infer import MCMC, NUTS, Predictive

import matplotlib.pyplot as plt

from tabascal.utils.yaml import Tee, load_config
from tabascal.utils.tle import get_satellite_positions, get_tles_by_id
from tabascal.jax.coordinates import calculate_sat_corr_time, orbit, itrf_to_uvw, itrf_to_xyz, calculate_fringe_frequency
from tabascal.jax.interferometry import int_sample_times

from astropy.time import Time
import numpy as np

from tab_opt.data import extract_data
from tab_opt.opt import run_svi, svi_predict, f_model_flat, flatten_obs, post_samples
from tab_opt.gp import (
    get_times,
    kernel,
    resampling_kernel,
)
from tab_opt.plot import plot_predictions
from tab_opt.vis import get_rfi_phase
from tab_opt.models import (
    fixed_orbit_rfi_fft_standard,
    fixed_orbit_rfi_full_fft_standard_model,
)
from tab_opt.transform import affine_transform_full_inv, affine_transform_diag_inv

import xarray as xr
from daskms import xds_from_ms, xds_from_table

def reduced_chi2(pred, true, noise):
    rchi2 = ((jnp.abs(pred - true) / noise) ** 2).sum() / (2 * true.size)
    return rchi2


def write_xds(vi_pred, times, file_path, overwrite=True):
    import dask.array as da
    import xarray as xr

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


def tabascal_subtraction(conf_path: str, sim_dir: str):

    log = open('log_tab.txt', 'w')
    backup = sys.stdout
    sys.stdout = Tee(sys.stdout, log)

    print()
    start_time = datetime.now()
    print(f"Start Time : {start_time}")

    key, subkey = random.split(random.PRNGKey(1))

    mem_i = 0

    ### Define Model
    vis_model = fixed_orbit_rfi_full_fft_standard_model


    def model(args, v_obs=None):
        return fixed_orbit_rfi_fft_standard(args, vis_model, v_obs)


    model_name = f"fixed_orbit_rfi"
    print(f"Model : {model_name}")
    results_name = f"fixed_orbit_rfi"

    config = load_config(conf_path, config_type="tab")

    if config["data"]["sim_dir"] is None:
        config["data"]["sim_dir"] = os.path.abspath(sim_dir)
    else:
        sim_dir = os.path.abspath(config["data"]["sim_dir"])
        config["data"]["sim_dir"] = sim_dir

    config["model"] = {"name": model_name, "func": vis_model.__name__} 


    if sim_dir[-1]=="/":
        sim_dir = sim_dir[:-1]
    f_name = os.path.split(sim_dir)[1]

    print()
    print(f_name)
    print()

    zarr_path = os.path.join(sim_dir, f"{f_name}.zarr")
    ms_path = os.path.join(sim_dir, f"{f_name}.ms")

    plot_dir = os.path.join(sim_dir, "plots")
    results_dir = os.path.join(sim_dir, "results")
    mem_dir = os.path.join(sim_dir, "memory_profiles")

    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(mem_dir, exist_ok=True)

    map_path = os.path.join(results_dir, f"map_results_{results_name}.zarr")
    fisher_path = os.path.join(results_dir, f"fisher_results_{results_name}.zarr")

    ####################################################

    n_rfi = len(config["satellites"]["norad_ids"]) + len(config["satellites"]["sat_ids"])
    if len(config["satellites"]["sat_ids"])>0:
        import pandas as pd
        ole_df = pd.read_csv(config["satellites"]["ole_path"])
        oles = ole_df[ole_df["sat_id"].isin(config["satellites"]["sat_ids"])][["elevation", "inclination", "lon_asc_node", "periapsis"]]
        rfi_orbit = jnp.atleast_2d(oles.values)

    #####################################################
    # Calculate parameters from MS file
    #####################################################
    
    xds = xds_from_ms(ms_path)[0]
    xds_ant = xds_from_table(ms_path+"::ANTENNA")[0]
    xds_spec = xds_from_table(ms_path+"::SPECTRAL_WINDOW")[0]
    xds_src = xds_from_table(ms_path+"::SOURCE")[0]

    ants_itrf = xds_ant.POSITION.data.compute()
    freqs = xds_spec.CHAN_FREQ.data[0].compute()
    ra, dec = jnp.rad2deg(xds_src.DIRECTION.data[0].compute())

    n_ant = ants_itrf.shape[0]
    n_time = len(jnp.unique(xds.TIME.data.compute()))
    n_bl = xds.DATA.data.shape[0] // n_time
    n_freq, n_corr = xds.DATA.data.shape[1:]

    a1 = xds.ANTENNA1.data.reshape(n_time, n_bl)[0,:].compute()
    a2 = xds.ANTENNA2.data.reshape(n_time, n_bl)[0,:].compute()

    vis_obs = xds.DATA.data.reshape(n_time, n_bl, n_freq, n_corr).compute()[:,:,0,0]
    noise = xds.SIGMA.data.mean().compute()

    times_jd = xds.TIME.data.reshape(n_time, n_bl)[:,0].compute()
    times = times_jd * 24 * 3600 # Convert Julian date in days to seconds
    int_time = jnp.diff(times)[0]

    if len(config["satellites"]["norad_ids"])>0:
        tles_df = get_tles_by_id(
            config["spacetrack"]["username"], 
            config["spacetrack"]["password"], 
            config["satellites"]["norad_ids"],
            jnp.mean(times_jd),
            tle_dir=config["satellites"]["tle_dir"],
            )
        tles = np.atleast_2d(tles_df[["TLE_LINE1", "TLE_LINE2"]].values)
        

    #######################
    # Check the required sampling rate of the RFI by checking the 
    # fringe frequency at a low sampling rate. Every minute is enough.
    #######################

    jd_minute = 1 / (24*60)
    times_jd_coarse =  jnp.arange(times_jd[0], times_jd[-1]+jd_minute, jd_minute)
    times_coarse = times_jd_coarse * 24 * 3600
    # times_coarse = Time(times_jd_coarse, format="jd").sidereal_time("mean", "greenwich").hour*3600 # Convert hours to seconds

    if len(config["satellites"]["norad_ids"])>0:
        rfi_xyz = get_satellite_positions(tles, np.array(times_jd_coarse))
    if len(config["satellites"]["sat_ids"])>0:
        rfi_xyz = jnp.array([orbit(times_coarse, *rfi_orb) for rfi_orb in rfi_orbit])

    gsa = Time(times_jd_coarse, format="jd").sidereal_time("mean", "greenwich").hour*15 # Convert hours to degrees
    gh0 = (gsa - ra) % 360
    ants_u = itrf_to_uvw(ants_itrf, gh0, dec)[:,:,0] # We want the uvw-coordinates at the coarse sampling rate for the fringe frequency prediction

    fringe_params = [{
        "times": times_coarse,
        "freq": freqs.max(),
        "rfi_xyz": rfi_xyz[i],
        "ants_itrf": ants_itrf,
        "ants_u": ants_u,
        "dec": dec,
    } for i in range(n_rfi)]

    fringe_freq = jnp.array([calculate_fringe_frequency(**f_params) for f_params in fringe_params])

    sample_freq = jnp.pi * jnp.max(jnp.abs(fringe_freq)) * jnp.sqrt(jnp.max(jnp.abs(vis_obs)) / (6 * noise))
    n_int_samples = jnp.ceil(int_time * sample_freq).astype(int)

    print(f"Using {n_int_samples} samples per time step for RFI prediction.")
    
    #####################
    # Now the required sampling rate has been determined we can create the times_fine array
    #####################

    times_fine = int_sample_times(times, n_int_samples)
    times_jd_fine = int_sample_times(times_jd, n_int_samples)

    ra, dec = jnp.rad2deg(xds_src.DIRECTION.data[0].compute())
    gsa = Time(times_jd_fine, format="jd").sidereal_time("mean", "greenwich").hour*15 # Convert hours to degrees
    gh0 = (gsa - ra) % 360

    ants_uvw = itrf_to_uvw(ants_itrf, gh0, dec)#[:,:,2] # We need the uvw-coordinates at the fine sampling rate for the RFI
    ants_xyz = itrf_to_xyz(ants_itrf, gsa)

    # To be replaced with TLE code for real data
    # phi_i = -2pi (|r_rfi_s - r_ant_i| + w_ant_i) / lambda

    def get_rfi_phase_from_pos(rfi_xyz, ants_w, ants_xyz, freqs):

        c = 299792458.0
        lam = c / freqs
        c_dist = jnp.linalg.norm(rfi_xyz[:,:,None,:] - ants_xyz[None,:,:,:], axis=-1) + ants_w[None,:,:]
        phase = -2.0 * jnp.pi * c_dist[:,:,:,None] / lam[None,None,None,:]

        return phase
    
    if len(config["satellites"]["norad_ids"])>0:
        rfi_xyz = get_satellite_positions(tles, np.array(times_jd_fine))
        rfi_phase = jnp.transpose(get_rfi_phase_from_pos(rfi_xyz, ants_uvw[...,-1], ants_xyz, freqs)[...,0], axes=(0,2,1))

    if len(config["satellites"]["sat_ids"])>0:
        rfi_phase = jnp.array(
            [
                get_rfi_phase(times_fine, orbit, ants_uvw, ants_xyz, freqs).T
                for orbit in rfi_orbit
            ]
        )

    print()
    print(f"Number of Antennas   : {n_ant: 4}")
    print(f"Number of Time Steps : {n_time: 4}")

    # Square root of the power spectrum in the time axis for the astronomical visibilities
    @jit
    def pow_spec_sqrt(k, P0=1e3, k0=1e-3, gamma=1.0):
        k_ = (k / k0) ** 2
        return P0 * 0.5 * (jnp.exp(-0.5 * k_) + 1.0 / ((1.0 + k_) ** (gamma / 2)))

    ### GP Parameters

    # Gain GP Parameters
    g_amp_var = (config["gains"]["amp_std"] / 100)**2 # convert % to decimal
    g_phase_var = jnp.deg2rad(config["gains"]["phase_std"])**2 # convert degrees to radians
    g_l = 60.0 * config["gains"]["corr_time"] # convert minutes to seconds

    # RFI GP Parameters
    if config["rfi"]["var"] is not None:
        rfi_var = config["rfi"]["var"]
    else:
        rfi_var = jnp.max(jnp.abs(vis_obs))
    # rfi_l can be calculated based on RFI positions. Check True value definitions
    rfi_l = config["rfi"]["corr_time"]

    ### Gain Sampling Times
    g_times = get_times(times, g_l)
    n_g_times = len(g_times)

    ### RFI Sampling Times
    rfi_times = get_times(times, rfi_l)
    n_rfi_times = len(rfi_times)

    print()
    print("Number of parameters per antenna/baseline")
    print(f"Gains : {n_g_times: 4}")
    print(f"RFI   : {n_rfi_times: 4}")
    print(f"AST   : {n_time: 4}")
    print()
    print(
        f"Number of parameters : {((2 * n_ant - 1) * n_g_times) + (2 * n_time * n_bl) + (2 * n_ant * n_rfi_times)}"
    )
    print(f"Number of data points: {2* n_bl * n_time}")

    ##############################################
    # Only include estimates derived from MS available data
    ##############################################
    # Astronomical and RFI estimates from observed data

    vis_ast_est = jnp.mean(vis_obs.T, axis=1, keepdims=True) * jnp.ones((n_bl, n_time))
    ast_k_est = jnp.fft.fft(vis_ast_est, axis=1)
    k_ast = jnp.fft.fftfreq(n_time, int_time)

    rfi_induce_est = (
        jnp.interp(rfi_times, times, jnp.sqrt(jnp.max(jnp.abs(vis_obs - vis_ast_est.T), axis=1)))[
            None, None, :
        ] # shape is now (1, 1, n_rfi_times)
        * jnp.ones((n_rfi, n_ant, n_rfi_times))
        / n_rfi
    )

    # Stack observed data into real-valued array
    v_obs_ri = jnp.concatenate([vis_obs.real, vis_obs.imag], axis=0).T

    # Set Constant Parameters
    args = {
        "noise": noise if noise > 0 else 0.2,
        "vis_ast_true": jnp.nan * jnp.zeros((n_bl, n_time), dtype=complex), 
        "vis_rfi_true": jnp.nan * jnp.zeros((n_bl, n_time), dtype=complex),
        "gains_true": jnp.nan * jnp.zeros((n_ant, n_time), dtype=complex),
        "times": times,
        "times_fine": times_fine,
        "g_times": g_times,
        "n_time": n_time,
        "n_ants": n_ant,
        "n_bl": n_bl,
        "a1": a1,
        "a2": a2,
        "bl": jnp.arange(n_bl),
    }

    # Condition array if anything needs access to the truth
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
    #############################
    # Calculate True Parameter Values
    #############################
    if jnp.any(truth_cond) :
        xds = xr.open_zarr(zarr_path)

        vis_ast = xds.vis_ast.data[:,:,0].compute()
        vis_rfi = xds.vis_rfi.data[:,:,0].compute()
        gains_ants = xds.gains_ants[:,:,0].data.compute()

        if len(config["satellites"]["sat_ids"])>0:
            corr_time_params = [{
                "sat_xyz": xds.rfi_sat_xyz.data[i,xds.time_idx].compute(),
                "ants_xyz": xds.ants_xyz.data[xds.time_idx].compute(),
                "orbit_el": xds.rfi_sat_orbit.data[i,0].compute(),
                "lat": xds.tel_latitude,
                "dish_d": xds.dish_diameter,
                "freqs": freqs,
            } for i in range(n_rfi)]

            l = jnp.min(jnp.array([calculate_sat_corr_time(**corr_time_params[i]) for i in range(n_rfi)]))

            rfi_induce = jnp.array(
                [
                    vmap(jnp.interp, in_axes=(None, None, 1), out_axes=(0))(
                        rfi_times, xds.time_fine.data, xds.rfi_sat_A[:,:,:,0].data.compute()[i]
                    )
                    for i in range(n_rfi)
                ]
            )


        def get_orbit_elevation(rfi_xyz, latitude):
            from tabascal.jax.coordinates import earth_radius
            R_rfi = jnp.max(jnp.linalg.norm(rfi_xyz, axis=-1))
            R_e = earth_radius(latitude)
            return R_rfi - R_e

        if len(config["satellites"]["norad_ids"])>0:
            corr_time_params = [{
                "sat_xyz": xds.rfi_tle_sat_xyz.data[i,xds.time_idx].compute(),
                "ants_xyz": xds.ants_xyz.data[xds.time_idx].compute(),
                "orbit_el": get_orbit_elevation(xds.rfi_tle_sat_xyz.data[i,xds.time_idx].compute(), xds.tel_latitude),
                "lat": xds.tel_latitude,
                "dish_d": xds.dish_diameter,
                "freqs": freqs,
            } for i in range(n_rfi)]

            l = jnp.min(jnp.array([calculate_sat_corr_time(**corr_time_params[i]) for i in range(n_rfi)]))

            rfi_induce = jnp.array(
                [
                    vmap(jnp.interp, in_axes=(None, None, 1), out_axes=(0))(
                        rfi_times, xds.time_fine.data, xds.rfi_tle_sat_A[:,:,:,0].data.compute()[i]
                    )
                    for i in range(n_rfi)
                ]
            )


        print()
        print(f"Minimum expected RFI correlation time : {l:.0f} s ({rfi_l:.0f} s used)")
        print()
        print(f"Mean RFI Amp. : {jnp.mean(jnp.abs(vis_rfi)):.1f} Jy")
        print(f"Mean AST Amp. : {jnp.mean(jnp.abs(vis_ast)):.1f} Jy")

        ast_k = jnp.fft.fft(vis_ast, axis=0).T

        ast_k_mean = jnp.fft.fft(
            vis_ast.mean(axis=0)[:, None] * jnp.ones((n_bl, n_time)), axis=1
        )

        gains_induce = vmap(jnp.interp, in_axes=(None, None, 1), out_axes=(0))(
            g_times, times, gains_ants
        )
        
        true_params = {
            **{f"g_amp_induce": jnp.abs(gains_induce)},
            **{f"g_phase_induce": jnp.angle(gains_induce[:-1])},
            **{f"rfi_r_induce": rfi_induce.real},
            **{f"rfi_i_induce": rfi_induce.imag},
            **{"ast_k_r": ast_k.real},
            **{"ast_k_i": ast_k.imag},
        }

        args.update({
            "vis_ast_true": vis_ast.T,
            "vis_rfi_true": vis_rfi.T,
            "gains_true": gains_ants.T,
        })

        ################
        del vis_ast
        del vis_rfi
        del gains_ants
        ################

    ###################################
    # End of True Parameter Definition
    ###################################

    # Set Gain Prior Mean
    if config["gains"]["amp_mean"] == "truth":
        g_amp_prior_mean = true_params["g_amp_induce"]
    elif isinstance(config["gains"]["amp_mean"], numbers.Number):
        g_amp_prior_mean = config["gains"]["amp_mean"] * jnp.ones((n_ant, n_g_times))
    else:
        ValueError("gains: amp_mean: must be a number or 'truth'.")

    if config["gains"]["phase_mean"] == "truth":
        g_phase_prior_mean = true_params["g_phase_induce"]
    elif isinstance(config["gains"]["phase_mean"], numbers.Number):
        g_phase_prior_mean = jnp.deg2rad(config["gains"]["phase_mean"]) * jnp.ones((n_ant-1, n_g_times))
    else:
        ValueError("gains: phase_mean: must be a number or 'truth'.")

    # Set Astronomical Prior Mean
    if config["ast"]["mean"]==0:
        ast_k_prior_mean = jnp.zeros((n_bl, n_time), dtype=complex)
    elif config["ast"]["mean"]=="est":
        ast_k_prior_mean = ast_k_est
    elif config["ast"]["mean"]=="truth":
        ast_k_prior_mean = ast_k
    elif config["ast"]["mean"]=="truth_mean":
        ast_k_prior_mean = ast_k_mean
    else:
        ValueError("ast: mean: must be one of (est, prior, truth, truth_mean)")

    # Set RFI Prior Mean
    if config["rfi"]["mean"]==0:
        rfi_prior_mean = jnp.zeros((n_rfi, n_ant, n_rfi_times), dtype=complex)
    elif config["rfi"]["mean"]=="est":
        rfi_prior_mean = rfi_induce_est
    elif config["rfi"]["mean"]=="truth":
        rfi_prior_mean = rfi_induce
    else:
        ValueError("rfi: mean: must be one of (est, prior, truth)")

    ### Define Prior Parameters
    args.update(
        {
            "mu_G_amp": g_amp_prior_mean,
            "mu_G_phase": g_phase_prior_mean,
            "mu_rfi_r": rfi_prior_mean.real,
            "mu_rfi_i": rfi_prior_mean.imag,
            "mu_ast_k_r": ast_k_prior_mean.real,
            "mu_ast_k_i": ast_k_prior_mean.imag,
            "L_G_amp": jnp.linalg.cholesky(kernel(g_times, g_times, g_amp_var, g_l, 1e-8)),
            "L_G_phase": jnp.linalg.cholesky(
                kernel(g_times, g_times, g_phase_var, g_l, 1e-8)
            ),
            "sigma_ast_k": jnp.array([pow_spec_sqrt(k_ast, **config["ast"]["pow_spec"]) for _ in range(n_bl)]),
            "L_RFI": jnp.linalg.cholesky(kernel(rfi_times, rfi_times, rfi_var, rfi_l)),
            "resample_g_amp": resampling_kernel(g_times, times, g_amp_var, g_l, 1e-8),
            "resample_g_phase": resampling_kernel(g_times, times, g_phase_var, g_l, 1e-8),
            "resample_rfi": resampling_kernel(rfi_times, times_fine, rfi_var, rfi_l, 1e-8),
            "rfi_phase": rfi_phase,
        }
    )

    ##############################################
    # Initial parameters for optimization
    ##############################################

    # Set RFI parameter initialisation
    if config["rfi"]["init"] == "est":
        rfi_induce_init = rfi_induce_est    # Estimate from data
    elif config["rfi"]["init"] == "prior":
        rfi_induce_init = rfi_prior_mean    # Prior mean value
    elif config["rfi"]["init"] == "truth":
        rfi_induce_init = rfi_induce        # True value
    else:
        ValueError("rfi: init: must be one of (est, prior, truth)")

    # Set Astronomical parameter initialisation
    if config["ast"]["init"] == "est":
        ast_k_init = ast_k_est              # Estimate from data
    elif config["ast"]["init"] == "prior":
        ast_k_init = ast_k_prior_mean       # Prior mean value
    elif config["ast"]["init"] == "truth":
        ast_k_init = ast_k                  # True value
    elif config["ast"]["init"] == "truth_mean":
        ast_k_init = ast_k_mean           # Mean of true value
    else:
        ValueError("ast: init: must be one of (est, prior, truth, truth_mean)")

    init_params = {
        "g_amp_induce": g_amp_prior_mean,
        "g_phase_induce": g_phase_prior_mean,
        "ast_k_r": ast_k_init.real,
        "ast_k_i": ast_k_init.imag,
        "rfi_r_induce": rfi_induce_init.real,
        "rfi_i_induce": rfi_induce_init.imag,
    }

    inv_scaling = {
        "L_RFI": jnp.linalg.inv(args["L_RFI"]),
        "L_G_amp": jnp.linalg.inv(args["L_G_amp"]),
        "L_G_phase": jnp.linalg.inv(args["L_G_phase"]),
        "sigma_ast_k": 1.0 / args["sigma_ast_k"],
    }

    print()
    end_start = datetime.now()
    print(f"Startup Time : {end_start - start_time}")
    print(f"{end_start}")

    mem_i += 1
    jax.profiler.save_device_memory_profile(
        os.path.join(mem_dir, f"memory_{mem_i}.prof")
    )

    guides = {
        "map": "AutoDelta",
    }

    ### Check and Plot Model at init params
    init_params_base = inv_transform(init_params, args, inv_scaling)
    pred = Predictive(
        model=model,
        posterior_samples=tree_map(lambda x: x[None, :], init_params_base),
        batch_ndims=1,
    )
    key, subkey = random.split(key)
    init_pred = pred(subkey, args=args)
    rchi2 = reduced_chi2(init_pred["vis_obs"][0], vis_obs.T, noise)
    print()
    print(f"Reduced Chi^2 @ init: {rchi2}")

    ### Check and Plot Model at true parameters
    
    if config["plots"]["init"]:
        start = datetime.now()
        print()
        print("Plotting Initial Parameters")
        plot_predictions(
            times=times,
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

    if config["plots"]["truth"]:

        start = datetime.now()
        print()
        print("Plotting True Parameters")
        true_params_base = inv_transform(true_params, args, inv_scaling)
        pred = Predictive(
        model=model,
        posterior_samples=tree_map(lambda x: x[None, :], true_params_base),
        batch_ndims=1,
        )
        key, subkey = random.split(key)
        true_pred = pred(subkey, args=args)
        rchi2 = reduced_chi2(true_pred["vis_obs"][0], vis_obs.T, noise)
        print()
        print(f"Reduced Chi^2 @ true: {rchi2}")
        print()
        plot_predictions(
            times=times,
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
    
    

    mem_i += 1
    jax.profiler.save_device_memory_profile(
        os.path.join(mem_dir, f"memory_{mem_i}.prof")
    )

    ### Check and Plot Model at prior parameters
    key, subkey = random.split(key)
    if config["plots"]["prior"]:
        start = datetime.now()
        n_prior = config["plots"]["prior_samples"]
        print()
        print(f"Plotting {n_prior:.0f} Prior Parameter Samples")
        pred = Predictive(model, num_samples=n_prior)
        prior_pred = pred(subkey, args=args)
        print("Prior Samples Drawn")
        plot_predictions(
            times=times,
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
    

    ### Run Inference
    key, *subkeys = random.split(key, 3)
    if config["inference"]["mcmc"]:
        num_warmup = 500
        num_samples = 1000
        print(f"Running MCMC with {num_warmup:.0f} Warm Up Samples and for {num_samples:.0f} Samples")
        start = datetime.now()

        nuts_kernel = NUTS(model, dense_mass=False)  # [('g_phase_0', 'g_phase_1')])
        mcmc = MCMC(nuts_kernel, num_warmup=num_warmup, num_samples=num_samples)
        mcmc.run(
            subkeys[0],
            args=args,
            v_obs=v_obs_ri,
            extra_fields=("potential_energy",),
            init_params=init_params_base,
        )
        print()
        print(f"MCMC Run Time : {datetime.now() - start}")
        print(f"{datetime.now()}")
        start = datetime.now()

        pred = Predictive(model, posterior_samples=mcmc.get_samples())
        mcmc_pred = pred(subkeys[1], args=args)
        plot_predictions(
            times=times,
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

    mem_i += 1
    jax.profiler.save_device_memory_profile(
        os.path.join(mem_dir, f"memory_{mem_i}.prof")
    )

    key, *subkeys = random.split(key, 3)
    if config["inference"]["opt"]:
        print()
        print("Running Optimization ...")
        guide_family = guides[config["opt"]["guide"]]
        vi_results, vi_guide = run_svi(
            model=model,
            args=args,
            obs=v_obs_ri,
            max_iter=config["opt"]["max_iter"],
            guide_family=guide_family,
            init_params={
                **{k + "_auto_loc": v for k, v in init_params_base.items()},
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

        map_xds = write_xds(vi_pred, times, map_path)

        plot_predictions(
            times,
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

        rchi2 = reduced_chi2(vi_pred["vis_obs"][0], vis_obs.T, noise)
        print()
        print(f"Reduced Chi^2 @ opt: {rchi2}")
        print()

        plt.semilogy(vi_results.losses)
        plt.savefig(os.path.join(plot_dir, f"{model_name}_opt_loss.pdf"), format="pdf")

        print()
        print("Copying tabascal results to MS file in TAB_DATA column")
        subprocess.run(f"tab2MS -m {ms_path} -z {map_path}", shell=True, executable="/bin/bash") 
        
        del vi_pred
        del vi_results

    mem_i += 1
    jax.profiler.save_device_memory_profile(
        os.path.join(mem_dir, f"memory_{mem_i}.prof")
    )

    key, *subkeys = random.split(key, 3)
    if config["inference"]["fisher"] and rchi2 < 1.1:

        start = datetime.now()
        n_fisher = config["fisher"]["n_samples"]
        print(f"Calculating {n_fisher:.0f} Fisher Samples ...")

        f_model = lambda params, args: vis_model(params, args)[0]
        model_flat = lambda params: f_model_flat(f_model, params, args)

        post_mean = {k[:-9]: v for k, v in vi_params.items()} if config["inference"]["opt"] else init_params_base

        dtheta = post_samples(
            model_flat,
            post_mean,
            flatten_obs(vis_obs),
            noise,
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

        fisher_xds = write_xds(fisher_pred, times, fisher_path)

        plot_predictions(
            times=times,
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

    print()
    end_time = datetime.now()
    print(f"End Time  : {end_time}")
    print(f"Total Time : {end_time - start_time}")

    mem_i += 1
    jax.profiler.save_device_memory_profile(
        os.path.join(mem_dir, f"memory_{mem_i}.prof")
    )   

    log.close()
    shutil.copy("log_tab.txt", sim_dir)
    os.remove("log_tab.txt")
    sys.stdout = backup

    with open(os.path.join(sim_dir, "tab_config.yaml"), "w") as fp:
        yaml.dump(config, fp)

    
def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Apply tabascal to a simulation."
    )
    parser.add_argument(
        "-s", "--sim_dir", help="Path to the directory of the simulation."
    )
    parser.add_argument(
        "-c", "--config", required=True, help="Path to the config file."
    )
    args = parser.parse_args()
    sim_dir = args.sim_dir
    conf_path = args.config   

    tabascal_subtraction(conf_path, sim_dir) 

if __name__=="__main__":
    main()