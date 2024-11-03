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
from tabascal.jax.coordinates import calculate_sat_corr_time, orbit
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

    (
        N_int_samples,
        N_ant,
        N_bl,
        a1,
        a2,
        times,
        times_fine,
        bl_uvw,
        ants_uvw,
        ants_xyz,
        vis_ast,
        vis_rfi,
        vis_obs,
        vis_cal,
        noise,
        noise_data,
        int_time,
        freqs,
        gains_ants,
        rfi_A_app,
        rfi_orbit,
    ) = extract_data(zarr_path, sampling=config["data"]["sampling"])

    N_rfi = len(rfi_orbit)
    N_time = len(times)

    ################
    del bl_uvw
    del vis_cal
    del noise_data
    #################

    print()
    print(f"Number of Antennas   : {N_ant: 4}")
    print(f"Number of Time Steps : {N_time: 4}")

    # Square root of the power spectrum in the time axis for the astronomical visibilities
    @jit
    def pow_spec(k, P0=1e3, k0=1e-3, gamma=1.0):
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
    rfi_l = config["rfi"]["corr_time"]

    ### Gain Sampling Times
    g_times = get_times(times, g_l)
    N_g_times = len(g_times)

    ### RFI Sampling Times
    rfi_times = get_times(times, rfi_l)
    N_rfi_times = len(rfi_times)

    print()
    print("Number of parameters per antenna/baseline")
    print(f"Gains : {N_g_times: 4}")
    print(f"RFI   : {N_rfi_times: 4}")
    print(f"AST   : {N_time: 4}")
    print()
    print(
        f"Number of parameters : {((2 * N_ant - 1) * N_g_times) + (2 * N_time * N_bl) + (2 * N_ant * N_rfi_times)}"
    )
    print(f"Number of data points: {2* N_bl * N_time}")

    # To be replaced with TLE code for real data
    rfi_phase = jnp.array(
        [
            get_rfi_phase(times_fine, orbit, ants_uvw, ants_xyz, freqs).T
            for orbit in rfi_orbit
        ]
    )

    ##############################################
    # Only include estimates derived from MS available data
    ##############################################
    # Astronomical and RFI estimates from observed data

    vis_ast_est = jnp.mean(vis_obs.T, axis=1, keepdims=True) * jnp.ones((N_bl, N_time))
    ast_k_est = jnp.fft.fft(vis_ast_est, axis=1)
    k_ast = jnp.fft.fftfreq(N_time, int_time)

    rfi_induce_est = (
        jnp.interp(rfi_times, times, jnp.sqrt(jnp.max(jnp.abs(vis_obs - vis_ast_est.T), axis=1)))[
            None, None, :
        ] # shape is now (1, 1, N_rfi_times)
        * jnp.ones((N_rfi, N_ant, N_rfi_times))
        / N_rfi
    )

    # Stack observed data into real-valued array
    v_obs_ri = jnp.concatenate([vis_obs.real, vis_obs.imag], axis=0).T

    # Set Constant Parameters
    args = {
        "noise": noise if noise > 0 else 0.2,
        "vis_ast_true": jnp.nan * jnp.zeros((N_bl, N_time), dtype=complex), 
        "vis_rfi_true": jnp.nan * jnp.zeros((N_bl, N_time), dtype=complex),
        "gains_true": jnp.nan * jnp.zeros((N_ant, N_time), dtype=complex),
        "times": times,
        "times_fine": times_fine,
        "g_times": g_times,
        "N_time": N_time,
        "N_ants": N_ant,
        "N_bl": N_bl,
        "a1": a1,
        "a2": a2,
        "bl": jnp.arange(N_bl),
        "n_int": int(N_int_samples),
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

    # Calculate True Values
    if jnp.any(truth_cond) :
        ### Define True Parameters
        gains_true = gains_ants
        vis_ast_true = vis_ast
        vis_rfi_true = vis_rfi

        xds = xr.open_zarr(zarr_path)

        corr_time_params = {
            "sat_xyz": orbit(times, *rfi_orbit[0]),
            "ants_xyz": ants_xyz[N_int_samples//2::N_int_samples],
            "orbit_el": rfi_orbit[0,0],
            "lat": xds.tel_latitude,
            "dish_d": xds.dish_diameter,
            "freqs": freqs,
        }
        l = calculate_sat_corr_time(**corr_time_params)

        print()
        print(f"Minimum expected RFI correlation time : {l:.0f} s ({rfi_l:.0f} s used)")
        print()
        print(f"Mean RFI Amp. : {jnp.mean(jnp.abs(vis_rfi_true)):.1f} Jy")
        print(f"Mean AST Amp. : {jnp.mean(jnp.abs(vis_ast_true)):.1f} Jy")

        ast_k = jnp.fft.fft(vis_ast_true, axis=0).T

        ast_k_mean = jnp.fft.fft(
            vis_ast_true.mean(axis=0)[:, None] * jnp.ones((N_bl, N_time)), axis=1
        )

        gains_induce = vmap(jnp.interp, in_axes=(None, None, 1))(
            g_times, times, gains_ants
        )
        rfi_induce = jnp.array(
            [
                vmap(jnp.interp, in_axes=(None, None, 1))(
                    rfi_times, times_fine, rfi_A_app[i]
                )
                for i in range(N_rfi)
            ]
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
            "vis_ast_true": vis_ast_true.T,
            "vis_rfi_true": vis_rfi_true.T,
            "gains_true": gains_true.T,
        })

    # Set Gain Prior Mean
    if config["gains"]["amp_mean"] == "truth":
        g_amp_prior_mean = true_params["g_amp_induce"]
    elif isinstance(config["gains"]["amp_mean"], numbers.Number):
        g_amp_prior_mean = config["gains"]["amp_mean"] * jnp.ones((N_ant, N_g_times))
    else:
        ValueError("gains: amp_mean: must be a number or 'truth'.")

    if config["gains"]["phase_mean"] == "truth":
        g_phase_prior_mean = true_params["g_phase_induce"]
    elif isinstance(config["gains"]["phase_mean"], numbers.Number):
        g_phase_prior_mean = jnp.deg2rad(config["gains"]["phase_mean"]) * jnp.ones((N_ant-1, N_g_times))
    else:
        ValueError("gains: phase_mean: must be a number or 'truth'.")

    # Set Astronomical Prior Mean
    if config["ast"]["mean"]==0:
        ast_k_prior_mean = jnp.zeros((N_bl, N_time), dtype=complex)
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
        rfi_prior_mean = jnp.zeros((N_rfi, N_ant, N_rfi_times), dtype=complex)
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
            "sigma_ast_k": jnp.array([pow_spec(k_ast, **config["ast"]["pow_spec"]) for _ in range(N_bl)]),
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

    ################
    del gains_ants
    del rfi_A_app
    ################

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

    print()
    print("Copying tabascal results to MS file in TAB_DATA column")
    subprocess.run(f"tab2MS -m {ms_path} -z {map_path}", shell=True, executable="/bin/bash")    

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