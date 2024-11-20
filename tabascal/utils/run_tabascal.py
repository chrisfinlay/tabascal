from datetime import datetime

import shutil
import os
import sys
import yaml

from frozendict import frozendict

# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"  # add this
# os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import jax
from jax import random
import jax.profiler
import jax.numpy as jnp

# jax.config.update("jax_platform_name", "cpu")

import numpy as np

from tabascal.utils.config import Tee, load_config, yaml_load

from tab_opt.gp import (
    kernel,
    resampling_kernel,
    cholesky,
    find_closest_factor_greater_than,
)
from tab_opt.models import (
    fixed_orbit_rfi_fft_standard,
    fixed_orbit_rfi_full_fft_standard_model,
    fixed_orbit_rfi_full_fft_standard_model_otf,
    fixed_orbit_rfi_full_fft_standard_model_otf_fft,
)

from tabascal.utils.tab_tools import (
    read_ms,
    get_tles,
    estimate_sampling,
    get_rfi_phase,
    get_gp_params,
    calculate_estimates,
    get_truth_conditional,
    calculate_true_values,
    get_prior_means,
    pow_spec_sqrt,
    save_memory,
    inv_transform,
    get_init_params,
    init_predict,
    plot_init,
    plot_truth,
    plot_prior,
    run_mcmc,
    run_opt,
    run_fisher,
    check_antenna_and_satellite_positions,
)


def tabascal_subtraction(
    config: dict,
    sim_dir: str,
    ms_path: str = None,
    spacetrack_path: str = None,
    norad_ids: list = [],
):

    log = open("log_tab.txt", "w")
    backup = sys.stdout
    sys.stdout = Tee(sys.stdout, log)

    print()
    start_time = datetime.now()
    print(f"Start Time : {start_time}")

    key, subkey = random.split(random.PRNGKey(1))

    mem_i = 0

    ### Define Model
    # vis_model = fixed_orbit_rfi_full_fft_standard_model
    # vis_model = fixed_orbit_rfi_full_fft_standard_model_otf # RFI resampling is performed On-The-Fly
    vis_model = fixed_orbit_rfi_full_fft_standard_model_otf_fft  # RFI resampling is performed On-The-Fly with FFT - Not working yet

    def model(args, v_obs=None):
        return fixed_orbit_rfi_fft_standard(args, vis_model, v_obs)

    model_name = f"fixed_orbit_rfi"
    print(f"Model : {model_name}")
    results_name = f"fixed_orbit_rfi"

    if config["data"]["sim_dir"] is None:
        config["data"]["sim_dir"] = os.path.abspath(sim_dir)
    else:
        sim_dir = os.path.abspath(config["data"]["sim_dir"])
        config["data"]["sim_dir"] = sim_dir

    config["model"] = {"name": model_name, "func": vis_model.__name__}

    if sim_dir[-1] == "/":
        sim_dir = sim_dir[:-1]
    f_name = os.path.split(sim_dir)[1]

    print()
    print(f_name)
    print()

    zarr_path = os.path.join(sim_dir, f"{f_name}.zarr")
    config["data"]["zarr_path"] = zarr_path

    if not ms_path:
        ms_path = os.path.join(sim_dir, f"{f_name}.ms")
    else:
        ms_path = os.path.abspath(ms_path)

    config["data"]["ms_path"] = ms_path

    plot_dir = os.path.join(sim_dir, "plots")
    results_dir = os.path.join(sim_dir, "results")
    mem_dir = os.path.join(sim_dir, "memory_profiles")

    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(mem_dir, exist_ok=True)

    map_path = os.path.join(results_dir, f"map_results_{results_name}.zarr")
    fisher_path = os.path.join(results_dir, f"fisher_results_{results_name}.zarr")

    #####################################################
    # Calculate parameters from MS file
    #####################################################

    ms_params = read_ms(ms_path, config["data"]["freq"], config["data"]["corr"])
    n_rfi, norad_ids, tles = get_tles(config, ms_params, norad_ids, spacetrack_path)

    config["satellites"]["norad_ids"] = norad_ids
    config["satellites"]["norad_ids_path"] = None

    ######################################################
    # Check satellite and antenna positions to simulation
    ######################################################
    # if get_truth_conditional(config) and len(tles) > 0:
    #     check_antenna_and_satellite_positions(config, ms_params, tles)
    ######################################################
    #######################
    # Check the required sampling rate of the RFI by checking the
    # fringe frequency at a low sampling rate. Every minute is enough.
    # Once required sampling rate is determined calculate rfi_phase and times_fine
    #######################
    n_int_samples = estimate_sampling(config, ms_params, n_rfi, norad_ids, tles)

    rfi_phase, times_fine = get_rfi_phase(ms_params, norad_ids, tles, n_int_samples)
    gp_params = get_gp_params(config, ms_params)

    if config["model"]["func"] == "fixed_orbit_rfi_full_fft_standard_model_otf_fft":
        gp_params["n_rfi_times"] = gp_params["n_rfi_times"] - 1
        gp_params["rfi_times"] = gp_params["rfi_times"][:-1]

    n_g_params = (2 * ms_params["n_ant"] - 1) * gp_params["n_g_times"]
    n_rfi_params = 2 * ms_params["n_ant"] * gp_params["n_rfi_times"]
    n_ast_params = 2 * ms_params["n_time"] * ms_params["n_bl"]
    n_data = 2 * ms_params["n_bl"] * ms_params["n_time"]

    print(f"Using {n_int_samples} samples per time step for RFI prediction.")
    print()
    print(f"Number of Antennas   : {ms_params['n_ant']: 4}")
    print(f"Number of Time Steps : {ms_params['n_time']: 4}")
    print()
    print("Number of parameters per antenna/baseline")
    print(f"Gains : {gp_params['n_g_times']: 4}")
    print(f"RFI   : {gp_params['n_rfi_times']: 4}")
    print(f"AST   : {ms_params['n_time']: 4}")
    print()
    print(f"Number of parameters : {n_g_params + n_rfi_params + n_ast_params}")
    print(f"Number of data points: {n_data}")

    ##############################################
    # Only include estimates derived from MS available data
    ##############################################
    # Astronomical and RFI estimates from observed data

    estimates = calculate_estimates(ms_params, n_rfi, gp_params["rfi_times"])

    #############################
    # Calculate True Parameter Values if required
    #############################

    if get_truth_conditional(config):
        true_params, truth_args = calculate_true_values(
            zarr_path, config, ms_params, gp_params, n_rfi, norad_ids
        )
    else:
        true_params = None

    # Define Prior Means based on config
    prior_means = get_prior_means(
        config, ms_params, estimates, true_params, n_rfi, gp_params
    )

    k_ast = jnp.fft.fftfreq(ms_params["n_time"], ms_params["int_time"])

    #######################################
    # Chunking of the baselines appears to not be needed. Only jax.checkpoint is needed to trade compute for memory
    #######################################
    # import jax

    # gpu_bytes = jax.devices()[0].memory_stats()["bytes_limit"]
    # needed_bytes = (
    #     n_rfi * n_int_samples * ms_params["n_time"] * ms_params["n_bl"] * 8 * 2 * 3
    # )
    # n_bl_chunk = int(
    #     find_closest_factor_greater_than(
    #         ms_params["n_bl"], int(4 * needed_bytes / gpu_bytes)
    #     )
    # )

    # print()
    # print(f"Total memory     : {gpu_bytes / (1024**3):.2f} GB")
    # print(f"Memory needed    : {needed_bytes / (1024**3):.2f} GB")
    # print(f"Number of chunks : {n_bl_chunk}")

    # Set Constant Parameters
    args = {
        "noise": ms_params["noise"] if ms_params["noise"] > 0 else 1.0,
        "vis_ast_true": jnp.nan
        * jnp.zeros((ms_params["n_bl"], ms_params["n_time"]), dtype=complex),
        "vis_rfi_true": jnp.nan
        * jnp.zeros((ms_params["n_bl"], ms_params["n_time"]), dtype=complex),
        "gains_true": jnp.nan
        * jnp.zeros((ms_params["n_ant"], ms_params["n_time"]), dtype=complex),
        "times": ms_params["times"],
        "times_fine": times_fine,
        "g_times": gp_params["g_times"],
        "rfi_times": gp_params["rfi_times"],
        "k_ast": k_ast,
        "rfi_var": gp_params["rfi_var"],
        "rfi_l": gp_params["rfi_l"],
        "n_int_samples": n_int_samples,
        "n_rfi_factor": ms_params["n_time"] * n_int_samples // gp_params["n_rfi_times"],
        "n_time": ms_params["n_time"],
        "n_ants": ms_params["n_ant"],
        "n_rfi": n_rfi,
        "n_bl": ms_params["n_bl"],
        # "n_bl_chunk": ms_params["n_bl"],
        # "n_bl_chunk": ms_params["n_ant"] // 2,
        # "n_bl_chunk": ms_params["n_ant"] // 4,
        # "n_bl_chunk": 2,
        "n_bl_chunk": 1,
        # "n_bl_chunk": n_bl_chunk,
        # "chunk_rfi": False,
        "a1": ms_params["a1"],
        "a2": ms_params["a2"],
        "bl": jnp.arange(ms_params["n_bl"]),
        "v_obs_ri": jnp.concatenate(
            [ms_params["vis_obs"].real, ms_params["vis_obs"].imag], axis=0
        ).T,
        ### Define Prior Parameters
        "mu_G_amp": prior_means["g_amp_prior_mean"],
        "mu_G_phase": prior_means["g_phase_prior_mean"],
        "mu_rfi_r": prior_means["rfi_prior_mean"].real,
        "mu_rfi_i": prior_means["rfi_prior_mean"].imag,
        "mu_ast_k_r": prior_means["ast_k_prior_mean"].real,
        "mu_ast_k_i": prior_means["ast_k_prior_mean"].imag,
        "L_G_amp": cholesky(
            gp_params["g_times"], gp_params["g_amp_var"], gp_params["g_l"], 1e-8
        ),
        "L_G_phase": cholesky(
            gp_params["g_times"], gp_params["g_phase_var"], gp_params["g_l"], 1e-8
        ),
        "L_RFI": cholesky(
            gp_params["rfi_times"], gp_params["rfi_var"], gp_params["rfi_l"], 1e-8
        ),
        "sigma_ast_k": jnp.array(
            [
                pow_spec_sqrt(k_ast, **config["ast"]["pow_spec"])
                for _ in range(ms_params["n_bl"])
            ]
        ),
        "resample_g_amp": resampling_kernel(
            gp_params["g_times"],
            ms_params["times"],
            gp_params["g_amp_var"],
            gp_params["g_l"],
            1e-8,
        ),
        "resample_g_phase": resampling_kernel(
            gp_params["g_times"],
            ms_params["times"],
            gp_params["g_phase_var"],
            gp_params["g_l"],
            1e-8,
        ),
        "rfi_phase": rfi_phase,
    }

    if config["model"]["func"] == "fixed_orbit_rfi_full_fft_standard_model":
        args.update(
            {
                "resample_rfi": resampling_kernel(
                    gp_params["rfi_times"],
                    times_fine,
                    gp_params["rfi_var"],
                    gp_params["rfi_l"],
                    1e-8,
                ),
            }
        )

    if get_truth_conditional(config):
        args.update(truth_args)

    ##############################################
    # Initial parameters for optimization
    ##############################################
    init_params = get_init_params(
        config, ms_params, prior_means, estimates, true_params
    )

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

    mem_i = save_memory(mem_dir, mem_i)

    ### Check and Plot Model at init params
    init_params_base = inv_transform(init_params, args, inv_scaling)

    key, subkey = random.split(key)
    init_pred = init_predict(ms_params, model, args, subkey, init_params_base)

    if config["plots"]["init"]:
        plot_init(ms_params, init_pred, args, model_name, plot_dir)

    ### Check and Plot Model at true parameters
    if config["plots"]["truth"]:
        key, subkey = random.split(key)
        plot_truth(
            zarr_path,
            ms_params,
            args,
            model,
            model_name,
            subkey,
            true_params,
            gp_params,
            inv_scaling,
            plot_dir,
        )

    mem_i = save_memory(mem_dir, mem_i)

    ### Check and Plot Model at prior parameters
    key, subkey = random.split(key)
    if config["plots"]["prior"]:
        plot_prior(config, ms_params, model, model_name, args, subkey, plot_dir)

    ### Run MCMC Inference
    key, *subkeys = random.split(key, 3)
    if config["inference"]["mcmc"]:
        mcmc_pred = run_mcmc(
            ms_params, model, model_name, subkeys, args, init_params_base, plot_dir
        )

    mem_i = save_memory(mem_dir, mem_i)

    ### Run Optimization
    key, *subkeys = random.split(key, 3)
    if config["inference"]["opt"]:
        vi_params, rchi2 = run_opt(
            config,
            ms_params,
            model,
            model_name,
            args,
            subkeys,
            init_params_base,
            plot_dir,
            ms_path,
            map_path,
        )

    mem_i = save_memory(mem_dir, mem_i)

    ### Run Fisher Covariance Prediction
    key, *subkeys = random.split(key, 3)
    if config["inference"]["fisher"] and rchi2 < 1.1:
        run_fisher(
            config,
            ms_params,
            model,
            model_name,
            subkeys,
            vis_model,
            args,
            vi_params,
            init_params_base,
            plot_dir,
            fisher_path,
        )

    print()
    end_time = datetime.now()
    print(f"End Time  : {end_time}")
    print(f"Total Time : {end_time - start_time}")

    mem_i = save_memory(mem_dir, mem_i)

    log.close()
    shutil.copy("log_tab.txt", sim_dir)
    os.remove("log_tab.txt")
    sys.stdout = backup

    with open(os.path.join(sim_dir, "tab_config.yaml"), "w") as fp:
        yaml.dump(config, fp)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Apply tabascal to a simulation.")
    parser.add_argument(
        "-c", "--config", required=True, help="Path to the config file."
    )
    parser.add_argument(
        "-s", "--sim_dir", help="Path to the directory of the simulation."
    )
    parser.add_argument("-m", "--ms_path", help="Path to Measurement Set.")
    parser.add_argument(
        "-np", "--norad_path", help="Path to text file containing NORAD IDs to include."
    )
    parser.add_argument(
        "-st", "--spacetrack", help="Path to Space-Track login details."
    )
    args = parser.parse_args()
    sim_dir = args.sim_dir
    conf_path = args.config
    spacetrack_path = args.spacetrack
    norad_path = args.norad_path
    if sim_dir:
        norad_path = os.path.join(sim_dir, "input_data/norad_ids.yaml")

    if norad_path:
        norad_ids = [int(x) for x in np.loadtxt(norad_path)]
    else:
        norad_ids = []

    config = load_config(conf_path, config_type="tab")

    config_st_path = config["satellites"]["spacetrack_path"]
    if spacetrack_path:
        config["satellites"]["spacetrack_path"] = os.path.abspath(spacetrack_path)
    elif config_st_path:
        config_st_path = os.path.abspath(config_st_path)
        config["satellites"]["spacetrack_path"] = config_st_path
        spacetrack_path = config_st_path

    tabascal_subtraction(config, sim_dir, args.ms_path, spacetrack_path, norad_ids)


if __name__ == "__main__":
    main()
