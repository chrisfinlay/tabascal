import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike

import jax.numpy as jnp

from tabascal.dask.observation import Observation
from tabascal.jax.coordinates import alt_az_of_source
import os

plt.rcParams["font.size"] = 18


def time_units(times: ArrayLike) -> tuple:
    """Scale the time axis to hours, minutes or seconds depending on the total range.

    Parameters
    ----------
    times : ArrayLike
        Times to consider.

    Returns
    -------
    tuple
        Rescaled times array and the scale unit as a string.
    """

    time_range = times[-1] - times[0]
    times = times - times[0]
    if time_range > 3600:
        units = "hr"
        times = times / 3600
    elif time_range > 60:
        units = "min"
        times = times / 60
    else:
        units = "s"

    return times, units


def plot_angular_seps(obs: Observation, save_path: str) -> None:
    """Plot the angular separations between the RFI sources and pointing direction.

    Parameters
    ----------
    obs : Observation
        Observation object with RFI sources added
    save_path : str
        Path to where to save the plots.
    """

    times, scale = time_units(obs.times_fine)
    n_time = len(times)
    if n_time > 100:
        idx = np.arange(0, n_time, n_time // 100)
    else:
        idx = np.arange(n_time)
    plt.figure(figsize=(10, 7))
    if obs.n_rfi_satellite > 0:
        ang_seps = np.concatenate(obs.rfi_satellite_ang_sep, axis=0).mean(axis=-1).T
        plt.plot(times[idx], ang_seps[idx], label="Satellite")
    if obs.n_rfi_stationary > 0:
        ang_seps = np.concatenate(obs.rfi_stationary_ang_sep, axis=0).mean(axis=-1).T
        plt.plot(times[idx], ang_seps[idx], label="Stationary")
    if obs.n_rfi_tle_satellite > 0:
        ang_seps = np.concatenate(obs.rfi_tle_satellite_ang_sep, axis=0).mean(axis=-1)
        for ang_sep, n_id in zip(np.atleast_2d(ang_seps), obs.norad_ids):
            plt.plot(times[idx], ang_sep[idx], label=n_id.compute())
    plt.xlabel(f"Time [{scale}]")
    plt.ylabel("Angular Separation [deg]")
    plt.legend()
    plt.savefig(os.path.join(save_path, "AngularSeps.png"), format="png", dpi=200)


def plot_src_alt(obs: Observation, save_path: str) -> None:
    """Plot the target source altitude over the period of the observation.

    Parameters
    ----------
    obs : Observation
        Observation object.
    save_path : str
        Path to where to save the plot.
    """

    times, scale = time_units(obs.times)
    lsa = obs.lsa.compute()
    alt = alt_az_of_source(
        lsa[obs.t_idx], *[x.compute() for x in [obs.latitude, obs.ra, obs.dec]]
    )[:, 0]
    plt.figure(figsize=(10, 7))
    plt.plot(times, alt, ".-")
    plt.xlabel(f"Time [{scale}]")
    plt.ylabel("Source Altitude [deg]")
    plt.savefig(os.path.join(save_path, "SourceAltitude.png"), format="png", dpi=200)


def plot_uv(obs: Observation, save_path: str) -> None:
    """Plot the uv coverage of the telescope for the given observation.

    Parameters
    ----------
    obs : Observation
        Observation object.
    save_path : str
        Path to where to save the plot.
    """

    plt.figure(figsize=(10, 10))
    if obs.n_time_fine > 100:
        time_step = int(obs.n_time_fine / 100)
    else:
        time_step = 1
    u = obs.bl_uvw[::time_step, :, 0].compute().flatten()
    v = obs.bl_uvw[::time_step, :, 1].compute().flatten()
    max_U = np.max(np.sqrt(u**2 + v**2))
    exp = float(np.floor(np.log10(max_U)))
    mantissa = np.ceil(10 ** (np.log10(max_U) - exp))
    lim = mantissa * 10**exp
    plt.plot(u, v, "k.", ms=1, alpha=0.3)
    plt.plot(-u, -v, "k.", ms=1, alpha=0.3)
    plt.grid()
    plt.xlim(-lim, lim)
    plt.ylim(-lim, lim)
    plt.xlabel("U [m]")
    plt.ylabel("V [m]")
    plt.savefig(os.path.join(save_path, "UV.png"), format="png", dpi=200)

    plt.rcParams["font.size"] = 16


def plot_comparison(
    ax, times, mean1, mean2, std1, std2, true1, true2, rmse, diff=False
):
    times, units = time_units(times)
    for i, a in enumerate(ax):
        a[0].plot(rmse[..., i], "o")
        a[0].set_xlabel("Sample")

        if diff:
            a[1].plot(times, mean1[i] - true1[i].real, label="Estimate")
            a[1].fill_between(times, -std1[i], std1[i], color="tab:orange", alpha=0.3)
            a[1].fill_between(
                times, -2 * std1[i], 2 * std1[i], color="tab:orange", alpha=0.3
            )
            a[2].plot(times, mean2[i] - true2[i], label="Estimate")
            a[2].fill_between(times, -std2[i], std2[i], color="tab:orange", alpha=0.3)
            a[2].fill_between(
                times, -2 * std2[i], 2 * std2[i], color="tab:orange", alpha=0.3
            )
        else:
            a[1].plot(times, true1[i], label="True")
            a[1].plot(times, mean1[i], label="Estimate")
            a[1].fill_between(
                times,
                mean1[i] - std1[i],
                mean1[i] + std1[i],
                color="tab:orange",
                alpha=0.3,
            )
            a[1].fill_between(
                times,
                mean1[i] - 2 * std1[i],
                mean1[i] + 2 * std1[i],
                color="tab:orange",
                alpha=0.3,
            )
            a[2].plot(times, true2[i], label="True")
            a[2].plot(times, mean2[i], label="Estimate")
            a[2].fill_between(
                times,
                mean2[i] - std2[i],
                mean2[i] + std2[i],
                color="tab:orange",
                alpha=0.3,
            )
            a[2].fill_between(
                times,
                mean2[i] - 2 * std2[i],
                mean2[i] + 2 * std2[i],
                color="tab:orange",
                alpha=0.3,
            )

        a[1].set_xlabel(f"Time [{units}]")
        a[1].legend()
        a[2].set_xlabel(f"Time [{units}]")
        a[2].legend()


def plot_complex_real_imag(
    times,
    param,
    true,
    rmse,
    name: str,
    save_name: str = None,
    diff: bool = False,
    max_plots: int = 10,
    save_dir: str = "plots/",
):
    n_params = min(param.shape[1], max_plots)
    # idx = np.random.permutation(param.shape[1])
    # print(param.shape, true.shape)
    # param = param[:, idx]
    # true = true[idx]
    mean_r = param.real.mean(axis=0)
    mean_i = param.imag.mean(axis=0)
    std_r = param.real.std(axis=0)
    std_i = param.imag.std(axis=0)

    fig, ax = plt.subplots(n_params, 3, figsize=(18, 4.5 * n_params))

    ax[0, 0].set_title("Root Mean Squared Error")
    ax[0, 1].set_title(f"{name} Real")
    ax[0, 2].set_title(f"{name} Imag")

    plot_comparison(
        ax, times, mean_r, mean_i, std_r, std_i, true.real, true.imag, rmse, diff=diff
    )

    if save_name is not None:
        fig.savefig(
            os.path.join(save_dir, f"{save_name}_real_imag.pdf"),
            format="pdf",
            bbox_inches="tight",
        )
    plt.close(fig)


def plot_complex_amp_phase(
    times,
    param,
    true,
    rmse,
    name: str,
    save_name: str = None,
    diff: bool = False,
    max_plots: int = 10,
    save_dir: str = "plots/",
):
    n_params = min(param.shape[1], max_plots)
    # idx = np.random.permutation(param.shape[1])
    # print(param.shape, true.shape)
    # param = param[:, idx]
    # true = true[idx]
    mean_amp = jnp.abs(param).mean(axis=0)
    mean_phase = jnp.rad2deg(jnp.angle(param)).mean(axis=0)
    std_amp = jnp.abs(param).std(axis=0)
    std_phase = jnp.rad2deg(jnp.angle(param)).std(axis=0)

    fig, ax = plt.subplots(n_params, 3, figsize=(18, 4.5 * n_params))

    ax[0, 0].set_title("Root Mean Squared Error")
    ax[0, 1].set_title(f"{name} Magnitude")
    ax[0, 2].set_title(f"{name} Phase")

    plot_comparison(
        ax,
        times,
        mean_amp,
        mean_phase,
        std_amp,
        std_phase,
        jnp.abs(true),
        jnp.rad2deg(jnp.angle(true)),
        rmse,
        diff=diff,
    )

    if save_name is not None:
        fig.savefig(
            os.path.join(save_dir, f"{save_name}_amp_phase.pdf"),
            format="pdf",
            bbox_inches="tight",
        )
    plt.close(fig)


def plot_predictions(
    times,
    pred,
    args,
    type: str = "",
    model_name: str = "",
    max_plots: int = 10,
    save_dir: str = "plots/",
):
    plot_complex_real_imag(
        times=times,
        param=pred["ast_vis"],
        true=args["vis_ast_true"],
        rmse=pred["rmse_ast"],
        name="Ast. Vis.",
        save_name=f"{model_name}_{type}_ast_vis",
        max_plots=max_plots,
        save_dir=save_dir,
    )

    plot_complex_amp_phase(
        times=times,
        param=pred["rfi_vis"],
        true=args["vis_rfi_true"],
        rmse=pred["rmse_rfi"],
        name="RFI Vis.",
        save_name=f"{model_name}_{type}_rfi_vis",
        diff=False,  # True,
        max_plots=max_plots,
        save_dir=save_dir,
    )

    plot_complex_amp_phase(
        times=times,
        param=pred["gains"],
        true=args["gains_true"],
        rmse=pred["rmse_gains"],
        name="Gains",
        save_name=f"{model_name}_{type}_gains",
        diff=False,
        max_plots=max_plots,
        save_dir=save_dir,
    )
