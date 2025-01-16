import numpy as np
import pandas as pd
from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.stats import norm

from daskms import xds_from_ms, xds_from_table
import dask.array as da
import xarray as xr

from tge import TGE, Cl

from glob import glob
import os

from tqdm import tqdm

plt.rcParams["font.size"] = 16


def get_stats(
    stats_base: dict, name: str, img_dirs: list, img_name: str, n_sigma: float = 3.0
):

    n_dir = len(img_dirs.flatten())
    shape = img_dirs.shape

    TP = np.zeros(n_dir)
    im_noise = np.nan * np.zeros(n_dir)
    mean_I_error = np.nan * np.zeros(n_dir)
    mean_abs_I_error = np.nan * np.zeros(n_dir)
    std_I_error = np.nan * np.zeros(n_dir)
    det = np.zeros(n_dir)

    no_src = []
    no_match = []

    for i, img_dir in enumerate(img_dirs.flatten()):
        img_path = os.path.join(img_dir, f"{img_name}.csv")
        bdsf_path = os.path.join(img_dir, f"{img_name}.pybdsf.csv")

        try:
            df = pd.read_csv(img_path)
            n_src = len(df)
            if n_src > 0:
                im_noise[i] = df["Image_noise [mJy/beam]"].values[0]
                # idx = np.where(df["SNR"].values > n_sigma)[0]
                idx = np.where(
                    df["Total_flux_image [mJy]"]
                    > n_sigma * 1e3 * stats_base["im_noise_theory"][i]
                )[0]
                n_match = len(idx)
                print(
                    f"{df['Total_flux_image [mJy]'].min():.2f}, {df['Total_flux_image [mJy]'].max():.2f}, {n_sigma * 1e3 * stats_base['im_noise_theory'][i]:.2f}, {n_match}"
                )

                if n_match > 0:
                    TP[i] = n_match
                    mean_I_error[i] = np.mean(
                        df["Total_flux_image [mJy]"].iloc[idx]
                        - df["Total_flux_true [mJy]"].iloc[idx]
                    )
                    mean_abs_I_error[i] = np.mean(
                        np.abs(
                            df["Total_flux_image [mJy]"].iloc[idx]
                            - df["Total_flux_true [mJy]"].iloc[idx]
                        )
                    )
                    std_I_error[i] = np.mean(df["Total_flux_std [mJy]"].iloc[idx])

                    bdsf_df = pd.read_csv(bdsf_path, skiprows=5)
                    if len(bdsf_df) > 0:
                        # det[i] = np.sum(
                        #     bdsf_df[" Total_flux"] > n_sigma * 1e-3 * im_noise[i]
                        # )
                        det[i] = np.sum(
                            bdsf_df[" Total_flux"]
                            > n_sigma * stats_base["im_noise_theory"][i]
                        )
                else:
                    no_match.append(img_path)
        except:
            no_src.append(img_path)

    stats = {
        "TP": TP,
        "im_noise": im_noise,
        "mean_I_error": np.abs(mean_I_error),
        "mean_abs_I_error": mean_abs_I_error,
        "std_I_error": std_I_error,
        "det": det,
    }
    stats = {key: val.reshape(*shape) for key, val in stats.items()}
    stats.update(stats_base)

    print()
    print(f"{name} results")
    print(f"Number of runs with no source extraction results {len(no_src)}")
    if len(no_src) > 0:
        print("Runs with no source extraction")
        for no in no_src:
            print(no)
    if len(no_match) > 0:
        print(f"Runs with no matches with SNR > {n_sigma:.1f}")
        for no in no_match:
            print(no)

    stats["FP"] = stats["det"] - stats["TP"]
    stats["FN"] = stats["P"] - stats["TP"]
    # stats["TN"] = unknown
    # stats["N"] = stats["FP"] + stats["TN"] => unknown
    stats["TPR"] = stats["TP"] / stats["P"]
    stats["FNR"] = stats["FN"] / stats["P"]
    # stats["FPR"] = stats["FP"] / stats["N"]
    # stats["TNR"] = stats["TN"] / stats["N"]
    stats["prec"] = stats["TP"] / (stats["TP"] + stats["FP"])  # Purity
    stats["prec"] = np.where(np.isnan(stats["prec"]), 0, stats["prec"])
    stats["rec"] = stats["TP"] / (stats["TP"] + stats["FN"])  # Completeness

    stats["SNR_RFI"] = stats["mean_rfi"] / stats["vis_noise"]
    stats["RFI/AST"] = stats["mean_rfi"] / stats["mean_ast"]

    return stats


def bin_stats(stats, n_bins, statistic, bin_array):

    bins = np.logspace(
        np.log10(np.min(bin_array)), np.log10(np.max(bin_array) + 1), n_bins + 1
    )
    bin_idx = [
        np.where((bin_array >= bins[i]) & (bin_array < bins[i + 1]))[0]
        for i in range(n_bins)
    ]

    binned = {}
    for key, value in stats.items():
        binned_stat = np.zeros(n_bins)
        for i, idx in enumerate(bin_idx):
            binned_value = value[idx]
            binned_stat[i] = statistic(binned_value[~np.isnan(binned_value)])
        binned[key] = binned_stat

    return binned


def get_gain_std(zarr_file, fish_file, data_col="DATA"):

    xds = xr.open_zarr(zarr_file)
    xds_fish = xr.open_zarr(fish_file)
    gains = (
        xds.gains_ants.data[:, :, 0]
        if data_col == "DATA"
        else np.ones_like(xds.gains_ants.data[:, :, 0])
    )
    fish_gains = xds_fish.gains.data
    amp_std = (
        100 * np.nanmean(np.std(np.abs(fish_gains), axis=0).T / np.abs(gains)).compute()
    )
    if len(np.where(np.std(np.angle(fish_gains), axis=0) == 0)[0]) > 0:
        print(os.path.split(zarr_file)[1])
    phase_std = np.rad2deg(np.nanmean(np.std(np.angle(fish_gains), axis=0))).compute()

    return amp_std, phase_std


def get_binned_gain_std(
    zarr_files, fish_files, bin_idx, percentiles=[16, 50, 84], data_col="DATA"
):

    binned_bias = np.zeros((2, len(bin_idx), 3))

    for i, idx in enumerate(bin_idx):

        biases = []
        for zarr_file, fish_file in zip(zarr_files[idx], fish_files[idx]):
            try:
                biases.append(get_gain_std(zarr_file, fish_file, data_col))
            except:
                pass

        if len(biases) > 0:
            binned_bias[:, i] = np.array(
                [np.percentile(biases, p, axis=0) for p in percentiles]
            ).T

    return binned_bias


def plot_gain_std(binned_snr, binned_bias, data_dir, tab_suffix):

    fig, ax = plt.subplots(1, 1, figsize=(7, 6))

    ax.plot(
        binned_snr,
        binned_bias[0, :, 1],
        "o-",
        label="Amplitude",
        color="tab:blue",
    )
    ax.fill_between(
        x=binned_snr,
        y1=binned_bias[0, :, 0],
        y2=binned_bias[0, :, 2],
        color="tab:blue",
        alpha=0.1,
    )
    ax.legend(loc="upper right")
    ax.set_xlabel("SNR$(|V^{RFI}|)$")
    ax.set_ylabel("Mean Gain Amp Post. Std [%]")
    # ax.semilogx()
    ax.loglog()

    ax2 = ax.twinx()
    ax2.plot(
        binned_snr,
        binned_bias[1, :, 1],
        "o-",
        label="Phase",
        color="tab:orange",
    )
    ax2.fill_between(
        x=binned_snr,
        y1=binned_bias[1, :, 0],
        y2=binned_bias[1, :, 2],
        color="tab:orange",
        alpha=0.1,
    )
    ax2.legend(loc="center right")
    ax2.loglog()
    ax2.set_ylabel("Gain Phase Post. Std [deg]")

    plt.savefig(
        os.path.join(data_dir, f"plots/Gain_Std{tab_suffix}.pdf"),
        format="pdf",
        dpi=200,
        bbox_inches="tight",
    )


def get_gain_bias(sim_file, fish_file, data_col="DATA"):

    xds = xr.open_zarr(sim_file)
    xds_fish = xr.open_zarr(fish_file)
    gains = (
        xds.gains_ants.data[:, :, 0].compute()
        if data_col == "DATA"
        else np.ones_like(xds.gains_ants.data[:, :, 0]).compute()
    )
    fish_gains = xds_fish.gains.data
    amp_bias = np.abs(
        (np.mean(np.abs(fish_gains), axis=0).T - np.abs(gains))
        / np.std(np.abs(fish_gains), axis=0).T
    ).compute()
    phase_bias = np.abs(
        (np.mean(np.angle(fish_gains), axis=0).T - np.angle(gains))
        / np.std(np.angle(fish_gains), axis=0).T
    ).compute()
    # Maybe this should be an RMSE
    error_amp = np.nanmean(amp_bias)
    error_phase = np.nanmean(phase_bias)

    return error_amp, error_phase


def get_binned_gain_bias(
    zarr_files, fish_files, bin_idx, percentiles=[16, 50, 84], data_col="DATA"
):

    binned_bias = np.zeros((2, len(bin_idx), 3))

    for i, idx in enumerate(bin_idx):

        biases = []
        for zarr_file, fish_file in zip(zarr_files[idx], fish_files[idx]):
            try:
                biases.append(get_gain_bias(zarr_file, fish_file, data_col))
            except:
                pass

        if len(biases) > 0:
            binned_bias[:, i] = np.array(
                [np.percentile(biases, p, axis=0) for p in percentiles]
            ).T

    # print(binned_errors[:, -1, :])

    return binned_bias


def plot_gain_bias(binned_snr, binned_bias, data_dir, tab_suffix):

    fig, ax = plt.subplots(1, 1, figsize=(7, 6))

    ax.plot(
        binned_snr,
        binned_bias[0, :, 1],
        "o-",
        label="Amplitude",
        color="tab:blue",
    )
    ax.fill_between(
        x=binned_snr,
        y1=binned_bias[0, :, 0],
        y2=binned_bias[0, :, 2],
        color="tab:blue",
        alpha=0.1,
    )
    ax.legend(loc="upper right")
    ax.set_xlabel("SNR$(|V^{RFI}|)$")
    ax.set_ylabel("Gain Amp Normalized Bias")
    # ax.semilogx()
    ax.loglog()

    ax2 = ax.twinx()
    ax2.plot(
        binned_snr,
        binned_bias[1, :, 1],
        "o-",
        label="Phase",
        color="tab:orange",
    )
    ax2.fill_between(
        x=binned_snr,
        y1=binned_bias[1, :, 0],
        y2=binned_bias[1, :, 2],
        color="tab:orange",
        alpha=0.1,
    )
    ax2.legend(loc="center right")
    ax2.loglog()
    ax2.set_ylabel("Gain Phase Normalized Bias")

    plt.savefig(
        os.path.join(data_dir, f"plots/Gain_Bias{tab_suffix}.pdf"),
        format="pdf",
        dpi=200,
        bbox_inches="tight",
    )


def get_gain_error(sim_file, tab_file, data_col="DATA"):

    xds = xr.open_zarr(sim_file)
    xds_tab = xr.open_zarr(tab_file)
    gains = (
        xds.gains_ants.data[:, :, 0]
        if data_col == "DATA"
        else np.ones_like(xds.gains_ants.data[:, :, 0])
    )
    error_amp = (
        100
        * np.sqrt(
            np.mean((np.abs(xds_tab.gains.data[0, :, :]).T - np.abs(gains)) ** 2)
        ).compute()
    )
    error_phase = np.sqrt(
        np.mean(
            np.rad2deg(
                (
                    np.unwrap(np.angle(xds_tab.gains.data[0, :, :]).T.compute(), axis=0)
                    - np.unwrap(np.angle(gains).compute(), axis=0)
                )
            )
            ** 2
        )
    )

    return error_amp, error_phase


def get_binned_gain_errors(
    zarr_files, tab_files, bin_idx, percentiles=[16, 50, 84], data_col="DATA"
):

    binned_errors = np.zeros((2, len(bin_idx), 3))

    for i, idx in enumerate(bin_idx):

        errors = np.array(
            [
                get_gain_error(zarr_file, tab_file, data_col)
                for zarr_file, tab_file in zip(zarr_files[idx], tab_files[idx])
            ]
        )
        binned_errors[:, i] = np.array(
            [np.percentile(errors, p, axis=0) for p in percentiles]
        ).T

    return binned_errors


def plot_gain_errors(binned_snr, binned_errors, data_dir, tab_suffix):

    fig, ax = plt.subplots(1, 1, figsize=(7, 6))

    ax.plot(
        binned_snr,
        binned_errors[0, :, 1],
        "o-",
        label="Amplitude",
        color="tab:blue",
    )
    ax.fill_between(
        x=binned_snr,
        y1=binned_errors[0, :, 0],
        y2=binned_errors[0, :, 2],
        color="tab:blue",
        alpha=0.1,
    )
    ax.legend(loc="upper right")
    ax.set_xlabel("SNR$(|V^{RFI}|)$")
    ax.set_ylabel("Gain Amp RMSE [%]")
    # ax.semilogx()
    ax.loglog()

    ax2 = ax.twinx()
    ax2.plot(
        binned_snr,
        binned_errors[1, :, 1],
        "o-",
        label="Phase",
        color="tab:orange",
    )
    ax2.fill_between(
        x=binned_snr,
        y1=binned_errors[1, :, 0],
        y2=binned_errors[1, :, 2],
        color="tab:orange",
        alpha=0.1,
    )
    ax2.legend(loc="center right")
    ax2.loglog()
    ax2.set_ylabel("Gain Phase RMSE [deg]")

    plt.savefig(
        os.path.join(data_dir, f"plots/Gain_Error{tab_suffix}.pdf"),
        format="pdf",
        dpi=200,
        bbox_inches="tight",
    )


def get_error(sim_file, tab_file):

    xds = xr.open_zarr(sim_file)
    xds_tab = xr.open_zarr(tab_file)
    error = (
        (xds.vis_ast.data[:, :, 0] - xds_tab.ast_vis.data[0, :, :].T)
        .flatten()
        .compute()
    )

    return np.concatenate([error.real, error.imag])


def get_aoflagged(sim_file, ms_file):

    xds = xr.open_zarr(sim_file)
    xds_ms = xds_from_ms(ms_file)[0]
    idx = np.where(xds_ms["AO_FLAGS"].data[:, 0, 0] == 0)[0].compute()
    error = (
        (xds.vis_ast.data[:, :, 0].flatten()[idx] - xds_ms.CAL_DATA.data[idx, 0, 0])
        .flatten()
        .compute()
    )

    return np.concatenate([error.real, error.imag])


def get_perfect_flagged(sim_file):

    xds = xr.open_zarr(sim_file)
    idx = np.where(xds.flags.data[:, :, 0].flatten() == 0)[0].compute()
    error = (
        (
            xds.vis_ast.data[:, :, 0].flatten()[idx]
            - xds.vis_calibrated.data[:, :, 0].flatten()[idx]
        )
        .flatten()
        .compute()
    )

    return np.concatenate([error.real, error.imag])


def get_noise(sim_file, gains=False):

    xds = xr.open_zarr(sim_file)
    if gains:
        a1, a2 = xds.antenna1.data.compute(), xds.antenna1.data.compute()
        gains = xds.gains_ants.data[:, :, 0].compute()
        noise = (
            xds.noise_data.data[:, :, 0].compute()
            * gains[:, a1]
            * np.conjugate(gains[:, a2])
        ).flatten()
    else:
        noise = xds.noise_data.data[:, :, 0].flatten().compute()

    return np.concatenate([noise.real, noise.imag])


def get_vis_cal_error(sim_file):

    xds = xr.open_zarr(sim_file)
    vis_cal = (
        (xds.vis_calibrated.data[:, :, 0] - xds.vis_ast.data[:, :, 0])
        .flatten()
        .compute()
    )

    return np.concatenate([vis_cal.real, vis_cal.imag])


def get_bins(bin_array, n_bins):

    bins = np.logspace(
        np.log10(np.min(bin_array)), np.log10(np.max(bin_array) + 1), n_bins + 1
    )
    bin_idx = [
        np.where((bin_array >= bins[i]) & (bin_array < bins[i + 1]))[0]
        for i in range(n_bins)
    ]

    return bin_idx


def get_all_errors(zarr_files, ms_files, tab_files, bin_idx):

    errors = []
    errors_flags1 = []
    errors_flags2 = []
    # noises = []
    # noises_g = []
    print()
    print("Calculating Errors")
    for idx in tqdm(bin_idx):

        errors.append(
            np.concatenate(
                [
                    get_error(zarr_file, tab_file)
                    for zarr_file, tab_file in zip(zarr_files[idx], tab_files[idx])
                ]
            )
        )
        errors_flags1.append(
            np.concatenate(
                [get_perfect_flagged(zarr_file) for zarr_file in zarr_files[idx]]
            )
        )
        errors_flags2.append(
            np.concatenate(
                [
                    get_aoflagged(zarr_file, ms_file)
                    for zarr_file, ms_file in zip(zarr_files[idx], ms_files[idx])
                ]
            )
        )
        # noises.append(np.concatenate([get_noise(zarr_file) for zarr_file in zarr_files[idx]]))
        # noises_g.append(np.concatenate([get_noise(zarr_file, gains=True) for zarr_file in zarr_files[idx]]))

    return errors, errors_flags1, errors_flags2


def get_file_names(
    data_dir: str, model_name: str, tab_suffix: str = "", data_col: str = "DATA"
):

    data_dirs = np.array(glob(os.path.join(data_dir, "*")))

    data_dirs = np.array([d for d in data_dirs if "SEED" in d])

    if len(data_dirs) == 0:
        raise ValueError(f"No data found in {data_dir}")

    img_dirs = np.array([os.path.join(d, "images") for d in data_dirs])
    ps_dirs = np.array([os.path.join(d, "power_spectrum") for d in data_dirs])

    ms_files = np.array(
        [os.path.join(d, os.path.split(d)[1] + ".ms") for d in data_dirs]
    )
    zarr_files = np.array(
        [os.path.join(d, os.path.split(d)[1] + ".zarr") for d in data_dirs]
    )
    tab_files = np.array(
        [
            os.path.join(
                os.path.join(d, "results"),
                f"{model_name}{tab_suffix}.zarr",
            )
            for d in data_dirs
        ]
    )
    fish_files = np.array(
        [
            os.path.join(
                os.path.join(d, "results"),
                f"fisher{model_name[3:]}{tab_suffix}.zarr",
            )
            for d in data_dirs
        ]
    )

    n_sim = len(data_dirs)

    im_noise_theory = np.zeros(n_sim)
    rchi2 = np.zeros(n_sim)
    mean_rfi = np.zeros(n_sim)
    max_rfi = np.zeros(n_sim)
    mean_ast = np.zeros(n_sim)
    vis_noise = np.zeros(n_sim)
    flags1 = np.zeros(n_sim)
    flags2 = np.zeros(n_sim)
    idx = []

    no_tab = []
    bad_tab = []
    no_fish = []

    for i in tqdm(range(n_sim)):

        try:
            xds = xr.open_zarr(fish_files[i])
        except:
            no_fish.append(data_dirs[i])

        try:
            xds_ms = xds_from_ms(ms_files[i])[0]
            xds = xr.open_zarr(zarr_files[i])
            xds_res = xr.open_zarr(tab_files[i])
            vis_obs = xds_ms[data_col].data[:, 0, 0].reshape(xds.n_time, xds.n_bl)
            # vis_obs = xds.vis_obs.data[:, :, 0]
            rchi2[i] = (
                np.mean(np.abs(vis_obs - xds_res.vis_obs.data[0, :, :].T) ** 2)
                / (2 * xds.noise_std.data**2)
            ).compute()[0]
            mean_rfi[i] = np.abs(xds_ms["RFI_MODEL_DATA"].data).mean().compute()
            max_rfi[i] = np.abs(xds_ms["RFI_MODEL_DATA"].data).max().compute()
            mean_ast[i] = np.abs(xds_ms["AST_MODEL_DATA"].data).mean().compute()
            # flags1[i] = np.abs(xds_ms["3S_FLAGS"]).mean().compute()
            flags1[i] = np.abs(xds.flags.mean()).compute()
            flags2[i] = np.abs(xds_ms["AO_FLAGS"].data).mean().compute()
            vis_noise[i] = np.std(xds_ms["NOISE_DATA"].data.real).compute()
            im_noise_theory[i] = vis_noise[i] / np.sqrt(xds_ms["DATA"].data.shape[0])
            if rchi2[i] < 1:
                idx.append(i)
            else:
                bad_tab.append(data_dirs[i])
        except:
            no_tab.append(data_dirs[i])

    idx = np.array(idx)
    bad_rchi2 = rchi2[rchi2 > 1]

    print()
    print(f"Total number of simulations      : {n_sim}")
    print(f"Number of sims with good results : {len(idx)}")
    print(f"Number of sims with rchi2 > 1    : {len(bad_tab)}")
    print(f"Number of sims with no results   : {len(no_tab)}")
    print(f"Number of sims with no Fisher    : {len(no_fish)}")
    if len(bad_tab) > 0:
        print()
        print("Simulations with rchi2 > 1")
        for bad, rchi2_ in zip(bad_tab, bad_rchi2):
            print(bad, rchi2_)
    if len(no_tab) > 0:
        print()
        print("Simulations with no results")
        for no in no_tab:
            print(no)
    if len(no_fish) > 0:
        print()
        print("Simulations with no Fisher")
        for no in no_fish:
            print(no)

    files = {
        "data_dirs": data_dirs[idx],
        "img_dirs": img_dirs[idx],
        "ps_dirs": ps_dirs[idx],
        "ms_files": ms_files[idx],
        "zarr_files": zarr_files[idx],
        "tab_files": tab_files[idx],
        "fish_files": fish_files[idx],
    }

    data = {
        "rchi2": rchi2[idx],
        "mean_rfi": mean_rfi[idx],
        "max_rfi": max_rfi[idx],
        "mean_ast": mean_ast[idx],
        "flags1": flags1[idx],
        "flags2": flags2[idx],
        "vis_noise": vis_noise[idx],
        "im_noise_theory": im_noise_theory[idx],
    }

    return files, data


def get_names(tab_suffix: str = ""):

    im_suffix = ""

    names = {
        "options": [
            # "perfect",
            "ideal",
            "tab",
            "flag1",
            "flag2",
        ],
        "names": {
            # "perfect": "No Noise, No RFI",
            "ideal": "Uncontaminated",
            "tab": "TABASCAL",
            "flag1": "Perfect Flagging",
            "flag2": "AOFlagger",
        },
        "colors": {
            # "perfect": "tab:purple",
            "ideal": "tab:blue",
            "tab": "tab:orange",
            "flag1": "tab:red",
            "flag2": "tab:green",
        },
        # "img_names": {
        #     "perfect": f"AST_MODEL_DATA_0.0sigma{suffix}",
        #     "ideal": f"AST_DATA_0.0sigma{suffix}",
        #     "tab": f"TAB_DATA_0.0sigma{suffix}",
        #     "flag1": f"CAL_DATA_3.0sigma{suffix}",
        #     "flag2": f"CAL_DATA_aoflagger{suffix}",
        # },
        "img_names": {
            # "perfect": f"AST_MODEL_DATA_0.0sigma{im_suffix}",
            "ideal": f"AST_DATA_0.0sigma{im_suffix}",
            "tab": f"TAB_DATA_0.0sigma{im_suffix}{tab_suffix}",
            "flag1": f"CAL_DATA_3.0sigma{im_suffix}",
            "flag2": f"CAL_DATA_aoflagger{im_suffix}",
        },
    }

    return names


def plot_errors(
    errors: dict, bin_idx: list, SNR: list, noise_std: float, hist_bins: int
):

    plt.figure(figsize=(15, 5))
    for i, idx in enumerate(bin_idx):
        x = np.linspace(errors[i].min(), errors[i].max(), 100)
        mean, std = norm.fit(errors[i])
        y = norm.pdf(x, mean, std)
        plt.hist(
            errors[i],
            bins=hist_bins,
            density=True,
            color=f"C{i}",
            alpha=0.8,
            label=f"{10*np.log10(np.mean(SNR[idx])): = .1f} dB",
            histtype="step",
        )
        plt.plot(x, y, color=f"C{i}", alpha=0.8)

    x = np.linspace(-5, 5, 100)
    plt.plot(x, norm.pdf(x, 0, noise_std), color="k", label="Vis Noise")
    plt.ylim(1e-6, 2e0)
    plt.xlim(x[0], x[-1])
    plt.semilogy()
    plt.xlabel("AST Vis Error [Jy]")
    plt.ylabel("Probability Density [Jy$^{-1}$]")
    plt.grid()


def plot_tab_errors(
    zarr_files: list,
    tab_files: list,
    bin_idx: list,
    SNR: list,
    noise_std: float,
    hist_bins: int,
    data_dir: str,
):

    errors = []
    for idx in tqdm(bin_idx):
        errors.append(
            np.concatenate(
                [
                    get_error(zarr_file, tab_file)
                    for zarr_file, tab_file in zip(zarr_files[idx], tab_files[idx])
                ]
            )
        )

    plot_errors(errors, bin_idx, SNR, noise_std, hist_bins)

    mean, std = norm.fit(errors[0])
    plt.axvline(mean - 4 * std, ls="dashed", color="tab:blue")
    plt.axvline(mean + 4 * std, ls="dashed", color="tab:blue", label="4$\\sigma$ Error")
    plt.legend(fontsize=14, title="SNR$(|V^{RFI}|)$")
    plt.title("TABASCAL Errors")

    plt.text(
        -4.5,
        0.5,
        f"Vis Noise std : {noise_std:.2f} Jy",
        {"fontsize": 14},
        bbox={"facecolor": "white", "boxstyle": "round", "pad": 0.4, "edgecolor": "k"},
    )
    plt.text(
        -4.5,
        0.1,
        f"TAB Error std : {std:.2f} Jy",
        {"fontsize": 14},
        bbox={"facecolor": "white", "boxstyle": "round", "pad": 0.4, "edgecolor": "k"},
    )

    plt.savefig(
        os.path.join(data_dir, "plots/TABASCAL_Errors.pdf"),
        format="pdf",
        dpi=200,
        bbox_inches="tight",
    )


def plot_perfect_flag_errors(
    zarr_files: list,
    bin_idx: list,
    SNR: list,
    noise_std: float,
    hist_bins: int,
    data_dir: str,
):
    errors = []
    for idx in tqdm(bin_idx):
        errors.append(
            np.concatenate(
                [get_perfect_flagged(zarr_file) for zarr_file in zarr_files[idx]]
            )
        )

    plot_errors(errors, bin_idx, SNR, noise_std, hist_bins)

    sig3 = 3 * noise_std * np.sqrt(2)
    plt.axvline(-sig3, ls="dashed", color="k")
    plt.axvline(sig3, ls="dashed", color="k", label="3$\\sigma$ Noise")
    plt.legend(fontsize=14, title="SNR$(|V^{RFI}|)$")
    plt.title("Perfectly Calibrated & 3$\\sigma$ Flagged Errors")

    plt.savefig(
        os.path.join(data_dir, "plots/Perfect_Flag_Errors.pdf"),
        format="pdf",
        dpi=200,
        bbox_inches="tight",
    )


def plot_aoflagger_errors(
    zarr_files: list,
    ms_files: list,
    bin_idx: list,
    SNR: list,
    noise_std: float,
    hist_bins: int,
    data_dir: str,
):

    errors = []
    for idx in bin_idx:
        errors.append(
            np.concatenate(
                [
                    get_aoflagged(zarr_file, ms_file)
                    for zarr_file, ms_file in zip(zarr_files[idx], ms_files[idx])
                ]
            )
        )

    plot_errors(errors, bin_idx, SNR, noise_std, hist_bins)

    sig3 = 3 * noise_std * np.sqrt(2)
    plt.axvline(-sig3, ls="dashed", color="k")
    plt.axvline(sig3, ls="dashed", color="k", label="3$\\sigma$ Noise")
    plt.legend(fontsize=14, title="SNR$(|V^{RFI}|)$")
    plt.title("AOFlagger Calibrated & Flagged Errors")

    plt.savefig(
        os.path.join(data_dir, "plots/AOFlagger_Errors.pdf"),
        format="pdf",
        dpi=200,
        bbox_inches="tight",
    )


def plot_vis_cal_errors(
    zarr_files: list,
    bin_idx: list,
    SNR: list,
    noise_std: float,
    hist_bins: int,
    data_dir: str,
):

    errors = []
    for idx in bin_idx:
        errors.append(
            np.concatenate(
                [get_vis_cal_error(zarr_file) for zarr_file in zarr_files[idx]]
            )
        )

    plot_errors(errors, bin_idx, SNR, noise_std, hist_bins)

    sig3 = 3 * noise_std * np.sqrt(2)
    plt.axvline(-sig3, ls="dashed", color="k")
    plt.axvline(sig3, ls="dashed", color="k", label="3$\\sigma$ Noise")
    plt.legend(fontsize=14, title="SNR$(|V^{RFI}|)$")
    plt.title("Calibrated, Unflagged Visibilities")

    plt.savefig(
        os.path.join(data_dir, "plots/Vis_Cal_Errors.pdf"),
        format="pdf",
        dpi=200,
        bbox_inches="tight",
    )


def plot_errors_ax(
    ax, errors: dict, bin_idx: list, SNR: list, noise_std: float, hist_bins: int
):

    for i, idx in enumerate(bin_idx):
        x = np.linspace(errors[i].min(), errors[i].max(), 100)
        mean, std = norm.fit(errors[i])
        y = norm.pdf(x, mean, std)
        ax.hist(
            errors[i],
            bins=hist_bins,
            density=True,
            color=f"C{i}",
            alpha=0.8,
            label=f"{10*np.log10(np.mean(SNR[idx])): = .1f} dB",
            histtype="step",
        )
        ax.plot(x, y, color=f"C{i}", alpha=0.8)

    x = np.linspace(-5, 5, 100)
    ax.plot(x, norm.pdf(x, 0, noise_std), color="k", label="Vis Noise")
    ax.semilogy()
    ax.set_ylabel("Probability Density [Jy$^{-1}$]")
    ax.grid()


def plot_tab_error_ax(
    ax, zarr_files, tab_files, bin_idx, SNR, noise_std, hist_bins=100
):

    errors = []
    for idx in tqdm(bin_idx):
        errors.append(
            np.concatenate(
                [
                    get_error(zarr_file, tab_file)
                    for zarr_file, tab_file in zip(zarr_files[idx], tab_files[idx])
                ]
            )
        )

    plot_errors_ax(ax, errors, bin_idx, SNR, noise_std, hist_bins)

    mean, std = norm.fit(errors[0])
    ax.axvline(mean - 4 * std, ls="dashed", color="tab:blue")
    ax.axvline(mean + 4 * std, ls="dashed", color="tab:blue", label="4$\\sigma$ Error")

    return std


def plot_flag_error_ax(ax, zarr_files, bin_idx, SNR, noise_std, hist_bins=100):

    errors = []
    for idx in tqdm(bin_idx):
        errors.append(
            np.concatenate(
                [get_perfect_flagged(zarr_file) for zarr_file in zarr_files[idx]]
            )
        )

    plot_errors_ax(ax, errors, bin_idx, SNR, noise_std, hist_bins)


def plot_aoflag_error_ax(
    ax, zarr_files, ms_files, bin_idx, SNR, noise_std, hist_bins=100
):

    errors = []
    for idx in tqdm(bin_idx):
        errors.append(
            np.concatenate(
                [
                    get_aoflagged(zarr_file, ms_file)
                    for zarr_file, ms_file in zip(zarr_files[idx], ms_files[idx])
                ]
            )
        )

    plot_errors_ax(ax, errors, bin_idx, SNR, noise_std, hist_bins)


def plot_errors_all(files, bin_idx, SNR, noise_std, hist_bins, data_dir, suffix):

    fig, ax = plt.subplots(3, 1, figsize=(15, 15))

    tab_std = plot_tab_error_ax(
        ax[0],
        files["zarr_files"],
        files["tab_files"],
        bin_idx,
        SNR,
        noise_std,
        hist_bins,
    )
    plot_aoflag_error_ax(
        ax[1],
        files["zarr_files"],
        files["ms_files"],
        bin_idx,
        SNR,
        noise_std,
        hist_bins,
    )
    plot_flag_error_ax(ax[2], files["zarr_files"], bin_idx, SNR, noise_std, hist_bins)

    ax[0].tick_params(axis="x", labelbottom=False)  # changes apply to the x-axis
    ax[1].tick_params(axis="x", labelbottom=False)  # changes apply to the x-axis

    ax[0].set_title("TABASCAL Errors")
    # ax[0].plot(x, t.pdf(x, *t.fit(errors[-1])))

    ax[0].text(
        -4.5,
        0.5,
        f"Vis Noise std : {noise_std:.2f} Jy",
        {"fontsize": 14},
        bbox={"facecolor": "white", "boxstyle": "round", "pad": 0.4, "edgecolor": "k"},
    )
    ax[0].text(
        -4.5,
        0.1,
        f"TAB Error std : {tab_std:.2f} Jy",
        {"fontsize": 14},
        bbox={"facecolor": "white", "boxstyle": "round", "pad": 0.4, "edgecolor": "k"},
    )

    sig3 = 3 * noise_std * np.sqrt(2)
    for a in ax:
        a.axvline(-sig3, ls="dashed", color="k")
        a.axvline(sig3, ls="dashed", color="k", label="3$\\sigma$ Noise")
        a.legend(fontsize=14, title="SNR$(|V^{RFI}|)$", loc="right")
        a.set_ylim(5e-6, 2e0)
        a.set_xlim(-5, 5)

    ax[1].text(
        -1.3,
        0.9,
        "AOFlagger, Perfect Calibration",
        fontsize=14,
        bbox={"facecolor": "white", "boxstyle": "round", "pad": 0.1, "edgecolor": "k"},
    )
    ax[2].text(
        -1.4,
        0.9,
        "3$\\sigma$ Flagging, Perfect Calibration",
        fontsize=14,
        bbox={"facecolor": "white", "boxstyle": "round", "pad": 0.1, "edgecolor": "k"},
    )

    ax[2].set_xlabel("AST Visibility Error [Jy]")

    plt.subplots_adjust(wspace=0, hspace=-0.03)  # figsize=(11.5,12)

    plt.savefig(
        os.path.join(data_dir, f"plots/Errors{suffix}.pdf"),
        format="pdf",
        dpi=200,
        bbox_inches="tight",
    )


def plot_sausage(all_med, all_q1, all_q2, names, x_name, y_name, error_alpha=0.1):

    plt.figure(figsize=(7, 6))
    fig, ax = plt.subplots(1, 1, figsize=(7, 6))

    for name, stat in all_med.items():
        marker = "o-" if name == "ideal" else ".-"
        ax.plot(
            stat[x_name],
            stat[y_name],
            marker,
            label=names["names"][name],
            color=names["colors"][name],
        )
        ax.fill_between(
            x=stat[x_name],
            y1=all_q1[name][y_name],
            y2=all_q2[name][y_name],
            color=names["colors"][name],
            alpha=error_alpha,
        )

    ax.legend()

    return ax


def plot_completeness(
    all_med: dict, all_q1: dict, all_q2: dict, names: dict, data_dir: str, suffix: str
):
    ax = plot_sausage(all_med, all_q1, all_q2, names, "SNR_RFI", "rec")

    ax.set_xlabel("SNR$(|V^{RFI}|)$")
    ax.set_ylabel("Completeness")
    ax.semilogx()

    ax2 = ax.twiny()
    ax2.plot(all_med["ideal"]["RFI/AST"], all_med["ideal"]["rec"], alpha=0)
    ax2.semilogx()
    ax2.set_xlabel("$|V^{RFI}| / |V^{AST}|$")

    plt.savefig(
        os.path.join(data_dir, f"plots/Completeness{suffix}.pdf"),
        format="pdf",
        dpi=200,
        bbox_inches="tight",
    )


def plot_purity(
    all_med: dict, all_q1: dict, all_q2: dict, names: dict, data_dir: str, suffix: str
):

    ax = plot_sausage(all_med, all_q1, all_q2, names, "SNR_RFI", "prec")

    ax.set_xlabel("SNR$(|V^{RFI}|)$")
    ax.set_ylabel("Purity")
    ax.semilogx()

    ax2 = ax.twiny()
    ax2.plot(all_med["ideal"]["RFI/AST"], all_med["ideal"]["prec"], alpha=0)
    ax2.semilogx()
    ax2.set_xlabel("$|V^{RFI}| / |V^{AST}|$")

    plt.savefig(
        os.path.join(data_dir, f"plots/Purity{suffix}.pdf"),
        format="pdf",
        dpi=200,
        bbox_inches="tight",
    )


def plot_image_noise(
    all_med, all_q1: dict, all_q2: dict, names, data_dir: str, suffix: str
):

    ax = plot_sausage(all_med, all_q1, all_q2, names, "SNR_RFI", "im_noise")

    ax.set_xlabel("SNR$(|V^{RFI}|)$")
    ax.set_ylabel("Image Noise [mJy/beam]")
    ax.semilogx()
    ax.semilogy()

    ax2 = ax.twiny()
    ax2.plot(all_med["ideal"]["RFI/AST"], all_med["ideal"]["im_noise"], alpha=0)
    ax2.semilogx()
    ax2.set_xlabel("$|V^{RFI}| / |V^{AST}|$")

    plt.savefig(
        os.path.join(data_dir, f"plots/ImageNoise{suffix}.pdf"),
        format="pdf",
        dpi=200,
        bbox_inches="tight",
    )


def plot_abs_flux_error(
    all_med,
    all_q1: dict,
    all_q2: dict,
    names,
    data_dir: str,
    vbounds: list,
    suffix: str,
):

    ax = plot_sausage(all_med, all_q1, all_q2, names, "SNR_RFI", "mean_abs_I_error")

    ax.axhline(0, ls="--", color="k")
    ax.set_xlabel("SNR$(|V^{RFI}|)$")
    ax.set_ylabel("Mean Absolute Flux Error [mJy]")
    # ax.semilogx()
    ax.loglog()
    # ax.xticks(10**np.arange(-1.0, 2.0, 1.0), 10**np.arange(-1.0, 2.0, 1.0))
    ax.set_ylim(vbounds[0], vbounds[1])

    ax2 = ax.twiny()
    ax2.plot(all_med["ideal"]["RFI/AST"], all_med["ideal"]["mean_abs_I_error"], alpha=0)
    ax2.semilogx()
    ax2.set_xlabel("$|V^{RFI}| / |V^{AST}|$")

    plt.savefig(
        os.path.join(data_dir, f"plots/AbsFluxError{suffix}.pdf"),
        format="pdf",
        dpi=200,
        bbox_inches="tight",
    )


def plot_flux_error(
    all_med,
    all_q1: dict,
    all_q2: dict,
    names,
    data_dir: str,
    vbounds: list,
    suffix: str,
):

    ax = plot_sausage(all_med, all_q1, all_q2, names, "SNR_RFI", "mean_I_error")

    ax.axhline(0, ls="--", color="k")
    ax.set_xlabel("SNR$(|V^{RFI}|)$")
    ax.set_ylabel("Mean Flux Error [mJy]")
    # ax.semilogx()
    ax.loglog()
    # ax.xticks(10**np.arange(-1.0, 2.0, 1.0), 10**np.arange(-1.0, 2.0, 1.0))
    ax.set_ylim(vbounds[0], vbounds[1])

    ax2 = ax.twiny()
    ax2.plot(all_med["ideal"]["RFI/AST"], all_med["ideal"]["mean_I_error"], alpha=0)
    ax2.semilogx()
    ax2.set_xlabel("$|V^{RFI}| / |V^{AST}|$")

    plt.savefig(
        os.path.join(data_dir, f"plots/FluxError{suffix}.pdf"),
        format="pdf",
        dpi=200,
        bbox_inches="tight",
    )


def plot_flag_noise(
    files: dict,
    noise_std: float,
    data: dict,
    all_stats: dict,
    names: dict,
    data_dir: str,
    suffix: str,
):

    flag_rate = np.linspace(0, 0.999, 100)
    n_row = xds_from_ms(files["ms_files"][0])[0].DATA.data.shape[0]
    ther_noise = 1e3 * noise_std / np.sqrt((1 - flag_rate) * n_row)

    plt.figure(figsize=(7, 6))
    for name, flag in zip(["flag1", "flag2"], [data["flags1"], data["flags2"]]):
        plt.plot(
            100 * flag,
            all_stats[name]["im_noise"],
            "o",
            label=names["names"][name],
            color=names["colors"][name],
        )

    plt.plot(100 * flag_rate, ther_noise, "k--", label="Theoretical")
    plt.xlabel("Flag Rate [%]")
    plt.ylabel("Actual Image Noise [mJy]")
    plt.semilogy()
    plt.legend()

    plt.savefig(
        os.path.join(data_dir, f"plots/ImageNoiseVsFlagRate{suffix}.pdf"),
        format="pdf",
        dpi=200,
        bbox_inches="tight",
    )


def plot_theoretical_noise(
    files: dict,
    noise_std: float,
    data: dict,
    all_stats: dict,
    names: dict,
    data_dir: str,
    suffix: str,
):

    n_row = xds_from_ms(files["ms_files"][0])[0].DATA.data.shape[0]

    ther_noise1 = np.array(
        [
            1e3 * noise_std / np.sqrt((1 - flag_rate) * n_row)
            for flag_rate in data["flags1"]
        ]
    )
    ther_noise2 = np.array(
        [
            1e3 * noise_std / np.sqrt((1 - flag_rate) * n_row)
            for flag_rate in data["flags2"]
        ]
    )

    sig_I = np.linspace(ther_noise1.min(), ther_noise1.max(), 100)

    plt.figure(figsize=(7, 6))
    for name, noise in zip(["flag1", "flag2"], [ther_noise1, ther_noise2]):
        plt.plot(
            noise,
            all_stats[name]["im_noise"],
            "o",
            label=names["names"][name],
            color=names["colors"][name],
        )

    plt.plot(sig_I, sig_I, "k--", label="Theoretical")
    plt.xlabel("Theoretical Image Noise [mJy]")
    plt.ylabel("Actual Image Noise [mJy]")
    plt.semilogy()
    plt.legend()

    plt.savefig(
        os.path.join(data_dir, f"plots/ImageNoiseVsTheory{suffix}.pdf"),
        format="pdf",
        dpi=200,
        bbox_inches="tight",
    )


def plot_image(
    img_dirs,
    img_names,
    names,
    options,
    mean_rfi,
    rfi_amp,
    data_dir,
    vbounds,
    suffix,
    log: bool = False,
    cmap: str = "Greys_r",
):

    idx = np.argmin(np.abs(mean_rfi - rfi_amp))
    imgs = [
        1e3 * fits.getdata(os.path.join(img_dirs[idx], name + "-image.fits"))[0, 0]
        for name in img_names.values()
    ]
    heads = [
        fits.getheader(os.path.join(img_dirs[idx], name + "-image.fits"))
        for name in img_names.values()
    ]
    noises = {
        key: 1e3
        * np.nanstd(fits.getdata(os.path.join(img_dirs[idx], name + "-residual.fits")))
        for key, name in img_names.items()
    }

    # cmap = 'PuOr_r'

    ##############################

    img_scale = "Log" if log else "Linear"
    # vbounds = [-5, 5] # 64 Ant
    vbounds = [-10, 10]
    vbounds = [-15, 15]
    # vbounds = [-100, 100]
    vbounds_log = [1e-1, 1e1]

    edge = np.abs(heads[0]["CDELT1"]) * heads[0]["NAXIS1"] / 2

    fig, ax = plt.subplots(2, 2, figsize=(12, 12))
    for i, a in enumerate(ax.flatten()):
        if log:
            im = a.imshow(
                np.abs(imgs[i]),
                cmap=cmap,
                origin="lower",
                extent=[-edge, edge, -edge, edge],
                norm=LogNorm(vmin=vbounds_log[0], vmax=vbounds_log[1]),
            )
        else:
            im = a.imshow(
                imgs[i],
                cmap=cmap,
                origin="lower",
                extent=[-edge, edge, -edge, edge],
                vmin=vbounds[0],
                vmax=vbounds[1],
            )
        a.text(
            -0.63,
            0.58,
            names[options[i]],
            {"fontsize": 20},
            bbox={
                "facecolor": "white",
                "boxstyle": "round",
                "pad": 0.2,
                "edgecolor": "k",
            },
        )
        a.text(
            -0.63,
            -0.63,
            r" $\sigma_I$:" + f" {noises[options[i]]:.2f} mJy/beam",
            {"fontsize": 16},
            bbox={
                "facecolor": "white",
                "boxstyle": "round",
                "pad": 0.2,
                "edgecolor": "k",
            },
        )

    # cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7]) # figsize=(15,11)
    cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="Flux mJy/beam")

    ax[0, 0].set_ylabel("DEC Offset [deg]")
    ax[1, 0].set_ylabel("DEC Offset [deg]")

    ax[1, 0].set_xlabel("RA Offset [deg]")
    ax[1, 1].set_xlabel("RA Offset [deg]")

    ax[0, 1].tick_params(
        axis="both",  # changes apply to the x-axis
        labelbottom=False,
        labelleft=False,
        direction="in",
    )
    ax[0, 0].tick_params(axis="x", labelbottom=False)  # changes apply to the x-axis
    ax[1, 1].tick_params(labelleft=False)

    for a in ax.flatten():
        a.tick_params(
            direction="in", left=True, right=True, top=True, bottom=True, labelsize=16
        )
    # plt.subplots_adjust(wspace=-0.4441, hspace=0) # figsize=(15,11)
    # plt.subplots_adjust(wspace=-0.2799, hspace=0.) # figsize=(13,12)
    plt.subplots_adjust(wspace=0, hspace=-0.03)  # figsize=(11.5,12)
    plt.suptitle(f"Mean RFI Amplitude: {mean_rfi[idx]:.2f} Jy", y=0.91)
    plt.savefig(
        os.path.join(
            data_dir,
            f"plots/CompImage{img_scale}_{cmap}_RFI_{mean_rfi[idx]:.2e}{suffix}.pdf",
        ),
        dpi=200,
        format="pdf",
        bbox_inches="tight",
    )


def plot_vis_ex(zarr_file, tab_file, data_dir, bl1=3e1, bl2=1e3, suffix=""):
    xds = xr.open_zarr(zarr_file)
    xds_tab = xr.open_zarr(tab_file)

    mean_rfi = np.mean(np.abs(xds.vis_rfi.data)).compute()

    bl_norm = np.linalg.norm(xds.bl_uvw.data[0, :, :-1], axis=-1).compute()

    bl1 = np.argmin(np.abs(bl_norm - bl1))
    bl2 = np.argmin(np.abs(bl_norm - bl2))

    time = xds.time.data / 60
    fig, ax = plt.subplots(2, 2, figsize=(10, 9))

    ax[0, 0].plot(time, np.abs(xds.vis_rfi.data[:, bl1]), label="Truth")
    ax[0, 0].plot(
        time, np.abs(xds_tab.rfi_vis.data[0, bl1]), alpha=0.5, label="tabascal"
    )
    ax[0, 0].set_ylabel("RFI Visibility Magnitude [Jy]")
    ax[0, 0].tick_params(axis="x", labelbottom=False)
    y_pos = ax[0, 0].get_ylim()[0] + np.diff(ax[0, 0].get_ylim()) * 0.9
    ax[0, 0].text(
        0.1,
        y_pos,
        f"|uv| = {bl_norm[bl1]:.0f} m",
        fontsize=14,
        bbox={"facecolor": "white", "boxstyle": "round", "pad": 0.3, "edgecolor": "k"},
    )
    # ax[0,0].legend(loc="center left")
    # ax[0, 0].legend()

    ax[0, 1].plot(time, np.abs(xds.vis_rfi.data[:, bl2]), label="Truth")
    ax[0, 1].plot(
        time, np.abs(xds_tab.rfi_vis.data[0, bl2]), alpha=0.5, label="tabascal"
    )
    ax[0, 1].tick_params(axis="x", labelbottom=False)
    ax[0, 1].tick_params(
        axis="y", labelleft=False, labelright=True, left=False, right=True
    )
    # ax[0,1].tick_params(axis='y')
    y_pos = ax[0, 1].get_ylim()[0] + np.diff(ax[0, 1].get_ylim()) * 0.90
    ax[0, 1].text(
        0.1,
        y_pos,
        f"|uv| = {bl_norm[bl2]:.0f} m",
        fontsize=14,
        bbox={"facecolor": "white", "boxstyle": "round", "pad": 0.3, "edgecolor": "k"},
    )
    # ax[0, 1].legend(loc="center left")
    # ax[0,1].legend()

    ax[1, 0].plot(time, np.abs(xds.vis_ast.data[:, bl1]), label="Truth")
    ax[1, 0].plot(
        time, np.abs(xds_tab.ast_vis.data[0, bl1]), alpha=0.5, label="tabascal"
    )
    ax[1, 0].set_xlabel("Time [min]")
    ax[1, 0].set_ylabel("AST Visibility Magnitude [Jy]")
    y_pos = ax[1, 0].get_ylim()[0] + np.diff(ax[1, 0].get_ylim()) * 0.9
    ax[1, 0].text(
        0.1,
        y_pos,
        f"|uv| = {bl_norm[bl1]:.0f} m",
        fontsize=14,
        bbox={"facecolor": "white", "boxstyle": "round", "pad": 0.3, "edgecolor": "k"},
    )
    # ax[1,0].legend(loc="center left")
    # ax[1, 0].legend()

    ax[1, 1].plot(time, np.abs(xds.vis_ast.data[:, bl2]), label="Truth")
    ax[1, 1].plot(
        time, np.abs(xds_tab.ast_vis.data[0, bl2]), alpha=0.5, label="tabascal"
    )
    ax[1, 1].set_xlabel("Time [min]")
    # ax[1,1].set_ylabel("AST Visibility Magnitude [Jy]", )
    ax[1, 1].tick_params(
        axis="y", labelleft=False, labelright=True, left=False, right=True
    )
    y_pos = ax[1, 1].get_ylim()[0] + np.diff(ax[1, 1].get_ylim()) * 0.90
    ax[1, 1].text(
        0.1,
        y_pos,
        f"|uv| = {bl_norm[bl2]:.0f} m",
        fontsize=14,
        bbox={"facecolor": "white", "boxstyle": "round", "pad": 0.3, "edgecolor": "k"},
    )
    # ax[1,1].legend(loc="center left")
    # ax[1, 1].legend()

    plt.subplots_adjust(wspace=0, hspace=-0.03)  # figsize=(11.5,12)

    for a in ax.flatten():
        a.grid()
        a.legend("lower right")

    plt.savefig(
        os.path.join(data_dir, f"plots/Vis_Ex_RFI_{mean_rfi:.2e}{suffix}.pdf"),
        format="pdf",
        dpi=200,
        bbox_inches="tight",
    )


def plot_vis_ex_multi(zarr_file, tab_files, data_dir, bl1=3e1, bl2=1e3, suffix=""):

    plt.rcParams["font.size"] = 14

    xds = xr.open_zarr(zarr_file)
    xds_tab = [xr.open_zarr(tab_file) for tab_file in tab_files]
    n_tab = len(tab_files)

    mean_rfi = np.mean(np.abs(xds.vis_rfi.data)).compute()

    bl_norm = np.linalg.norm(xds.bl_uvw.data[0, :, :-1], axis=-1).compute()

    bl1 = np.argmin(np.abs(bl_norm - bl1))
    bl2 = np.argmin(np.abs(bl_norm - bl2))

    time = xds.time.data / 60
    fig, ax = plt.subplots(n_tab + 1, 2, figsize=(7, 11))

    ax[0, 0].plot(time, np.abs(xds.vis_rfi.data[:, bl1]), label="Truth")
    ax[0, 0].plot(
        time, np.abs(xds_tab[0].rfi_vis.data[0, bl1]), ".-", alpha=0.5, label="tabascal"
    )
    ax[0, 0].set_ylabel("$|V^\\text{RFI}|$ [Jy]")  # , fontsize=11)
    ax[0, 0].tick_params(axis="x", labelbottom=False)
    # y_pos = ax[0, 0].get_ylim()[0] + np.diff(ax[0, 0].get_ylim()) * 0.9
    # ax[0, 0].text(
    #     0.1,
    #     y_pos,
    #     f"|uv| = {bl_norm[bl1]:.0f} m",
    #     fontsize=14,
    #     bbox={"facecolor": "white", "boxstyle": "round", "pad": 0.3, "edgecolor": "k"},
    # )
    ax[0, 0].set_title(f"|uv| = {bl_norm[bl1]:.0f} m")

    ax[0, 1].plot(time, np.abs(xds.vis_rfi.data[:, bl2]), label="Truth")
    ax[0, 1].plot(
        time, np.abs(xds_tab[0].rfi_vis.data[0, bl2]), ".-", alpha=0.5, label="tabascal"
    )
    ax[0, 1].tick_params(axis="x", labelbottom=False)
    ax[0, 1].tick_params(
        axis="y", labelleft=False, labelright=True, left=False, right=True
    )
    # y_pos = ax[0, 1].get_ylim()[0] + np.diff(ax[0, 1].get_ylim()) * 0.90
    # ax[0, 1].text(
    #     0.1,
    #     y_pos,
    #     f"|uv| = {bl_norm[bl2]:.0f} m",
    #     fontsize=14,
    #     bbox={"facecolor": "white", "boxstyle": "round", "pad": 0.3, "edgecolor": "k"},
    # )
    ax[0, 1].set_title(f"|uv| = {bl_norm[bl2]:.0f} m")

    y_range1 = [
        1.1 * np.min([np.min(np.real(tab.ast_vis.data[0, bl1])) for tab in xds_tab]),
        1.1 * np.max([np.max(np.real(tab.ast_vis.data[0, bl1])) for tab in xds_tab]),
    ]
    y_range2 = [
        1.1 * np.min([np.min(np.real(tab.ast_vis.data[0, bl2])) for tab in xds_tab]),
        1.1 * np.max([np.max(np.real(tab.ast_vis.data[0, bl2])) for tab in xds_tab]),
    ]

    k0 = [-2, -3, -4]

    for i in range(n_tab):

        # Left Panel
        ax[i + 1, 0].plot(time, np.real(xds.vis_ast.data[:, bl1]), label="Truth")
        ax[i + 1, 0].plot(
            time,
            np.real(xds_tab[i].ast_vis.data[0, bl1]),
            ".-",
            alpha=0.5,
            label="tabascal",
        )
        ax[i + 1, 0].set_ylim(*y_range1)
        ax[i + 1, 0].set_ylabel("$Re(V^\\text{AST})$ [Jy]")  # , fontsize=12)
        ax[i + 1, 0].tick_params(axis="x", labelbottom=False, bottom=False)
        y_pos = ax[i + 1, 0].get_ylim()[0] + np.diff(ax[i + 1, 0].get_ylim()) * 0.9
        # ax[i + 1, 0].text(
        #     0.1,
        #     y_pos,
        #     f"|uv| = {bl_norm[bl1]:.0f} m",
        #     fontsize=14,
        #     bbox={
        #         "facecolor": "white",
        #         "boxstyle": "round",
        #         "pad": 0.3,
        #         "edgecolor": "k",
        #     },
        # )
        ax[i + 1, 0].text(
            7.5,
            y_pos,
            "$k_0 = 10^{" + f"{k0[i]}" + "}$ Hz",
            fontsize=14,
            horizontalalignment="center",
            bbox={
                "facecolor": "white",
                "boxstyle": "round",
                "pad": 0.3,
                "edgecolor": "k",
            },
        )

        # Right Panel
        ax[i + 1, 1].plot(time, np.real(xds.vis_ast.data[:, bl2]), label="Truth")
        ax[i + 1, 1].plot(
            time,
            np.real(xds_tab[i].ast_vis.data[0, bl2]),
            ".-",
            alpha=0.5,
            label="tabascal",
        )
        ax[i + 1, 1].set_ylim(*y_range2)
        ax[i + 1, 1].tick_params(
            axis="y", labelleft=False, labelright=True, left=False, right=True
        )
        ax[i + 1, 1].tick_params(axis="x", labelbottom=False, bottom=False)
        y_pos = ax[i + 1, 1].get_ylim()[0] + np.diff(ax[i + 1, 1].get_ylim()) * 0.9
        # ax[i + 1, 1].text(
        #     0.1,
        #     y_pos,
        #     f"|uv| = {bl_norm[bl2]:.0f} m",
        #     fontsize=14,
        #     bbox={
        #         "facecolor": "white",
        #         "boxstyle": "round",
        #         "pad": 0.3,
        #         "edgecolor": "k",
        #     },
        # )
        ax[i + 1, 1].text(
            7.5,
            y_pos,
            "$k_0 = 10^{" + f"{k0[i]}" + "}$ Hz",
            fontsize=14,
            horizontalalignment="center",
            bbox={
                "facecolor": "white",
                "boxstyle": "round",
                "pad": 0.3,
                "edgecolor": "k",
            },
        )

    ax[-1, 0].tick_params(axis="x", labelbottom=True)
    ax[-1, 1].tick_params(axis="x", labelbottom=True)
    ax[-1, 0].set_xlabel("Time [min]")
    ax[-1, 1].set_xlabel("Time [min]")
    plt.subplots_adjust(wspace=0, hspace=-0.03)  # figsize=(11.5,12)
    plt.subplots_adjust(wspace=0.03, hspace=0.0)  # figsize=(11.5,12)
    # plt.subplots_adjust(wspace=0.0, hspace=0.0)  # figsize=(11.5,12)

    for a in ax.flatten():
        a.grid()
        a.legend(loc="lower right")

    plt.savefig(
        os.path.join(data_dir, f"plots/Vis_Ex_Multi_RFI_{mean_rfi:.2e}{suffix}.pdf"),
        format="pdf",
        dpi=200,
        bbox_inches="tight",
    )
    print("Saved")


def extract_ms_data(ms_path: str, name: str) -> dict:

    xds = xds_from_ms(ms_path)[0]

    if name == "flag2":
        flags = np.invert(xds["AO_FLAGS"].data[:, 0, 0].compute().astype(bool))
        data_col = "CAL_DATA"
    elif name == "flag1":
        # flags = np.invert(xds["3S_FLAGS"].data[:, 0, 0].compute().astype(bool))
        flags = np.where(
            np.abs(xds.CAL_DATA.data[:, 0, 0] - xds.AST_MODEL_DATA.data[:, 0, 0])
            < 3 * np.sqrt(2) * xds.NOISE_DATA.data[0, 0, 0]
        )[0]
        data_col = "CAL_DATA"
    elif name == "ideal":
        flags = np.arange(xds.DATA.data.shape[0])
        data_col = "AST_DATA"
    elif name == "tab":
        flags = np.arange(xds.DATA.data.shape[0])
        data_col = "TAB_DATA"
    elif name == "perfect":
        flags = np.arange(xds.DATA.data.shape[0])
        data_col = "AST_MODEL_DATA"
    else:
        print(f"Invalid name : {name} given.")

    data = {
        "dish_d": xds_from_table(ms_path + "::ANTENNA")[0]
        .DISH_DIAMETER.data[0]
        .compute(),
        "freq": xds_from_table(ms_path + "::SPECTRAL_WINDOW")[0]
        .CHAN_FREQ.data[0, 0]
        .compute(),
        "vis": xds[data_col].data[flags, 0, 0].compute(),
        "uv": xds.UVW.data[flags, :2].compute(),
        "noise_std": xds.SIGMA.data[flags].mean().compute(),
    }

    return data


def calculate_pow_spec(
    data: dict,
    n_grid: int = 256,
    n_bins: int = 10,
):

    tge_ps = TGE(dish_d=data["dish_d"], ref_freq=data["freq"], f=1, N_grid_x=n_grid)

    l_b, Cl_b_norm, d_Cl_b_norm = tge_ps.estimate_Cl(
        uv=data["uv"] / tge_ps.lamda,
        V=data["vis"],
        sigma_n=data["noise_std"],
        n_bins=n_bins,
    )

    l = np.logspace(np.log10(np.nanmin(l_b) / 2), np.log10(np.nanmax(l_b) * 2))
    norm = l * (l + 1) / (2 * np.pi)
    C_l_norm = norm * Cl(l)

    xds = xr.Dataset(
        {
            # Gridded variables
            "U_g": (("uv_grid"), da.asarray(tge_ps.U_g.flatten())),
            "V_cg": (("uv_grid"), da.asarray(tge_ps.V_cg)),
            "K1g": (("uv_grid"), da.asarray(tge_ps.K1g)),
            "B_cg": (("uv_grid"), da.asarray(tge_ps.B_cg)),
            "K2gg": (("uv_grid"), da.asarray(tge_ps.K2gg)),
            "E_g": (("uv_grid"), da.asarray(tge_ps.E_g)),
            # Binned variables
            "U_b": (("l_bin"), da.asarray(tge_ps.U_b)),
            "l_b": (("l_bin"), da.asarray(tge_ps.l_b)),
            "Cl_norm": (("l_bin"), da.asarray(tge_ps.Cl_norm)),
            "Cl_b": (("l_bin"), da.asarray(tge_ps.Cl_b)),
            "delta_Cl_b": (("l_bin"), da.asarray(tge_ps.delta_Cl_b)),
            "Cl_b_normed": (("l_bin"), da.asarray(Cl_b_norm)),
            "d_Cl_b_normed": (("l_bin"), da.asarray(d_Cl_b_norm)),
            "l": (("l_fine"), da.asarray(l)),
            "Cl_normed": (("l_fine"), da.asarray(C_l_norm)),
        }
    )

    return xds


def extract_pow_spec_all(ms_files, SNR, bin_idx, names, data_dir):

    ps_dir = os.path.join(data_dir, "plots")
    n_rfi_bins = len(bin_idx)

    for i, idx in tqdm(enumerate(bin_idx)):
        xdt = xr.DataTree(name=f"rfi_bin_{i:02}")
        print(f"Calculating PS from : {len(idx)} datasets")
        ps_name = f"ps_rfi_{i:02}_{n_rfi_bins:02}.zarr"
        print(f"Writing : {ps_name}")
        for name in tqdm(names["options"]):
            if len(idx) > 0:
                data = [extract_ms_data(ms, name) for ms in ms_files[idx]]
                data = {
                    key: np.concatenate([np.atleast_1d(d[key]) for d in data])
                    for key in data[0].keys()
                }
                for key in ["dish_d", "freq", "noise_std"]:
                    data[key] = data[key][0]
                xds = calculate_pow_spec(data)
                xdt[name] = xds

        xdt["SNR"] = np.mean(SNR[idx])
        xdt.to_zarr(os.path.join(ps_dir, ps_name), mode="w")

    return ps_dir


def plot_pow_spec(ps_file, names, SNR, ps_dir):

    xdt = xr.open_datatree(ps_file, engine="zarr")
    ps_name = os.path.splitext(os.path.split(ps_file)[1])[0]

    plot_name = f"PS_Recovery_{ps_name}.pdf"
    i = int(ps_name.split("_")[-2])

    plt.figure(figsize=(15, 11))
    for name in names["options"]:
        xds = xdt[name]
        if name in ["ideal", "tab", "flag1", "flag2"]:
            marker = "o" if name == "ideal" else "."
            capsize = 10 if name == "ideal" else 5
            if name == "ideal":
                plt.loglog(xds["l"], xds["Cl_normed"] * 1e6, "k", label="$C_l$")
            plt.errorbar(
                xds["l_b"],
                xds["Cl_b_normed"] * 1e6,
                yerr=xds["d_Cl_b_normed"] * 1e6,
                fmt=marker,
                label=names["names"][name],
                color=names["colors"][name],
                capsize=capsize,
            )

    plt.title("Recovered Power Spectrum : SNR($|V^{RFI}|$) Amp = " + f"{SNR[i]:.2f} Jy")
    plt.legend()
    plt.xlabel("l")
    plt.ylabel("l(l+1)C$_l$/2$\\pi$ [mK$^2$]")
    # plt.ylim(1e7, 5e9)
    plt.ylim(1e6, 5e11)
    plt.xlim(1e2, 1.5e4)
    plt.savefig(
        os.path.join(ps_dir, plot_name),
        format="pdf",
        dpi=200,
        bbox_inches="tight",
    )


def plot(
    data_dir: str,
    n_bins: int,
    n_hist_bins: int,
    n_sigma: float,
    plots: dict,
    rfi_amps: list,
    vbounds: list,
    vbounds_flux: list,
    ps_dir: str,
    tab_suffix: str = "",
    data_col: str = "DATA",
    model_name: str = "map_pred_fixed_orbit_rfi_full_fft_standard_padded_model",
):
    if tab_suffix:
        tab_suffix = "_" + tab_suffix

    files, data = get_file_names(data_dir, model_name, tab_suffix, data_col)

    # print(data["max_rfi"].max())

    # import sys

    # sys.exit(0)

    names = get_names(tab_suffix=tab_suffix)

    noise_std = np.mean(data["vis_noise"])
    SNR = data["mean_rfi"] / data["vis_noise"]
    bin_idx = get_bins(SNR, n_bins)

    if plots["gains"]:
        percentiles = [5, 50, 95]
        binned_snr = [np.median(SNR[idx]) for idx in bin_idx]
        binned_errors = get_binned_gain_errors(
            files["zarr_files"], files["tab_files"], bin_idx, percentiles, data_col
        )

        plot_gain_errors(binned_snr, binned_errors, data_dir, tab_suffix)

        binned_bias = get_binned_gain_bias(
            files["zarr_files"], files["fish_files"], bin_idx, percentiles, data_col
        )

        plot_gain_bias(binned_snr, binned_bias, data_dir, tab_suffix)

        binned_bias = get_binned_gain_std(
            files["fish_files"], bin_idx, percentiles, data_col
        )

        plot_gain_std(binned_snr, binned_bias, data_dir, tab_suffix)

    #############################################################
    # Visibility Errors
    #############################################################

    if plots["error"]:

        plot_errors_all(
            files, bin_idx, SNR, noise_std, n_hist_bins, data_dir, tab_suffix
        )

        # plot_tab_errors(
        #     files["zarr_files"],
        #     files["tab_files"],
        #     bin_idx,
        #     SNR,
        #     noise_std,
        #     n_hist_bins,
        #     data_dir,
        # )

        # plot_perfect_flag_errors(
        #     files["zarr_files"], bin_idx, SNR, noise_std, n_hist_bins, data_dir
        # )

        # plot_aoflagger_errors(
        #     files["zarr_files"],
        #     files["ms_files"],
        #     bin_idx,
        #     SNR,
        #     noise_std,
        #     n_hist_bins,
        #     data_dir,
        # )

        # plot_vis_cal_errors(
        #     files["zarr_files"], bin_idx, SNR, noise_std, n_hist_bins, data_dir
        # )

    if plots["img"]:
        keys = ["ideal", "tab", "flag1", "flag2"]
        img_names = {key: names["img_names"][key] for key in keys}
        for rfi_amp in rfi_amps:
            try:
                plot_image(
                    files["img_dirs"],
                    img_names,
                    names["names"],
                    keys,
                    data["mean_rfi"],
                    rfi_amp,
                    data_dir,
                    vbounds,
                    log=False,
                    suffix=tab_suffix,
                )
            except:
                print(f"Unable to create image for RFI amp closest to {rfi_amp}")

    if plots["vis"]:
        for rfi_amp in rfi_amps:
            idx = np.argmin(np.abs(data["mean_rfi"] - rfi_amp))
            plot_vis_ex(
                files["zarr_files"][idx],
                files["tab_files"][idx],
                data_dir,
                bl1=3e1,
                bl2=1e3,
                suffix=tab_suffix,
            )
            suffixes = [tab_suffix, "_k0_1e-3", "_k0_1e-4"]
            tab_files = 3 * [
                files["tab_files"][idx],
            ]
            tab_files = [
                tab_file.replace(tab_suffix, suffix)
                for tab_file, suffix in zip(tab_files, suffixes)
            ]
            # plot_vis_ex_multi(
            #     files["zarr_files"][idx],
            #     tab_files,
            #     data_dir,
            #     bl1=3e1,
            #     bl2=1e3,
            #     suffix=tab_suffix,
            # )

    # tab_files = [
    #     "../../data/pnt_src_32A/pnt_src_obs_32A_450T-0000-0898_1025I_001F-1.227e+09-1.227e+09_100PAST_000GAST_000EAST_6SAT_0GRD_3.0e+01RFI_78RSEED/results/map_pred_fixed_orbit_rfi_full_fft_standard_padded_model_k0_1e-2.zarr",
    #     "../../data/pnt_src_32A/pnt_src_obs_32A_450T-0000-0898_1025I_001F-1.227e+09-1.227e+09_100PAST_000GAST_000EAST_6SAT_0GRD_3.0e+01RFI_78RSEED/results/map_pred_fixed_orbit_rfi_full_fft_standard_padded_model_k0_1e-3.zarr",
    #     "../../data/pnt_src_32A/pnt_src_obs_32A_450T-0000-0898_1025I_001F-1.227e+09-1.227e+09_100PAST_000GAST_000EAST_6SAT_0GRD_3.0e+01RFI_78RSEED/results/map_pred_fixed_orbit_rfi_full_fft_standard_padded_model_k0_1e-4.zarr",
    # ]
    # zarr_file = "../../data/pnt_src_32A/pnt_src_obs_32A_450T-0000-0898_1025I_001F-1.227e+09-1.227e+09_100PAST_000GAST_000EAST_6SAT_0GRD_3.0e+01RFI_78RSEED/pnt_src_obs_32A_450T-0000-0898_1025I_001F-1.227e+09-1.227e+09_100PAST_000GAST_000EAST_6SAT_0GRD_3.0e+01RFI_78RSEED.zarr"
    # plot_vis_ex_multi(
    #     zarr_file,
    #     tab_files,
    #     data_dir,
    #     bl1=3e1,
    #     bl2=1e3,
    #     suffix=tab_suffix,
    # )

    #############################################################
    # Point Source Recovery Statistics
    #############################################################

    if plots["src"]:
        n_true = np.array(
            [
                len(pd.read_csv(os.path.join(d, "true_sources.csv")))
                for d in files["img_dirs"]
            ]
        )

        base_stats = {
            "mean_ast": data["mean_ast"],
            "mean_rfi": data["mean_rfi"],
            "vis_noise": data["vis_noise"],
            "im_noise_theory": data["im_noise_theory"],
            "P": n_true,
        }

        all_stats = {
            name: get_stats(
                base_stats,
                names["names"][name],
                files["img_dirs"],
                names["img_names"][name],
                n_sigma,
            )
            for name in names["options"]
        }

        percentiles = [2.5, 50, 97.5]
        percentiles = [16, 50, 84]

        all_q1 = {
            name: bin_stats(
                stat,
                n_bins,
                lambda x: np.percentile(x, percentiles[0]),
                data["mean_rfi"],
            )
            for name, stat in all_stats.items()
        }
        all_med = {
            name: bin_stats(
                stat,
                n_bins,
                lambda x: np.percentile(x, percentiles[1]),
                data["mean_rfi"],
            )
            for name, stat in all_stats.items()
        }
        all_q2 = {
            name: bin_stats(
                stat,
                n_bins,
                lambda x: np.percentile(x, percentiles[2]),
                data["mean_rfi"],
            )
            for name, stat in all_stats.items()
        }
        print()
        plot_completeness(all_med, all_q1, all_q2, names, data_dir, tab_suffix)
        plot_purity(all_med, all_q1, all_q2, names, data_dir, tab_suffix)
        plot_image_noise(all_med, all_q1, all_q2, names, data_dir, tab_suffix)
        plot_flux_error(
            all_med, all_q1, all_q2, names, data_dir, vbounds_flux, tab_suffix
        )
        plot_abs_flux_error(
            all_med, all_q1, all_q2, names, data_dir, vbounds_flux, tab_suffix
        )

        # plot_flag_noise(files, noise_std, data, all_stats, names, data_dir)
        # plot_theoretical_noise(files, noise_std, data, all_stats, names, data_dir)

    ###########################################
    # Plot Power Spectrum Recovery
    ###########################################

    if plots["pow_spec"]:
        if ps_dir:
            ps_files = glob(os.path.join(ps_dir, "ps_rfi_*"))
        else:
            ps_dir = extract_pow_spec_all(
                files["ms_files"], SNR, bin_idx, names, data_dir
            )
            ps_files = glob(os.path.join(ps_dir, "ps_rfi_*"))
        for ps_file in ps_files:
            snr = [np.mean(SNR[idx]) for idx in bin_idx]
            plot_pow_spec(ps_file, names, snr, ps_dir)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyse tabascal simulations, recoveries and point source extractions."
    )
    parser.add_argument(
        "-d",
        "--data_dir",
        required=True,
        help="Path to the data directory containing all the simulations.",
    )
    parser.add_argument(
        "-dc",
        "--data_col",
        default="DATA",
        help="Data columns name in the MS file. Default is 'DATA'.",
    )
    parser.add_argument(
        "-b", "--n_bins", default=5, type=int, help="Number of RFI bins. Default is 5."
    )
    parser.add_argument(
        "-hb",
        "--hist_bins",
        default=100,
        type=int,
        help="Number of histogram bins in error plots. Default is 100.",
    )
    parser.add_argument(
        "-n",
        "--n_sigma",
        default=3.0,
        type=float,
        help="Number of sigma of the image noise to consider a detection. Default is 3.",
    )
    parser.add_argument(
        "-p",
        "--plots",
        default="gains,error,vis,img,src",
        help="Which plots to create. Default is 'error,img,src'. Options are {'error', 'vis', 'img', 'src', 'pow_spec'}",
    )
    parser.add_argument(
        "-v",
        "--vbounds",
        default="-15,15",
        help="Bounds on the image plot.",
    )
    parser.add_argument(
        "-vf",
        "--vbounds_flux",
        default="-15,15",
        help="Bounds on the flux error plot.",
    )
    parser.add_argument(
        "-r",
        "--rfi_amps",
        default="100",
        help="RFI amplitude for image plot.",
    )
    parser.add_argument(
        "-ps",
        "--ps_dir",
        default=None,
        help="Directory with precalculated power spectrum results.",
    )
    parser.add_argument(
        "-sx",
        "--suffix",
        default="",
        help="Data suffix.",
    )
    args = parser.parse_args()
    data_dir = args.data_dir
    plots_types = args.plots.split(",")
    vbounds = [float(x) for x in args.vbounds.split(",")]
    vbounds_flux = [float(x) for x in args.vbounds_flux.split(",")]
    rfi_amps = [float(x) for x in args.rfi_amps.split(",")]

    os.makedirs(os.path.join(data_dir, "plots"), exist_ok=True)

    plots = {
        "gains": False,
        "error": False,
        "vis": False,
        "img": False,
        "src": False,
        "pow_spec": False,
    }

    for plot_type in plots_types:
        plots[plot_type] = True

    plot(
        data_dir,
        args.n_bins,
        args.hist_bins,
        args.n_sigma,
        plots,
        rfi_amps,
        vbounds,
        vbounds_flux,
        args.ps_dir,
        tab_suffix=args.suffix,
        data_col=args.data_col,
    )


if __name__ == "__main__":

    main()
