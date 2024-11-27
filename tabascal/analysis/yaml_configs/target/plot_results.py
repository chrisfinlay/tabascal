import numpy as np
import pandas as pd
from astropy.io import fits
import matplotlib.pyplot as plt

from daskms import xds_from_ms
import xarray as xr

from glob import glob
import os

plt.rcParams["font.size"] = 16


def get_stats(stats_base, img_dirs, img_name):

    n_dir = len(img_dirs.flatten())
    shape = img_dirs.shape

    TP = np.zeros(n_dir)
    im_noise = np.nan * np.zeros(n_dir)
    mean_I_error = np.nan * np.zeros(n_dir)
    std_I_error = np.nan * np.zeros(n_dir)
    det = np.zeros(n_dir)

    for i, img_dir in enumerate(img_dirs.flatten()):
        img_path = os.path.join(img_dir, f"{img_name}.csv")
        bdsf_path = os.path.join(img_dir, f"{img_name}.pybdsf.csv")

        try:
            df = pd.read_csv(img_path)
            n_src = len(df)
            if n_src > 0:
                TP[i] = n_src
                im_noise[i] = df["Image_noise [mJy/beam]"].values[0]
                mean_I_error[i] = np.mean(
                    df["Total_flux_image [mJy]"] - df["Total_flux_true [mJy]"]
                )
                std_I_error[i] = np.mean(df["Total_flux_std [mJy]"])
        except:
            print(f"No source extraction file found at {img_path}")

        try:
            bdsf_df = pd.read_csv(bdsf_path, skiprows=5)
            if len(bdsf_df) > 0:
                det[i] = np.sum(bdsf_df[" Total_flux"] > 3e-3 * im_noise[i])
        except:
            print(f"No pyBDSF file found at {bdsf_path}")

    stats = {
        "TP": TP,
        "im_noise": im_noise,
        "mean_I_error": mean_I_error,
        "std_I_error": std_I_error,
        "det": det,
    }
    stats = {key: val.reshape(*shape) for key, val in stats.items()}
    stats.update(stats_base)

    stats["FP"] = stats["det"] - stats["TP"]
    stats["FN"] = stats["P"] - stats["TP"]
    # stats["TN"] = unknown
    # stats["N"] = stats["FP"] + stats["TN"] => unknown
    stats["TPR"] = stats["TP"] / stats["P"]
    stats["FNR"] = stats["FN"] / stats["P"]
    # stats["FPR"] = stats["FP"] / stats["N"]
    # stats["TNR"] = stats["TN"] / stats["N"]
    stats["prec"] = stats["TP"] / (stats["TP"] + stats["FP"])  # Purity
    stats["rec"] = stats["TP"] / (stats["TP"] + stats["FN"])  # Completeness

    stats["SNR_RFI"] = stats["mean_rfi"] / stats["vis_noise"]
    stats["RFI/AST"] = stats["mean_rfi"] / stats["mean_ast"]

    return stats


def bin_stats(stats, n_bins, bin_stat, bin_array):

    bins = np.logspace(
        np.log10(np.min(bin_array)), np.log10(np.max(bin_array) + 1), n_bins + 1
    )
    bin_idx = [
        np.where((bin_array >= bins[i]) & (bin_array < bins[i + 1]))[0]
        for i in range(n_bins)
    ]

    binned = {
        key: np.array([bin_stat(value[idx]) for idx in bin_idx])
        for key, value in stats.items()
    }

    return binned


suffix = ""
ideal_name = f"AST_DATA_0.0sigma{suffix}"
tab_name = f"TAB_DATA_0.0sigma{suffix}"
flag1_name = f"CAL_DATA_3.0sigma{suffix}"
flag2_name = f"CAL_DATA_aoflagger{suffix}"

options = [
    "ideal",
    "tab",
    "flag1",
    "flag2",
]

names = {
    "ideal": "Uncontaminated",
    "tab": "TABASCAL",
    "flag1": "Perfect Flagging",
    "flag2": "AOFlagger",
}

colors = {
    "ideal": "tab:blue",
    "tab": "tab:orange",
    "flag1": "tab:red",
    "flag2": "tab:green",
}

img_names = {
    "ideal": ideal_name,
    "tab": tab_name,
    "flag1": flag1_name,
    "flag2": flag2_name,
}


def main(data_dir, n_bins, bin_stat):

    data_dirs = np.array(glob(os.path.join(data_dir, "*")))

    data_dirs = np.array([d for d in data_dirs if "plots" not in d])

    img_dirs = np.array([os.path.join(d, "images") for d in data_dirs])

    ms_files = np.array(
        [os.path.join(d, os.path.split(d)[1] + ".ms") for d in data_dirs]
    )
    zarr_files = np.array(
        [os.path.join(d, os.path.split(d)[1] + ".zarr") for d in data_dirs]
    )

    n_sim = len(data_dirs)

    mean_rfi = np.zeros(n_sim)
    mean_ast = np.zeros(n_sim)
    vis_noise = np.zeros(n_sim)
    flags1 = np.zeros(n_sim)
    flags2 = np.zeros(n_sim)

    for i, ms in enumerate(ms_files):
        xds_ms = xds_from_ms(ms)[0]
        xds = xr.open_zarr(zarr_files[i])
        mean_rfi[i] = np.abs(xds_ms.RFI_MODEL_DATA.data).mean().compute()
        mean_ast[i] = np.abs(xds_ms.AST_MODEL_DATA.data).mean().compute()
        flags1[i] = np.abs(xds.flags).mean().compute()
        flags2[i] = np.abs(xds_ms.FLAG.data).mean().compute()
        vis_noise[i] = np.std(xds_ms.NOISE_DATA.data.real).compute()

    n_true = np.array(
        [len(pd.read_csv(os.path.join(d, "true_sources.csv"))) for d in img_dirs]
    )

    base_stats = {
        "mean_ast": mean_ast,
        "mean_rfi": mean_rfi,
        "vis_noise": vis_noise,
        "P": n_true,
    }

    all_stats = {
        name: get_stats(base_stats, img_dirs, img_names[name]) for name in options
    }

    all_med = {
        name: bin_stats(stat, n_bins, bin_stat, mean_rfi)
        for name, stat in all_stats.items()
    }

    ################################################################################
    # SNR vs Completeness
    ################################################################################

    plt.figure(figsize=(7, 6))
    fig, ax = plt.subplots(1, 1, figsize=(7, 6))

    for name, stat in all_med.items():
        marker = "o-" if name == "ideal" else ".-"
        ax.plot(
            stat["SNR_RFI"], stat["rec"], marker, label=names[name], color=colors[name]
        )

    ax.legend()
    ax.set_xlabel("SNR$(|V^{RFI}|)$")
    ax.set_ylabel("Completeness")
    ax.semilogx()

    ax2 = ax.twiny()
    ax2.plot(all_med["flag2"]["RFI/AST"], all_med["flag2"]["prec"], alpha=0)
    ax2.semilogx()
    ax2.set_xlabel("$|V^{RFI}| / |V^{AST}|$")

    plt.savefig(os.path.join(data_dir, "plots/Completeness.pdf"), format="pdf", dpi=200)

    # ################################################################################
    # # SNR vs Purity
    # ################################################################################

    plt.figure(figsize=(7, 6))
    fig, ax = plt.subplots(1, 1, figsize=(7, 6))

    for name, stat in all_med.items():
        marker = "o-" if name == "ideal" else ".-"
        ax.plot(
            stat["SNR_RFI"], stat["prec"], marker, label=names[name], color=colors[name]
        )

    ax.legend()
    ax.set_xlabel("SNR$(|V^{RFI}|)$")
    ax.set_ylabel("Purity")
    ax.semilogx()

    ax2 = ax.twiny()
    ax2.plot(all_med["flag2"]["RFI/AST"], all_med["flag2"]["prec"], alpha=0)
    ax2.semilogx()
    ax2.set_xlabel("$|V^{RFI}| / |V^{AST}|$")

    plt.savefig(os.path.join(data_dir, "plots/Purity.pdf"), format="pdf", dpi=200)

    ################################################################################
    # SNR vs Image Noise
    ################################################################################

    plt.figure(figsize=(7, 6))
    fig, ax = plt.subplots(1, 1, figsize=(7, 6))

    for name, stat in all_med.items():
        marker = "o-" if name == "ideal" else ".-"
        ax.plot(
            stat["SNR_RFI"],
            stat["im_noise"],
            marker,
            label=names[name],
            color=colors[name],
        )

    ax.legend()
    ax.set_xlabel("SNR$(|V^{RFI}|)$")
    ax.set_ylabel("Image Noise [mJy/beam]")
    ax.semilogx()
    ax.semilogy()

    ax2 = ax.twiny()
    ax2.plot(all_med["flag2"]["RFI/AST"], all_med["flag2"]["im_noise"], alpha=0)
    ax2.semilogx()
    ax2.set_xlabel("$|V^{RFI}| / |V^{AST}|$")

    plt.savefig(os.path.join(data_dir, "plots/ImageNoise.pdf"), format="pdf", dpi=200)

    ################################################################################
    # SNR vs Flux Error
    ################################################################################

    plt.figure(figsize=(7, 6))
    fig, ax = plt.subplots(1, 1, figsize=(7, 6))

    for name, stat in all_med.items():
        marker = "o-" if name == "ideal" else ".-"
        ax.errorbar(
            x=stat["SNR_RFI"],
            y=stat["mean_I_error"],
            yerr=stat["std_I_error"],
            fmt=marker,
            label=names[name],
        )

    ax.legend()
    ax.axhline(0, ls="--", color="k")
    ax.set_xlabel("SNR$(|V^{RFI}|)$")
    ax.set_ylabel("Flux Error [mJy]")
    ax.semilogx()
    # ax.xticks(10**np.arange(-1.0, 2.0, 1.0), 10**np.arange(-1.0, 2.0, 1.0))
    ax.set_ylim(-100, 100)

    ax2 = ax.twiny()
    ax2.plot(all_med["flag2"]["RFI/AST"], all_med["flag2"]["mean_I_error"], alpha=0)
    ax2.semilogx()
    ax2.set_xlabel("$|V^{RFI}| / |V^{AST}|$")

    plt.savefig(os.path.join(data_dir, "plots/FluxError.pdf"), format="pdf", dpi=200)

    ################################################################################
    # Flag Rate vs Image Noise
    ################################################################################

    flag_rate = np.linspace(0, 0.999, 100)
    ther_noise = (
        1e3 * vis_noise.mean() / np.sqrt((1 - flag_rate) * xds.n_time * xds.n_bl)
    )
    ther_noise1 = np.array(
        [
            1e3 * vis_noise.mean() / np.sqrt((1 - flag_rate) * xds.n_time * xds.n_bl)
            for flag_rate in flags1
        ]
    )
    ther_noise2 = np.array(
        [
            1e3 * vis_noise.mean() / np.sqrt((1 - flag_rate) * xds.n_time * xds.n_bl)
            for flag_rate in flags2
        ]
    )

    plt.figure(figsize=(7, 6))
    plt.plot(
        100 * flags1,
        all_stats["flag1"]["im_noise"],
        "o",
        label=names["flag1"],
        color=colors["flag1"],
    )
    plt.plot(
        100 * flags2,
        all_stats["flag2"]["im_noise"],
        "o",
        label=names["flag2"],
        color=colors["flag2"],
    )
    plt.plot(100 * flag_rate, ther_noise, "k--", label="Theoretical")
    plt.xlabel("Flag Rate [%]")
    plt.ylabel("Actual Image Noise [mJy]")
    plt.semilogy()
    plt.legend()

    plt.savefig(
        os.path.join(data_dir, "plots/ImageNoiseVsFlagRate.pdf"), format="pdf", dpi=200
    )

    ################################################################################
    # Theoretical Image Noise vs Image Noise
    ################################################################################

    sig_I = np.linspace(ther_noise1.min(), ther_noise1.max(), 100)

    plt.figure(figsize=(7, 6))
    plt.plot(
        ther_noise1,
        all_stats["flag1"]["im_noise"],
        "o",
        label=names["flag1"],
        color=colors["flag1"],
    )
    plt.plot(
        ther_noise2,
        all_stats["flag2"]["im_noise"],
        "o",
        label=names["flag2"],
        color=colors["flag2"],
    )
    plt.plot(sig_I, sig_I, "k--", label="Theoretical")
    plt.xlabel("Theoretical Image Noise [mJy]")
    plt.ylabel("Actual Image Noise [mJy]")
    plt.semilogy()
    plt.legend()

    plt.savefig(
        os.path.join(data_dir, "plots/ImageNoiseVsTheory.pdf"), format="pdf", dpi=200
    )


if __name__ == "__main__":

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
        "-b", "--n_bins", default=5, help="Number of RFI bins. Default is 5."
    )
    parser.add_argument(
        "-s",
        "--stat",
        default="median",
        help="Type of statistic. Options are {'median', 'mean'}. Default is 'median'.",
    )

    args = parser.parse_args()
    data_dir = args.data_dir
    n_bins = args.n_bins

    stat_opts = {
        "median": np.median,
        "mean": np.mean,
    }

    os.makedirs(os.path.join(data_dir, "plots"), exist_ok=True)

    main(data_dir, n_bins, stat_opts[args.stat])
