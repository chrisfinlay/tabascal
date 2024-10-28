from tge import TGE, Cl
from daskms import xds_from_ms, xds_from_table
import xarray as xr
import argparse
import numpy as np
import matplotlib.pyplot as plt
import os

from tabascal.utils.yaml import load_config

def extract_ms_data(ms_path: str, data_col: str) -> tuple:

    xds = xds_from_ms(ms_path)[0]
    flags = np.astype(xds.FLAG.data[:,0,0].compute(), bool)
    data = {
        "dish_d": xds_from_ms(ms_path+"::ANTENNA")[0].DISH_DIAMETER.data.compute(),
        "freq": xds_from_table(ms_path+"::SPECTRAL_WINDOW").CHAN_FREQ.data.compute(),
        "vis": xds[data_col].data[not flags].compute(),
        "uv": xds.UVW.data[not flags,:2].compute(),
        "noise_std": xds.SIGMA.data[not flags].mean().compute()
    }

    return data

def extract_pow_spec(zarr_path: str=None, ms_path: str=None, data_col: str="TAB_DATA", n_grid: int=256, n_bins: int=20, suffix: str=""): 

    if suffix is not "":
        suffix = "_" + suffix

    sim_dir = os.path.split(ms_path)[0]

    data = extract_ms_data(ms_path, data_col)

    tge_ps = TGE(dish_d=data["dish_d"], ref_freq=data["freq"], f=1, N_grid_x=n_grid)

    l_b, Cl_b_norm, d_Cl_b_norm = tge_ps.estimate_Cl(
        uv=data["uv"]/tge_ps.lamda, 
        V=data["vis"], 
        sigma_n=data["noise_std"], 
        n_bins=n_bins
        )
    
    l = np.logspace(np.log10(np.nanmin(l_b)/2), np.log10(np.nanmax(l_b)*2))
    norm = l * (l+1) / (2*np.pi)
    C_l_norm = norm*Cl(l)

    plt.figure(figsize=(10,7))

    plt.loglog(l, C_l_norm*1e6, 'k', label="$C_l$")
    plt.errorbar(l_b, Cl_b_norm*1e6, yerr=d_Cl_b_norm*1e6, fmt='.', color="tab:orange")

    plt.title("Recovered Power Spectrum")
    plt.xlabel("l")
    plt.ylabel("l(l+1)C$_l$/2$\\pi$ [mK$^2$]")
    plt.ylim(1e7, 5e9)
    plt.xlim(1e2, 1.5e4)
    plt.savefig(os.path.join(sim_dir, f"plots/PS_Recovery_{data_col}{suffix}.pdf"), format="pdf", dpi=200)

def main():
    parser = argparse.ArgumentParser(
        description="Calculate angular power spectrum using the tapered gridded estimator."
    )
    parser.add_argument(
        "-z", "--zarr_path", default=None, help="Path to the zarr simulation file."
    )
    parser.add_argument(
        "-m", "--ms_path", defualt=None, help="Path to the MS file."
    )
    parser.add_argument(
        "-c", "--config", required=True, help="Path to the config file."
    )
    args = parser.parse_args()
    zarr_path = args.zarr_path
    ms_path = args.ms_path
    conf_path = args.config   

    config = load_config(conf_path, config_type="pow_spec")

    ps_conf = config["tge"]

    for data_type in config.keys():
        conf = config[data_type]
        if conf["suffix"] is None:
            conf["suffix"] = ""
        extract_pow_spec(zarr_path, ms_path, conf["data_col"], ps_conf["n_grid"], ps_conf["n_bins"], conf["suffix"]) 


if __name__=="__main__":
    main()