from tge import TGE, Cl
from daskms import xds_from_ms, xds_from_table
import xarray as xr
import dask.array as da
import argparse
import numpy as np
import matplotlib.pyplot as plt
import os

from tabascal.utils.config import load_config
from tabascal.utils.flag_data import write_perfect_flags, run_aoflagger

def extract_ms_data(ms_path: str, data_col: str) -> tuple:

    xds = xds_from_ms(ms_path)[0]
    flags = np.invert(xds.FLAG.data[:,0,0].compute().astype(bool))
    data = {
        "dish_d": xds_from_table(ms_path+"::ANTENNA")[0].DISH_DIAMETER.data.compute()[0],
        "freq": xds_from_table(ms_path+"::SPECTRAL_WINDOW")[0].CHAN_FREQ.data.compute()[0,0],
        "vis": xds[data_col].data[flags,0,0].compute(),
        "uv": xds.UVW.data[flags,:2].compute(),
        "noise_std": xds.SIGMA.data[flags].mean().compute()
    }

    return data

def extract_pow_spec(ms_path: str=None, data_col: str="TAB_DATA", n_grid: int=256, n_bins: int=20, suffix: str=""): 

    if suffix != "":
        suffix = "_" + suffix

    sim_dir = os.path.split(ms_path)[0]
    ps_dir = os.path.join(sim_dir, "power_spectrum")
    os.makedirs(ps_dir, exist_ok=True)

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
            "Cl_b": (("l_bin"), da.asarray(tge_ps.Cl_b)),
            "Cl_norm": (("l_bin"), da.asarray(tge_ps.Cl_norm)),
            "delta_Cl_b": (("l_bin"), da.asarray(tge_ps.delta_Cl_b)),
        }
    )
    xds.to_zarr(os.path.join(ps_dir, f"PS_results_{data_col}{suffix}.zarr"), mode="w")

    plt.figure(figsize=(10,7))

    plt.loglog(l, C_l_norm*1e6, 'k', label="$C_l$")
    plt.errorbar(l_b, Cl_b_norm*1e6, yerr=d_Cl_b_norm*1e6, fmt='.', color="tab:orange")

    plt.title("Recovered Power Spectrum")
    plt.xlabel("l")
    plt.ylabel("l(l+1)C$_l$/2$\\pi$ [mK$^2$]")
    plt.ylim(1e7, 5e9)
    plt.xlim(1e2, 1.5e4)
    plt.savefig(os.path.join(ps_dir, f"PS_Recovery_{data_col}{suffix}.pdf"), format="pdf", dpi=200)

def main():
    parser = argparse.ArgumentParser(
        description="Calculate angular power spectrum using the tapered gridded estimator."
    )
    parser.add_argument(
        "-m", "--ms_path", default=None, help="Path to the MS file."
    )
    parser.add_argument(
        "-d", "--data", default="ideal,tab,flag1,flag2", help="The data types to analyse. {'ideal', 'tab', 'flag1', 'flag2'}"
    )
    parser.add_argument(
        "-c", "--config", required=True, help="Path to the config file."
    )
    args = parser.parse_args()
    ms_path = args.ms_path
    conf_path = args.config  
    data_types = args.data.split(",") 

    if ms_path[-1]=="/":
        ms_path = ms_path[:-1]

    config = load_config(conf_path, config_type="pow_spec")

    tge_conf = config["tge"]

    for key in data_types:
        conf = config[key]
        if conf["suffix"] is None:
            suffix = ""
        else:
            suffix = "_" + conf["suffix"]

        data_col = conf["data_col"]
        flag_type = conf["flag"]["type"]
        if flag_type=="perfect":
            thresh = config[key]["flag"]["thresh"]
            write_perfect_flags(ms_path, thresh)
            name = f"{thresh:.1f}sigma{suffix}"
        elif flag_type=="aoflagger":
            run_aoflagger(ms_path, data_col, config[key]["flag"]["strategies"])
            name = f"aoflagger{suffix}" 
        else:
            ValueError("Incorrect flagging type chosen. Must be one of {perfect, aoflagger}.")

        extract_pow_spec(ms_path, conf["data_col"], tge_conf["n_grid"], tge_conf["n_bins"], name) 


if __name__=="__main__":
    main()