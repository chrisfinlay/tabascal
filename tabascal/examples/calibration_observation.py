#!/usr/bin/env python

import argparse

import numpy as np
import xarray as xr
import dask.array as da

from tabascal.utils.write import write_ms, mk_obs_name, mk_obs_dir
from tabascal.utils.plot import plot_uv, plot_src_alt, plot_angular_seps
from tabascal.utils.tools import str2bool, load_antennas
from tabascal.dask.observation import Observation


def main():

    parser = argparse.ArgumentParser(
        description="Simulate a calibrator observation contaminated by RFI."
    )
    parser.add_argument(
        "--f_name", default="calibrator", help="File name to save the observations."
    )
    parser.add_argument(
        "--o_path", default="./data/", help="Path to save the observations."
    )
    parser.add_argument(
        "--SEFD",
        default=420.0,
        type=float,
        help="System Equivalent flux density in Jy. Same across frequency and antennas.",
    )
    parser.add_argument("--t_0", default=0.0, type=float, help="Start time in seconds.")
    parser.add_argument("--delta_t", default=2.0, type=float, help="Time step in seconds.")
    parser.add_argument("--N_t", default=150, type=int, help="Number of time steps.")
    parser.add_argument(
        "--N_int", default=4, type=int, help="Number of integration samples."
    )
    parser.add_argument("--N_f", default=1, type=int, help="Number of frequency channels.")
    parser.add_argument("--freq_start", default=1227e6, type=float, help="Start frequency.")
    parser.add_argument("--chan_width", default=209e3, type=float, help="End frequency.")
    parser.add_argument("--N_a", default=64, type=int, help="Number of antennas.")
    parser.add_argument("--RFIamp", default=1.0, type=float, help="RFI amplitude.")
    parser.add_argument("--CALamp", default=1.0, type=float, help="Calibrator amplitude.")
    parser.add_argument("--seed", default=0, type=int, help="Random seed.")
    parser.add_argument(
        "--N_sat", default=2, type=int, help="Number of satellite-based RFI sources."
    )
    parser.add_argument(
        "--N_grd", default=3, type=int, help="Number of ground-based RFI sources."
    )
    parser.add_argument(
        "--overwrite", default="no", type=str2bool, help="Overwrite existing observation."
    )
    parser.add_argument(
        "--chunksize",
        default=100.0,
        type=float,
        help="Chunksize for Dask visibility array in MB.",
    )

    args = parser.parse_args()
    f_name = args.f_name
    output_path = args.o_path
    SEFD = args.SEFD
    t_0 = args.t_0
    dT = args.delta_t
    N_t = args.N_t
    N_int = args.N_int
    N_freq = args.N_f
    N_ant = args.N_a
    RFI_amp = args.RFIamp
    CAL_amp = args.CALamp
    seed = args.seed
    N_sat = args.N_sat
    N_grd = args.N_grd
    overwrite = args.overwrite
    chunksize = args.chunksize
    freq_start = args.freq_start
    chan_width = args.chan_width
        

    rng = np.random.default_rng(12345)
    ants_enu = rng.permutation(load_antennas("MeerKAT"))[:N_ant]
    # ants_enu = load_antennas("MeerKAT")[:N_ant]
    # itrf_path = "../data/Meerkat.itrf.txt"

    times = da.arange(t_0, t_0 + N_t * dT, dT)
    freqs = da.arange(freq_start, freq_start + N_freq * chan_width, chan_width)

    obs = Observation(
        latitude=-30.0,
        longitude=21.0,
        elevation=1050.0,
        ra=21.0,
        dec=-30.0,
        times=times,
        freqs=freqs,
        SEFD=SEFD,
        ENU_array=ants_enu,
        # ITRF_path=itrf_path, 
        n_int_samples=N_int,
        max_chunk_MB=chunksize,
    )

    print()
    print('Adding "Calibrator" source ...')
    obs.addAstro(I=CAL_amp * np.ones((len(freqs),)), ra=obs.ra, dec=obs.dec)

    #### Satellite-based RFI ####

    print('Adding "Satellite" sources ...')

    rfi_P = np.array(
        [
            RFI_amp * 5.8e-6 * np.exp(-0.5 * ((freqs - 1.227e9) / 5e6) ** 2),
            RFI_amp * 0.6e-4 * np.exp(-0.5 * ((freqs - 1.2e9) / 5e6) ** 2),
            RFI_amp * 2 * 0.6e-4 * np.exp(-0.5 * ((freqs - 1.2e9) / 5e6) ** 2),
        ]
    )

    elevation = [20200e3, 19140e3]
    inclination = [55.0, 64.8]
    lon_asc_node = [41.0, 17.0]
    periapsis = [-45., -31.]

    if N_sat > 0 and N_sat <= 2:
        obs.addSatelliteRFI(
            Pv=rfi_P[:N_sat, None, :] * da.ones((N_sat, obs.n_time_fine, obs.n_freq)),
            elevation=elevation[:N_sat],
            inclination=inclination[:N_sat],
            lon_asc_node=lon_asc_node[:N_sat],
            periapsis=periapsis[:N_sat],
        )
    elif N_sat > 2:
        raise ValueError("Maximum number of satellite-based RFI sources is 2.")

    #### Ground-based RFI ####

    print('Adding "Ground" sources ...')

    rfi_P = np.array(
        [
            RFI_amp * 6e-4 * np.exp(-0.5 * ((freqs - 1.22e9) / 3e6) ** 2),
            RFI_amp * 1.5e-4 * np.exp(-0.5 * ((freqs - 1.22e9) / 3e6) ** 2),
            RFI_amp * 0.4e-4 * np.exp(-0.5 * ((freqs - 1.22e9) / 3e6) ** 2),
        ]
    )
    latitude = [-20.0, -20.0, -25.0]
    longitude = [30.0, 20.0, 20.0]
    elevation = [obs.elevation, obs.elevation, obs.elevation]

    if N_grd > 0 and N_grd <= 3:
        obs.addStationaryRFI(
            Pv=rfi_P[:N_grd, None, :] * np.ones((N_grd, obs.n_time_fine, obs.n_freq)),
            latitude=latitude[:N_grd],
            longitude=longitude[:N_grd],
            elevation=elevation[:N_grd],
        )
    elif N_grd > 3:
        raise ValueError("Maximum number of ground-based RFI sources is 3.")

    print('Adding "Gains" ...')

    obs.addGains(G0_mean=1.0, G0_std=0.05, Gt_std_amp=1e-5, Gt_std_phase=np.deg2rad(1e-3))

    print("Calculating visibilities ...")

    obs.calculate_vis()

    obs_name = mk_obs_name(f_name, obs)
    save_path, zarr_path, ms_path = mk_obs_dir(output_path, obs_name, overwrite)

    plot_src_alt(obs, save_path)
    plot_uv(obs, save_path)
    plot_angular_seps(obs, save_path)

    print(obs)

    print()
    print("Saving observation simulation files to:")
    print("----------------------")
    print(save_path)

    print()
    print("Saving observation zarr file ...")
    obs.write_to_zarr(zarr_path, overwrite)

    print()
    print(f"Flag Rate      : {100*obs.flags.mean().compute(): .1f} %")

    print()
    print("Saving observation MS file ...")
    xds = xr.open_zarr(zarr_path)
    write_ms(xds, ms_path, overwrite=overwrite)
