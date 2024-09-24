#!/usr/bin/env python

import argparse
import os

import numpy as np
import xarray as xr

from tabascal.utils.sky import generate_random_sky
from tabascal.utils.tools import load_antennas, str2bool
from tabascal.utils.write import write_ms, mk_obs_name, mk_obs_dir
from tabascal.dask.observation import Observation

parser = argparse.ArgumentParser(
    description="Simulate a target observation contaminated by RFI."
)
parser.add_argument(
    "--f_name", default="mixed_sources", help="File name to save the observations."
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
parser.add_argument("--t_0", default=440.0, type=float, help="Start time in seconds.")
parser.add_argument("--delta_t", default=2.0, type=float, help="Time step in seconds.")
parser.add_argument("--N_t", default=450, type=int, help="Number of time steps.")
parser.add_argument(
    "--N_int", default=4, type=int, help="Number of integration samples."
)
parser.add_argument("--N_f", default=1, type=int, help="Number of frequency channels.")
parser.add_argument("--freq_start", default=1227e6, type=float, help="Start frequency.")
parser.add_argument("--freq_end", default=1227.209e6, type=float, help="End frequency.")
parser.add_argument("--N_a", default=64, type=int, help="Number of antennas.")
parser.add_argument("--RFIamp", default=1.0, type=float, help="RFI amplitude.")
parser.add_argument("--seed", default=0, type=int, help="Random seed.")
parser.add_argument(
    "--N_sat", default=2, type=int, help="Number of satellite-based RFI sources."
)
parser.add_argument(
    "--N_grd", default=3, type=int, help="Number of ground-based RFI sources."
)
parser.add_argument(
    "--N_p_ast", default=10, type=int, help="Number of astronomical point sources."
)
parser.add_argument(
    "--N_g_ast", default=10, type=int, help="Number of astronomical Gaussian sources."
)
parser.add_argument(
    "--N_e_ast", default=10, type=int, help="Number of astronomical exponential sources."
)
parser.add_argument(
    "--src_size", default=50.0, type=int, help="Characterisitic source size."
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
seed = args.seed
N_sat = args.N_sat
N_grd = args.N_grd
N_p_ast = args.N_p_ast
N_g_ast = args.N_g_ast
N_e_ast = args.N_e_ast
src_size = args.src_size
overwrite = args.overwrite
chunksize = args.chunksize
freq_start = args.freq_start
freq_end = args.freq_end

N_ast = N_p_ast + N_g_ast + N_e_ast

rng = np.random.default_rng(12345)
ants_enu = rng.permutation(load_antennas("MeerKAT"))[:N_ant]
# itrf_path = "../data/Meerkat.itrf.txt"

times = np.arange(t_0, t_0 + N_t * dT, dT)
freqs = np.linspace(freq_start, freq_end, N_freq)

obs = Observation(
    latitude=-30.0,
    longitude=21.0,
    elevation=1050.0,
    ra=27.0,
    dec=-30.0,
    times=times,
    freqs=freqs,
    SEFD=SEFD,
    ENU_array=ants_enu,
    # ITRF_path=itrf_path, 
    n_int_samples=N_int,
    max_chunk_MB=chunksize,
)

max_bw = 36 / 3600
beam_width = obs.syn_bw if obs.syn_bw < max_bw else max_bw
n_beam = 5
min_I = 3 * np.mean(obs.noise_std) / np.sqrt(N_t * N_ant * (N_ant - 1) / 2)

print()
print(f'Generating "Astro" sources with >{3600*n_beam*beam_width:.0f}" separation ...')
print(
    f"Sources lie within {obs.fov/2:.2f} degrees with minimum flux of {min_I.compute()*1e3:.1f} mJy"
)


I, d_ra, d_dec = generate_random_sky(
    n_src=N_ast,
    min_I=min_I,
    max_I=1.0,
    freqs=obs.freqs,
    fov=obs.fov,
    beam_width=beam_width,
    random_seed=seed,
    n_beam=n_beam,
)

print('Adding "Astro point sources ...')

if N_p_ast>0:
    obs.addAstro(
        I=I[:N_p_ast, None, :] * np.ones((1, obs.n_time_fine, 1)),
        ra=obs.ra + d_ra[:N_p_ast],
        dec=obs.dec + d_dec[:N_p_ast],
    )

print('Adding "Astro Gauss" sources ...')

if N_g_ast>0:
    sizes = np.abs(rng.normal(scale=src_size, size=(N_g_ast,2)))
    obs.addAstroGauss(
        I=I[N_p_ast:N_p_ast+N_g_ast, None, :] * np.ones((1, obs.n_time_fine, 1)),
        major=sizes.max(axis=1),
        minor=sizes.min(axis=1),
        pos_angle=rng.uniform(low=0.0, high=360.0, size=(N_g_ast)),
        ra=obs.ra + d_ra[N_p_ast:N_p_ast+N_g_ast],
        dec=obs.dec + d_dec[N_p_ast:N_p_ast+N_g_ast],
    )

print('Adding "Astro exponential" sources ...')

if N_e_ast>0:
    obs.addAstroExp(
        I=I[N_p_ast+N_g_ast:, None, :] * np.ones((1, obs.n_time_fine, 1)),
        shape=rng.normal(scale=src_size, size=(N_e_ast,)),
        ra=obs.ra + d_ra[N_p_ast+N_g_ast:],
        dec=obs.dec + d_dec[N_p_ast+N_g_ast:],
    )


#### Satellite-based RFI ####

print('Adding "Satellite" sources ...')

rfi_P = np.array(
    [
        RFI_amp * 0.6e-4 * np.exp(-0.5 * ((freqs - 1.227e9) / 5e6) ** 2),
        RFI_amp * 2 * 0.6e-4 * np.exp(-0.5 * ((freqs - 1.227e9) / 5e6) ** 2),
    ]
)

elevation = [20200e3, 19140e3]
inclination = [55.0, 64.8]
lon_asc_node = [41.0, 17.0]
periapsis = [-45., -31.]

if N_sat > 0 and N_sat <= 2:
    obs.addSatelliteRFI(
        Pv=rfi_P[:N_sat, None, :] * np.ones((N_sat, obs.n_time_fine, obs.n_freq)),
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

print(obs)

print()
print("Saving observation simulation files to:")
print("----------------------")
print(save_path)

print()
print("Saving observation zarr file ...")
obs.write_to_zarr(zarr_path, overwrite)

print()
print("Saving observation MS file ...")
xds = xr.open_zarr(zarr_path)
write_ms(xds, ms_path, overwrite=overwrite)
