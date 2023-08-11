#!/usr/bin/env python

import argparse
import os

import numpy as np

from tabascal.utils.sky import generate_random_sky
from tabascal.utils.tools import load_antennas, str2bool

parser = argparse.ArgumentParser(
    description="Simulate a target observation contaminated by RFI."
)
parser.add_argument(
    "--f_name", default="target", help="File name to save the observations."
)
parser.add_argument("--o_path", default="./", help="Path to save the observations.")
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
    "--N_int", default=128, type=int, help="Number of integration samples."
)
parser.add_argument(
    "--N_f", default=128, type=int, help="Number of frequency channels."
)
parser.add_argument(
    "--freq_start", default=1.227e9, type=float, help="Start frequency."
)
parser.add_argument("--freq_end", default=1.226752e9, type=float, help="End frequency.")
parser.add_argument("--N_a", default=64, type=int, help="Number of antennas.")
parser.add_argument("--RFIamp", default=1.0, type=float, help="RFI amplitude.")
parser.add_argument("--seed", default=0, type=int, help="Random seed.")
parser.add_argument(
    "--N_sat", default=1, type=int, help="Number of satellite-based RFI sources."
)
parser.add_argument(
    "--N_grd", default=1, type=int, help="Number of ground-based RFI sources."
)
parser.add_argument(
    "--backend", default="dask", type=str, help="Use pure JAX or Dask backend."
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
overwrite = args.overwrite
chunksize = args.chunksize
freq_start = args.freq_start
freq_end = args.freq_end

if args.backend.lower() == "jax":
    from tabascal.jax.observation import Observation

    print()
    print("Using JAX backend")
    print()
else:
    from tabascal.dask.observation import Observation

    print()
    print("Using Dask backend")
    print()

rng = np.random.default_rng(12345)
ants_enu = rng.permutation(load_antennas("MeerKAT"))[:N_ant]

times = np.arange(t_0, t_0 + N_t * dT, dT)
freqs = np.linspace(freq_start, freq_end, N_freq)

obs = Observation(
    latitude=-30.0,
    longitude=21.0,
    elevation=1050.0,
    ra=27.0,
    dec=15.0,
    times=times,
    freqs=freqs,
    SEFD=SEFD,
    ENU_array=ants_enu,
    n_int_samples=N_int,
    max_chunk_MB=chunksize,
)

beam_width = obs.syn_bw if obs.syn_bw < 1e-2 else 1e-2

print(f'Generating "Astro" sources with >{3600*5*beam_width:.0f}" separation ...')


I, d_ra, d_dec = generate_random_sky(
    n_src=100,
    min_I=np.mean(obs.noise_std) / 5.0,
    max_I=1.0,
    freqs=obs.freqs,
    fov=obs.fov,
    beam_width=beam_width,
    random_seed=seed,
)

print('Adding "Astro" sources ...')

obs.addAstro(I=I, ra=obs.ra + d_ra, dec=obs.dec + d_dec)

#### Satellite-based RFI ####

print('Adding "Satellite" sources ...')

rfi_P = [
    RFI_amp * 0.6e-4 * np.exp(-0.5 * ((freqs - 1.227e9) / 5e6) ** 2),
    RFI_amp * 2 * 0.6e-4 * np.exp(-0.5 * ((freqs - 1.227e9) / 5e6) ** 2),
]

elevation = [20200e3, 19140e3]
inclination = [55.0, 64.8]
lon_asc_node = [21.0, 17.0]
periapsis = [7.0, 1.0]

if N_sat > 0 and N_sat <= 2:
    obs.addSatelliteRFI(
        Pv=rfi_P[:N_sat],
        elevation=elevation[:N_sat],
        inclination=inclination[:N_sat],
        lon_asc_node=lon_asc_node[:N_sat],
        periapsis=periapsis[:N_sat],
    )
elif N_sat > 2:
    raise ValueError("Maximum number of satellite-based RFI sources is 2.")

#### Ground-based RFI ####

print('Adding "Ground" sources ...')

rfi_P = [
    RFI_amp * 6e-4 * np.exp(-0.5 * ((freqs - 1.22e9) / 3e6) ** 2),
    RFI_amp * 1.5e-4 * np.exp(-0.5 * ((freqs - 1.22e9) / 3e6) ** 2),
    RFI_amp * 0.4e-4 * np.exp(-0.5 * ((freqs - 1.22e9) / 3e6) ** 2),
]
latitude = [-20.0, -20.0, -25.0]
longitude = [30.0, 20.0, 20.0]
elevation = [obs.elevation, obs.elevation, obs.elevation]

if N_grd > 0 and N_grd <= 3:
    obs.addStationaryRFI(
        Pv=rfi_P[:N_grd],
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

f_name = (
    f"{f_name}_obs_{obs.n_ant:0>2}A_{obs.n_time:0>3}T-{int(obs.times[0]):0>4}-{int(obs.times[-1]):0>4}"
    + f"_{obs.n_int_samples:0>3}I_{obs.n_freq:0>3}F-{float(obs.freqs[0]):.3e}-{float(obs.freqs[-1]):.3e}"
    + f"_{obs.n_ast:0>3}AST_{obs.n_rfi_satellite}SAT_{obs.n_rfi_stationary}GRD"
)

save_path = os.path.join(output_path, f_name)

print(obs)

print()
print("Saving observation zarr file to:")
print("----------------------")
print(save_path)

obs.write_to_zarr(save_path, overwrite)

print()
print("Saving observation MS file to:")
print("----------------------")
print(save_path + ".ms")

obs.write_to_ms(save_path + ".ms", overwrite)
