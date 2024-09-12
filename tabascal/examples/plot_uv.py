#!/usr/bin/env python

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

from tabascal.utils.tools import (
    load_antennas,
    str2bool,
)

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
parser.add_argument("--t_0", default=-300.0, type=float, help="Start time in seconds.")
parser.add_argument("--delta_t", default=6.0, type=float, help="Time step in seconds.")
parser.add_argument("--N_t", default=100, type=int, help="Number of time steps.")
parser.add_argument(
    "--N_int", default=1, type=int, help="Number of integration samples."
)
parser.add_argument("--N_f", default=1, type=int, help="Number of frequency channels.")
parser.add_argument("--freq_start", default=1227e6, type=float, help="Start frequency.")
parser.add_argument("--freq_end", default=1227.209e6, type=float, help="End frequency.")
parser.add_argument("--N_a", default=64, type=int, help="Number of antennas.")
parser.add_argument("--RFIamp", default=1.0, type=float, help="RFI amplitude.")
parser.add_argument("--CALamp", default=1.0, type=float, help="Calibrator amplitude.")
parser.add_argument("--seed", default=0, type=int, help="Random seed.")
parser.add_argument(
    "--N_sat", default=0, type=int, help="Number of satellite-based RFI sources."
)
parser.add_argument(
    "--N_grd", default=0, type=int, help="Number of ground-based RFI sources."
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
CAL_amp = args.CALamp
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
    ra=21.0,
    dec=30.0,
    times=times,
    freqs=freqs,
    SEFD=SEFD,
    ENU_array=ants_enu,
    n_int_samples=N_int,
    max_chunk_MB=chunksize,
)

from tabascal.jax.coordinates import alt_az_of_source, gmst_to_lst

lst = gmst_to_lst(obs.times.compute(), obs.longitude.compute())
alt = alt_az_of_source(lst, *[x.compute() for x in [obs.latitude, obs.ra, obs.dec]])[:,0]

plt.rcParams["font.size"] = 18

total_time = times[-1] - times[0]
if total_time>3600:
    times = times/3600
    scale = "hr"
elif total_time>60:
    times = times/60
    scale = "min"
else:
    scale = "sec"

plt.figure(figsize=(10,7))
plt.plot(times, alt, '.-')
plt.xlabel(f"Time [{scale}]")
plt.ylabel("Source Altitude [deg]")
plt.savefig("SourceAltitude.png", format="png", dpi=200)

plt.figure(figsize=(10,10))
u = obs.bl_uvw[:,:,0].compute().flatten()
v = obs.bl_uvw[:,:,1].compute().flatten()
plt.plot(u, v, 'k.', ms=1, alpha=0.3)
plt.plot(-u, -v, 'k.', ms=1, alpha=0.3)
plt.grid()
plt.xlim(-8e3, 8e3)
plt.ylim(-8e3, 8e3)
plt.xlabel("U [m]")
plt.ylabel("V [m]")
plt.savefig("UV.png", format="png", dpi=200)
