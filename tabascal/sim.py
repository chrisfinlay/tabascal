import argparse
import os

import jax.numpy as jnp
from jax import random

from tabascal import Observation
from tabascal.utils.tools import (
    generate_random_sky,
    load_antennas,
    save_observations,
    str2bool,
)

parser = argparse.ArgumentParser(description="Simulate RFI contaminated visibilities.")
parser.add_argument(
    "--f_name", default="test", help="File name to save the observations."
)
parser.add_argument("--o_path", default="./", help="Path to save the observations.")
parser.add_argument("--noise", default=0.65, type=float, help="Noise level in Jy.")
parser.add_argument("--T_0", default=440.0, type=float, help="Start time in seconds.")
parser.add_argument("--dT", default=2.0, type=float, help="Time step in seconds.")
parser.add_argument("--N_t", default=10, type=int, help="Number of time steps.")
parser.add_argument(
    "--N_int", default=16, type=int, help="Number of integration samples."
)
parser.add_argument("--N_a", default=8, type=int, help="Number of antennas.")
parser.add_argument("--RFIamp", default=1.0, type=float, help="RFI amplitude.")
parser.add_argument("--Rkey", default=0, type=int, help="Random key.")
parser.add_argument(
    "--satRFI", default=True, type=str2bool, help="Include satellite-based RFI source."
)
parser.add_argument(
    "--grdRFI", default=True, type=str2bool, help="Include ground-based RFI source."
)

args = parser.parse_args()
f_name = args.f_name
output_path = args.o_path
noise = args.noise
T_0 = args.T_0
dT = args.dT
N_t = args.N_t
N_int = args.N_int
N_ant = args.N_a
RFI_amp = args.RFIamp
Rkey = args.Rkey
satRFI = args.satRFI
grdRFI = args.grdRFI

ants_enu = random.permutation(random.PRNGKey(19), load_antennas("MeerKAT"))[:N_ant]
obs = Observation(
    latitude=-30.0,
    longitude=21.0,
    elevation=1050.0,
    ra=27.0,
    dec=15.0,
    times=jnp.arange(T_0, T_0 + N_t * dT, dT),
    freqs=jnp.array([1.227e9]),
    ENU_array=ants_enu,
    n_int_samples=N_int,
)

n_src = 100
mean_I = 0.1

I, d_ra, d_dec = generate_random_sky(n_src, mean_I, obs.fov, obs.syn_bw)
spectral_indices = 0.7 + 0.2 * random.normal(random.PRNGKey(101), (n_src,))
I = I[:, None] * (obs.freqs / obs.freqs[0])[None, :] ** (
    -1.0 * spectral_indices[:, None]
)

obs.addAstro(I=I, ra=obs.ra + d_ra, dec=obs.dec + d_dec)

# rfi_P = jnp.array([6e-4 * jnp.exp(-0.5 * ((obs.freqs - 1.2e9) / 2e7) ** 2)])
rfi_P = RFI_amp * 200.0 * 0.29e-6 * jnp.ones((1, 1))

if satRFI:
    obs.addSatelliteRFI(
        Pv=rfi_P,
        elevation=jnp.array([202e5]),
        inclination=jnp.array([55.0]),
        lon_asc_node=jnp.array([21.0]),
        periapsis=jnp.array([5.0]),
    )

# rfi_P = jnp.array([6e-4 * jnp.exp(-0.5 * ((obs.freqs - 1.5e9) / 2e7) ** 2)])
rfi_P = RFI_amp * 1e-6 * jnp.ones((1, 1))

if grdRFI:
    obs.addStationaryRFI(
        Pv=rfi_P,
        latitude=jnp.array([-20.0]),
        longitude=jnp.array([30.0]),
        elevation=jnp.array([obs.elevation]),
    )

obs.addGains(G0_mean=1.0, G0_std=0.05, Gt_std_amp=1e-5, Gt_std_phase=jnp.deg2rad(1e-3))

obs.calculate_vis()
obs.addNoise(noise=noise, key=random.PRNGKey(104 + int(T_0)))

f_name = (
    f"{f_name}_obs_{obs.n_ant:0>2}A_{obs.n_time:0>3}T_{obs.n_freq:0>3}F"
    + f"_{obs.n_ast:0>3}AST_{len(obs.rfi_orbit.keys())}SAT_{len(obs.rfi_geo.keys())}GRD"
)

save_path = os.path.join(output_path, f_name)

save_observations(
    save_path,
    [
        obs,
    ],
)

print()
print("Saved observations to:")
print("----------------------")
print(save_path + ".h5")

print(obs)