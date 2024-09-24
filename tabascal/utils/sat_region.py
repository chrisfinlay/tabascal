from astropy.coordinates import SkyCoord
import astropy.units as u
from regions import CircleSkyRegion, RectangleSkyRegion

import xarray as xr
import numpy as np

import argparse

import os

import matplotlib.colors as mcolors

colors = list(mcolors.TABLEAU_COLORS.values())
n_c = len(colors)

parser = argparse.ArgumentParser(
    description="Extract satellite paths to ds9 region file."
)
parser.add_argument(
    "--zarr_path", required=True, help="File path to the tabascal zarr simulation file."
)
parser.add_argument(
    "--max_d", default=30.0, type=float, help="Maximum angular distance from phase centre (in degrees) of the satellite path to include."
)

args = parser.parse_args()
zarr_path = args.zarr_path
max_d = args.max_d

if zarr_path[-1]=="/":
    zarr_path = zarr_path[:-1]

region_path = os.path.join(os.path.split(zarr_path)[0], "satelitte_paths.ds9")

def xyz_to_radec(xyz):
    if xyz.ndim==2:
        xyz = xyz[None,:,:]

    xyz = xyz / np.linalg.norm(xyz, axis=-1, keepdims=True)
    radec = np.zeros((*xyz.shape[:2], 2))
    radec[:,:,0] = np.arctan2(xyz[:,:,1], xyz[:,:,0])
    radec[:,:,1] = np.arcsin(xyz[:,:,2])

    return np.rad2deg(radec)

xds = xr.open_zarr(zarr_path)

times = np.linspace(xds.time_fine.min(), xds.time_fine.max(), 100)

# xyz = (xds.rfi_sat_xyz - xds.ants_xyz.mean(dim="ant")).sel(time_fine=times, method="nearest")
xyz = xds.rfi_sat_xyz.sel(time_fine=times, method="nearest")

radec = xyz_to_radec(xyz.compute())

c0 = SkyCoord(xds.target_ra, xds.target_dec, unit='deg', frame='fk5')
c = SkyCoord(radec[:,:,0], radec[:,:,1], unit='deg', frame='fk5')

min_sep = c0.separation(c).min(axis=1).deg
print(min_sep)
idx = np.where(min_sep<max_d)[0]

min_idx = np.argmin(c0.separation(c)[:,:-1], axis=1)

ang_v = np.diff(radec, axis=1)
ang_theta = np.arctan2(ang_v[:,:,1], ang_v[:,:,0])

with open(region_path, "w") as fp:
    for c_i, i in enumerate(idx):
        fp.write(RectangleSkyRegion(c0, 0.05*u.deg, 0.3*u.deg, ang_theta[i,min_idx[i]]*u.rad, visual={"color": colors[c_i%n_c]}).serialize(format="ds9"))
        for j in range(len(times)):
            fp.write(CircleSkyRegion(c[i,j], radius=0.1*u.deg, visual={"color": colors[c_i%n_c]}).serialize(format="ds9"))