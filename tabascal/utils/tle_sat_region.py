from astropy.coordinates import SkyCoord
import astropy.units as u
from regions import CircleSkyRegion, RectangleSkyRegion

import xarray as xr
import numpy as np

from daskms import xds_from_ms, xds_from_table

import argparse

import os

import matplotlib.colors as mcolors

from tabascal.utils.config import yaml_load
from tabascal.utils.tle import get_tles_by_id, get_satellite_positions
from tabascal.jax.coordinates import itrf_to_xyz, gmsa_from_jd


def main():

    colors = list(mcolors.TABLEAU_COLORS.values())
    n_c = len(colors)

    parser = argparse.ArgumentParser(
        description="Extract satellite paths to ds9 region file."
    )
    parser.add_argument(
        "-m", "--ms_path", required=True, help="File path to the tabascal zarr simulation file."
    )
    parser.add_argument(
        "-d", "--max_d", default=90.0, type=float, help="Maximum angular distance from phase centre (in degrees) of the satellite path to include."
    )
    parser.add_argument(
        "-st", "--spacetrack", help="Path to YAML config file containing Space-Track login details with 'username' and 'password'."
    )
    parser.add_argument(
        "-ni", "--norad_ids", help="NORAD IDs of satellites to include."
    )
    parser.add_argument(
        "-np", "--norad_path", help="Path to YAML config file containing list of norad_ids."
    )
    parser.add_argument(
        "-td", "--tle_dir", default="./tles", help="Path to directory containing TLEs."
    )
    
    args = parser.parse_args()
    ms_path = args.ms_path
    max_d = args.max_d
    spacetrack = yaml_load(args.spacetrack)
    tle_dir = args.tle_dir
    norad_ids = []
    if args.norad_ids:
        norad_ids += args.norad_ids.split(",")
    if args.norad_path:
        norad_ids += yaml_load(args.norad_path).split()

    os.makedirs(tle_dir, exist_ok=True)

    if ms_path[-1]=="/":
        ms_path = ms_path[:-1]

    region_path = os.path.join(os.path.split(ms_path)[0], "satelitte_paths.ds9")

    def xyz_to_radec(xyz):
        if xyz.ndim==2:
            xyz = xyz[None,:,:]

        xyz = xyz / np.linalg.norm(xyz, axis=-1, keepdims=True)
        radec = np.zeros((*xyz.shape[:2], 2))
        radec[:,:,0] = np.arctan2(xyz[:,:,1], xyz[:,:,0])
        radec[:,:,1] = np.arcsin(xyz[:,:,2])

        return np.rad2deg(radec)

    xds = xds_from_ms(ms_path)[0]

    times = np.unique(xds.TIME.data.compute())
    times_jd = np.linspace(times[0], times[-1], 100)

    xds_ants = xds_from_table(ms_path+"::ANTENNA")[0]
    ants_itrf = np.mean(xds_ants.POSITION.data.compute(), axis=0, keepdims=True)
    ants_xyz = itrf_to_xyz(ants_itrf, gmsa_from_jd(times_jd))[:,0]

    xds_src = xds_from_table(ms_path+"::SOURCE")[0]
    ra, dec = np.rad2deg(xds_src.DIRECTION.data[0].compute())

    epoch_jd = np.mean(times)
    tles_df = get_tles_by_id(
        spacetrack["username"], 
        spacetrack["password"], 
        norad_ids,
        epoch_jd,
        tle_dir=tle_dir,
    )
    tles = np.atleast_2d(tles_df[["TLE_LINE1", "TLE_LINE2"]].values)
    n_tles = len(tles)

    print(f"Found {n_tles} matching TLEs.")

    if n_tles>0:
        rfi_xyz = get_satellite_positions(tles, times_jd)

        xyz = rfi_xyz - ants_xyz[None,:,:]

        radec = xyz_to_radec(xyz)

        c0 = SkyCoord(ra, dec, unit='deg', frame='fk5')
        c = SkyCoord(radec[:,:,0], radec[:,:,1], unit='deg', frame='fk5')

        min_sep = c0.separation(c).min(axis=1).deg
        print(f"Minimum angular separation from target : {[round(x, 1) for x in min_sep]} deg.")
        print(f"Only including satellites within {max_d:.1f} degrees of pointing direction.")
        idx = np.where(min_sep<max_d)[0]

        min_idx = np.argmin(c0.separation(c)[:,:-1], axis=1)

        ang_v = np.diff(radec, axis=1)
        ang_theta = np.arctan2(ang_v[:,:,1], ang_v[:,:,0])

        with open(region_path, "w") as fp:
            for c_i, i in enumerate(idx):
                fp.write(RectangleSkyRegion(c0, 0.05*u.deg, 0.3*u.deg, ang_theta[i,min_idx[i]]*u.rad, visual={"color": colors[c_i%n_c]}).serialize(format="ds9"))
                for j in range(len(times)):
                    fp.write(CircleSkyRegion(c[i,j], radius=0.1*u.deg, visual={"color": colors[c_i%n_c]}).serialize(format="ds9"))

if __name__=="__main__":
    main()