from astropy.coordinates import SkyCoord
import astropy.units as u
from regions import CircleSkyRegion, RectangleSkyRegion

import xarray as xr
import numpy as np

import argparse

import os

import matplotlib.colors as mcolors


def main():

    colors = list(mcolors.TABLEAU_COLORS.values())
    # n_c = len(colors)

    parser = argparse.ArgumentParser(
        description="Extract astronomical positions to ds9 region file."
    )
    parser.add_argument(
        "-z",
        "--zarr_path",
        required=True,
        help="File path to the tabascal zarr simulation file.",
    )

    args = parser.parse_args()
    zarr_path = args.zarr_path

    if zarr_path[-1] == "/":
        zarr_path = zarr_path[:-1]

    region_path = os.path.join(
        os.path.split(zarr_path)[0], "astronomical_positions.ds9"
    )

    xds = xr.open_zarr(zarr_path)

    ra, dec = xds.ast_p_radec.data.compute().T
    c = SkyCoord(ra, dec, unit="deg", frame="fk5")

    max_U = np.max(np.linalg.norm(xds.bl_uvw.data[0], axis=-1))
    syn_bw = (3e8 / (max_U * xds.freq.data.max())).compute()

    with open(region_path, "w") as fp:
        for i in range(len(ra)):
            fp.write(
                CircleSkyRegion(
                    c[i], radius=syn_bw * u.rad, visual={"color": colors[0]}
                ).serialize(format="ds9")
            )

    print(f"Region file written to : {region_path}")
