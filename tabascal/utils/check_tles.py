from tabascal.utils.tle import get_tles_by_id, get_satellite_positions
from tabascal.utils.config import yaml_load
from tabascal.utils.plot_results import get_file_names
from tabascal.jax.coordinates import mjd_to_jd
import numpy as np
from tqdm import tqdm
import xarray as xr
import os

import argparse


def calculate_tle_pos_errors(
    data_dir: str,
    tle_offset: int,
    tle_dir: str,
    tab_suffix: str,
    st_config: dict,
    model_name="map_pred_fixed_orbit_rfi_full_fft_standard_padded_model",
):

    files, data = get_file_names(data_dir, model_name, tab_suffix)

    minmax = []
    missing = []
    for i in tqdm(range(len(files["zarr_files"]))):
        xds = xr.open_zarr(files["zarr_files"][i])
        sat_pos_true = xds.rfi_tle_sat_xyz.data
        norad_ids = xds.norad_ids.data.compute()
        times_mjd_fine = xds.time_mjd_fine.data
        time_mjd = np.mean(times_mjd_fine)
        time_jd = mjd_to_jd(time_mjd)
        tles_df = get_tles_by_id(
            st_config["username"],
            st_config["password"],
            norad_ids,
            time_jd + tle_offset,
            tle_dir=tle_dir,
        )
        tles = np.atleast_2d(tles_df[["TLE_LINE1", "TLE_LINE2"]].values)
        sat_pos = get_satellite_positions(tles, mjd_to_jd(times_mjd_fine))
        n_idx = [
            i
            for i, n_id in enumerate(norad_ids)
            if n_id in tles_df["NORAD_CAT_ID"].values
        ]
        max_d = np.max(
            np.linalg.norm(sat_pos - sat_pos_true[n_idx], axis=-1), axis=1
        ).compute()
        minmax.append([np.min(max_d), np.max(max_d)])

        if len(n_idx) != len(sat_pos_true):
            missing.append(i)

    missing = [os.path.split(files["zarr_files"][i])[1] for i in missing]
    # rchi2 = [data["rchi2"][i] for i in missing]

    print(
        f"Minimum and Maximum TLE position errors : ({np.nanmin(minmax):.0f}, {np.nanmax(minmax):.0f}) m"
    )
    print(f"Data files with some TLEs not found : {len(missing)} files")
    for m in missing:
        print(f"{m}")
    # for m, r in zip(missing, rchi2):
    #     print(f"{m} : {r:.2f}")

    return minmax, missing  # , rchi2


def main():

    parser = argparse.ArgumentParser(
        description="Analyse tabascal simulations, recoveries and point source extractions."
    )
    parser.add_argument(
        "-d",
        "--data_dir",
        required=True,
        help="Path to the data directory containing all the simulations.",
    )
    parser.add_argument(
        "-t",
        "--tle_offset",
        default=-1,
        type=int,
        help="TLE offset to check in days.",
    )
    parser.add_argument(
        "-td",
        "--tle_dir",
        default="./tles",
        help="TLE directory.",
    )
    parser.add_argument(
        "-st",
        "--spacetrack_login",
        required=True,
        help="Path to the spacetrack login config file.",
    )
    args = parser.parse_args()
    data_dir = args.data_dir
    tle_offset = args.tle_offset
    tle_dir = args.tle_dir
    tab_suffix = "_offset_-1"
    tab_suffix = "_k0_1e-2"

    st_config = yaml_load(args.spacetrack_login)

    calculate_tle_pos_errors(data_dir, tle_offset, tle_dir, tab_suffix, st_config)


if __name__ == "__main__":
    main()
