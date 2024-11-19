import argparse
import subprocess
import os

from datetime import datetime

import numpy as np

from tabascal.utils.config import load_config, run_sim_config
from tabascal.utils.run_tabascal import tabascal_subtraction

def main():
    
    parser = argparse.ArgumentParser(
        description="Run and end-to-end simulation and source extraction analysis."
    )
    parser.add_argument(
        "-s", "--sim_config", required=True, help="Path to the simulation config file."
    )
    parser.add_argument(
        "-t", "--tab_config", required=True, help="Path to the tabascal config file."
    )
    parser.add_argument(
        "-e", "--extract_config", required=True, help="Path to the extraction config file."
    )
    parser.add_argument(
        "-r", "--rfi_amp", default=1.0, type=float, help="Path to the extraction config file."
    )
    parser.add_argument(
        "-rr", "--random_seed_offset", default=0, type=int, help="Offset to random seeds used."
    )
    parser.add_argument(
        "-ra", "--ra", type=float, help="Right Ascension of the observation."
    )
    parser.add_argument(
        "-dec", "--dec", type=float, help="Declination of the observation."
    )
    args = parser.parse_args()
    config_path = args.sim_config
    r_seed_offset = args.random_seed_offset
    config = load_config(config_path, config_type="sim")

    if args.ra is not None:
        config["observation"]["ra"] = args.ra

    if args.dec is not None:
        config["observation"]["dec"] = args.dec

    spacetrack_path = os.path.abspath(config["rfi_sources"]["tle_satellite"]["spacetrack_path"])
    config["rfi_sources"]["tle_satellite"]["spacetrack_path"] = spacetrack_path

    config["rfi_sources"]["tle_satellite"]["power_scale"] *= args.rfi_amp
    config["rfi_sources"]["satellite"]["power_scale"] *= args.rfi_amp
    config["rfi_sources"]["stationary"]["power_scale"] *= args.rfi_amp

    config["observation"]["random_seed"] += r_seed_offset
    config["ast_sources"]["pow_spec"]["random"]["random_seed"] += r_seed_offset
    config["gains"]["random_seed"] += r_seed_offset
    
    config["output"]["suffix"] = f"{args.rfi_amp:.1e}RFI_{r_seed_offset:.0f}RSEED"

    times = {"t0": datetime.now(),}

    obs, sim_dir = run_sim_config(obs_spec=config, spacetrack_path=spacetrack_path)

    sim_name = os.path.split(sim_dir)[1]
    ms_path = os.path.join(sim_dir, sim_name+".ms")

    times["t1"] = datetime.now()

    print("======================================================================")
    norad_path = os.path.join(sim_dir, "input_data/norad_ids.yaml")
    norad_ids = [int(x) for x in np.loadtxt(norad_path)]
    tab_config = load_config(args.tab_config, config_type="tab")
    tabascal_subtraction(config=tab_config, sim_dir=sim_dir, spacetrack_path=spacetrack_path, norad_ids=norad_ids)
    # subprocess.run(f"tabascal -c {args.tab_config} -s {sim_dir}", shell=True, executable="/bin/bash")

    times["t2"] = datetime.now()

    print("======================================================================")
    subprocess.run(f"ps-extract -c {args.extract_config} -m {ms_path}", shell=True, executable="/bin/bash")

    times["t3"] = datetime.now()
    print("\n==================================================================\n")
    print(f"Simualtion time : {times['t1']-times['t0']}")
    print(f"TABASCAL time   : {times['t2']-times['t1']}")
    print(f"Extraction time : {times['t3']-times['t2']}")
    print("========================================")
    print(f"Total time      : {times['t3']-times['t0']}")

if __name__=="__main__":
    main()