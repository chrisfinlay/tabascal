import os
import argparse
import subprocess

from datetime import datetime

from tqdm import tqdm

import numpy as np

from tabascal.utils.config import load_config, run_sim_config
from tabascal.utils.run_tabascal import tabascal_subtraction


def main():

    parser = argparse.ArgumentParser(
        description="Run and end-to-end simulation and source extraction analysis."
    )
    parser.add_argument(
        "-s", "--sim_config", help="Path to the simulation config file."
    )
    parser.add_argument("-t", "--tab_config", help="Path to the tabascal config file.")
    parser.add_argument(
        "-e",
        "--extract_config",
        help="Path to the extraction config file.",
    )
    parser.add_argument(
        "-sp",
        "--sim_path",
        help="Path to the simulation files with prefix given, e.g. '/path/to/sims/sim_name*' ",
    )
    parser.add_argument(
        "-r", "--rfi_amp", type=float, help="Path to the extraction config file."
    )
    parser.add_argument(
        "-rr",
        "--random_seed_offset",
        default=0,
        type=int,
        help="Offset to random seeds used.",
    )
    parser.add_argument(
        "-ra", "--ra", type=float, help="Right Ascension of the observation."
    )
    parser.add_argument(
        "-dec", "--dec", type=float, help="Declination of the observation."
    )
    parser.add_argument("-sx", "--suffix", default="run1", help="Image name suffix.")
    args = parser.parse_args()
    config_path = args.sim_config
    rfi_amp = args.rfi_amp
    r_seed_offset = args.random_seed_offset
    sim_dirs = args.sim_path

    times = {
        "t0": datetime.now(),
    }

    if config_path:
        config = load_config(config_path, config_type="sim")

        if args.ra is not None:
            config["observation"]["ra"] = args.ra

        if args.dec is not None:
            config["observation"]["dec"] = args.dec

        spacetrack_path = os.path.abspath(
            config["rfi_sources"]["tle_satellite"]["spacetrack_path"]
        )
        config["rfi_sources"]["tle_satellite"]["spacetrack_path"] = spacetrack_path

        config["rfi_sources"]["tle_satellite"]["power_scale"] *= rfi_amp
        config["rfi_sources"]["satellite"]["power_scale"] *= rfi_amp
        config["rfi_sources"]["stationary"]["power_scale"] *= rfi_amp

        config["observation"]["random_seed"] += r_seed_offset
        config["ast_sources"]["point"]["random"]["random_seed"] += r_seed_offset
        config["gains"]["random_seed"] += r_seed_offset

        config["output"]["suffix"] = f"{args.rfi_amp:.1e}RFI_{r_seed_offset:.0f}RSEED"

        obs, sim_dir = run_sim_config(obs_spec=config, spacetrack_path=spacetrack_path)
        sim_dirs = [sim_dir]

        times["t0.1"] = datetime.now()
    elif sim_dirs:
        from glob import glob

        sim_dirs = sim_dirs if sim_dirs[-1] == "*" else sim_dirs + "*"
        sim_dirs = glob(sim_dirs)

    if args.tab_config:
        for sim_dir in tqdm(sim_dirs):
            print(f"Running TABASCAL on sim : {sim_dir}")
            times["t1"] = datetime.now()

            print(
                "======================================================================"
            )
            norad_path = os.path.join(sim_dir, "input_data/norad_ids.yaml")
            norad_ids = [int(x) for x in np.loadtxt(norad_path)]
            tab_config = load_config(args.tab_config, config_type="tab")
            tabascal_subtraction(
                config=tab_config,
                sim_dir=sim_dir,
                spacetrack_path=spacetrack_path,
                norad_ids=norad_ids,
            )
            # subprocess.run(f"tabascal -c {args.tab_config} -s {sim_dir}", shell=True, executable="/bin/bash")

            times["t1.1"] = datetime.now()
    if args.extract_config:
        for sim_dir in tqdm(sim_dirs):
            print(f"Running SOURCE EXTRACTION on sim : {sim_dir}")
            times["t2"] = datetime.now()

            print(
                "======================================================================"
            )
            subprocess.run(
                f"extract -c {args.extract_config} -s {sim_dir} -sx {args.suffix}",
                shell=True,
                executable="/bin/bash",
            )
            times["t2.1"] = datetime.now()
    times["t3"] = datetime.now()
    print("\n==================================================================\n")
    try:
        print(f"Simualtion time : {times['t0.1']-times['t0']}")
    except:
        pass
    try:
        print(f"TABASCAL time   : {times['t1.1']-times['t1']}")
    except:
        pass
    try:
        print(f"Extraction time : {times['t2.1']-times['t2']}")
    except:
        pass
    print("========================================")
    print(f"Total time      : {times['t3']-times['t0']}")


if __name__ == "__main__":
    main()
