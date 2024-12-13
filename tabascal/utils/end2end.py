import os
import argparse
import subprocess

from datetime import datetime

from tqdm import tqdm

import numpy as np

from tabascal.utils.config import load_config, run_sim_config


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
        "-st",
        "--spacetrack_path",
        help="Path to the SpaceTrack login details.",
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
    all_sim_dir = os.path.split(sim_dirs)[0]
    spacetrack_path = args.spacetrack_path
    suffix = args.suffix
    tab_config = args.tab_config

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
        print(f"Simualtion time : {times['t0.1']-times['t0']}")

    elif sim_dirs:
        from glob import glob

        sim_dirs = sim_dirs if sim_dirs[-1] == "*" else sim_dirs + "*"
        sim_dirs = glob(sim_dirs)

    if tab_config:

        completed = []
        failed = []
        n_sim = len(sim_dirs)
        for i, sim_dir in enumerate(sim_dirs):
            sim_name = os.path.split(sim_dir)[1]
            print(f"Completed sims : {len(completed)}")
            print(f"Failed sims    : {len(failed)}")
            print(f"Running sim number : {i} / {n_sim}")
            print(f"Running TABASCAL on sim : {sim_dir}")
            times["t1"] = datetime.now()

            print(
                "======================================================================"
            )
            sub = subprocess.run(
                f"tabascal -c {tab_config} -s {sim_dir} -st {spacetrack_path} -sx {suffix}",
                shell=True,
                executable="/bin/bash",
            )
            if sub.returncode == 0:
                completed.append(sim_name)
            else:
                print(f"TABASCAL FAILED for {sim_dir}")
                failed.append(sim_name)

            times["t1.1"] = datetime.now()
            print(f"TABASCAL time   : {times['t1.1']-times['t1']}")

    if args.extract_config:
        for sim_dir in tqdm(sim_dirs):
            print(f"Running SOURCE EXTRACTION on sim : {sim_dir}")
            times["t2"] = datetime.now()

            print(
                "======================================================================"
            )
            subprocess.run(
                f"extract -c {args.extract_config} -s {sim_dir} -sx {suffix}",
                shell=True,
                executable="/bin/bash",
            )
            times["t2.1"] = datetime.now()
            print(f"Extraction time : {times['t2.1']-times['t2']}")

    times["t3"] = datetime.now()
    print("========================================")
    print(f"Total time      : {times['t3']-times['t0']}")

    if tab_config:
        print()
        print(f"Completed TABASCAL runs : {len(completed)}")
        print(f"Failed TABASCAL runs : {len(failed)}")
        for fail in failed:
            print(fail)

        with open(os.path.join(all_sim_dir, f"tab_failed_{suffix}.txt"), "w") as fp:
            fp.write(f"Failed : {len(failed)} / {len(sim_dirs)} \n")
            for fail in failed:
                fp.write(f"{fail} \n")
        with open(os.path.join(all_sim_dir, f"tab_completed_{suffix}.txt"), "w") as fp:
            fp.write(f"Completed : {len(completed)} / {len(sim_dirs)} \n")
            for success in completed:
                fp.write(f"{success} \n")


if __name__ == "__main__":
    main()
