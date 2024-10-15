import argparse
import subprocess
from datetime import datetime
from tabascal.utils.yaml import load_config, run_sim_config
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
        "-r", "--rfi_amp", type=float, help="Path to the extraction config file."
    )
    args = parser.parse_args()
    config_path = args.sim_config
    config = load_config(config_path)

    if args.rfi_amp is not None:
        config["rfi_sources"]["satellite"]["power_scale"] = args.rfi_amp
        config["rfi_sources"]["stationary"]["power_scale"] = args.rfi_amp

    config["output"]["suffix"] = f"{args.rfi_amp:.1e}RFI"

    times = {"t0": datetime.now(),}

    obs, sim_dir = run_sim_config(obs_spec=config)

    times["t1"] = datetime.now()

    print("======================================================================")
    tabascal_subtraction(args.tab_config, sim_dir)
    # subprocess.run(f"tabascal -c {args.tab_config} -s {sim_dir}", shell=True, executable="/bin/bash")

    times["t2"] = datetime.now()

    print("======================================================================")
    subprocess.run(f"extract -c {args.extract_config} -s {sim_dir}", shell=True, executable="/bin/bash")

    times["t3"] = datetime.now()
    print("\n==================================================================\n")
    print(f"Simualtion time : {times["t1"]-times["t0"]}")
    print(f"TABASCAL time   : {times["t2"]-times["t1"]}")
    print(f"Extraction time : {times["t3"]-times["t2"]}")
    print("========================================")
    print(f"Total time      : {times["t3"]-times["t0"]}")

if __name__=="__main__":
    main()