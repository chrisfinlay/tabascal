import argparse

from tabascal.utils.yaml import load_sim_config, run_sim_config
from tabascal.utils.tools import str2bool

def main():
    parser = argparse.ArgumentParser(
        description="Simulate an observation defined by a YAML config file."
    )
    parser.add_argument(
        "-c", "--config_path", help="File path to the observation config file."
    )
    parser.add_argument(
        "-r", "--rfi_amp", default=None, type=float, help="Scale the RFI power."
    )
    parser.add_argument(
        "-a", "--n_ant", default=None, type=int, help="Number of antennas to include."
    )
    parser.add_argument(
        "-i", "--n_int", default=None, type=int, help="Number of integration samples."
    )
    parser.add_argument(
        "-n", "--SEFD", default=None, type=float, help="System Equivalent flux density in Jy. Same across frequency and antennas."
    )
    parser.add_argument(
        "-dt", "--int_time", default=None, type=float, help="Time step in seconds."
    )
    parser.add_argument(
        "-nt", "--n_time", default=None, type=int, help="Number of time steps."
    )
    parser.add_argument(
        "-o", "--overwrite", default="no", type=str2bool, help="Overwrite existing observation."
    )
    args = parser.parse_args()

    obs_spec = load_sim_config(args.config_path)

    obs_spec["output"]["overwrite"] = args.overwrite

    if args.n_ant is not None:
        obs_spec["telescope"]["n_ant"] = args.n_ant

    if args.n_int is not None:
        obs_spec["observation"]["n_int"] = args.n_int

    if args.rfi_amp is not None:
        obs_spec["rfi_sources"]["satellite"]["power_scale"] = args.rfi_amp
        obs_spec["rfi_sources"]["stationary"]["power_scale"] = args.rfi_amp

    obs_spec["output"]["suffix"] = f"{args.rfi_amp:.1e}RFI"

    if args.SEFD is not None:
        obs_spec["observation"]["SEFD"] = args.SEFD

    if args.int_time is not None:
        obs_spec["observation"]["int_time"] = args.int_time

    if args.n_time is not None:
        obs_spec["observation"]["n_time"] = args.n_time
    
    return run_sim_config(obs_spec=obs_spec)

if __name__=="__main__":
    main()