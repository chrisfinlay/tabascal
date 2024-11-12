import argparse
import os

from tabascal.utils.config import load_config, run_sim_config

def main():
    parser = argparse.ArgumentParser(
        description="Simulate an observation defined by a YAML config file."
    )
    parser.add_argument(
        "-c", "--config_path", help="File path to the observation config file."
    )
    parser.add_argument(
        "-r", "--rfi_amp", default=1.0, type=float, help="Scale the RFI power. Default is 1.0"
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
        "-o", "--overwrite", default=False, action=argparse.BooleanOptionalAction, help="Overwrite existing observation."
    )
    parser.add_argument(
        "-st", "--spacetrack", help="Path to Space-Track login details."
    )
    args = parser.parse_args()
    rfi_amp = args.rfi_amp
    spacetrack_path = args.spacetrack
    
    obs_spec = load_config(args.config_path, config_type="sim")

    config_st_path = obs_spec["rfi_sources"]["tle_satellite"]["spacetrack_path"]
    if spacetrack_path:
        obs_spec["rfi_sources"]["tle_satellite"]["spacetrack_path"] = os.path.abspath(spacetrack_path)
    elif config_st_path:
        config_st_path = os.path.abspath(config_st_path)
        obs_spec["rfi_sources"]["tle_satellite"]["spacetrack_path"] = config_st_path
        spacetrack_path = config_st_path

    obs_spec["rfi_sources"]["tle_satellite"]["power_scale"] *= rfi_amp
    obs_spec["rfi_sources"]["satellite"]["power_scale"] *= rfi_amp
    obs_spec["rfi_sources"]["stationary"]["power_scale"] *= rfi_amp

    obs_spec["output"]["overwrite"] = args.overwrite

    if args.n_ant is not None:
        obs_spec["telescope"]["n_ant"] = args.n_ant

    if args.n_int is not None:
        obs_spec["observation"]["n_int"] = args.n_int

    suffix = obs_spec["output"]["suffix"]
    if suffix:
        suffix = f"{rfi_amp:.1e}RFI_" + suffix
    else:
        suffix = f"_{rfi_amp:.1e}RFI"
        obs_spec["output"]["suffix"] = f"{rfi_amp:.1e}RFI"

    if args.SEFD is not None:
        obs_spec["observation"]["SEFD"] = args.SEFD

    if args.int_time is not None:
        obs_spec["observation"]["int_time"] = args.int_time

    if args.n_time is not None:
        obs_spec["observation"]["n_time"] = args.n_time
    
    return run_sim_config(obs_spec=obs_spec, spacetrack_path=spacetrack_path)

if __name__=="__main__":
    obs, obs_path = main()