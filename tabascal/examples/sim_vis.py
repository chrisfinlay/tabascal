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
        "-r", "--rfi_amp", default=1.0, type=float, help="Scale the RFI power."
    )
    parser.add_argument(
        "-o", "--overwrite", default="no", type=str2bool, help="Overwrite existing observation."
    )
    args = parser.parse_args()

    obs_spec = load_sim_config(args.config_path)

    obs_spec["output"]["overwrite"] = args.overwrite
    obs_spec["rfi_sources"]["satellite"]["power_scale"] = args.rfi_amp
    obs_spec["rfi_sources"]["stationary"]["power_scale"] = args.rfi_amp
    
    run_sim_config(obs_spec=obs_spec)