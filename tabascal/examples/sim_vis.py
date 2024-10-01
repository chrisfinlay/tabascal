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
        "-o", "--overwrite", default="no", type=str2bool, help="Overwrite existing observation."
    )
    args = parser.parse_args()

    obs_spec = load_sim_config(args.config_path)
    obs_spec["output"]["overwrite"] = args.overwrite
    run_sim_config(obs_spec=obs_spec)