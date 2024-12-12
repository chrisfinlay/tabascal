#!/usr/bin/env python

from daskms import xds_from_ms, xds_to_table

import xarray as xr
import dask.array as da
import dask
import os

import argparse
import subprocess
import shutil

# aoflagger -strategy ../yaml_configs/target/firstpass.rfis tab1_obs_16A_450T-0440-1338_256I_001F-1.227e+09-1.227e+09_100PAST_000GAST_000EAST_1SAT_0GRD_1.0e+00RFI/tab1_obs_16A_450T-0440-1338_256I_001F-1.227e+09-1.227e+09_100PAST_000GAST_000EAST_1SAT_0GRD_1.0e+00RFI.ms/


def write_perfect_flags(ms_path: str, n_sigma: float = 3.0):

    xds = xds_from_ms(ms_path)[0]

    if n_sigma > 0:
        flags = da.abs(xds.CAL_DATA - xds.AST_MODEL_DATA) > (
            n_sigma * xds.SIGMA * da.sqrt(2)
        )
    else:
        flags = xr.zeros_like(xds.DATA).astype(bool)

    xds = xds.assign(FLAG=flags)
    dask.compute(
        xds_to_table(
            [
                xds,
            ],
            ms_path,
            ["FLAG"],
        )
    )

    print()
    print(f"Flag Threshold : {n_sigma: .1f} sigma")
    print(f"Flag Rate      : {100*flags.data.mean().compute(): .1f} %")

    return flags


def run_aoflagger(
    ms_path: str,
    data_column: str = "DATA",
    strategy_paths: list = None,
    sif_path: str = None,
    bash_exec: str = "/bin/bash",
):

    if ms_path[-1] == "/":
        ms_path = ms_path[:-1]

    data_dir, ms_file = os.path.split(os.path.abspath(ms_path))

    docker_opts = "--rm -v /etc/group:/etc/group -v /etc/passwd:/etc/passwd -v /etc/shadow:/etc/shadow -v/etc/sudoers.d:/etc/sudoers.d -e HOME=${HOME} --user=`id -ur`"
    container_cmd = f"docker run {docker_opts} -v {data_dir}:/data --workdir /data stimela/aoflagger:latest"

    if sif_path is not None:
        sif_path = os.path.abspath(sif_path)
        container_cmd = (
            f"singularity exec --bind {data_dir}:/data --pwd /data {sif_path}"
        )

    if strategy_paths is not None:
        write_perfect_flags(ms_path, 0)
        for strategy_path in strategy_paths:

            strategy_path = os.path.abspath(strategy_path)
            shutil.copy(strategy_path, data_dir)
            strategy_file = os.path.split(strategy_path)[1]
            strategy = f"-strategy /data/{strategy_file}"

            aoflag_cmd = f"{container_cmd} aoflagger -column {data_column} {strategy} /data/{ms_file}"
            subprocess.run(aoflag_cmd, shell=True, executable=bash_exec)

        print()
        print(
            f"Strategies : {[os.path.split(strategy)[1] for strategy in strategy_paths]}"
        )

    else:
        aoflag_cmd = f"{container_cmd} aoflagger /data/{ms_file}"
        subprocess.run(aoflag_cmd, shell=True, executable=bash_exec)
        print()

    flags = xds_from_ms(ms_path)[0].FLAG.data
    print(f"Flag Rate      : {100*flags.mean().compute(): .1f} %")

    return flags


def main():

    parser = argparse.ArgumentParser(
        description="Flag CAL_DATA lying a certain threshold away from the AST_MODEL_DATA. Requires tabascal simulated MS files."
    )
    parser.add_argument(
        "-m", "--ms_path", required=True, help="File path to the Measurement Set."
    )
    parser.add_argument(
        "-s",
        "--n_sigma",
        default=3.0,
        type=float,
        help="Threshold in number of std of noise given by SIGMA column. 0 unflags everything.",
    )
    parser.add_argument(
        "-d", "--data_col", default="DATA", help="Data column to run AOFlagger on."
    )
    parser.add_argument(
        "-ao",
        "--aoflagger",
        default=None,
        action=argparse.BooleanOptionalAction,
        help="Whether to use AOFlagger to flag the data.",
    )
    parser.add_argument(
        "-st",
        "--strategy_paths",
        default=None,
        help="list of paths to AOFlagger strategies.",
    )
    parser.add_argument(
        "-sp",
        "--sif_path",
        default=None,
        help="Paths to AOFlagger singularity image.",
    )
    args = parser.parse_args()
    ms_path = args.ms_path
    n_sigma = args.n_sigma
    strategy_paths = args.strategy_paths
    data_col = args.data_col

    if args.aoflagger is not None:
        if strategy_paths is not None:
            strategy_paths = strategy_paths.split(",")
        flags = run_aoflagger(ms_path, data_col, strategy_paths, args.sif_path)
    else:
        flags = write_perfect_flags(ms_path, n_sigma)


if __name__ == "__main__":
    main()
