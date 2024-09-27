import argparse
import subprocess
import os

def main():

    parser = argparse.ArgumentParser(
        description="Use WSclean to image columns of an MS file."
    )
    parser.add_argument(
        "--ms_path", required=True, help="File path to the MS file."
    )
    parser.add_argument(
        "--data_col", default="DATA", type=str, help="The names of the data columns in the MS file to image."
    )
    parser.add_argument(
        "--name_suffix", default=None, type=str, help="Image name suffix. Iamge name will be '{--data_col}_{--name_suffix}-image.fits'."
    )
    parser.add_argument(
        "--wsclean_opts", default="-size 2048 2048 -scale 2asec -niter 50000 -mgain 0.3 -auto-threshold 1 -pol xx -weight briggs -0.5 -auto-mask 3", help="WSclean options."
    )
    parser.add_argument(
        "--bash_exec", default="/bin/bash", help="Path to the bash exectuable used to run docker."
    )

    args = parser.parse_args()
    ms_path = os.path.abspath(args.ms_path)
    bash_exec = args.bash_exec
    wsclean_opts = args.wsclean_opts
    suffix = args.name_suffix

    if suffix is not None:
        suffix = "_" + suffix

    data_cols = args.data_col.upper().split(",")

    if ms_path[-1]=="/":
        ms_path = ms_path[:-1]
        
    data_dir, ms_file = os.path.split(ms_path)

    docker_opts = "--rm -v /etc/group:/etc/group -v /etc/passwd:/etc/passwd -v /etc/shadow:/etc/shadow -v/etc/sudoers.d:/etc/sudoers.d -e HOME=${HOME} --user=`id -ur`"
    docker_cmd = f"docker run {docker_opts} -v {data_dir}:/data --workdir /data chrisjfinlay/wsclean:kern8"

    for data_col in data_cols:

        if "flag" in data_col.lower():
            n_sigma = data_col.lower().replace("flag", "")
            flag_cmd = f"flag-data --ms_path {ms_path} --n_sigma {n_sigma}"
            subprocess.run(flag_cmd, shell=True, executable=bash_exec)
        else:
            wsclean_cmd = f"{docker_cmd} wsclean {wsclean_opts} -data-column {data_col} -name {data_col}{suffix} {ms_file}"
            subprocess.run(wsclean_cmd, shell=True, executable=bash_exec)

