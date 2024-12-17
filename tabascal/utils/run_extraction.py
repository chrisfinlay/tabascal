import argparse
import shutil
import glob
import yaml
import sys
import os
import subprocess

from tabascal.utils.config import load_config, Tee
from tabascal.utils.extract import extract
from tabascal.utils.flag_data import write_perfect_flags, run_aoflagger
from tabascal.utils.results_to_MS import write_results


def main():
    parser = argparse.ArgumentParser(
        description="Process a simulation file that has potentially had tabascal run on it."
    )
    parser.add_argument(
        "-c",
        "--config_path",
        required=True,
        help="File path to the source extraction config file.",
    )
    parser.add_argument(
        "-s", "--sim_dir", help="Path to the directory of the simulation."
    )
    parser.add_argument(
        "-d",
        "--data",
        default="perfect,ideal,tab,flag1,flag2",
        help="The data types to analyse. {'perfect', 'ideal', 'tab', 'flag1', 'flag2'}",
    )
    parser.add_argument(
        "-p",
        "--processes",
        default="image,extract",
        help="The types of processing to do. {'image', 'extract', 'pow_spec'}",
    )
    parser.add_argument(
        "-b",
        "--bash_exec",
        default="/bin/bash",
        help="Path to the bash exectuable used to run docker. Default is /bin/bash.",
    )
    parser.add_argument(
        "-sp",
        "--sif_path",
        default=None,
        help="Singularity image path if using singularity.",
    )
    parser.add_argument("-isx", "--im_suffix", default=None, help="Image name suffix.")
    parser.add_argument("-tsx", "--tab_suffix", default=None, help="Image name suffix.")
    args = parser.parse_args()
    bash = args.bash_exec
    sim_dir = args.sim_dir
    sif_path = args.sif_path
    im_suffix = args.im_suffix
    tab_suffix = args.tab_suffix

    im_suffix, tab_suffix = [
        "_" + suffix if suffix else "" for suffix in [im_suffix, tab_suffix]
    ]

    model_name = "fixed_orbit_rfi_full_fft_standard_padded_model"
    tab_path = os.path.join(sim_dir, f"results/map_pred_{model_name}{tab_suffix}.zarr")

    from tabascal.utils.tle import id_generator

    log_path = f"log_extract_{id_generator()}.txt"
    log = open(log_path, "w")
    backup = sys.stdout
    sys.stdout = Tee(sys.stdout, log)

    config = load_config(args.config_path)

    if sim_dir is not None:
        sim_dir = os.path.abspath(sim_dir)
        config["data"]["sim_dir"] = sim_dir
    elif config["data"]["sim_dir"] is not None:
        sim_dir = os.path.abs(config["data"]["sim_dir"])
        config["data"]["sim_dir"] = sim_dir
    else:
        raise KeyError(
            "'sim_dir' must be specified in either the config file or as a command line argument."
        )

    if sif_path is not None:
        sif_path = os.path.abspath(sif_path)
        config["image"]["sif_path"] = sif_path
    elif config["image"]["sif_path"] is not None:
        sif_path = config["image"]["sif_path"]

    if sim_dir[-1] == "/":
        sim_dir = sim_dir[:-1]

    f_name = os.path.split(sim_dir)[1]
    zarr_path = os.path.join(sim_dir, f"{f_name}.zarr")
    ms_path = os.path.join(sim_dir, f"{f_name}.ms")
    img_dir = os.path.join(sim_dir, "images")

    os.makedirs(img_dir, exist_ok=True)

    # print(config)
    # sys.exit(0)

    print()
    print(f"Working on {ms_path}")

    data = args.data.lower().split(",")
    procs = args.processes.lower().split(",")

    if sif_path is not None:
        singularity = f" -s {sif_path}"
    else:
        singularity = ""

    for key in data:

        data_col = config[key]["data_col"]
        flag_type = config[key]["flag"]["type"]

        if "image" in procs:
            print(
                "\n\n================================================================================"
            )
            print()
            print(f"Flagging {data_col} column of the MS file.")

            if flag_type == "perfect":
                thresh = config[key]["flag"]["thresh"]
                write_perfect_flags(ms_path, thresh)
                if key == "tab":
                    name = f"{thresh:.1f}sigma{im_suffix}{tab_suffix}"
                    write_results(ms_path, tab_path)
                else:
                    name = f"{thresh:.1f}sigma{im_suffix}"

                reflagged = True

            elif flag_type == "aoflagger":
                rerun_aoflagger = False
                flags, reflagged = run_aoflagger(
                    ms_path,
                    data_col,
                    config[key]["flag"]["strategies"],
                    config[key]["flag"]["sif_path"],
                    rerun_aoflagger=rerun_aoflagger,
                )
                name = f"aoflagger{im_suffix}"
            else:
                print(
                    "Incorrect flagging type chosen. Must be one of {perfect, aoflagger}."
                )

            if reflagged:
                wsclean_opts = "".join(
                    [f" -{k} {v}" for k, v in config["image"]["params"].items()]
                )
                img_cmd = f"image{singularity} -m {ms_path} -d {data_col} -n {name} -w '{wsclean_opts}'"
                print()
                print(f"Imaging {data_col} column of the MS file.\nUsing {img_cmd}")
                subprocess.run(img_cmd, shell=True, executable=bash)

        else:
            reflagged = False

        if "extract" in procs and reflagged:

            if flag_type == "aoflagger":
                name = f"aoflagger{im_suffix}"
            elif flag_type == "perfect":
                if key == "tab":
                    name = f"{thresh:.1f}sigma{im_suffix}{tab_suffix}"
                else:
                    name = f"{thresh:.1f}sigma{im_suffix}"
                thresh = config[key]["flag"]["thresh"]
            else:
                print(
                    "Incorrect flagging type chosen. Must be one of {perfect, aoflagger}."
                )

            img_path = os.path.join(img_dir, f"{data_col}_{name}-image.fits")
            print()
            print(f"Extracting sources from {img_path}")
            extract(
                img_path,
                zarr_path,
                config["extract"]["sigma_cut"],
                config["extract"]["beam_cut"],
                config["extract"]["thresh_isl"],
                config["extract"]["thresh_pix"],
            )

    log.close()
    shutil.copy(log_path, sim_dir)
    os.remove(log_path)
    sys.stdout = backup

    with open(
        os.path.join(img_dir, f"extract_config{im_suffix}{tab_suffix}.yaml"), "w"
    ) as fp:
        yaml.dump(config, fp)


if __name__ == "__main__":
    main()
