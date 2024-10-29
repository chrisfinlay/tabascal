import argparse
import shutil
import glob
import yaml
import sys
import os
import subprocess

from tabascal.utils.yaml import load_config, Tee
from tabascal.utils.extract import extract
from tabascal.utils.flag_data import write_perfect_flags, run_aoflagger

def main():
    parser = argparse.ArgumentParser(
        description="Process a simulation file that has potentially had tabascal run on it."
    )
    parser.add_argument(
        "-c", "--config_path", help="File path to the source extraction config file."
    )
    parser.add_argument(
        "-s", "--sim_dir", help="Path to the directory of the simulation."
    )
    parser.add_argument(
        "-d", "--data", default="ideal,tab,flag1,flag2", help="The data types to analyse. {'ideal', 'tab', 'flag1', 'flag2'}"
    )
    parser.add_argument(
        "-p", "--processes", default="image,extract", help="The types of processing to do. {'image', 'extract', 'pow_spec'}"
    )
    parser.add_argument(
        "-b", "--bash_exec", default="/bin/bash", help="Path to the bash exectuable used to run docker. Default is /bin/bash."
    )
    parser.add_argument(
        "-sp", "--sif_path", default=None, help="Singularity image path if using singularity."
    )
    parser.add_argument(
        "-sx", "--suffix", default=None, help="Image name suffix."
    )
    args = parser.parse_args()
    bash = args.bash_exec
    sim_dir = args.sim_dir
    sif_path = args.sif_path
    suffix = args.suffix

    if suffix is not None:
        suffix = "_" + suffix
    else:
        suffix = ""


    log = open('log_extract.txt', 'w')
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
        raise KeyError("'sim_dir' must be specified in either the config file or as a command line argument.")

    if sif_path is not None:
        sif_path = os.path.abspath(sif_path)
        config["image"]["sif_path"] = sif_path 
    elif config["image"]["sif_path"] is not None:
        sif_path = config["image"]["sif_path"]

    if sim_dir[-1]=="/":
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
            print("\n\n================================================================================")
            print()
            print(f"Flagging {data_col} column of the MS file.")
            
            if flag_type=="perfect":
                thresh = config[key]["flag"]["thresh"]
                write_perfect_flags(ms_path, thresh)
                name = f"{thresh:.1f}sigma{suffix}"
            elif flag_type=="aoflagger":
                run_aoflagger(ms_path, data_col, config[key]["flag"]["strategies"])
                name = f"aoflagger{suffix}" 
            else:
                print("Incorrect flagging type chosen. Must be one of {perfect, aoflagger}.")
            
            wsclean_opts = "".join([f" -{k} {v}" for k, v in config["image"]["params"].items()])
            img_cmd = f"image{singularity} -m {ms_path} -d {data_col} -n {name} -w '{wsclean_opts}'"
            print()
            print(f"Imaging {data_col} column of the MS file.\nUsing {img_cmd}")
            subprocess.run(img_cmd, shell=True, executable=bash)
        
        if "extract" in procs:
            
            if flag_type=="aoflagger":
                name = f"aoflagger{suffix}" 
            elif flag_type=="perfect":
                thresh = config[key]["flag"]["thresh"]
                name = f"{thresh:.1f}sigma{suffix}"
            else:
                print("Incorrect flagging type chosen. Must be one of {perfect, aoflagger}.")
            
            img_path = os.path.join(img_dir, f"{data_col}_{name}-image.fits")
            print()
            print(f"Extracting sources from {img_path}")
            extract(img_path, zarr_path, config["extract"]["sigma_cut"], config["extract"]["beam_cut"], 
                    config["extract"]["thresh_isl"], config["extract"]["thresh_pix"])

        # if "pow_spec" in procs:

    log.close()
    shutil.copy("log_extract.txt", sim_dir)
    os.remove("log_extract.txt")
    sys.stdout = backup

    with open(os.path.join(sim_dir, "extract_config.yaml"), "w") as fp:
        yaml.dump(config, fp)
    

if __name__=="__main__":
    main()
