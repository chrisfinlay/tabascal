# tabascal

[![DOI:10.1093/mnras/stad1979](https://zenodo.org/badge/DOI/10.1093/mnras/stad1979.svg)](https://doi.org/10.1093/mnras/stad1979)
[![DOI:10.48550/arXiv.2502.00106](https://img.shields.io/badge/arXiv-2502.00106-b31b1b.svg)](https://doi.org/10.48550/arXiv.2502.00106)
[![Documentation Status](https://readthedocs.org/projects/tabascal/badge/?version=latest)](https://tabascal.readthedocs.io/en/latest/?badge=latest)

**T**r**A**jectory **BA**sed RFI **S**ubtraction and **CAL**ibration (tabascal)
of radio interferometry data. A source to visibility model for RFI sources
including certain near-field effects. Visibility data is jointly calibrated and
cleaned from specific RFI contamination by modelling the RFI signal in the
visibilities.

`tabascal` is written in [JAX](https://jax.readthedocs.io/en/latest/notebooks/quickstart.html) 
and [Dask](https://www.dask.org) and can therefore use GPUs and/or CPUs and be distributed across clusters of these compute units.

## Installation

The following instructions are expected to work on Linux machine. If you are running Windows it is recommended to use WSL. If you are running Mac then your mileage may vary. If all else fails there is the [Docker install](#docker).

```bash
git clone https://github.com/chrisfinlay/tabascal.git
```
<!-- 
### Conda Environment (Recommended)

Create a conda environment with all the dependencies including JAX with optional GPU support.

#### GPU Enabled
```bash
conda env create -n tab_env -f tabascal/env_gpu.yaml
```
or
#### CPU Only
```bash
conda env create -n tab_env -f tabascal/env_cpu.yaml
```
Then proceed to activate the conda environment and install `tabascal`
```bash
conda activate tab_env
pip install -e tabascal/
``` -->

### Pure `pip` install

You can install `tabascal` with pip alone inside an environment of your choice with optional GPU support.

#### GPU Enabled
```bash
pip install -e ./tabascal/[gpu]
```
or
#### CPU Only
```bash
pip install -e ./tabascal/
```


### Docker

If you are having trouble with the `pip` install method you can try with [Docker](https://www.docker.com/) instead. The provided [Dockerfile](Dockerfile) can be used to build an image which should, "in principle", run on any machine.

Assuming you have cloned this git repository into you current working directory then you can either:

1. build an image with the latest tabascal.
2. download a working but older version of a tabascal image. 

First we can simplify by setting the `TAB_DIR` environment variable using

```bash
TAB_DIR=$(pwd)
```

#### 1. Build an image

```bash
TAB_IMG=tabascal:latest
docker build -t ${TAB_IMG} ./tabascal/ 
```

#### 2. Download an image

```bash
TAB_IMG=chrisjfinlay/tabascal:0.0.1
docker pull ${TAB_IMG}
```

After running one of the above you can run the docker image using the appropriate command below:

#### Linux
```bash
docker run -it -v ${TAB_DIR}:/data -u $(id -u):$(id -g) ${TAB_IMG} bash
```

#### Mac
```bash
docker run -it -v ${TAB_DIR}:/data ${TAB_IMG} bash
```

For more complex tabascal installs using docker you can adapt the [Dockerfile](Dockerfile) to your needs. 


<!-- ### GPU 
 
To enable GPU compute you need the GPU version of `jaxlib` installed. The easiest way is using pip, as is done using the `env_gpu.yaml`, otherwise, refer to the JAX installation [documentation](https://jax.readthedocs.io/en/latest/installation.html). -->

## Simulations and Analysis

`tabascal` now includes the facility to define a simulation using a YaML configuration file. There is a general command line interface to run these simulations allowing one to change certain parameters on the file as well as in the configuration file. All input data is copied into the output simulation directory to allow one to run an identical simulation with ease. Inside [tabascal/analysis/yaml_configs/target](https://github.com/chrisfinlay/tabascal/tree/main/tabascal/analysis/yaml_configs/target) are a set of config files to get you started. There are also example data files which are used for including predefined astronomical and rfi models. They are all `csv` files with file extensions to help distinguish them. These reside in [tabascal/analysis/yaml_configs/aux_data](https://github.com/chrisfinlay/tabascal/tree/main/tabascal/analysis/yaml_configs/aux_data).

### Including TLE-based satelllites

You will need to provide [Space-Track](https://www.space-track.org/auth/login) login details as a YaML file. The filename can be `spacetrack_login.yaml` for example and should look like 

```yaml
username: user@email.com
password: password123
```

### Running a simulation

To run a simulation of a target field with 100 randomly distributed point sources and some GPS satellites simply run 

```bash
sim-vis -c sim_target_32A.yaml -st spacetrack_login.yaml
```

You can run the help function to see what other command line options there are.

```bash
sim-vis -h
```

### Analysis

Downstream analysis such as flagging, RFI subtraction, imaging, and source extraction can be performed through such configuration files as well. This is currently still in development where the `tabascal` RFI subtraction algorithm itself is not yet publically available. However, a full end to end analysis pipeline is available. Individual portions can be accessed through the command line scripts: `flag-data`, `image`, and  `src-extract`, with example configs in [tabascal/analysis/yaml_configs/target](https://github.com/chrisfinlay/tabascal/tree/main/tabascal/analysis/yaml_configs/target). All three of these can be perfomed in a single command line script by using `extract`. See the help documentation of these scripts for further details.  

### Config File Definitions

The configuration files for simulation and analysis have many options with set defaults such that minimal configurations can be set unless more is required.

The base configuration files with defualts and definitions reside in [tabascal/tabascal/data/config_files](https://github.com/chrisfinlay/tabascal/tree/main/tabascal/data/config_files)

## Output Data Structure

Once a simulation has been run then a directory will be created to store all input and output data used for the simulation. The location for this directory is defined in the simulation config file under 

```yaml
output:
    path: output_directory_path
    prefix: simulation_name_prefix
```

The directory name will have a prefix as defined in the config file and include many of the other simulation configuration parameters in the directory name. For example, when using the `target_obs_32A.yaml` config file, the name will be `pnt_src_obs_32A_450T-0000-0898_1025I_001F-1.227e+09-1.227e+09_100PAST_000GAST_000EAST_3SAT_0GRD_1.0e+00RFI`. The resulting directory structure inside this base directory is as follows:

```
- sim_name/
    - sim_name.zarr/
    - sim_name.ms/
    - AngularSeps.png
    - SourceAltitude.png
    - UV.png
    - log_sim_xxxxxx.txt
    - input_data/
        - MeerKAT.itrf.txt
        - norad_ids.yaml
        - norad_satellite.rfimodel
        - sim_config.yaml 
```

The `.zarr` and `.ms` files contain the actual visibilities that are simulated with the `.zarr` file containing intermediate values used in calculating certain quantities. The `.png` files are diagnostoc plots to check the angular separations between the RFI sources and the target direction, The source altitude of the target direction, and the UV coverage of the baselines. `log_sim_xxxxxx.txt` contains the output that was displayed when originally running the simulation. Finally, `input_data` contains all the required data to rerun the simulation exactly for reproducibility especially if the data hungry visibilities need to be deleted for some reason.

## Zarr Output

The `.zarr` files is most easily read using [Xarray](https://xarray.dev/) in a [Jupyter](https://jupyter.org/) notebook. The following code will read the simulation data into an [`xarray.Dataset`](https://docs.xarray.dev/en/stable/generated/xarray.Dataset.html) object.

```python
import xarray as xr

zarr_path = "path/to/data_dir/sim_name/sim_name.zarr/"
xds = xr.open_zarr(zarr_path)

xds
```

The structure of the `.zarr` file is as follows:

### Coordinates

| Name | Description  |
|-------|-------------|
| `ant` | Antenna index |
| `bl`  | Baseline index |
| `enu` | East, North Up {m} |
| `freq` | Frequencies {Hz} |
| `geo` | Latitude, Longitude, Elevation {deg, deg, m} |
| `itrf` | International Terrestrial Reference Frame (ECEF) {m} |
| `lmn` | Local astronomcal cosine coordinates |
| `radec` | Right Ascension, Declination {deg} |
| `time` | Elapsed observation time centroids {s} |
| `time_fine` | Fine grain time centroids {s} |
| `time_mjd` | Modified Julian Date time {days} |
| `time_mjd_fine` | Fine grain Modified Julian Date time {days} |
| `tle` | Two-line elements for satellites |
| `uvw` | Local antenna coordinates {m} |
| `xyz` | Geocentric Celestial Reference Frame (ECI) {m} |

### Data Variables

| Name | Description |
|------|-------------|
| `SEFD` | System Equivalent Flux Density {Jy} |
| `antenna1` | Antenna 1 index |
| `antenna2` | Antenna 2 index |
| `ants_itrf` | Antenna ITRF coordinates {m} |
| `ants_uvw` | Antenna UVW coordinates {m} |
| `ants_xyz` | Antenna XYZ (ECI) coorindates {m} |
| `ast_p_I` | Astronomical point source intensities {Jy} |
| `ast_p_lmn` | Astronomical point source positions |
| `bl_uvw` | Baseline UVW coordinates {m} |
| `flags` | RFI flags based on 3sigma from truth |
| `gains_ants` | Antenna gains |
| `noise_data` | Visibility noise realisation {Jy} |
| `noise_std` | Visibility noise standard deviation {Jy} |
| `norad_ids` | NORAD IDs for TLE-based satellites |
| `rfi_tle_sat_A` | Modulated satellite signal amplitudes {Jy^0.5} |
| `rfi_tle_sat_ang_sep` | Satellite angular separation from target direction {deg} |
| `rfi_tle_sat_orbit` | Satellite TLE orbit parameters |
| `rfi_tle_sat_xyz` | Satellite XYZ (ECI) coordinates {m} |
| `time_idx` | Time index to map from `time_fine` to `time` |
| `vis_ast` | Astronomical visibility component {Jy} |
| `vis_calibrated` | Perfectlty calibrated visibilities {Jy} |
| `vis_obs` | Observed (uncalibrated) visibilities {Jy} |
| `vis_rfi` | RFI visibility component {Jy} |

### Attributes

| Name | Description |
|------|-------------|
| `chan_width` | Frequency channel bandwidth {Hz} |
| `dish_diameter` | Dish diameter {m} |
| `int_time` | Integration time per sample {s} |
| `n_ant` | Number of antennas |
| `n_ast_e_src` | Number of astronomical exponential profile sources |
| `n_ast_g_src` | Number of astronomical Gaussian profile sources |
| `n_ast_p_src` | Number of astronomical point sources |
| `n_ast_src` | Number of astronomical sources `n_ast_e_src` + `n_ast_g_src` + `n_ast_p_src` |
| `n_bl` | Number of baselines |
| `n_freq` | Number of frequency channels |
| `n_int_samples` | Number of integration samples per time sample |
| `n_sat_src` | Number of satellite RFI sources |
| `n_stat_src` | Number of stationary RFI sources |
| `n_time` | Number of time steps |
| `n_time_fine` | Number of fine grained time steps |
| `target_dec` | Declination of the target direction {deg} |
| `target_name` | Name of the target |
| `target_ra` | Right Ascension of the target direction {deg} |
| `tel_elevation` | Telescope elevation {m} |
| `tel_latitude` | Telescope latitude {deg} |
| `tel_longitude` | Telescope logitude {deg} |
| `tel_name` | Telescope name |

## Measurement Set Output

Measurement sets allow the addition of non-standard data columns. The simulator in tabascal takes advantage of this and adds the following columns to help with debugging and analysis.

### Standard Columns

* `DATA` : Observed data which includes gains and noise.
* `CORRECTED_DATA` : Filled with zeros or the data of ones choice when calling the `write_ms` function.
* `MODEL_DATA` : Filled with zeros as it will be used by `WSCLEAN` when imaging.

### Non-standard Columns

* `CAL_DATA` : Observed data (`DATA`) where the true gain solutions have been applied.
* `AST_MODEL_DATA` : The astronomical visibilities only with perfect gains and no noise. 
* `RFI_MODEL_DATA` : The RFI visibilities only with perfect gains and no noise.
* `AST_DATA` : The same as `AST_MODEL_DATA` but with the noise added. 
* `RFI_DATA` : The same as `RFI_MODEL_DATA` but with the noise added. 
* `NOISE_DATA` : The complex noise that is added to the above datasets. 

<!-- ## Documentation

[https://tabascal.readthedocs.io/en/latest/](https://tabascal.readthedocs.io/en/latest/) -->

## Citing tabascal

```
@ARTICLE{Finlay2023,
       author = {{Finlay}, Chris and {Bassett}, Bruce A. and {Kunz}, Martin and {Oozeer}, Nadeem},
        title = "{Trajectory-based RFI subtraction and calibration for radio interferometry}",
      journal = {Monthly Notices of the Royal Astronomical Society},
         year = 2023,
        month = sep,
       volume = {524},
       number = {3},
        pages = {3231-3251},
          doi = {10.1093/mnras/stad1979},
archivePrefix = {arXiv},
       eprint = {2301.04188},
}
```

```
@ARTICLE{Finlay2025,
       author = {{Finlay}, Chris and {Bassett}, Bruce A. and {Kunz}, Martin and {Oozeer}, Nadeem},
        title = "{TABASCAL II: Removing Multi-Satellite Interference from Point-Source Radio Astronomy Observations}",
      journal = {arXiv e-prints},
         year = 2025,
        month = jan,
          doi = {10.48550/arXiv.2502.00106},
archivePrefix = {arXiv},
       eprint = {2502.00106},
}
```
