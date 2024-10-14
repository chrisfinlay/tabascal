# tabascal

[![DOI:10.1093/mnras/stad1979](https://zenodo.org/badge/DOI/10.1093/mnras/stad1979.svg)](https://doi.org/10.1093/mnras/stad1979)
[![Documentation Status](https://readthedocs.org/projects/tabascal/badge/?version=latest)](https://tabascal.readthedocs.io/en/latest/?badge=latest)

**T**r**A**jectory **BA**sed RFI **S**ubtraction and **CAL**ibration (tabascal)
of radio interferometry data. A source to visibility model for RFI sources
including certain near-field effects. Visibility data is jointly calibrated and
cleaned from specific RFI contamination by modelling the RFI signal in the
visibilities.

`tabascal` is written in [JAX](https://jax.readthedocs.io/en/latest/notebooks/quickstart.html) 
and [Dask](https://www.dask.org) and can therefore use GPUs and/or CPUs and be distributed across clusters of these compute units.

## Installation

```bash
git clone https://github.com/chrisfinlay/tabascal.git
```

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
```

### Pure `pip` install

Alternatively, you can install `tabascal` with pip alone inside an enivironment of your choice, again, with optional GPU support.

#### GPU Enabled
```bash
pip install -e ./tabascal/[gpu]
```
or
#### CPU Only
```bash
pip install -e ./tabascal/
```

### GPU 
 
To enable GPU compute you need the GPU version of `jaxlib` installed. The easiest way is using pip, as is done using the `env_gpu.yaml`, otherwise, refer to the JAX installation [documentation](https://jax.readthedocs.io/en/latest/installation.html).

## Simulate a contaminated MeerKAT observation

```bash
sim-target
```

### Help function

```bash
sim-target --help
```

### Configuration file based simulations and analysis
`tabascal` now includes the facility to define a simulation using a YaML configuration file. There is a general command line interface to run these simulations allowing one to change certain parameters on the file as well as in the configuration file. All input data is copied into the output simulation directory to allow one to run an identical simulation with ease. Inside [tabascal/examples/yaml_obs](tabascal/examples/yaml_obs) are a set of config files to get you started. There are also example data files which are used for including predefined astronomical and rfi models. They are all `csv` files with file extensions to help distinguish them. To run a simulation of a target field with 100 randomly distributed point sources (just like is shown above) simply run 

```bash
sim-vis -c target_obs.yaml
```

Again, as before, you can run the help function to see what other command line options there are.

```bash
sim-vis -h
```

Downstream analysis such as flagging, RFI subtraction, imaging, and source extraction can be performed through such configuration files as well. This is currently still in development where the `tabascal` RFI subtraction algorithm itself is not yet publically available. However, a full end to end analysis pipeline is available. Individual portions can be accessed through the command line scripts: `flag-data`, `image`, and  `src-extract`, with example configs in [tabascal/analysis/yaml_configs/target](tabascal/analysis/yaml_configs/target). All three of these can be perfomed in a single command line script by using `extract`. See the help documentation of these scripts for further details.  

## Measurement Set output

Measurement sets allow the addition of non-standard data columns. The simulator in tabascal takes advantage of this and adds the following columns to help with debugging and analysis.

### Standard

* `DATA` : Observed data which includes gains and noise.
* `CORRECTED_DATA` : Filled with zeros or the data of ones choice when calling the `write_ms` function.
* `MODEL_DATA` : Filled with zeros as it will be used by `WSCLEAN` when imaging.

### Non-standard

* `CAL_DATA` : Observed data (`DATA`) where the true gain solutions have been applied.
* `AST_MODEL_DATA` : The astronomical visibilities only with perfect gains and no noise. 
* `RFI_MODEL_DATA` : The RFI visibilities only with perfect gains and no noise.
* `AST_DATA` : The same as `AST_MODEL_DATA` but with the noise added. 
* `RFI_DATA` : The same as `RFI_MODEL_DATA` but with the noise added. 
* `NOISE_DATA` : The complex noise that is added to the above datasets. 

## Documentation

[https://tabascal.readthedocs.io/en/latest/](https://tabascal.readthedocs.io/en/latest/)

## Citing tabascal

```
@ARTICLE{Finlay2023,
       author = {{Finlay}, Chris and {Bassett}, Bruce A. and {Kunz}, Martin and {Oozeer}, Nadeem},
        title = "{Trajectory-based RFI subtraction and calibration for radio interferometry}",
      journal = {\mnras},
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
