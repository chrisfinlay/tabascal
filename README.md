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

Alternatively, you can install `tabascal` with pip alone inside enivironment of your choice again with optional GPU support.

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
python tabascal/examples/target_observation.py
```

### Help function

```bash
python tabascal/examples/target_observation.py --help
```

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
