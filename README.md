# tabascal

[https://arxiv.org/abs/2301.04188](https://arxiv.org/abs/2301.04188)

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

```bash
conda env create -f tabascal/env.yaml
conda develop ./tabascal/
```

### GPU 
 
To enable GPU compute you need the GPU version of `jaxlib` installed. The easiest way is using conda, otherwise, refer to the JAX installation [documentation](https://github.com/google/jax#installation).

```bash
conda install jaxlib=*=*cuda* jax cuda-nvcc -c conda-forge -c nvidia
```

## Simulate a contaminated MeerKAT observation

```bash
python tabascal/examples/sim.py
```

### Help function

```bash
python tabascal/examples/sim.py --help
```

## Documentation

[https://tabascal.readthedocs.io/en/latest/](https://tabascal.readthedocs.io/en/latest/)

## Citing tabascal

```
@article{finlay2023trajectory,
  title={Trajectory Based RFI Subtraction and Calibration for Radio Interferometry},
  author={Finlay, Chris and Bassett, Bruce A and Kunz, Martin and Oozeer, Nadeem},
  journal={arXiv preprint arXiv:2301.04188},
  year={2023}
}
```
