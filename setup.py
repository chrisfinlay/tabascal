from setuptools import setup, find_packages

description = """Trajectory based Radio Frequency Interference (RFI) subtraction
                 and calibration using Bayesian methods for radio
                 interferometeric data."""

setup(
    name="tabascal",
    version="0.0.1",
    description=description,
    url="http://github.com/chrisfinlay/tabascal",
    author="Chris Finlay",
    author_email="christopher.finlay@unige.ch",
    license="MIT",
    packages=find_packages(),
    package_data={"tabascal": ["tabascal/data/*"]},
    entry_points={"console_scripts": [
            "sim-target=tabascal.examples.target_observation:main",
            "sim-calib=tabascal.examples.calibration_observation:main",
            "sim-mixed=tabascal.examples.mixed_source_observation:main",
            "sim-low=tabascal.examples.low_freq_obs:main",
            "flag-data=tabascal.utils.flag_data:main",
            "sat-region=tabascal.utils.sat_region:main",
            "ast-region=tabascal.utils.ast_region:main",
            "tab2MS=tabascal.utils.results_to_MS:main",
            "image=tabascal.utils.wsclean_image:main",
            "extract=tabascal.utils.extract:main",
            ]
        },
    install_requires=["jax", "dask", "xarray", "zarr", "dask-ms", "scipy", "tqdm", "matplotlib"],
    extras_require = {
        "gpu": ["jax[cuda12]"],
        "sat": ["astropy", "regions"],
        "astro": ["bdsf"],
    },
    zip_safe=False,
)
