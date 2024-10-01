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
            # Not Needed
            "sim-target=tabascal.examples.target_observation:main",
            "sim-calib=tabascal.examples.calibration_observation:main",
            "sim-mixed=tabascal.examples.mixed_source_observation:main",
            "sim-low=tabascal.examples.low_freq_obs:main",
            # Useful extras
            "sat-region=tabascal.utils.sat_region:main",
            "ast-region=tabascal.utils.ast_region:main",
            # Base scripts
            "tab2MS=tabascal.utils.results_to_MS:main",
            "flag-data=tabascal.utils.flag_data:main",
            "image=tabascal.utils.wsclean_image:main",
            "src-extract=tabascal.utils.extract:main",
            # Config enabled
            "sim-vis=tabascal.examples.sim_vis:main",
            "tabascal=tabascal.utils.run_tabascal:main",
            "extract=tabascal.utils.run_extraction:main",
            "end2end=tabascal.utils.end2end:main",
            ]
        },
    install_requires=["jax", "dask", "xarray", "zarr", 
                      "dask-ms", "scipy", "tqdm", "pandas", "matplotlib"],
    extras_require = {
        "gpu": ["jax[cuda12]"],
        "sat": ["astropy", "regions"],
        "astro": ["bdsf"],
        "all": ["jax[cuda12]", "astropy", "regions", "bdsf"],
        "all_cpu": ["astropy", "regions", "bdsf"],
    },
    zip_safe=False,
)
