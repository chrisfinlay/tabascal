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
    entry_points="""
        [console_scripts]
        sim-vis=tabascal.examples.target_observation:cli
    """,
    install_requires=["jax", "dask", "xarray", "dask-ms"],
    zip_safe=False,
)
