#!/usr/bin/env python

from daskms import xds_from_ms, xds_to_table

import xarray as xr
import dask.array as da
import dask

import argparse

def write_flags(ms_path: str, n_sigma: float = 3.0):

    xds = xds_from_ms(ms_path)[0]

    if n_sigma > 0:
        flags = da.abs(xds.CAL_DATA - xds.AST_MODEL_DATA) > (n_sigma * xds.SIGMA * da.sqrt(2))
    else:
        flags = xr.zeros_like(xds.DATA).astype(bool)

    xds = xds.assign(FLAG=flags)
    dask.compute(xds_to_table([xds,], ms_path, ["FLAG"]))   

    return flags

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Flag DATA lying a certain threshold away from the MODEL_DATA "
    )
    parser.add_argument(
        "--ms_path", required=True, help="File path to the Measurement Set."
    )
    parser.add_argument(
        "--n_sigma", default=3.0, type=float, help="Threshold in number of std of noise given by SIGMA column. 0 unflags everything."
    )

    args = parser.parse_args()
    flags = write_flags(ms_path=args.ms_path, n_sigma=args.n_sigma)
   
    print()
    print(f"Flag Threshold : {args.n_sigma: .1f} sigma")
    print(f"Flag Rate      : {100*flags.data.mean().compute(): .1f} %")
