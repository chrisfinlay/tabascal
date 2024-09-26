#!/usr/bin/env python

from daskms import xds_from_ms, xds_to_table

import numpy as np

import xarray as xr
import dask.array as da
import dask

import argparse

def write_results(ms_path: str, zarr_path: str):

    xds_ms = xds_from_ms(ms_path)[0]
    xds_tab = xr.open_zarr(zarr_path)

    dims = ["row", "chan", "corr"]
    chunks = {k: v for k, v in xds_ms.chunks.items() if k in dims}

    vis_ast = xds_tab.ast_vis.data.astype(np.complex64).mean(axis=0).T.flatten()

    vis_ast = xr.DataArray(da.expand_dims(vis_ast, axis=(1,2)), dims=dims).chunk(chunks)

    xds_ms = xds_ms.assign(TAB_DATA=vis_ast)

    print("Writing tabascal results to 'TAB_DATA' column in MS file.")

    dask.compute(xds_to_table([xds_ms,], ms_path, ["TAB_DATA"], column_keywords={"TAB_DATA": {"UNIT": "Jy"}}))

def main():

    parser = argparse.ArgumentParser(
        description="Copy recovered data from a tabascal run and save in a MS file under the column named 'TAB_DATA'."
    )
    parser.add_argument(
        "--ms_path", required=True, help="File path to the Measurement Set."
    )
    parser.add_argument(
        "--zarr_path", required=True, help="File path to the zarr file containing results."
    )

    args = parser.parse_args()
    write_results(ms_path=args.ms_path, zarr_path=args.zarr_path)
