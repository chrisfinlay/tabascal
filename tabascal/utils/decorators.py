import dask.array as da
from dask import delayed
import xarray as xr
import numpy as np
from functools import partial
import inspect
from jax import jit

from typing import Callable
from numpy.typing import ArrayLike

# def arr_to_xarr(x: ArrayLike, name: str, dims: list[str]):
#     return xr.DataArray(data=x, dims=dims, name=name)


def jit_with_doc(func: Callable) -> Callable:
    def inner(*args, **kwargs):
        return jit(func)(*args, **kwargs)

    inner.__doc__ = func.__doc__
    inner.__signature__ = inspect.signature(func)

    return inner


def xarrayify(
    func: Callable, dims_in: list[list[str]], dims_out: list[str]
) -> Callable:
    def inner(shape_out: tuple, *args, **kwargs):

        # Create a keyword argument dictionary from combination of unnamed arguments and keyword arguments
        arg_names = inspect.getfullargspec(func).args
        all_kwargs = dict(
            {arg_name: arg for arg_name, arg in zip(arg_names[: len(args)], args)},
            **kwargs
        )

        in_xds = xr.Dataset(
            {
                name: xr.DataArray(data=da.atleast_1d(x), dims=dims, name=name)
                for name, x, dims in zip(
                    all_kwargs.keys(), all_kwargs.values(), dims_in
                )
            }
        )

        out_xds = xr.Dataset({"result": (dims_out, da.zeros(shape_out))})

        def delayed_func(xds):
            delayed_args = [xds[name].data for name in arg_names]
            result = delayed(func, pure=True)(*delayed_args).compute()
            xds_out = xr.Dataset({"result": (dims_out, result)})
            return xds_out

        xds = xr.map_blocks(delayed_func, in_xds, template=out_xds)

        return xds.result.data

    inner.__doc__ = func.__doc__
    sig = inspect.Signature(
        [
            inspect.Parameter(
                "shape_out", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=tuple
            ),
            *inspect.signature(func).parameters.values(),
        ]
    )
    inner.__signature__ = sig

    return inner


# def xarrayify_multi(func, dims_in, dims_out, names_out):
#     def inner(*args, **kwargs):

#         arg_names = inspect.getfullargspec(func).args
#         all_kwargs = dict({arg_name: arg for arg_name, arg in zip(arg_names[:len(args)], args)}, **kwargs)
#         in_xds = xr.Dataset({name: arr_to_xarr(x, name, dims) for name, x, dims in zip(all_kwargs.keys(), all_kwargs.values(), dims_in)})

#         shapes_out = []
#         for dim in dims_out:
#             shapes_out.append([in_xds.sizes[dim[i]] for i in range(len(dim))])
#         out_xds = xr.Dataset({name: (dim, da.zeros(shape)) for name, dim, shape in zip(names_out, dims_out, shapes_out)})

#         def delayed_func(xds):
#             delayed_args = [xds[name].data for name in arg_names]
#             results = delayed(func, pure=True)(*delayed_args).compute()
#             xds_out = xr.Dataset({name: (out_xds[name].dims, result) for name, result in zip(out_xds.data_vars.keys(), results)})
#             return xds_out

#         results = xr.map_blocks(delayed_func, in_xds, template=out_xds)

#         return [results[name].data for name in results.data_vars.keys()]
#     return inner
