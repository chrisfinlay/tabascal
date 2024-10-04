import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike

from tabascal.dask.observation import Observation
from tabascal.jax.coordinates import alt_az_of_source, gmst_to_lst
import os

plt.rcParams["font.size"] = 18

def time_units(times: ArrayLike) -> tuple:
    """Scale the time axis to hours, minutes or seconds depending on the total range.

    Parameters
    ----------
    times : ArrayLike
        Times to consider.

    Returns
    -------
    tuple
        Rescaled times array and the scale unit as a string.
    """

    time_range = times[-1] - times[0]
    if time_range>3600:
        scale = "hr"
        times = times / 3600
    elif time_range>60:
        scale = "min"
        times = times / 60
    else:
        scale = "s"

    return times, scale

def plot_angular_seps(obs: Observation, save_path: str) -> None:
    """Plot the angular separations between the RFI sources and pointing direction.

    Parameters
    ----------
    obs : Observation
        Observation object with RFI sources added
    save_path : str
        Path to where to save the plots.
    """
    
    times, scale = time_units(obs.times_fine)
    plt.figure(figsize=(10,7))
    if obs.n_rfi_satellite>0:
        ang_seps = np.concatenate(obs.rfi_satellite_ang_sep, axis=0).mean(axis=-1).T
        plt.plot(times, ang_seps, label="Satellite")
    if obs.n_rfi_stationary>0:
        ang_seps = np.concatenate(obs.rfi_stationary_ang_sep, axis=0).mean(axis=-1).T
        plt.plot(times, ang_seps, label="Stationary")
    plt.xlabel(f"Time [{scale}]")
    plt.ylabel("Angular Separation [deg]")
    plt.legend()
    plt.savefig(os.path.join(save_path, "AngularSeps.png"), format="png", dpi=200)


def plot_src_alt(obs: Observation, save_path: str) -> None:
    """Plot the target source altitude over the period of the observation.

    Parameters
    ----------
    obs : Observation
        Observation object.
    save_path : str
        Path to where to save the plot.
    """

    times, scale = time_units(obs.times)
    lst = gmst_to_lst(obs.times.compute(), obs.longitude.compute())
    alt = alt_az_of_source(lst, *[x.compute() for x in [obs.latitude, obs.ra, obs.dec]])[:,0]
    plt.figure(figsize=(10,7))
    plt.plot(times, alt, '.-')
    plt.xlabel(f"Time [{scale}]")
    plt.ylabel("Source Altitude [deg]")
    plt.savefig(os.path.join(save_path, "SourceAltitude.png"), format="png", dpi=200)


def plot_uv(obs: Observation, save_path: str) -> None:
    """Plot the uv coverage of the telescope for the given observation.

    Parameters
    ----------
    obs : Observation
        Observation object.
    save_path : str
        Path to where to save the plot.
    """   

    plt.figure(figsize=(10,10))
    if obs.n_time_fine > 100:
        time_step = int(obs.n_time_fine/100)
    else:
        time_step = 1
    u = obs.bl_uvw[::time_step,:,0].compute().flatten()
    v = obs.bl_uvw[::time_step,:,1].compute().flatten()
    max_U = np.max(np.sqrt(u**2 + v**2))
    exp = float(np.floor(np.log10(max_U)))
    mantissa = np.ceil(10**(np.log10(max_U)-exp))
    lim = mantissa * 10**exp
    plt.plot(u, v, 'k.', ms=1, alpha=0.3)
    plt.plot(-u, -v, 'k.', ms=1, alpha=0.3)
    plt.grid()
    plt.xlim(-lim, lim)
    plt.ylim(-lim, lim)
    plt.xlabel("U [m]")
    plt.ylabel("V [m]")
    plt.savefig(os.path.join(save_path, "UV.png"), format="png", dpi=200)