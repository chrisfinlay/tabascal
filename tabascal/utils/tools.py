import argparse
import os
from pathlib import Path

import jax.numpy as jnp
import numpy as np

pkg_dir = Path(__file__).parent.absolute()


import signal
from contextlib import contextmanager


class TimeoutException(Exception):
    pass


@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")

    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


def beam_size(diameter: float, frequency: float, fwhp: bool = True):
    """
    Calculate the beam size of an antenna or an array. For an array use
    fwhp = True. This assumes an Airy disk primary beam pattern.

    Parameters:
    -----------
    diameter: float
        Diameter of the dish or baseline in metres.
    frequency: float
        Observation frequency in Hertz.
    fwhp: bool
        True if you want the field of view to be the full width at half power
        and False if you want the first null.

    Returns:
    --------
    beam_width: float
        beam_width in degrees.
    """
    diameter = jnp.asarray(diameter)
    frequency = jnp.asarray(frequency)

    c = 299792458.0
    lamda = c / frequency
    beam_width = 1.02 * lamda / diameter if fwhp else 1.22 * lamda / diameter

    return jnp.rad2deg(beam_width)


def load_antennas(telescope: str = "MeerKAT"):
    """
    Load the ENU coordinates for a telescope. Currently only MeerKAT is
    included.

    Parameters:
    -----------
    telescope: str
        The name of the telescope.

    Returns:
    --------
    enu: array_like (n_ant, 3)
        The East, North, Up coordinates of each antenna relative to a reference
        position.
    """
    if telescope == "MeerKAT":
        enu = np.loadtxt(os.path.join(pkg_dir, "../data/Meerkat.enu.txt"))
    else:
        print("Only MeerKAT antennas are currentyl available.")
        enu = None
    return enu


def str2bool(v: str):
    """
    Convert string to boolean.

    Parameters:
    -----------
    v: str
        String to convert to boolean.

    Raises:
    -------
        argparse.ArgumentTypeError: If the string is not a boolean.

    Returns:
    --------
    bool: bool
        The boolean value of the string.
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")
