from jax import jit
import jax.numpy as jnp
from jax.config import config

config.update("jax_enable_x64", True)

c = 2.99792458e8


def rfi_vis(app_amplitude, c_distances, freqs, a1, a2):
    """
    Calculate visibilities from distances to rfi sources.

    Parameters
    ----------
    app_amplitude: array_like (n_time, n_ant, n_freq, n_src)
        Apparent amplitude at the antennas.
    c_distances: array_like (n_time, n_ant, n_src)
        The phase corrected distances between the rfi sources and the antennas in metres.
    freqs: array_like (n_freq,)
        Frequencies in Hz.
    a1: array_like (n_bl,)
        Antenna 1 indexes, between 0 and n_ant-1.
    a2: array_like (n_bl,)
        Antenna 2 indexes, between 0 and n_ant-1.

    Returns
    -------
    vis: array_like (n_time, n_bl, n_freq)
        The visibilities.
    """

    return _rfi_vis(app_amplitude, c_distances, freqs, a1, a2)


def astro_vis(sources, uvw, lmn, freqs):
    """
    Calculate visibilities from a set of point sources using DFT.

    Parameters
    ----------
    sources: array_like (n_freq, n_src)
        Array of point source intensities in Jy.
    uvw: array_like (ntime, n_bl, 3)
        (u,v,w) coordinates of each baseline.
    lmn: array_like (n_src, 3)
        (l,m,n) coordinate of each source.
    freqs: array_like (n_freq,)
        Frequencies in Hz.

    Returns
    -------
    vis: array_like (n_time, n_bl, n_freq)
        Visibilities of the given set of sources and baselines.
    """

    return _astro_vis(sources, uvw, lmn, freqs)


def ants_to_bl(G, a1, a2):
    """
    Calculate the complex gains for each baseline given the per antenna gains.

    Parameters
    ----------
    G: array_like (n_time, n_ant)
        Complex gains at each antenna over time.

    Returns
    -------
    G_bl: array_like (n_time, n_bl)
        Complex gains on each baseline over time.
    """

    return _ants_to_bl(G, a1, a2)


@jit
def minus_two_pi_over_lamda(freqs):
    """Calculate -2pi/lambda for each frequency.

    Args:
        freqs (jnp.ndarray): Frequencies in Hz. (n_freq,)

    Returns:
        jnp.ndarray: -2pi/lambda for each frequency. (n_freq,)
    """
    return -2.0 * jnp.pi * freqs / c


@jit
def amp_to_intensity(amps, a1, a2):
    """Calculate intensity on a baseline ffrom the amplitudes at each antenna.

    Args:
        amps (jnp.ndarray): Amplitudes at the antennas. (n_time, n_ant, n_freq, n_src)
        a1 (jnp.ndarray): Antenna 1 indexes, between 0 and n_ant-1. (n_bl,)
        a2 (jnp.ndarray): Antenna 2 indexes, between 0 and n_ant-1. (n_bl,)

    Returns:
        jnp.ndarray: Intensity on baselines.
    """
    return amps[:, a1] * jnp.conjugate(amps[:, a2])


@jit
def phase_from_distances(distances, a1, a2, freqs):
    """Calculate phase differences between antennas from distances.

    Args:
        distances (jnp.ndarray): Distances to antennas. (n_time, n_ant, n_src)
        a1 (jnp.ndarray): Antenna 1 indexes, between 0 and n_ant-1. (n_bl,)
        a2 (jnp.ndarray): Antenna 2 indexes, between 0 and n_ant-1. (n_bl,)
        freqs (jnp.ndarray): Frequencies in Hz. (n_freq,)

    Returns:
        jnp.ndarray: Phases on baselines.
    """
    freqs = freqs[None, None, :, None]
    distances = distances[:, :, None, :]

    phases = minus_two_pi_over_lamda(freqs) * (
        distances[:, a1, :] - distances[:, a2, :]
    )

    return phases


@jit
def _rfi_vis(app_amplitude, c_distances, freqs, a1, a2):
    # Create array of shape (n_time, n_bl, n_freq, n_src), then sum over n_src

    app_amplitude = jnp.asarray(app_amplitude)
    c_distances = jnp.asarray(c_distances)
    freqs = jnp.asarray(freqs)
    a1 = jnp.asarray(a1)
    a2 = jnp.asarray(a2)

    phase = phase_from_distances(c_distances, a1, a2, freqs)
    intensity = amp_to_intensity(app_amplitude, a1, a2)

    vis = jnp.sum(intensity * jnp.exp(1.0j * phase), axis=-1)

    return vis


@jit
def _astro_vis(sources, uvw, lmn, freqs):
    #     Create array of shape (n_time, n_bl, n_freq, n_src), then sum over n_src

    sources = jnp.asarray(
        sources[None, None, :, :]
    )  #     (1,      1,    n_freq, n_src)
    freqs = jnp.asarray(freqs[None, None, :, None])  #      (1,      1,    n_freq, 1)
    uvw = jnp.asarray(
        uvw[:, :, None, None, :]
    )  #          (n_time, n_bl, 1,      1,     3)
    lmn = jnp.asarray(
        lmn[None, None, None, :, :]
    )  #       (1,      1,    1,      n_src, 3)
    s0 = jnp.array([0, 0, 1])[
        None, None, None, None, :
    ]  # (1,      1,    1,      1,     3)

    phase = minus_two_pi_over_lamda(freqs) * jnp.sum(uvw * (lmn - s0), axis=-1)

    vis = jnp.sum(sources * jnp.exp(1.0j * phase), axis=-1)

    return vis


@jit
def _ants_to_bl(G, a1, a2):
    G_bl = G[:, a1] * jnp.conjugate(G[:, a2])

    return G_bl
