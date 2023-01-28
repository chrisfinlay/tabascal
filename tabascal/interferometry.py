from jax import jit
import jax.numpy as jnp
from jax.config import config
config.update("jax_enable_x64", True)

@jit
def rfi_vis(app_amplitude, c_distances, freqs):
    """
    Calculate visibilities from distances to rfi sources.

    Parameters:
    -----------
    app_amplitude: array_like (n_time, n_ant, n_freq, n_src)
        Apparent amplitude at the antennas.
    c_distances: array_like (n_time, n_ant, n_src)
        The phase corrected distances between the rfi sources and the antennas.
    freqs: array_like (n_freq,)
        Frequencies.

    Returns:
    --------
    vis: array_like (n_time, n_bl, n_freq)
        The visibilities.
    """
    n_time, n_ant, n_freq, n_src = app_amplitude.shape
    c = 2.99792458e8

    # Create array of shape (n_time, n_bl, n_freq, n_src), then sum over n_src

    minus_two_pi_over_lamda = (-2.0*jnp.pi*freqs/c).reshape(1,1,n_freq,1)

    c_distances = c_distances.reshape(n_time,n_ant,1,n_src)

    a1, a2 = jnp.triu_indices(n_ant, 1)

    phase = minus_two_pi_over_lamda*(c_distances[:,a1]-c_distances[:,a2])
    intensities_app = app_amplitude[:,a1]*app_amplitude[:,a2]

    vis = jnp.sum(intensities_app*jnp.exp(-1.j*phase), axis=-1)

    return vis

@jit
def astro_vis(sources, uvw, lmn, freqs):
    """
    Calculate visibilities from a set of point sources using DFT.

    Parameters:
    -----------
    sources: array_like (n_freq, n_src)
        Array of point source intensities in Jy.
    uvw: array_like (ntime, n_bl, 3)
        (u,v,w) coordinates of each baseline.
    lmn: array_like (n_src, 3)
        (l,m,n) coordinate of each source.
    freqs: array_like (n_freq,)
        Frequencies in Hz.

    Returns:
    --------
    vis: array_like (n_time, n_bl, n_freq)
        Visibilities of the given set of sources and baselines.
    """
    c = 2.99792458e8

#     Create array of shape (n_time, n_bl, n_freq, n_src), then sum over n_src

    sources = sources[None,None,:,:]
    uvw = uvw[:,:,None,None,:]
    lmn = lmn[None,None,None,:,:]
    freqs = freqs[None,None,:,None]

    minus_two_pi_over_lamda = -2.*jnp.pi * freqs/c

    lmn = lmn - jnp.array([0,0,1])[None,:]

    phase = minus_two_pi_over_lamda*jnp.sum(uvw*lmn, axis=-1)

    vis = jnp.sum(sources*jnp.exp(-1.j*phase), axis=-1)

    return vis

@jit
def ants_to_bl(G):
    """
    Calculate the complex gains for each baseline given the per antenna gains.

    Parameters:
    -----------
    G: array_like (n_time, n_ant)
        Complex gains at each antenna over time.

    Returns:
    --------
    G_bl: array_like (n_time, n_bl)
        Complex gains on each baseline over time.
    """

    n_ant = G.shape[-1]

    a1, a2 = jnp.triu_indices(n_ant, 1)

    G_bl = G[:,a1]*G[:,a2].conjugate()

    return G_bl
