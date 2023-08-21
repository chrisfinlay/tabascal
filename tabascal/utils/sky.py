import numpy as np
import jax.numpy as jnp
from numpy.random import Generator


def uniform_points_disk(
    radius: float, n_src: int, random_seed: int | Generator | None = None
):
    """
    Generate uniformly distributed random points on a disk.

    Parameters:
    -----------
    radius: float
        Radius of the disk.
    n_src: int
        The number of sources/points to generate.
    seed: int
        Random number generator seed/key.

    Returns:
    --------
    points: array_like (2, n_src)
        The coordinate positions of the random points centred on (0,0).
    """
    rng = np.random.default_rng(random_seed)
    r = jnp.sqrt(rng.uniform(low=0.0, high=radius, size=(n_src,)))
    theta = rng.uniform(low=0.0, high=2.0 * jnp.pi, size=(n_src,))

    return r * jnp.array([jnp.cos(theta), jnp.sin(theta)])


def random_power_law(
    n_src: int,
    I_min: float = 1e-4,
    I_max: float = 1.0,
    alpha: float = 1.6,
    random_seed: int | Generator | None = None,
):
    """Generate a random power law distribution of source fluxes with minimum source
    flux defined by `I0`.

    Parameters:
    -----------
    n_src: int
        Number of source fluxes to draw.
    I0: float
        Minimum source flux.
    alpha: float
        Power law index.

    Returns:
    --------
    I: array_like (n_src,)
        Array of source fluxes."""

    def inv_cdf(x):
        return I_min * (1.0 - x) ** (1.0 / (1.0 - alpha))

    rng = np.random.default_rng(random_seed)
    rand_unif = rng.uniform(size=(n_src,))
    I = np.array(inv_cdf(rand_unif))
    while np.any(I > I_max):
        idx = np.where(I > I_max)[0]
        rand_unif = rng.uniform(size=(idx.shape[0],))
        I[idx] = inv_cdf(rand_unif)

    return I


def generate_random_sky(
    n_src: int,
    freqs: jnp.ndarray,
    min_I: float = 1e-4,
    max_I: float = 1.0,
    I_power_law: float = 1.6,
    spec_idx_mean: float = 0.7,
    spec_idx_std: float = 0.2,
    fov: float = 1.0,
    beam_width: float = 0.0,
    random_seed: int | Generator | None = None,
):
    """
    Generate uniformly distributed point sources inside the field of view with
    an exponential intensity distribution. Setting the beam width will make
    sure souces are separated by 5 beam widths apart.

    Parameters:
    -----------
    n_src: int
        Number of sources to generate.
    mean_I: float
        Mean intensity of the sources.
    freqs: array_like (n_freq,)
        Frequencies to generate the sources at.
    spec_idx_mean: float
        Mean spectral index of the sources.
    spec_idx_std: float
        Standard deviation of the spectral index of the sources.
    fov: float
        Field of view to generate positions within. Same units as beam_width.
    beam_width: float
        Width of the resolving beam to ensure sources do not overlap. Sources will be
        separated by >5*`beam_width`. Same units as fov.
    random_seed: int
        Random number generator seed/key.

    Returns:
    --------
    I: array_like (n_src, n_freq)
        The sources intensities.
    delta_ra: array_like
        The sources right ascensions relative to (0,0).
    delta_dec: array_like
        The sources declinations relative to (0,0).
    """
    rng = np.random.default_rng(random_seed)

    I = random_power_law(n_src, min_I, max_I, I_power_law, rng)
    positions = uniform_points_disk(fov / 2.0, 1, rng)
    while positions.shape[1] < n_src:
        n_sample = 2 * (n_src - positions.shape[1])
        new_positions = uniform_points_disk(fov / 2.0, n_sample, rng)
        positions = np.concatenate([positions, new_positions], axis=1)
        s1, s2 = np.triu_indices(positions.shape[1], 1)
        d = np.linalg.norm(positions[:, s1] - positions[:, s2], axis=0)
        idx = np.where(d < 5 * beam_width)[0]
        remove_source_idx = np.unique(jnp.concatenate([s1[idx], s2[idx]]))
        positions = np.delete(positions, remove_source_idx, axis=1)

    d_ra, d_dec = positions[:, :n_src]

    spectral_indices = rng.normal(loc=spec_idx_mean, scale=spec_idx_std, size=(n_src,))
    I = I[:, None] * ((freqs[None, :] / freqs[0]) ** -spectral_indices[:, None])

    return I, d_ra, d_dec
