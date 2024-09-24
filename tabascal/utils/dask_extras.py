import jax.numpy as jnp

def get_factors(n: int):
    """Get the integer factors of n.

    Parameters:
    -----------
    n: int
        Number to get the factors of.

    Returns:
    --------
    factors: array_like
        The factors of n.
    """
    factors = [1, n]
    root_n = jnp.sqrt(n)
    if root_n == int(root_n):
        factors.append(root_n)
    for i in range(1, int(root_n)):
        if n % i == 0:
            factors += [i, n // i]
    return jnp.unique(jnp.array(factors).sort().astype(int))


def get_chunksizes(n_t: int, n_f: int, n_int: int, n_bl: int, MB_max: float):
    """Get the chunk sizes for a given number of time and frequency samples.

    Parameters:
    -----------
    n_t: int
        Number of time samples.
    n_f: int
        Number of frequency samples.
    n_int: int
        Number of integration samples per time sample.
    n_bl: int
        Number of baselines.
    MB_max: float
        Maximum megabytes to use for a chunk.

    Returns:
    --------
    chunksize: dict
        Dictionary containing the time and frequency chunk sizes.
    """
    time_factors = get_factors(n_t)
    freq_factors = get_factors(n_f)
    extra = MB_max * 1e6 / (16 * n_int * n_bl)
    tt, ff = jnp.meshgrid(time_factors, freq_factors)
    idx = jnp.argmin(jnp.abs(tt * ff - extra))
    time_chunksize = int(tt.flatten()[idx])
    freq_chunksize = int(ff.flatten()[idx])
    chunk_bytes = 16 * n_int * n_bl * time_chunksize * freq_chunksize
    chunksize = {
        "time": time_chunksize,
        "freq": freq_chunksize,
        "chunk_bytes": f"{chunk_bytes/1e6:.0f} MB",
    }
    return chunksize