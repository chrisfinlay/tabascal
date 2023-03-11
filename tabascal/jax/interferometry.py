import sys

import jax.numpy as jnp
from jax import jit, random
from jax.config import config
from jax.lax import scan
from scipy.special import jv

config.update("jax_enable_x64", True)

c = 2.99792458e8


@jit
def rfi_vis(app_amplitude, c_distances, freqs, a1, a2):
    """
    Calculate visibilities from distances to rfi sources.

    Parameters
    ----------
    app_amplitude: array_like (n_src, n_time, n_ant, n_freq)
        Apparent amplitude at the antennas.
    c_distances: array_like (n_src, n_time, n_ant)
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
    n_src = app_amplitude.shape[0]
    vis = _rfi_vis(app_amplitude[0, None], c_distances[0, None], freqs, a1, a2)

    # This is a scan over the sources, but we can't use scan it unless we jit decorate this function
    def _add_vis(vis, i):
        return (
            vis + _rfi_vis(app_amplitude[i, None], c_distances[i, None], freqs, a1, a2),
            i,
        )

    return scan(_add_vis, vis, jnp.arange(1, n_src))[0]
    # return _rfi_vis(app_amplitude, c_distances, freqs, a1, a2)


@jit
def astro_vis(sources, uvw, lmn, freqs):
    """
    Calculate visibilities from a set of point sources using DFT.

    Parameters
    ----------
    sources: array_like (n_src, n_freq)
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
    n_src = sources.shape[0]
    vis = _astro_vis(sources[0, None], uvw, lmn[0, None], freqs)

    # This is a scan over the sources, but we can't use scan it unless we jit decorate this function
    def _add_vis(vis, i):
        return vis + _astro_vis(sources[i, None], uvw, lmn[i, None], freqs), i

    return scan(_add_vis, vis, jnp.arange(1, n_src))[0]


def ants_to_bl(G, a1, a2):
    """
    Calculate the complex gains for each baseline given the per antenna gains.

    Parameters
    ----------
    G: array_like (n_time, n_ant, n_freq)
        Complex gains at each antenna over time.
    a1: array_like (n_bl,)
        Antenna 1 indexes, between 0 and n_ant-1.
    a2: array_like (n_bl,)
        Antenna 2 indexes, between 0 and n_ant-1.

    Returns
    -------
    G_bl: array_like (n_time, n_bl, n_freq)
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
        amps (jnp.ndarray): Amplitudes at the antennas. (n_src, n_time, n_ant, n_freq)
        a1 (jnp.ndarray): Antenna 1 indexes, between 0 and n_ant-1. (n_bl,)
        a2 (jnp.ndarray): Antenna 2 indexes, between 0 and n_ant-1. (n_bl,)

    Returns:
        jnp.ndarray: Intensity on baselines.
    """
    return amps[:, :, a1] * jnp.conjugate(amps[:, :, a2])


@jit
def phase_from_distances(distances, a1, a2, freqs):
    """Calculate phase differences between antennas from distances.

    Args:
        distances (jnp.ndarray): Distances to antennas. (n_src, n_time, n_ant)
        a1 (jnp.ndarray): Antenna 1 indexes, between 0 and n_ant-1. (n_bl,)
        a2 (jnp.ndarray): Antenna 2 indexes, between 0 and n_ant-1. (n_bl,)
        freqs (jnp.ndarray): Frequencies in Hz. (n_freq,)

    Returns:
        jnp.ndarray: Phases on baselines.
    """
    # Create array of shape (n_src, n_time, n_bl, n_freq)
    freqs = freqs[None, None, None, :]
    distances = distances[:, :, :, None]

    phases = minus_two_pi_over_lamda(freqs) * (
        distances[:, :, a1, :] - distances[:, :, a2, :]
    )

    return phases


@jit
def _rfi_vis(app_amplitude, c_distances, freqs, a1, a2):
    # Create array of shape (n_src, n_time, n_bl, n_freq), then sum over n_src

    app_amplitude = jnp.asarray(app_amplitude)
    c_distances = jnp.asarray(c_distances)
    freqs = jnp.asarray(freqs)
    a1 = jnp.asarray(a1)
    a2 = jnp.asarray(a2)

    phase = phase_from_distances(c_distances, a1, a2, freqs)
    intensity = amp_to_intensity(app_amplitude, a1, a2)

    vis = jnp.sum(intensity * jnp.exp(1.0j * phase), axis=0)

    return vis


@jit
def _astro_vis(sources, uvw, lmn, freqs):
    #     Create array of shape (n_src, n_time, n_bl, n_freq), then sum over n_src

    sources = jnp.asarray(sources[:, None, None, :])  #     (n_src, 1, n_freq)
    freqs = jnp.asarray(freqs[None, None, None, :])  #      (1, 1, 1, n_freq)
    uvw = jnp.asarray(uvw[None, :, :, None, :])  #          (1, n_time, n_bl, 1, 3)
    lmn = jnp.asarray(lmn[:, None, None, None, :])  #       (n_src, 1, 1, 1, 3)
    s0 = jnp.array([0, 0, 1])[None, None, None, None, :]  # (1, 1, 1, 1, 3)

    phase = minus_two_pi_over_lamda(freqs) * jnp.sum(uvw * (lmn - s0), axis=-1)

    vis = jnp.sum(sources * jnp.exp(1.0j * phase), axis=0)

    return vis


@jit
def _ants_to_bl(G, a1, a2):
    G_bl = G[:, a1, :] * jnp.conjugate(G[:, a2, :])

    return G_bl


def airy_beam(theta: jnp.ndarray, freqs: jnp.ndarray, dish_d: float):
    """
    Calculate the primary beam voltage at a given angular distance from the
    pointing direction. The beam intensity model is the Airy disk as
    defined by the dish diameter. This is the same a the CASA default.

    Parameters
    ----------
    theta: (n_src, n_time, n_ant)
        The angular separation between the pointing direction and the
        source.
    freqs: (n_freq,)
        The frequencies at which to calculate the beam in Hz.
    dish_d: float
        The diameter of the dish in meters.

    Returns
    -------
    E: ndarray (n_src, n_time, n_ant, n_freq)
        The beam voltage at each frequency.
    """
    theta = jnp.asarray(theta[:, :, :, None])
    freqs = jnp.asarray(freqs)
    dish_d = jnp.asarray(dish_d).flatten()[0]
    mask = jnp.where(theta > 90.0, 0, 1)
    theta = jnp.deg2rad(theta)
    x = jnp.where(
        theta == 0.0,
        sys.float_info.epsilon,
        jnp.pi * freqs[None, None, None, :] * dish_d * jnp.sin(theta) / c,
    )

    return (2 * jv(1, x) / x) * mask


def Pv_to_Sv(Pv: jnp.ndarray, d: jnp.ndarray) -> jnp.ndarray:
    """
    Convert emission power to received intensity in Jy. Assumes constant
    power across the bandwidth. Calculated from

    .. math::
        Jy = \frac{P G_a}{FSPL \Delta\nu A_e}
            = \frac{P G_a \lambda^2 4\pi}{(4\pi d)^2 \Delta\nu \lambda^2}
            = \frac{P G_a}{4\pi d^2 \Delta \nu}

    Parameters
    ----------
    Pv: ndarray (n_src, n_freq)
        Specific emission power in W/Hz.
    d: ndarray (n_src, n_time, n_ant)
        Distances from source to receiving antennas in m.

    Returns
    -------
    Sv: ndarray (n_src, n_time, n_ant, n_freq)
        Spectral flux density at the receiving antennas in Jy.
    """
    Pv = jnp.asarray(Pv)
    d = jnp.asarray(d)
    return Pv[:, None, None, :] / (4 * jnp.pi * d[:, :, :, None] ** 2) * 1e26


def add_noise(vis: jnp.ndarray, noise_std: jnp.ndarray, key: jnp.ndarray):
    """
    Add complex gaussian noise to the integrated visibilities. The real and
    imaginary components will each get this level of noise.

    Parameters
    ----------
    vis: ndarray (n_time, n_bl, n_freq)
        The visibilities to add noise to.
    noise_std: (n_freq, )
        Standard deviation of the complex noise.
    key: jax.random.PRNGKey
        Random number generator key.
    """
    vis = jnp.asarray(vis)
    noise_std = jnp.asarray(noise_std)
    key = jnp.asarray(key)
    noise = (
        random.normal(key, shape=vis.shape, dtype=jnp.complex128)
        * noise_std[None, None, :]
    )
    return vis + noise, noise


def SEFD_to_noise_std(
    SEFD: jnp.ndarray, chan_width: jnp.ndarray, int_time: jnp.ndarray
):
    """Calculate the standard deviation of the complex noise in a visibility
    given the system equivalent flux density, the channel width and integration time.

    Parameters
    ----------
    SEFD: ndarray (n_freq, )
        System equivalent flux density in Jy.
    chan_width: ndarray (n_time, n_ant, n_freq)
        Channel width in Hz.
    int_time: float
        Integration time in seconds.

    Returns
    -------
    noise_std: ndarray (n_time, n_ant, n_freq)
        Standard deviation of the complex noise in a visibility.
    """
    SEFD = jnp.asarray(SEFD)
    chan_width = jnp.asarray(chan_width)
    int_time = jnp.asarray(int_time)
    return SEFD / jnp.sqrt(2 * chan_width * int_time)


def int_sample_times(times: jnp.ndarray, n_int_samples: int = 1):
    """Calculate the times at which to sample the visibilities for

    Parameters
    ----------
    times: ndarray (n_time, )
        The time centroids at which to sample the visibilities.
    n_int_samples: int
        The number of samples to take per integration time.

    Returns
    -------
    times_fine: ndarray (n_time * n_int_samples, )
        The times at which to sample the visibilities.
    """
    times = jnp.asarray(times)
    n_int_samples = jnp.asarray(n_int_samples)
    int_time = times[1] - times[0]
    times_fine = int_time / (2 * n_int_samples) + jnp.arange(
        times[0] - int_time / 2,
        times[-1] + int_time / 2,
        int_time / n_int_samples,
    )
    return times_fine


def generate_gains(
    G0_mean: complex,
    G0_std: float,
    Gt_std_amp: float,
    Gt_std_phase: float,
    times: jnp.ndarray,
    n_ant: int,
    n_freq: int,
    key: jnp.ndarray,
):
    """
    Generate complex antenna gains. Gain amplitudes and phases
    are modelled as linear time-variates. Gains for all antennas at t = 0
    are randomly sampled from a Gaussian described by the G0 parameters.
    The rate of change of both ampltudes and phases are sampled from a zero
    mean Gaussian with standard deviation as provided.

    Parameters
    ----------
    G0_mean: complex
        Mean of Gaussian at t = 0.
    G0_std: float
        Standard deviation of Gaussian at t = 0.
    Gt_std_amp: float
        Standard deviation of Gaussian describing the rate of change in the
        gain amplitudes in 1/seconds.
    Gt_std_phase: float
        Standard deviation of Gaussian describing the rate of change in the
        gain phases in rad/seconds.
    key: jax.random.PRNGKey
        Random number generator key.
    """
    G0_mean = jnp.asarray(G0_mean)
    G0_std = jnp.asarray(G0_std)
    Gt_std_amp = jnp.asarray(Gt_std_amp)
    Gt_std_phase = jnp.asarray(Gt_std_phase)
    times = jnp.asarray(times)
    n_ant = jnp.asarray(n_ant)
    n_freq = jnp.asarray(n_freq)
    key = jnp.asarray(key)
    G0 = G0_mean * jnp.exp(
        1.0j * jnp.pi * (random.uniform(key, (1, n_ant, n_freq)) - 0.5)
    )
    key, subkey = random.split(key)
    gains_noise = G0_std * random.normal(key, (n_ant, n_freq), dtype=jnp.complex128)
    key, subkey = random.split(key)

    gains_amp = Gt_std_amp * random.normal(key, (1, n_ant, 1)) * (times)[:, None, None]
    key, subkey = random.split(key)
    gains_phase = (
        Gt_std_phase * random.normal(key, (1, n_ant, 1)) * (times)[:, None, None]
    )
    key, subkey = random.split(key)
    gains_ants = G0 + gains_noise + gains_amp * jnp.exp(1.0j * gains_phase)
    gains_ants = gains_ants.at[:, -1, :].set(jnp.abs(gains_ants[:, -1, :]))
    return gains_ants


def apply_gains(
    vis_ast: jnp.ndarray,
    vis_rfi: jnp.ndarray,
    gains: jnp.ndarray,
    a1: jnp.ndarray,
    a2: jnp.ndarray,
):
    """Apply antenna gains to visibilities.

    Parameters
    ----------
    vis_ast: ndarray (n_time, n_bl, n_freq)
        The astronomical visibilities.
    vis_rfi: ndarray (n_time, n_bl, n_freq)
        The RFI visibilities.
    gains: ndarray (n_time, n_ant, n_freq)
        The antenna gains.
    a1: ndarray (n_bl,)
        The first antenna index for each baseline.
    a2: ndarray (n_bl,)
        The second antenna index for each baseline.

    Returns
    -------
    vis_obs: ndarray (n_time, n_bl, n_freq)
        The visibilities with gains applied.
    """
    vis_ast = jnp.asarray(vis_ast)
    vis_rfi = jnp.asarray(vis_rfi)
    gains = jnp.asarray(gains)
    a1 = jnp.asarray(a1)
    a2 = jnp.asarray(a2)
    vis_obs = gains[:, a1] * (vis_ast + vis_rfi) * jnp.conj(gains[:, a2])
    return vis_obs


def time_avg(vis: jnp.ndarray, n_int_samples: int = 1):
    """Average visibilities in time.

    Parameters
    ----------
    vis: ndarray (n_time_fine, n_bl, n_freq)
        The visibilities to average in time.
    n_int_samples: int
        The number of samples to take per integration time.

    Returns
    -------
    vis_avg: ndarray (n_time, n_bl, n_freq)
        The averaged visibilities.
    """
    vis = jnp.asarray(vis)
    n_int_samples = jnp.asarray(n_int_samples)
    vis_avg = jnp.mean(
        jnp.reshape(vis, (-1, n_int_samples, vis.shape[1], vis.shape[2])),
        axis=1,
    )
    return vis_avg


def db_to_lin(dB: float):
    """
    Convert deciBels to linear units.

    Parameters
    ----------
    dB: float, ndarray
        deciBel value to convert.

    Returns
    -------
    lin: float, ndarray
    """
    dB = jnp.asarray(dB)
    return 10.0 ** (dB / 10.0)
