import jax.numpy as jnp
from jax import random
import jax
import h5py
import numpy as np
import os
from pathlib import Path
pkg_dir = Path(__file__).parent.absolute()


def uniform_points_disk(radius, n_src, key=None):
    """
    Generate uniformly distributed random points on a disk.

    Parameters:
    -----------
    radius: float
        Radius of the disk.
    n_src: int
        The number of sources/points to generate.
    key: jax.random.PRNGKey
        Random number generator seed/key.

    Returns:
    --------
    points: array_like (2, n_src)
        The coordinate positions of the random points centred on (0,0).
    """
    if not isinstance(key, jax.interpreters.xla._DeviceArray):
        key = random.PRNGKey(101)
    r = radius*jnp.sqrt(random.uniform(key, (n_src,)))
    key, subkey = random.split(key)
    theta = 2.*jnp.pi*random.uniform(key, (n_src,))

    return r*jnp.array([jnp.cos(theta), jnp.sin(theta)])

def beam_size(diameter, frequency, fwhp=True):
    """
    Calculate the beam size of an antenna or an array. For an array use fwhp = True. This assumes an Airy disk primary beam pattern.

    Parameters:
    -----------
    diameter: float
        Diameter of the dish or baseline in metres.
    frequency: float
        Observation frequency in Hertz.
    fwhp: bool
        True if you want the field of view to be the full width at half power and False if you want the first null.

    Returns:
    --------
    fov: float
        Field of view in degrees.
    """
    c = 299792458.
    lamda = c/frequency
    fov = 1.02 * lamda / diameter if fwhp else 1.22 * lamda / diameter

    return jnp.rad2deg(fov)

def generate_random_sky(n_src, mean_I, fov, beam_width=0., key=None):
    """
    Generate uniformly distributed point sources inside the field of view with an exponential intensity distribution. Setting the beam width will make sure souces are separated by 5 beam widths apart.

    Parameters:
    -----------
    n_src: int
        Number of sources to generate.
    mean_I: float
        Mean intensity of the sources.
    fov: float
        Field of view to generate positions within. Same units as beam_width.
    beam_width: float
        Width of the resolving beam to ensure sources do not overlap. Sources will be separated by >5*beam_width. Same units as fov.
    key: jax.random.PRNGKey
        Random number generator seed/key.

    Returns:
    --------
    I: array_like
        The sources intensities.
    delta_ra: array_like
        The sources right ascensions relative to (0,0).
    delta_dec: array_like
        The sources declinations relative to (0,0).
    """
    if not isinstance(key, jax.interpreters.xla._DeviceArray):
        key = random.PRNGKey(121)
    subkey, key = random.split(key)
    I = mean_I*random.exponential(key, (2*n_src,))
    positions = uniform_points_disk(fov/2., 2*n_src, subkey)
    source_d = jnp.linalg.norm(positions[:,:,None]-positions[:,None,:], axis=0)
    source_d = source_d + jnp.triu((5*beam_width+1.)*jnp.ones(source_d.shape))

    idx = list(jnp.arange(2*n_src))
    for i in jnp.unique(jnp.where(source_d<5*beam_width)[0]):
        idx.remove(i)
    idx = jnp.array(idx)

    return I[idx[:n_src]], positions[0,idx[:n_src]], positions[1,idx[:n_src]]

def save_observations(file_path, observations):
    """
    Save a list of observations to HDF5 file format.

    Parameters:
    -----------
    file_path: str
        File path (including file name) location to save the HDF5 file.
    observations: list
        List of observation class instances.
    """
    with h5py.File(file_path+'.h5', 'w') as fp:

        for i, obs in enumerate(observations):

            fp[f'track{i}/n_ant'] = obs.n_ant
            fp[f'track{i}/n_time'] = obs.n_time
            fp[f'track{i}/n_freq'] = obs.n_freq
            fp[f'track{i}/n_int_samples'] = obs.n_int_samples
            fp[f'track{i}/target'] = [obs.ra, obs.dec]
            fp[f'track{i}/int_time'] = obs.int_time
            fp[f'track{i}/times'] = obs.times
            fp[f'track{i}/times_fine'] = obs.times_fine
            fp[f'track{i}/freqs'] = obs.freqs
            fp[f'track{i}/ants_ENU'] = obs.ENU
            fp[f'track{i}/ants_XYZ'] = obs.ants_xyz
            fp[f'track{i}/ants_UVW'] = obs.ants_uvw

            fp[f'track{i}/latitude'] = obs.latitude
            fp[f'track{i}/longitude'] = obs.longitude
            fp[f'track{i}/elevation'] = obs.elevation
            fp[f'track{i}/antenna1'] = obs.a1
            fp[f'track{i}/antenna2'] = obs.a2
            fp[f'track{i}/vis_ast'] = obs.vis_ast
            fp[f'track{i}/vis_rfi'] = obs.vis_rfi
            fp[f'track{i}/vis_obs'] = obs.vis_obs
            fp[f'track{i}/noise'] = obs.noise_data

            fp[f'track{i}/gains'] = obs.gains_ants

            fp[f'track{i}/ast_I'] = obs.ast_I
            fp[f'track{i}/ast_radec'] = obs.ast_radec

            fp[f'track{i}/rfi_XYZ'] = obs.rfi_xyz
            fp[f'track{i}/rfi_orbit'] = obs.rfi_orbit
            fp[f'track{i}/rfi_A'] = obs.rfi_A_app

def load_antennas(telescope='MeerKAT'):
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
    if telescope == 'MeerKAT':
        enu = np.loadtxt(os.path.join(pkg_dir, '../data/MeerKAT.enu.txt'))
    else:
        print('Only MeerKAT antennas are currentyl available.')
        enu = None
    return enu
