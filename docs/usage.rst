Usage
=====

.. _installation:

Installation
------------

To use tabascal, first install it using pip:

.. code-block:: console

   (venv) $ pip install git+https://github.com/chrisfinlay/tabascal.git

Defining an observation
-----------------------

To create an observation, instantiate the
:py:class:`tabascal.observation.Observation` class.

For example:

.. code-block:: python

    from tabascal.observation import Observation
    from tabascal.utils.tools import load_antennas
    import jax.numpy as jnp

    ants_enu = load_antennas('MeerKAT')
    obs = Observation(latitude=-30., longitude=21., elevation=1050.,
                      ra=27., dec=-15., times=jnp.arange(0, 10., 2),
                      freqs=jnp.linspace(1e9, 2e9, 128),
                      ENU_array=ants_enu, n_int_samples=4)

Adding astronomical sources
---------------------------

Adding astronomical sources to an observation is done through the
:py:meth:`tabascal.observation.Observation.addAstro` method of the
:py:class:`tabascal.observation.Observation` class.

For example:

.. code-block:: python

    from jax import random

    n_src = 100
    mean_I = 0.1

    fov = beam_size(obs.dish_d, obs.freqs[-1])
    max_bl_length = jnp.max(jnp.linalg.norm(obs.bl_uvw[0], axis=-1))
    beam_width = beam_size(max_bl_length, obs.freqs[-1])
    I, d_ra, d_dec = generate_random_sky(n_src, mean_I, fov, beam_width)
    spectral_indices = 0.7 + 0.2*random.normal(random.PRNGKey(101), (n_src,))
    I = I[:,None] * (obs.freqs/obs.freqs[0])[None,:] ** -1.*spectral_indices[:,None]

    obs.addAstro(I=I, ra=obs.ra+d_ra, dec=obs.dec+d_dec)

Adding a satellite RFI source
-----------------------------

Adding a satellite based RFI source is done through the
:py:meth:`tabascal.observation.Observation.addSat` method of the
:py:class:`tabascal.observation.Observation` class.

For example:

.. code-block:: python

    rfi_P = 6e-4 * jnp.exp( -0.5 * ((obs.freqs-1.5e9)/2e7) ** 2 )

    obs.addSat(Pv=rfi_P, elevation=202e5, inclination=55.0,
               lon_asc_node=21.0, periapsis=5.0)

Adding some time and frequency dependent antenna gains
------------------------------------------------------

This done through the :py:meth:`tabascal.observation.Observation.addGains`
method.

For example:

.. code-block:: python

    obs.addGains(G0_mean=1.0, G0_std=0.05, Gt_std_amp=1e-5,
                 Gt_std_phase=jnp.deg2rad(1e-3))

Finally, calculate the visibilities and add some noise
------------------------------------------------------

This done through the :py:meth:`tabascal.observation.Observation.calculate_vis`
and :py:meth:`tabascal.observation.Observation.addNoise` methods.

For example:

.. code-block:: python

    obs.calculate_vis()
    obs.addNoise(noise=0.65, key=random.PRNGKey(999))
