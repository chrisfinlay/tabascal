Usage
=====

.. _installation:

Installation
------------

To use tabascal, first install it using pip:

.. code-block:: console

   (venv) $ pip install git+https://github.com/chrisfinlay/tabascal.git

Example observation script 
--------------------------

.. code-block:: console

    (venv) $ python examples/sim.py --satRFI yes --grdRFI yes --N_t 450 --N_f 128 --backend dask

The example script has help documentation

.. code-block:: console

    (venv) $ python examples/sim.py --help

Defining an observation
-----------------------

To create an observation, instantiate the
:py:class:`tabascal.jax.observation.Observation` class.

For example:

.. code-block:: python

    from tabascal.jax.observation import Observation
    from tabascal.utils.tools import load_antennas
    import numpy as np

    times = np.arange(0., 10., 2.)
    freqs = np.linspace(1.3, 1.6, 32)
    SEFD = 420. # Jy

    ants_enu = load_antennas('MeerKAT')
    obs = Observation(latitude=-30.0, longitude=21.0, elevation=1050.0,
                      ra=27.0, dec=15.0, times=times, freqs=freqs, 
                      SEFD=SEFD, ENU_array=ants_enu, n_int_samples=16)

Adding astronomical sources
---------------------------

Adding astronomical sources to an observation is done through the
:py:meth:`tabascal.jax.observation.Observation.addAstro` method of the
:py:class:`tabascal.jax.observation.Observation` class.

For example:

.. code-block:: python

    from tabascal.utils.tools import generate_random_sky

    I, d_ra, d_dec = generate_random_sky(n_src=100, mean_I=0.1, 
                                         freqs=obs.freqs, fov=obs.fov,
                                         beam_width=obs.syn_bw,
                                         random_seed=123)

    obs.addAstro(I=I, ra=obs.ra+d_ra, dec=obs.dec+d_dec)

Adding a satellite RFI source
-----------------------------

Adding a satellite based RFI source is done through the
:py:meth:`tabascal.jax.observation.Observation.addSatelliteRFI` method of the
:py:class:`tabascal.jax.observation.Observation` class.

For example:

.. code-block:: python

    rfi_P = 6e-4 * jnp.exp( -0.5 * ((obs.freqs-1.4e9)/2e7) ** 2 )

    obs.addSatelliteRFI(Pv=rfi_P, elevation=jnp.array([202e5]), 
                        inclination=jnp.array([55.0]),
                        lon_asc_node=jnp.array([21.0]), 
                        periapsis=jnp.array([5.0]))

Adding a stationary RFI source
------------------------------

Adding a stationary RFI source is done through the
:py:meth:`tabascal.jax.observation.Observation.addStationaryRFI` method of the
:py:class:`tabascal.jax.observation.Observation` class.

For example:

.. code-block:: python

    rfi_P = 6e-4 * jnp.exp( -0.5 * ((obs.freqs-1.5e9)/2e7) ** 2 )

    obs.addStationaryRFI(Pv=rfi_P, latitude=jnp.array([-20.]), 
                         longitude=jnp.array([30.]), 
                         elevation=jnp.array([tar.elevation]))

Adding some time and frequency dependent antenna gains
------------------------------------------------------

This done through the :py:meth:`tabascal.jax.observation.Observation.addGains`
method.

For example:

.. code-block:: python

    obs.addGains(G0_mean=1.0, G0_std=0.05, Gt_std_amp=1e-5,
                 Gt_std_phase=jnp.deg2rad(1e-3))

Calculating the observed visibilities
-------------------------------------

This done through the :py:meth:`tabascal.jax.observation.Observation.calculate_vis`
method.

For example:

.. code-block:: python

    obs.calculate_vis()

Saving the observation to a Measurement Set
-------------------------------------------

This is done through the :py:meth:`tabascal.jax.observation.Observation.write_to_ms`
method.

For example:

.. code-block:: python

    obs.write_to_ms("example_observation", overwrite=True)