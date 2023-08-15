Programming References
======================

Observation Class (JAX)
-----------------------

.. autoclass:: tabascal.jax.observation.Observation

.. automethod:: tabascal.jax.observation.Observation.addAstro

.. automethod:: tabascal.jax.observation.Observation.addSatelliteRFI

.. automethod:: tabascal.jax.observation.Observation.addStationaryRFI

.. automethod:: tabascal.jax.observation.Observation.addGains

.. automethod:: tabascal.jax.observation.Observation.calculate_vis

.. automethod:: tabascal.jax.observation.Observation.write_to_zarr

.. automethod:: tabascal.jax.observation.Observation.write_to_ms


Observation Class (Dask)
-----------------------

.. autoclass:: tabascal.dask.observation.Observation

.. automethod:: tabascal.dask.observation.Observation.addAstro

.. automethod:: tabascal.dask.observation.Observation.addSatelliteRFI

.. automethod:: tabascal.dask.observation.Observation.addStationaryRFI

.. automethod:: tabascal.dask.observation.Observation.addGains

.. automethod:: tabascal.dask.observation.Observation.calculate_vis

.. automethod:: tabascal.dask.observation.Observation.write_to_zarr

.. automethod:: tabascal.dask.observation.Observation.write_to_ms

Coordinates
-----------

.. autofunction:: tabascal.jax.coordinates.radec_to_lmn

.. autofunction:: tabascal.jax.coordinates.radec_to_XYZ

.. autofunction:: tabascal.jax.coordinates.ENU_to_GEO

.. autofunction:: tabascal.jax.coordinates.GEO_to_XYZ

.. autofunction:: tabascal.jax.coordinates.ENU_to_ITRF

.. autofunction:: tabascal.jax.coordinates.ENU_to_UVW

.. autofunction:: tabascal.jax.coordinates.Rotx

.. autofunction:: tabascal.jax.coordinates.Rotz

.. autofunction:: tabascal.jax.coordinates.orbit

.. autofunction:: tabascal.jax.coordinates.orbit_velocity

.. autofunction:: tabascal.jax.coordinates.RIC_dev

.. autofunction:: tabascal.jax.coordinates.orbit_fisher

Interferometry
--------------

.. autofunction:: tabascal.jax.interferometry.rfi_vis

.. autofunction:: tabascal.jax.interferometry.astro_vis

.. autofunction:: tabascal.jax.interferometry.ants_to_bl

.. autofunction:: tabascal.jax.interferometry.airy_beam

.. autofunction:: tabascal.jax.interferometry.Pv_to_Sv

.. autofunction:: tabascal.jax.interferometry.SEFD_to_noise_std

.. autofunction:: tabascal.jax.interferometry.int_sample_times

.. autofunction:: tabascal.jax.interferometry.generate_gains

.. autofunction:: tabascal.jax.interferometry.apply_gains

.. autofunction:: tabascal.jax.interferometry.time_avg