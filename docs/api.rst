Programming References
======================

Observation Class
-----------------

.. autoclass:: tabascal.jax.observation.Observation

.. automethod:: tabascal.jax.observation.Observation.addAstro

.. automethod:: tabascal.jax.observation.Observation.addSatelliteRFI

.. automethod:: tabascal.jax.observation.Observation.addStationaryRFI

.. automethod:: tabascal.jax.observation.Observation.addGains

.. automethod:: tabascal.jax.observation.Observation.calculate_vis

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
