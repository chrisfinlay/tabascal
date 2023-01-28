API
===

Observation Class
-----------------

.. autoclass:: tabascal.observation.Observation

.. automethod:: tabascal.observation.Observation.addAstro

.. automethod:: tabascal.observation.Observation.addSat

.. automethod:: tabascal.observation.Observation.addGains

.. automethod:: tabascal.observation.Observation.addNoise

.. automethod:: tabascal.observation.Observation.calculate_vis

Coordinates
-----------

.. autofunction:: tabascal.coordinates.radec_to_lmn

.. autofunction:: tabascal.coordinates.radec_to_XYZ

.. autofunction:: tabascal.coordinates.ENU_to_GEO

.. autofunction:: tabascal.coordinates.GEO_to_XYZ

.. autofunction:: tabascal.coordinates.ENU_to_ITRF

.. autofunction:: tabascal.coordinates.ENU_to_UVW

.. autofunction:: tabascal.coordinates.Rotx

.. autofunction:: tabascal.coordinates.Rotz

.. autofunction:: tabascal.coordinates.orbit

.. autofunction:: tabascal.coordinates.orbit_velocity

.. autofunction:: tabascal.coordinates.RIC_dev

.. autofunction:: tabascal.coordinates.orbit_fisher

Interferometry
--------------

.. autofunction:: tabascal.interferometry.rfi_vis

.. autofunction:: tabascal.interferometry.astro_vis

.. autofunction:: tabascal.interferometry.ants_to_bl
