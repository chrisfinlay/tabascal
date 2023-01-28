Usage
=====

.. _installation:

Installation
------------

To use tabascal, first install it using pip:

.. code-block:: console

   (venv) $ pip install git+https://github.com/chrisfinlay/tabascal.git

Creating an observation
----------------

To create an observation instantiate the
``tabascal.observation.Observation()`` class:

.. autoclass:: tabascal.observation.Observation

For example:

>>> from tabascal.observation import Observation
>>> from tabascal.utils.tools import load_antennas
>>> import numpy as np
>>> ants_enu = load_antennas('MeerKAT')
>>> obs = Observation(latitude=-30., longitude=21., elevation=1050.,
                  ra=27., dec=-15., times=np.arange(0, 10., 2),
                  freqs=np.linspace(1e9, 2e9, 128),
                  ENU_array=ants_enu, n_int_samples=4)
