from tabascal.jax.coordinates import radec_to_lmn
import pytest
import numpy as np


def test_radec_to_lmn1():
    ra, dec, phase_centre = 0.0, 0.0, (0.0, 0.0)
    lmn = radec_to_lmn(ra, dec, phase_centre)
    assert np.allclose(lmn, np.array([0, 0, 1]))


def test_radec_to_lmn2():
    ra, dec, phase_centre = 0.0, 0.0, (0.0, 0.0)
    with pytest.raises(TypeError):
        ra = "0"
        radec_to_lmn(ra, dec, phase_centre)


def test_radec_to_lmn3():
    ra, dec, phase_centre = 0.0, 0.0, (0.0, 0.0)
    with pytest.raises(TypeError):
        dec = "0"
        radec_to_lmn(ra, dec, phase_centre)


def test_radec_to_lmn4():
    ra, dec, phase_centre = 0.0, 0.0, (0.0, 0.0)
    with pytest.raises(TypeError):
        phase_centre = "0,0"
        radec_to_lmn(ra, dec, phase_centre)
