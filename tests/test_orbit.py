"""Unit tests exercising the :mod:`Orbit` helper class.

The tests here ensure that a simple ``Orbit`` instance can return
interpolated positions and velocities at arbitrary epochs.  The helper
function :func:`create_test_orbit` builds a tiny synthetic orbit used by
the test cases.
"""

import os
import sys
import numpy as np
from astropy.time import Time
from astropy.coordinates import CartesianRepresentation, CartesianDifferential
from astropy import units as u

# Ensure the package root is on the path when running via ``pytest -q``.
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from Orbit import Orbit


def create_test_orbit():
    """Return a minimal ``Orbit`` instance for testing.

    The orbit is constructed directly without reading from files and
    contains a few epochs with associated Cartesian position and
    velocity vectors so that interpolation can be tested deterministically.
    """
    epochs = Time([0, 1, 2], format="jd")
    positions = CartesianRepresentation(
        [0, 1, 2] * u.au,
        [10, 10, 10] * u.au,
        [0, -1, -2] * u.au,
    )
    velocities = CartesianDifferential(
        [1, 1, 1] * u.au / u.day,
        [0, 0, 0] * u.au / u.day,
        [2, 2, 2] * u.au / u.day,
    )
    orbit = Orbit.__new__(Orbit)
    orbit.epochs = epochs
    orbit.positions = positions
    orbit.velocities = velocities
    return orbit


def test_get_pos_and_vel():
    """Check ``Orbit.get_pos`` and ``Orbit.get_vel`` interpolation.

    The test queries an ``Orbit`` built by :func:`create_test_orbit` at
    epochs between the defined samples and verifies that the returned
    arrays have the expected shapes, values and units.
    """

    orbit = create_test_orbit()
    times = [0, 1.5]
    pos = orbit.get_pos(times)
    vel = orbit.get_vel(times)

    expected_pos = np.array([[0.0, 1.5], [10.0, 10.0], [0.0, -1.5]])
    expected_vel = (
        np.array([[1.0, 1.0], [0.0, 0.0], [2.0, 2.0]]) * u.au / u.day
    )

    assert pos.shape == expected_pos.shape
    assert np.allclose(pos, expected_pos)

    assert vel.shape == expected_vel.shape
    assert np.allclose(vel.value, expected_vel.value)
    assert vel.unit == expected_vel.unit
