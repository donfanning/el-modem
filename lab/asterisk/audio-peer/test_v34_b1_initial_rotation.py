"""An odd initial differential rotation is a quarter turn plus initial state [0,1,0,0].

The effective B1 initial state [0,1,0,0] seen on several recorded transmitters is not a
trellis reset: with the encoder state zero, a differential-encoder start Z0 that is odd
produces the same B1 symbols as state [0,1,0,0] with Z0 even, up to a constant quarter
turn that a receiver's complex gain absorbs. Even Z0 (including the 180 degree symmetry)
is the standard stream. See lab/V34_B1_INITIAL_ROTATION.md.
"""
import unittest

import numpy as np

import v34_b1_reference as ref

# Receive profiles (rate, baud, shaping, precoder) taken from retained recordings.
PROFILES = (
    (33600, 3428.5714285714284, True, (835, 259, -842, -106, 775, -157)),
    (24000, 3428.5714285714284, True, (589, 275, -597, -45, 562, -106)),
    (31200, 3200.0, True, (203, 56, -154, -88, 124, 62)),
    (31200, 3200.0, True, (0, 0, 0, 0, 0, 0)),
)


def x(profile, state, rotation):
    rate, baud, shaping, precoder = profile
    return np.array(ref.build(rate, baud, 'call', shaping, precoder, state, initial_rotation=rotation)['x'])


def rotated_equal(a, b):
    """Smallest max difference between a and any quarter-turn rotation of b."""
    return min(float(np.max(abs(a-(1j**k)*b))) for k in range(4))


class InitialRotationTests(unittest.TestCase):
    def test_default_is_unchanged(self):
        for profile in PROFILES:
            rate, baud, shaping, precoder = profile
            np.testing.assert_array_equal(
                ref.build(rate, baud, 'call', shaping, precoder)['x'],
                ref.build(rate, baud, 'call', shaping, precoder, initial_rotation=0)['x'])

    def test_odd_rotation_with_zero_state_is_state_0100_up_to_a_quarter_turn(self):
        for profile in PROFILES:
            for z0 in (1, 3):
                self.assertLess(rotated_equal(x(profile, (0, 0, 0, 0), z0), x(profile, (0, 1, 0, 0), 0)), 1e-9)

    def test_even_rotation_is_the_standard_stream_up_to_a_quarter_turn(self):
        for profile in PROFILES:
            self.assertLess(rotated_equal(x(profile, (0, 0, 0, 0), 2), x(profile, (0, 0, 0, 0), 0)), 1e-9)
            self.assertLess(rotated_equal(x(profile, (0, 1, 0, 0), 1), x(profile, (0, 0, 0, 0), 0)), 1e-9)

    def test_the_two_states_are_genuinely_different_streams(self):
        # Without the rotation the identity would be vacuous: the zero-state and
        # state-0100 streams differ by far more than a quarter turn.
        for profile in PROFILES[:3]:
            self.assertGreater(rotated_equal(x(profile, (0, 1, 0, 0), 0), x(profile, (0, 0, 0, 0), 0)), 1.0)

    def test_invalid_rotation_is_rejected(self):
        with self.assertRaises(ValueError):
            ref.build(33600, 3428.5714285714284, 'call', True, PROFILES[0][3], initial_rotation=4)


if __name__ == '__main__':
    unittest.main()
