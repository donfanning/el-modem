"""B1 replay must preserve calibration and reject inconsistent physical rows."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent/'tools'))
from b1_postcal_tracking_replay import replay
import v34_receive as receive
import v34_negotiation as negotiation


class PostcalTrackingTests(unittest.TestCase):
    def fixture(self):
        profile = negotiation.DirectionParameters(rate=33600, baud=24000/7,
            carrier=1959.1836734693877, trellis=0, shaping=True,
            nonlinear=True, precoder=(129,188,-297,-160,222,-69))
        reference = receive.Receiver(profile, 'answer')
        projected = receive.projection(reference.linear, 675.)
        rows = np.zeros((len(projected),48), complex)
        rows[:,24] = (.11+.002j)*projected
        taps = np.zeros(48, complex); taps[24] = 1.
        return profile, dict(windows=rows, taps_before=taps.copy(), taps_after=taps.copy(),
            observed_before=rows@taps, observed_after=rows@taps)

    def test_clean_independent_b1_reference_stays_qualified(self):
        profile, capture = self.fixture()
        for track in (False, True):
            row, output = replay(capture, profile, (0,0,0,0), track=track)
            self.assertEqual(row['qualification']['bit_errors'], 0)
            np.testing.assert_array_equal(output[:len(output)//2], capture['observed_after'][:len(output)//2])

    def test_late_channel_change_cannot_change_calibration_observations(self):
        profile, capture = self.fixture()
        first = len(capture['windows'])//2
        capture['windows'][first:] *= 1.03
        capture['observed_before'] = capture['windows']@capture['taps_before']
        capture['observed_after'] = capture['windows']@capture['taps_after']
        row, output = replay(capture, profile, (0,0,0,0), track=True)
        self.assertGreater(row['correction_energy_fraction'], 0.)
        np.testing.assert_array_equal(output[:first], capture['observed_after'][:first])

    def test_inconsistent_saved_observations_abort_before_decoding(self):
        profile, capture = self.fixture()
        capture['observed_after'][70] += 1.
        with self.assertRaisesRegex(ValueError, 'reconstruction failed'):
            replay(capture, profile, (0,0,0,0), track=True)


if __name__ == '__main__':
    unittest.main()
