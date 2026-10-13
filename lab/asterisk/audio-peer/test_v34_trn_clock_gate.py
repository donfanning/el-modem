"""TRN clock acceptance: primary validation rule plus the slope rule for moderate drift."""
import unittest
from v34_trn_clock import decide, MIN_SLOPE_PER_1000


def bounds(lo):
    return {8: (lo, 1.), 16: (lo, 1.), 32: (lo, 1.)}


class DecideTests(unittest.TestCase):
    def test_primary_rule_is_unchanged(self):
        self.assertEqual(decide(.92, bounds(.4), 3e-4), (True, 'validation'))
        self.assertEqual(decide(.5, bounds(.4), 0.), (True, 'validation'))
        self.assertEqual(decide(.12, bounds(-.001), 0.), (False, None))      # bootstrap not positive, no slope

    def test_undrifted_and_noise_lanes_stay_rejected(self):
        for improvement, slope in ((-.035, 3e-6), (-.05, 6e-6), (-.075, 7e-6), (.0, 7e-6), (.03, 5e-6)):
            self.assertEqual(decide(improvement, bounds(-.0005), slope), (False, None), (improvement, slope))

    def test_moderate_drift_with_small_improvement_is_accepted_on_slope(self):
        # slopes and improvements measured at +200, +225, +250 ppm (see the lane sweep)
        for improvement, slope in ((.023, -.443e-3), (.126, -.483e-3), (.088, -.560e-3)):
            self.assertEqual(decide(improvement, bounds(-.004), slope), (True, 'slope'), (improvement, slope))

    def test_slope_alone_is_not_enough_and_threshold_is_sane(self):
        self.assertEqual(decide(-.01, bounds(-.01), -5e-4), (False, None))   # a worse fit is never accepted
        self.assertEqual(decide(.0, bounds(-.01), -5e-4), (False, None))
        self.assertEqual(decide(.05, bounds(-.01), (MIN_SLOPE_PER_1000*.99)/1000), (False, None))
        self.assertGreater(MIN_SLOPE_PER_1000, 7*.007)                        # well above the undrifted maximum seen


if __name__ == '__main__':
    unittest.main()
