"""Focused tests for the opt-in direct-V.34/no-V.8 answer detector."""

import unittest

import numpy as np

from v34_info import InfoModulator, build_info0
from v34_no_v8_detector import NoV8V34Detector


class NoV8V34DetectorTests(unittest.TestCase):
    def test_crc_valid_info0_followed_by_non_tone_is_rejected_at_tone_gate(self):
        info0c = InfoModulator(
            "call", amplitude=900.0, guard=False
        ).modulate(build_info0(ack=0, rate3429=1))
        signal = np.concatenate((
            np.rint(info0c).astype(np.int16),
            np.zeros(2000, dtype=np.int16),
        ))
        detector = NoV8V34Detector()

        matches = []
        for start in range(0, len(signal), 160):
            matches.append(detector.feed(signal[start:start + 160]))

        report = detector.report()
        self.assertFalse(any(matches))
        self.assertEqual(report["crc_valid_info0c_candidates"], 1)
        self.assertIsNone(report["match"])
        self.assertLess(report["best_following_tone_ratio"],
                        report["tone_ratio_threshold"])
        self.assertGreater(report["tone_ratio_shortfall"], 0.0)


if __name__ == "__main__":
    unittest.main()
