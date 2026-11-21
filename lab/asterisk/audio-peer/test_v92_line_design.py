"""Known-response checks for the quantized upstream channel-design objective."""
import unittest

import numpy as np

import v34_g711
from v92_digital_session import _quantisation_isi


class QuantizedDesignTests(unittest.TestCase):
    def plan(self, law):
        levels = sorted({int(x) for x in v34_g711.decode(np.arange(256), law) if x > 0})[2:50]
        return {"constellation": levels, "moduli": [30, 30, 30, 15] * 3,
                "trellis": 1, "drn": 9}

    def test_delayed_identity_has_zero_distortion_over_full_prefilter_support(self):
        for law in ("pcmu", "pcma"):
            for delay in (0, 47, 48, 90, 255):
                with self.subTest(law=law, delay=delay):
                    # A quantized gain of 1/2 followed by channel gain 2 is
                    # exactly a delayed identity, independent of random data.
                    precoder = {"p1": [], "z2": [0] * delay + [16384]}
                    error, found = _quantisation_isi([2.0], precoder, self.plan(law), law)
                    self.assertEqual(found, delay)
                    self.assertLess(error, 1e-12)

    def test_delay_alignment_preserves_a_known_gain_error(self):
        precoder = {"p1": [], "z2": [0] * 90 + [16384]}
        error, found = _quantisation_isi([1.8], precoder, self.plan("pcmu"), "pcmu")
        self.assertEqual(found, 90)
        self.assertAlmostEqual(error, 0.1, places=12)

    def test_probe_must_extend_beyond_the_warmup(self):
        with self.assertRaisesRegex(ValueError, "300 samples"):
            _quantisation_isi([2.0], {"p1": [], "z2": [16384]},
                              self.plan("pcmu"), "pcmu", frames=25)


if __name__ == "__main__":
    unittest.main()
