"""Capture regression for the V.92 fixed-lag 32-state upstream decoder.

The compact fixture contains equalized constellation samples from the local
SmartLink arm-10c capture, beginning at B1u and ending at DATA frame 608. It is
not raw PCM. Eight traceback groups first commit a parity-inconsistent survivor
near frame 656; 64 groups preserves the valid path through that interval.
"""
from pathlib import Path
import unittest

import numpy as np

import v92_pcm_upstream as pu
from v92_pcm_upstream_rx import PcmUpstreamStreamReceiver


FIXTURE = Path(__file__).with_name("testdata") / "v92_arm10c_b1u_to_data608.npz"
LEVELS = (96, 292, 492, 716, 988, 1244, 1500, 1756, 1980, 2236)
PROFILE = pu.PcmUpstreamProfile(
    constellations=(LEVELS,), interval_set=(0,) * pu.INTERVALS,
    moduli=(10, 10, 10, 5) * 3, trellis=1, bits=36,
)


@unittest.skipUnless(FIXTURE.exists(), "private capture fixture not present")
class V92TracebackCaptureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with np.load(FIXTURE, allow_pickle=False) as data:
            cls.samples = np.asarray(data["equalized"], dtype=float).reshape(-1)

    def test_viterbi32_default_uses_64_group_traceback(self):
        rx = PcmUpstreamStreamReceiver(PROFILE, "pcmu", decoder_mode="viterbi32")
        self.assertEqual(rx._soft.traceback_groups, 64)

    def test_long_traceback_preserves_frame_656_path(self):
        long = PcmUpstreamStreamReceiver(PROFILE, "pcmu", decoder_mode="viterbi32")
        long.feed_samples(self.samples)
        self.assertEqual(long.state, long.DATA)
        self.assertEqual(long.b1_errors, 0)
        self.assertEqual(long.trellis_violations, 0)
        self.assertGreaterEqual(long.frames, 656)

        short = PcmUpstreamStreamReceiver(
            PROFILE, "pcmu", decoder_mode="viterbi32", viterbi_traceback_groups=8)
        short.feed_samples(self.samples)
        self.assertGreater(short.trellis_violations, 0)


if __name__ == "__main__":
    unittest.main()
