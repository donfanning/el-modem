"""Incremental V.90 S-transition detector checks."""

import sys
import unittest
from pathlib import Path

import numpy as np

import v34_tx
from v90_s_transition import V90STransitionDetector

import v90_test_ja_waveform as control


class V90STransitionDetectorTests(unittest.TestCase):
    def test_finds_each_s_phase_inversion_across_packet_boundaries(self):
        pair = np.concatenate((v34_tx.s_signal(128), v34_tx.s_signal(16, True)))
        samples = control._modulate(np.tile(pair, 3))
        detector = V90STransitionDetector(control.BAUD, control.CARRIER)
        events = []
        for offset in range(0, len(samples), 160):
            events.extend(detector.feed(samples[offset:offset + 160]))

        expected_first = int(128 * control.FS / control.BAUD)
        expected_period = int(144 * control.FS / control.BAUD)
        self.assertEqual([event for event, _at in events],
                         ["S", "flip"] * 3)
        for index in range(3):
            s_at = index * expected_period
            flip_at = s_at + expected_first
            self.assertLess(abs(events[2 * index][1] - s_at), 32, events)
            self.assertLess(abs(events[2 * index + 1][1] - flip_at), 32, events)
        self.assertEqual(len(detector.transitions), 3)

    def test_reports_s_onset_without_waiting_for_s_to_sbar(self):
        samples = control._modulate(v34_tx.s_signal(128))
        detector = V90STransitionDetector(control.BAUD, control.CARRIER)
        events = []
        for offset in range(0, len(samples), 160):
            events.extend(detector.feed(samples[offset:offset + 160]))
        self.assertEqual([name for name, _at in events], ["S"])
        self.assertLess(abs(events[0][1]), 32)
        self.assertEqual(detector.transitions, [])

    def test_rejects_discontinuity_until_explicit_reset(self):
        detector = V90STransitionDetector(control.BAUD, control.CARRIER)
        detector.feed(np.zeros(160), sample_at=1000)
        with self.assertRaisesRegex(ValueError, "discontinuity"):
            detector.feed(np.zeros(160), sample_at=1320)
        detector.reset(sample_at=1320)
        self.assertEqual(detector.feed(np.zeros(160), sample_at=1320), [])

    def test_bounded_history_still_finds_s_pair_after_long_silence(self):
        detector = V90STransitionDetector(control.BAUD, control.CARRIER)
        for offset in range(0, 16000, 160):
            detector.feed(np.zeros(160), sample_at=offset)
            self.assertLessEqual(len(detector._pcm), int(0.25 * control.FS))

        pair = np.concatenate((v34_tx.s_signal(128), v34_tx.s_signal(16, True)))
        samples = control._modulate(pair)
        events = []
        for offset in range(0, len(samples), 160):
            events.extend(detector.feed(
                samples[offset:offset + 160], sample_at=16000 + offset))

        self.assertEqual([name for name, _at in events], ["S", "flip"])
        expected_flip = 16000 + int(128 * control.FS / control.BAUD)
        self.assertLess(abs(events[1][1] - expected_flip), 32, events)


if __name__ == "__main__":
    unittest.main()
