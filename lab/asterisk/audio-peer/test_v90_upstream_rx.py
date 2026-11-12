"""Synthetic end-to-end test of V.90 upstream Phase 3 Ja recovery."""

import sys
import unittest
from collections import deque
from pathlib import Path

import numpy as np

import v34_rx
import v34_tx
import v90_info_reference as v90
from v90_ja import V90JaScanner
from v90_upstream_rx import V90UpstreamPhase3Receiver

import v90_test_ja_waveform as control


class V90UpstreamPhase3ReceiverTests(unittest.TestCase):
    def test_four_point_slicer_reuses_its_rotation_decision(self):
        receiver = object.__new__(V90UpstreamPhase3Receiver)
        angles = np.linspace(-np.pi, np.pi, 2049, endpoint=False)
        # Include exact constellation points and dense points around each
        # nearest-point boundary, where a separate angle calculation matters.
        angles = np.concatenate((angles, np.asarray([
            np.pi / 4 + offset
            for offset in (-1e-12, -1e-15, 0.0, 1e-15, 1e-12)
        ])))
        for angle in angles:
            y = 1.37 * complex(np.cos(angle), np.sin(angle))
            decided = receiver._decide4_with_rotation(y)
            expected = v34_rx.decide4(y)
            old_rotation = int(round(-np.arctan2(
                (expected / v34_tx.P0).imag,
                (expected / v34_tx.P0).real) / (np.pi / 2))) % 4
            self.assertEqual(decided, expected)
            self.assertEqual(receiver._last_rotation, old_rotation)

    def test_phase4_16point_slicer_emits_quarter_and_rotation_decisions(self):
        class DecisionEq:
            def __init__(self, values):
                self.values = list(values)
                self.k = 0
                self.phase = 0.0

            def available(self):
                return len(self.values)

            def window(self, _k):
                return None

            def step(self, *, decide, _window=None):
                value = self.values[self.k]
                decide(value)
                self.k += 1
                return value

        receiver = object.__new__(V90UpstreamPhase3Receiver)
        receiver.eq = DecisionEq(v34_rx._C16_PTS_TUPLE)
        receiver._data_decider = None
        receiver._symbol_callback = None
        receiver.role = "call"
        receiver.scrambler = v34_tx.Scrambler("call")
        receiver.indices = []
        receiver._index_storage = np.empty(0, dtype=np.int64)
        receiver.quarters = []
        receiver._equalized_pending = deque()
        receiver.mismatch = []
        receiver.trn_end = 0
        receiver.scanner = V90JaScanner()
        receiver._symbol_cursor = 0
        receiver._decision_cursor = 0
        receiver._phase4_active = True
        receiver._phase4_constellation = 16
        receiver._sq = None
        receiver._last = 0j
        receiver._ja_symbol_cursor = 0
        receiver._stage_timing_enabled = True
        receiver._stage_timing_rows = deque(maxlen=2048)

        receiver.feed()
        decisions = receiver.read_new_symbol_decisions()
        self.assertEqual(len(decisions), 16)
        self.assertTrue(all(0 <= rotation < 4 and 0 <= quarter < 4
                            for rotation, quarter in decisions))
        self.assertTrue(any(quarter != 0 for _, quarter in decisions))
        timing = receiver.timing_report()
        self.assertEqual(timing["count"], 1)
        self.assertEqual(receiver._stage_timing_rows[-1]["symbols"], 16)
        self.assertEqual(set(timing["stages"]),
                         {"total", "equalizer", "symbol_routing", "ja_scan"})

    def test_recovers_repeated_crc_valid_ja_from_independent_waveform(self):
        descriptor_bits = control._ja_descriptor(7, 19, 31)
        expected = v90.parse_ja_descriptor(descriptor_bits)
        sequence, scrambler_state, rotation_state = control._preamble_and_trn()
        ja_bits = descriptor_bits * 3
        ja_symbols, _ = control._differential(ja_bits, rotation_state[0],
                                              scrambler_state)
        samples = control._modulate(np.concatenate((sequence, ja_symbols)))

        frontend = v34_rx.Phase3Rx(control.BAUD, control.CARRIER, "answer")
        acquired = frontend.acquire(samples)
        self.assertIsNotNone(acquired)
        mse, role = frontend.train_trn_auto()
        self.assertLess(mse, 0.1)

        # Keep the fitted frontend, but replay newly arriving PCM in 20 ms
        # increments through its append-only baseband updater.
        frontend._bb_input_len = 0
        frontend._bb_origin = 0
        frontend._bb_storage = None
        frontend._z = np.zeros(0, dtype=np.complex128)
        receiver = V90UpstreamPhase3Receiver(frontend, role)
        pcm = np.zeros(0, dtype=float)
        descriptors = []
        for offset in range(0, len(samples), 160):
            pcm = np.concatenate((pcm, samples[offset:offset + 160]))
            frontend._z = frontend.update_baseband(pcm)
            descriptors.extend(receiver.feed())
        self.assertEqual(receiver.trn_end, 1401)
        self.assertGreaterEqual(len(descriptors), 1)
        self.assertTrue(all(item["crc_valid"] and item["valid"]
                            for item in descriptors))
        self.assertEqual(descriptors[0]["N"], 7)
        self.assertEqual(descriptors[0]["LSP"], 19)
        self.assertEqual(descriptors[0]["LTP"], 31)
        self.assertEqual(descriptors[0]["Ucodes"], expected["Ucodes"])
        self.assertEqual(receiver.feed(), [])  # no duplicate emission on an empty poll
        self.assertEqual(receiver.read_new_symbol_rotations(), receiver.indices)
        self.assertEqual(receiver.read_new_symbol_rotations(), [])

    def test_index_storage_grows_geometrically_and_matches_decision_list(self):
        receiver = object.__new__(V90UpstreamPhase3Receiver)
        receiver.indices = []
        receiver._index_storage = np.empty(0, dtype=np.int64)
        expected = [i % 4 for i in range(5000)]
        for decision in expected:
            receiver._append_rotation(decision)

        view = receiver._indices_view()
        np.testing.assert_array_equal(view, expected)
        self.assertTrue(np.shares_memory(view, receiver._index_storage))
        self.assertGreaterEqual(receiver._index_storage.size, len(expected))
        self.assertLessEqual(receiver._index_storage.size, 2 * len(expected))


if __name__ == "__main__":
    unittest.main()
