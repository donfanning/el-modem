"""Streaming acquisition/training qualification for V.90 upstream Phase 3."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from unittest.mock import patch

import v34_tx
import v34_rx
import v90_info_reference as v90
from v90_upstream_session import V90UpstreamPhase3Session

import v90_test_ja_waveform as control


class V90UpstreamPhase3SessionTests(unittest.TestCase):
    def test_tracking_cap_outpaces_every_upstream_symbol_rate(self):
        """The per-packet tracking cap must drain faster than the line delivers.

        feed() runs once per RTP packet, and our SIP stacks offer only 10 or
        20 ms packets. A cap below the symbol rate lets the backlog grow, as
        the V.34 Phase-4 tracker did on the 2026-11-09 captured peer calls.
        """
        packets_per_second = 8000 / 160
        for baud in (3000.0, 3200.0, 3428.5714285714):
            with self.subTest(baud=baud):
                self.assertGreaterEqual(
                    V90UpstreamPhase3Session.MAX_SYMBOLS_PER_MEDIA_FEED
                    * packets_per_second, 1.25 * baud)

    def test_tracking_catchup_is_bounded_per_media_feed(self):
        session = V90UpstreamPhase3Session(3428.5714285714, 1959.1836734694)
        session.state = session.TRACKING
        session.frontend = SimpleNamespace(
            update_baseband=lambda pcm: np.asarray(pcm, dtype=np.complex128))
        calls = []

        class Receiver:
            eq = SimpleNamespace(k=432)

            def feed(self, *, max_symbols=None):
                calls.append(max_symbols)
                return []

        session.receiver = Receiver()
        session.feed(np.ones(160, dtype=np.float64))
        self.assertEqual(calls, [session.MAX_SYMBOLS_PER_MEDIA_FEED])
        self.assertEqual(session.MAX_SYMBOLS_PER_MEDIA_FEED, 128)

    def test_startup_trace_is_opt_in_and_reports_acquisition_work(self):
        with patch.dict("os.environ", {"V90_UPSTREAM_TRACE": "1"}):
            session = V90UpstreamPhase3Session(
                3428.5714285714, 1959.1836734694)
        self.assertEqual(session.startup_trace_report()["count"], 0)
        session._startup_trace.append({
            "kind": "acquire", "samples_seen": 8000,
            "capture_samples": 8000, "accepted": False,
            "metric": None, "cpu_ns": 123456,
        })
        report = session.startup_trace_report()
        self.assertEqual(report["max_cpu_ns"], 123456)
        self.assertEqual(report["rows"][0]["capture_samples"], 8000)

    def test_startup_trace_keeps_the_best_rejected_acquisition_candidate(self):
        with patch.dict("os.environ", {"V90_UPSTREAM_TRACE": "1"}):
            session = V90UpstreamPhase3Session(
                3428.5714285714, 1959.1836734694)

        candidate = {"metric": 0.31, "lag": 440, "delay": 0.5}

        def reject_with_candidate(frontend, _samples, **_kwargs):
            frontend.last_best = candidate
            return None

        with patch("v90_upstream_session.v34_rx.Phase3Rx.acquire",
                   autospec=True, side_effect=reject_with_candidate):
            for _ in range(14):
                session.feed(np.full(160, 500.0))

        rows = [row for row in session.startup_trace_report()["rows"]
                if row["kind"] == "acquire"]
        self.assertTrue(rows)
        self.assertEqual(rows[-1]["best_candidate"], candidate)

    def test_streaming_acquisition_waits_for_full_candidate_window(self):
        session = V90UpstreamPhase3Session(
            3428.5714285714, 1959.1836734694)

        def candidate(frontend, _samples, **_kwargs):
            frontend.lag = 1000
            frontend.delay = 0.5
            frontend.gain = 1.0
            return {"metric": 0.9, "lag": 1000,
                    "delay": 0.5, "gain": 1.0}

        with patch("v90_upstream_session.v34_rx.Phase3Rx.acquire",
                   autospec=True, side_effect=candidate), \
                patch.object(V90UpstreamPhase3Session,
                             "_refine_acquisition",
                             autospec=True,
                             side_effect=lambda _session, frontend, rough: rough):
            session.feed(np.zeros(2000))
            self.assertEqual(session.state, session.ACQUIRING)
            session.feed(np.zeros(160))

        self.assertEqual(session.state, session.TRAINING)
        self.assertIsNone(session.training_mse)

    def test_a_silent_line_stops_the_acquisition_search_and_a_signal_resumes_it(self):
        session = V90UpstreamPhase3Session(3428.5714285714, 1959.1836734694)
        with patch("v90_upstream_session.v34_rx.Phase3Rx.acquire",
                   autospec=True, return_value=None) as acquire:
            for idle in (np.zeros(160), np.full(160, 8.0)):          # mu-law and A-law idle
                acquire.reset_mock()
                session = V90UpstreamPhase3Session(3428.5714285714, 1959.1836734694)
                for _ in range(100):
                    session.feed(idle)
                calls = acquire.call_count
                for _ in range(100):
                    session.feed(idle)
                self.assertEqual(acquire.call_count, calls)         # nothing more after silence
                noisy = np.random.default_rng(1).normal(0, 500, 160)
                session.feed(noisy)
                for _ in range(10):
                    session.feed(noisy)
                self.assertEqual(acquire.call_count, calls + 1)     # signal: searching again

    def test_synthetic_waveform_uses_selected_v34_carrier(self):
        # Carrier Table 2 is d/e times the selected symbol rate, not the
        # 2400-bit/s rate-grid base. This guards the PCM fixture that feeds
        # the Phase 3/B1 answerer tests.
        self.assertEqual(control.CARRIER, v34_tx.CARRIERS[round(control.BAUD)][0])

    def test_info1d_probe_offset_centers_upstream_receive_profile(self):
        session = V90UpstreamPhase3Session(
            3428.5714285714, 1959.1836734694,
            probe_frequency_offset_units=5)
        ratio = 0.1 / 1050.0 + 1.0
        self.assertAlmostEqual(session.receive_clock_ratio, ratio)
        self.assertAlmostEqual(session.receive_baud, 3428.5714285714 * ratio)
        self.assertAlmostEqual(session.receive_carrier, 1959.1836734694 * ratio)
        ignored = V90UpstreamPhase3Session(
            3428.5714285714, 1959.1836734694,
            probe_frequency_offset_units=-512)
        self.assertEqual(ignored.receive_clock_ratio, 1.0)
        with self.assertRaisesRegex(ValueError, "frequency offset"):
            V90UpstreamPhase3Session(
                3428.5714285714, 1959.1836734694,
                probe_frequency_offset_units=512)

    def test_streaming_acquisition_training_and_ja_decode(self):
        descriptor_bits = control._ja_descriptor(7, 19, 31)
        expected = v90.parse_ja_descriptor(descriptor_bits)
        sequence, _, _ = control._preamble_and_trn()
        scrambled, scrambler_state = control._scramble([1] * (2 * 512))
        trn = []
        for offset in range(0, len(scrambled), 2):
            i1, i2 = scrambled[offset:offset + 2]
            rotation = (2 * i2 + i1) % 4
            trn.append((1 + 1j) * [1, -1j, -1, 1j][rotation])
        rotation_state = [(2 * scrambled[-1] + scrambled[-2]) % 4]
        ja_bits = descriptor_bits * 3
        ja_symbols, _ = control._differential(
            ja_bits, rotation_state[0], scrambler_state)
        samples = control._modulate(np.concatenate((sequence[:432], trn, ja_symbols)))

        session = V90UpstreamPhase3Session(
            control.BAUD, control.CARRIER, trn_scrambler_role="answer")
        descriptors = []
        for offset in range(0, len(samples), 160):
            descriptors.extend(session.feed(samples[offset:offset + 160]))

        self.assertEqual(session.state, session.COMPLETE)
        self.assertEqual(session.receiver.trn_end, 512)
        self.assertLess(session.training_mse, 0.1)
        self.assertIn(session.detected_trn_role, ("answer", "call"))
        self.assertGreaterEqual(len(descriptors), 1)
        self.assertTrue(all(item["crc_valid"] and item["valid"]
                            for item in descriptors))
        self.assertEqual(descriptors[0]["N"], 7)
        self.assertEqual(descriptors[0]["LSP"], 19)
        self.assertEqual(descriptors[0]["LTP"], 31)
        self.assertEqual(descriptors[0]["Ucodes"], expected["Ucodes"])

    def test_segmented_acquisition_refines_level_changed_preamble(self):
        baud = 3200.0
        carrier = v34_tx.CARRIERS[3200][0]
        preamble = np.concatenate((
            v34_tx.s_signal(128) / np.sqrt(2.0),
            v34_tx.s_signal(16, True) / np.sqrt(2.0),
            v34_tx.pp_signal(),
        ))
        prefix = np.zeros(800)
        waveform = v34_tx.qam_modulate(
            preamble, baud, carrier, 1.0, delay=0.25)
        samples = np.concatenate((prefix, waveform, np.zeros(500)))
        frontend = v34_rx.Phase3Rx(baud, carrier, "answer")
        rough = frontend.acquire(samples, min_metric=0.0)
        self.assertIsNotNone(rough)

        session = V90UpstreamPhase3Session(baud, carrier)
        refined = session._refine_acquisition(frontend, rough)

        self.assertIsNotNone(refined)
        self.assertAlmostEqual(refined["lag"], len(prefix), delta=1)
        self.assertEqual(refined["delay"], 0.25)
        self.assertGreater(refined["segment_metrics"]["s_sbar"], 0.8)
        self.assertGreater(refined["segment_metrics"]["pp"], 0.8)

    def test_trn_fit_handles_standard_power_normalized_four_point_symbols(self):
        baud = 3200.0
        carrier = v34_tx.CARRIERS[3200][0]
        trn, _ = v34_tx.trn_symbols(512, "answer")
        symbols = np.concatenate((
            v34_tx.s_signal(128) / np.sqrt(2.0),
            v34_tx.s_signal(16, True) / np.sqrt(2.0),
            v34_tx.pp_signal()[:288],
            trn / np.sqrt(2.0),
        ))
        waveform = v34_tx.qam_modulate(
            symbols, baud, carrier, 1.0, delay=0.25)
        frontend = v34_rx.Phase3Rx(baud, carrier, "answer")
        frontend.lag = 0
        frontend.delay = 0.25
        frontend._z = frontend.update_baseband(waveform)
        session = V90UpstreamPhase3Session(baud, carrier)

        mse, role, scale = session._train_power_normalized(frontend)

        self.assertEqual(role, "answer")
        self.assertLess(mse, 0.01)
        self.assertAlmostEqual(abs(scale), 1.0 / np.sqrt(2.0), delta=0.03)
        np.testing.assert_allclose(
            frontend.equalize(432, 512), trn, atol=0.06, rtol=0.06)

    def test_startup_pcm_retention_is_bounded(self):
        session = V90UpstreamPhase3Session(
            control.BAUD, control.CARRIER, max_seconds=0.1)
        session.feed(np.zeros(800))
        self.assertEqual(session.state, session.ACQUIRING)
        session.feed(np.zeros(1))
        self.assertEqual(session.state, session.FAILED)
        self.assertIn("bounded", session.failure_reason)

    def test_data_phase_has_its_own_larger_bound(self):
        # Regression: the 40 s startup bound used to apply to the whole call, so any V.90
        # transfer that ran past 40 s of upstream audio died with "startup exceeded".
        session = V90UpstreamPhase3Session(
            control.BAUD, control.CARRIER, max_seconds=0.1, max_data_seconds=0.3)
        session.feed(np.zeros(800))
        session._data_phase = True          # what begin_data_tracking() sets
        session.feed(np.zeros(1600))        # past the startup bound, within the data bound
        self.assertNotEqual(session.state, session.FAILED)
        session.feed(np.zeros(100))         # 2,500 samples > 0.3 s
        self.assertEqual(session.state, session.FAILED)
        self.assertIn("data phase exceeded", session.failure_reason)
        with self.assertRaises(ValueError):
            V90UpstreamPhase3Session(control.BAUD, control.CARRIER,
                                     max_seconds=5.0, max_data_seconds=1.0)

    def test_pcm_history_grows_geometrically_and_preserves_samples(self):
        session = V90UpstreamPhase3Session(
            control.BAUD, control.CARRIER, max_seconds=0.1,
            max_data_seconds=0.5)
        chunks = [np.arange(160, dtype=np.float64) + 160 * index
                  for index in range(25)]
        for chunk in chunks:
            session._append_pcm(chunk)

        expected = np.concatenate(chunks)
        np.testing.assert_array_equal(session._pcm, expected)
        self.assertEqual(session._pcm_length, len(expected))
        self.assertGreaterEqual(session._pcm_storage.size, len(expected))
        self.assertLessEqual(session._pcm_storage.size, session.max_data_samples)
        # Geometric growth leaves less than one retained-buffer copy's slack.
        self.assertLessEqual(session._pcm_storage.size, 2 * len(expected))

    def test_pcm_history_capacity_never_exceeds_data_retention_bound(self):
        session = V90UpstreamPhase3Session(
            control.BAUD, control.CARRIER, max_seconds=0.1,
            max_data_seconds=0.3)
        session._data_phase = True
        for _ in range(15):
            session.feed(np.zeros(160))
        self.assertEqual(session.samples_seen, session.max_data_samples)
        self.assertEqual(len(session._pcm), session.max_data_samples)
        self.assertLessEqual(session._pcm_storage.size, session.max_data_samples)
        session.feed(np.zeros(1))
        self.assertEqual(session.state, session.FAILED)
        self.assertIn("data phase exceeded", session.failure_reason)


if __name__ == "__main__":
    unittest.main()
