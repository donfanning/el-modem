import json
from pathlib import Path
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np

import v34_info
import v90_info_reference as v90
from v90_digital_phase2 import V90DigitalPhase2Signals
from autochain import V8Chain
from g711 import alaw_to_linear, linear_to_ulaw, ulaw_to_linear
from v8 import V8_MOD_V34, V8_PCM_DIGITAL, V8_PROTOCOL_LAPM_V42
from v90_digital_phase2_engine import V90DigitalAnswererPhase2Engine
from v90_media_clock import RtpMediaClock
from v90_digital_phase2_control import Phase2Action


class V90DigitalAnswererPhase2EngineTests(unittest.TestCase):
    def test_optin_fallback_tx_bypasses_phase_lock_held_by_receive(self):
        engine = self.engine()

        class Fallback:
            def tx_samples(self, count):
                return [0] * count

        engine._split_fallback_tx_lock = True
        engine._publish_fallback_tx(Fallback())
        initial_tx_media = engine._tx_media
        finished = threading.Event()
        result = {}

        def render():
            result["payload"] = engine.tx_g711(160, "pcmu")
            finished.set()

        engine._lock.acquire()
        worker = threading.Thread(target=render)
        try:
            worker.start()
            self.assertTrue(finished.wait(1.0),
                            "fallback TX blocked behind the V.90 receive lock")
        finally:
            engine._lock.release()
            worker.join(2.0)
            engine.close()
        self.assertFalse(worker.is_alive())
        self.assertEqual(result["payload"], bytes([0xff]) * 160)
        self.assertEqual(engine._tx_media, initial_tx_media + 160)

    def test_fallback_retirement_waits_for_fast_tx_lease(self):
        engine = self.engine()
        tx_started = threading.Event()
        allow_tx_to_finish = threading.Event()
        retired = threading.Event()

        class Fallback:
            def tx_samples(self, count):
                tx_started.set()
                if not allow_tx_to_finish.wait(2.0):
                    raise TimeoutError("test did not release fallback TX")
                return [0] * count

        engine._split_fallback_tx_lock = True
        engine._publish_fallback_tx(Fallback())
        tx_worker = threading.Thread(
            target=lambda: engine.tx_g711(160, "pcmu"))
        retire_worker = threading.Thread(
            target=lambda: (engine._retire_fallback_tx(), retired.set()))
        try:
            tx_worker.start()
            self.assertTrue(tx_started.wait(1.0))
            retire_worker.start()
            self.assertFalse(retired.wait(0.02),
                             "retirement did not wait for in-flight TX")
            allow_tx_to_finish.set()
            tx_worker.join(2.0)
            retire_worker.join(2.0)
            self.assertFalse(tx_worker.is_alive())
            self.assertFalse(retire_worker.is_alive())
            self.assertTrue(retired.is_set())
            self.assertIsNone(engine._fallback_tx_engine)
        finally:
            allow_tx_to_finish.set()
            tx_worker.join(2.0)
            if retire_worker.ident is not None:
                retire_worker.join(2.0)
            engine.close()

    def test_info1a_deadline_uses_rendered_info1d_completion(self):
        engine = V90DigitalAnswererPhase2Engine(
            role="digital-answerer",
            negotiation={"pcm_negotiated_bits": V8_PCM_DIGITAL},
            media_at=0, media_origin=1000.0, law="pcmu",
            info0d_fields={"rate3429": 1}, info1d_fields={},
            tone_b_peak=900, probe_rms=450,
            probe_builder=lambda duration, rms: np.full(round(duration * 8000), rms),
            phase2_info1a_timeout_retrain=True)
        try:
            engine._phase2.state = engine._phase2.WAIT_INFO1A
            start = engine._renderer._last_finite_end + 800
            engine._submit_phase2_actions([Phase2Action(start, "info1d")])
            complete = engine._renderer._completion["info1d-complete"]
            self.assertGreater(complete, start)
            self.assertEqual(engine._phase2._info1a_deadline_at,
                             complete + int(.700 * 8000)
                             + engine._phase2.rtd_samples)
        finally:
            engine.close()

    def test_clock_bridge_waits_for_configured_fit_span(self):
        clock = RtpMediaClock(
            origin=200.0, media_at=24000, max_anchors=128,
            min_fit_span_samples=12800)
        for index in range(81):
            rx_at = index * 160
            media_at = 24000 + rx_at * 1.0001
            deadline = 200.0 + (media_at - 24000) / 8000
            clock.observe(rx_at, {
                "packet_playout_deadline_monotonic": deadline})
            if rx_at < 12800:
                self.assertIsNone(clock.last_fit)
        self.assertEqual(clock.last_fit["sample_span"], 12800.0)
        self.assertAlmostEqual(clock.slope, 1.0001, places=7)
        self.assertEqual(clock.last_fit["anchor_count"], 81)

    def test_clock_bridge_ignores_short_jitter_then_tracks_rate_offset(self):
        clock = RtpMediaClock(origin=200.0, media_at=24000)
        for index in range(8):
            rx_at = index * 160
            jitter = (-1) ** index * 0.004
            deadline = 200.0 + rx_at / 8000 + jitter
            clock.observe(rx_at, {"packet_playout_deadline_monotonic": deadline})
        self.assertEqual(clock.slope, 1.0)

        clock = RtpMediaClock(origin=200.0, media_at=24000)
        for index in range(64):
            rx_at = index * 160
            media_at = 24000 + rx_at * 1.0001
            deadline = 200.0 + (media_at - 24000) / 8000
            clock.observe(rx_at, {"packet_playout_deadline_monotonic": deadline})
        self.assertAlmostEqual(clock.slope, 1.0001, places=7)
        self.assertAlmostEqual(clock.rx_to_media(4000), 28000.4, delta=1)

    def test_clock_bridge_ignores_burst_delay_step_without_losing_rate(self):
        clock = RtpMediaClock(origin=200.0, media_at=24000)
        for index in range(128):
            rx_at = index * 160
            media_at = 24000 + rx_at * 1.0001
            # A bounded arrival/playout delay step must not be mistaken for
            # a >2% RTP sample-clock-rate change.
            burst_delay = 0.040 if 36 <= index < 82 else 0.0
            deadline = 200.0 + (media_at - 24000) / 8000 + burst_delay
            clock.observe(rx_at, {
                "packet_playout_deadline_monotonic": deadline})
        self.assertAlmostEqual(clock.slope, 1.0001, places=4)
        self.assertAlmostEqual(clock.rx_to_media(20000), 44002.0, delta=2)

    def test_clock_bridge_restarts_fit_after_persistent_arrival_delay_step(self):
        clock = RtpMediaClock(origin=200.0, media_at=24000)
        for index in range(192):
            rx_at = index * 160
            media_at = 24000 + rx_at * 1.0001
            # Model a persistent 40 ms arrival-delay change, plus periodic
            # one-packet bursts. The step changes clock offset, not rate.
            delay_step = 0.040 if index >= 64 else 0.0
            burst = 0.015 if index % 10 == 9 else 0.0
            deadline = (200.0 + (media_at - 24000) / 8000
                        + delay_step + burst)
            clock.observe(rx_at, {
                "packet_playout_deadline_monotonic": deadline})
        self.assertAlmostEqual(clock.slope, 1.0001, places=4)
        self.assertAlmostEqual(clock.rx_to_media(30000), 54323.0, delta=3)

    def test_clock_opt_in_detects_persistent_5p6ms_playout_step(self):
        clocks = {
            threshold: RtpMediaClock(
                origin=200.0, media_at=0, max_rate_error_ppm=2000,
                offset_step_samples=threshold)
            for threshold in (80, 40)
        }
        for index in range(64):
            rx_at = index * 160
            # A 45-sample persistent playout-delay step is not clock drift.
            offset = 0 if index < 8 else 45
            deadline = 200.0 + (rx_at + offset) / 8000
            for clock in clocks.values():
                clock.observe(rx_at, {
                    "packet_playout_deadline_monotonic": deadline})

        self.assertGreater(abs(clocks[80].last_fit["error_ppm"]), 5000)
        self.assertAlmostEqual(clocks[40].slope, 1.0, places=5)
        self.assertEqual(
            clocks[40].last_fit["persistent_offset_step_samples"], 40)

    def test_clock_offset_step_candidate_replays_retained_failed_fit(self):
        fixture_path = (Path(__file__).parent / "testdata" /
                        "v90_phase2_clock_offset_step_20261106.json")
        fixture = json.loads(fixture_path.read_text())
        anchors = fixture["anchors"]
        origin = 200.0
        media_at = anchors[0][1]
        results = {}
        for threshold in (80, 40):
            clock = RtpMediaClock(
                origin=origin, media_at=media_at, max_anchors=64,
                max_rate_error_ppm=2000,
                offset_step_samples=threshold)
            failure = None
            for rx_sample, media_sample in anchors:
                deadline = origin + (media_sample - media_at) / 8000.0
                try:
                    clock.observe(int(rx_sample), {
                        "packet_playout_deadline_monotonic": deadline})
                except ValueError as exc:
                    failure = str(exc)
                    break
            results[threshold] = (clock, failure)

        baseline, baseline_failure = results[80]
        self.assertIsNotNone(baseline_failure)
        self.assertIn("fit=1.020240", baseline_failure)
        self.assertAlmostEqual(
            baseline.last_fit["error_ppm"], fixture["saved_fit_error_ppm"],
            delta=0.5)
        # The fixture stores the already-admitted fit anchors. Feeding them
        # back through observe() applies the live jitter-admission filter a
        # second time, so the replay can retain fewer rows while reproducing
        # the same fitted slope and rejection.
        self.assertLess(baseline.last_fit["anchor_count"],
                        fixture["saved_anchor_count"])

        candidate, candidate_failure = results[40]
        self.assertIsNone(candidate_failure)
        self.assertAlmostEqual(candidate.last_fit["slope"], 1.0, delta=0.001)
        self.assertLess(abs(candidate.last_fit["error_ppm"]), 1000)
        self.assertEqual(candidate.last_fit["persistent_offset_step_samples"], 40)

    def test_clock_bridge_discards_two_stale_v8_preroll_anchors_after_handoff_step(self):
        clock = RtpMediaClock(origin=200.0, media_at=24000)
        for rx_at in (0, 160):
            clock.observe(rx_at, {
                "packet_playout_deadline_monotonic": 200.0 + rx_at / 8000,
            })
        # The selected V.90 engine's next packet is on a new media offset,
        # 40 ms later. Three consecutive observations establish a step, while
        # the two leading pre-roll samples must not poison the rate fit.
        for rx_at in (800, 960, 1120):
            clock.observe(rx_at, {
                "packet_playout_deadline_monotonic": (
                    200.0 + (rx_at + 320) / 8000),
            })
        self.assertEqual(len(clock.anchors), 1)
        for rx_at in range(1280, 12000, 160):
            clock.observe(rx_at, {
                "packet_playout_deadline_monotonic": (
                    200.0 + (rx_at + 320) / 8000),
            })
        self.assertIsNotNone(clock.last_fit)
        self.assertAlmostEqual(clock.last_fit["slope"], 1.0, places=5)
        self.assertGreaterEqual(clock.anchors[0][0], 1120)

    def test_clock_bridge_rejects_sustained_out_of_envelope_rate(self):
        clock = RtpMediaClock(origin=200.0, media_at=24000)
        with self.assertRaisesRegex(ValueError, "exceeds configured V.90 timing"):
            for index in range(128):
                rx_at = index * 160
                media_at = 24000 + rx_at * 1.03
                deadline = 200.0 + (media_at - 24000) / 8000
                clock.observe(rx_at, {
                    "packet_playout_deadline_monotonic": deadline})

    def test_rtp_engine_pumps_missing_tone_a_retrain_deadline(self):
        engine = self.engine(media_at=0, media_origin=1000.0)
        engine._phase2.state = engine._phase2.TX_PROBE
        engine._phase2.l2_started_at = 0
        engine._phase2._tone_a_deadline_at = 5200
        engine.feed_g711(bytes([0xff]) * 160, "pcmu", {
            "packet_playout_deadline_monotonic": 1000.25,
            "payload_samples": 160,
        })
        self.assertEqual(engine.phase2_state, engine._phase2.FAILED)
        self.assertIn(("phase2-state", engine._phase2.FAILED), engine.events)
        self.assertTrue(engine._failed)
        failures = [value for name, value in engine.events
                    if name == "phase2-failed"]
        self.assertEqual(failures, [{
            "reason": "V.90 digital Phase 2 failed or timed out",
        }])
        engine.close()

    def test_phase1_silence_is_emitted_on_the_dce_sample_clock(self):
        engine = self.engine(media_at=12000, media_origin=1000.0,
                             phase1_silence_samples=600)
        try:
            self.assertEqual(engine.tx_g711(600, "pcmu"), bytes([0xff]) * 600)
            signal = engine.tx_g711(160, "pcmu")
            self.assertEqual(len(signal), 160)
            self.assertNotEqual(signal, bytes([0xff]) * 160)
            self.assertEqual(engine._renderer.cursor, 12760)
        finally:
            engine.close()

    @staticmethod
    def engine(law="pcmu", *, media_at=16000, media_origin=1000.0,
               phase3_factory=None, phase3_rate_mask=None,
               info1d_fields=None, phase3_enable_lapm=False,
               phase1_silence_samples=0, protocols=0):
        return V90DigitalAnswererPhase2Engine(
            role="digital-answerer",
            negotiation={"pcm_negotiated_bits": V8_PCM_DIGITAL,
                         "protocols": protocols},
            media_at=media_at, media_origin=media_origin, law=law,
            info0d_fields={"rate3429": 1}, info1d_fields=info1d_fields or {},
            tone_b_peak=900, probe_rms=450,
            probe_builder=lambda duration, rms: np.full(round(duration * 8000), rms),
            phase3_factory=phase3_factory,
            phase3_rate_mask=phase3_rate_mask,
            phase3_enable_lapm=phase3_enable_lapm,
            phase1_silence_samples=phase1_silence_samples,
        )

    def test_fallback_preserves_lapm_establishment_history(self):
        engine = self.engine()

        class Fallback:
            _v42_ever_established = True
            v42_mode = "lost"

        try:
            engine._v34_fallback_engine = Fallback()
            self.assertEqual(engine.v42_mode, "lost")
            self.assertTrue(engine.v42_ever_established)
        finally:
            engine.close()

    def test_default_phase3_factory_propagates_opt_in_lapm(self):
        engine = self.engine(phase3_rate_mask=0xFFFF,
                             phase3_enable_lapm=True,
                             protocols=V8_PROTOCOL_LAPM_V42)
        session = engine._default_phase3_factory(
            info1a={"v90_uinfo": 90, "v90_analog_symbol_rate": 5},
            law="pcmu", handoff_at=0, media_origin=1000.0,
            media_clock=engine._clock)
        self.assertTrue(session.enable_lapm)
        self.assertTrue(engine.supports_v42)
        self.assertEqual(engine.v42_mode, "off")
        engine.close()

    def test_v90_profile_does_not_start_lapm_without_v8_agreement(self):
        engine = self.engine(phase3_rate_mask=0xFFFF,
                             phase3_enable_lapm=True,
                             protocols=0)
        try:
            self.assertTrue(engine._phase3_lapm_configured)
            self.assertFalse(engine._v8_lapm_negotiated)
            self.assertFalse(engine._phase3_enable_lapm)
            self.assertFalse(engine.supports_v42)

            # Exercise the actual V.90 Phase-3 factory boundary, not only the
            # pre-handoff capability flags. A local profile may enable LAPM,
            # but without mutual V.8 LAPM negotiation the selected session
            # must not install a LAPM link.
            engine._phase2.info0a = {"clock": 0}  # no peer digital PCM capability
            engine._phase2.info1a = {
                "v90_uinfo": 67,
                "v90_analog_symbol_rate": 5,
                "v90_digital_symbol_rate": 6,
            }
            handoff = type("Handoff", (), {"at": 0})()
            engine._start_phase3(handoff)
            self.assertIsNotNone(engine._phase3)
            self.assertFalse(engine._phase3.enable_lapm)
            self.assertIsNone(engine._phase3.lapm)
        finally:
            engine.close()



    def test_default_factory_uses_its_info1d_probe_offset_for_upstream(self):
        blocks = [{} for _ in range(6)]
        blocks[5] = {"hi_carrier": 1}
        engine = self.engine(phase3_rate_mask=0xFFFF,
                             info1d_fields={"blocks": blocks, "freq_offset": 5})
        engine._phase2.rtd_samples = 960
        session = engine._default_phase3_factory(
            info1a={"v90_uinfo": 90, "v90_analog_symbol_rate": 5},
            law="pcmu", handoff_at=0, media_origin=1000.0,
            media_clock=engine._clock)
        self.assertEqual(session.upstream_probe_frequency_offset_units, 5)
        self.assertEqual(session.transmitter.round_trip_delay_samples, 960)
        self.assertAlmostEqual(session.upstream.receive_clock_ratio,
                               0.1 / 1050.0 + 1.0)
        engine.close()

    def test_measured_l2_offset_replaces_static_info1d_value(self):
        engine = self.engine(info1d_fields={"freq_offset": 0})
        engine._phase2.info0a = {"rate2743": 1, "rate2800": 1,
                                 "rate3429": 1, "disallow3429": 1,
                                 "pts1664": 1}
        engine._adapter.signals.probe_frequency_offset_units = -5
        engine._apply_measured_info1d_offset([Phase2Action(0, "info1d")])
        self.assertEqual(engine._signals.info1d_fields["freq_offset"], -5)
        engine._adapter.signals.probe_frequency_offset_units = -512
        engine._apply_measured_info1d_offset([Phase2Action(0, "info1d")])
        self.assertEqual(engine._signals.info1d_fields["freq_offset"], -512)
        engine.close()

    def test_info1d_is_constrained_by_received_info0a_before_transmit(self):
        blocks = [{"max_rate": 10, "preemph": 2} for _ in range(6)]
        blocks[3]["hi_carrier"] = 1
        blocks[4]["hi_carrier"] = 0
        engine = self.engine(info1d_fields={"blocks": blocks})
        engine._phase2.info0a = {
            "rate2743": 0, "rate2800": 1, "rate3429": 1,
            "lo3000": 1, "hi3000": 0,
            "lo3200": 0, "hi3200": 1,
            "disallow3429": 1, "pts1664": 0,
        }
        engine._adapter.signals.probe_quality = {"snr_db": 20.0}
        engine._apply_measured_info1d_offset([Phase2Action(0, "info1d")])
        selected = engine._signals.info1d_fields["blocks"]
        self.assertEqual(selected[1]["max_rate"], 0)
        self.assertEqual((selected[3]["hi_carrier"], selected[3]["max_rate"]),
                         (0, 5))
        self.assertEqual((selected[4]["hi_carrier"], selected[4]["max_rate"]),
                         (1, 5))
        self.assertEqual(selected[5]["max_rate"], 5)
        self.assertEqual(engine._phase3_info1d["blocks"], selected)
        self.assertIn(("phase2-probe-quality", {"snr_db": 20.0}),
                      engine.events)
        engine.close()

    def test_explicit_profile_can_use_configured_info1d_caps_without_probe_projection(self):
        blocks = [{"max_rate": 12} for _ in range(6)]
        engine = V90DigitalAnswererPhase2Engine(
            role="digital-answerer",
            negotiation={"pcm_negotiated_bits": V8_PCM_DIGITAL},
            media_at=0, media_origin=1000.0, law="pcmu",
            info0d_fields={"rate3429": 1},
            info1d_fields={"blocks": blocks},
            tone_b_peak=900, probe_rms=450,
            probe_builder=lambda duration, rms: np.full(round(duration * 8000), rms),
            phase2_project_info1d_rates_from_probe=False)
        try:
            engine._phase2.info0a = {"rate3429": 1, "allow3429": 1}
            engine._adapter.signals.probe_quality = {
                "snr_db": 20.0,
                "tone_snr_db_by_hz": {1500: -30.0, 1650: -30.0},
            }
            engine._apply_measured_info1d_offset([Phase2Action(0, "info1d")])
            rates = [b["max_rate"] for b in engine._signals.info1d_fields["blocks"]]
            # INFO0a still suppresses unsupported intermediate baud rates.
            self.assertEqual(rates, [12, 0, 0, 0, 0, 12])
            self.assertIn(("phase2-info1d-projection", {
                "enabled": False, "probe_snr_db": 20.0,
                "rates": [12, 0, 0, 0, 0, 12],
            }), engine.events)
        finally:
            engine.close()

    def test_v90_handoff_builds_phase3_and_switches_to_exact_g711(self):
        captured = {}

        class FakePhase3:
            transmit_ready = True
            state = "initial-signals"

            def feed_pcm(self, samples, *, sample_at):
                captured["pcm"] = np.asarray(samples).copy()
                captured["sample_at"] = sample_at

            def read_g711(self, count, law):
                captured["law"] = law
                return bytes([0x12 if law == "pcmu" else 0x34]) * count

        def factory(**kwargs):
            captured["factory"] = kwargs
            return FakePhase3()

        engine = self.engine("pcmu", phase3_factory=factory)
        engine._phase2.info1a = {
            "v90_uinfo": 90,
            "v90_analog_symbol_rate": 5,
            "v90_digital_symbol_rate": 6,
            "v90_freq_offset": -3,
        }
        from v90_digital_phase2_tx import RenderedHandoff
        engine._start_phase3(RenderedHandoff(24000, "handoff-v90-phase3", 6))
        self.assertEqual(captured["factory"]["info1a"]["v90_uinfo"], 90)
        self.assertEqual(captured["factory"]["law"], "pcmu")
        self.assertEqual(engine.tx_g711(160, "pcmu"), bytes([0x12]) * 160)

        engine._phase2.state = engine._phase2.COMPLETE
        payload = bytes([0xff]) * 160
        engine.feed_g711(payload, "pcmu", {
            "packet_playout_deadline_monotonic": engine._clock.origin,
            "payload_samples": len(payload),
        })
        self.assertEqual(captured["sample_at"], 0)
        self.assertEqual(captured["law"], "pcmu")
        self.assertEqual(len(captured["pcm"]), 160)
        engine.close()

    def test_v34_fallback_gap_uses_pcm_wrapper_and_advances_receive_clock(self):
        captured = {}

        class Fallback:
            def note_concealed_rx(self, count):
                captured["concealed"] = count

            def set_rx_media_deadline(self, deadline):
                captured["deadline"] = deadline

            def feed_pcm(self, pcm):
                captured["pcm"] = pcm
                return b"decoded"

        engine = self.engine("pcmu")
        try:
            engine._v34_fallback_engine = Fallback()
            result = engine.feed_gap(160, "pcmu", {
                "packet_playout_deadline_monotonic": engine._clock.origin,
            })
            self.assertEqual(result, b"decoded")
            self.assertEqual(captured["concealed"], 160)
            self.assertEqual(captured["deadline"], engine._clock.origin)
            self.assertEqual(len(captured["pcm"]), 320)
            self.assertEqual(engine._rx_at, 160)
        finally:
            engine.close()

    def test_v34_fallback_preroll_batch_preserves_pcm_and_media_origin(self):
        captured = {"feeds": [], "deadlines": []}

        class Fallback:
            def set_rx_media_deadline(self, deadline):
                captured["deadlines"].append(deadline)

            def feed_pcm(self, pcm):
                captured["feeds"].append(bytes(pcm))

        engine = self.engine()
        try:
            engine._v34_fallback_engine = Fallback()
            engine._rx_at = 100
            engine._remember_phase2_pcm(np.arange(160, dtype=float))
            engine._rx_at = 260
            engine._remember_phase2_pcm(np.arange(160, dtype=float) + 160)
            engine._rx_at = 420
            engine._adapter.info1a_complete_rx_sample = 180

            with patch.dict("os.environ", {"V90_FALLBACK_PREROLL_BATCH": "1"}):
                engine._replay_v34_fallback_pcm_from_info1a()

            self.assertEqual(len(captured["feeds"]), 1)
            decoded = np.frombuffer(captured["feeds"][0], dtype="<i2")
            np.testing.assert_array_equal(decoded, np.arange(80, 320, dtype="<i2"))
            self.assertEqual(captured["deadlines"], [
                engine._clock.origin + 180 / 8000,
            ])
            details = next(value for name, value in engine.events
                           if name == "phase3-v34-fallback-preroll-replayed")
            self.assertEqual(details["start_rx_sample"], 180)
            self.assertEqual(details["end_rx_sample"], 420)
            self.assertEqual(details["samples"], 240)
            self.assertEqual(details["chunks"], 2)
            self.assertTrue(details["batched"])
            self.assertGreaterEqual(details["wall_ms"], 0.0)
            self.assertGreaterEqual(details["thread_cpu_ms"], 0.0)
        finally:
            engine.close()

    def test_phase3_handoff_replays_bounded_pcm_from_info1a_boundary(self):
        captured = {"chunks": []}

        class FakePhase3:
            transmit_ready = False
            state = "acquiring"

            def feed_pcm(self, samples, *, sample_at):
                captured["chunks"].append((sample_at, np.asarray(samples).copy()))

            def read_g711(self, count, law):
                return bytes([0xff if law == "pcmu" else 0xd5]) * count

        engine = self.engine(phase3_factory=lambda **_kwargs: FakePhase3())
        engine._phase2.info1a = {
            "v90_uinfo": 90,
            "v90_analog_symbol_rate": 5,
            "v90_digital_symbol_rate": 6,
        }
        engine._rx_at = 100
        engine._remember_phase2_pcm(np.arange(160, dtype=float))
        engine._rx_at = 260
        engine._remember_phase2_pcm(np.arange(160, dtype=float) + 160)
        engine._rx_at = 420
        engine._adapter.info1a_complete_rx_sample = 180

        from v90_digital_phase2_tx import RenderedHandoff
        engine._start_phase3(RenderedHandoff(
            24000, "handoff-v90-phase3", 6))

        self.assertEqual([start for start, _ in captured["chunks"]], [180, 260])
        np.testing.assert_array_equal(captured["chunks"][0][1], np.arange(80, 160))
        np.testing.assert_array_equal(captured["chunks"][1][1], np.arange(160, 320))
        self.assertIn(("phase3-preroll-replayed", {
            "start_rx_sample": 180, "end_rx_sample": 420, "samples": 240,
        }), engine.events)
        engine.close()

    def test_phase3_factory_contract_is_checked_before_install(self):
        engine = self.engine(phase3_factory=lambda **kwargs: object())
        engine._phase2.info1a = {
            "v90_uinfo": 90,
            "v90_analog_symbol_rate": 5,
            "v90_digital_symbol_rate": 6,
        }
        from v90_digital_phase2_tx import RenderedHandoff
        with self.assertRaisesRegex(TypeError, "feed_pcm, read_g711"):
            engine._start_phase3(
                RenderedHandoff(24000, "handoff-v90-phase3", 6))
        self.assertIsNone(engine._phase3)
        engine.close()

    def test_bad_phase3_transmit_fails_closed_to_law_silence(self):
        class BadPhase3:
            transmit_ready = True
            state = "initial-signals"

            def feed_pcm(self, samples, *, sample_at):
                pass

            def read_g711(self, count, law):
                return b"too long"

        engine = self.engine(phase3_factory=lambda **kwargs: BadPhase3())
        engine._phase2.info1a = {
            "v90_uinfo": 90,
            "v90_analog_symbol_rate": 5,
            "v90_digital_symbol_rate": 6,
        }
        from v90_digital_phase2_tx import RenderedHandoff
        engine._start_phase3(RenderedHandoff(24000, "handoff-v90-phase3", 6))
        self.assertEqual(engine.tx_g711(4, "pcmu"), b"\xff" * 4)
        self.assertTrue(engine._failed)
        self.assertEqual(engine.events[-1][0], "phase3-failed")
        engine.close()

    def test_phase3_s_timeout_starts_bounded_retrain_with_specific_reason(self):
        engine = V90DigitalAnswererPhase2Engine(
            role="digital-answerer",
            negotiation={"pcm_negotiated_bits": V8_PCM_DIGITAL},
            media_at=16000, media_origin=1000.0, law="pcmu",
            info0d_fields={"rate3429": 1}, info1d_fields={},
            tone_b_peak=900, probe_rms=450,
            probe_builder=lambda duration, rms: np.full(round(duration * 8000), rms),
            retrain_on_failure=True)

        class TimedOutPhase3:
            state = "failed"
            failure_reason = "V.90 §9.3.1.5: no analogue S within 5100 ms plus RTD of TRN1d"
            upstream_data = None
            upstream = None

        engine._phase3 = TimedOutPhase3()
        engine.tx_g711(160, "pcmu")

        starts = [data for name, data in engine.events if name == "retrain-start"]
        self.assertEqual(len(starts), 1)
        self.assertTrue(starts[0]["initiating"])
        self.assertEqual(starts[0]["reason"], TimedOutPhase3.failure_reason)
        self.assertEqual(engine.retrain_count, 1)
        engine.close()

    def test_rate_mask_rejects_non_integer_configuration(self):
        with self.assertRaisesRegex(ValueError, "must be an integer"):
            self.engine(phase3_rate_mask=1.5)
        with self.assertRaisesRegex(ValueError, "must be an integer"):
            self.engine(phase3_rate_mask=True)

    def test_configured_rate_mask_builds_profile_from_v90_info1a(self):
        for law in ("pcmu", "pcma"):
            with self.subTest(law=law):
                blocks = [{} for _ in range(6)]
                blocks[5] = {"hi_carrier": 1}
                engine = self.engine(
                    law, phase3_rate_mask=0xFFFF,
                    info1d_fields={"blocks": blocks})
                engine._phase2.info1a = {
                    "v90_uinfo": 90,
                    "v90_analog_symbol_rate": 5,
                    "v90_digital_symbol_rate": 6,
                }
                from v90_digital_phase2_tx import RenderedHandoff
                engine._start_phase3(
                    RenderedHandoff(24000, "handoff-v90-phase3", 6))
                self.assertEqual(engine._phase3.baud, 2400 * 10 / 7)
                self.assertEqual(engine._phase3.carrier, (2400 * 10 / 7) * 4 / 7)
                self.assertEqual(engine._phase3.law, law)
                self.assertEqual(engine.phase3_state, engine._phase3.upstream.ACQUIRING)
                engine.close()

    def test_requires_digital_v8_role_and_matching_g711_law(self):
        with self.assertRaisesRegex(ValueError, "did not negotiate"):
            V90DigitalAnswererPhase2Engine(
                role="digital-answerer", negotiation={"pcm_negotiated_bits": 0},
                media_at=0, media_origin=0.0, law="pcmu", info0d_fields={},
                info1d_fields={}, tone_b_peak=1, probe_rms=1)
        engine = self.engine()
        with self.assertRaisesRegex(ValueError, "codec changed"):
            engine.tx_g711(160, "pcma")
        engine.close()

    def test_engine_factory_is_selected_by_v8_and_exposes_raw_g711_surface(self):
        created = []

        def factory(**kwargs):
            engine = V90DigitalAnswererPhase2Engine(
                **kwargs, law="pcmu", info0d_fields={"rate3429": 1},
                info1d_fields={}, tone_b_peak=900, probe_rms=450,
                probe_builder=lambda duration, rms:
                    np.full(round(duration * 8000), rms))
            created.append(engine)
            return engine

        chain = V8Chain(False, allow=V8_MOD_V34,
                        pcm_availability=V8_PCM_DIGITAL,
                        v90_engine_factory=factory)
        chain._v8.status = 2
        chain._v8.result = {
            "modulations": V8_MOD_V34,
            "protocols": 0,
            "pcm_negotiated": True,
            "pcm_negotiated_bits": V8_PCM_DIGITAL,
        }
        try:
            chain._step_v8()
            self.assertEqual(chain.kind, "v90")
            self.assertTrue(callable(getattr(created[0], "feed_g711", None)))
            self.assertEqual(len(chain.tx_g711(160, "pcmu")), 160)
            self.assertEqual(chain.feed_g711(bytes([0xff]), "pcmu", {
                "packet_playout_deadline_monotonic":
                    created[0]._clock.origin,
                "payload_samples": 1,
            }), b"")
        finally:
            chain.close()

    def test_transmit_is_exact_selected_law_and_renders_info0d(self):
        for law, silence in (("pcmu", 0xff), ("pcma", 0xd5)):
            with self.subTest(law=law):
                engine = self.engine(law)
                reference = self.engine(law)
                payload = engine.tx_g711(160, law)
                self.assertEqual(len(payload), 160)
                self.assertTrue(any(value != silence for value in payload))
                self.assertEqual(payload, reference.tx_g711(160, law))
                engine.close()
                reference.close()

    def test_rendered_info0d_survives_both_rtp_codecs(self):
        for law, decoder, expected_alaw in (
                ("pcmu", ulaw_to_linear, 0),
                ("pcma", alaw_to_linear, 1)):
            with self.subTest(law=law):
                engine = self.engine(law)
                try:
                    codewords = engine.tx_g711(1200, law)
                    pcm = np.fromiter((decoder(word) for word in codewords),
                                      dtype=float, count=len(codewords))
                    decoded = v90.decode_info0d(pcm, sender_role="call", phases=16)
                    frame = next(row for row in decoded
                                 if "_start_sample" in row and "ack_info0a" in row)
                    self.assertEqual(frame["ack_info0a"], 0)
                    self.assertEqual(frame["alaw"], expected_alaw)
                finally:
                    engine.close()

    def test_exact_g711_receive_decodes_info0a_on_common_sample_clock(self):
        engine = self.engine()
        info0a = v34_info.InfoModulator("answer", amplitude=3000, guard=True).modulate(
            v34_info.build_info0(rate3429=1, ack=0), tail_bits=2)
        samples = np.rint(np.concatenate((np.zeros(800), info0a, np.zeros(1600)))) \
            .astype(np.int16)
        payload = bytes(linear_to_ulaw(int(sample)) for sample in samples)
        offset = 0
        try:
            for start in range(0, len(payload), 160):
                chunk = payload[start:start + 160]
                context = {
                    "packet_playout_deadline_monotonic":
                        engine._clock.origin + offset / 8000,
                    "payload_samples": len(chunk),
                    "timestamp": offset,
                }
                self.assertEqual(engine.feed_g711(chunk, "pcmu", context), b"")
                offset += len(chunk)
            self.assertIsNotNone(engine._phase2.info0a)
            self.assertEqual(engine._phase2.info0a["ack"], 0)
            self.assertEqual(engine._clock.rx_to_media(320), engine._media_at + 320)
        finally:
            engine.close()

    def test_engine_restarts_tone_b_after_duplicate_info0a_ack(self):
        """Exercise duplicate INFO0 recovery through the RTP-facing engine."""
        engine = V90DigitalAnswererPhase2Engine(
            role="digital-answerer",
            negotiation={"pcm_negotiated_bits": V8_PCM_DIGITAL},
            media_at=0, media_origin=time.monotonic(), law="pcmu",
            info0d_fields={"rate3429": 1}, info1d_fields={},
            tone_b_peak=900, probe_rms=450,
            probe_builder=lambda duration, rms: np.full(
                round(duration * 8000), rms))
        engine._adapter.trace_events = True
        mod = v34_info.InfoModulator("answer", amplitude=3000, guard=True)
        info0a = mod.modulate(v34_info.build_info0(rate3429=1, ack=0),
                              tail_bits=2)
        received_pcm = np.concatenate((np.zeros(800), info0a, np.zeros(4000),
                                       info0a, np.zeros(5000)))
        received = bytes(linear_to_ulaw(int(sample)) for sample in
                         np.rint(received_pcm).astype(np.int32))
        rendered = []
        ack_starts = []
        offset = 0
        try:
            while offset < len(received):
                chunk = received[offset:offset + 160]
                prior_events = len(engine._adapter.event_trace)
                context = {
                    "packet_playout_deadline_monotonic":
                        engine._clock.origin + offset / 8000,
                    "payload_samples": len(chunk),
                    "timestamp": offset,
                }
                self.assertEqual(engine.feed_g711(chunk, "pcmu", context), b"")
                new_events = engine._adapter.event_trace[prior_events:]
                if any("info0d" in row["actions"] for row in new_events):
                    ack_starts.append(engine._renderer.cursor)
                rendered.append(engine.tx_g711(len(chunk), "pcmu"))
                offset += len(chunk)

            self.assertEqual(len(ack_starts), 1, engine._adapter.event_trace)
            ack_at = ack_starts[0]
            expected = V90DigitalPhase2Signals(
                info0d={"rate3429": 1}, amplitude=3000)
            expected.set_phase_at_sample(ack_at)
            ack_pcm = expected.modulate_info0d(ack_info0a=1)
            ack_codewords = bytes(linear_to_ulaw(int(round(sample)))
                                  for sample in ack_pcm)
            tx_stream = b"".join(rendered)
            self.assertEqual(tx_stream[ack_at:ack_at + len(ack_codewords)],
                             ack_codewords)
            tone_at = ack_at + len(ack_codewords)
            t = np.arange(tone_at, tone_at + 80)
            tone_codewords = bytes(
                linear_to_ulaw(int(round(900 * np.cos(2 * np.pi * 1200 * n / 8000))))
                for n in t)
            self.assertEqual(tx_stream[tone_at:tone_at + 80], tone_codewords)
        finally:
            engine.close()

    def test_v34_selection_handoff_starts_seeded_v34_call_side(self):
        engine = self.engine()
        try:
            engine._phase2.info1a = {
                "mode": 5, "sym_call": 5, "sym_ans": 5,
                "max_rate": 12, "hi_carrier": 0, "preemph": 0,
                "power_min": 0, "power_extra": 0,
            }
            engine._start_v34_fallback()
            fallback = engine._v34_fallback_engine
            self.assertIsNotNone(fallback)
            self.assertEqual(fallback.caller.stage, "phase3")
            self.assertAlmostEqual(fallback.caller.call_baud, 3428.57142857)
            self.assertFalse(engine.connected)
            self.assertEqual(engine.phase3_state, "phase3")
            self.assertTrue(any(name == "phase3-v34-fallback-started"
                                for name, _ in engine.events))
        finally:
            engine.close()

    def test_phase2_fails_closed_on_rtp_gap(self):
        engine = self.engine()
        try:
            self.assertEqual(engine.feed_gap(80, "pcmu", {"sequence": 3}), b"")
            self.assertEqual(engine.phase2_state, engine._phase2.FAILED)
            silence = engine.tx_g711(160, "pcmu")
            self.assertEqual(silence, bytes([0xff]) * 160)
            self.assertIn("RTP sample gap during Phase 2", str(engine.events))
        finally:
            engine.close()

    def test_phase3_rtp_gap_preserves_session_and_advances_sample_clock(self):
        class Phase3:
            transmit_ready = True
            state = "data"

            def __init__(self):
                self.gaps = []

            def feed_pcm(self, samples, *, sample_at):
                self.gaps.append((np.asarray(samples).copy(), sample_at))

            def read_g711(self, count, law):
                return bytes([0xff if law == "pcmu" else 0xd5]) * count

            def read_received_data(self):
                return b""

        phase3 = Phase3()
        engine = self.engine(phase3_factory=lambda **_kwargs: phase3)
        engine._phase2.state = engine._phase2.COMPLETE
        engine._phase3 = phase3
        try:
            origin = engine._clock.origin
            self.assertEqual(engine.feed_gap(160, "pcmu", {
                "sequence": 3, "timestamp": 160,
                "packet_playout_deadline_monotonic": origin + 0.020,
            }), b"")
            self.assertFalse(engine._failed)
            self.assertEqual(engine.phase3_state, "data")
            self.assertEqual(engine._rx_at, 160)
            self.assertEqual(len(phase3.gaps), 1)
            missing, sample_at = phase3.gaps[0]
            self.assertEqual(sample_at, 0)
            self.assertEqual(missing.shape, (160,))
            self.assertTrue(np.all(missing == ulaw_to_linear(0xff)))
            self.assertIn("phase3-rx-gap", str(engine.events))
            self.assertEqual(engine.tx_g711(4, "pcmu"), b"\xff" * 4)
        finally:
            engine.close()


if __name__ == "__main__":
    unittest.main()
