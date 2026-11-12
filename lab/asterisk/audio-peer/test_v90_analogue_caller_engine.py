import os
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from v8 import V8_PCM_ANALOG, V8_PROTOCOL_LAPM_V42
from v90_analogue_caller_engine import (
    V90AnalogueCallerPhase2Engine,
    _phase_failure_detail,
    build_v90_analogue_caller_engine,
    select_v34_fallback_symbol_rate,
    select_v90_analogue_symbol_rate,
)
from v90_analogue_caller_phase2 import AnaloguePhase2Action
from v90_analogue_caller_phase3 import V90AnalogueCallerPhase3
from v90_analogue_phase2_tx import AnalogueTxHandoff
import v34_info
import v90_info_reference as v90
from v90_dce_factory import install_v90_analogue_caller


class PhaseFailureDiagnosticsTests(unittest.TestCase):
    def test_traceback_is_opt_in(self):
        try:
            raise TypeError("bad diagnostic value")
        except TypeError as exc:
            with patch.dict(os.environ, {"V90_DEBUG_TRACEBACK": "0"}):
                self.assertEqual(_phase_failure_detail(exc),
                                 "bad diagnostic value")
                with patch.dict(os.environ, {"V90_DEBUG_TRACEBACK": "1"}):
                    detail = _phase_failure_detail(exc)
        self.assertIn("bad diagnostic value", detail)
        self.assertIn("Traceback (most recent call last)", detail)


def profile():
    return {
        "info0a_fields": {"rate3429": 1, "disallow3429": 1},
        "info1a_fields": {
            "md_len": 0, "uinfo": 90, "analog_symbol_rate": 5,
            "digital_symbol_rate": 6, "freq_offset": 0,
        },
        "nominal_peak": 1200,
        "probe_rms": 300,
    }


class V90AnalogueCallerEngineTests(unittest.TestCase):
    def make_engine(self, law="pcmu", configured=None):
        engine = V90AnalogueCallerPhase2Engine(
            role="analogue-caller",
            negotiation={"pcm_negotiated_bits": V8_PCM_ANALOG},
            media_at=0, media_origin=100.0, law=law,
            profile=profile() if configured is None else configured)
        engine._phase3_jd_descriptor = {"rate_mask": (1 << 22) - 1}
        return engine

    def test_b1d_failure_walks_down_three_jd_rates_via_retrain(self):
        engine = self.make_engine()
        try:
            engine._controller.info0d = {"valid": True}
            with patch.object(engine, "_start_phase3_retrain") as retrain:
                for failed_drn, retry_cap in ((21, 20), (20, 19), (19, 18)):
                    engine._phase4_downstream_selection = {
                        "drn": failed_drn,
                        "jd_rate_mask": (1 << failed_drn) - 1}
                    self.assertTrue(engine._try_phase4_rate_stepdown(
                        "b1d_not_all_ones", "V.90 B1d frame was not all ones"))
                    self.assertEqual(engine._phase4_downstream_drn_cap,
                                     retry_cap)
                    self.assertEqual(engine._phase4_rate_fallback_attempts,
                                     21 - retry_cap)
                self.assertEqual(retrain.call_count, 3)
                self.assertIn("DRN 21", retrain.call_args_list[0].kwargs["reason"])
                self.assertTrue(all(call.kwargs["rate_fallback"]
                                    for call in retrain.call_args_list))
                engine._phase4_downstream_selection = {
                    "drn": 18, "jd_rate_mask": (1 << 18) - 1}
                self.assertFalse(engine._try_phase4_rate_stepdown(
                    "b1d_not_all_ones", "V.90 B1d frame was not all ones"))
                self.assertEqual(engine._phase4_rate_fallback_attempts, 3)
                self.assertEqual(engine.events[-1][0],
                                 "phase4-rate-walkdown-exhausted")
        finally:
            engine.close()

    def test_b1d_walkdown_stops_at_connect_time_budget(self):
        engine = self.make_engine()
        try:
            engine._phase4_downstream_selection = {
                "drn": 21, "jd_rate_mask": (1 << 21) - 1}
            engine._controller.info0d = {"valid": True}
            engine._phase4_rate_fallback_started_at = 0
            engine._tx_media = 90 * 8000
            with patch.object(engine, "_start_phase3_retrain") as retrain:
                self.assertFalse(engine._try_phase4_rate_stepdown(
                    "b1d_not_all_ones", "B1d failed after budget"))
                retrain.assert_not_called()
            self.assertEqual(engine.events[-1][0],
                             "phase4-rate-walkdown-exhausted")
            self.assertEqual(engine.events[-1][1]["reason"],
                             "connect_time_budget")
        finally:
            engine.close()

    def test_active_rate_walkdown_fails_closed_at_hard_deadline(self):
        engine = self.make_engine()
        try:
            engine._phase4_rate_fallback_started_at = 0
            engine._phase4_rate_fallback_attempts = 1
            with patch.object(engine._clock, "tx_now", return_value=90 * 8000):
                engine._pump_timer()
            self.assertTrue(engine._failed)
            self.assertEqual(engine.events[-1][0],
                             "phase4-rate-walkdown-exhausted")
            self.assertEqual(engine.events[-1][1]["reason"],
                             "connect_time_budget_expired")
        finally:
            engine.close()

    def test_unrelated_phase4_failure_does_not_step_down(self):
        engine = self.make_engine()
        try:
            engine._phase4_downstream_selection = {
                "drn": 21, "jd_rate_mask": (1 << 21) - 1}
            with patch.object(engine, "_start_phase3_retrain") as retrain:
                self.assertFalse(engine._try_phase4_rate_stepdown(
                    "phase4_control_failure", "invalid CP frame"))
                retrain.assert_not_called()
                self.assertEqual(engine._phase4_downstream_drn_cap, 22)
        finally:
            engine.close()

    def test_b1d_failure_event_enters_rate_stepdown_path(self):
        engine = self.make_engine()
        try:
            reason = "V.90 B1d frame was not all ones"
            engine._phase4_downstream_selection = {
                "drn": 19, "jd_rate_mask": (1 << 19) - 1}
            engine._controller.info0d = {"valid": True}
            engine._phase4_controller = SimpleNamespace(
                FAILED="failed", DATA="data", state="failed",
                _failure_reason=reason, _failure_code="b1d_not_all_ones",
                events=[("phase4-failed", reason)], lapm=None)
            engine._phase4_media = SimpleNamespace(
                feed_downstream=Mock(return_value=[]))
            with patch.object(engine, "_start_phase3_retrain") as retrain:
                engine._feed_phase4(b"\x00")
            self.assertEqual(engine._phase4_downstream_drn_cap, 18)
            retrain.assert_called_once()
            self.assertIn(("phase4-failed", reason), engine.events)
            self.assertEqual(engine.events[-1][0], "phase4-rate-stepdown")
            self.assertFalse(engine._failed)
        finally:
            engine.close()

    def test_phase4_ucode_zero_omission_is_opt_in_for_cpt_and_cp(self):
        for enabled, expected in ((False, True), (True, False)):
            configured = profile()
            if enabled:
                configured["phase4_omit_ucode0"] = True
            engine = V90AnalogueCallerPhase2Engine(
                role="analogue-caller",
                negotiation={"pcm_negotiated_bits": V8_PCM_ANALOG},
                media_at=0, media_origin=100.0, law="pcmu",
                profile=configured)
            try:
                engine._phase3_baud = 2400.0 * 10 / 7
                engine._phase3_carrier = 1959.1836734694
                engine._phase3_jd_descriptor = {"rate_mask": (1 << 22) - 1}
                engine._start_phase4(ready_at=0)
                for phase_profile in (
                        engine._phase4_controller.cpt_profile,
                        engine._phase4_controller.cp_profile):
                    for mask in phase_profile["constellations"]:
                        self.assertEqual(0 in mask, expected)
            finally:
                engine.close()

    def test_phase4_lapm_starts_directly_after_v8_lapm_agreement(self):
        cases = (
            ({}, None, True),
            ({"protocols": V8_PROTOCOL_LAPM_V42}, None, False),
            ({"protocols": V8_PROTOCOL_LAPM_V42}, "1", True),
        )
        for extra, force, expected in cases:
            with self.subTest(extra=extra, force=force), \
                    patch.dict(os.environ, {}, clear=False):
                os.environ.pop("V42_FORCE_DETECT_AFTER_V8", None)
                if force is not None:
                    os.environ["V42_FORCE_DETECT_AFTER_V8"] = force
                engine = V90AnalogueCallerPhase2Engine(
                    role="analogue-caller",
                    negotiation={"pcm_negotiated_bits": V8_PCM_ANALOG, **extra},
                    media_at=0, media_origin=100.0, law="pcmu",
                    profile=profile())
                try:
                    engine._phase3_baud = 2400.0 * 10 / 7
                    engine._phase3_carrier = 1959.1836734694
                    engine._phase3_jd_descriptor = {"rate_mask": (1 << 22) - 1}
                    engine._start_phase4(ready_at=0)
                    controller = engine._phase4_controller
                    self.assertIs(controller.lapm_detect, expected)
                    with patch("v90_analogue_phase4.select_analog_upstream_profile",
                               return_value=SimpleNamespace(rate=21600)), \
                            patch("v90_analogue_phase4.V90AnalogueUpstreamTransmitter"), \
                            patch("v90_analogue_phase4.upstream_power_estimate_job"), \
                            patch("v90_analogue_phase4.V90LapmLink") as link:
                        controller._accept_mp({"ack": 0, "type": 1})
                    self.assertIs(link.call_args.kwargs["detect"], expected)
                finally:
                    engine.close()

    def test_phase4_data_fault_hold_is_opt_in(self):
        for hold_ms, frames in ((None, 0), (1500, 2000)):
            configured = profile()
            if hold_ms is not None:
                configured["phase4_data_fault_hold_ms"] = hold_ms
            engine = V90AnalogueCallerPhase2Engine(
                role="analogue-caller",
                negotiation={"pcm_negotiated_bits": V8_PCM_ANALOG},
                media_at=0, media_origin=100.0, law="pcmu",
                profile=configured)
            try:
                engine._phase3_baud = 2400.0 * 10 / 7
                engine._phase3_carrier = 1959.1836734694
                engine._phase3_jd_descriptor = {"rate_mask": (1 << 22) - 1}
                engine._start_phase4(ready_at=0)
                self.assertEqual(
                    engine._phase4_controller.data_fault_hold_frames, frames)
            finally:
                engine.close()
        for bad in (True, -1, 10001, 1.5):
            configured = profile()
            configured["phase4_data_fault_hold_ms"] = bad
            with self.assertRaisesRegex(ValueError, "phase4_data_fault_hold_ms"):
                V90AnalogueCallerPhase2Engine(
                    role="analogue-caller",
                    negotiation={"pcm_negotiated_bits": V8_PCM_ANALOG},
                    media_at=0, media_origin=100.0, law="pcmu",
                    profile=configured)

    def test_phase4_cp_only_ucode_zero_omission_keeps_cpt_mask(self):
        for enabled in (False, True):
            configured = profile()
            if enabled:
                configured["phase4_cp_omit_ucode0"] = True
            engine = V90AnalogueCallerPhase2Engine(
                role="analogue-caller",
                negotiation={"pcm_negotiated_bits": V8_PCM_ANALOG},
                media_at=0, media_origin=100.0, law="pcmu",
                profile=configured)
            try:
                engine._phase3_baud = 2400.0 * 10 / 7
                engine._phase3_carrier = 1959.1836734694
                engine._phase3_jd_descriptor = {"rate_mask": (1 << 22) - 1}
                engine._start_phase4(ready_at=0)
                controller = engine._phase4_controller
                for mask in controller.cpt_profile["constellations"]:
                    self.assertIn(0, mask)
                for mask in controller.cp_profile["constellations"]:
                    self.assertEqual(0 in mask, not enabled)
            finally:
                engine.close()
        configured = profile()
        configured["phase4_cp_omit_ucode0"] = "yes"
        with self.assertRaisesRegex(ValueError, "phase4_cp_omit_ucode0"):
            V90AnalogueCallerPhase2Engine(
                role="analogue-caller",
                negotiation={"pcm_negotiated_bits": V8_PCM_ANALOG},
                media_at=0, media_origin=100.0, law="pcmu",
                profile=configured)

    def test_phase4_ucode_zero_omission_requires_boolean(self):
        configured = profile()
        configured["phase4_omit_ucode0"] = 1
        with self.assertRaisesRegex(ValueError, "must be a boolean"):
            V90AnalogueCallerPhase2Engine(
                role="analogue-caller",
                negotiation={"pcm_negotiated_bits": V8_PCM_ANALOG},
                media_at=0, media_origin=100.0, law="pcmu",
                profile=configured)

    def test_rx_bit_rate_reports_v90_downstream_not_upstream(self):
        engine = self.make_engine()
        try:
            engine._phase4_controller = SimpleNamespace(
                DATA="data", state="data",
                cp_profile={"rate": 36000.0, "drn": 7},
                upstream_profile=SimpleNamespace(rate=28800))
            # Transmit stays the V.34 upstream; receive is the V.90 downstream.
            self.assertEqual(engine.bit_rate, 28800)
            self.assertEqual(engine.rx_bit_rate, 36000)
            # Before DATA the downstream is not yet established.
            engine._phase4_controller.state = "wait-mp"
            self.assertEqual(engine.rx_bit_rate, 0)
            # A V.34 fallback is symmetric; it reports its own receive rate.
            engine._phase4_controller = None
            engine._v34_fallback_engine = SimpleNamespace(
                bit_rate=28800, rx_bit_rate=28800)
            self.assertEqual(engine.rx_bit_rate, 28800)
        finally:
            engine.close()

    def test_fallback_preserves_lapm_establishment_history(self):
        engine = self.make_engine()
        try:
            engine._v34_fallback_engine = SimpleNamespace(
                _v42_ever_established=True, v42_mode="lost")
            self.assertEqual(engine.v42_mode, "lost")
            self.assertTrue(engine.v42_ever_established)
        finally:
            engine.close()

    def test_fallback_diagnostics_include_transport_gap_feedback(self):
        engine = self.make_engine()
        expected = {"registered_gaps": 1, "tainted_symbols": 12,
                    "payload_truth_used": False}
        guard = SimpleNamespace(report=Mock(return_value=expected))
        receiver = SimpleNamespace(
            eq=SimpleNamespace(_transport_guard=guard, slips=0,
                               timing_snapshot=lambda: {}),
            idx=[], frames=[], trn_end=None, _end_est=None, b1_symbol=None)
        answerer = SimpleNamespace(
            _p4rcv=receiver, rx_qualification=None, stage="data",
            _p4_state="data", _data_rx_stage_profile=None,
            _p4_poll_profile=None, t=0, rx_t=0, mp_ans=None,
            _mp_ack=0, _mp_ack_sent=False, mp_call=None, events=[])
        fallback = SimpleNamespace(
            answerer=answerer, _v42=None, state="data", connected=True,
            bit_rate=28800, rx_bit_rate=28800, failed_reason=None,
            v42_mode="lapm", events=[])
        engine._v34_fallback_engine = fallback
        try:
            report = engine.v34_fallback_diagnostics
            self.assertEqual(report["phase4_rx"]["gap_feedback"], expected)
            guard.report.assert_called_once_with()
        finally:
            engine.close()

    def test_missing_phase3_sd_retrains_once_then_fails_closed_without_tone_b(self):
        engine = self.make_engine()
        try:
            engine._controller.info0d = {
                "rate3429": 1, "v90_3429_upstream": 1,
            }
            ja = v90.build_ja_descriptor(
                N=0, LSP=1, LTP=1, SP=[0], TP=[0], H=[0] * 8,
                REF=[0] * 8, Ucodes=[])
            descriptor = v90.parse_ja_descriptor(ja)
            phase3 = V90AnalogueCallerPhase3(
                baud=3428.5714285714, ja_bits=ja, descriptor=descriptor)
            phase3.start(0)
            engine._phase3_controller = phase3
            engine._phase3_jd_descriptor = {"rate_mask": 1}
            engine._phase4_downstream_selection = {"drn": 1}

            # Drive the real Phase-3 1500 ms deadline through the production
            # engine timer. It must start, not discard, the analogue retrain.
            with patch.object(engine._clock, "tx_now",
                              return_value=phase3._deadline):
                engine._pump_timer()
            self.assertFalse(engine._failed)
            self.assertEqual(engine._phase3_retrain_attempts, 1)
            self.assertIsNone(engine._phase3_controller)
            self.assertIsNone(engine._phase3_jd_descriptor)
            self.assertIsNone(engine._phase4_downstream_selection)
            self.assertEqual(engine._controller.state,
                             engine._controller.WAIT_TONE_B)
            started = [value for name, value in engine.events
                       if name == "phase3-retrain-started"]
            self.assertEqual(len(started), 1)
            self.assertEqual(started[0]["reason"],
                             "V.90 Phase-3 recovery timer expired")

            # With no peer response, the separate 5 s local watchdog emits a
            # bounded retrain handoff; the production failure path reports a
            # safe terminal failure instead of waiting for the call watchdog.
            phase2_deadline = engine._controller._deadline_at
            with patch.object(engine._clock, "tx_now",
                              return_value=phase2_deadline):
                engine._pump_timer()
                engine._pump_tx_events()
            self.assertTrue(engine._failed)
            self.assertTrue(any(name == "phase2-failed"
                                for name, _ in engine.events))
        finally:
            engine.close()

    def test_v90_rate_selection_obeys_info1d_and_info0d_bit_40(self):
        blocks = [{"max_rate": 0} for _ in range(6)]
        blocks[4] = {"hi_carrier": 0, "max_rate": 8}
        blocks[5] = {"hi_carrier": 0, "max_rate": 12}
        # 3429 has an INFO1d V.34 profile, but is not a V.90 upstream rate
        # unless INFO0d bit 40 advertises that capability.
        self.assertEqual(select_v90_analogue_symbol_rate(
            {"lo3200": 1}, {"v90_3429_upstream": 0},
            {"blocks": blocks}, 5), 4)
        self.assertEqual(select_v90_analogue_symbol_rate(
            {"rate3429": 1, "disallow3429": 1},
            {"v90_3429_upstream": 1}, {"blocks": blocks}, 5), 5)
        blocks[4] = {"max_rate": 0}
        self.assertIsNone(select_v90_analogue_symbol_rate(
            {"rate3429": 1, "disallow3429": 1},
            {"v90_3429_upstream": 0}, {"blocks": blocks}, 5))

    def test_v90_3000_upstream_uses_advertised_low_carrier_block(self):
        blocks = [{"max_rate": 0} for _ in range(6)]
        blocks[3] = {"hi_carrier": 0, "max_rate": 8}
        blocks[4] = {"hi_carrier": 0, "max_rate": 12}
        info0a = {"lo3000": 1, "lo3200": 1}
        info0d = {"v90_3429_upstream": 0}

        # A preference of 3000 selects that rate even when 3200 is offered;
        # an INFO1d carrier mismatch makes that rate ineligible.
        self.assertEqual(select_v90_analogue_symbol_rate(
            info0a, info0d, {"blocks": blocks}, 3), 3)
        self.assertEqual(select_v90_analogue_symbol_rate(
            info0a, info0d, {"blocks": blocks}, 4), 4)
        blocks[3]["hi_carrier"] = 1
        self.assertIsNone(select_v90_analogue_symbol_rate(
            info0a, info0d, {"blocks": blocks}, 3))







    def test_call13_profile_selects_v34_not_illegal_v90_3429_request(self):
        engine = self.make_engine()
        try:
            engine._controller.info0d = {"v90_3429_upstream": 0}
            engine._controller.info1d = {
                "blocks": [{"max_rate": 0} for _ in range(5)]
                + [{"hi_carrier": 0, "preemph": 2, "max_rate": 12}],
            }
            actions = [AnaloguePhase2Action(100, "info1a", 6),
                       AnaloguePhase2Action(None, "handoff-v90-phase3", 6,
                                            after="info1a-complete")]
            result = engine._prepare_v90_info1a(actions)
            self.assertEqual(result[0].value, 5)
            self.assertEqual(result[1].signal, "handoff-v34-answer-phase3")
            self.assertEqual(result[1].value, 5)
            self.assertEqual(engine._controller.mode, 5)
            self.assertFalse(engine._failed)
            self.assertEqual(engine.events[-1][0], "phase2-v34-fallback-selected")
            decoded = v34_info.parse_info1a(engine._build_info1a(5))
            self.assertEqual(decoded["sym_ans"], 5)
            self.assertEqual(decoded["sym_call"], 5)
            self.assertEqual(decoded["max_rate"], 12)
        finally:
            engine.close()

    def test_v34_fallback_selects_lower_mutually_supported_block(self):
        blocks = [{"max_rate": 0} for _ in range(6)]
        blocks[4] = {"hi_carrier": 0, "max_rate": 8}
        blocks[5] = {"hi_carrier": 0, "max_rate": 12}
        self.assertEqual(select_v34_fallback_symbol_rate(
            {"lo3200": 1, "rate3429": 1, "allow3429": 1},
            {"blocks": blocks}, 5), 5)
        self.assertEqual(select_v34_fallback_symbol_rate(
            {"lo3200": 1, "rate3429": 0}, {"blocks": blocks}, 5), 4)
        self.assertIsNone(select_v34_fallback_symbol_rate(
            {"rate3429": 1, "allow3429": 0}, {"blocks": blocks}, 5))

    def test_selected_v34_fallback_starts_answer_phase3_after_info1a(self):
        engine = self.make_engine()
        try:
            engine._controller.info0d = {"v90_3429_upstream": 0}
            blocks = [{"max_rate": 0} for _ in range(5)]
            blocks.append({"hi_carrier": 0, "preemph": 0, "max_rate": 12})
            engine._controller.info1d = {"blocks": blocks}
            actions = [AnaloguePhase2Action(100, "info1a", 6),
                       AnaloguePhase2Action(None, "handoff-v90-phase3", 6,
                                            after="info1a-complete")]
            engine._prepare_v90_info1a(actions)
            engine._renderer.handoffs.append(AnalogueTxHandoff(
                1000, "handoff-v34-answer-phase3", 5))
            engine._pump_tx_events()
            self.assertIsNone(engine._v34_fallback_engine,
                               "V.34 must not replace the renderer before INFO1a ends")
            engine._renderer.cursor = 1000
            engine._pump_tx_events()
            fallback = engine._v34_fallback_engine
            self.assertIsNotNone(fallback)
            self.assertEqual(fallback.answerer.stage,
                             "v90_v34_fallback_silence")
            self.assertEqual(max(map(abs, fallback.tx_samples(560)), default=0), 0)
            fallback.feed_pcm(bytes(320))
            fallback.tx_samples(1)  # drain the final 70-ms silence marker
            self.assertEqual(fallback.answerer.stage, "p3_tx")
            self.assertGreater(max(map(abs, fallback.tx_samples(160))), 0)
            self.assertTrue(any(name == "phase2-v34-fallback-started"
                                for name, _ in engine.events))
        finally:
            engine.close()

    def test_v34_fallback_gap_uses_pcm_wrapper_and_advances_receive_clock(self):
        class Fallback:
            def __init__(self):
                self.concealed = 0
                self.deadline = None
                self.pcm = b""

            def note_concealed_rx(self, count):
                self.concealed = count

            def set_rx_media_deadline(self, deadline):
                self.deadline = deadline

            def feed_pcm(self, pcm):
                self.pcm = pcm
                return b"decoded"

        engine = self.make_engine()
        try:
            fallback = Fallback()
            engine._v34_fallback_engine = fallback
            result = engine.feed_gap(160, "pcmu", {
                "packet_playout_deadline_monotonic": 100.0,
            })
            self.assertEqual(result, b"decoded")
            self.assertEqual(fallback.concealed, 160)
            self.assertEqual(fallback.deadline, 100.0)
            self.assertEqual(len(fallback.pcm), 320)
            self.assertEqual(engine._rx_at, 160)
        finally:
            engine.close()

    def test_engine_reduces_v90_upstream_rate_to_peer_supported_block(self):
        engine = self.make_engine()
        try:
            engine._profile["info0a_fields"]["lo3200"] = 1
            engine._controller.info0d = {"v90_3429_upstream": 0}
            blocks = [{"max_rate": 0} for _ in range(6)]
            blocks[4] = {"hi_carrier": 0, "max_rate": 8}
            blocks[5] = {"hi_carrier": 0, "max_rate": 12}
            engine._controller.info1d = {"blocks": blocks}
            actions = [AnaloguePhase2Action(100, "info1a", 6)]
            self.assertEqual(engine._prepare_v90_info1a(actions), actions)
            self.assertEqual(engine._profile_info1a_fields["analog_symbol_rate"], 4)
            self.assertEqual(engine.events[-1][0], "phase2-rate-selected")
        finally:
            engine.close()

    def test_requires_the_analogue_caller_pcm_role(self):
        with self.assertRaisesRegex(ValueError, "analogue-caller role"):
            V90AnalogueCallerPhase2Engine(
                role="digital-answerer",
                negotiation={"pcm_negotiated_bits": V8_PCM_ANALOG},
                media_at=0, media_origin=100.0, law="pcmu", profile=profile())
        with self.assertRaisesRegex(ValueError, "did not negotiate"):
            V90AnalogueCallerPhase2Engine(
                role="analogue-caller",
                negotiation={"pcm_negotiated_bits": 0},
                media_at=0, media_origin=100.0, law="pcmu", profile=profile())

    def test_phase2_transmitter_uses_selected_g711_law(self):
        for law in ("pcmu", "pcma"):
            with self.subTest(law=law):
                engine = self.make_engine(law)
                try:
                    samples = engine.tx_samples(160)
                    codewords = engine.tx_g711(160, law)
                    self.assertEqual(len(samples), 160)
                    self.assertEqual(len(codewords), 160)
                    self.assertEqual(engine.kind, "v90")
                    self.assertFalse(engine.connected)
                    self.assertEqual(engine.bit_rate, 0)
                finally:
                    engine.close()

    def test_phase1_silence_is_emitted_on_the_dce_sample_clock(self):
        engine = V90AnalogueCallerPhase2Engine(
            role="analogue-caller",
            negotiation={"pcm_negotiated_bits": V8_PCM_ANALOG},
            media_at=12000, media_origin=100.0, law="pcmu", profile=profile(),
            phase1_silence_samples=600)
        try:
            self.assertEqual(engine.tx_samples(600), [0] * 600)
            self.assertGreater(max(abs(x) for x in engine.tx_samples(160)), 0)
            self.assertEqual(engine._renderer.cursor, 12760)
        finally:
            engine.close()

    def test_raw_g711_receive_preserves_phase2_clocked_input(self):
        engine = self.make_engine("pcmu")
        try:
            received = engine.feed_g711(
                bytes([0xff]) * 160, "pcmu",
                {"packet_playout_deadline_monotonic": 100.0,
                 "payload_samples": 160})
            self.assertEqual(received, b"")
            self.assertEqual(engine._rx_at, 160)
            self.assertFalse(engine._failed)
        finally:
            engine.close()

    def test_gap_fails_startup_instead_of_inventing_timing(self):
        engine = self.make_engine("pcma")
        try:
            self.assertEqual(engine.feed_gap(
                160, "pcma", {"reason": "test"}), b"")
            self.assertTrue(engine._failed)
            self.assertEqual(engine.events[-1][0], "phase2-failed")
        finally:
            engine.close()

    def test_opt_in_phase2_gap_fill_advances_clock_through_bounded_silence(self):
        p = profile()
        p.update(phase2_rtp_gap_policy="fill_silence",
                 phase2_rtp_gap_max_samples=320)
        engine = V90AnalogueCallerPhase2Engine(
            role="analogue-caller",
            negotiation={"pcm_negotiated_bits": V8_PCM_ANALOG},
            media_at=0, media_origin=100.0, law="pcmu", profile=p)
        try:
            engine.feed_gap(160, "pcmu", {
                "sequence": 12,
                "packet_playout_deadline_monotonic": 100.0,
            })
            self.assertFalse(engine._failed)
            self.assertEqual(engine._rx_at, 160)
            self.assertEqual(engine.events[-1][0], "phase2-rx-gap-filled")
            self.assertEqual(engine.events[-1][1]["samples"], 160)
        finally:
            engine.close()

    def test_opt_in_phase2_gap_fill_still_fails_above_bound(self):
        p = profile()
        p.update(phase2_rtp_gap_policy="fill_silence",
                 phase2_rtp_gap_max_samples=160)
        engine = V90AnalogueCallerPhase2Engine(
            role="analogue-caller",
            negotiation={"pcm_negotiated_bits": V8_PCM_ANALOG},
            media_at=0, media_origin=100.0, law="pcmu", profile=p)
        try:
            engine.feed_gap(320, "pcmu", {"sequence": 12})
            self.assertTrue(engine._failed)
            self.assertEqual(engine.events[-1][0], "phase2-failed")
        finally:
            engine.close()

    def test_phase2_clock_fit_overrides_remain_profile_scoped(self):
        default_engine = self.make_engine()
        try:
            self.assertEqual(default_engine._clock.low_delay_percentile, 15.0)
            self.assertEqual(default_engine._clock.max_rate_error, 0.002)
            self.assertEqual(default_engine._clock.anchors.maxlen, 64)
            self.assertEqual(default_engine._clock.min_fit_span_samples, 6400)
        finally:
            default_engine.close()
        p = profile()
        p.update(phase2_clock_low_delay_percentile=25,
                 phase2_clock_max_rate_error_ppm=25000,
                 phase2_clock_anchor_limit=128,
                 phase2_clock_min_fit_span_samples=12800)
        engine = V90AnalogueCallerPhase2Engine(
            role="analogue-caller",
            negotiation={"pcm_negotiated_bits": V8_PCM_ANALOG},
            media_at=0, media_origin=100.0, law="pcmu", profile=p)
        try:
            self.assertEqual(engine._clock.low_delay_percentile, 25.0)
            self.assertAlmostEqual(engine._clock.max_rate_error, 0.025)
            self.assertEqual(engine._clock.anchors.maxlen, 128)
            self.assertEqual(engine._clock.min_fit_span_samples, 12800)
        finally:
            engine.close()


    def test_factory_binds_the_profile_and_law(self):
        factory = build_v90_analogue_caller_engine(
            law="pcmu", profile=profile())
        engine = factory(
            role="analogue-caller",
            negotiation={"pcm_negotiated_bits": V8_PCM_ANALOG},
            media_at=0, media_origin=100.0)
        try:
            self.assertEqual(engine.law, "pcmu")
        finally:
            engine.close()

    def test_peer_recovery_route_defaults_on_and_can_be_disabled(self):
        default_engine = self.make_engine()
        try:
            self.assertEqual(default_engine._controller.late_l2_echo_policy,
                             "fail")
            self.assertTrue(
                default_engine._controller.accept_early_peer_recovery)
        finally:
            default_engine.close()
        configured = profile()
        configured["phase2_accept_early_peer_recovery"] = False
        disabled = V90AnalogueCallerPhase2Engine(
            role="analogue-caller",
            negotiation={"pcm_negotiated_bits": V8_PCM_ANALOG},
            media_at=0, media_origin=100.0, law="pcmu", profile=configured)
        try:
            self.assertFalse(disabled._controller.accept_early_peer_recovery)
        finally:
            disabled.close()
        configured = profile()
        configured["phase2_late_l2_echo_policy"] = "continue"
        engine = V90AnalogueCallerPhase2Engine(
            role="analogue-caller",
            negotiation={"pcm_negotiated_bits": V8_PCM_ANALOG},
            media_at=0, media_origin=100.0, law="pcmu", profile=configured)
        try:
            self.assertEqual(engine._controller.late_l2_echo_policy, "continue")
        finally:
            engine.close()

    def test_early_peer_recovery_profile_can_override_default(self):
        configured = profile()
        configured.update(phase2_late_l2_echo_policy="bounded",
                          phase2_late_l2_echo_max_samples=20000)
        options = {"v90_analogue_caller": configured}
        self.assertTrue(install_v90_analogue_caller(
            options, tx_role="originate", law="pcmu"))
        engine = options["v90_engine_factory"](
            role="analogue-caller",
            negotiation={"pcm_negotiated_bits": V8_PCM_ANALOG},
            media_at=0, media_origin=100.0)
        try:
            self.assertTrue(engine._controller.accept_early_peer_recovery)
        finally:
            engine.close()

        configured["phase2_accept_early_peer_recovery"] = False
        options = {"v90_analogue_caller": configured}
        self.assertTrue(install_v90_analogue_caller(
            options, tx_role="originate", law="pcmu"))
        disabled = options["v90_engine_factory"](
            role="analogue-caller",
            negotiation={"pcm_negotiated_bits": V8_PCM_ANALOG},
            media_at=0, media_origin=100.0)
        try:
            self.assertFalse(disabled._controller.accept_early_peer_recovery)
        finally:
            disabled.close()

        configured["phase2_accept_early_peer_recovery"] = 1
        with self.assertRaisesRegex(ValueError, "must be boolean"):
            install_v90_analogue_caller(
                {"v90_analogue_caller": configured},
                tx_role="originate", law="pcmu")

    def test_info1d_timeout_retrain_is_profile_opt_in(self):
        configured = profile()
        configured["phase2_info1d_timeout_retrain"] = True
        options = {"v90_analogue_caller": configured}
        self.assertTrue(install_v90_analogue_caller(
            options, tx_role="originate", law="pcmu"))
        engine = options["v90_engine_factory"](
            role="analogue-caller",
            negotiation={"pcm_negotiated_bits": V8_PCM_ANALOG},
            media_at=0, media_origin=100.0)
        try:
            self.assertTrue(engine._controller.info1d_timeout_retrain)
            self.assertFalse(engine._failed)
        finally:
            engine.close()

        configured["phase2_info1d_timeout_retrain"] = 1
        with self.assertRaisesRegex(ValueError, "must be boolean"):
            install_v90_analogue_caller(
                {"v90_analogue_caller": configured},
                tx_role="originate", law="pcmu")

    def test_dce_profile_factory_accepts_bounded_echo_policy(self):
        configured = profile()
        configured.update(phase2_late_l2_echo_policy="bounded",
                          phase2_late_l2_echo_max_samples=20000)
        options = {"v90_analogue_caller": configured}
        self.assertTrue(install_v90_analogue_caller(
            options, tx_role="originate", law="pcmu"))
        self.assertIn("v90_engine_factory", options)

        configured = profile()
        configured.update(phase2_late_l2_echo_policy="bounded",
                          phase2_late_l2_echo_max_samples=20000)
        engine = V90AnalogueCallerPhase2Engine(
            role="analogue-caller",
            negotiation={"pcm_negotiated_bits": V8_PCM_ANALOG},
            media_at=0, media_origin=100.0, law="pcmu", profile=configured)
        try:
            self.assertEqual(engine._controller.late_l2_echo_policy, "bounded")
            self.assertEqual(engine._controller.late_l2_echo_max_samples, 20000)
        finally:
            engine.close()

    def test_dce_profile_factory_accepts_advance_after_window_policy(self):
        configured = profile()
        configured["phase2_late_l2_echo_policy"] = "advance-after-window"
        options = {"v90_analogue_caller": configured}
        self.assertTrue(install_v90_analogue_caller(
            options, tx_role="originate", law="pcmu"))
        self.assertIn("v90_engine_factory", options)

    def _advance_to_phase3_handoff(self, engine, at=0):
        engine._controller.state = engine._controller.COMPLETE
        engine._controller.mode = 6
        engine._controller.info1d = {
            "blocks": [{"hi_carrier": 0} for _ in range(6)]}
        engine._phase3_jd_descriptor = {"rate_mask": (1 << 22) - 1}
        engine._renderer._completion["info1a-complete"] = at
        engine._renderer.cursor = at
        engine._tx_media = at
        engine._renderer._events.clear()
        engine._renderer._mode = "silence"
        engine._renderer._array = None
        engine._maybe_start_phase3()

    def test_phase3_uses_one_level_dil_and_preserves_the_70ms_silence(self):
        engine = self.make_engine()
        try:
            self._advance_to_phase3_handoff(engine, at=1600)
            self.assertEqual(engine._phase3_start_at, 1600)
            self.assertEqual(len(engine._phase3_expected_dil), 768)
            self.assertEqual(engine._phase3_ja_descriptor["N"], 1)
            self.assertEqual(engine._phase3_ja_descriptor["Ucodes"], [0])
            self.assertEqual(engine._phase3_ja_descriptor["TP"], [0])
            pcm = engine.tx_samples(800)
            self.assertEqual(pcm[:560], [0] * 560)
            self.assertGreater(max(abs(sample) for sample in pcm[560:]), 1)
            self.assertFalse(engine.connected)
        finally:
            engine.close()

    def test_phase4_cp_rate_uses_jd_mask_and_profile_cap(self):
        for mask, cap, expected in (
                ((1 << 22) - 1, 22, 22),
                (1 << 9, 22, 10),
                ((1 << 22) - 1, 7, 7),
                ((1 << 22) - 1, 22, 22)):
            with self.subTest(mask=mask, cap=cap):
                engine = self.make_engine(configured={
                    **profile(), "phase4_downstream_drn_cap": cap})
                try:
                    self._advance_to_phase3_handoff(engine, at=0)
                    engine._phase3_jd_descriptor = {"rate_mask": mask}
                    engine._start_phase4(0)
                    self.assertFalse(engine._failed)
                    self.assertEqual(
                        engine._phase4_controller.cp_profile["drn"], expected)
                    self.assertEqual(
                        engine._phase4_downstream_selection["drn"], expected)
                finally:
                    engine.close()

    def test_phase4_exact_drn_pin_is_explicit_and_must_be_advertised(self):
        pinned = self.make_engine(configured={
            **profile(), "phase4_downstream_drn_pin": 7})
        try:
            self._advance_to_phase3_handoff(pinned, at=0)
            pinned._phase3_jd_descriptor = {"rate_mask": (1 << 22) - 1}
            pinned._start_phase4(0)
            self.assertEqual(pinned._phase4_controller.cp_profile["drn"], 7)
            self.assertEqual(pinned._phase4_downstream_selection["pinned_drn"], 7)
            self.assertTrue(pinned._phase4_downstream_selection["requested_pin_met"])
        finally:
            pinned.close()

        for mask, expected_reason in ((0b1, "absent from Jd"),):
            engine = self.make_engine(configured={
                **profile(), "phase4_downstream_drn_pin": 7})
            try:
                self._advance_to_phase3_handoff(engine, at=0)
                engine._phase3_jd_descriptor = {"rate_mask": mask}
                engine._start_phase4(0)
                self.assertTrue(engine._failed)
                self.assertIsNone(engine._phase4_controller)
                self.assertIn(expected_reason, engine.events[-1][1]["reason"])
            finally:
                engine.close()

        missing = self.make_engine()
        try:
            self._advance_to_phase3_handoff(missing, at=0)
            missing._phase3_jd_descriptor = None
            missing._start_phase4(0)
            self.assertTrue(missing._failed)
            self.assertIsNone(missing._phase4_controller)
            self.assertIn("missing CRC-valid Jd",
                          missing.events[-1][1]["reason"])
        finally:
            missing.close()

    def test_phase4_invalid_or_disjoint_jd_mask_fails_closed_without_exception(self):
        engine = self.make_engine(configured={
            **profile(), "phase4_downstream_drn_cap": 5})
        try:
            self._advance_to_phase3_handoff(engine, at=0)
            engine._phase3_jd_descriptor = {"rate_mask": 1 << 10}
            engine._start_phase4(0)
            self.assertTrue(engine._failed)
            self.assertIsNone(engine._phase4_controller)
            self.assertEqual(engine.events[-1][0], "phase4-rate-rejected")
        finally:
            engine.close()

    def test_phase4_timeout_does_not_impersonate_b1d_failure(self):
        engine = self.make_engine()
        try:
            engine._phase4_timeout_at = 0
            with patch.object(engine, "_try_phase4_rate_stepdown",
                              return_value=True) as stepdown:
                engine._pump_timer()
            stepdown.assert_not_called()
            self.assertTrue(engine._failed)
            self.assertEqual(engine.events[-1][0], "phase4-timeout")
        finally:
            engine.close()

    def test_phase3_receiver_gets_exact_raw_g711_after_handoff(self):
        engine = self.make_engine("pcma")
        try:
            self._advance_to_phase3_handoff(engine, at=80)
            engine._clock.observe(0, {
                "packet_playout_deadline_monotonic": 100.0,
                "payload_samples": 160,
            })
            session = Mock()
            session.feed_downstream_g711.return_value = []
            engine._phase3_session = session
            payload = bytes(range(160))
            engine._rx_at = len(payload)
            engine._feed_phase3(payload)
            received, keywords = session.feed_downstream_g711.call_args
            self.assertEqual(received[0], payload[80:])
            self.assertEqual(keywords["at"], 80)
        finally:
            engine.close()

    def test_phase3_dil_length_rejects_invalid_profile_values(self):
        for value in (-6, 5, 774, True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                V90AnalogueCallerPhase2Engine(
                    role="analogue-caller",
                    negotiation={"pcm_negotiated_bits": V8_PCM_ANALOG},
                    media_at=0, media_origin=100.0, law="pcmu",
                    profile={**profile(), "dil_codewords": value})


if __name__ == "__main__":
    unittest.main()
