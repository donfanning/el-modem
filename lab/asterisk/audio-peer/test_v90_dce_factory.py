import os
import unittest
from unittest import mock

from autochain import V8Chain
from v8 import V8_MOD_V34, V8_PCM_DIGITAL, V8_PROTOCOL_LAPM_V42
from v90_dce_factory import (V90_PHASE1_SILENCE_SAMPLES,
                             install_v90_analogue_caller,
                             install_v90_digital_answerer)


def profile():
    blocks = [{} for _ in range(6)]
    blocks[5] = {"hi_carrier": 1}
    return {
        "info0d_fields": {"rate3429": 1, "v90_3429_upstream": 1},
        "info1d_fields": {"blocks": blocks},
        "tone_b_peak": 900,
        "probe_rms": 450,
        "phase3_rate_mask": 0xFFFF,
    }


class V90DceFactoryTests(unittest.TestCase):
    def test_explicit_type1_mp_profile_is_opt_in_and_reaches_engine(self):
        configured = profile()
        self.assertNotIn("phase4_mp_profile", configured)
        configured["phase4_mp_profile"] = {
            "mp_type": 1,
            "max_a2d_drn": 12,
            "trellis": 0,
            "nonlinear": 1,
            "shaping": 1,
            "rate_mask": 0x1FFF,
            "precoder": [520, -52, -119, 1, 116, 137],
        }
        options = {"v90_digital_answerer": configured}
        install_v90_digital_answerer(options, tx_role="answer", law="pcmu")
        engine = options["v90_engine_factory"](
            role="digital-answerer",
            negotiation={"pcm_negotiated_bits": V8_PCM_DIGITAL},
            media_at=0, media_origin=100.0)
        try:
            self.assertEqual(engine._phase4_mp_profile, {
                "mp_type": 1, "max_a2d_drn": 12, "trellis": 0,
                "nonlinear": 1, "shaping": 1, "rate_mask": 0x1FFF,
                "precoder": (520, -52, -119, 1, 116, 137),
            })
        finally:
            engine.close()

    def test_v92_capability_profile_is_opt_in_and_pcm_upstream_stays_off(self):
        configured = profile()
        configured["v92"] = True
        configured["info1d_fields"]["blocks"][5]["hi_carrier"] = 0
        options = {"v90_digital_answerer": configured}
        install_v90_digital_answerer(options, tx_role="answer", law="pcmu")
        engine = options["v90_engine_factory"](
            role="digital-answerer",
            negotiation={"pcm_negotiated_bits": V8_PCM_DIGITAL},
            media_at=0, media_origin=100.0)
        try:
            self.assertTrue(engine.v92_advertised)
            self.assertEqual(engine._signals.info0d_fields["reserved"] & 2, 2)
            self.assertEqual(engine._signals.info1d_fields["blocks"][5]["hi_carrier"], 0)
            self.assertIsNone(engine._phase3_factory)
            self.assertEqual(engine._phase1_silence_samples,
                             V90_PHASE1_SILENCE_SAMPLES)
        finally:
            engine.close()

    def test_quick_connect_does_not_repeat_post_toneq_phase1_silence(self):
        configured = profile()
        configured.update(v92=True, v92_pcm_upstream=True,
                          v92_quick_connect=True)
        options = {"v90_digital_answerer": configured}
        install_v90_digital_answerer(options, tx_role="answer", law="pcmu")
        engine = options["v90_engine_factory"](
            role="digital-answerer",
            negotiation={"pcm_negotiated_bits": V8_PCM_DIGITAL,
                         "quick_connect": True},
            media_at=33280, media_origin=100.0)
        try:
            self.assertEqual(engine._phase1_silence_samples, 0)
            self.assertEqual(engine._phase2_start_at, 33280)
        finally:
            engine.close()

    def test_quick_connect_requires_pcm_upstream_and_flows_only_when_opted_in(self):
        configured = profile()
        configured.update(v92=True, v92_pcm_upstream=True,
                          v92_quick_connect=True)
        options = {"v90_digital_answerer": configured}
        install_v90_digital_answerer(options, tx_role="answer", law="pcmu")
        self.assertTrue(options["v92_quick_connect"])
        self.assertEqual(options["v92_quick_connect_law"], "pcmu")
        self.assertEqual(options["v92_quick_connect_level_code"], 0b01)

        configured["v92_pcm_upstream"] = False
        with self.assertRaisesRegex(ValueError, "needs V.92 PCM upstream"):
            install_v90_digital_answerer(
                {"v90_digital_answerer": configured},
                tx_role="answer", law="pcmu")

    def test_v92_receiver_and_cpd_profile_keys_reach_the_session(self):
        def session_for(configured):
            options = {"v90_digital_answerer": configured}
            install_v90_digital_answerer(options, tx_role="answer", law="pcmu")
            engine = options["v90_engine_factory"](
                role="digital-answerer",
                negotiation={"pcm_negotiated_bits": V8_PCM_DIGITAL},
                media_at=0, media_origin=100.0)
            try:
                return engine, engine._phase3_factory(
                    info1a={"v90_uinfo": 67}, law="pcmu", handoff_at=0, media_origin=100.0)
            finally:
                engine.close()

        configured = profile()
        configured.update(v92=True, v92_pcm_upstream=True)
        env = {k: v for k, v in os.environ.items()
               if k not in ("EL_MODEM_V92_B1U_RX", "EL_MODEM_V92_CPD_CONSTELLATION")}
        with mock.patch.dict(os.environ, env, clear=True):
            engine, session = session_for(configured)
            # Unset keys: the pair validated against SmartLink.
            self.assertEqual(session._b1u_rx_mode, "trn2u-fit-viterbi")
            self.assertEqual(session._cpd_constellation_mode, "adaptive")
            self.assertEqual(engine._clock.offset_step_samples, 40)
            self.assertEqual(engine._clock.violation_epochs, 3)
            configured.update(v92_b1u_rx_mode="off", v92_cpd_constellation="default")
            _, session = session_for(configured)
            self.assertEqual(session._b1u_rx_mode, "off")
            self.assertEqual(session._cpd_constellation_mode, "default")
        for key, value in (("v92_b1u_rx_mode", "hard"), ("v92_cpd_constellation", "dense")):
            bad = dict(configured, **{key: value})
            with self.assertRaises(ValueError):
                install_v90_digital_answerer({"v90_digital_answerer": bad}, tx_role="answer", law="pcmu")
        plain = profile()
        plain["v92_b1u_rx_mode"] = "trn2u-fit-viterbi"
        with self.assertRaises(ValueError):
            install_v90_digital_answerer({"v90_digital_answerer": plain}, tx_role="answer", law="pcmu")
        # FPE keeps the validated receiver (it carries the TRN2u fit across the exchange).
        fpe = profile()
        fpe.update(v92=True, v92_pcm_upstream=True, v92_fast_parameter_exchange=True, phase3_enable_lapm=True)
        with mock.patch.dict(os.environ, env, clear=True):
            _, session = session_for(fpe)
        self.assertEqual(session._b1u_rx_mode, "trn2u-fit-viterbi")
        self.assertEqual(session._cpd_constellation_mode, "adaptive")
        self.assertTrue(session._enable_fpe)

    def test_fractional_rx_timing_is_opt_in_and_reaches_v92_phase3(self):
        configured = profile()
        configured.update(v92=True, v92_pcm_upstream=True,
                          v92_fractional_rx_timing=True)
        options = {"v90_digital_answerer": configured}
        install_v90_digital_answerer(options, tx_role="answer", law="pcmu")
        engine = options["v90_engine_factory"](
            role="digital-answerer",
            negotiation={"pcm_negotiated_bits": V8_PCM_DIGITAL},
            media_at=0, media_origin=100.0)
        try:
            session = engine._phase3_factory(
                info1a={"v90_uinfo": 67}, law="pcmu", handoff_at=0,
                media_origin=100.0)
            self.assertTrue(session._p3.fractional_rx_timing)
            self.assertTrue(session._p3.fractional_rx_reacquisition)
            self.assertIsNone(session._p3.rx_fractional_phase)
        finally:
            engine.close()

        configured["v92_fractional_rx_reacquisition"] = False
        options = {"v90_digital_answerer": configured}
        install_v90_digital_answerer(options, tx_role="answer", law="pcmu")
        engine = options["v90_engine_factory"](
            role="digital-answerer",
            negotiation={"pcm_negotiated_bits": V8_PCM_DIGITAL},
            media_at=0, media_origin=100.0)
        try:
            session = engine._phase3_factory(
                info1a={"v90_uinfo": 67}, law="pcmu", handoff_at=0,
                media_origin=100.0)
            self.assertTrue(session._p3.fractional_rx_timing)
            self.assertFalse(session._p3.fractional_rx_reacquisition)
        finally:
            engine.close()

        configured["v92_fractional_rx_timing"] = False
        configured["v92_fractional_rx_reacquisition"] = True
        with self.assertRaisesRegex(ValueError, "needs v92_fractional_rx_timing"):
            install_v90_digital_answerer(
                {"v90_digital_answerer": configured},
                tx_role="answer", law="pcmu")

        configured["v92_pcm_upstream"] = False
        with self.assertRaisesRegex(ValueError, "needs v92_pcm_upstream"):
            install_v90_digital_answerer(
                {"v90_digital_answerer": configured},
                tx_role="answer", law="pcmu")

    def test_v42bis_enabled_without_lapm_does_not_reject_non_pcm_profile(self):
        options = {"v90_digital_answerer": profile()}
        self.assertTrue(install_v90_digital_answerer(
            options, tx_role="answer", law="pcmu", v42bis=True))


    def test_probe_rate_projection_setting_is_opt_in_and_validated(self):
        configured = profile()
        options = {"v90_digital_answerer": configured}
        install_v90_digital_answerer(options, tx_role="answer", law="pcmu")
        engine = options["v90_engine_factory"](
            role="digital-answerer",
            negotiation={"pcm_negotiated_bits": V8_PCM_DIGITAL},
            media_at=0, media_origin=100.0)
        try:
            self.assertTrue(engine._phase2_project_info1d_rates_from_probe)
        finally:
            engine.close()

        configured["phase2_project_info1d_rates_from_probe"] = False
        options = {"v90_digital_answerer": configured}
        install_v90_digital_answerer(options, tx_role="answer", law="pcmu")
        engine = options["v90_engine_factory"](
            role="digital-answerer",
            negotiation={"pcm_negotiated_bits": V8_PCM_DIGITAL},
            media_at=0, media_origin=100.0)
        try:
            self.assertFalse(engine._phase2_project_info1d_rates_from_probe)
        finally:
            engine.close()

        for bad in (0, 1, None, "false"):
            configured["phase2_project_info1d_rates_from_probe"] = bad
            with self.assertRaisesRegex(ValueError, "project_info1d_rates"):
                install_v90_digital_answerer(
                    {"v90_digital_answerer": configured},
                    tx_role="answer", law="pcmu")

    def test_absent_profile_keeps_pcm_unadvertised(self):
        options = {"allow": V8_MOD_V34}
        self.assertFalse(install_v90_digital_answerer(
            options, tx_role="answer", law="pcmu"))
        chain = V8Chain(False, **options)
        try:
            self.assertEqual(chain.pcm_availability, 0)
            self.assertIsNone(chain.v90_engine_factory)
        finally:
            chain.close()

    def test_explicit_profile_selects_phase2_engine_on_digital_v8_role(self):
        for protocols in (0, V8_PROTOCOL_LAPM_V42):
            with self.subTest(protocols=protocols):
                configured = profile()
                configured["phase3_enable_lapm"] = True
                options = {"allow": V8_MOD_V34,
                           "protocols": protocols,
                           "v90_digital_answerer": configured}
                self.assertTrue(install_v90_digital_answerer(
                    options, tx_role="answer", law="pcmu"))
                self.assertEqual(options["pcm_availability"], V8_PCM_DIGITAL)
                chain = V8Chain(False, **options)
                try:
                    chain._v8.status = 2
                    chain._v8.result = {
                        "modulations": V8_MOD_V34,
                        "protocols": protocols,
                        "pcm_negotiated": True,
                        "pcm_negotiated_bits": V8_PCM_DIGITAL,
                    }
                    chain._step_v8()
                    self.assertEqual(chain.kind, "v90")
                    self.assertEqual(chain.engine.kind, "v90")
                    self.assertFalse(chain.engine.connected)
                    self.assertEqual(chain.engine.law, "pcmu")
                    # Without V.8 LAPM the V.42 detection phase decides.
                    self.assertTrue(chain.engine.supports_v42)
                finally:
                    chain.close()

    def test_profile_requires_complete_explicit_signal_and_rate_choices(self):
        options = {"v90_digital_answerer": {"tone_b_peak": 900}}
        with self.assertRaisesRegex(ValueError, "missing"):
            install_v90_digital_answerer(options, tx_role="answer", law="pcmu")

        bad_profile = profile()
        bad_profile["phase3_rate_mask"] = True
        with self.assertRaisesRegex(ValueError, "rate_mask must be an integer"):
            install_v90_digital_answerer(
                {"v90_digital_answerer": bad_profile},
                tx_role="answer", law="pcmu")

    def test_call_roles_let_either_profile_serve_the_other_direction(self):
        options = {"v90_digital_answerer": dict(
            profile(), call_roles=["originate"], v92=True, v92_pcm_upstream=True,
            v92_quick_connect=True)}
        self.assertTrue(install_v90_digital_answerer(options, tx_role="originate", law="pcmu"))
        self.assertEqual(options["pcm_availability"], V8_PCM_DIGITAL)
        # Quick Connect is an analogue caller's request.
        self.assertFalse(options["v92_quick_connect"])
        with self.assertRaisesRegex(ValueError, "answer role"):
            install_v90_digital_answerer(
                {"v90_digital_answerer": dict(profile(), call_roles=["originate"])},
                tx_role="answer", law="pcmu")
        for bad in ("answer", [], ["answer", "answer"], ["dial"]):
            with self.subTest(call_roles=bad), self.assertRaisesRegex(ValueError, "call_roles"):
                install_v90_digital_answerer(
                    {"v90_digital_answerer": dict(profile(), call_roles=bad)},
                    tx_role="answer", law="pcmu")
        caller = {"info0a_fields": {}, "info1a_fields": {}, "nominal_peak": 900,
                  "probe_rms": 450}
        default = {"v90_analogue_caller": dict(caller)}
        self.assertFalse(install_v90_analogue_caller(default, tx_role="answer", law="pcmu"))
        self.assertNotIn("pcm_availability", default)
        answering = {"v90_analogue_caller": dict(caller, call_roles=["answer"])}
        self.assertTrue(install_v90_analogue_caller(answering, tx_role="answer", law="pcmu"))
        self.assertEqual(answering["pcm_availability"], 0x20)

    def test_profile_cannot_enable_wrong_role_or_replace_pcm_role(self):
        with self.assertRaisesRegex(ValueError, "originate role"):
            install_v90_digital_answerer(
                {"v90_digital_answerer": profile()},
                tx_role="originate", law="pcmu")
        with self.assertRaisesRegex(ValueError, "cannot override"):
            install_v90_digital_answerer(
                {"pcm_availability": 0x40,
                 "v90_digital_answerer": profile()},
                tx_role="answer", law="pcmu")

    def test_selected_g711_law_is_bound_to_factory(self):
        options = {"v90_digital_answerer": profile()}
        install_v90_digital_answerer(options, tx_role="answer", law="pcma")
        engine = options["v90_engine_factory"](
            role="digital-answerer",
            negotiation={"pcm_negotiated_bits": V8_PCM_DIGITAL},
            media_at=0, media_origin=100.0)
        try:
            self.assertEqual(engine.law, "pcma")
            output = engine.tx_g711(160, "pcma")
            self.assertEqual(len(output), 160)
        finally:
            engine.close()


    def test_upstream_equalizer_step_is_an_opt_in_profile_setting(self):
        def build(configured):
            options = {"v90_digital_answerer": configured}
            install_v90_digital_answerer(options, tx_role="answer", law="pcmu")
            engine = options["v90_engine_factory"](
                role="digital-answerer",
                negotiation={"pcm_negotiated_bits": V8_PCM_DIGITAL},
                media_at=0, media_origin=100.0)
            return engine

        engine = build(profile())
        try:
            self.assertIsNone(engine._phase3_upstream_equalizer_mu)   # default unchanged
        finally:
            engine.close()
        configured = profile()
        configured["phase3_upstream_equalizer_mu"] = 0.1
        engine = build(configured)
        try:
            self.assertEqual(engine._phase3_upstream_equalizer_mu, 0.1)
        finally:
            engine.close()
        for bad in (0, -0.1, 1.5, True, "0.1", float("nan")):
            configured["phase3_upstream_equalizer_mu"] = bad
            with self.assertRaisesRegex(ValueError, "equalizer_mu"):
                install_v90_digital_answerer({"v90_digital_answerer": configured},
                                             tx_role="answer", law="pcmu")

    def test_l2_echo_tolerance_is_an_opt_in_profile_setting(self):
        def build(configured):
            options = {"v90_digital_answerer": configured}
            install_v90_digital_answerer(options, tx_role="answer", law="pcmu")
            return options["v90_engine_factory"](
                role="digital-answerer",
                negotiation={"pcm_negotiated_bits": V8_PCM_DIGITAL},
                media_at=0, media_origin=100.0)

        engine = build(profile())
        try:
            self.assertEqual(engine._phase2.l2_echo_tolerance_samples, 0)   # default exact
        finally:
            engine.close()
        configured = profile()
        configured["phase2_l2_echo_tolerance_samples"] = 8
        engine = build(configured)
        try:
            self.assertEqual(engine._phase2.l2_echo_tolerance_samples, 8)
        finally:
            engine.close()
        for bad in (-1, 81, True, 1.5, "8"):
            configured["phase2_l2_echo_tolerance_samples"] = bad
            with self.assertRaisesRegex(ValueError, "l2_echo_tolerance"):
                install_v90_digital_answerer({"v90_digital_answerer": configured},
                                             tx_role="answer", law="pcmu")

    def test_l2_echo_window_policy_is_opt_in_and_validated(self):
        def build(configured):
            options = {"v90_digital_answerer": configured}
            install_v90_digital_answerer(options, tx_role="answer", law="pcmu")
            return options["v90_engine_factory"](
                role="digital-answerer",
                negotiation={"pcm_negotiated_bits": V8_PCM_DIGITAL},
                media_at=0, media_origin=100.0)

        engine = build(profile())
        try:
            self.assertEqual(engine._phase2.l2_echo_window_policy, "require")
        finally:
            engine.close()

        configured = profile()
        configured["phase2_l2_echo_window_policy"] = "advance-after-window"
        engine = build(configured)
        try:
            self.assertEqual(
                engine._phase2.l2_echo_window_policy, "advance-after-window")
        finally:
            engine.close()
        configured = profile()
        configured["phase2_l2_echo_window_policy"] = "on-tone-a"
        engine = build(configured)
        try:
            self.assertEqual(engine._phase2.l2_echo_window_policy, "on-tone-a")
        finally:
            engine.close()
        for bad in (None, True, "continue", "advance"):
            configured["phase2_l2_echo_window_policy"] = bad
            with self.assertRaisesRegex(ValueError, "l2_echo_window_policy"):
                install_v90_digital_answerer(
                    {"v90_digital_answerer": configured},
                    tx_role="answer", law="pcmu")

    def test_clock_offset_step_threshold_is_an_opt_in_profile_setting(self):
        def build(configured):
            options = {"v90_digital_answerer": configured}
            install_v90_digital_answerer(options, tx_role="answer", law="pcmu")
            return options["v90_engine_factory"](
                role="digital-answerer",
                negotiation={"pcm_negotiated_bits": V8_PCM_DIGITAL},
                media_at=0, media_origin=100.0)

        engine = build(profile())
        try:
            self.assertEqual(engine._clock.offset_step_samples, 80)
        finally:
            engine.close()
        configured = profile()
        configured["phase2_clock_offset_step_samples"] = 40
        engine = build(configured)
        try:
            self.assertEqual(engine._clock.offset_step_samples, 40)
        finally:
            engine.close()
        for bad in (0, 8001, True, 1.5, "40"):
            configured["phase2_clock_offset_step_samples"] = bad
            with self.assertRaisesRegex(ValueError, "clock_offset_step_samples"):
                install_v90_digital_answerer(
                    {"v90_digital_answerer": configured},
                    tx_role="answer", law="pcmu")

    def test_retrain_on_failure_is_an_opt_in_profile_setting(self):
        def build(configured):
            options = {"v90_digital_answerer": configured}
            install_v90_digital_answerer(options, tx_role="answer", law="pcmu")
            return options["v90_engine_factory"](
                role="digital-answerer",
                negotiation={"pcm_negotiated_bits": V8_PCM_DIGITAL},
                media_at=0, media_origin=100.0)

        engine = build(profile())
        try:
            self.assertFalse(engine._retrain_on_failure)
            self.assertEqual(engine._max_retrains, 2)
        finally:
            engine.close()
        configured = profile()
        configured.update(retrain_on_failure=True, max_retrains=3)
        engine = build(configured)
        try:
            self.assertTrue(engine._retrain_on_failure)
            self.assertEqual(engine._max_retrains, 3)
        finally:
            engine.close()
        for key, bad in (("retrain_on_failure", 1), ("max_retrains", 0), ("max_retrains", 6),
                         ("max_retrains", True)):
            configured = profile()
            configured[key] = bad
            with self.assertRaisesRegex(ValueError, key):
                install_v90_digital_answerer({"v90_digital_answerer": configured},
                                             tx_role="answer", law="pcmu")





    def test_upstream_equalizer_step_reaches_the_upstream_session(self):
        from v90_digital_phase3_session import V90DigitalAnswererPhase3Session
        kwargs = dict(baud=3429.0, carrier=1959.0, law="pcmu", uinfo=90, rate_mask=0xFFFF)
        self.assertIsNone(V90DigitalAnswererPhase3Session(**kwargs).upstream.equalizer_mu)
        self.assertEqual(V90DigitalAnswererPhase3Session(
            **kwargs, upstream_equalizer_mu=0.1).upstream.equalizer_mu, 0.1)


if __name__ == "__main__":
    unittest.main()
