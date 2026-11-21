import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np

from autochain import V8Chain
import v34_g711
import v92_analogue_session as v92_analogue
from v8 import V8_MOD_V34, V8_PCM_DIGITAL, V8_PROTOCOL_LAPM_V42
from v90_dce_factory import install_v90_digital_answerer
import v90_info_reference as v90_info


def profile(*, v92=None, hi_carrier=0):
    blocks = [{} for _ in range(6)]
    blocks[5] = {"hi_carrier": hi_carrier}
    p = {"info0d_fields": {"rate3429": 1, "v90_3429_upstream": 1},
         "info1d_fields": {"blocks": blocks}, "tone_b_peak": 900, "probe_rms": 450,
         "phase3_rate_mask": 0xFFFF}
    if v92 is not None:
        p["v92"] = v92
    return p


def engine_for(p, law="pcmu", protocols=0):
    options = {"allow": V8_MOD_V34, "v90_digital_answerer": p}
    assert install_v90_digital_answerer(options, tx_role="answer", law=law)
    chain = V8Chain(False, **options)
    chain._v8.status = 2
    chain._v8.result = {"modulations": V8_MOD_V34, "protocols": protocols, "pcm_negotiated": True,
                        "pcm_negotiated_bits": V8_PCM_DIGITAL}
    chain._step_v8()
    return chain


class V92InV90ModeTests(unittest.TestCase):
    def test_off_by_default_changes_nothing(self):
        chain = engine_for(profile(hi_carrier=1))
        try:
            self.assertEqual(int(chain.engine._signals.info0d_fields.get("reserved", 0)), 0)
            info0d = v90_info.parse_info0d(chain.engine._signals.info0d_bits())
            self.assertEqual(info0d["max_power"], 0)
            self.assertEqual(info0d["power_at_codec"], 0)
            self.assertEqual(chain.engine.v92_report,
                             {"advertised": False, "peer_capable": None, "negotiated": False,
                              "fast_parameter_exchange_events": []})
        finally:
            chain.close()

    def test_v92_sets_info0d_bit_27_and_keeps_info1d_bit_70_clear(self):
        chain = engine_for(profile(v92=True, hi_carrier=0))
        try:
            signals = chain.engine._signals
            self.assertEqual(int(signals.info0d_fields["reserved"]) & 2, 2)     # bit 27
            self.assertEqual(int(signals.info0d_fields["reserved"]) & 1, 0)     # bit 26 (short Phase 2) clear
            self.assertEqual(signals.info1d_fields["blocks"][5].get("hi_carrier", 0), 0)
            bits = signals.build_info0d_bits() if hasattr(signals, "build_info0d_bits") else None
            self.assertTrue(chain.engine.v92_report["advertised"])
        finally:
            chain.close()

    def test_v92_defaults_to_codec_output_minus_12_dbm0(self):
        chain = engine_for(profile(v92=True))
        try:
            signals = chain.engine._signals
            decoded = v90_info.parse_info0d(signals.info0d_bits())
            self.assertIsNotNone(decoded)
            self.assertEqual(decoded["max_power"], 23)
            self.assertEqual(decoded["power_at_codec"], 1)  # INFO0d bit 38
        finally:
            chain.close()

    def test_v92_info0d_power_defaults_are_overridable(self):
        configured = profile(v92=True)
        configured["info0d_fields"].update(max_power=19, power_at_codec=0)
        chain = engine_for(configured)
        try:
            decoded = v90_info.parse_info0d(chain.engine._signals.info0d_bits())
            self.assertEqual(decoded["max_power"], 19)
            self.assertEqual(decoded["power_at_codec"], 0)
        finally:
            chain.close()

    def test_a_v92_profile_may_not_offer_pcm_upstream(self):
        with self.assertRaisesRegex(ValueError, "bit 70"):
            install_v90_digital_answerer({"allow": V8_MOD_V34, "v90_digital_answerer":
                                          profile(v92=True, hi_carrier=1)},
                                         tx_role="answer", law="pcmu")

    def test_the_flag_must_be_boolean(self):
        with self.assertRaises(ValueError):
            install_v90_digital_answerer({"allow": V8_MOD_V34, "v90_digital_answerer":
                                          profile(v92=1)}, tx_role="answer", law="pcmu")

    def test_negotiation_needs_both_capability_bits(self):
        chain = engine_for(profile(v92=True))
        try:
            engine = chain.engine
            for clock, peer, negotiated in ((0, False, False), (1, True, True), (2, False, False),
                                            (3, True, True)):
                engine._phase2.info0a = {"clock": clock}
                self.assertEqual(engine.v92_report,
                                 {"advertised": True, "peer_capable": peer,
                                  "negotiated": negotiated,
                                  "fast_parameter_exchange_events": []})
            engine._phase2.info0a = None
            self.assertIsNone(engine.v92_report["peer_capable"])
        finally:
            chain.close()


class V92PcmUpstreamKeyTests(unittest.TestCase):
    def pcm_profile(self, **extra):
        p = profile(v92=True)
        p["v92_pcm_upstream"] = True
        p["phase3_enable_lapm"] = True
        p.update(extra)
        return p

    def test_negotiated_v8_v92_handoff_runs_direct_lapm_with_exact_payload(self):
        """Exercise the production V.8/V.90 handoff into the paired V.92 link."""
        for law in ("pcmu", "pcma"):
            with self.subTest(law=law):
                chain = engine_for(
                    self.pcm_profile(), law=law,
                    protocols=V8_PROTOCOL_LAPM_V42)
                analogue = None
                try:
                    engine = chain.engine
                    engine._phase2.info0a = {"clock": 1}  # peer advertises V.92
                    engine._phase2.info1a = {
                        "v90_uinfo": 67,
                        "v90_analog_symbol_rate": 5,
                        "v90_digital_symbol_rate": 6,
                    }
                    engine._start_phase3(SimpleNamespace(at=0))
                    digital = engine._phase3
                    self.assertIsNotNone(digital)
                    self.assertFalse(digital._lapm_detect)
                    self.assertTrue(digital.enable_lapm)

                    analogue = v92_analogue.V92AnalogueSession(
                        law=law, lapm_detect=False)
                    up = bytes((i * 37 + 5) & 0xff for i in range(128))
                    down = bytes((i * 91 + 17) & 0xff for i in range(128))
                    analogue.feed_bytes(up)
                    digital.feed_bytes(down)
                    silence = int(v34_g711.encode(np.zeros(1), law)[0]) & 0xff
                    to_digital = [0.0] * 240
                    to_analogue = [float(v34_g711.decode(
                        np.array([silence], dtype=np.uint8), law)[0])] * 240
                    got_up, got_down = bytearray(), bytearray()
                    for _ in range(500):
                        analogue_tx = analogue.read(160)
                        digital_tx = digital.read_g711(160, law)
                        to_digital.extend(v34_g711.roundtrip(
                            np.rint(analogue_tx), law).astype(float))
                        to_analogue.extend(v34_g711.decode(
                            np.frombuffer(digital_tx, dtype=np.uint8), law).astype(float))
                        digital.feed_pcm(np.asarray(to_digital[:160]))
                        del to_digital[:160]
                        # The production chain fits the TRN2u equaliser on a thread; this loop runs faster
                        # than real time, so let a pending fit finish before more media arrives.
                        fit = getattr(getattr(digital._p4, "_decoder", None), "_fit_thread", None)
                        if fit is not None:
                            fit.join(timeout=10.0)
                        analogue.feed(np.asarray(to_analogue[:160]))
                        del to_analogue[:160]
                        got_up.extend(digital.read_received_data())
                        got_down.extend(analogue.take_received())
                        if got_up == up and got_down == down:
                            break

                    self.assertEqual(bytes(got_up), up)
                    self.assertEqual(bytes(got_down), down)
                    self.assertTrue(analogue.connected)
                    self.assertEqual(digital.state, "data")
                    self.assertTrue(analogue.lapm._v42_direct)
                    self.assertTrue(digital.lapm._v42_direct)
                    self.assertNotIn("ODP received", digital.lapm._v42.events)
                    self.assertNotIn("ADP received", analogue.lapm._v42.events)
                finally:
                    chain.close()

    def test_it_needs_v92_and_a_boolean(self):
        p = profile()
        p["v92_pcm_upstream"] = True
        with self.assertRaisesRegex(ValueError, "needs v92"):
            install_v90_digital_answerer({"allow": V8_MOD_V34, "v90_digital_answerer": p},
                                         tx_role="answer", law="pcmu")
        with self.assertRaises(ValueError):
            install_v90_digital_answerer({"allow": V8_MOD_V34, "v90_digital_answerer":
                                          self.pcm_profile(v92_pcm_upstream=1)}, tx_role="answer", law="pcmu")

    def test_v92_session_factory_receives_the_negotiated_lapm_detection_policy(self):
        from v92_digital_session import factory

        make = factory(law="pcmu", enable_lapm=True, lapm_detect=True)
        session = make(info1a={"v90_uinfo": 67}, law="pcmu",
                       handoff_at=0, media_origin=100.0,
                       lapm_detect=False)
        self.assertFalse(session._lapm_detect)

    def test_it_offers_pcm_upstream_and_installs_the_v92_session(self):
        chain = engine_for(self.pcm_profile())
        try:
            engine = chain.engine
            self.assertEqual(engine._signals.info1d_fields["blocks"][5].get("hi_carrier"), 1)    # bit 70 set
            self.assertEqual(int(engine._signals.info0d_fields["reserved"]) & 2, 2)              # bit 27
            self.assertIsNotNone(engine._phase3_factory)
            self.assertEqual(engine._phase3_rate_mask, 0xFFFF)    # plain V.90 for a peer without V.92
        finally:
            chain.close()

    def test_lost_frame_allowance_reaches_the_live_session_factory(self):
        chain = engine_for(self.pcm_profile(
            v92_pcm_upstream_max_lost_frames=16))
        try:
            session = chain.engine._phase3_factory(
                info1a={"v90_uinfo": 67}, law="pcmu", handoff_at=0,
                media_origin=100.0)
            self.assertEqual(session._pcm_upstream_max_lost_frames, 16)
            self.assertEqual(session._p4, None)  # Phase 4 receiver is built at the Phase 3 handoff.
        finally:
            chain.close()

    def test_the_v92_session_has_the_surface_the_engine_and_the_metrics_read(self):
        import v92_digital_session as ds
        session = ds.V92DigitalSession(law="pcmu")
        for name in ("feed_pcm", "read_g711", "feed_bytes", "read_received_data"):
            self.assertTrue(callable(getattr(session, name)))
        self.assertIsInstance(session.transmit_ready, bool)
        self.assertIsInstance(session.state, str)
        self.assertEqual(session.phase4.transmitter.DATA_READY, "data-ready")
        self.assertIsNone(session.upstream)
        self.assertIsNone(session.upstream_data)
        self.assertEqual(len(session.read_g711(160, "pcmu")), 160)            # silence before anything arrives

    def test_dce_engine_exposes_only_an_explicit_active_session_rate_request(self):
        chain = engine_for(self.pcm_profile())
        try:
            engine = chain.engine
            requests = []
            engine._phase3 = SimpleNamespace(
                request_rate_renegotiation=lambda drn, *, silent_period=False, silent_early_rt_after=None:
                    requests.append((drn, silent_period, silent_early_rt_after)))
            self.assertTrue(engine.request_v92_rate_renegotiation(8))
            self.assertEqual(requests, [(8, False, None)])
            self.assertEqual(engine.events[-1],
                             ("v92-rate-renegotiation-requested",
                              {"drn_up": 8, "silent_period": False,
                               "silent_early_rt_after": None}))
            engine._phase3 = None
            self.assertFalse(engine.request_v92_rate_renegotiation(7))
        finally:
            chain.close()

    def test_test_only_rate_trigger_waits_for_v92_data_and_fires_once(self):
        with patch.dict("os.environ", {
                "EL_MODEM_TEST_V92_RENEGOTIATE_AFTER_MS": "20",
                "EL_MODEM_TEST_V92_RENEGOTIATE_DRN_UP": "8"}):
            chain = engine_for(self.pcm_profile())
        try:
            engine = chain.engine
            requests = []
            engine._phase3 = SimpleNamespace(
                stage="data", lapm=SimpleNamespace(v42_mode="lapm", read_tx_bits=None), transmit_ready=True,
                read_g711=lambda count, _law, **_: bytes([0xFF]) * count,
                request_rate_renegotiation=lambda drn, *, silent_period=False, silent_early_rt_after=None:
                    requests.append((drn, silent_period, silent_early_rt_after)))
            engine._phase3.lapm.v42_mode = "detect"
            for _ in range(5):
                engine.tx_g711(160, "pcmu")
            self.assertEqual(requests, [])
            self.assertIsNone(engine._test_v92_data_started_at)
            engine._phase3.lapm.v42_mode = "lapm"
            engine.tx_g711(160, "pcmu")  # begin data timer after LAPM connects
            self.assertEqual(requests, [])
            engine.tx_g711(160, "pcmu")  # 20 ms elapsed
            engine.tx_g711(160, "pcmu")  # one-shot
            self.assertEqual(requests, [(8, False, None)])
        finally:
            chain.close()

    def test_test_only_rate_trigger_can_select_silent_period(self):
        with patch.dict("os.environ", {
                "EL_MODEM_TEST_V92_RENEGOTIATE_AFTER_MS": "20",
                "EL_MODEM_TEST_V92_RENEGOTIATE_DRN_UP": "8",
                "EL_MODEM_TEST_V92_RENEGOTIATE_SILENT_PERIOD": "1"}):
            chain = engine_for(self.pcm_profile())
        try:
            engine = chain.engine
            requests = []
            engine._phase3 = SimpleNamespace(
                stage="data", lapm=SimpleNamespace(v42_mode="lapm", read_tx_bits=None), transmit_ready=True,
                read_g711=lambda count, _law, **_: bytes([0xFF]) * count,
                request_rate_renegotiation=lambda drn, *, silent_period=False, silent_early_rt_after=None:
                    requests.append((drn, silent_period, silent_early_rt_after)))
            engine.tx_g711(160, "pcmu")
            engine.tx_g711(160, "pcmu")
            self.assertEqual(requests, [(8, True, None)])
        finally:
            chain.close()

    def test_test_only_rate_trigger_can_select_early_silent_rt(self):
        with patch.dict("os.environ", {
                "EL_MODEM_TEST_V92_RENEGOTIATE_AFTER_MS": "20",
                "EL_MODEM_TEST_V92_RENEGOTIATE_DRN_UP": "8",
                "EL_MODEM_TEST_V92_RENEGOTIATE_SILENT_PERIOD": "1",
                "EL_MODEM_TEST_V92_RENEGOTIATE_EARLY_RT_AFTER": "4008"}):
            chain = engine_for(self.pcm_profile())
        try:
            engine = chain.engine
            requests = []
            engine._phase3 = SimpleNamespace(
                stage="data", lapm=SimpleNamespace(v42_mode="lapm", read_tx_bits=None), transmit_ready=True,
                read_g711=lambda count, _law, **_: bytes([0xFF]) * count,
                request_rate_renegotiation=lambda drn, *, silent_period=False, silent_early_rt_after=None:
                    requests.append((drn, silent_period, silent_early_rt_after)))
            engine.tx_g711(160, "pcmu")
            engine.tx_g711(160, "pcmu")
            self.assertEqual(requests, [(8, True, 4008)])
        finally:
            chain.close()

    def test_test_only_rate_trigger_rejects_a_non_v92_profile(self):
        with patch.dict("os.environ", {
                "EL_MODEM_TEST_V92_RENEGOTIATE_AFTER_MS": "0"}):
            chain = engine_for(profile())
        try:
            self.assertIn("v92-test-hook-ignored",
                          [name for name, _ in chain.engine.events])
        finally:
            chain.close()

    def test_v92_pcm_session_does_not_start_lapm_without_v8_agreement(self):
        p = self.pcm_profile(phase3_enable_lapm=True)
        chain = engine_for(p, protocols=0)
        try:
            session = chain.engine._phase3_factory(
                info1a={"v90_uinfo": 67}, law="pcmu", handoff_at=0,
                media_origin=100.0)
            self.assertFalse(session.enable_lapm)
        finally:
            chain.close()


if __name__ == "__main__":
    unittest.main()
