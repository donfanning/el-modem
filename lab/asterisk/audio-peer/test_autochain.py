"""V8Chain: automatic selection between V.34, V.32bis and V.22bis, and the legacy fallback."""
import struct
import unittest
import math
import threading
from unittest.mock import patch

import v8
from autochain import V8Chain
from v8 import V8_MOD_V21, V8_MOD_V22, V8_MOD_V32, V8_MOD_V34
from v8 import V8_PCM_ANALOG, V8_PCM_DIGITAL
from g711 import linear_to_ulaw, ulaw_to_linear


def loop(a, b, seconds, delay=40, until=None):
    to_a, to_b = [0] * delay, [0] * delay
    got_a, got_b = bytearray(), bytearray()
    for tick in range(int(seconds * 50)):
        sa, sb = a.tx_samples(160), b.tx_samples(160)
        to_a += sb
        to_b += sa
        pa, pb = to_a[:160], to_b[:160]
        del to_a[:160], to_b[:160]
        got_a += a.feed_pcm(struct.pack("<160h", *[int(max(-32768, min(32767, v))) for v in pa]))
        got_b += b.feed_pcm(struct.pack("<160h", *[int(max(-32768, min(32767, v))) for v in pb]))
        if until and until(a, b, tick):
            break
    return bytes(got_a), bytes(got_b)


def exchange(a, b, seconds, delay=40):
    state = {"sent": False}

    def until(a, b, tick):
        if not state["sent"] and a.is_settled() and b.is_settled():
            a.feed_bytes(b"caller says hi\r\n")
            b.feed_bytes(b"answerer says hi\r\n")
            state["sent"] = True
            state["at"] = tick
        return state["sent"] and tick > state["at"] + 400

    return loop(a, b, seconds, delay=delay, until=until)


@unittest.skipUnless(v8.available(), "numpy not installed: V.8 tones cannot be generated")
class ChainTests(unittest.TestCase):
    def test_no_v8_falls_back_legacy_v34_then_v32_then_v22(self):
        allow = V8_MOD_V34 | V8_MOD_V32 | V8_MOD_V22
        for calling in (True, False):
            with self.subTest(calling=calling):
                chain = V8Chain(calling, allow=allow)
                try:
                    # Status 4 is the ANSam-seen, CM/JM-incomplete case;
                    # only that case may enter legacy V.34 Phase 1.
                    chain._v8.status = 4
                    chain._t = 11 * 8000
                    chain._step_v8()
                    self.assertIsNone(chain.engine)
                    self.assertEqual(chain.kind, "v8")

                    chain._t = 12 * 8000
                    chain._step_v8()
                    self.assertEqual(chain.kind, "v34")
                    self.assertEqual(chain._no_v8_legacy_next_at, 24 * 8000)

                    chain._t = 24 * 8000
                    chain._advance_no_v8_legacy_fallback()
                    self.assertEqual(chain.kind, "v32")
                    self.assertEqual(chain._no_v8_legacy_next_at, 36 * 8000)

                    chain._t = 36 * 8000
                    chain._advance_no_v8_legacy_fallback()
                    self.assertEqual(chain.kind, "v22")
                    self.assertEqual(chain._no_v8_legacy_stage, 0)
                finally:
                    chain.close()

    def test_no_ansam_skips_legacy_v34_and_uses_v32_or_v22(self):
        for calling, expected in ((True, "v22"), (False, "v32")):
            with self.subTest(calling=calling):
                chain = V8Chain(calling, allow=V8_MOD_V34 | V8_MOD_V32 | V8_MOD_V22)
                try:
                    chain._v8.status = 3  # no ANSam / no CM
                    chain._t = 12 * 8000
                    chain._step_v8()
                    self.assertEqual(chain.kind, expected)
                    self.assertNotEqual(chain.kind, "v34")
                    if expected == "v32":
                        self.assertEqual(chain._legacy_probe_at, chain._t)
                finally:
                    chain.close()

    def test_plain_ans_uses_v22_after_ac_wait_and_gates_answer_tone(self):
        chain = V8Chain(True, allow=V8_MOD_V34 | V8_MOD_V32 | V8_MOD_V22)
        try:
            chain._v8.status = 0
            chain._v8.plain_ans = True
            chain._t = int((chain.patience + .1) * 8000)
            chain._step_v8()
            self.assertEqual(chain.kind, "v22")
            self.assertIsNotNone(chain._tone_gate)
            self.assertEqual(chain._no_v8_legacy_stage, 0)
        finally:
            chain.close()

    def test_real_plain_ans_after_pcmu_still_selects_legacy_v22(self):
        import numpy as np

        chain = V8Chain(True, allow=V8_MOD_V34 | V8_MOD_V22)
        try:
            for tick in range(400):
                n = tick * 160 + np.arange(160)
                reversal = (n // int(0.450 * v8.FS)) % 2
                frame = 2500.0 * np.sin(2 * math.pi * 2100.0 * n / v8.FS + math.pi * reversal)
                pcm = [ulaw_to_linear(linear_to_ulaw(int(round(float(sample))))) for sample in frame]
                chain.feed_pcm(struct.pack("<160h", *pcm))
                if chain.kind != "v8":
                    break

            self.assertEqual(chain.kind, "v22")
            self.assertIsNotNone(chain.engine)
            self.assertTrue(any("plain ANS (no V.8 partner)" in message
                                for _, message in chain.events))
        finally:
            chain.close()

    def test_late_legacy_v34_deadline_is_relative_to_actual_start(self):
        chain = V8Chain(True, allow=V8_MOD_V34 | V8_MOD_V32 | V8_MOD_V22)
        try:
            chain._v8.status = 4
            chain._t = 15 * 8000
            chain._step_v8()
            self.assertEqual(chain.kind, "v34")
            self.assertEqual(chain._no_v8_legacy_next_at, 27 * 8000)
        finally:
            chain.close()

    def test_ansam_with_missed_cm_does_not_skip_legacy_v34(self):
        chain = V8Chain(True, allow=V8_MOD_V34 | V8_MOD_V32 | V8_MOD_V22)
        try:
            chain._v8.status = 4
            chain._v8._stage = "cm"  # ANSam was recognized; JM/CM response was missed.
            chain._t = 8 * 8000
            chain._step_v8()
            self.assertEqual(chain.kind, "v8")
            self.assertIsNone(chain.engine)

            chain._t = 12 * 8000
            chain._step_v8()
            self.assertEqual(chain.kind, "v34")
            self.assertTrue(any("starting legacy V.34" in str(event)
                                for event in chain.events))
        finally:
            chain.close()

    def test_v8_pcm_role_requires_engine_before_it_can_be_advertised(self):
        with self.assertRaisesRegex(ValueError, "requires a V.90 engine factory"):
            V8Chain(False, pcm_availability=V8_PCM_DIGITAL)

    def test_v90_digital_answerer_marks_digital_network_access(self):
        for pcm, expected in ((V8_PCM_DIGITAL, v8.V8_PSTN_ACCESS_DIGITAL_NETWORK),
                              (0, 0)):
            with self.subTest(pcm=pcm):
                chain = V8Chain(False, allow=V8_MOD_V34, pcm_availability=pcm,
                                v90_engine_factory=(lambda **_kwargs: object())
                                if pcm else None)
                try:
                    self.assertEqual(chain._v8.pstn_access_bits, expected)
                finally:
                    chain.close()

    def test_v90_factory_must_supply_exact_raw_media_interfaces(self):
        chain = V8Chain(False, allow=V8_MOD_V34,
                        pcm_availability=V8_PCM_DIGITAL,
                        v90_engine_factory=lambda **_kwargs: object())
        chain._v8.status = 2
        chain._v8.result = {
            "modulations": V8_MOD_V34,
            "protocols": 0,
            "pcm_negotiated": True,
            "pcm_negotiated_bits": V8_PCM_DIGITAL,
        }
        try:
            with self.assertRaisesRegex(TypeError, "incomplete engine"):
                chain._step_v8()
        finally:
            chain.close()

    def test_v8_pcm_negotiation_selects_role_specific_engine(self):
        class Engine:
            events = []

            def tx_samples(self, count):
                return [0] * count

            def feed_pcm(self, _pcm):
                return b""

            def feed_bytes(self, _data):
                pass

            def tx_g711(self, count, codec):
                return bytes([0xff if codec == "pcmu" else 0xd5]) * count

            def feed_g711(self, payload, _codec, _context):
                return bytes(payload)

            def feed_gap(self, _samples, _codec, _context):
                return b""

        for calling, local_pcm, negotiated_pcm, expected_role in (
                (False, V8_PCM_DIGITAL, V8_PCM_DIGITAL, "digital-answerer"),
                (True, V8_PCM_ANALOG, V8_PCM_ANALOG, "analogue-caller")):
            with self.subTest(role=expected_role):
                selected = []

                def factory(*, role, negotiation, media_at, media_origin):
                    selected.append((role, negotiation["pcm_negotiated_bits"],
                                     media_at, media_origin))
                    return Engine()

                chain = V8Chain(calling, allow=V8_MOD_V34,
                                pcm_availability=local_pcm,
                                v90_engine_factory=factory)
                chain._v8.status = 2
                chain._v8.result = {
                    "modulations": V8_MOD_V34,
                    "protocols": 0,
                    "pcm_negotiated": True,
                    "pcm_negotiated_bits": negotiated_pcm,
                }
                chain._step_v8()
                self.assertEqual(chain.kind, "v90")
                self.assertEqual(selected[0][:3], (expected_role, negotiated_pcm, 0))
                self.assertGreater(selected[0][3], 0.0)
                self.assertEqual(chain.tx_g711(3, "pcmu"), bytes([0xff]) * 3)
                self.assertEqual(chain.feed_g711(b"raw", "pcmu", {}), b"raw")
                chain.close()

    def test_v90_handoff_uses_rtp_transmit_media_cursor(self):
        class Engine:
            events = []
            def tx_samples(self, count): return [0] * count
            def feed_pcm(self, _pcm): return b""
            def feed_bytes(self, _data): pass
            def tx_g711(self, count, codec): return bytes([0xff]) * count
            def feed_g711(self, payload, _codec, _context): return bytes(payload)
            def feed_gap(self, _samples, _codec, _context): return b""

        selected = []
        chain = V8Chain(False, allow=V8_MOD_V34,
                        pcm_availability=V8_PCM_DIGITAL,
                        v90_engine_factory=lambda **kwargs: (selected.append(kwargs) or Engine()))
        try:
            chain.set_tx_media_deadline(50.0)
            chain.tx_samples(160)
            chain._v8.status = 2
            chain._v8.result = {
                "modulations": V8_MOD_V34,
                "protocols": 0,
                "pcm_negotiated_bits": V8_PCM_DIGITAL,
            }
            chain._step_v8()
            self.assertEqual(selected[0]["media_at"], 160)
            self.assertAlmostEqual(selected[0]["media_origin"], 50.02)
        finally:
            chain.close()

    def test_v90_enabled_answerer_uses_v34_when_pcm_role_is_not_negotiated(self):
        """An opt-in digital-answerer profile must preserve the V.34 fallback."""
        factory_calls = []

        def v90_factory(**kwargs):
            factory_calls.append(kwargs)
            raise AssertionError("V.90 engine must not start without a negotiated PCM role")

        caller = V8Chain(True, allow=V8_MOD_V34, v34_rate=9600)
        chain = V8Chain(False, allow=V8_MOD_V34,
                        v34_rate=9600,
                        pcm_availability=V8_PCM_DIGITAL,
                        v90_engine_factory=v90_factory)
        try:
            caller_rx, answerer_rx = exchange(caller, chain, 90)

            self.assertEqual(caller.kind, "v34")
            self.assertEqual(chain.kind, "v34")
            self.assertIsNotNone(chain.engine)
            self.assertEqual(factory_calls, [])
            self.assertEqual(caller_rx, b"answerer says hi\r\n")
            self.assertEqual(answerer_rx, b"caller says hi\r\n")
        finally:
            caller.close()
            chain.close()

    def test_unavailable_negotiated_pcm_role_does_not_fall_back_to_v34(self):
        """Never reinterpret a peer's already-negotiated digital PCM as voiceband."""
        chain = V8Chain(False, allow=V8_MOD_V34)
        try:
            chain._v8.status = 2
            chain._v8.result = {
                "modulations": V8_MOD_V34,
                "protocols": 0,
                "pcm_negotiated": True,
                "pcm_negotiated_bits": V8_PCM_DIGITAL,
            }

            chain._step_v8()

            self.assertEqual(chain.kind, "v90-unavailable")
            self.assertTrue(chain._failed)
            self.assertIsNone(chain.engine)
        finally:
            chain.close()

    def test_v8_selects_v21_and_exchanges_exact_payload_in_both_roles(self):
        for calling in (True, False):
            with self.subTest(calling=calling):
                caller = V8Chain(calling, allow=V8_MOD_V21)
                answer = V8Chain(not calling, allow=V8_MOD_V21)
                ga, gb = exchange(caller, answer, 20)
                self.assertEqual((caller.kind, answer.kind), ("v21", "v21"))
                self.assertEqual((caller.bit_rate, answer.bit_rate), (300, 300))
                self.assertEqual(gb, b"caller says hi\r\n")
                self.assertEqual(ga, b"answerer says hi\r\n")
                self.assertFalse(caller.supports_v42)
                self.assertFalse(answer.supports_v42)

    def test_selected_v34_receive_does_not_block_transmit_but_close_waits(self):
        entered, release, transmitted, closed = (threading.Event() for _ in range(4))
        class Engine:
            events=[]
            def feed_pcm(self,pcm):
                entered.set()
                if not release.wait(3):raise TimeoutError('test receive release')
                return b'received'
            def tx_samples(self,count):
                transmitted.set()
                return [7]*count
            def close(self):closed.set()
        chain=V8Chain(True)
        chain.engine,chain.kind=Engine(),'v34'
        got=[]
        rx=threading.Thread(target=lambda:got.append(chain.feed_pcm(b'pcm')))
        tx=threading.Thread(target=lambda:got.append(chain.tx_samples(2)))
        shutdown=threading.Thread(target=chain.close)
        rx.start()
        try:
            self.assertTrue(entered.wait(1))
            tx.start()
            self.assertTrue(transmitted.wait(1),'RX held the transmit dispatch lock')
            shutdown.start()
            self.assertFalse(closed.wait(.05),'shutdown raced active receive')
        finally:
            release.set()
            for thread in (rx,tx,shutdown):
                if thread.ident is not None:thread.join(3)
        self.assertTrue(closed.is_set())
        self.assertIn(b'received',got)
        self.assertIn([7,7],got)

    def test_explicit_v34_call_ceiling_wins_over_process_default_in_both_roles(self):
        for calling in (True, False):
            with self.subTest(calling=calling), patch.dict('os.environ', {'V34_RATE':'33600'}):
                chain=V8Chain(calling,v34_rate=24000)
                chain._choose('v34')
                core=chain.engine.caller if calling else chain.engine.answerer
                self.assertEqual(core.data_rate,24000)
                if hasattr(chain.engine,'_v8') and chain.engine._v8 is not None:
                    chain.engine._v8.close()

    def test_tx_trace_attributes_lock_wait_without_changing_pcm(self):
        class Engine:
            events=[]
            def tx_samples(self,count):return list(range(count))
        with patch.dict('os.environ',{'EL_MODEM_TIMING_TRACE':'1'}):
            chain=V8Chain(True)
        chain.engine=Engine()
        held=threading.Event()
        release=threading.Event()
        def hold():
            with chain._lock:
                held.set()
                release.wait(1)
        worker=threading.Thread(target=hold)
        worker.start()
        self.assertTrue(held.wait(1))
        timer=threading.Timer(.03,release.set)
        timer.start()
        try:
            self.assertEqual(chain.tx_samples(160),list(range(160)))
        finally:
            release.set();worker.join(1);timer.join(1)
        timing=chain._tx_timing
        self.assertGreater(timing['lock_wait_seconds'],.015)
        self.assertLess(timing['lock_wait_cpu_seconds'],timing['lock_wait_seconds'])
        self.assertLess(timing['engine_seconds'],timing['lock_wait_seconds'])

    def test_legacy_usb1_selects_before_remote_timeout(self):
        import v22
        a = V8Chain(True, v22_rate=2400)
        b = v22.V22Modem(False, 2400, ans_tone_ms=0)
        loop(a, b, 1)
        self.assertEqual(a.kind, "v22")
        self.assertTrue(any("USB1 detected" in event[1] for event in a.events))
        ga, gb = exchange(a, b, 25)
        self.assertEqual(gb, b"caller says hi\r\n")
        self.assertEqual(ga, b"answerer says hi\r\n")

    def test_usb1_rejects_other_startup_tones_and_short_bursts(self):
        for frequency in (2100, 2225, 2400, 1800):
            with self.subTest(frequency=frequency):
                a = V8Chain(True)
                for tick in range(50):
                    a.tx_samples(160)
                    x = [int(2000 * math.sin(2 * math.pi * frequency * (tick * 160 + n) / 8000))
                         for n in range(160)]
                    a.feed_pcm(struct.pack("<160h", *x))
                self.assertEqual(a.kind, "v8")
        a = V8Chain(True)
        for tick in range(30):
            a.tx_samples(160)
            x = [int(2000 * math.sin(2 * math.pi * 2250 * (tick * 160 + n) / 8000))
                 if tick < 10 else 0 for n in range(160)]
            a.feed_pcm(struct.pack("<160h", *x))
        self.assertEqual(a.kind, "v8")

    def test_usb1_respects_mask_and_established_v8_offer(self):
        for allow, status in ((V8_MOD_V34, 0), (V8_MOD_V22 | V8_MOD_V34, 1)):
            a = V8Chain(True, allow=allow)
            a._v8.status = status
            for tick in range(30):
                a.tx_samples(160)
                x = [int(2000 * math.sin(2 * math.pi * 2250 * (tick * 160 + n) / 8000))
                     for n in range(160)]
                a.feed_pcm(struct.pack("<160h", *x))
            self.assertEqual(a.kind, "v8")

    def test_legacy_v32_both_roles_exchange(self):
        from v32_stream import V32Modem
        from v32_answer import V32Answerer
        for calling in (True, False):
            with self.subTest(calling=calling):
                chain = V8Chain(calling, allow=V8_MOD_V32 | V8_MOD_V22,
                                v32_rates=(9600,))
                peer = (V32Answerer if calling else V32Modem)(
                    fs=8000, my_rates=(9600,), use_v8=False)
                a, b = (chain, peer) if calling else (peer, chain)
                ga, gb = exchange(a, b, 45)
                self.assertEqual(chain.kind, "v32")
                self.assertEqual(gb, b"caller says hi\r\n")
                self.assertEqual(ga, b"answerer says hi\r\n")
                self.assertTrue(any("legacy V.32" in e[1] for e in chain.events))

    def test_legacy_v32_14400_with_half_second_round_trip(self):
        from v32_stream import V32Modem
        from v32_answer import V32Answerer
        for calling in (True, False):
            with self.subTest(calling=calling):
                chain = V8Chain(calling, allow=V8_MOD_V32 | V8_MOD_V22,
                                v32_rates=(14400,))
                peer = (V32Answerer if calling else V32Modem)(
                    fs=8000, my_rates=(14400,), use_v8=False)
                a, b = (chain, peer) if calling else (peer, chain)
                ga, gb = exchange(a, b, 45, delay=2000)
                self.assertEqual(chain.kind, "v32")
                self.assertEqual((a.bit_rate, b.bit_rate), (14400, 14400))
                self.assertEqual(gb, b"caller says hi\r\n")
                self.assertEqual(ga, b"answerer says hi\r\n")

    def test_ac_detector_chunking_and_rejections(self):
        from autochain import _LegacyAC
        for chunk in (64, 160, 320):
            detector = _LegacyAC()
            detected = False
            for start in range(0, 1280, chunk):
                samples = [1000 * (math.cos(2*math.pi*600*n/8000) +
                                   math.cos(2*math.pi*3000*n/8000))
                           for n in range(start, min(start+chunk, 1280))]
                detected |= detector.feed(samples)
            self.assertTrue(detected)
        for frequencies in ((600,), (3000,), (2100,), (2250,), (600, 2400)):
            detector = _LegacyAC()
            samples = [sum(1000*math.cos(2*math.pi*f*n/8000) for f in frequencies)
                       for n in range(1600)]
            self.assertFalse(detector.feed(samples))
        detector = _LegacyAC()
        self.assertFalse(detector.feed([1000*(math.cos(2*math.pi*600*n/8000) +
                                             math.cos(2*math.pi*3000*n/8000))
                                        for n in range(640)]))
        self.assertFalse(detector.feed([0]*1280))

    def test_ac_respects_mask_and_established_offer(self):
        for allow, status in ((V8_MOD_V22, 0), (V8_MOD_V32 | V8_MOD_V22, 1)):
            a = V8Chain(True, allow=allow)
            a._v8.status = status
            for tick in range(12):
                a.tx_samples(160)
                x = [int(1000*(math.cos(2*math.pi*600*n/8000) +
                               math.cos(2*math.pi*3000*n/8000)))
                     for n in range(tick*160, (tick+1)*160)]
                a.feed_pcm(struct.pack("<160h", *x))
            self.assertEqual(a.kind, "v8")

    def test_ac_resolves_unanswered_cm(self):
        a = V8Chain(True, v32_rates=(9600,))
        a._v8.status, a._v8._stage = 1, "cm"
        for tick in range(12):
            a.tx_samples(160)
            x = [int(1000*(math.cos(2*math.pi*600*n/8000) +
                           math.cos(2*math.pi*3000*n/8000)))
                 for n in range(tick*160, (tick+1)*160)]
            a.feed_pcm(struct.pack("<160h", *x))
        self.assertEqual(a.kind, "v32")

    def test_v32_only_probe_does_not_select_disallowed_v22(self):
        a = V8Chain(False, allow=V8_MOD_V32, patience=.1)
        for _ in range(130):
            a.tx_samples(160)
            a.feed_pcm(bytes(320))
        self.assertEqual(a.kind, "v32")

    def test_probe_preserves_queued_payload_on_v22_fallback(self):
        a = V8Chain(False, allow=V8_MOD_V32 | V8_MOD_V22, patience=.1)
        a.feed_bytes(b"queued before selection")
        for _ in range(120):
            a.tx_samples(160)
            a.feed_pcm(bytes(320))
        self.assertEqual(a.kind, "v22")
        self.assertGreater(a.pending_bits(), 0)
        self.assertEqual(a._queued, b"")
        self.assertTrue(any("probe unanswered" in e[1] for e in a.events))

    def test_v22_when_only_v22_is_common(self):
        a = V8Chain(True, allow=V8_MOD_V22, v22_rate=2400)
        b = V8Chain(False, allow=V8_MOD_V22, v22_rate=2400)
        ga, gb = exchange(a, b, 40)
        self.assertEqual((a.kind, b.kind), ("v22", "v22"))
        self.assertEqual(gb, b"caller says hi\r\n")
        self.assertEqual(ga, b"answerer says hi\r\n")

    def test_v32_beats_v22(self):
        from v8 import V8_PROTOCOL_LAPM_V42
        a = V8Chain(True, allow=V8_MOD_V32 | V8_MOD_V22, v32_rates=(9600,),
                    protocols=V8_PROTOCOL_LAPM_V42)
        b = V8Chain(False, allow=V8_MOD_V32 | V8_MOD_V22, v32_rates=(9600,),
                    protocols=V8_PROTOCOL_LAPM_V42)
        ga, gb = exchange(a, b, 60)
        self.assertEqual((a.kind, b.kind), ("v32", "v32"))
        self.assertEqual((a.negotiated_protocols, b.negotiated_protocols),
                         (V8_PROTOCOL_LAPM_V42, V8_PROTOCOL_LAPM_V42))
        self.assertEqual(gb, b"caller says hi\r\n")
        self.assertEqual(ga, b"answerer says hi\r\n")

    def test_v34_chosen_when_both_offer_it(self):
        a = V8Chain(True, v34_rate=9600)
        b = V8Chain(False, v34_rate=9600)
        ga, gb = exchange(a, b, 90)
        self.assertEqual((a.kind, b.kind), ("v34", "v34"))
        self.assertEqual(gb, b"caller says hi\r\n")
        self.assertEqual(ga, b"answerer says hi\r\n")

    def test_caller_offering_more_is_limited_by_answerer(self):
        a = V8Chain(True, allow=V8_MOD_V34 | V8_MOD_V32 | V8_MOD_V22, v32_rates=(9600,))
        b = V8Chain(False, allow=V8_MOD_V32 | V8_MOD_V22, v32_rates=(9600,))
        exchange(a, b, 60)
        self.assertEqual((a.kind, b.kind), ("v32", "v32"))

    def test_legacy_v22_answerer_gets_fallback(self):
        import v22
        a = V8Chain(True, v22_rate=2400, patience=3.0)
        b = v22.V22Modem(False, 2400, ans_tone_ms=0)
        ga, gb = exchange(a, b, 30)
        self.assertEqual(a.kind, "v22")
        self.assertEqual(gb, b"caller says hi\r\n")
        self.assertEqual(ga, b"answerer says hi\r\n")

    def test_plain_ans_starts_v22_before_the_v8_timeout(self):
        import v22
        # The V.22bis-only automode (line_from_ms for V22B): plain ANS must beat the 6 s patience.
        a = V8Chain(True, v22_rate=2400, allow=V8_MOD_V22)
        b = v22.V22Modem(False, 2400, ans_tone_ms=3000)           # V.25 answer tone for 3 s, then V.22bis
        ga, gb = exchange(a, b, 40)
        chosen = [t for t, text in a.events if text == "line mode: v22"]
        self.assertTrue(chosen and chosen[0] < 4.0, a.events)     # not the 12 s V.8 timeout
        self.assertEqual(a.kind, "v22")
        self.assertEqual(gb, b"caller says hi\r\n")
        self.assertEqual(ga, b"answerer says hi\r\n")

    def test_silence_to_plain_ans_onset_is_not_ansam(self):
        # The onset window of a plain 2100 Hz tone looks AM-deep; it must not latch ANSam.
        n = v8.V8Negotiator(True, modulations=V8_MOD_V22)
        pcm = [0] * 4000 + [int(3000 * math.sin(2 * math.pi * 2100 * i / 8000)) for i in range(8000 * 6)]
        for i in range(0, len(pcm), 160):
            n.rx(pcm[i:i + 160])
        self.assertTrue(n.plain_ans)
        self.assertNotEqual(n.status, 1)                          # 1 = ANSam accepted

    def test_legacy_v22_caller_gets_fallback(self):
        import v22
        a = v22.V22Modem(True, 2400)
        b = V8Chain(False, v22_rate=2400, patience=3.0)
        ga, gb = exchange(a, b, 30)
        self.assertEqual(b.kind, "v22")
        self.assertEqual(gb, b"caller says hi\r\n")
        self.assertEqual(ga, b"answerer says hi\r\n")


if __name__ == "__main__":
    unittest.main()


class NoCommonModulationTests(unittest.TestCase):
    """V.8 completing with nothing in common must not leave the line without an engine (found in review)."""

    class _Done:
        status = 2
        def __init__(self, mods): self.result = {'modulations': mods}
        def close(self): pass

    def chain(self, allow):
        c = V8Chain(True, allow=allow)
        c._v8 = self._Done(V8_MOD_V34)                     # peer offers V.34 only
        return c

    def test_falls_back_to_v22_when_allowed(self):
        c = self.chain(V8_MOD_V32 | V8_MOD_V22)
        c._step_v8()
        self.assertEqual(c.kind, 'v22')
        self.assertIsNotNone(c.engine)

    def test_without_v22_it_logs_once_not_every_block(self):
        c = self.chain(V8_MOD_V32)
        for _ in range(5):
            c._step_v8()
        self.assertIsNone(c.engine)
        self.assertEqual(sum('without an allowed common modulation' in str(e) for e in c.events), 1)
