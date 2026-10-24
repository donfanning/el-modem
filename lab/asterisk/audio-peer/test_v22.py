#!/usr/bin/env python3
"""V.22 / V.22bis (v22.V22Modem): our modem against itself and against SpanDSP's as an independent reference."""
import random
import struct
import time
import unittest

try:
    import numpy  # noqa: F401
except ImportError:
    numpy = None

if numpy is not None:                     # v22.py's signal chain is built on numpy
    import v22
    from modem_dce import AtDce
    from v34_online_control import OnlineControl
    from audio_path_impairment import AudioPathImpairment
    from v22_audio_loop import V22SoftwareAudioLoop


def link(a, b, seconds=12.0, delay=40, noise=0.0, ppm=0.0, seed=1, until=None):
    # random.Random rather than numpy: this suite must run on a bare checkout, and the noise
    # source does not need numpy's determinism guarantees to be reproducible.
    rng = random.Random(seed)
    to_a, to_b = [0] * delay, [0] * delay
    got_a, got_b = bytearray(), bytearray()
    for tick in range(int(seconds * 50)):
        sa, sb = a.tx_samples(160), b.tx_samples(160)
        to_a += sb
        to_b += sa
        xa, xb = to_a[:160], to_b[:160]
        del to_a[:160], to_b[:160]
        if noise:
            xa = [v + rng.gauss(0, noise) for v in xa]
            xb = [v + rng.gauss(0, noise) for v in xb]
        got_a += a.feed_pcm(struct.pack("<160h", *[int(max(-32768, min(32767, v))) for v in xa]))
        got_b += b.feed_pcm(struct.pack("<160h", *[int(max(-32768, min(32767, v))) for v in xb]))
        if until and until(a, b, tick):
            break
    return bytes(got_a), bytes(got_b)


@unittest.skipIf(numpy is None, "numpy not installed")
class SelfTests(unittest.TestCase):
    def test_software_audio_loop_runs_v22_v22bis_and_bell212a_over_pcmu(self):
        caller_payload = b"caller\x00\xff V22"
        answer_payload = b"answer\xff\x00 V22"
        for rate, profile in ((600, "v22"), (1200, "v22"),
                              (1200, "bell212a"), (2400, "v22bis")):
            for calling_party in (True, False):
                with self.subTest(rate=rate, profile=profile,
                                  calling_party=calling_party):
                    impairment = AudioPathImpairment(
                        gain_db=3, dc_offset=256, noise_rms=64,
                        clip_level=32767, seed=7)
                    loop = V22SoftwareAudioLoop(
                        bit_rate=rate, profile=profile,
                        calling_party=calling_party,
                        local_to_peer_impairment=impairment,
                        peer_to_local_impairment=AudioPathImpairment(
                            gain_db=3, dc_offset=-256, noise_rms=64,
                            clip_level=10000, seed=9))
                    loop.feed_bytes(caller_payload)
                    loop.feed_peer_bytes(answer_payload)
                    got_local, got_peer = bytearray(), bytearray()
                    for _ in range(250):
                        local_rx, peer_rx = loop.step_duplex()
                        got_local.extend(local_rx)
                        got_peer.extend(peer_rx)
                        if (answer_payload in got_local
                                and caller_payload in got_peer):
                            break
                    self.assertIn(answer_payload, got_local)
                    self.assertIn(caller_payload, got_peer)
                    status = loop.status()
                    self.assertTrue(status["endpoint_settled"])
                    self.assertTrue(status["peer_settled"])
                    self.assertGreater(impairment.report()["samples"], 0)
                    loop.close()

    def test_v22bis_line_signal_return_clamps_until_retrain_window_expires(self):
        modem = v22.V22Modem(True, 2400, profile="v22bis")
        modem.connected = True
        modem._carrier = True
        modem._rx_buf[:] = [1, 0, 1]
        modem._deframe.buf[:] = [0, 1, 0]

        modem._carrier_gone()
        self.assertTrue(modem._rx_data_clamped)
        self.assertEqual(modem._rx_buf, [])
        self.assertEqual(modem._deframe.buf, [])
        modem._rx_ready = True
        modem._on_bit(1)
        self.assertEqual(modem._rx_buf, [])

        modem._rb.carrier = True
        modem._on_symbol(1 + 0j)
        deadline = modem._rx_clamp_until_symbol
        self.assertIsNotNone(deadline)
        modem._rb.symbols = deadline - 1
        modem._release_data_clamp_after_recovery_window()
        self.assertTrue(modem._rx_data_clamped)

        modem._rb.symbols = deadline
        modem._release_data_clamp_after_recovery_window()
        self.assertFalse(modem._rx_data_clamped)
        modem._on_bit(1)
        self.assertTrue(modem._rx_buf)

    def test_v22bis_returned_retrain_pattern_keeps_data_clamped(self):
        modem = v22.V22Modem(True, 2400)
        modem.connected = True
        modem._rx_data_clamped = True
        modem._rx_clamp_until_symbol = 100
        modem._rx_retrain_pending = True
        modem._rb.symbols = 100
        modem._release_data_clamp_after_recovery_window()
        self.assertTrue(modem._rx_data_clamped)

    def test_v22bis_retrain_uses_1200_then_2400_sequence_and_requalifies(self):
        modem = v22.V22Modem(True, 2400, profile="v22bis")
        modem.connected = True
        modem._rate = 2400
        modem._state = "bis"
        modem._rx_ready = True
        modem._start_retrain()
        self.assertEqual(modem._tx_mode, "s1")
        self.assertEqual(modem._tx_rate_now, 1200)

        for _ in range(60):
            modem._next_symbol()
        self.assertEqual((modem._tx_mode, modem._tx_rate_now), ("s1", 1200))
        modem._next_symbol()
        self.assertEqual((modem._tx_mode, modem._tx_rate_now), ("scr1", 1200))
        for _ in range(299):
            modem._next_symbol()
        self.assertEqual((modem._tx_mode, modem._tx_rate_now), ("scr1", 1200))
        modem._next_symbol()
        self.assertEqual((modem._tx_mode, modem._tx_rate_now), ("scr1", 2400))
        for _ in range(119):
            modem._next_symbol()
        self.assertEqual((modem._tx_mode, modem._tx_rate_now), ("scr1", 2400))
        modem._next_symbol()
        self.assertEqual((modem._tx_mode, modem._tx_rate_now), ("data", 2400))

        modem._qam_from = 0
        modem._rb.symbols = 1
        modem._qam_ones = 32
        modem._after_symbol(0b01, 4)
        self.assertEqual(modem._state, "bis")
        self.assertTrue(modem._rx_ready)
        self.assertFalse(modem._rx_data_clamped)
        self.assertIn(("RETRAIN_SUCCEEDED", 2400), modem.events)

    def test_v22bis_one_sided_retrain_requalifies_pair_and_resumes_payload(self):
        caller = v22.V22Modem(True, 2400, profile="v22bis")
        answerer = v22.V22Modem(False, 2400, profile="v22bis")
        # Establish the production waveform state before injecting the remote
        # S1/retrain request. This is a paired DSP check, not independent-peer
        # or RTP/DCE qualification.
        for _ in range(500):
            tx_caller, tx_answerer = caller.tx_samples(160), answerer.tx_samples(160)
            caller.feed_pcm(struct.pack("<160h", *tx_answerer))
            answerer.feed_pcm(struct.pack("<160h", *tx_caller))
        self.assertTrue(caller.connected and answerer.connected)

        caller._start_retrain()
        success = False
        for _ in range(100):
            tx_caller, tx_answerer = caller.tx_samples(160), answerer.tx_samples(160)
            caller.feed_pcm(struct.pack("<160h", *tx_answerer))
            answerer.feed_pcm(struct.pack("<160h", *tx_caller))
            success = (
                ("RETRAIN_SUCCEEDED", 2400) in caller.events
                and ("RETRAIN_SUCCEEDED", 2400) in answerer.events
            )
            if success:
                break
        self.assertTrue(success)
        self.assertEqual((caller._state, answerer._state), ("bis", "bis"))

        sent_caller = b"caller-after-retrain-" + bytes(range(32))
        sent_answerer = b"answer-after-retrain-" + bytes(range(32))
        caller.feed_bytes(sent_caller)
        answerer.feed_bytes(sent_answerer)
        received_caller, received_answerer = bytearray(), bytearray()
        for _ in range(150):
            tx_caller, tx_answerer = caller.tx_samples(160), answerer.tx_samples(160)
            received_caller.extend(caller.feed_pcm(struct.pack("<160h", *tx_answerer)))
            received_answerer.extend(answerer.feed_pcm(struct.pack("<160h", *tx_caller)))
        self.assertEqual(bytes(received_caller), sent_answerer)
        self.assertEqual(bytes(received_answerer), sent_caller)

    def test_v22bis_optional_rate_change_down_and_up_resumes_payload(self):
        caller = v22.V22Modem(True, 2400, profile="v22bis")
        answerer = v22.V22Modem(False, 2400, profile="v22bis")
        for _ in range(500):
            tx_caller, tx_answerer = caller.tx_samples(160), answerer.tx_samples(160)
            caller.feed_pcm(struct.pack("<160h", *tx_answerer))
            answerer.feed_pcm(struct.pack("<160h", *tx_caller))
        self.assertTrue(caller.connected and answerer.connected)
        online_control = OnlineControl(lambda: caller)
        dce = AtDce()
        dce.feed(b"ATE0V1X4\r")
        dce.online = True
        dce.attach_online_control(online_control.request)
        dce.attach_resume_online(lambda: None)

        for target, parameter in ((1200, 4), (2400, 5)):
            self.assertEqual(dce.feed(f"ATO{parameter}\r".encode()), b"")
            self.assertTrue(dce.online_return_pending)
            for _ in range(125):
                tx_caller, tx_answerer = caller.tx_samples(160), answerer.tx_samples(160)
                caller.feed_pcm(struct.pack("<160h", *tx_answerer))
                answerer.feed_pcm(struct.pack("<160h", *tx_caller))
            self.assertEqual((caller.bit_rate, answerer.bit_rate), (target, target),
                             (caller.events[-12:], answerer.events[-12:]))
            self.assertTrue(caller.is_settled() and answerer.is_settled())
            self.assertIn(("RATE_CHANGE_SUCCEEDED", target), caller.events)
            self.assertIn(("RATE_CHANGE_SUCCEEDED", target), answerer.events)
            result = online_control.poll()
            self.assertEqual(result["status"], "rate-exchanged")
            self.assertEqual((result["tx_rate"], result["rx_rate"]), (target, target))
            dce.finish_online_control(target, "")
            self.assertIn(f"CONNECT {target}".encode(), dce.take_notifications())
            sent_caller = bytes((i * 29 + target) & 0xff for i in range(48))
            sent_answerer = bytes((i * 17 + target) & 0xff for i in range(48))
            caller.feed_bytes(sent_caller)
            answerer.feed_bytes(sent_answerer)
            received_caller, received_answerer = bytearray(), bytearray()
            for _ in range(100):
                tx_caller, tx_answerer = caller.tx_samples(160), answerer.tx_samples(160)
                received_caller.extend(caller.feed_pcm(struct.pack("<160h", *tx_answerer)))
                received_answerer.extend(answerer.feed_pcm(struct.pack("<160h", *tx_caller)))
            self.assertEqual(bytes(received_caller), sent_answerer)
            self.assertEqual(bytes(received_answerer), sent_caller)

    def test_v22bis_optional_rate_change_retries_when_peer_s1_is_absent(self):
        modem = v22.V22Modem(True, 2400, profile="v22bis")
        modem.connected = True
        modem._state = "bis"
        modem._rate = 2400
        modem._rx_ready = True
        modem.request_rate_change(1200)
        for _ in range(65):
            modem.tx_samples(160)
        self.assertIn(("RATE_CHANGE_RETRY", 1), modem.events)
        self.assertEqual(modem._rate_change_attempts, 2)
        self.assertEqual(modem._rate_control, "initiator_wait_s1")

    def test_v22bis_online_rate_change_updates_lapm_clock_and_payload(self):
        caller = v22.V22Modem(True, 2400, profile="v22bis")
        answerer = v22.V22Modem(False, 2400, profile="v22bis")
        for _ in range(500):
            tx_caller, tx_answerer = caller.tx_samples(160), answerer.tx_samples(160)
            caller.feed_pcm(struct.pack("<160h", *tx_answerer))
            answerer.feed_pcm(struct.pack("<160h", *tx_caller))
        self.assertTrue(caller.connected and answerer.connected)

        caller.start_v42(True, detect=False, rate=2400)
        answerer.start_v42(False, detect=False, rate=2400)
        for _ in range(200):
            tx_caller, tx_answerer = caller.tx_samples(160), answerer.tx_samples(160)
            caller.feed_pcm(struct.pack("<160h", *tx_answerer))
            answerer.feed_pcm(struct.pack("<160h", *tx_caller))
            if caller.v42_mode == answerer.v42_mode == "lapm":
                break
        self.assertEqual((caller.v42_mode, answerer.v42_mode), ("lapm", "lapm"))

        for initiator, target in ((caller, 1200), (answerer, 2400)):
            initiator.request_rate_change(target)
            for _ in range(125):
                tx_caller, tx_answerer = caller.tx_samples(160), answerer.tx_samples(160)
                caller.feed_pcm(struct.pack("<160h", *tx_answerer))
                answerer.feed_pcm(struct.pack("<160h", *tx_caller))
            self.assertEqual((caller.bit_rate, answerer.bit_rate), (target, target))
            self.assertEqual((caller._v42.rate, answerer._v42.rate), (target, target))
            self.assertEqual((caller.v42_mode, answerer.v42_mode), ("lapm", "lapm"))

            sent_caller = bytes((i * 13 + target + 0x52) & 0xff for i in range(128))
            sent_answerer = bytes((i * 19 + target + 0xA7) & 0xff for i in range(128))
            caller.feed_bytes(sent_caller)
            answerer.feed_bytes(sent_answerer)
            received_caller, received_answerer = bytearray(), bytearray()
            for _ in range(400):
                tx_caller, tx_answerer = caller.tx_samples(160), answerer.tx_samples(160)
                received_caller.extend(caller.feed_pcm(struct.pack("<160h", *tx_answerer)))
                received_answerer.extend(answerer.feed_pcm(struct.pack("<160h", *tx_caller)))
                if len(received_caller) == len(sent_answerer) and len(received_answerer) == len(sent_caller):
                    break
            self.assertEqual(bytes(received_caller), sent_answerer)
            self.assertEqual(bytes(received_answerer), sent_caller)

    def test_transmitted_dbm0_matches_measured_pcm_power(self):
        # Independent physical reference: full-scale sine = +3.14 dBm0.
        # Measure generated random-data audio, rather than just its gain field.
        import math
        import numpy as np
        for calling in (True, False):
            for requested in (-20.0, -10.0):
                with self.subTest(calling=calling, requested=requested):
                    m = v22.V22Modem(calling, 2400, tx_power_dbm0=requested)
                    try:
                        m._actions.clear()
                        m._set_tx("data", 2400)
                        rng = random.Random(882)
                        m.feed_bits([rng.randrange(2) for _ in range(24000)])
                        pcm = np.asarray(m.tx_samples(80000), dtype=float)[8000:72000]
                        measured = 10 * math.log10(float(np.mean(pcm ** 2)) /
                                                   (32768 ** 2 / 2)) + 3.14
                        self.assertAlmostEqual(measured, requested, delta=0.12)
                        self.assertLess(float(np.max(np.abs(pcm))), 32767)
                    finally:
                        m.close()

    def test_default_transmit_level_meets_v2_one_minute_mean_cap(self):
        # V.2 section 2.3(i): one-minute mean power shall not exceed -13 dBm0 (switched network).
        import math
        m = v22.V22Modem(True, 2400)
        try:
            default_dbm0 = 20 * math.log10(m.rms / v22._ZERO_DBM0_RMS)
            self.assertLessEqual(default_dbm0, -13.0)
        finally:
            m.close()

    def test_guard_tone_is_answer_high_channel_only_at_declared_level(self):
        import math
        import numpy as np

        def capture(calling, guard):
            modem = v22.V22Modem(calling, 2400, profile="v22bis", guard_tone=guard)
            try:
                modem._actions.clear()
                modem._set_tx("scr1", 2400)
                return np.asarray(modem.tx_samples(40000), dtype=float)[8000:32000]
            finally:
                modem.close()

        def tone_rms(signal, hz):
            n = np.arange(len(signal), dtype=float)
            phase = 2 * math.pi * hz * n / v22.FS
            sine = np.sin(phase)
            cosine = np.cos(phase)
            a = 2.0 * float(signal @ sine) / len(signal)
            b = 2.0 * float(signal @ cosine) / len(signal)
            return math.hypot(a, b) / math.sqrt(2.0)

        answer_plain = capture(False, 0)
        caller_plain = capture(True, 0)
        self.assertTrue(np.array_equal(caller_plain, capture(True, 2)))
        for setting, hz in ((1, 550.0), (2, 1800.0)):
            with self.subTest(setting=setting, hz=hz):
                guarded = capture(False, setting)
                guard_rms = tone_rms(guarded, hz)
                data_rms = math.sqrt(max(0.0, float(np.mean(guarded ** 2)) - guard_rms ** 2))
                self.assertAlmostEqual(20 * math.log10(guard_rms / data_rms), -6.0, delta=0.15)
                self.assertAlmostEqual(float(np.sqrt(np.mean(guarded ** 2))),
                                       float(np.sqrt(np.mean(answer_plain ** 2))), delta=0.01 * float(np.sqrt(np.mean(answer_plain ** 2))))
        self.assertTrue(np.array_equal(capture(True, 0), capture(True, 2)))

    def test_receive_dbm0_uses_pcm_reference_and_explicit_rms_is_preserved(self):
        m = v22.V22Modem(True, 2400, rms=1234.0)
        try:
            self.assertEqual(m.rms, 1234.0)
            m._rb.power = (32768 ** 2 / 2) * 10 ** ((-18.0 - 3.14) / 10)
            self.assertAlmostEqual(m.rx_signal_power_dbm0, -18.0)
            m._rb.power = 0
            self.assertEqual(m.rx_signal_power_dbm0, -99.0)
        finally:
            m.close()

    def test_scrambler_roundtrip_and_constellation(self):
        s, d = v22.Scrambler(), v22.Descrambler()
        bits = [(i * 7 + (i >> 3)) & 1 for i in range(500)]
        out = [d.step(s.step(b)) for b in bits]
        self.assertEqual(out[20:], bits[20:])
        self.assertEqual(len(v22.POINTS), 16)
        self.assertAlmostEqual(sum(abs(p[2]) ** 2 for p in v22.POINTS) / 16, 10)

    def test_handshake_and_data_at_2400(self):
        a = v22.V22Modem(True, 2400)
        b = v22.V22Modem(False, 2400)
        sent = {"done": False}

        def until(a, b, tick):
            if a.is_settled() and b.is_settled() and not sent["done"]:
                a.feed_bytes(b"hello from the caller\r\n")
                b.feed_bytes(b"hello from the answerer\r\n")
                sent["done"] = True
            return False

        ga, gb = link(a, b, seconds=14, until=until)
        self.assertEqual((a.bit_rate, b.bit_rate), (2400, 2400))
        self.assertEqual(gb, b"hello from the caller\r\n")
        self.assertEqual(ga, b"hello from the answerer\r\n")

    def test_handshake_and_data_at_optional_600(self):
        caller = v22.V22Modem(True, 600, profile="v22")
        answerer = v22.V22Modem(False, 600, profile="v22")
        sent = {"done": False}

        def until(a, b, _tick):
            if a.is_settled() and b.is_settled() and not sent["done"]:
                a.feed_bytes(b"CALLER-600\r\n")
                b.feed_bytes(b"ANSWER-600\r\n")
                sent["done"] = True
            return False

        try:
            got_caller, got_answerer = link(caller, answerer, seconds=12, until=until)
            self.assertTrue(sent["done"], (caller.events, answerer.events, caller.state, answerer.state))
            self.assertEqual((caller.bit_rate, answerer.bit_rate), (600, 600))
            self.assertEqual(got_answerer, b"CALLER-600\r\n")
            self.assertEqual(got_caller, b"ANSWER-600\r\n")
        finally:
            caller.close()
            answerer.close()

    def test_rate_fallback_below_configured_minimum_fails_training(self):
        modem = v22.V22Modem(True, 2400, min_rate=2400)
        try:
            modem.connected = True
            modem._state = "bis"
            modem._rate = 2400
            modem._fall_back_to_1200()
            self.assertTrue(modem._hold_failed)
            self.assertEqual(modem.state, "failed")
            self.assertEqual(modem._state, "failed")
            self.assertEqual(modem.bit_rate, 2400)
            self.assertIn(("RATE_BELOW_MINIMUM", 1200), modem.events)
        finally:
            modem.close()

    def test_initial_1200_training_is_rejected_when_minimum_is_2400(self):
        modem = v22.V22Modem(True, 2400, min_rate=2400)
        try:
            modem._enter_v22()
            self.assertEqual(modem.state, "failed")
            self.assertFalse(modem.connected)
            self.assertIn(("RATE_BELOW_MINIMUM", 1200), modem.events)
        finally:
            modem.close()

    def test_bell212_fallback_respects_requested_minimum(self):
        modem = v22.V22Modem(False, 1200, profile="bell212a", min_rate=600)
        try:
            modem._select_bell103_fallback()
            self.assertFalse(modem._bell103_fallback)
            self.assertEqual(modem.state, "failed")
            self.assertIn(("RATE_BELOW_MINIMUM", 300), modem.events)
        finally:
            modem.close()

    def test_bell212a_answer_tone_is_2225_hz_after_billing_silence(self):
        import numpy as np
        answerer = v22.V22Modem(False, 1200, profile="bell212a")
        silence = np.asarray([sample for _ in range(100) for sample in answerer.tx_samples(160)], dtype=float)
        self.assertEqual(float(np.max(np.abs(silence))), 0.0)
        tone = np.asarray([sample for _ in range(5) for sample in answerer.tx_samples(160)], dtype=float)
        n = np.arange(len(tone))
        power_2225 = abs(np.dot(tone, np.exp(-2j * np.pi * 2225 * n / 8000))) ** 2
        power_2100 = abs(np.dot(tone, np.exp(-2j * np.pi * 2100 * n / 8000))) ** 2
        self.assertGreater(power_2225, 100 * power_2100)

    def test_bell212a_caller_qualifies_2225_not_v25_2100(self):
        import numpy as np
        caller = v22.V22Modem(True, 1200, profile="bell212a")
        n = np.arange(8000)
        for hz in (2100, 2225):
            pcm = np.asarray(3000 * np.sin(2 * np.pi * hz * n / 8000), dtype="<i2").tobytes()
            caller.feed_pcm(pcm)
            if hz == 2100:
                self.assertNotIn(("BELL212A_ANSWER_TONE", 2225), caller.events)
                self.assertEqual(caller._state, "bell_call_wait_tone")
                caller._answer_tone_samples.clear()
                caller._answer_tone_pending.clear()
                caller._answer_tone_run = 0
            else:
                self.assertIn(("BELL212A_ANSWER_TONE", 2225), caller.events)
                self.assertIn(caller._state, ("bell_call_wait_mark", "bell_call_mark"))

    def test_v22_caller_accepts_national_2225_hz_answer_tone(self):
        import numpy as np
        caller = v22.V22Modem(True, 1200, profile="v22")
        rx_sample = 0
        try:
            # The V.22 national variant substitutes the answer tone for the
            # decoded unscrambled mark. Once qualified, the caller waits the
            # normal 456 ms before transmitting its low-channel scrambled 1s.
            for _ in range(25):
                n = np.arange(rx_sample, rx_sample + 160)
                pcm = np.asarray(3000 * np.sin(2 * np.pi * 2225 * n / 8000), dtype="<i2")
                caller.feed_pcm(pcm.tobytes())
                caller.tx_samples(160)
                rx_sample += len(n)
            self.assertIn(("V22_NATIONAL_ANSWER_TONE", 2225), caller.events)
            self.assertEqual(caller._state, "hs1")
            self.assertEqual(caller._tx_mode, "silence")
            for _ in range(12):
                caller.feed_pcm(bytes(320))
                caller.tx_samples(160)
            self.assertEqual(caller._tx_mode, "scr1")
        finally:
            caller.close()

    def test_v22_caller_does_not_treat_v25_ans_as_national_2225(self):
        import numpy as np
        caller = v22.V22Modem(True, 1200, profile="v22")
        try:
            for offset in range(0, 8000, 160):
                n = np.arange(offset, offset + 160)
                pcm = np.asarray(3000 * np.sin(2 * np.pi * 2100 * n / 8000), dtype="<i2")
                caller.feed_pcm(pcm.tobytes())
                caller.tx_samples(160)
            self.assertNotIn(("V22_NATIONAL_ANSWER_TONE", 2225), caller.events)
            self.assertEqual(caller._state, "wait")
            self.assertEqual(caller._tx_mode, "silence")
        finally:
            caller.close()

    def test_v22bis_2400_caller_uses_s1_after_national_2225_tone(self):
        import numpy as np
        caller = v22.V22Modem(True, 2400, profile="v22bis")
        rx_sample = 0
        try:
            for _ in range(25):
                n = np.arange(rx_sample, rx_sample + 160)
                pcm = np.asarray(3000 * np.sin(2 * np.pi * 2225 * n / 8000), dtype="<i2")
                caller.feed_pcm(pcm.tobytes())
                caller.tx_samples(160)
                rx_sample += len(n)
            self.assertIn(("V22_NATIONAL_ANSWER_TONE", 2225), caller.events)
            self.assertEqual(caller._state, "hs1")
            self.assertEqual(caller._tx_mode, "silence")
            # The first interval ends in S1; after its 100 ms the modem
            # switches to the scrambled mark sequence.
            for _ in range(8):
                caller.feed_pcm(bytes(320))
                caller.tx_samples(160)
            self.assertEqual(caller._tx_mode, "s1")
            for _ in range(6):
                caller.feed_pcm(bytes(320))
                caller.tx_samples(160)
            self.assertEqual(caller._tx_mode, "scr1")
        finally:
            caller.close()

    def test_bell212a_answer_tone_detection_timing(self):
        # UDS 212A-compatible timing allows 155 +/- 50 ms for detection.
        import numpy as np
        caller = v22.V22Modem(True, 1200, profile="bell212a")
        detected_at = None
        original_event = caller._event

        def record_event(name, code=0):
            nonlocal detected_at
            if name == "BELL212A_ANSWER_TONE" and detected_at is None:
                detected_at = caller._rx_samples
            original_event(name, code)

        caller._event = record_event
        elapsed = 0
        while elapsed < 4000 and detected_at is None:
            n = np.arange(elapsed, elapsed + 160)
            pcm = np.asarray(3000 * np.sin(2 * np.pi * 2225 * n / 8000), dtype="<i2")
            caller.feed_pcm(pcm.tobytes())
            elapsed += len(n)
        self.assertIsNotNone(detected_at)
        detection_ms = detected_at * 1000 / 8000
        self.assertGreaterEqual(detection_ms, 105.0)
        self.assertLessEqual(detection_ms, 205.0)

    def test_bell212a_handshake_transition_timing(self):
        # UDS 212A-compatible timing: caller mark 456 +/- 10 ms after
        # answer-tone detection, answer mark after 270 +/- 40 ms of caller
        # mark, then each side's data transition 765 +/- 10 ms after its
        # mark qualification. Measure the modem's own transmit symbol clock,
        # not the 20 ms test-link polling cadence.
        caller = v22.V22Modem(True, 1200, profile="bell212a")
        answerer = v22.V22Modem(False, 1200, profile="bell212a")
        marks = {"caller": [], "answerer": []}
        modems = (("caller", caller), ("answerer", answerer))
        for name, modem in modems:
            original_set_tx = modem._set_tx
            original_event = modem._event

            def set_tx(mode, rate=1200, *, _name=name, _modem=modem,
                       _original=original_set_tx):
                if _modem._tx_mode != mode:
                    marks[_name].append((mode, _modem._tx_sym))
                return _original(mode, rate)

            def event(event_name, code=0, *, _name=name, _modem=modem,
                      _original=original_event):
                if event_name in ("BELL212A_ANSWER_TONE", "TRAINING_SUCCEEDED"):
                    marks[_name].append((event_name, _modem._tx_sym))
                return _original(event_name, code)

            modem._set_tx = set_tx
            modem._event = event

        try:
            link(caller, answerer, seconds=8)
            tx_baud = v22.BAUD
            caller_marks = dict(marks["caller"])
            answerer_marks = dict(marks["answerer"])

            def elapsed_ms(start, end):
                return (end - start) * 1000.0 / tx_baud

            caller_mark_delay = elapsed_ms(
                caller_marks["BELL212A_ANSWER_TONE"], caller_marks["scr1"])
            answerer_turnaround = elapsed_ms(
                caller_marks["scr1"], answerer_marks["scr1"])
            answerer_data_delay = elapsed_ms(
                answerer_marks["scr1"], answerer_marks["data"])
            caller_data_delay = elapsed_ms(
                caller_marks["TRAINING_SUCCEEDED"], caller_marks["data"])

            self.assertAlmostEqual(caller_mark_delay, 456.0, delta=10.0)
            self.assertGreaterEqual(answerer_turnaround, 230.0)
            self.assertLessEqual(answerer_turnaround, 310.0)
            self.assertAlmostEqual(answerer_data_delay, 765.0, delta=10.0)
            self.assertAlmostEqual(caller_data_delay, 765.0, delta=10.0)
        finally:
            caller.close()
            answerer.close()

    def test_bell212a_handshake_and_short_payload_exchange(self):
        caller = v22.V22Modem(True, 1200, profile="bell212a")
        answerer = v22.V22Modem(False, 1200, profile="bell212a")
        sent = {"done": False}

        def on_tick(a, b, tick):
            if a.is_settled() and b.is_settled() and not sent["done"]:
                a.feed_bytes(b"CALLER TEST\r\n")
                b.feed_bytes(b"ANSWER TEST\r\n")
                sent["done"] = True

        got_caller, got_answerer = link(caller, answerer, seconds=8, until=on_tick)
        self.assertTrue(sent["done"])
        self.assertTrue(caller.connected and answerer.connected)
        self.assertEqual((caller.bit_rate, answerer.bit_rate), (1200, 1200))
        self.assertFalse(answerer.bell103_fallback)
        self.assertEqual(got_caller, b"ANSWER TEST\r\n")
        self.assertEqual(got_answerer, b"CALLER TEST\r\n")

    def test_bell212a_leased_line_role_start_and_payload_exchange(self):
        # Dedicated lines have no dialed-call answer tone. The configured
        # originate role starts the low-channel mark; answer waits for it and
        # returns its high-channel mark before both enter data mode.
        caller = v22.V22Modem(True, 1200, profile="bell212a", bell212_leased_line=True)
        answerer = v22.V22Modem(False, 1200, profile="bell212a", bell212_leased_line=True)
        sent = {"done": False}

        def on_tick(a, b, tick):
            if a.is_settled() and b.is_settled() and not sent["done"]:
                a.feed_bytes(b"LEASE CALLER\x00\xff\r\n")
                b.feed_bytes(b"LEASE ANSWER\xff\x00\r\n")
                sent["done"] = True

        try:
            self.assertEqual(caller._state, "bell_leased_call_mark")
            self.assertEqual(caller._tx_mode, "scr1")
            self.assertEqual(answerer._state, "bell_leased_answer_wait_mark")
            self.assertEqual(answerer.tx_samples(160), [0] * 160)
            got_caller, got_answerer = link(caller, answerer, seconds=8, until=on_tick)
            self.assertTrue(sent["done"])
            self.assertTrue(caller.connected and answerer.connected)
            self.assertEqual(got_caller, b"LEASE ANSWER\xff\x00\r\n")
            self.assertEqual(got_answerer, b"LEASE CALLER\x00\xff\r\n")
        finally:
            caller.close()
            answerer.close()

    def test_bell212a_leased_line_retrains_after_sustained_carrier_loss(self):
        caller = v22.V22Modem(True, 1200, profile="bell212a", bell212_leased_line=True)
        answerer = v22.V22Modem(False, 1200, profile="bell212a", bell212_leased_line=True)
        try:
            link(caller, answerer, seconds=3)
            self.assertTrue(caller.connected and answerer.connected)

            # A short interruption is clamped/discarded but does not restart
            # the modem roles; carrier return resumes the existing bearer.
            silence = struct.pack("<160h", *([0] * 160))
            for _ in range(10):
                caller.tx_samples(160)
                answerer.tx_samples(160)
                caller.feed_pcm(silence)
                answerer.feed_pcm(silence)
            self.assertTrue(caller.connected and answerer.connected)
            self.assertTrue(caller._rx_data_clamped and answerer._rx_data_clamped)
            for _ in range(100):
                caller_pcm = caller.tx_samples(160)
                answerer_pcm = answerer.tx_samples(160)
                caller.feed_pcm(struct.pack("<160h", *answerer_pcm))
                answerer.feed_pcm(struct.pack("<160h", *caller_pcm))
                if not caller._rx_data_clamped and not answerer._rx_data_clamped:
                    break
            self.assertFalse(caller._bell_leased_recovering)
            self.assertFalse(answerer._bell_leased_recovering)
            self.assertFalse(caller._rx_data_clamped or answerer._rx_data_clamped)

            caller.start_v42(True, detect=False, rate=1200)
            answerer.start_v42(False, detect=False, rate=1200)
            for _ in range(300):
                tx_caller = caller.tx_samples(160)
                tx_answerer = answerer.tx_samples(160)
                caller.feed_pcm(struct.pack("<160h", *tx_answerer))
                answerer.feed_pcm(struct.pack("<160h", *tx_caller))
                if caller.v42_mode == answerer.v42_mode == "lapm":
                    break
            self.assertEqual((caller.v42_mode, answerer.v42_mode), ("lapm", "lapm"))

            # Simulate a 1.6-second two-way circuit interruption while both
            # modem clocks continue to run and transmit audio is lost.
            for _ in range(80):
                caller.tx_samples(160)
                answerer.tx_samples(160)
                caller.feed_pcm(silence)
                answerer.feed_pcm(silence)
            self.assertFalse(caller.connected)
            self.assertFalse(answerer.connected)
            self.assertIn("BELL212A_LEASE_RECOVERY_STARTED",
                          [event for event, _ in caller.events])
            self.assertIn("BELL212A_LEASE_RECOVERY_STARTED",
                          [event for event, _ in answerer.events])

            sent = {"done": False}

            def on_tick(a, b, tick):
                if a.is_settled() and b.is_settled() and not sent["done"]:
                    a.feed_bytes(b"AFTER RECOVERY A\x00\xff\r\n")
                    b.feed_bytes(b"AFTER RECOVERY B\xff\x00\r\n")
                    sent["done"] = True

            got_caller, got_answerer = link(caller, answerer, seconds=8, until=on_tick)
            self.assertTrue(sent["done"])
            self.assertTrue(caller.connected and answerer.connected)
            self.assertEqual((caller.v42_mode, answerer.v42_mode), ("lapm", "lapm"))
            self.assertEqual(got_caller, b"AFTER RECOVERY B\xff\x00\r\n")
            self.assertEqual(got_answerer, b"AFTER RECOVERY A\x00\xff\r\n")
        finally:
            caller.close()
            answerer.close()

    def test_bell212a_answerer_falls_back_to_bell103_after_qualified_calling_mark(self):
        from bell103 import Bell103Receiver, Bell103StreamTransmitter

        answerer = v22.V22Modem(False, 1200, profile="bell212a")
        caller_tx = Bell103StreamTransmitter("originate")
        caller_rx = Bell103Receiver("answer")
        caller_payload = b"BELL103 CALLER\r\n"
        answer_payload = b"BELL103 ANSWER\r\n"
        mark_at = 2 * 8000 + round((0.155 + 0.456) * 8000)
        caller_data_at = 2 * 8000 + round((0.155 + 0.765) * 8000)
        caller_sent = answer_sent = False
        fallback_at = None
        from_caller = bytearray()
        from_answerer = bytearray()
        original_event = answerer._event

        def record_fallback(name, code=0):
            nonlocal fallback_at
            if name == "BELL212A_FALLBACK_103" and fallback_at is None:
                fallback_at = answerer._rx_samples
            original_event(name, code)

        answerer._event = record_fallback

        for sample in range(0, 6 * 8000, 160):
            answer_pcm = answerer.tx_samples(160)
            if sample >= caller_data_at and not caller_sent:
                caller_tx.feed_bytes(caller_payload)
                caller_sent = True
            caller_pcm = caller_tx.samples(160) if sample >= mark_at else [0] * 160
            from_caller.extend(answerer.feed_pcm(struct.pack("<160h", *caller_pcm)))
            from_answerer.extend(caller_rx.feed_pcm(struct.pack("<160h", *answer_pcm)))
            if answerer.is_settled() and not answer_sent:
                answerer.feed_bytes(answer_payload)
                answer_sent = True

        self.assertTrue(caller_sent)
        self.assertTrue(answer_sent)
        self.assertTrue(answerer.bell103_fallback)
        self.assertEqual(answerer.bit_rate, 300)
        self.assertIsNotNone(fallback_at)
        detection_ms = (fallback_at - mark_at) * 1000 / 8000
        self.assertGreaterEqual(detection_ms, 105.0)
        self.assertLessEqual(detection_ms, 205.0)
        self.assertEqual(bytes(from_caller), caller_payload)
        self.assertEqual(bytes(from_answerer), answer_payload)

    def test_bell212a_fallback_preserves_mark_for_immediate_connect_data(self):
        """The DTE may queue its first byte as soon as fallback CONNECT is ready."""
        from bell103 import Bell103Receiver, Bell103StreamTransmitter

        answerer = v22.V22Modem(False, 1200, profile="bell212a")
        caller_tx = Bell103StreamTransmitter("originate")
        caller_rx = Bell103Receiver("answer")
        payload = bytes(range(128))
        mark_at = 2 * 8000 + round(0.611 * 8000)
        queued = False
        got = bytearray()

        for sample in range(0, 8 * 8000, 160):
            answer_pcm = answerer.tx_samples(160)
            if sample >= mark_at:
                if answerer.is_settled() and not queued:
                    caller_tx.feed_bytes(payload)
                    queued = True
                caller_pcm = caller_tx.samples(160)
            else:
                caller_pcm = [0] * 160
            got.extend(answerer.feed_pcm(struct.pack("<160h", *caller_pcm)))
            caller_rx.feed_pcm(struct.pack("<160h", *answer_pcm))

        self.assertTrue(queued)
        self.assertTrue(answerer.bell103_fallback)
        self.assertEqual(bytes(got), payload)

    def test_bell212a_fallback_decodes_independent_minimodem_vector(self):
        """Exercise the fallback path with an independent Bell 103 waveform."""
        import math
        from pathlib import Path
        from bell103 import SAMPLE_RATE

        fixture = (Path(__file__).parent / "fixtures" / "reference" /
                   "independent-fsk" / "bell103a2-orig.raw").read_bytes()
        expected = b"ELMOD B103 ORIGINATE 0123456789\r"
        independent_samples = list(struct.unpack(f"<{len(fixture) // 2}h", fixture))

        # The answerer has sent two seconds of silence and its 2225-Hz answer
        # mark. The caller's 1270-Hz mark begins at the UDS turnaround point,
        # followed by the independent minimodem vector after a 300-ms mark.
        caller_mark_samples = round(0.611 * SAMPLE_RATE)
        caller_mark = [
            round(12000 * math.sin(2 * math.pi * 1270 * i / SAMPLE_RATE))
            for i in range(round(0.300 * SAMPLE_RATE))
        ]
        incoming = caller_mark + independent_samples
        start = 2 * SAMPLE_RATE + caller_mark_samples
        answerer = v22.V22Modem(False, 1200, profile="bell212a")
        decoded = bytearray()
        total_samples = start + len(incoming) + SAMPLE_RATE
        for offset in range(0, total_samples, 160):
            answerer.tx_samples(160)
            data_offset = offset - start
            block = incoming[data_offset:data_offset + 160] if data_offset >= 0 else []
            block += [0] * (160 - len(block))
            decoded.extend(answerer.feed_pcm(struct.pack("<160h", *block)))

        self.assertTrue(answerer.bell103_fallback)
        self.assertEqual(answerer.bit_rate, 300)
        self.assertEqual(bytes(decoded), expected)

    def test_bell212a_9_bit_async_format_transports_seven_data_bits(self):
        caller = v22.V22Modem(True, 1200, profile="bell212a", bell212_char_bits=9)
        answerer = v22.V22Modem(False, 1200, profile="bell212a", bell212_char_bits=9)
        sent = {"done": False}
        expected_caller = bytes(value & 0x7f for value in b"CALLER 7BIT\r\n")
        expected_answerer = bytes(value & 0x7f for value in b"ANSWER 7BIT\r\n")

        def on_tick(a, b, tick):
            if a.is_settled() and b.is_settled() and not sent["done"]:
                a.feed_bytes(b"CALLER 7BIT\r\n")
                b.feed_bytes(b"ANSWER 7BIT\r\n")
                sent["done"] = True

        got_caller, got_answerer = link(caller, answerer, seconds=8, until=on_tick)
        self.assertTrue(sent["done"])
        self.assertEqual(got_caller, expected_answerer)
        self.assertEqual(got_answerer, expected_caller)

    def test_bell212a_mode_b_stop_adaptation_at_dte_rate_limits(self):
        # UDS 212A-compatible Mode B accepts 1200 bps +1%/-2.5% and adapts
        # the stop element while keeping start/data positions fixed.
        for char_bits in (9, 10, 11, 12):
            rates = (1170, 1200, 1212) if char_bits <= 10 else (1200,)
            for dte_baud in rates:
                with self.subTest(char_bits=char_bits, dte_baud=dte_baud):
                    caller = v22.V22Modem(
                        True, 1200, profile="bell212a", bell212_char_bits=char_bits,
                        bell212_parity="O", bell212_dte_baud=dte_baud,
                    )
                    answerer = v22.V22Modem(
                        False, 1200, profile="bell212a", bell212_char_bits=char_bits,
                        bell212_parity="O", bell212_dte_baud=dte_baud,
                    )
                    data_bits = 7 if char_bits == 9 else 8
                    data_mask = (1 << data_bits) - 1
                    sent = {"done": False}
                    caller_payload = bytes((i * 37 + 11) & data_mask for i in range(48))
                    answerer_payload = bytes((i * 53 + 7) & data_mask for i in range(48))

                    def on_tick(a, b, tick):
                        if a.is_settled() and b.is_settled() and not sent["done"]:
                            a.feed_bytes(caller_payload)
                            b.feed_bytes(answerer_payload)
                            sent["done"] = True

                    got_caller, got_answerer = link(
                        caller, answerer, seconds=7.5, until=on_tick,
                    )
                    self.assertTrue(sent["done"])
                    self.assertEqual(got_caller, answerer_payload)
                    self.assertEqual(got_answerer, caller_payload)
                    caller.close()
                    answerer.close()

    def test_bell212a_rejects_out_of_range_mode_b_dte_baud(self):
        for rate in (1169, 1213):
            with self.subTest(rate=rate), self.assertRaisesRegex(ValueError, "1170..1212"):
                v22.V22Modem(True, 1200, profile="bell212a", bell212_dte_baud=rate)

    def test_bell212a_mode_b_adaptation_is_directional(self):
        # Each DTE clock is independent. The receive deframer must accept the
        # other endpoint's stop-bit pattern rather than assuming symmetry.
        caller = v22.V22Modem(
            True, 1200, profile="bell212a", bell212_char_bits=9,
            bell212_dte_baud=1212,
        )
        answerer = v22.V22Modem(
            False, 1200, profile="bell212a", bell212_char_bits=9,
            bell212_dte_baud=1170,
        )
        caller_payload = bytes((i * 29 + 3) & 0x7f for i in range(64))
        answerer_payload = bytes((i * 43 + 5) & 0x7f for i in range(64))
        sent = {"done": False}

        def on_tick(a, b, tick):
            if a.is_settled() and b.is_settled() and not sent["done"]:
                a.feed_bytes(caller_payload)
                b.feed_bytes(answerer_payload)
                sent["done"] = True

        got_caller, got_answerer = link(caller, answerer, seconds=8, until=on_tick)
        self.assertTrue(sent["done"])
        self.assertEqual(got_caller, answerer_payload)
        self.assertEqual(got_answerer, caller_payload)
        caller.close()
        answerer.close()

    def test_bell212a_mode_b_stop_schedule_meets_rate_and_deletion_limits(self):
        payload = bytes((i * 73 + 19) & 0xff for i in range(128))
        for char_bits in (9, 10, 11, 12):
            data_bits = 7 if char_bits == 9 else 8
            parity_bits = 1 if char_bits >= 11 else 0
            nominal_stops = char_bits - (1 + data_bits + parity_bits)
            data_mask = (1 << data_bits) - 1
            for rate in (1170, 1200, 1212):
                with self.subTest(char_bits=char_bits, rate=rate):
                    modem = v22.V22Modem(
                        True, 1200, profile="bell212a",
                        bell212_char_bits=char_bits, bell212_parity="O", bell212_dte_baud=rate,
                    )
                    expected = bytes(value & data_mask for value in payload)
                    modem._plain_feed_bytes(payload)
                    wire = list(modem._app_bits)
                    stops = []
                    offset = 0
                    for value in expected:
                        self.assertEqual(wire[offset], 0)  # start
                        offset += 1
                        got = sum(wire[offset + bit] << bit for bit in range(data_bits))
                        self.assertEqual(got, value)
                        offset += data_bits
                        if parity_bits:
                            self.assertEqual(wire[offset], (value.bit_count() & 1) ^ 1)
                            offset += 1
                        stop_start = offset
                        while offset < len(wire) and wire[offset] == 1:
                            offset += 1
                        stops.append(offset - stop_start)
                    self.assertEqual(offset, len(wire))

                    # Check the independent rational rate budget: total output
                    # bits track N * character_bits * 1200 / DTE baud to <1 bit.
                    numerator = len(expected) * char_bits * 1200
                    self.assertLessEqual(abs(len(wire) * rate - numerator), rate // 2)
                    if rate == 1200:
                        self.assertEqual(stops, [nominal_stops] * len(expected))
                    elif rate < 1200:
                        self.assertTrue(all(count in (nominal_stops, nominal_stops + 1) for count in stops))
                        self.assertIn(nominal_stops + 1, stops)
                    else:
                        self.assertTrue(all(count in (nominal_stops - 1, nominal_stops) for count in stops))
                        self.assertIn(nominal_stops - 1, stops)
                        self.assertTrue(all(sum(count == nominal_stops - 1 for count in stops[i:i + 8]) <= 1
                                            for i in range(len(stops) - 7)))

    def test_bell212a_parity_modes_roundtrip_and_reject_bad_parity(self):
        for parity in ("E", "O", "M", "S"):
            with self.subTest(parity=parity):
                def frame(value, corrupt=False):
                    data = [(value >> bit) & 1 for bit in range(8)]
                    ones = value.bit_count() & 1
                    p = {"E": ones, "O": ones ^ 1, "M": 1, "S": 0}[parity]
                    return [0] + data + [p ^ int(corrupt), 1]

                receiver = v22._BellAsyncDeframer(8, parity=parity, character_bits=11)
                self.assertEqual(receiver.push(frame(0x33) + frame(0xA5) + [1]), bytes((0x33, 0xA5)))
                receiver = v22._BellAsyncDeframer(8, parity=parity, character_bits=11)
                self.assertEqual(receiver.push(frame(0x33, corrupt=True) + frame(0xA5) + [1]), bytes((0xA5,)))

    def test_bell212a_mode_b_break_mapping_and_character_resynchronization(self):
        # UDS 212A-compatible spec §3.3: normalize an M..(2M+3)-bit
        # start-polarity run to 2M+3 bits; pass longer runs, then recover at
        # the next stop-to-start transition.
        for data_bits in (7, 8):
            with self.subTest(data_bits=data_bits):
                decoder = v22._BellAsyncDeframer(data_bits)

                def frame(value):
                    return [0] + [(value >> bit) & 1 for bit in range(data_bits)] + [1]

                prefix = [0x00, (1 << data_bits) - 1]
                suffix = [0x55 & ((1 << data_bits) - 1), 0x00, 0x7f & ((1 << data_bits) - 1)]
                m = data_bits + 2
                break_bits = 2 * m + 9  # longer than the normalization threshold
                wire = []
                for value in prefix:
                    wire.extend(frame(value))
                wire.extend([0] * break_bits)
                wire.extend([1] * (2 * m))
                for value in suffix:
                    wire.extend(frame(value))

                output = bytearray()
                # Irregular chunks ensure break detection does not depend on
                # audio/decoder callback boundaries.
                for start in range(0, len(wire), 7):
                    output.extend(decoder.push(wire[start:start + 7]))
                self.assertEqual(output, bytes(prefix + suffix))
                events = decoder.take_break_events()
                # Longer BREAKs pass through, rather than being shortened to
                # the minimum normalization threshold.
                self.assertEqual(events, [break_bits])
                self.assertEqual(decoder.take_break_events(), [])

    def test_bell212a_mode_b_break_tx_uses_normalized_start_polarity(self):
        for char_bits in (9, 10):
            with self.subTest(char_bits=char_bits):
                modem = v22.V22Modem(True, 1200, profile="bell212a",
                                     bell212_char_bits=char_bits)
                try:
                    modem._app_bits.clear()
                    self.assertTrue(modem.bell212_feed_break())
                    self.assertEqual(list(modem._app_bits),
                                     [0] * (2 * char_bits + 3) + [1] * (2 * char_bits))
                    modem._app_bits.clear()
                    modem.v42_mode = "recovering"
                    self.assertFalse(modem.bell212_feed_break())
                    self.assertEqual(list(modem._app_bits), [])
                    other = v22.V22Modem(True, 1200)
                    try:
                        self.assertFalse(other.bell212_feed_break())
                    finally:
                        other.close()
                finally:
                    modem.close()

    def test_bell212a_mode_b_break_roundtrips_with_payload_resync(self):
        caller = v22.V22Modem(True, 1200, profile="bell212a")
        answerer = v22.V22Modem(False, 1200, profile="bell212a")
        before = b"BEFORE\x00\xff"
        after = b"AFTER\x00\x55"
        sent = {"done": False}

        def on_tick(a, b, tick):
            if a.is_settled() and b.is_settled() and not sent["done"]:
                a.feed_bytes(before)
                self.assertTrue(a.bell212_feed_break())
                a.feed_bytes(after)
                sent["done"] = True

        try:
            got_caller, got_answerer = link(caller, answerer, seconds=8, until=on_tick)
            self.assertTrue(sent["done"])
            self.assertEqual(got_caller, b"")
            self.assertEqual(got_answerer, before + after)
            self.assertEqual(answerer.bell212_read_break_events(), [
                {"type": "break_indication", "start_bits": 23, "character_bits": 10},
            ])
            self.assertEqual(answerer.bell212_read_break_events(), [])
        finally:
            caller.close()
            answerer.close()

    def test_bell212a_idle_mark_does_not_decode_spurious_bytes(self):
        # The 2225 Hz startup detector must not notch the scrambled mark/data
        # stream after its bounded handshake window.
        caller = v22.V22Modem(True, 1200, profile="bell212a")
        answerer = v22.V22Modem(False, 1200, profile="bell212a")
        got_caller, got_answerer = link(caller, answerer, seconds=30)
        self.assertEqual((got_caller, got_answerer), (b"", b""))


def _exchange(a, b, seconds=16):
    sent = {"done": False}

    def until(a, b, tick):
        if a.is_settled() and b.is_settled() and not sent["done"]:
            a.feed_bytes(b"from caller\r\n")
            b.feed_bytes(b"from answerer\r\n")
            sent["done"] = True
        return False

    return link(a, b, seconds=seconds, until=until)


@unittest.skipIf(numpy is None, "numpy not installed")
class SpandspInterop(unittest.TestCase):
    """SpanDSP is only a test reference; skipped where its library is not built."""

    def setUp(self):
        try:
            import spandsp_v22
            self.ref = spandsp_v22.V22bisModem
            self.ref(True, 1200)
        except Exception as exc:                                  # pragma: no cover
            self.skipTest(f"SpanDSP reference unavailable: {exc}")

    def _run(self, ours_calls, rate):
        ours = v22.V22Modem(ours_calls, rate)
        ref = self.ref(not ours_calls, rate, bell212a=False)
        a, b = (ours, ref) if ours_calls else (ref, ours)
        ga, gb = _exchange(a, b)
        self.assertEqual((ours.bit_rate, ref.bit_rate), (rate, rate))
        got_ours, got_ref = (ga, gb) if ours_calls else (gb, ga)
        self.assertEqual(got_ours, b"from answerer\r\n" if ours_calls else b"from caller\r\n")
        self.assertEqual(got_ref, b"from caller\r\n" if ours_calls else b"from answerer\r\n")

    def test_ours_calls_2400(self): self._run(True, 2400)
    def test_ours_answers_2400(self): self._run(False, 2400)
    def test_ours_calls_1200(self): self._run(True, 1200)
    def test_ours_answers_1200(self): self._run(False, 1200)

if __name__ == "__main__":
    unittest.main()
