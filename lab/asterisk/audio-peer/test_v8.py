#!/usr/bin/env python3
"""V.8: our negotiator against itself, with echo/noise, and against SpanDSP's V.8 as an independent reference."""
import random
import unittest

import v8


def loop(call, ans, seconds=14.0, delay=40, noise=0.0, seed=1):
    rng = random.Random(seed)
    to_a, to_c = [0] * delay, [0] * delay
    for _ in range(int(seconds * 50)):
        c, a = call.tx(160), ans.tx(160)
        to_a += c
        to_c += a
        xa, xc = to_a[:160], to_c[:160]
        del to_a[:160], to_c[:160]
        if noise:
            xa = [v + rng.gauss(0, noise) for v in xa]
            xc = [v + rng.gauss(0, noise) for v in xc]
        call.rx(xc)
        ans.rx(xa)
        if call.status in (2, 3, 4) and ans.status in (2, 3, 4):
            break


class SelfTests(unittest.TestCase):
    def test_message_coding(self):
        self.assertEqual(v8.call_function_octet(6), 0xC1)
        self.assertEqual(v8.modulation_octets(v8.V8_MOD_V34 | v8.V8_MOD_V32 | v8.V8_MOD_V21)[0], 0x45)
        bits = v8.build_message(6, v8.V8_MOD_V32)
        self.assertEqual(bits[:20], [1] * 10 + v8.SYNC_CM_JM)
        self.assertEqual(v8.parse_sequence(bits[20:]), [0xC1, 0x05, 0x11, 0x10])
        bits_with_lapm = v8.build_message(6, v8.V8_MOD_V34, v8.V8_PROTOCOL_LAPM_V42)
        self.assertEqual(v8.parse_sequence(bits_with_lapm[20:]), [0xC1, 0x45, 0x10, 0x10, 0x2A])
        self.assertEqual(v8.decode_modulations(v8.modulation_octets(v8.V8_MOD_V34 | v8.V8_MOD_V22)),
                         v8.V8_MOD_V34 | v8.V8_MOD_V22)

    @unittest.skipUnless(v8.available(), "numpy not installed: V.8 tones cannot be generated")
    def test_caller_and_answerer_negotiate_the_common_modulation(self):
        call = v8.V8Negotiator(True, modulations=v8.V8_MOD_V34 | v8.V8_MOD_V32, protocols=v8.V8_PROTOCOL_LAPM_V42)
        ans = v8.V8Negotiator(False, modulations=v8.V8_MOD_V32 | v8.V8_MOD_V22)
        loop(call, ans)
        self.assertEqual((call.status, ans.status), (2, 2), (call.status, ans.status))
        self.assertEqual(call.result["modulations"], v8.V8_MOD_V32)
        self.assertEqual(ans.result["modulations"], v8.V8_MOD_V32)
        self.assertEqual(call.result["call_function"], 6)
        self.assertEqual(call.result["protocols"], v8.V8_PROTOCOL_NONE)
        self.assertEqual(ans.result["protocols"], v8.V8_PROTOCOL_NONE)

    @unittest.skipUnless(v8.available(), "numpy not installed: V.8 tones cannot be generated")
    def test_lapm_is_a_mutual_protocol_and_is_reported_by_both_sides(self):
        call = v8.V8Negotiator(True, modulations=v8.V8_MOD_V34,
                               protocols=v8.V8_PROTOCOL_LAPM_V42)
        ans = v8.V8Negotiator(False, modulations=v8.V8_MOD_V34,
                              protocols=v8.V8_PROTOCOL_LAPM_V42)
        loop(call, ans)
        self.assertEqual((call.status, ans.status), (2, 2))
        self.assertEqual(call.result["protocols"], v8.V8_PROTOCOL_LAPM_V42)
        self.assertEqual(ans.result["protocols"], v8.V8_PROTOCOL_LAPM_V42)

    @unittest.skipUnless(v8.available(), "numpy not installed: V.8 tones cannot be generated")
    def test_lapm_is_not_reported_for_a_v21_only_common_modulation(self):
        call = v8.V8Negotiator(True, modulations=v8.V8_MOD_V21 | v8.V8_MOD_V22,
                               protocols=v8.V8_PROTOCOL_LAPM_V42)
        ans = v8.V8Negotiator(False, modulations=v8.V8_MOD_V21,
                              protocols=v8.V8_PROTOCOL_LAPM_V42)
        loop(call, ans)
        self.assertEqual((call.result["modulations"], ans.result["modulations"]),
                         (v8.V8_MOD_V21, v8.V8_MOD_V21))
        self.assertEqual((call.result["protocols"], ans.result["protocols"]),
                         (v8.V8_PROTOCOL_NONE, v8.V8_PROTOCOL_NONE))

    @unittest.skipUnless(v8.available(), "numpy not installed: V.8 tones cannot be generated")
    def test_with_delay_and_noise(self):
        call = v8.V8Negotiator(True, modulations=v8.V8_MOD_V34)
        ans = v8.V8Negotiator(False, modulations=v8.V8_MOD_V34)
        loop(call, ans, delay=1200, noise=150)
        self.assertEqual((call.status, ans.status), (2, 2))
        self.assertEqual(call.result["modulations"], v8.V8_MOD_V34)

    @unittest.skipUnless(v8.available(), "numpy not installed: V.8 tones cannot be generated")
    def test_answerer_waits_at_least_200ms_before_ansam(self):
        answer = v8.V8Negotiator(False, modulations=v8.V8_MOD_V21)
        initial = [answer.tx(160) for _ in range(10)]
        self.assertTrue(all(not any(frame) for frame in initial),
                        "answerer must remain silent for the first 200 ms")
        self.assertTrue(any(answer.tx(160)), "ANSam should begin after the 200 ms guard")

    @unittest.skipUnless(v8.available(), "numpy not installed: V.8 tones cannot be generated")
    def test_caller_sends_required_silence_after_cj_before_v21(self):
        call = v8.V8Negotiator(True, modulations=v8.V8_MOD_V21)
        answer = v8.V8Negotiator(False, modulations=v8.V8_MOD_V21)
        te_silence_samples = 0
        caller_silence = None
        silence_samples = 0
        answer_silence = None
        answer_silence_samples = 0

        # Drive the negotiated audio as 20 ms RTP frames, as the production
        # DCE does. Capture the caller's samples after CJ completes and before
        # the V.8 state becomes available for the selected V.21 engine.
        for _ in range(14 * 50):
            stage_before_tx = call._stage
            answer_stage_before_tx = answer._stage
            caller_pcm = call.tx(160)
            answer_pcm = answer.tx(160)
            if stage_before_tx == "te":
                te_silence_samples += len(caller_pcm)
                self.assertEqual(caller_pcm, [0] * len(caller_pcm))
            if caller_silence is not None and stage_before_tx == "post_cj":
                silence_samples += len(caller_pcm)
                self.assertEqual(caller_pcm, [0] * len(caller_pcm))
            elif call._stage == "post_cj":
                caller_silence = call.t
            if answer_silence is not None and answer_stage_before_tx == "post_cj":
                answer_silence_samples += len(answer_pcm)
                self.assertEqual(answer_pcm, [0] * len(answer_pcm))

            call.rx(answer_pcm)
            answer.rx(caller_pcm)
            if answer_silence is None and answer._stage == "post_cj":
                answer_silence = answer.t
            if call.status == answer.status == 2:
                break

        self.assertEqual((call.status, answer.status), (2, 2))
        self.assertGreaterEqual(te_silence_samples, int(0.5 * v8.FS))
        self.assertIsNotNone(caller_silence)
        # 75 ± 5 ms, quantized to the 20 ms media cadence, permits 80 ms.
        self.assertGreaterEqual(silence_samples, int(0.070 * v8.FS))
        self.assertLessEqual(silence_samples, int(0.080 * v8.FS))
        self.assertIsNotNone(answer_silence)
        self.assertGreaterEqual(answer_silence_samples, int(0.070 * v8.FS))
        self.assertLessEqual(answer_silence_samples, int(0.080 * v8.FS))

    @unittest.skipUnless(v8.available(), "numpy not installed: V.8 tones cannot be generated")
    def test_answerer_reports_a_non_v8_call_when_no_cm_arrives(self):
        ans = v8.V8Negotiator(False, modulations=v8.V8_MOD_V32)
        for _ in range(50 * 7):
            ans.tx(160)
            ans.rx([0] * 160)
        self.assertEqual(ans.status, 3)


def _scripted_answerer_caller(jm_repeats, then_tone_a=True, seconds=9.0):
    """Our caller against a scripted far end: ANSam, `jm_repeats` JM sequences, then 2400 Hz (V.34 tone A) or silence."""
    import numpy as np
    call = v8.V8Negotiator(True, modulations=v8.V8_MOD_V32 | v8.V8_MOD_V34)
    ansam = v8._AnsAm(3000.0)
    jm = v8._Fsk(v8.V21_H, 1500.0)
    jm.bits = v8.build_message(6, v8.V8_MOD_V32 | v8.V8_MOD_V34, 0) * jm_repeats
    n_ans, jm_started, phase = 0, None, 0
    for tick in range(int(seconds * 50)):
        call.tx(160)
        t = tick * 0.02
        if t < 1.5:
            x = np.zeros(160)
        elif call.status == 0 or call._stage == 'wait_ans':
            x = ansam.read(160)
        elif call._stage in ('te', 'cm') and (jm_started is None and call._stage == 'cm' or jm_started is not None):
            if jm_started is None:
                jm_started = t
            if jm.bits:
                x = jm.read(160)
            elif then_tone_a:
                x = 3000.0 * np.sin(2 * np.pi * 2400.0 * (np.arange(160) + phase) / 8000.0); phase += 160
            else:
                x = np.zeros(160)
        else:
            x = np.zeros(160)
        call.rx(x)
        if call.status in (2, 3, 4):
            break
    return call


class SingleJmTests(unittest.TestCase):
    def test_one_jm_then_tone_a_completes_v8(self):
        call = _scripted_answerer_caller(jm_repeats=1)
        self.assertEqual(call.status, 2)
        self.assertTrue(call.single_jm)

    def test_one_jm_then_silence_does_not_complete(self):
        call = _scripted_answerer_caller(jm_repeats=1, then_tone_a=False)
        self.assertNotEqual(call.status, 2)

    def test_switch_disables_it(self):
        import os
        from unittest.mock import patch
        with patch.dict(os.environ, {'V34_V8_SINGLE_JM': '0'}):
            call = _scripted_answerer_caller(jm_repeats=1)
        self.assertNotEqual(call.status, 2)

    def test_repeated_jm_completes_the_normal_way(self):
        call = _scripted_answerer_caller(jm_repeats=5)
        self.assertEqual(call.status, 2)
        self.assertFalse(call.single_jm)


def _ref():
    try:
        import spandsp_v8
        return spandsp_v8.V8Negotiator if spandsp_v8.available() else None
    except Exception:
        return None


@unittest.skipUnless(_ref(), "libspandsp (reference implementation) not available")
class InteropTests(unittest.TestCase):
    def test_our_caller_with_the_reference_answerer_selects_v21(self):
        import spandsp_v8
        call = v8.V8Negotiator(True, modulations=v8.V8_MOD_V21)
        ans = _ref()(False, modulations=spandsp_v8.V8_MOD_V21)
        loop(call, ans)
        self.assertEqual((call.status, ans.status), (2, 2), (call.status, ans.status))
        self.assertEqual(call.result["modulations"], v8.V8_MOD_V21)
        self.assertEqual(ans.result["modulations"], spandsp_v8.V8_MOD_V21)
        self.assertEqual(call.result["call_function"], v8.V8_CALL_V_SERIES)

    def test_reference_caller_with_our_answerer_selects_v21(self):
        import spandsp_v8
        call = _ref()(True, modulations=spandsp_v8.V8_MOD_V21)
        ans = v8.V8Negotiator(False, modulations=v8.V8_MOD_V21)
        loop(call, ans)
        self.assertEqual((call.status, ans.status), (2, 2), (call.status, ans.status))
        self.assertEqual(call.result["modulations"], spandsp_v8.V8_MOD_V21)
        self.assertEqual(ans.result["modulations"], v8.V8_MOD_V21)

    def test_our_caller_with_the_reference_answerer(self):
        import spandsp_v8
        call = v8.V8Negotiator(True, modulations=v8.V8_MOD_V34 | v8.V8_MOD_V32)
        ans = _ref()(False, modulations=spandsp_v8.V8_MOD_V34 | spandsp_v8.V8_MOD_V32)
        loop(call, ans)
        self.assertEqual((call.status, ans.status), (2, 2), (call.status, ans.status))
        self.assertEqual(call.result["modulations"] & v8.V8_MOD_V34, v8.V8_MOD_V34)

    def test_the_reference_caller_with_our_answerer(self):
        import spandsp_v8
        call = _ref()(True, modulations=spandsp_v8.V8_MOD_V34 | spandsp_v8.V8_MOD_V32)
        ans = v8.V8Negotiator(False, modulations=v8.V8_MOD_V34 | v8.V8_MOD_V32)
        loop(call, ans)
        self.assertEqual((call.status, ans.status), (2, 2), (call.status, ans.status))
        self.assertEqual(ans.result["modulations"] & v8.V8_MOD_V34, v8.V8_MOD_V34)


if __name__ == "__main__":
    unittest.main()
