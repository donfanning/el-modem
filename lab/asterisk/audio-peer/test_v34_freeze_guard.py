import os
import unittest

import numpy as np

from v34_freeze_guard import FreezeGuard


def noise(n=160, seed=0, amp=3000):
    return np.round(np.random.default_rng(seed).normal(0, amp, n))


class FreezeGuardTests(unittest.TestCase):
    def test_distinct_blocks_pass_unchanged(self):
        g = FreezeGuard()
        for i in range(50):
            x = noise(seed=i)
            self.assertTrue(np.array_equal(g.filter(x, i * 160), x))
        self.assertEqual(g.frozen_blocks, 0)
        self.assertIsNone(g.pending_event)

    def test_a_repeated_block_is_replaced_by_zeros_until_the_stream_moves_again(self):
        g = FreezeGuard()
        a, b = noise(seed=1), noise(seed=2)
        self.assertTrue(np.array_equal(g.filter(a, 0), a))
        out = [g.filter(a.copy(), 160 * (i + 1)) for i in range(5)]
        self.assertTrue(all(not o.any() for o in out))
        self.assertEqual((g.frozen_blocks, g.longest_run), (5, 5))
        self.assertEqual(g.pending_event, ("start", 160))
        g.pending_event = None
        self.assertTrue(np.array_equal(g.filter(b, 960), b))             # the stream moved on: passed again
        self.assertEqual(g.pending_event, ("end", 5))
        self.assertEqual(g.runs, [[160, 5]])

    def test_tone_a_and_tone_b_repeat_but_are_not_a_frozen_stream(self):
        n = np.arange(160)
        tone_a = np.round(3000 * (np.sin(2 * np.pi * 2400 * n / 8000) + np.sin(2 * np.pi * 1800 * n / 8000)))
        tone_b = np.round(6000 * np.sin(2 * np.pi * 1200 * n / 8000))
        for tone in (tone_a, tone_b):
            g = FreezeGuard()
            for i in range(60):
                self.assertTrue(np.array_equal(g.filter(tone.copy(), i * 160), tone))
            self.assertEqual((g.frozen_blocks, g.longest_run, g.tone_blocks), (0, 0, 59))
            self.assertFalse(g.holding)
            self.assertIsNone(g.pending_event)

    def test_all_zero_blocks_are_lost_audio_not_a_frozen_frame(self):
        g = FreezeGuard()
        a = noise(seed=9)
        g.filter(a, 0)
        out = g.filter(np.zeros(160), 160)                                # a 20 ms dropout of exact zeros
        self.assertFalse(out.any())
        self.assertEqual((g.zero_blocks, g.frozen_blocks), (1, 0))
        self.assertTrue(g.holding)

    def test_hold_lasts_through_the_lost_stretch_and_the_margin_after_it(self):
        from v34_freeze_guard import HOLD_MARGIN
        g = FreezeGuard()
        g.filter(noise(seed=10), 0)
        for i in range(3):
            g.filter(np.zeros(160), 160 * (i + 1))
        self.assertTrue(g.holding)
        g.filter(noise(seed=11), 640)                                     # first real block closes the run
        self.assertTrue(g.holding)                                        # margin still to run
        blocks = HOLD_MARGIN // 160
        for i in range(blocks):
            self.assertTrue(g.holding, i)
            g.filter(noise(seed=20 + i), 800 + 160 * i)
        self.assertFalse(g.holding)

    def test_quiet_repeats_below_the_rms_floor_are_not_flagged(self):
        g = FreezeGuard()
        q = np.full(160, 5.0)
        for i in range(10):
            g.filter(q, i * 160)
        self.assertEqual(g.frozen_blocks, 0)

    def test_block_length_changes_reset_the_comparison(self):
        g = FreezeGuard()
        a = noise(160, 3)
        g.filter(a, 0)
        out = g.filter(a[:100], 160)                                      # same start, different length
        self.assertTrue(out.any())
        self.assertEqual(g.frozen_blocks, 0)

    def test_comparison_uses_the_received_audio_not_the_zero_filled_copy(self):
        g = FreezeGuard()
        a = noise(seed=4)
        g.filter(a, 0)
        for i in range(3):
            self.assertFalse(g.filter(a.copy(), 160 * (i + 1)).any())
        self.assertEqual(g.longest_run, 3)

    def test_report(self):
        g = FreezeGuard()
        a = noise(seed=5)
        g.filter(a, 0)
        g.filter(a.copy(), 160)
        r = g.report()
        self.assertEqual((r["frozen_blocks"], r["frozen_samples"], r["longest_run_blocks"]), (1, 160, 1))


class EngineIntegrationTests(unittest.TestCase):
    def engine(self, role, enabled):
        old = os.environ.get("V34_FREEZE_GUARD")
        if enabled:
            os.environ["V34_FREEZE_GUARD"] = "1"
        else:
            os.environ.pop("V34_FREEZE_GUARD", None)
        self.addCleanup(lambda: os.environ.__setitem__("V34_FREEZE_GUARD", old) if old is not None
                        else os.environ.pop("V34_FREEZE_GUARD", None))
        if role == "call":
            import v34_call
            return v34_call.V34Caller(symbol_rates=[2400], data_rate=4800)
        import v34_ans
        return v34_ans.V34Answerer(symbol_rates=[2400], data_rate=4800)

    def feed(self, e, blocks):
        e._step = lambda: None                                            # only the ingestion step is under test
        for b in blocks:
            e.rx(b)

    def test_default_is_off(self):
        for role in ("call", "answer"):
            self.assertIsNone(self.engine(role, False)._freeze)

    def test_only_active_once_the_data_phase_has_started(self):
        for role in ("call", "answer"):
            e = self.engine(role, True)
            a = noise(seed=6)
            self.feed(e, [a, a.copy(), a.copy()])
            self.assertEqual(len(e._rx), 480)
            self.assertTrue(e._rx.any())                                  # handshake audio is never touched
            self.assertEqual(e._freeze.frozen_blocks, 0)

    def test_in_the_data_phase_adaptation_is_held_and_then_restored(self):
        import types
        for role in ("call", "answer"):
            e = self.engine(role, True)
            e._rx_state = {}
            e._p4rcv = types.SimpleNamespace(eq=types.SimpleNamespace(adapt=True))
            self.feed(e, [noise(seed=30)])
            self.assertTrue(e._p4rcv.eq.adapt)
            self.feed(e, [np.zeros(160)])                                 # a dropout
            self.assertFalse(e._p4rcv.eq.adapt)
            self.feed(e, [noise(seed=31 + i) for i in range(8)])          # the stream is back; margin of 4 blocks
            self.assertTrue(e._p4rcv.eq.adapt)

    def test_a_pre_existing_adapt_false_is_not_turned_on_by_the_release(self):
        import types
        e = self.engine("call", True)
        e._rx_state = {}
        e._p4rcv = types.SimpleNamespace(eq=types.SimpleNamespace(adapt=False))
        self.feed(e, [noise(seed=40), np.zeros(160)] + [noise(seed=41 + i) for i in range(8)])
        self.assertFalse(e._p4rcv.eq.adapt)

    def test_in_the_data_phase_repeats_are_zeroed_and_logged(self):
        for role in ("call", "answer"):
            e = self.engine(role, True)
            e._rx_state = {}                                              # data phase has started
            a, b = noise(seed=7), noise(seed=8)
            self.feed(e, [a, a.copy(), a.copy(), b])
            rx = e._rx
            self.assertTrue(np.array_equal(rx[:160], a))
            self.assertFalse(rx[160:480].any())
            self.assertTrue(np.array_equal(rx[480:], b))
            texts = [t for _, t in e.events]
            self.assertTrue(any("stream lost" in t for t in texts), texts)
            self.assertTrue(any("resumed after 2 lost blocks" in t for t in texts), texts)


class RunSamplesTests(unittest.TestCase):
    def test_run_samples_accumulates_and_resets(self):
        g = FreezeGuard()
        g.filter(noise(seed=50), 0)
        for i in range(5):
            g.filter(np.zeros(160), 160 * (i + 1))
        self.assertEqual(g.run_samples, 800)
        g.filter(noise(seed=51), 960)
        self.assertEqual(g.run_samples, 0)


class OneSecondLimitTests(unittest.TestCase):
    def data_phase_caller(self, limit=None):
        env = {"V34_FREEZE_GUARD": "1"}
        if limit is not None:
            env["V34_FREEZE_LIMIT_S"] = str(limit)
        old = {k: os.environ.get(k) for k in ("V34_FREEZE_GUARD", "V34_FREEZE_LIMIT_S")}
        os.environ.update(env)
        self.addCleanup(lambda: [os.environ.__setitem__(k, v) if v is not None else os.environ.pop(k, None) for k, v in old.items()])
        import v34_call
        c = v34_call.V34Caller(symbol_rates=[2400], data_rate=4800)
        c._step = lambda: None
        c._rx_state = {}
        return c

    def test_a_short_outage_does_not_raise_the_flag(self):
        c = self.data_phase_caller()
        for i in range(30):                                               # 0.6 s of zeros
            c.rx(np.zeros(160))
        self.assertFalse(c.lost_too_long)

    def test_one_second_of_loss_raises_the_flag_once(self):
        c = self.data_phase_caller()
        c.rx(noise(seed=60))
        for i in range(60):
            c.rx(np.zeros(160))
        self.assertTrue(c.lost_too_long)
        self.assertEqual(sum("this data link is gone" in t for _, t in c.events), 1)

    def test_frozen_frames_count_like_zeros(self):
        c = self.data_phase_caller()
        a = noise(seed=61)
        c.rx(a)
        for i in range(60):
            c.rx(a.copy())
        self.assertTrue(c.lost_too_long)

    def test_the_limit_is_configurable(self):
        c = self.data_phase_caller(limit=0.2)
        c.rx(noise(seed=62))
        for i in range(12):
            c.rx(np.zeros(160))
        self.assertTrue(c.lost_too_long)


class RetrainOpeningTests(unittest.TestCase):
    def test_begin_retrain_sends_70_ms_of_silence_then_tone_b(self):
        import v34_call
        c = v34_call.V34Caller(symbol_rates=[2400], data_rate=4800)
        c.begin_retrain()
        self.assertEqual(c.stage, "info0_ack")
        out = np.asarray(c.tx(8000), dtype=float)
        self.assertFalse(out[:int(0.065 * 8000)].any())                   # 70 +- 5 ms of silence (11.5.1.1)
        tone = out[int(0.080 * 8000):int(0.5 * 8000)]
        self.assertGreater(float(np.sqrt(np.mean(tone ** 2))), 500)
        spec = np.abs(np.fft.rfft(tone * np.hanning(len(tone))))
        peak = np.fft.rfftfreq(len(tone), 1 / 8000)[int(np.argmax(spec))]
        self.assertAlmostEqual(peak, 1200.0, delta=15.0)                  # Tone B


class ModemRetrainTests(unittest.TestCase):
    def modem(self, **env):
        old = {k: os.environ.get(k) for k in ("V34_FREEZE_GUARD", "V34_FREEZE_RETRAIN")}
        os.environ["V34_FREEZE_GUARD"] = "1"
        os.environ.pop("V34_FREEZE_RETRAIN", None)
        os.environ.update(env)
        self.addCleanup(lambda: [os.environ.__setitem__(k, v) if v is not None else os.environ.pop(k, None) for k, v in old.items()])
        from v34_modem import V34Modem
        m = V34Modem(data_rate=4800, with_v8=False)
        m.state, m.bit_rate = "data", 4800
        return m

    def lose(self, m):
        m.caller.lost_too_long = True
        m.feed_pcm(np.zeros(160, dtype="<i2").tobytes())

    def test_a_lost_stream_swaps_in_a_fresh_caller_and_stays_connected(self):
        m = self.modem()
        old = m.caller
        self.lose(m)
        self.assertIsNot(m.caller, old)
        self.assertTrue(m._retraining)
        self.assertTrue(m.connected)                                      # the host sees no carrier loss
        self.assertEqual(m.caller.stage, "info0_ack")
        self.assertEqual(m.caller.bit_sink, m._bit_sink)                 # the V.42 layer stays wired to the new caller
        self.assertTrue(any("retrain 1/2" in t for _, t in m.events), m.events)

    def test_retrain_gives_up_after_the_second_attempt(self):
        m = self.modem()
        self.lose(m)
        m._retraining = False                                             # pretend it completed
        self.lose(m)
        self.assertEqual(m._retrains, 2)
        m._retraining = False
        self.lose(m)
        self.assertEqual(m.state, "failed")
        self.assertFalse(m.connected)
        self.assertIn("after 2 retrains", m.failed_reason)

    def test_retrain_can_be_turned_off(self):
        m = self.modem(V34_FREEZE_RETRAIN="0")
        self.lose(m)
        self.assertEqual(m.state, "failed")
        self.assertIn("carrier stream lost", m.failed_reason)

    def test_a_retrain_that_does_not_finish_fails_after_the_timeout(self):
        m = self.modem()
        self.lose(m)
        m._t += int((m.RETRAIN_TIMEOUT + 1) * m.fs)
        m.feed_pcm(np.zeros(160, dtype="<i2").tobytes())
        self.assertEqual(m.state, "failed")
        self.assertIn("retrain did not reach data mode", m.failed_reason)

    def test_a_retrain_that_reaches_data_mode_is_reported_complete(self):
        m = self.modem()
        self.lose(m)
        m.caller._step = lambda: None                                      # only the modem's bookkeeping is under test
        m.caller.stage, m.caller._rx_state = "data", {}
        m.caller.data_rate, m.caller.rx_rate = 4800, 4800
        m.feed_pcm(np.zeros(160, dtype="<i2").tobytes())
        self.assertFalse(m._retraining)
        self.assertTrue(any("retrain complete" in t for _, t in m.events), m.events)
        self.assertEqual(m.state, "data")

def tone_a(n=160, amp=3000, t0=0):
    k = np.arange(t0, t0 + n)
    return np.round(amp * (np.sin(2 * np.pi * 2400 * k / 8000) + np.sin(2 * np.pi * 1800 * k / 8000)))


class AnswererRetrainTests(unittest.TestCase):
    def caller(self, enabled=True):
        old = os.environ.get("V34_ANSWERER_RETRAIN")
        if enabled:
            os.environ["V34_ANSWERER_RETRAIN"] = "1"
        else:
            os.environ.pop("V34_ANSWERER_RETRAIN", None)
        self.addCleanup(lambda: os.environ.__setitem__("V34_ANSWERER_RETRAIN", old) if old is not None else os.environ.pop("V34_ANSWERER_RETRAIN", None))
        import v34_call
        c = v34_call.V34Caller(symbol_rates=[2400], data_rate=4800)
        c._step = lambda: None
        c._rx_state = {}
        return c

    def test_tone_a_in_data_mode_raises_the_flag_after_80_ms(self):
        c = self.caller()
        for i in range(3):
            c.rx(tone_a(t0=i * 160))
        self.assertFalse(c.answerer_retrain)                               # 60 ms
        c.rx(tone_a(t0=480))
        self.assertTrue(c.answerer_retrain)
        self.assertEqual(sum("initiated a retrain" in t for _, t in c.events), 1)

    def test_smartlink_retrain_without_guard_is_recognized_after_80_ms(self):
        c = self.caller()
        for i in range(4):
            n = np.arange(160)+160*i
            c.rx(np.round(4000*np.sin(2*np.pi*2400*n/8000)))
            self.assertEqual(c.answerer_retrain, i == 3)

    def test_data_audio_guard_only_other_tones_and_silence_do_not(self):
        c = self.caller()
        for i in range(100):
            c.rx(noise(seed=i))
        n = np.arange(160)
        for frequency in (1200,1800,2100,2300,2500):
            for i in range(20):
                c.rx(np.round(4000 * np.sin(2 * np.pi * frequency * (n + 160 * i) / 8000)))
        for i in range(20):
            c.rx(np.zeros(160))
        self.assertFalse(c.answerer_retrain)

    def test_a_gap_resets_the_run(self):
        c = self.caller()
        for i in range(3):
            c.rx(tone_a(t0=i * 160))
        c.rx(noise(seed=5))
        c.rx(tone_a())
        self.assertFalse(c.answerer_retrain)

    def test_off_by_default(self):
        c = self.caller(enabled=False)
        for i in range(20):
            c.rx(tone_a(t0=i * 160))
        self.assertFalse(c.answerer_retrain)

    def test_the_modem_answers_with_a_fresh_caller_sending_silence_then_tone_b(self):
        m = ModemRetrainTests.modem(self)
        old = m.caller
        old.answerer_retrain = True
        m.feed_pcm(np.zeros(160, dtype="<i2").tobytes())
        self.assertIsNot(m.caller, old)
        self.assertTrue(m._retraining)
        self.assertTrue(m.connected)
        self.assertEqual(m.caller.stage, "info0_ack")
        self.assertEqual(m._retrains, 1)

    def test_a_third_answerer_retrain_ends_the_call(self):
        m = ModemRetrainTests.modem(self)
        for _ in range(2):
            m.caller.answerer_retrain = True
            m.feed_pcm(np.zeros(160, dtype="<i2").tobytes())
            m._retraining = False
        m.caller.answerer_retrain = True
        m.feed_pcm(np.zeros(160, dtype="<i2").tobytes())
        self.assertFalse(m.connected)


if __name__ == "__main__":
    unittest.main()
