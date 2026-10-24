"""INFO0c repeat when INFO0a was lost (V.34 11.2.2.1.1): the caller no longer waits forever."""
import os
import unittest
from unittest.mock import patch
import numpy as np

import v34_ans
import v34_call
import v34_info as vi


def loop(seconds=14.0, delay_ms=25.0, damage=None, rate=4800, syms=(2400,)):
    """In-process caller <-> answerer over a delay line; `damage(tick, chunk)` may alter what the caller receives."""
    call = v34_call.V34Caller(symbol_rates=list(syms), data_rate=rate)
    ans = v34_ans.V34Answerer(symbol_rates=list(syms), data_rate=rate)
    d = int(delay_ms*8)
    to_ans, to_call = [np.zeros(d)], [np.zeros(d)]
    reached = None
    for tick in range(int(seconds*50)):
        to_ans.append(call.tx(160));to_call.append(ans.tx(160))
        xa = np.concatenate(to_ans)[:160];xc = np.concatenate(to_call)[:160]
        to_ans = [np.concatenate(to_ans)[160:]];to_call = [np.concatenate(to_call)[160:]]
        if damage:xc = damage(tick, xc)
        call.rx(xc);ans.rx(xa)
        if reached is None and call.stage == 'data' and ans.stage == 'data':reached = tick*0.02
    return call, ans, reached


def lose_info0a(count=2):
    """Replace the first `count` 20 ms chunks of the answerer's first signal (its INFO0a) with silence, as a lost RTP stretch
    at the start of the frame would (compare the dialup.world call whose INFO0a was damaged at 4.95 s)."""
    state = {'first': None}

    def damage(tick, chunk):
        if state['first'] is None and np.abs(chunk).max() > 10:
            state['first'] = tick
        if state['first'] is not None and tick < state['first'] + count:
            return np.zeros(160)
        return chunk
    return damage


class Info0RecoveryTests(unittest.TestCase):
    def received_repeat(self, call, ack):
        signal = vi.InfoModulator('answer', amplitude=1200, guard=True).modulate(
            vi.build_info0(ack=ack), tail_bits=0)
        call._rx = np.tile(signal, 4)
        call.rx_t = len(call._rx)
        call.info0a = {'ack': 0}
        call._try_info0a()

    def test_repeat_does_not_truncate_inflight_caller_frame(self):
        call = v34_call.V34Caller()
        call.stage = 'info0_ack'
        signal = vi.InfoModulator('call', amplitude=1200, guard=False).modulate(
            vi.build_info0(ack=1), tail_bits=2)
        source = v34_call.ArraySrc(signal)
        source.pos = 17
        call._play(source, [v34_call.ToneSrc(1200, call.amp)])
        self.received_repeat(call, 0)
        self.assertIs(call._src, source)
        self.assertEqual(source.pos, 17)
        self.assertEqual(len(call._queue), 1)

    def test_acknowledged_answer_frames_leave_tone_b_alone(self):
        call = v34_call.V34Caller()
        call.stage = 'info0_ack'
        source = v34_call.ToneSrc(1200, call.amp)
        call._play(source)
        self.received_repeat(call, 1)
        self.assertIs(call._src, source)
        self.assertEqual(call._queue, [])

    def test_old_repeats_do_not_interrupt_confirmed_probing(self):
        call = v34_call.V34Caller()
        call.stage = 'probing'
        call.l1_start = 100
        call.probe_start = 0
        source = v34_call.ToneSrc(1200, call.amp)
        call._play(source)
        self.received_repeat(call, 0)
        self.assertIs(call._src, source)
        self.assertEqual(call.stage, 'probing')

    def test_repeated_answer_frames_survive_lost_first_acknowledgment(self):
        """A recovery peer ignores the first complete ack, then acknowledges the next.

        Decode emitted waveform, rather than count calls to the sender helper.
        The fixture only implements INFO0/Tone A; normal duplex tests cover data.
        """
        call = v34_call.V34Caller()
        repeat = vi.InfoModulator('answer', amplitude=1200, guard=True).modulate(
            vi.build_info0(ack=0), tail_bits=0)
        acknowledge = vi.InfoModulator('answer', amplitude=1200, guard=True).modulate(
            vi.build_info0(ack=1), tail_bits=0)
        sent, peer_pos, ack_pos = [], 0, None
        frames = []
        for tick in range(160):
            sent.extend(call.tx(160))
            # This peer acts on the SECOND complete acknowledgment, modelling
            # loss of the first without changing any of our receiver symbols.
            frames = vi.InfoDemodulator('call', phases=8).decode(np.asarray(sent), 'info0')
            received_acks = []
            for frame in sorted((f for f in frames if f['ack']),
                                key=lambda f:f['_bit_index']+f['_phase']/8):
                position = frame['_bit_index']+frame['_phase']/8
                if not received_acks or position-received_acks[-1] >= 4:
                    received_acks.append(position)
            if len(received_acks) >= 2 and ack_pos is None:
                ack_pos = 0
            if ack_pos is None:
                incoming = repeat[(np.arange(160)+peer_pos) % len(repeat)]
                peer_pos += 160
            else:
                at = np.arange(160)+ack_pos
                incoming = 1200*np.cos(2*np.pi*2400*at/8000)
                inside = at < len(acknowledge)
                incoming[inside] = acknowledge[at[inside]]
                # First Tone A reversal after a complete ack and 80ms tone.
                incoming[at >= len(acknowledge)+640] *= -1
                ack_pos += 160
            call.rx(incoming)
            if call.stage == 'b_done':
                break
        self.assertIsNotNone(ack_pos, 'peer never received a second complete acknowledgment')
        self.assertEqual(call.stage, 'b_done', 'recovery did not resume normal tone reversal exchange')
        self.assertGreaterEqual(len(received_acks), 2)
        self.assertLessEqual(len(call._queue), 1, 'stale acknowledgment frames accumulated')

    def test_undamaged_call_needs_no_repeat(self):
        call, ans, reached = loop()
        self.assertIsNotNone(reached)
        self.assertEqual(call._info0_repeats, 0)

    def test_lost_info0a_freezes_the_caller_without_the_recovery(self):
        with patch.dict(os.environ, {'V34_INFO0_RECOVERY': '0'}):
            call, ans, reached = loop(damage=lose_info0a())
        self.assertIsNone(reached)
        self.assertEqual(call.stage, 'info0')                 # the real failure: waiting for an INFO0a that never comes
        self.assertIsNone(call.info0a)

    def test_lost_info0a_is_recovered_by_repeating_info0c(self):
        call, ans, reached = loop(damage=lose_info0a())
        self.assertIsNotNone(reached, 'caller never reached data')
        self.assertGreaterEqual(call._info0_repeats, 1)
        self.assertLessEqual(call._info0_repeats, call.INFO0_RECOVERY_MAX)
        self.assertIsNotNone(call.info0a)

    def test_answerer_preserves_inflight_b_reversal_across_repeated_info0c(self):
        """A B reversal detected as repeated INFO0c recovery resumes Tone A must not be cleared."""
        answerer = v34_ans.V34Answerer()
        samples = np.arange(8960)
        signal = (answerer.amp * np.cos(2 * np.pi * 1200 * samples / 8000)).astype(float)
        signal[8922:] *= -1
        answerer._rx = signal[:8800].copy()
        answerer.tone_b.feed(answerer._rx)
        answerer.stage = 'toneA'
        answerer._src = v34_ans.ToneASrc(answerer.amp)
        answerer.info0c = {'ack': 1}
        answerer._t_a = 8400
        answerer.a_rev1_t = 5920
        answerer._a_rev1_rx = 5920
        answerer.t = 8960
        answerer.rx_t = 8800

        with patch.object(answerer, '_info0c_repeated', return_value=False):
            answerer.rx(signal[8800:8960])

        self.assertEqual(answerer.tone_b.reversals, [])
        self.assertEqual(answerer.stage, 'a_rev1')
        self.assertEqual(answerer.a_rev1_t, 5920)
        self.assertEqual(answerer.rtd_samples, 8922 - 5920 - 320)
        self.assertEqual(answerer._actions[0][0], 9242)

    def test_repeats_are_bounded_when_the_answerer_never_answers(self):
        call = v34_call.V34Caller()
        call.stage = 'info0'
        call.a_reversals = [1000.0]
        call.rx_t = 1000
        sent = []
        with patch.object(v34_call.V34Caller, '_send_info0', lambda self, ack: sent.append(ack)):
            for k in range(200):
                call.rx_t = 1000 + 400 * k
                call._info0_recovery()
        self.assertEqual(len(sent), call.INFO0_RECOVERY_MAX)
        self.assertEqual(set(sent), {0})                       # bit 28 stays 0: INFO0a was never received

    def test_no_repeat_before_a_tone_a_reversal_or_inside_the_grace(self):
        call = v34_call.V34Caller()
        call.stage = 'info0'
        sent = []
        with patch.object(v34_call.V34Caller, '_send_info0', lambda self, ack: sent.append(ack)):
            call.a_reversals = [1000.0]
            call.rx_t = 1000 + int(0.05*8000)                    # inside the 150 ms grace
            call._info0_recovery()
            self.assertEqual(sent, [])
            call.rx_t = 1000 + int(0.20*8000)
            call._info0_recovery()
            self.assertEqual(sent, [0])


if __name__ == '__main__':
    unittest.main()
