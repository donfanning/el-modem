#!/usr/bin/env python3
"""Self-tests for caller-side line-mode autodetection."""

import math
import struct
import unittest

import spandsp_v22
from autodetect import AnsGate, AutoCallerLine, tone_fraction
from bell103 import Bell103Receiver, Bell103StreamTransmitter

FRAME = 160


def tone(freq, seconds, amp=8000, phase=0.0):
    n = int(seconds * 8000)
    return [round(amp * math.sin(2 * math.pi * freq * i / 8000 + phase)) for i in range(n)]


def ans_tone(seconds=2.0):
    """V.25 ANS: 2100 Hz with a 180-degree phase reversal every 450 ms."""
    out, t, flip = [], 0.0, 0
    while t < seconds:
        seg = tone(2100, min(0.45, seconds - t), phase=math.pi * flip)
        out += seg
        t += 0.45
        flip ^= 1
    return out


def frames(samples):
    for i in range(0, len(samples) - FRAME + 1, FRAME):
        yield samples[i:i + FRAME]


class ToneTests(unittest.TestCase):
    def test_tone_fraction_discriminates(self):
        s = tone(2100, 0.05)
        self.assertGreater(tone_fraction(s, 2100)[0], 0.9)
        self.assertLess(tone_fraction(s, 2225)[0], 0.1)

    def test_ans_gate_silences_tone_even_across_phase_reversals(self):
        gate = AnsGate()
        out = []
        for f in frames(ans_tone(2.0) + [0] * 1600):
            out += gate.process(f)
        tone_part = out[: 2 * 8000 - FRAME]
        self.assertEqual(max(abs(x) for x in tone_part[FRAME:]), 0)
        self.assertTrue(gate.seen)
        self.assertFalse(gate.active)

    def test_ans_gate_passes_other_audio(self):
        gate = AnsGate()
        s = tone(2225, 0.2)
        out = []
        for f in frames(s):
            out += gate.process(f)
        self.assertEqual(out, s[: len(out)])


def make_line(rate=1200, **kw):
    v22 = spandsp_v22.V22bisModem(calling_party=True, bit_rate=rate)
    return AutoCallerLine(
        v22,
        lambda: Bell103StreamTransmitter("originate"),
        lambda: Bell103Receiver("answer"),
        **kw,
    )


@unittest.skipUnless(spandsp_v22.available(), "libspandsp not installed")
class AutoCallerTests(unittest.TestCase):
    def test_picks_v22_for_a_v22_answerer_after_answer_tone(self):
        line = make_line()
        answerer = spandsp_v22.V22bisModem(calling_party=False)
        rx = bytearray()
        sent = False
        # The answerer plays ANS (+75 ms silence) first, then runs V.22.
        lead = ans_tone(2.0) + [0] * 600
        pos = 0
        for tick in range(50 * 14):
            if pos + FRAME <= len(lead):
                far = lead[pos:pos + FRAME]
                pos += FRAME
            else:
                far = answerer.tx_samples(FRAME)
            rx += line.feed_pcm(struct.pack("<160h", *far))
            ours = line.tx_samples(FRAME)
            if pos + FRAME > len(lead):
                answerer.feed_pcm(struct.pack("<160h", *ours))
            if line.connected and not sent:
                answerer.feed_bytes(b"HELLO\r\n")
                line.feed_bytes(b"AT\r")
                sent = True
        self.assertEqual(line.state, "v22", line.reason)
        self.assertEqual(line.bit_rate, 1200)
        self.assertTrue(rx.endswith(b"HELLO\r\n"), rx)

    def test_ans_leading_edge_leak_does_not_commit_to_v22_during_the_tone(self):
        # The tone starts mid-frame (so its first frame is only partly tone and
        # slips past the gate) and answerer V.22 starts right as ANS ends, as
        # NIST does. The line must stay in "listen" for the whole tone.
        line = make_line()
        answerer = spandsp_v22.V22bisModem(calling_party=False)
        ans = [0] * 83 + ans_tone(3.0)
        ans_frames = len(ans) // FRAME
        states, rx, sent = [], bytearray(), False
        pos = 0
        for tick in range(50 * 16):
            if pos + FRAME <= len(ans):
                far = ans[pos:pos + FRAME]
                pos += FRAME
            else:
                far = answerer.tx_samples(FRAME)
            rx += line.feed_pcm(struct.pack("<160h", *far))
            ours = line.tx_samples(FRAME)
            if pos + FRAME > len(ans):
                answerer.feed_pcm(struct.pack("<160h", *ours))
            if tick < ans_frames:
                states.append(line.state)
            if line.connected and not sent:
                answerer.feed_bytes(b"READY\r\n")
                sent = True
        self.assertEqual(set(states), {"listen"}, "committed during the answer tone")
        self.assertEqual(line.state, "v22", line.reason)
        self.assertTrue(rx.endswith(b"READY\r\n"), rx)

    def test_answer_tone_alone_does_not_connect(self):
        line = make_line()
        for f in frames(ans_tone(3.0)):
            line.feed_pcm(struct.pack("<160h", *f))
            line.tx_samples(FRAME)
        self.assertFalse(line.connected)
        self.assertEqual(line.state, "listen")

    def test_picks_bell103_when_answerer_sends_2225_mark(self):
        line = make_line()
        answerer = Bell103StreamTransmitter("answer")
        answer_rx = Bell103Receiver("originate")
        got = bytearray()
        sent = False
        lead = ans_tone(2.0) + [0] * 600
        pos = 0
        heard = bytearray()
        settled = 0
        for tick in range(50 * 10):
            if pos + FRAME <= len(lead):
                far = lead[pos:pos + FRAME]
                pos += FRAME
            else:
                far = answerer.samples(FRAME)
            got += line.feed_pcm(struct.pack("<160h", *far))
            ours = line.tx_samples(FRAME)
            if line.state == "bell103":
                heard += answer_rx.feed_pcm(struct.pack("<160h", *ours))
                settled += 1
                # A real answerer sends its first data after a moment of idle
                # mark; the receiver needs that to lock onto character timing.
                if not sent and settled >= 25:
                    answerer.feed_bytes(b"Hi there\r\n")
                    line.feed_bytes(b"ping")
                    sent = True
        self.assertEqual(line.state, "bell103", line.reason)
        self.assertEqual(line.bit_rate, 300)
        self.assertTrue(line.connected)
        self.assertIn(b"Hi there", bytes(got))
        self.assertIn(b"ping", bytes(heard))

    def test_silent_answerer_falls_back_to_bell103_after_timeout(self):
        line = make_line(fallback_seconds=2.0)
        for _ in range(50 * 4):
            line.feed_pcm(b"\x00\x00" * FRAME)
            line.tx_samples(FRAME)
        self.assertEqual(line.state, "bell103")
        self.assertIn("trying Bell 103", line.reason)


if __name__ == "__main__":
    unittest.main()
