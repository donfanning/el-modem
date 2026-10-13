#!/usr/bin/env python3
"""Self-tests for the SpanDSP V.42 (LAPM) binding, wired bit-to-bit."""

import random
import unittest

import spandsp_v42
from spandsp_v42 import V42Link


def run(caller, answerer, bits, on_bit=None, flip=None):
    """Clock two links against each other: one bit each way per step.
    ``flip(i)`` may return True to corrupt the caller->answerer bit."""
    for i in range(bits):
        bc, ba = caller.tx_bit(), answerer.tx_bit()
        if flip and flip(i):
            bc ^= 1
        answerer.rx_bit(bc)
        caller.rx_bit(ba)
        if on_bit:
            on_bit(i)


@unittest.skipUnless(spandsp_v42.available(), "libspandsp not installed")
class V42Tests(unittest.TestCase):
    def test_detection_then_lapm_and_data_both_ways(self):
        caller, answerer = V42Link(True), V42Link(False)
        a_msg = b"hello from the caller " * 40
        b_msg = bytes(range(256)) * 4
        sent = {"done": False}

        def tick(i):
            if caller.state == "lapm" == answerer.state and not sent["done"]:
                caller.write(a_msg)
                answerer.write(b_msg)
                sent["done"] = True

        run(caller, answerer, 60000, tick)
        self.assertEqual((caller.state, answerer.state), ("lapm", "lapm"),
                         (caller.events, answerer.events))
        self.assertEqual(answerer.read(), a_msg)
        self.assertEqual(caller.read(), b_msg)

    def test_data_survives_a_noisy_line(self):
        # LAPM retransmits: corrupt about 1 bit in 2000 and the payload must
        # still arrive intact (this is what the lab ARQ could not give a real
        # peer, and what raw mode cannot give at all).
        rng = random.Random(7)
        caller, answerer = V42Link(True), V42Link(False)
        payload = bytes(rng.randrange(256) for _ in range(1500))
        state = {"sent": False, "armed": False}

        def tick(i):
            if caller.state == "lapm" == answerer.state and not state["sent"]:
                caller.write(payload)
                state.update(sent=True, armed=True)

        # SpanDSP's retransmit timers are counted in bits at an assumed
        # 28800 bps, so recovering from a lost frame takes many bit-times.
        run(caller, answerer, 1500000, tick,
            flip=lambda i: state["armed"] and rng.randrange(4000) == 0)
        self.assertEqual(answerer.read(), payload)

    def test_non_v42_peer_is_reported_unsupported(self):
        caller = V42Link(True)
        # A peer that sends only marks (idle line) never answers the ODP.
        for _ in range(30000):
            caller.tx_bit()
            caller.rx_bit(1)
        self.assertEqual(caller.state, "unsupported", caller.events)


@unittest.skipUnless(spandsp_v42.available(), "libspandsp not installed")
class V42TimerTests(unittest.TestCase):
    def test_corrected_timers_keep_a_multi_window_transfer_near_wire_speed(self):
        # With SpanDSP's default 28800 bps timer assumption a 2400 bps link
        # stalls waiting on timers: a 6000-byte transfer took 361 s against a
        # ~25 s wire minimum. With the line rate set it takes ~31 s.
        rate, nbytes = 2400, 4000  # bigger than the 15 x 128 byte send window
        caller = V42Link(True, bit_rate=rate)
        answerer = V42Link(False, bit_rate=rate)
        if not caller.timers_scaled:
            self.skipTest("libspandsp lacks v42_set_bit_rate (unpatched build)")
        payload = bytes(range(256)) * (nbytes // 256) + bytes(nbytes % 256)
        got, start = bytearray(), None
        for i in range(2_000_000):
            if start is None and caller.state == "lapm" == answerer.state:
                caller.write(payload)
                start = i
            bc, ba = caller.tx_bit(), answerer.tx_bit()
            answerer.rx_bit(bc)
            caller.rx_bit(ba)
            got += answerer.read()
            if len(got) >= nbytes:
                break
        self.assertEqual(bytes(got), payload)
        seconds = (i - start) / rate
        wire_minimum = nbytes * 10 / rate
        self.assertLess(seconds, 2.0 * wire_minimum,
                        f"{seconds:.0f}s for a {wire_minimum:.0f}s transfer")


@unittest.skipUnless(spandsp_v42.available(), "libspandsp not installed")
class V42AckStallTests(unittest.TestCase):
    def test_a_window_full_of_small_frames_does_not_stall_for_the_poll_timer(self):
        # SpanDSP built every ack with the F bit set, and the sender discards
        # such an ack unless a poll is outstanding, so a one-way sender waited
        # for the 10 s T403 timer each time its window filled: 800 bytes in
        # 32-byte frames (25 frames, window 15) took 11 s against a 4 s wire
        # time. spandsp-v42-tuning.patch echoes the P bit instead.
        rate, nbytes = 2400, 800
        caller = V42Link(True, bit_rate=rate, frame_size=32)
        answerer = V42Link(False, bit_rate=rate, frame_size=32)
        if not caller.frame_params_set:
            self.skipTest("libspandsp lacks v42_set_frame_params (unpatched build)")
        payload = bytes(range(256)) * 4
        payload = payload[:nbytes]
        got, start = bytearray(), None
        for i in range(1_000_000):
            if start is None and caller.state == "lapm" == answerer.state:
                caller.write(payload)
                start = i
            bc, ba = caller.tx_bit(), answerer.tx_bit()
            answerer.rx_bit(bc)
            caller.rx_bit(ba)
            got += answerer.read()
            if len(got) >= nbytes:
                break
        self.assertEqual(bytes(got), payload)
        seconds = (i - start) / rate
        self.assertLess(seconds, 2.0 * (nbytes * 10 / rate), f"{seconds:.1f}s")


if __name__ == "__main__":
    unittest.main()
