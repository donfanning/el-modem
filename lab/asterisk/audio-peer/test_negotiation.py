#!/usr/bin/env python3
"""Self-tests for the lab's simplified bidirectional mode negotiation."""

import unittest

from negotiation import (
    KIND_OFFER,
    MODE_BELL103,
    MODE_BELL212,
    FrameParser,
    ModeNegotiator,
    choose_mode,
    encode_frame,
    line_speed,
)


def exchange(a, b, a_start, b_start, now=0.0, drop=lambda frame_no, side: False):
    """Deliver bytes between two negotiators until both go quiet."""
    queue = [("b", a_start), ("a", b_start)]  # (destination, bytes)
    frame_no = 0
    while queue:
        dest, data = queue.pop(0)
        frame_no += 1
        if drop(frame_no, dest) or not data:
            continue
        target, name = (a, "b") if dest == "a" else (b, "a")
        reply = target.feed(data, now)
        if reply:
            queue.append((name, reply))


class FrameTests(unittest.TestCase):
    def test_round_trip(self) -> None:
        frames = FrameParser().feed(encode_frame(KIND_OFFER, b"\x01\x02"))
        self.assertEqual(frames, [(KIND_OFFER, b"\x01\x02")])

    def test_ignores_noise_and_split_delivery(self) -> None:
        parser = FrameParser()
        data = b"\x00\xff\x16junk" + encode_frame(KIND_OFFER, b"\x01") + b"\x16"
        out = []
        for byte in data:
            out.extend(parser.feed(bytes((byte,))))
        self.assertEqual(out, [(KIND_OFFER, b"\x01")])

    def test_bad_checksum_dropped_and_next_frame_recovered(self) -> None:
        bad = bytearray(encode_frame(KIND_OFFER, b"\x01\x02"))
        bad[-1] ^= 0xFF
        good = encode_frame(KIND_OFFER, b"\x01")
        self.assertEqual(FrameParser().feed(bytes(bad) + good), [(KIND_OFFER, b"\x01")])

    def test_choose_mode_picks_highest_common(self) -> None:
        self.assertEqual(choose_mode((1, 2), (1, 2)), MODE_BELL212)
        self.assertEqual(choose_mode((1, 2), (1,)), MODE_BELL103)
        self.assertEqual(choose_mode((1,), (1, 2)), MODE_BELL103)
        self.assertEqual(line_speed(MODE_BELL212), 1200)


class NegotiatorTests(unittest.TestCase):
    def test_bell103_must_be_supported(self) -> None:
        with self.assertRaises(ValueError):
            ModeNegotiator(supported=(MODE_BELL212,))

    def test_both_sides_agree_on_bell212(self) -> None:
        a, b = ModeNegotiator(), ModeNegotiator()
        exchange(a, b, a.start(0.0), b.start(0.0))
        self.assertTrue(a.complete and b.complete)
        self.assertEqual((a.mode, b.mode), (MODE_BELL212, MODE_BELL212))

    def test_falls_to_lowest_common_mode_either_direction(self) -> None:
        for sa, sb in (((1, 2), (1,)), ((1,), (1, 2))):
            a, b = ModeNegotiator(sa), ModeNegotiator(sb)
            exchange(a, b, a.start(0.0), b.start(0.0))
            self.assertEqual((a.mode, b.mode), (MODE_BELL103, MODE_BELL103))
            self.assertFalse(a.fell_back)

    def test_lost_offer_is_resent_and_still_converges(self) -> None:
        a, b = ModeNegotiator(retry_seconds=1.0), ModeNegotiator(retry_seconds=1.0)
        a_start, b_start = a.start(0.0), b.start(0.0)
        # b never hears a's first OFFER.
        exchange(a, b, a_start, b_start, drop=lambda n, dest: dest == "b" and n == 1)
        self.assertFalse(b.peer_confirmed and a.peer_confirmed)
        retry = b.poll(1.5), a.poll(1.5)  # b resends nothing it already heard; a resends
        self.assertTrue(retry[1] or retry[0])
        exchange(a, b, retry[1], retry[0], now=1.5)
        self.assertTrue(a.complete and b.complete)
        self.assertEqual((a.mode, b.mode), (MODE_BELL212, MODE_BELL212))

    def test_silent_peer_falls_back_to_bell103(self) -> None:
        a = ModeNegotiator(timeout_seconds=3.0)
        a.start(0.0)
        self.assertIsNone(a.mode)
        a.poll(2.0)
        self.assertFalse(a.complete)
        a.poll(3.5)
        self.assertTrue(a.complete and a.fell_back)
        self.assertEqual(a.mode, MODE_BELL103)

    def test_poll_before_start_is_inert(self) -> None:
        a = ModeNegotiator()
        self.assertEqual(a.poll(100.0), b"")
        self.assertFalse(a.complete)


if __name__ == "__main__":
    unittest.main()
