#!/usr/bin/env python3
"""Unit tests for the lab's simplified ARQ/framing layer (lapm_lite.py)."""

import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lapm_lite import (
    ArqSession,
    FrameParser,
    crc16_ccitt,
    encode_frame,
    FRAME_I,
    FRAME_RR,
)


class FramingTests(unittest.TestCase):
    def test_encode_decode_round_trip(self) -> None:
        encoded = encode_frame(FRAME_I, 5, b"hello")
        parser = FrameParser()
        frames = parser.feed(encoded)
        self.assertEqual(len(frames), 1)
        self.assertEqual(frames[0].frame_type, FRAME_I)
        self.assertEqual(frames[0].seq, 5)
        self.assertEqual(frames[0].payload, b"hello")

    def test_payload_containing_flag_and_escape_bytes_survives(self) -> None:
        payload = bytes((0x7E, 0x7D, 0x00, 0xFF, 0x7E, 0x7D))
        encoded = encode_frame(FRAME_I, 1, payload)
        frames = FrameParser().feed(encoded)
        self.assertEqual(frames[0].payload, payload)

    def test_corrupted_frame_is_dropped_not_misdelivered(self) -> None:
        encoded = bytearray(encode_frame(FRAME_I, 2, b"data"))
        # Flip a bit in the middle of the stuffed body (not a flag byte).
        encoded[2] ^= 0x01
        frames = FrameParser().feed(bytes(encoded))
        self.assertEqual(frames, [])

    def test_crc_detects_single_bit_errors(self) -> None:
        body = b"\x49\x03payload"
        good = crc16_ccitt(body)
        for bit in range(8 * len(body)):
            corrupted = bytearray(body)
            corrupted[bit // 8] ^= 1 << (bit % 8)
            self.assertNotEqual(crc16_ccitt(bytes(corrupted)), good)


class ArqSessionTests(unittest.TestCase):
    def _run_pair(
        self,
        payload_a: bytes,
        payload_b: bytes,
        drop_probability: float,
        seed: int,
        max_iterations: int = 20000,
    ) -> tuple[bytes, bytes]:
        rng = random.Random(seed)
        a = ArqSession(max_payload=8, retransmit_timeout=0.2)
        b = ArqSession(max_payload=8, retransmit_timeout=0.2)
        a.feed_terminal_bytes(payload_a)
        b.feed_terminal_bytes(payload_b)
        received_by_a = bytearray()
        received_by_b = bytearray()
        now = 0.0
        step = 0.01
        for _ in range(max_iterations):
            now += step
            out_a = a.pump_outgoing(now)
            out_b = b.pump_outgoing(now)
            if out_a and rng.random() >= drop_probability:
                received_by_b.extend(b.feed_received(out_a))
            if out_b and rng.random() >= drop_probability:
                received_by_a.extend(a.feed_received(out_b))
            if a.idle() and b.idle() and not out_a and not out_b:
                break
        return bytes(received_by_a), bytes(received_by_b)

    def test_clean_channel_delivers_both_directions_exactly(self) -> None:
        received_a, received_b = self._run_pair(b"PING" * 10, b"PONG" * 10, 0.0, 1)
        self.assertEqual(received_b, b"PING" * 10)
        self.assertEqual(received_a, b"PONG" * 10)

    def test_survives_lossy_channel_byte_exact(self) -> None:
        payload = bytes(range(256)) * 4
        received_a, received_b = self._run_pair(
            payload, b"", drop_probability=0.15, seed=42
        )
        self.assertEqual(received_b, payload)

    def test_survives_lossy_channel_both_directions(self) -> None:
        payload_a = os.urandom(500)
        payload_b = os.urandom(500)
        received_a, received_b = self._run_pair(
            payload_a, payload_b, drop_probability=0.1, seed=7
        )
        self.assertEqual(received_b, payload_a)
        self.assertEqual(received_a, payload_b)

    def test_idle_reports_false_until_fully_acknowledged(self) -> None:
        session = ArqSession(max_payload=4)
        self.assertTrue(session.idle())
        session.feed_terminal_bytes(b"data")
        self.assertFalse(session.idle())
        frame = session.pump_outgoing(0.0)
        self.assertFalse(session.idle())
        peer = ArqSession(max_payload=4)
        ack = peer.feed_received(frame)
        self.assertEqual(ack, b"data")
        reply = peer.pump_outgoing(0.0)
        session.feed_received(reply)
        self.assertTrue(session.idle())


if __name__ == "__main__":
    unittest.main()
