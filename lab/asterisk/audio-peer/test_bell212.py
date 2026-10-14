#!/usr/bin/env python3
"""Self-tests for the lab's Bell 212A (1200 bps DPSK) endpoint."""

import struct
import random
import unittest

from bell212 import Bell212Receiver, Bell212StreamTransmitter


def pcm_bytes(samples: list[int]) -> bytes:
    return struct.pack(f"<{len(samples)}h", *samples)


def transfer(
    role: str,
    payload: bytes,
    *,
    chunk_size: int = 80,
    noise: int = 0,
    max_samples_factor: int = 400,
) -> bytes:
    """Feed ``payload`` through a transmitter/receiver pair of the given
    role and return whatever the receiver decodes, stopping once the whole
    payload has come back (or a generous sample budget is exhausted)."""
    tx = Bell212StreamTransmitter(role)
    rx = Bell212Receiver(role)
    tx.feed_bytes(payload)
    decoded = bytearray()
    total = 0
    budget = len(payload) * max_samples_factor + 10_000
    rng = random.Random(212) if noise else None
    while total < budget and len(decoded) < len(payload):
        chunk = tx.samples(chunk_size)
        total += chunk_size
        if noise:
            assert rng is not None
            chunk = [
                max(-32768, min(32767, sample + rng.randint(-noise, noise)))
                for sample in chunk
            ]
        decoded.extend(rx.feed_pcm(pcm_bytes(chunk)))
    return bytes(decoded)


class Bell212Tests(unittest.TestCase):
    def test_decodes_ascii_payload_originate_role(self) -> None:
        payload = b"Hello Bell212!"
        self.assertEqual(transfer("originate", payload), payload)

    def test_decodes_ascii_payload_answer_role(self) -> None:
        payload = b"Testing Bell212 answer side!"
        self.assertEqual(transfer("answer", payload), payload)

    def test_all_byte_values_round_trip(self) -> None:
        payload = bytes(range(256))
        self.assertEqual(transfer("originate", payload, max_samples_factor=200), payload)

    def test_random_bytes_round_trip(self) -> None:
        payload = bytes(random.Random(1).randrange(256) for _ in range(500))
        self.assertEqual(transfer("originate", payload, max_samples_factor=200), payload)

    def test_tolerates_bounded_additive_noise(self) -> None:
        payload = b"Noise tolerance test 1234567890"
        self.assertEqual(transfer("originate", payload, noise=300), payload)

    def test_survives_irregular_audio_chunk_sizes(self) -> None:
        payload = b"irregular chunking test payload"
        self.assertEqual(transfer("originate", payload, chunk_size=37), payload)

    def test_full_duplex_stream_pair_exchanges_independent_messages(self) -> None:
        originate_tx = Bell212StreamTransmitter("originate")
        answer_tx = Bell212StreamTransmitter("answer")
        originate_rx = Bell212Receiver("answer")
        answer_rx = Bell212Receiver("originate")
        from_originate = b"modem one -> modem two\r\n"
        from_answer = b"modem two -> modem one\r\n"
        originate_tx.feed_bytes(from_originate)
        answer_tx.feed_bytes(from_answer)

        seen_by_answer_side = bytearray()
        seen_by_originate_side = bytearray()
        for _ in range(400):
            seen_by_answer_side.extend(
                originate_rx.feed_pcm(pcm_bytes(answer_tx.samples(160)))
            )
            seen_by_originate_side.extend(
                answer_rx.feed_pcm(pcm_bytes(originate_tx.samples(160)))
            )

        self.assertEqual(bytes(seen_by_answer_side), from_answer)
        self.assertEqual(bytes(seen_by_originate_side), from_originate)

    def test_idles_on_steady_carrier_then_encodes_bytes_fed_later(self) -> None:
        tx = Bell212StreamTransmitter("originate")
        rx = Bell212Receiver("originate")

        # A stretch of idle (no phase change) carrier before any data is
        # queued must not be misdecoded as a start bit/data frame.
        idle_decoded = rx.feed_pcm(pcm_bytes(tx.samples(4000)))
        self.assertEqual(idle_decoded, b"")

        tx.feed_bytes(b"HELLO")
        decoded = bytearray()
        for _ in range(60):
            decoded.extend(rx.feed_pcm(pcm_bytes(tx.samples(160))))
        self.assertEqual(bytes(decoded), b"HELLO")


if __name__ == "__main__":
    unittest.main()
