#!/usr/bin/env python3
"""Self-tests for the lab's Bell 103 FSK endpoint."""

import struct
import random
import unittest

from bell103 import Bell103Receiver, Bell103StreamTransmitter, Bell103Transmitter


def pcm_bytes(samples: list[int]) -> bytes:
    return struct.pack(f"<{len(samples)}h", *samples)


class Bell103Tests(unittest.TestCase):
    def test_pair_decodes_payload_in_both_directions(self) -> None:
        originate_payload = b"ORIGINATE 0123456789\r"
        answer_payload = b"ANSWER abcdefghij\r"
        caller_tx = Bell103Transmitter("originate", originate_payload)
        answer_tx = Bell103Transmitter("answer", answer_payload)
        caller_rx = Bell103Receiver("answer")
        answer_rx = Bell103Receiver("originate")
        caller_data = bytearray()
        answer_data = bytearray()

        for _ in range(250):
            caller_data.extend(caller_rx.feed_pcm(pcm_bytes(answer_tx.samples(160))))
            answer_data.extend(answer_rx.feed_pcm(pcm_bytes(caller_tx.samples(160))))

        self.assertEqual(bytes(caller_data), answer_payload)
        self.assertEqual(bytes(answer_data), originate_payload)

    def test_v21_tones_decode_in_both_directions_and_with_noise(self) -> None:
        from bell103 import V21_FREQUENCIES
        self.assertEqual(V21_FREQUENCIES["originate"], (980.0, 1180.0))
        self.assertEqual(V21_FREQUENCIES["answer"], (1650.0, 1850.0))
        originate_payload = b"V21 CALLER 0123456789\r"
        answer_payload = b"V21 ANSWER abcdefghij\r"
        caller_tx = Bell103Transmitter("originate", originate_payload, standard="v21")
        answer_tx = Bell103Transmitter("answer", answer_payload, standard="v21")
        caller_rx = Bell103Receiver("answer", standard="v21")
        answer_rx = Bell103Receiver("originate", standard="v21")
        caller_data, answer_data = bytearray(), bytearray()
        noise = random.Random(21)
        for _ in range(250):
            a = [x + noise.randint(-1000, 1000) for x in answer_tx.samples(160)]
            b = [x + noise.randint(-1000, 1000) for x in caller_tx.samples(160)]
            caller_data.extend(caller_rx.feed_pcm(pcm_bytes(a)))
            answer_data.extend(answer_rx.feed_pcm(pcm_bytes(b)))
        self.assertEqual(bytes(caller_data), answer_payload)
        self.assertEqual(bytes(answer_data), originate_payload)

    def test_modest_baud_error_and_white_noise(self) -> None:
        originate_payload = b"CALLER 987654321\r"
        answer_payload = b"ANSWER 123456789\r"
        caller_tx = Bell103Transmitter("originate", originate_payload, baud=297)
        answer_tx = Bell103Transmitter("answer", answer_payload, baud=303)
        caller_rx = Bell103Receiver("answer")
        answer_rx = Bell103Receiver("originate")
        caller_data = bytearray()
        answer_data = bytearray()
        noise = random.Random(103)

        for _ in range(250):
            caller_samples = answer_tx.samples(160)
            answer_samples = caller_tx.samples(160)
            caller_samples = [sample + noise.randint(-1200, 1200) for sample in caller_samples]
            answer_samples = [sample + noise.randint(-1200, 1200) for sample in answer_samples]
            caller_data.extend(caller_rx.feed_pcm(pcm_bytes(caller_samples)))
            answer_data.extend(answer_rx.feed_pcm(pcm_bytes(answer_samples)))

        self.assertEqual(bytes(caller_data), answer_payload)
        self.assertEqual(bytes(answer_data), originate_payload)

    def test_binary_bytes_survive_irregular_audio_chunks(self) -> None:
        originate_payload = b"\x00\xff\x55\xaa\x01\x80"
        answer_payload = b"\x13\x37\x00\xff\x81\x7e"
        caller_tx = Bell103Transmitter("originate", originate_payload)
        answer_tx = Bell103Transmitter("answer", answer_payload)
        caller_rx = Bell103Receiver("answer")
        answer_rx = Bell103Receiver("originate")
        caller_data = bytearray()
        answer_data = bytearray()
        chunk_sizes = (157, 173, 80, 239, 160)

        for index in range(50):
            count = chunk_sizes[index % len(chunk_sizes)]
            caller_data.extend(caller_rx.feed_pcm(pcm_bytes(answer_tx.samples(count))))
            answer_data.extend(answer_rx.feed_pcm(pcm_bytes(caller_tx.samples(count))))

        self.assertEqual(bytes(caller_data), answer_payload)
        self.assertEqual(bytes(answer_data), originate_payload)


class Bell103StreamTransmitterTests(unittest.TestCase):
    def test_idles_on_mark_then_encodes_bytes_fed_later(self) -> None:
        tx = Bell103StreamTransmitter("originate")
        rx = Bell103Receiver("originate")

        # A stretch of idle mark tone before any data is queued must not be
        # misdecoded as a start bit/data frame.
        idle_decoded = rx.feed_pcm(pcm_bytes(tx.samples(4000)))
        self.assertEqual(idle_decoded, b"")

        tx.feed_bytes(b"HELLO")
        decoded = bytearray()
        for _ in range(60):
            decoded.extend(rx.feed_pcm(pcm_bytes(tx.samples(160))))
        self.assertEqual(bytes(decoded), b"HELLO")

    def test_bytes_fed_incrementally_in_small_bursts_decode_in_order(self) -> None:
        tx = Bell103StreamTransmitter("answer")
        rx = Bell103Receiver("answer")
        message = b"the quick brown fox 0123456789"
        decoded = bytearray()
        for index, byte in enumerate(message):
            tx.feed_bytes(bytes([byte]))
            for _ in range(6):
                decoded.extend(rx.feed_pcm(pcm_bytes(tx.samples(160))))
        # Drain any bits still in flight once the message is fully queued.
        for _ in range(20):
            decoded.extend(rx.feed_pcm(pcm_bytes(tx.samples(160))))
        self.assertEqual(bytes(decoded), message)

    def test_full_duplex_stream_pair_exchanges_independent_messages(self) -> None:
        originate_tx = Bell103StreamTransmitter("originate")
        answer_tx = Bell103StreamTransmitter("answer")
        originate_rx = Bell103Receiver("answer")
        answer_rx = Bell103Receiver("originate")
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


if __name__ == "__main__":
    unittest.main()
