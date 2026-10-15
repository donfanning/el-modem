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

    def test_v21_answer_guard_tone_is_present_and_data_still_decodes(self) -> None:
        # ITU-T V.21 answerers carry a continuous 1800 Hz guard tone (echo
        # suppressor disabling for international circuits). It must be
        # energetically present without breaking the 1650/1850 mark/space
        # decision at the originate receiver.
        import math
        answer_tx = Bell103StreamTransmitter("answer", standard="v21",
                                             guard_hz=1800.0, guard_amp=6000.0)
        caller_tx = Bell103StreamTransmitter("originate", standard="v21")
        caller_rx = Bell103Receiver("answer", standard="v21")
        answer_rx = Bell103Receiver("originate", standard="v21")
        answer_tx.feed_bytes(b"GUARDED ANSWER abc\r")
        caller_tx.feed_bytes(b"CALLER def\r")
        caller_data, answer_data = bytearray(), bytearray()
        guard_windows = 0

        def energy(samples, hz):
            w = 2.0 * math.pi * hz / 8000
            c = 2.0 * math.cos(w)
            s0 = s1 = s2 = 0.0
            for x in samples:
                s0 = x + c * s1 - s2
                s2, s1 = s1, s0
            return s1 * s1 + s2 * s2 - c * s1 * s2

        for _ in range(250):
            a = answer_tx.samples(160)
            e1800 = energy(a, 1800)
            e1650 = energy(a, 1650)
            e1850 = energy(a, 1850)
            if e1800 > 0.15 * (e1650 + e1850):
                guard_windows += 1
            caller_data.extend(caller_rx.feed_pcm(pcm_bytes(a)))
            answer_data.extend(answer_rx.feed_pcm(pcm_bytes(caller_tx.samples(160))))
        self.assertGreater(guard_windows, 200, "1800 Hz guard tone not energetically present")
        self.assertEqual(bytes(caller_data), b"GUARDED ANSWER abc\r")
        self.assertEqual(bytes(answer_data), b"CALLER def\r")

    def test_no_guard_tone_without_opt_in(self) -> None:
        import math
        tx = Bell103StreamTransmitter("answer", standard="v21")
        a = tx.samples(800)
        w = 2.0 * math.pi * 1800 / 8000
        c = 2.0 * math.cos(w)
        s0 = s1 = s2 = 0.0
        for x in a:
            s0 = x + c * s1 - s2
            s2, s1 = s1, s0
        e1800 = s1 * s1 + s2 * s2 - c * s1 * s2
        self.assertAlmostEqual(e1800, 0.0, delta=1e6)

    def test_v25_answer_prologue_timing_tone_and_reversals(self) -> None:
        # V.25 §4.3: 1.8–2.5 s billing silence, 2100±15 Hz for 3.3±0.7 s with
        # 180° reversals every 425–475 ms, 75±20 ms silence, then data.
        import math
        tx = Bell103StreamTransmitter("answer", standard="v21",
                                      guard_hz=1800.0, guard_amp=6000.0,
                                      v25_answer=True)
        tx.feed_bytes(b"HI\r")
        out = tx.samples(int(6.5 * 8000))

        def energy(seg, hz):
            w = 2.0 * math.pi * hz / 8000
            c = 2.0 * math.cos(w)
            s0 = s1 = s2 = 0.0
            for x in seg:
                s0 = x + c * s1 - s2
                s2, s1 = s1, s0
            return s1 * s1 + s2 * s2 - c * s1 * s2

        def rms(seg):
            return (sum(x * x for x in seg) / len(seg)) ** 0.5

        self.assertLess(rms(out[:16000]), 1.0, "billing silence should be silent")
        tone = out[16000:16000 + 26400]
        self.assertGreater(energy(tone, 2100), 0.5 * energy(tone, 1650),
                           "answer tone should dominate at 2100 Hz")
        # Adjacent 450 ms blocks of the tone must anti-correlate: the 180°
        # reversals flip the sign block to block.
        def block(k):
            return tone[k * 3600:(k + 1) * 3600]
        dots = [sum(a * b for a, b in zip(block(k), block(k + 1))) for k in range(6)]
        self.assertTrue(all(d < 0 for d in dots), f"reversals missing: {dots}")
        gap = out[16000 + 26400:16000 + 26400 + 600]
        self.assertLess(rms(gap), 1.0, "post-tone silence should be silent")
        # Then modulated data with the guard tone on top.
        data = out[16000 + 26400 + 600:]
        self.assertGreater(energy(data, 1650), energy(data, 2100))
        self.assertGreater(energy(data, 1800), 0.15 * (energy(data, 1650) + energy(data, 1850)))

    def test_v25_prologue_then_data_flows_to_a_waiting_originate(self) -> None:
        # The originate side hears 5.4 s of prologue before the mark; it must
        # simply wait, then decode normally once data starts. Each 180-degree
        # reversal momentarily breaks the mark energy and reads as one 0xFF
        # framing artifact (7 reversals in 3.3 s); pre-data line garbage is
        # normal on PSTN and resync happens on the real banner, so the
        # assertion pins the exact artifact. But the first real character
        # must survive: silence retimes tone tracking (see the None branch in
        # feed_pcm), so the onset chatter cannot wedge the framer through it.
        answer_tx = Bell103StreamTransmitter("answer", standard="v21",
                                             guard_hz=1800.0, guard_amp=6000.0,
                                             v25_answer=True)
        caller_rx = Bell103Receiver("answer", standard="v21")
        answer_tx.feed_bytes(b"AFTER PROLOGUE ok\r")
        got = bytearray()
        for _ in range(500):
            got.extend(caller_rx.feed_pcm(pcm_bytes(answer_tx.samples(160))))
        self.assertEqual(bytes(got), b"\xff" * 7 + b"AFTER PROLOGUE ok\r")

    def test_v25_answer_rejected_for_originate_role(self) -> None:
        with self.assertRaises(ValueError):
            Bell103StreamTransmitter("originate", standard="v21", v25_answer=True)

    def test_calling_tone_bursts_then_mark(self) -> None:
        # Opt-in V.25 calling tone (commercial `^`): 1300 Hz bursts 0.6 s ON /
        # 1.75 s OFF, then the normal mark idle. Off unless asked.
        import math
        tx = Bell103StreamTransmitter("originate", standard="v21", calling_tone=2)
        tx.feed_bytes(b"Q\r")
        out = tx.samples(int(6.0 * 8000))

        def energy(seg, hz):
            w = 2.0 * math.pi * hz / 8000
            c = 2.0 * math.cos(w)
            s0 = s1 = s2 = 0.0
            for x in seg:
                s0 = x + c * s1 - s2
                s2, s1 = s1, s0
            return s1 * s1 + s2 * s2 - c * s1 * s2

        def rms(seg):
            return (sum(x * x for x in seg) / len(seg)) ** 0.5

        on, off = int(0.6 * 8000), int(1.75 * 8000)
        for burst in range(2):
            tone = out[burst * (on + off):burst * (on + off) + on]
            gap = out[burst * (on + off) + on:(burst + 1) * (on + off)]
            self.assertGreater(energy(tone, 1300), 0.5 * energy(tone, 980))
            self.assertLess(rms(gap), 1.0)
        # Then mark idle and normal data to a waiting answerer.
        rx = Bell103Receiver("originate", standard="v21")
        got = bytearray()
        for i in range(2 * (on + off), len(out), 160):
            got.extend(rx.feed_pcm(pcm_bytes(out[i:i + 160])))
        self.assertEqual(bytes(got), b"Q\r")

    def test_calling_tone_rejected_for_answer_role(self) -> None:
        with self.assertRaises(ValueError):
            Bell103StreamTransmitter("answer", standard="v21", calling_tone=1)

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
