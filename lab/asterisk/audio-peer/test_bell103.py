#!/usr/bin/env python3
"""Self-tests for the lab's Bell 103 FSK endpoint."""

import math
import struct
import random
import shutil
import subprocess
import tempfile
import unittest
import wave
from pathlib import Path

from bell103 import Bell103Receiver, Bell103StreamTransmitter, Bell103Transmitter


def pcm_bytes(samples: list[int]) -> bytes:
    return struct.pack(f"<{len(samples)}h", *samples)


class Bell103Tests(unittest.TestCase):
    def test_rejects_early_bad_stop_without_blocking_overlapping_valid_frame(self):
        rx = Bell103Receiver("originate")
        early_start, valid_start = 0, 60
        rx._start_candidate(early_start)
        rx._start_candidate(valid_start)

        early_targets = rx._candidates[0]["targets"]
        valid_targets = rx._candidates[1]["targets"]
        expected = 0xA5
        tones = {}
        tones[early_targets[0]] = False  # valid start center
        for index, target in enumerate(early_targets[1:9]):
            tones[target] = bool(0x55 & (1 << index))
        tones[early_targets[9]] = False  # The tentative frame has a bad stop bit.
        tones[valid_targets[0]] = False
        for index, target in enumerate(valid_targets[1:9]):
            tones[target] = bool(expected & (1 << index))
        tones[valid_targets[9]] = True

        emitted = bytearray()
        for center in sorted(tones):
            emitted.extend(rx._advance_candidates(center, tones[center]))

        self.assertEqual(emitted, bytes([expected]))

    def test_rejects_false_start_even_when_stop_sample_is_mark(self):
        rx = Bell103Receiver("originate")
        rx._start_candidate(0)
        targets = rx._candidates[0]["targets"]
        emitted = bytearray()
        for center, tone in zip(targets, [True] + [False] * 8 + [True]):
            emitted.extend(rx._advance_candidates(center, tone))
        self.assertEqual(bytes(emitted), b"")
        self.assertEqual(rx._candidates, [])

    def test_bell_103a2_frequency_plan_is_present_on_the_wire(self) -> None:
        """Check Bell 103A2 mark/space assignments independently of our RX."""
        from bell103 import FREQUENCIES

        self.assertEqual(FREQUENCIES["originate"], (1270.0, 1070.0))
        self.assertEqual(FREQUENCIES["answer"], (2225.0, 2025.0))

        def energy(segment: list[int], hz: float) -> float:
            omega = 2.0 * math.pi * hz / 8000.0
            real = sum(x * math.cos(omega * i) for i, x in enumerate(segment))
            imag = sum(x * math.sin(omega * i) for i, x in enumerate(segment))
            return real * real + imag * imag

        for role, expected_mark, expected_space in (
            ("originate", 1270.0, 1070.0),
            ("answer", 2225.0, 2025.0),
        ):
            # The 90-bit lead is mark; the first asynchronous start bit is
            # space. Measure stable interior windows, avoiding both edges.
            tx = Bell103Transmitter(role, b"\x00")
            pcm = tx.samples(3200)
            mark_window = pcm[800:1600]
            space_window = pcm[2500:2700]
            mark_correct = energy(mark_window, expected_mark)
            mark_reversed = energy(mark_window, expected_space)
            space_correct = energy(space_window, expected_space)
            space_reversed = energy(space_window, expected_mark)
            self.assertGreater(mark_correct, 20 * mark_reversed, role)
            self.assertGreater(space_correct, 3 * space_reversed, role)

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

    def test_v21_transmit_characteristic_frequencies_within_six_hz(self) -> None:
        """Measure emitted mark and space from production stream PCM (V.21 §3)."""
        def estimate_frequency(samples: list[int]) -> float:
            # For a single stationary sinusoid, the lag-one normalized
            # autocorrelation is cos(2*pi*f/fs). Mean removal and interior
            # windows avoid the startup/data transitions.
            values = [float(value) for value in samples]
            mean = sum(values) / len(values)
            values = [value - mean for value in values]
            numerator = sum(a * b for a, b in zip(values, values[1:]))
            denominator = sum(value * value for value in values[:-1])
            return math.acos(max(-1.0, min(1.0, numerator / denominator))) * 8000 / (2 * math.pi)

        for role, expected_mark, expected_space in (
            ("originate", 980.0, 1180.0),
            ("answer", 1650.0, 1850.0),
        ):
            with self.subTest(role=role):
                tx = Bell103StreamTransmitter(role, standard="v21")
                mark_pcm = tx.samples(400)
                tx.feed_bytes(b"\x00" * 5)
                data_pcm = tx.samples(1400)
                measured_mark = estimate_frequency(mark_pcm[100:380])
                # The first bit is the asynchronous start bit; the following
                # eight zero data bits form a long stable space interval.
                measured_space = estimate_frequency(data_pcm[70:220])
                self.assertLessEqual(abs(measured_mark - expected_mark), 6.0)
                self.assertLessEqual(abs(measured_space - expected_space), 6.0)

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


class IndependentFskVectorTests(unittest.TestCase):
    """Receive vectors generated by minimodem, independent of our TX path.

    The checked-in files are 8 kHz mono signed-16 PCM converted from FLAC
    generated by minimodem 0.24. The test payloads and exact generation
    commands are documented in lab/MODEM_CORE_P0_PROGRESS_2026_10_22.md.
    """

    FIXTURES = Path(__file__).parent / "fixtures" / "reference" / "independent-fsk"
    VECTORS = (
        ("v21-orig.raw", "v21", "originate", b"ELMOD V21 ORIGINATE 0123456789\r"),
        ("v21-answer.raw", "v21", "answer", b"ELMOD V21 ANSWER 9876543210\r"),
        # V.21 §3 requires demodulation to tolerate ±12 Hz from nominal after
        # the transmitter and line frequency-drift allowances are combined.
        # These were generated independently by minimodem at that boundary.
        ("v21-orig-plus12.raw", "v21", "originate",
         b"ELMOD V21 ORIGINATE PLUS 12 HZ 0123456789\r"),
        ("v21-orig-minus12.raw", "v21", "originate",
         b"ELMOD V21 ORIGINATE MINUS 12 HZ 0123456789\r"),
        ("v21-answer-plus12.raw", "v21", "answer",
         b"ELMOD V21 ANSWER PLUS 12 HZ 9876543210\r"),
        ("v21-answer-minus12.raw", "v21", "answer",
         b"ELMOD V21 ANSWER MINUS 12 HZ 9876543210\r"),
        ("bell103a2-orig.raw", "bell103", "originate", b"ELMOD B103 ORIGINATE 0123456789\r"),
        ("bell103a2-answer.raw", "bell103", "answer", b"ELMOD B103 ANSWER 9876543210\r"),
    )

    def test_independent_minimodem_receive_vectors_both_standards_and_roles(self):
        for filename, standard, remote_role, expected in self.VECTORS:
            with self.subTest(vector=filename):
                pcm = (self.FIXTURES / filename).read_bytes()
                self.assertEqual(len(pcm) % 2, 0)
                rx = Bell103Receiver(remote_role, standard=standard)
                decoded = bytearray()
                # Deliberately use irregular chunk sizes to exercise the
                # streaming receiver independently of the generator's blocks.
                sizes = (314, 480, 202, 638)
                offset = 0
                index = 0
                while offset < len(pcm):
                    size = sizes[index % len(sizes)] * 2
                    decoded.extend(rx.feed_pcm(pcm[offset:offset + size]))
                    offset += size
                    index += 1
                self.assertEqual(bytes(decoded), expected)

    @unittest.skipUnless(shutil.which("minimodem"), "optional independent minimodem is not installed")
    def test_our_transmitters_decode_in_minimodem_both_standards_and_roles(self):
        tool = shutil.which("minimodem")
        payload = b"ELMODEM TX TO MINIMODEM 0123456789\r"
        plans = (
            ("v21", "originate", 980, 1180),
            ("v21", "answer", 1650, 1850),
            ("bell103", "originate", 1270, 1070),
            ("bell103", "answer", 2225, 2025),
        )
        for standard, role, mark_hz, space_hz in plans:
            with self.subTest(standard=standard, role=role):
                tx = Bell103Transmitter(role, payload, standard=standard)
                sample_count = round((90 + 10 * len(payload) + 50) * 8000 / 300)
                with tempfile.NamedTemporaryFile(suffix=".wav") as audio:
                    with wave.open(audio.name, "wb") as wav:
                        wav.setnchannels(1)
                        wav.setsampwidth(2)
                        wav.setframerate(8000)
                        wav.writeframes(pcm_bytes(tx.samples(sample_count)))
                    result = subprocess.run(
                        [tool, "--rx", "--file", audio.name, "--mark", str(mark_hz),
                         "--space", str(space_hz), "300"],
                        capture_output=True, timeout=15, check=False)
                self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
                self.assertEqual(result.stdout, payload, result.stderr.decode(errors="replace"))


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
