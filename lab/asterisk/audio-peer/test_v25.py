"""Focused tests for the V.25 automatic-caller ANS/end-silence procedure."""

import math
import struct
import unittest

from bell103 import Bell103StreamTransmitter
from v25 import FRAME_SAMPLES, V25CallerProcedure


def pcm_tone(frequency: float, samples: int, *, amplitude: float = 7000,
             reverse_every: int | None = None, start: int = 0) -> bytes:
    values = []
    for index in range(start, start + samples):
        phase = 2 * math.pi * frequency * index / 8000
        if reverse_every:
            phase += math.pi * (index // reverse_every)
        values.append(round(amplitude * math.sin(phase)))
    return struct.pack(f"<{len(values)}h", *values)


def pcm_silence(samples: int) -> bytes:
    return bytes(samples * 2)


def feed_frames(procedure: V25CallerProcedure, pcm: bytes) -> str:
    frame_bytes = FRAME_SAMPLES * 2
    state = procedure.state
    for offset in range(0, len(pcm), frame_bytes):
        state = procedure.feed_pcm(pcm[offset:offset + frame_bytes])
    return state


class V25CallerProcedureTests(unittest.TestCase):
    def test_answer_tone_with_phase_reversals_then_80ms_silence_completes(self):
        procedure = V25CallerProcedure()
        tone = pcm_tone(2100, int(3.3 * 8000), reverse_every=3600)
        self.assertEqual(feed_frames(procedure, tone), "answer_tone")
        self.assertTrue(procedure.answer_detected)
        self.assertEqual(feed_frames(procedure, pcm_silence(3 * FRAME_SAMPLES)), "answer_tone")
        self.assertEqual(feed_frames(procedure, pcm_silence(FRAME_SAMPLES)), "complete")

    def test_short_answer_tone_does_not_qualify(self):
        procedure = V25CallerProcedure()
        self.assertEqual(feed_frames(procedure, pcm_tone(2100, 4 * FRAME_SAMPLES)),
                         "waiting_answer")
        self.assertEqual(feed_frames(procedure, pcm_silence(4 * FRAME_SAMPLES)),
                         "waiting_answer")

    def test_ans_frequency_tolerance_edges_qualify(self):
        for frequency in (2085.0, 2115.0):
            with self.subTest(frequency=frequency):
                procedure = V25CallerProcedure()
                self.assertEqual(feed_frames(
                    procedure, pcm_tone(frequency, 5 * FRAME_SAMPLES)),
                    "answer_tone")
                self.assertEqual(feed_frames(
                    procedure, pcm_silence(4 * FRAME_SAMPLES)), "complete")

    def test_calling_tone_and_v21_data_do_not_qualify_as_ans(self):
        procedure = V25CallerProcedure()
        self.assertEqual(feed_frames(procedure, pcm_tone(1300, 8000)), "waiting_answer")
        self.assertEqual(feed_frames(procedure, pcm_tone(1650, 8000)), "waiting_answer")
        self.assertEqual(feed_frames(procedure, pcm_tone(1850, 8000)), "waiting_answer")

    def test_comparable_speech_or_service_tone_inhibits_ans_qualification(self):
        # V.25 §11 suggests rejecting ANS when a comparable-level component
        # accompanies it in the speech/service bands.
        for interfering_hz in (1000.0, 3000.0):
            with self.subTest(interfering_hz=interfering_hz):
                procedure = V25CallerProcedure()
                mixed = []
                for i in range(5 * FRAME_SAMPLES):
                    t = i / 8000
                    mixed.append(round(5000 * math.sin(2 * math.pi * 2100 * t)
                                       + 5000 * math.sin(2 * math.pi * interfering_hz * t)))
                pcm = struct.pack(f"<{len(mixed)}h", *mixed)
                self.assertEqual(feed_frames(procedure, pcm), "waiting_answer")

    def test_no_answer_timeout_is_bounded(self):
        procedure = V25CallerProcedure(started_at=10.0, answer_timeout=30.0)
        self.assertEqual(procedure.poll(39.999), "waiting_answer")
        self.assertEqual(procedure.poll(40.0), "failed")

    def test_stalled_stream_after_ans_is_bounded(self):
        procedure = V25CallerProcedure()
        feed_frames(procedure, pcm_tone(2100, 5 * FRAME_SAMPLES))
        self.assertEqual(procedure.state, "answer_tone")
        procedure.answer_tone_started_at = 10.0
        self.assertEqual(procedure.poll(14.5), "failed")

    def test_answer_tone_duration_over_limit_fails(self):
        procedure = V25CallerProcedure()
        feed_frames(procedure, pcm_tone(2100, 8000))
        self.assertEqual(procedure.state, "answer_tone")
        feed_frames(procedure, pcm_tone(2100, int(4.7 * 8000), start=8000))
        self.assertEqual(procedure.state, "failed")

    def test_transmitter_repeats_call_tone_and_stops_to_silence(self):
        tx = Bell103StreamTransmitter("originate", standard="v21",
                                      v25_call_sequence=True)
        tx.feed_bytes(b"DATA")
        first = tx.samples(FRAME_SAMPLES)
        self.assertTrue(any(first))
        tx.samples(int(0.6 * 8000) - FRAME_SAMPLES)
        self.assertEqual(tx.samples(int(1.75 * 8000)), [0] * int(1.75 * 8000))
        repeated = tx.samples(FRAME_SAMPLES)
        self.assertTrue(any(repeated), "V.25 calling tone did not repeat")
        tx.stop_calling_tone()
        self.assertEqual(tx.prologue_remaining, 0)
        self.assertEqual(tx.samples(FRAME_SAMPLES), [0] * FRAME_SAMPLES)
        tx.release_call_sequence()
        resumed = tx.samples(FRAME_SAMPLES)
        self.assertTrue(any(resumed))


if __name__ == "__main__":
    unittest.main()
