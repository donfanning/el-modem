import unittest
import sys
from collections import deque
from pathlib import Path

import numpy as np

from v90_analogue_downstream_rx import (
    V90AnalogueDownstreamPhase3Receiver,
    parse_jd_bits,
)
from v90_analogue_caller_phase3 import (
    AnaloguePhase3Action,
    V90AnalogueCallerPhase3,
)
from v90_analogue_phase3_session import V90AnaloguePhase3Session
from v90_analogue_phase3_tx import V90AnaloguePhase3Transmitter
import v90_info_reference as v90
from v90_digital_tx import (
    build_jd_bits,
    V90DigitalPhase3Transmitter,
    sd_codewords,
    sd_prime_codewords,
)
import v34_info
import v34_g711
from v90_digital_phase3_session import V90DigitalAnswererPhase3Session

import v90_test_ja_waveform as control


class V90AnalogueDownstreamReceiverTests(unittest.TestCase):
    def test_pcmu_marker_accepts_positive_zero_for_negative_zero_positions(self):
        uinfo, law = 90, "pcmu"
        tx = V90DigitalPhase3Transmitter(uinfo, law)
        wire = bytearray(
            sd_codewords(uinfo, law)
            + sd_prime_codewords(uinfo, law)
            + tx.trn1d()
            + tx.jd_frame(1 << 7))
        # Some μ-law RTP paths canonicalize both signed-zero codewords to
        # +0. The physical marker is unchanged, so the receiver must still
        # detect S_d/S'_d and continue decoding the following exact symbols.
        for index in range(384 + 48):
            if wire[index] == 0x7f:
                wire[index] = 0xff
        rx = V90AnalogueDownstreamPhase3Receiver(uinfo, law)
        events = rx.feed(wire, at=0)
        self.assertEqual([event["event"] for event in events],
                         ["sd_mark", "sd_transition",
                          "trn1d_min_complete", "jd"])
        self.assertEqual(events[0]["at"], 0)
        self.assertEqual(events[1]["at"], 384)

    def test_marker_zero_tolerance_is_not_applied_to_alaw(self):
        uinfo, law = 80, "pcma"
        wire = bytearray(sd_codewords(uinfo, law))
        # Flip the sign bit of one zero-level marker octet. The PCMU alias
        # rule must not make this different negotiated-law byte sequence pass.
        wire[1] ^= 0x80
        rx = V90AnalogueDownstreamPhase3Receiver(uinfo, law)
        events = rx.feed(wire, at=0)
        self.assertNotIn("sd_mark", [event["event"] for event in events])

    def test_pcmu_marker_still_rejects_nonzero_codeword_errors(self):
        wire = bytearray(sd_codewords(90, "pcmu"))
        wire[0] ^= 0x01
        rx = V90AnalogueDownstreamPhase3Receiver(90, "pcmu")
        events = rx.feed(wire, at=0)
        self.assertNotIn("sd_mark", [event["event"] for event in events])

    def test_parses_generated_jd_after_chunked_marks_and_extended_training(self):
        uinfo, law = 83, "pcmu"
        rate_mask = (1 << 0) | (1 << 4) | (1 << 21)
        tx = V90DigitalPhase3Transmitter(uinfo, law)
        trn_symbols = 2058
        wire = (sd_codewords(uinfo, law)
                + sd_prime_codewords(uinfo, law)
                + tx.trn1d(trn_symbols)
                + tx.jd_frame(rate_mask, phase3_16point=True,
                              renegotiation_16point=True, max_lookahead=3))
        rx = V90AnalogueDownstreamPhase3Receiver(uinfo, law)

        events = []
        offset = 0
        for size in (1, 127, 160, 511, 2048, 93, 4096):
            if offset >= len(wire):
                break
            chunk = wire[offset:offset + size]
            events.extend(rx.feed(chunk, at=offset))
            offset += len(chunk)
        if offset < len(wire):
            events.extend(rx.feed(wire[offset:], at=offset))

        names = [event["event"] for event in events]
        self.assertEqual(names, ["sd_mark", "sd_transition",
                                 "trn1d_min_complete", "jd"])
        self.assertEqual(events[0]["at"], 0)
        self.assertEqual(events[1]["at"], 384)
        self.assertEqual(events[2]["at"], 384 + 48 + 2040)
        self.assertEqual(events[3]["frame_start"], 384 + 48 + trn_symbols)
        self.assertEqual(events[3]["descriptor"]["rate_mask"], rate_mask)
        self.assertTrue(events[3]["descriptor"]["phase3_16point"])
        self.assertTrue(events[3]["descriptor"]["renegotiation_16point"])
        self.assertEqual(events[3]["descriptor"]["max_lookahead"], 3)

    def test_jprime_wait_accepts_more_than_sixteen_repeated_jd_frames(self):
        uinfo, law = 83, "pcmu"
        tx = V90DigitalPhase3Transmitter(uinfo, law)
        rx = V90AnalogueDownstreamPhase3Receiver(uinfo, law)
        prefix = (sd_codewords(uinfo, law)
                  + sd_prime_codewords(uinfo, law)
                  + tx.trn1d(2040) + tx.jd_frame(1 << 7))
        events = rx.feed(prefix, at=0)
        self.assertIn("jd", [event["event"] for event in events])
        rx.notify_caller_s_sent(len(prefix))

        # The caller must keep listening while S propagates to the digital
        # modem and it completes the current Jd frame. Sixteen frames was an
        # arbitrary short cap that failed on the live SIP loopback.
        continuation = b"".join(tx.jd_frame(1 << 7) for _ in range(18))
        continuation += tx.jprime_d()
        events = rx.feed(continuation, at=len(prefix))
        self.assertIn("jd_prime", [event["event"] for event in events])
        self.assertEqual(rx.codeword_errors, 0)
        self.assertEqual(rx.state, rx.PHASE3_DONE)

    def test_bad_crc_or_alignment_is_not_accepted(self):
        tx = V90DigitalPhase3Transmitter(80, "pcma")
        tx.trn1d()
        good = tx.jd_frame(0x12345)
        bad = bytearray(good)
        bad[8] ^= 0x80
        rx = V90AnalogueDownstreamPhase3Receiver(80, "pcma")
        wire = (sd_codewords(80, "pcma") + sd_prime_codewords(80, "pcma")
                + V90DigitalPhase3Transmitter(80, "pcma").trn1d()
                + bytes(bad))
        events = rx.feed(wire, at=0)
        self.assertNotIn("jd", [event["event"] for event in events])
        self.assertIsNone(parse_jd_bits([0] * 72))

    def test_rejects_reserved_lookahead_value_even_with_valid_crc(self):
        bits = build_jd_bits(0)
        bits[49:51] = [0, 0]
        information = bits[18:34] + bits[35:51]
        crc = v34_info.crc16(information)
        bits[52:68] = [(crc >> index) & 1 for index in range(16)]
        self.assertIsNone(parse_jd_bits(bits))

    def test_rejects_noncontiguous_media_chunks(self):
        rx = V90AnalogueDownstreamPhase3Receiver(80, "pcmu")
        rx.feed(b"abc", at=100)
        with self.assertRaisesRegex(ValueError, "contiguous"):
            rx.feed(b"d", at=104)

    def test_contiguous_octets_allow_rounded_media_clock_chunk_steps(self):
        rx = V90AnalogueDownstreamPhase3Receiver(80, "pcmu")
        rx.feed(b"abc", at=100, sequence_at=1000)
        # A sub-sample clock ratio can round this media step to three even
        # though the next receive sample index correctly advances by 3.
        rx.feed(b"def", at=103, sequence_at=1003)
        with self.assertRaisesRegex(ValueError, "contiguous"):
            rx.feed(b"g", at=104, sequence_at=1007)

    def test_session_reacts_to_received_marks_training_and_jd(self):
        ja = v90.build_ja_descriptor(
            N=0, LSP=1, LTP=1, SP=[0], TP=[0], H=[0] * 8,
            REF=[0] * 8, Ucodes=[])
        descriptor = v90.parse_ja_descriptor(ja)
        controller = V90AnalogueCallerPhase3(
            baud=2400 * 10 / 7, ja_bits=ja, descriptor=descriptor)
        receiver = V90AnalogueDownstreamPhase3Receiver(83, "pcma")
        session = V90AnaloguePhase3Session(controller, receiver)
        session.start(100)

        tx = V90DigitalPhase3Transmitter(83, "pcma")
        trn_symbols = 2040
        downstream = (sd_codewords(83, "pcma")
                      + sd_prime_codewords(83, "pcma")
                      + tx.trn1d(trn_symbols)
                      + tx.jd_frame(1 << 7))
        ja_start = controller._ja_start
        result = session.feed_downstream_g711(downstream, at=ja_start)

        self.assertEqual([item["event"] for item in result],
                         ["sd_mark", "sd_transition",
                          "trn1d_min_complete", "jd"])
        self.assertEqual(controller.state, controller.WAIT_JD_PRIME)
        self.assertEqual([a.signal for a in result[-1]["actions"]],
                         ["S", "listen-jd-prime"])
        self.assertEqual(result[-1]["details"]["descriptor"]["rate_mask"],
                         1 << 7)

    def test_session_handles_jprime_and_requested_dil(self):
        ja = v90.build_ja_descriptor(
            N=1, LSP=1, LTP=1, SP=[0], TP=[0], H=[0] * 8,
            REF=[0] * 8, Ucodes=[3])
        descriptor = v90.parse_ja_descriptor(ja)
        controller = V90AnalogueCallerPhase3(
            baud=2400 * 10 / 7, ja_bits=ja, descriptor=descriptor)
        receiver = V90AnalogueDownstreamPhase3Receiver(83, "pcmu")
        session = V90AnaloguePhase3Session(controller, receiver)
        session.start(0)

        tx = V90DigitalPhase3Transmitter(83, "pcmu")
        prefix = (sd_codewords(83, "pcmu")
                  + sd_prime_codewords(83, "pcmu") + tx.trn1d()
                  + tx.jd_frame(1 << 7))
        stream_at = controller._ja_start
        first = session.feed_downstream_g711(prefix, at=stream_at)
        self.assertEqual(first[-1]["event"], "jd")
        session.notify_caller_s_sent(stream_at + len(prefix))

        continuation = tx.jd_frame(1 << 7) + tx.jprime_d()
        dil = tx.dil_stream(descriptor)
        self.assertIsNotNone(dil)
        continuation += dil.read(controller.dil_symbols)
        second = session.feed_downstream_g711(
            continuation, at=stream_at + len(prefix))

        self.assertEqual([item["event"] for item in second],
                         ["jd_prime", "dil_codewords"])
        self.assertEqual(second[-1]["details"]["received"],
                         controller.dil_symbols)
        self.assertEqual(controller.state, controller.PHASE4_READY)
        self.assertTrue(controller.phase4_ready)

    def test_phase3_transmitter_schedules_reactive_s_waveform(self):
        ja = v90.build_ja_descriptor(
            N=0, LSP=1, LTP=1, SP=[0], TP=[0], H=[0] * 8,
            REF=[0] * 8, Ucodes=[])
        controller = V90AnalogueCallerPhase3(
            baud=2400 * 10 / 7, ja_bits=ja,
            descriptor=v90.parse_ja_descriptor(ja))
        transmitter = V90AnaloguePhase3Transmitter(
            controller, carrier=1800, rms=1000)
        transmitter.start(0)
        until = controller._ja_start + 160
        transmitter.read_pcm(until)
        transmitter.notify_sd_transition(transmitter.cursor)
        begin = transmitter.cursor + 320
        duration = round(128 * 8000 / controller.baud)
        transmitter.schedule_actions([AnaloguePhase3Action(
            begin, "S", duration_samples=duration, symbols=128)])

        np.testing.assert_array_equal(transmitter.read_pcm(320), np.zeros(320))
        response = transmitter.read_pcm(duration)
        self.assertGreater(float(np.max(np.abs(response))), 1.0)
        self.assertEqual(transmitter.drain_completed_actions(),
                         [{"signal": "S", "at": begin + duration}])

    def test_session_renders_reactive_s_sbar_and_final_dil_responses(self):
        ja = v90.build_ja_descriptor(
            N=1, LSP=1, LTP=1, SP=[0], TP=[0], H=[0] * 8,
            REF=[0] * 8, Ucodes=[3])
        controller = V90AnalogueCallerPhase3(
            baud=2400 * 10 / 7, ja_bits=ja,
            descriptor=v90.parse_ja_descriptor(ja))
        transmitter = V90AnaloguePhase3Transmitter(
            controller, carrier=1800, rms=1000)
        receiver = V90AnalogueDownstreamPhase3Receiver(83, "pcmu")
        session = V90AnaloguePhase3Session(controller, receiver, transmitter)
        session.start(0)

        tx = V90DigitalPhase3Transmitter(83, "pcmu")
        initial = (sd_codewords(83, "pcmu")
                   + sd_prime_codewords(83, "pcmu") + tx.trn1d()
                   + tx.jd_frame(1 << 7))
        stream_at = controller._ja_start
        transmitter.read_pcm(stream_at)
        first_events = []
        for offset in range(0, len(initial), 160):
            chunk = initial[offset:offset + 160]
            transmitter.read_pcm(len(chunk))
            first_events.extend(session.feed_downstream_g711(
                chunk, at=stream_at + offset))
        self.assertEqual(first_events[-1]["event"], "jd")
        s_action = first_events[-1]["actions"][0]
        s_audio = transmitter.read_pcm(s_action.duration_samples)
        self.assertGreater(float(np.max(np.abs(s_audio))), 1.0)

        continuation = tx.jd_frame(1 << 7) + tx.jprime_d()
        dil = tx.dil_stream(v90.parse_ja_descriptor(ja))
        continuation += dil.read(controller.dil_symbols)
        next_at = stream_at + len(initial)
        continuation_events = []
        for offset in range(0, len(continuation), 160):
            chunk = continuation[offset:offset + 160]
            transmitter.read_pcm(len(chunk))
            continuation_events.extend(session.feed_downstream_g711(
                chunk, at=next_at + offset))
        self.assertIn("jd_prime", [item["event"]
                                   for item in continuation_events])
        self.assertTrue(controller.phase4_ready)
        final_audio = transmitter.read_pcm(500)
        self.assertGreater(float(np.max(np.abs(final_audio))), 1.0)
        completed = transmitter.drain_completed_actions()
        self.assertIn("Sbar", [item["signal"] for item in completed])

    def test_paired_phase3_sessions_react_to_jd_and_jprime_on_one_clock(self):
        """Join real answerer G.711 with caller parsing/rendering and PCM return."""
        for law in ("pcmu", "pcma"):
            with self.subTest(law=law):
                baud = control.BAUD
                carrier = control.CARRIER
                ja = control._ja_descriptor(0, 19, 31)
                descriptor = v90.parse_ja_descriptor(ja)
                self.assertEqual(descriptor["N"], 0)

                caller_controller = V90AnalogueCallerPhase3(
                    baud=baud, ja_bits=ja, descriptor=descriptor)
                caller_transmitter = V90AnaloguePhase3Transmitter(
                    caller_controller, carrier=carrier, rms=2000.0)
                caller_receiver = V90AnalogueDownstreamPhase3Receiver(67, law)
                caller = V90AnaloguePhase3Session(
                    caller_controller, caller_receiver, caller_transmitter)
                caller.start(0)

                answerer = V90DigitalAnswererPhase3Session(
                    baud=baud, carrier=carrier, law=law, uinfo=67,
                    rate_mask=1 << 7, trn1d_symbols=2040)
                sample_at = 0
                silence = 0xFF if law == "pcmu" else 0xD5
                caller_events = []
                max_ticks = 500

                for _ in range(max_ticks):
                    # Until Ja is accepted, the peer has no Phase 3 codewords.
                    if answerer.transmit_ready:
                        downstream = answerer.read_g711(160, law)
                        self.assertEqual(len(downstream), 160)
                    else:
                        downstream = bytes([silence]) * 160
                    caller_events.extend(caller.feed_downstream_g711(
                        downstream, at=sample_at))

                    # Return caller PCM through the negotiated G.711 law.
                    pcm = caller_transmitter.read_pcm(160)
                    descriptors = answerer.feed_pcm(
                        v34_g711.roundtrip(pcm, law), sample_at=sample_at)
                    self.assertLessEqual(len(descriptors), 1)

                    sample_at += 160
                    if (caller_controller.phase4_ready
                            and answerer.phase4 is not None
                            and answerer._phase4_receive_armed):
                        break

                self.assertLess(sample_at, max_ticks * 160,
                                (law, caller_controller.state,
                                 answerer.transmitter.state,
                                 caller_receiver.codeword_errors))
                caller_names = [item["event"] for item in caller_events]
                self.assertIn("jd", caller_names)
                self.assertIn("jd_prime", caller_names)
                self.assertEqual(caller_controller.state,
                                 caller_controller.PHASE4_READY)
                self.assertEqual(answerer.transmitter.state,
                                 answerer.transmitter.PHASE4_READY)
                self.assertTrue(answerer._phase4_receive_armed)
                self.assertFalse(answerer._ignore_initial_phase3_flip)
                self.assertTrue(any(name == "s-detected"
                                    for name, _ in answerer.events))
                self.assertTrue(any(name == "initial-s-to-sbar"
                                    for name, _ in answerer.events))
                self.assertEqual(caller_receiver.codeword_errors, 0)
                self.assertEqual(caller_transmitter.late_stop_samples, 0)

    def test_paired_phase3_uinfo90_accepts_caller_levels(self):
        """Exercise the captured peer probe's UINFO and Phase 3 level locally."""
        for law in ("pcmu", "pcma"):
            for rms in (1000.0, 2000.0, 4000.0):
                with self.subTest(law=law, rms=rms):
                    baud = control.BAUD
                    carrier = control.CARRIER
                    ja = control._ja_descriptor(0, 19, 31)
                    descriptor = v90.parse_ja_descriptor(ja)
                    caller_controller = V90AnalogueCallerPhase3(
                        baud=baud, ja_bits=ja, descriptor=descriptor)
                    caller_transmitter = V90AnaloguePhase3Transmitter(
                        caller_controller, carrier=carrier, rms=rms)
                    caller_receiver = V90AnalogueDownstreamPhase3Receiver(
                        90, law)
                    caller = V90AnaloguePhase3Session(
                        caller_controller, caller_receiver,
                        caller_transmitter)
                    caller.start(0)

                    answerer = V90DigitalAnswererPhase3Session(
                        baud=baud, carrier=carrier, law=law, uinfo=90,
                        rate_mask=1 << 7, trn1d_symbols=2040)
                    sample_at = 0
                    silence = 0xFF if law == "pcmu" else 0xD5
                    caller_events = []
                    max_ticks = 500

                    for _ in range(max_ticks):
                        if answerer.transmit_ready:
                            downstream = answerer.read_g711(160, law)
                        else:
                            downstream = bytes([silence]) * 160
                        caller_events.extend(caller.feed_downstream_g711(
                            downstream, at=sample_at))

                        pcm = caller_transmitter.read_pcm(160)
                        answerer.feed_pcm(v34_g711.roundtrip(pcm, law),
                                          sample_at=sample_at)
                        sample_at += 160
                        if (caller_controller.phase4_ready
                                and answerer.phase4 is not None
                                and answerer._phase4_receive_armed):
                            break

                    self.assertLess(sample_at, max_ticks * 160,
                                    (law, rms, caller_controller.state,
                                     answerer.transmitter.state,
                                     caller_receiver.codeword_errors))
                    names = [item["event"] for item in caller_events]
                    self.assertIn("sd_transition", names, (law, rms, names))
                    self.assertIn("jd", names)
                    self.assertIn("jd_prime", names)
                    self.assertTrue(any(name == "s-detected"
                                        for name, _ in answerer.events))
                    self.assertTrue(any(name == "initial-s-to-sbar"
                                        for name, _ in answerer.events))
                    self.assertEqual(caller_receiver.codeword_errors, 0)
                    self.assertEqual(caller_transmitter.late_stop_samples, 0)

    def test_paired_phase3_retrains_when_analogue_s_never_arrives(self):
        """Digital endpoint bounds Jd wait at 5100 ms plus measured RTD."""
        law = "pcmu"
        baud, carrier = control.BAUD, control.CARRIER
        ja = control._ja_descriptor(0, 19, 31)
        descriptor = v90.parse_ja_descriptor(ja)
        caller_controller = V90AnalogueCallerPhase3(
            baud=baud, ja_bits=ja, descriptor=descriptor)
        caller_transmitter = V90AnaloguePhase3Transmitter(
            caller_controller, carrier=carrier, rms=2000.0)
        caller = V90AnaloguePhase3Session(
            caller_controller, V90AnalogueDownstreamPhase3Receiver(67, law),
            caller_transmitter)
        caller.start(0)

        rtd_samples = 1600
        answerer = V90DigitalAnswererPhase3Session(
            baud=baud, carrier=carrier, law=law, uinfo=67,
            rate_mask=1 << 7, round_trip_delay_samples=rtd_samples)
        sample_at = 0
        silence = bytes([0xFF]) * 160
        suppressed = False
        failure = None

        for _ in range(700):
            try:
                downstream = (answerer.read_g711(160, law)
                              if answerer.transmit_ready else silence)
            except RuntimeError as exc:
                failure = str(exc)
                break
            caller.feed_downstream_g711(downstream, at=sample_at)
            pcm = caller_transmitter.read_pcm(160)
            if caller_controller.state == caller_controller.WAIT_JD_PRIME:
                suppressed = True
            if suppressed:
                pcm = np.zeros_like(pcm)
            answerer.feed_pcm(v34_g711.roundtrip(pcm, law),
                              sample_at=sample_at)
            sample_at += 160

        self.assertTrue(suppressed)
        self.assertIsNotNone(failure)
        self.assertEqual(answerer.state, answerer.transmitter.FAILED)
        self.assertIn("5100 ms plus RTD", failure)
        self.assertFalse(any(name == "s-detected"
                             for name, _ in answerer.events))
        self.assertTrue(any(name == "phase3-s-timeout"
                            for name, _ in answerer.events))

    def test_paired_phase3_progresses_with_symmetric_and_asymmetric_media_delay(self):
        """RTD-matched asymmetric delay must not look like a missing Phase-3 S."""
        delay_pairs = (
            (0, 0), (1200, 1200), (1600, 1600), (4000, 4000), (8000, 8000),
            (1600, 6400), (6400, 1600),
        )  # Includes 300 ms total RTD, near the measured captured peer range.
        for law in ("pcmu", "pcma"):
            for down_samples, up_samples in delay_pairs:
                with self.subTest(law=law, down_samples=down_samples,
                                  up_samples=up_samples):
                    baud, carrier = control.BAUD, control.CARRIER
                    ja = control._ja_descriptor(0, 19, 31)
                    descriptor = v90.parse_ja_descriptor(ja)
                    caller_controller = V90AnalogueCallerPhase3(
                        baud=baud, ja_bits=ja, descriptor=descriptor)
                    caller_transmitter = V90AnaloguePhase3Transmitter(
                        caller_controller, carrier=carrier, rms=2000.0)
                    caller_receiver = V90AnalogueDownstreamPhase3Receiver(
                        90, law)
                    caller = V90AnaloguePhase3Session(
                        caller_controller, caller_receiver,
                        caller_transmitter)
                    caller.start(0)

                    answerer = V90DigitalAnswererPhase3Session(
                        baud=baud, carrier=carrier, law=law, uinfo=90,
                        rate_mask=1 << 7,
                        round_trip_delay_samples=down_samples + up_samples)
                    silence = 0xFF if law == "pcmu" else 0xD5
                    down_delay = deque(
                        [bytes([silence]) * 160]
                        * (down_samples // 160))
                    up_delay = deque(
                        [np.zeros(160, dtype=float)]
                        * (up_samples // 160))
                    caller_events = []
                    sample_at = 0

                    for _ in range(900):
                        downstream = (
                            answerer.read_g711(160, law)
                            if answerer.transmit_ready else
                            bytes([silence]) * 160)
                        down_delay.append(downstream)
                        caller_events.extend(caller.feed_downstream_g711(
                            down_delay.popleft(), at=sample_at))

                        up_delay.append(caller_transmitter.read_pcm(160))
                        answerer.feed_pcm(
                            v34_g711.roundtrip(up_delay.popleft(), law),
                            sample_at=sample_at)
                        sample_at += 160
                        if (caller_controller.phase4_ready
                                and answerer.phase4 is not None
                                and answerer._phase4_receive_armed):
                            break

                    self.assertLess(sample_at, 900 * 160,
                                    (law, down_samples, up_samples,
                                     caller_controller.state,
                                     answerer.state,
                                     answerer.failure_reason))
                    self.assertEqual(caller_controller.state,
                                     caller_controller.PHASE4_READY)
                    self.assertEqual(answerer.state,
                                     answerer.transmitter.PHASE4_READY)
                    self.assertTrue(answerer._phase4_receive_armed)
                    caller_signals = [event["event"]
                                      for event in caller_events]
                    self.assertIn("jd", caller_signals)
                    self.assertIn("jd_prime", caller_signals)
                    self.assertTrue(any(name == "s-detected"
                                        for name, _ in answerer.events),
                                    answerer.events[-12:])
                    self.assertFalse(any(name == "phase3-s-timeout"
                                         for name, _ in answerer.events),
                                     answerer.events[-12:])

    def test_paired_phase3_sd_transition_delay_uses_caller_deadline(self):
        """Exercise measured-RTD and delayed-Sd cases against the caller wait."""
        for law in ("pcmu", "pcma"):
            for response_delay_samples, expected in (
                    (2000, {1500: True, 3000: True}),
                    (12800, {1500: False, 3000: True})):
                for timeout_ms, expect_ready in expected.items():
                    with self.subTest(law=law,
                                      delay_ms=response_delay_samples / 8,
                                      timeout_ms=timeout_ms):
                        baud, carrier = control.BAUD, control.CARRIER
                        ja = control._ja_descriptor(0, 19, 31)
                        descriptor = v90.parse_ja_descriptor(ja)
                        caller_controller = V90AnalogueCallerPhase3(
                            baud=baud, ja_bits=ja, descriptor=descriptor,
                            sd_timeout_ms=timeout_ms)
                        caller_transmitter = V90AnaloguePhase3Transmitter(
                            caller_controller, carrier=carrier, rms=2000.0)
                        caller_receiver = V90AnalogueDownstreamPhase3Receiver(
                            90, law)
                        caller = V90AnaloguePhase3Session(
                            caller_controller, caller_receiver,
                            caller_transmitter)
                        caller.start(0)

                        # The digital answerer knows the injected one-way
                        # delay; keep its own RTD-based S watchdog out of
                        # the independent caller-side Sd deadline comparison.
                        answerer = V90DigitalAnswererPhase3Session(
                            baud=baud, carrier=carrier, law=law, uinfo=90,
                            rate_mask=1 << 7,
                            round_trip_delay_samples=response_delay_samples)
                        silence = 0xFF if law == "pcmu" else 0xD5
                        delayed_downstream = deque(
                            [bytes([silence]) * 160]
                            * (response_delay_samples // 160))
                        events = []
                        sample_at = 0

                        for _ in range(1200):
                            produced = (answerer.read_g711(160, law)
                                        if answerer.transmit_ready
                                        else bytes([silence]) * 160)
                            delayed_downstream.append(produced)
                            events.extend(caller.feed_downstream_g711(
                                delayed_downstream.popleft(), at=sample_at))

                            answerer.feed_pcm(
                                v34_g711.roundtrip(
                                    caller_transmitter.read_pcm(160), law),
                                sample_at=sample_at)
                            caller_controller.tick(sample_at + 160)
                            sample_at += 160
                            if (caller_controller.state
                                    in (caller_controller.RETRAIN,
                                        caller_controller.FAILED)
                                    or (caller_controller.phase4_ready
                                        and answerer._phase4_receive_armed)):
                                break

                        if expect_ready:
                            self.assertEqual(caller_controller.state,
                                             caller_controller.PHASE4_READY)
                            self.assertTrue(answerer._phase4_receive_armed)
                            self.assertIn(
                                "sd_transition",
                                [event["event"] for event in events])
                            self.assertIn(
                                "jd", [event["event"] for event in events])
                        else:
                            self.assertEqual(caller_controller.state,
                                             caller_controller.RETRAIN)
                            self.assertNotIn(
                                "sd_transition",
                                [event["event"] for event in events])
                            self.assertEqual(answerer.state,
                                             answerer.transmitter.JD)

    def test_paired_phase3_s_watchdog_rtd_bias_transition(self):
        """Bracket the RTD estimate at which delayed S beats the watchdog."""
        actual_down_samples = 40000  # 5 s
        actual_up_samples = 40000    # 5 s
        actual_rtd = actual_down_samples + actual_up_samples
        estimates = (0, 40000, 44000, actual_rtd)

        for law in ("pcmu", "pcma"):
            for estimated_rtd in estimates:
                with self.subTest(law=law, estimated_rtd=estimated_rtd):
                    baud, carrier = control.BAUD, control.CARRIER
                    ja = control._ja_descriptor(0, 19, 31)
                    descriptor = v90.parse_ja_descriptor(ja)
                    caller_controller = V90AnalogueCallerPhase3(
                        baud=baud, ja_bits=ja, descriptor=descriptor)
                    caller_transmitter = V90AnaloguePhase3Transmitter(
                        caller_controller, carrier=carrier, rms=2000.0)
                    caller_receiver = V90AnalogueDownstreamPhase3Receiver(
                        90, law)
                    caller = V90AnaloguePhase3Session(
                        caller_controller, caller_receiver, caller_transmitter)
                    caller.start(0)

                    answerer = V90DigitalAnswererPhase3Session(
                        baud=baud, carrier=carrier, law=law, uinfo=90,
                        rate_mask=1 << 7,
                        round_trip_delay_samples=estimated_rtd)
                    silence = 0xFF if law == "pcmu" else 0xD5
                    down_delay = deque(
                        [bytes([silence]) * 160]
                        * (actual_down_samples // 160))
                    up_delay = deque(
                        [np.zeros(160, dtype=float)]
                        * (actual_up_samples // 160))
                    caller_events = []
                    sample_at = 0

                    for _ in range(1200):
                        downstream = (
                            answerer.read_g711(160, law)
                            if answerer.transmit_ready else
                            bytes([silence]) * 160)
                        down_delay.append(downstream)
                        caller_events.extend(caller.feed_downstream_g711(
                            down_delay.popleft(), at=sample_at))

                        up_delay.append(caller_transmitter.read_pcm(160))
                        answerer.feed_pcm(
                            v34_g711.roundtrip(up_delay.popleft(), law),
                            sample_at=sample_at)
                        sample_at += 160
                        if (answerer.state == answerer.transmitter.FAILED
                                or answerer._phase4_receive_armed):
                            break

                    timed_out = any(name == "phase3-s-timeout"
                                    for name, _ in answerer.events)
                    if estimated_rtd < 44000:
                        self.assertTrue(timed_out, answerer.events[-12:])
                        self.assertEqual(answerer.state,
                                         answerer.transmitter.FAILED)
                        self.assertFalse(answerer._phase4_receive_armed)
                    else:
                        self.assertFalse(timed_out, answerer.events[-12:])
                        self.assertEqual(answerer.state,
                                         answerer.transmitter.PHASE4_READY)
                        self.assertTrue(answerer._phase4_receive_armed)
                        self.assertTrue(any(name == "s-detected"
                                            for name, _ in answerer.events),
                                        answerer.events[-12:])
                        self.assertIn("jd", [event["event"]
                                              for event in caller_events])

    def test_paired_phase3_accepts_longer_trn1d_runs(self):
        """Jd boundary search remains valid after extended training runs."""
        for law in ("pcmu", "pcma"):
            # MICA documents TRN1d as a minimum that can extend to 4.092 s.
            # At 8 kUcode/s, the corresponding 32,736-Ucode run is the useful
            # upper-bound compatibility case; prior coverage stopped at 8,190.
            for trn1d_symbols in (2040, 4098, 8190, 32736):
                with self.subTest(law=law,
                                  trn1d_symbols=trn1d_symbols):
                    baud, carrier = control.BAUD, control.CARRIER
                    ja = control._ja_descriptor(0, 19, 31)
                    descriptor = v90.parse_ja_descriptor(ja)
                    caller_controller = V90AnalogueCallerPhase3(
                        baud=baud, ja_bits=ja, descriptor=descriptor)
                    caller_transmitter = V90AnaloguePhase3Transmitter(
                        caller_controller, carrier=carrier, rms=2000.0)
                    caller_receiver = V90AnalogueDownstreamPhase3Receiver(
                        90, law, trn1d_min_symbols=2040)
                    caller = V90AnaloguePhase3Session(
                        caller_controller, caller_receiver,
                        caller_transmitter)
                    caller.start(0)

                    answerer = V90DigitalAnswererPhase3Session(
                        baud=baud, carrier=carrier, law=law, uinfo=90,
                        rate_mask=1 << 7,
                        trn1d_symbols=trn1d_symbols)
                    silence = 0xFF if law == "pcmu" else 0xD5
                    caller_events = []
                    sample_at = 0

                    for _ in range(900):
                        downstream = (
                            answerer.read_g711(160, law)
                            if answerer.transmit_ready else
                            bytes([silence]) * 160)
                        caller_events.extend(caller.feed_downstream_g711(
                            downstream, at=sample_at))
                        answerer.feed_pcm(
                            v34_g711.roundtrip(
                                caller_transmitter.read_pcm(160), law),
                            sample_at=sample_at)
                        sample_at += 160
                        if (caller_controller.phase4_ready
                                and answerer.phase4 is not None
                                and answerer._phase4_receive_armed):
                            break

                    self.assertLess(sample_at, 900 * 160,
                                    (law, trn1d_symbols,
                                     caller_controller.state,
                                     answerer.state,
                                     answerer.failure_reason))
                    self.assertEqual(caller_controller.state,
                                     caller_controller.PHASE4_READY)
                    self.assertEqual(answerer.state,
                                     answerer.transmitter.PHASE4_READY)
                    self.assertTrue(answerer._phase4_receive_armed)
                    caller_signals = [event["event"]
                                      for event in caller_events]
                    self.assertIn("trn1d_min_complete", caller_signals)
                    self.assertIn("jd", caller_signals)
                    self.assertIn("jd_prime", caller_signals)
                    self.assertTrue(any(name == "s-detected"
                                        for name, _ in answerer.events),
                                    answerer.events[-12:])
                    self.assertFalse(any(name == "phase3-s-timeout"
                                         for name, _ in answerer.events),
                                     answerer.events[-12:])

    def test_paired_phase3_uinfo90_survives_canonicalized_pcmu_zero(self):
        """Reproduce the captured peer zero-sign rewrite across the paired sessions."""
        law = "pcmu"
        baud, carrier = control.BAUD, control.CARRIER
        ja = control._ja_descriptor(0, 19, 31)
        descriptor = v90.parse_ja_descriptor(ja)
        caller_controller = V90AnalogueCallerPhase3(
            baud=baud, ja_bits=ja, descriptor=descriptor)
        caller_transmitter = V90AnaloguePhase3Transmitter(
            caller_controller, carrier=carrier, rms=4000.0)
        caller_receiver = V90AnalogueDownstreamPhase3Receiver(90, law)
        caller = V90AnaloguePhase3Session(
            caller_controller, caller_receiver, caller_transmitter)
        caller.start(0)

        answerer = V90DigitalAnswererPhase3Session(
            baud=baud, carrier=carrier, law=law, uinfo=90,
            rate_mask=1 << 7, trn1d_symbols=2040)
        answerer_offset = 0
        sample_at = 0
        caller_events = []
        for _ in range(500):
            if answerer.transmit_ready:
                downstream = bytearray(answerer.read_g711(160, law))
                # Model the capture's μ-law canonicalization only in S_d and
                # S'_d. Training, Jd, and later codewords remain untouched.
                stop = max(0, min(len(downstream), 432 - answerer_offset))
                for index in range(stop):
                    if downstream[index] == 0x7f:
                        downstream[index] = 0xff
                answerer_offset += len(downstream)
                downstream = bytes(downstream)
                self.assertEqual(len(downstream), 160)
            else:
                downstream = bytes([0xff]) * 160

            caller_events.extend(caller.feed_downstream_g711(
                downstream, at=sample_at))
            pcm = caller_transmitter.read_pcm(160)
            answerer.feed_pcm(v34_g711.roundtrip(pcm, law),
                              sample_at=sample_at)
            sample_at += 160
            if (caller_controller.phase4_ready
                    and answerer.phase4 is not None
                    and answerer._phase4_receive_armed):
                break

        names = [event["event"] for event in caller_events]
        self.assertIn("sd_mark", names)
        self.assertIn("sd_transition", names)
        self.assertIn("jd", names)
        self.assertIn("jd_prime", names)
        self.assertTrue(caller_controller.phase4_ready)
        self.assertEqual(caller_receiver.codeword_errors, 0)
        self.assertEqual(caller_transmitter.late_stop_samples, 0)
        self.assertTrue(answerer._phase4_receive_armed)
        self.assertTrue(any(name == "s-detected"
                            for name, _ in answerer.events))


if __name__ == "__main__":
    unittest.main()
