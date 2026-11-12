import math
import unittest
from unittest.mock import patch

import numpy as np
import v34_data
import v34_tx
import v34_g711 as g711
from v90_analogue_upstream_tx import V90AnalogueUpstreamTransmitter
from v90_analogue_phase4 import (V90AnaloguePhase4Controller,
                                 V90AnaloguePhase4Encoder,
                                 V90AnaloguePhase4MediaTx, qam_symbols)
from v90_cp import V90CpSymbolDecoder, build_cp_frame, parse_cp
from v90_digital_phase4_session import V90DigitalAnswererPhase4Session
from v90_digital_phase4 import (V90DigitalPhase4Encoder, ri_codewords,
                                rbar_i_codewords)
from v90_digital_phase4_rx import V90DigitalPhase4Receiver
from v90_dil import g711_ucode_codewords
from v90_lapm import V90LapmLink
from v90_mp_policy import answerer_mp_profile
from v90_upstream_data import V90UpstreamDataReceiver


class V90AnaloguePhase4EncoderTests(unittest.TestCase):
    @staticmethod
    def fields(kind, *, law="pcmu", ack=1, drn=7, codec_masks=None,
               analog_rate_mask=0x109):
        return dict(
            kind=kind, drn=drn, selectors=(0, 1, 2, 0, 1, 2),
            tx_masks=(tuple(range(0, 16)), tuple(range(16, 48)),
                      tuple(range(80, 128))),
            codec_masks=codec_masks, law=law, ack=ack,
            spectral_sign_bits=0, analog_rate_mask=analog_rate_mask,
            lookahead=1, gain_q13=0x4000,
            shape_q6=(0x10, -16, 0x08, -24))

    def test_built_cpt_cp_and_cp_prime_round_trip_through_symbol_decoder(self):
        for constellation in (4, 16):
            for law in ("pcmu", "pcma"):
                with self.subTest(constellation=constellation, law=law):
                    encoder = V90AnaloguePhase4Encoder(constellation)
                    decoder = V90CpSymbolDecoder(constellation)
                    cpt_bits, cpt_symbols = encoder.encode_cp_frame(
                        **self.fields("cpt", law=law))
                    cp_bits, cp_symbols = encoder.encode_cp_frame(
                        **self.fields("cp", law=law, ack=0,
                                      drn=5,
                                      codec_masks=((2, 3), (10, 11), (110, 111))))
                    cp_prime_bits, cp_prime_symbols = encoder.encode_cp_frame(
                        **self.fields("cp", law=law, ack=1, drn=5,
                                      codec_masks=((2, 3), (10, 11), (110, 111))))
                    parsed = []
                    stream = cpt_symbols + cp_symbols + cp_prime_symbols
                    for offset in range(0, len(stream), 19):
                        parsed.extend(decoder.feed(stream[offset:offset + 19]))
                    self.assertEqual([item["kind"] for item in parsed],
                                     ["cpt", "cp", "cp"])
                    for index, bits in enumerate(
                            (cpt_bits, cp_bits, cp_prime_bits)):
                        received = dict(parsed[index])
                        received.pop("start_bit")
                        self.assertEqual(received, parse_cp(bits))
                    self.assertEqual(parsed[0]["codec"], law)
                    self.assertEqual(parsed[1]["ack"], 0)
                    self.assertEqual(parsed[2]["ack"], 1)
                    self.assertEqual(parsed[1]["codec_constellations"],
                                     ((2, 3), (10, 11), (110, 111)))

    def test_frame_builder_rejects_inconsistent_or_invalid_profiles(self):
        fields = self.fields("cpt")
        fields["selectors"] = (0, 1, 2, 0, 1)
        with self.assertRaisesRegex(ValueError, "six constellation"):
            build_cp_frame(**fields)

        fields = self.fields("cp")
        fields["tx_masks"] = ((0, 0), (16, 47), (80, 127))
        with self.assertRaisesRegex(ValueError, "unique nonempty"):
            build_cp_frame(**fields)

    def test_symbol_encoder_requires_complete_symbols(self):
        encoder = V90AnaloguePhase4Encoder(16)
        with self.assertRaisesRegex(ValueError, "align"):
            encoder.encode_bits([0, 1, 0])
        with self.assertRaisesRegex(ValueError, "contain bits"):
            V90AnaloguePhase4Encoder(4).encode_bits([0, 2])
        with self.assertRaisesRegex(ValueError, "contain bits"):
            V90AnaloguePhase4Encoder(4).encode_bits([0.0, 1.0])

    def test_controller_requires_start_before_downstream_input(self):
        controller = V90AnaloguePhase4Controller(
            cpt_fields=self.fields("cpt"),
            cp_fields=self.fields("cp", ack=0), law="pcmu",
            constellation=4, baud=3428.5714285714,
            carrier=1959.1836734694)
        with self.assertRaisesRegex(RuntimeError, "has not started"):
            controller.feed_downstream(b"\xff" * 6)

    REFERENCE_PEER_TYPE1_MP = {
        "type": 1, "ack": 0, "max_a2d_drn": 12, "trellis": 0,
        "nonlinear": 1, "shaping": 1, "rate_mask": 0x1FFF,
        "precoder": (691, 43, -234, -3, 154, 88),
    }

    def started_type1_caller(self):
        controller = V90AnaloguePhase4Controller(
            cpt_fields=self.fields("cpt"),
            cp_fields=self.fields("cp", ack=0, analog_rate_mask=0x1FFF),
            law="pcmu", constellation=4, baud=3200.0,
            carrier=1828.5714285714)
        controller.start()
        return controller

    def test_mp_acceptance_defers_upstream_power_estimate_to_media_feed(self):
        # The inline estimate is about 100 ms of work; at MP acceptance it
        # stalled the live sender in the middle of CP.
        controller = self.started_type1_caller()
        with patch.object(v34_data.DataTx, "_estimate_power",
                          side_effect=AssertionError("inline power estimate")):
            self.assertTrue(controller._accept_mp(dict(self.REFERENCE_PEER_TYPE1_MP)))
            self.assertIsNone(controller.upstream_transmitter)
            controller._advance_upstream_preparation(32 * 79)
            self.assertIsNone(controller.upstream_transmitter)
            controller._advance_upstream_preparation(32)
        prepared = controller.upstream_transmitter
        self.assertIsNotNone(prepared)
        reference = V90AnalogueUpstreamTransmitter(controller.upstream_profile)
        self.assertEqual(prepared.mapper.avg_power, reference.mapper.avg_power)
        self.assertEqual(prepared.mapper._linear_power,
                         reference.mapper._linear_power)
        self.assertEqual(prepared.encode_b1(), reference.encode_b1())
        bits = [i % 2 for i in range(2 * reference.bits_per_data_frame)]
        self.assertEqual(prepared.encode_bits(bits), reference.encode_bits(bits))

    def test_mp_prime_completes_pending_upstream_preparation(self):
        controller = self.started_type1_caller()
        controller._accept_mp(dict(self.REFERENCE_PEER_TYPE1_MP))
        controller._advance_upstream_preparation(32 * 10)
        self.assertIsNone(controller.upstream_transmitter)
        controller._accept_mp(dict(self.REFERENCE_PEER_TYPE1_MP, ack=1))
        self.assertEqual(controller.state, controller.WAIT_ED)
        reference = V90AnalogueUpstreamTransmitter(controller.upstream_profile)
        self.assertEqual(controller.upstream_transmitter.mapper.avg_power,
                         reference.mapper.avg_power)

    def test_data_fault_hold_is_opt_in_and_bounded(self):
        class Receiver:
            def __init__(self):
                self._pending = []

            def decode_codewords(self, frame):
                self._pending.extend(frame)
                if bytes(frame) == b"\xff" * 6:
                    raise ValueError("received mixed-radix label exceeds negotiated rate")
                self._pending.clear()
                return [1] * 27

        good = b"\x10" * 6
        silent = b"\xff" * 6
        for hold, expect_failed in ((0, True), (2, True), (3, False)):
            with self.subTest(hold=hold):
                controller = V90AnaloguePhase4Controller(
                    cpt_fields=self.fields("cpt"),
                    cp_fields=self.fields("cp", ack=0), law="pcmu",
                    constellation=4, baud=3428.5714285714,
                    carrier=1959.1836734694, data_fault_hold_frames=hold)
                controller.start()
                controller.state = controller.DATA
                controller._awaiting_trn2d = False
                controller._receiver = Receiver()
                controller.feed_downstream(good + silent * 3 + good)
                self.assertEqual(controller.state == controller.FAILED,
                                 expect_failed)
                names = [name for name, _ in controller.events]
                if hold == 3:
                    self.assertEqual(names[-2:], ["data-fault-start", "data-fault-end"])
                    self.assertEqual(controller.events[-1][1]["frames"], 3)
                    self.assertEqual(controller.events[-2][1]["codewords"], "ff" * 6)
                    self.assertEqual(controller._receiver._pending, [])
        with self.assertRaisesRegex(ValueError, "non-negative integer"):
            V90AnaloguePhase4Controller(
                cpt_fields=self.fields("cpt"),
                cp_fields=self.fields("cp", ack=0), law="pcmu",
                constellation=4, baud=3428.5714285714,
                carrier=1959.1836734694, data_fault_hold_frames=-1)

    def test_ed_wait_is_bounded_on_the_g711_frame_clock(self):
        controllers = [
            V90AnaloguePhase4Controller(
                cpt_fields=self.fields("cpt"),
                cp_fields=self.fields("cp", ack=0), law="pcmu",
                constellation=4, baud=baud, carrier=1959.1836734694)
            for baud in (2400.0, 3428.5714285714)
        ]
        self.assertEqual([item._ed_wait_limit_frames for item in controllers],
                         [4000, 4000])

        class RepeatingMpPrime:
            @staticmethod
            def decode_codewords(_frame):
                return [1] * 12

        controller = controllers[-1]
        controller._started = True
        controller.state = controller.WAIT_ED
        controller._awaiting_trn2d = False
        controller._ed_zero_run = 0
        controller._receiver = RepeatingMpPrime()
        controller.feed_downstream(b"\x00" * (6 * 4001))
        self.assertEqual(controller.state, controller.FAILED)
        self.assertEqual(controller._ed_wait_frames, 4001)
        self.assertEqual(controller.events[-1], (
            "phase4-failed", "timed out waiting for V.90 ED after MP-prime"))

    def test_ed_transition_preserves_cp_state_and_reports_primary_failure(self):
        class ScriptedReceiver:
            def __init__(self, profile, _law):
                self.kind = profile["kind"]
                self.frames = 0

            def decode_codewords(self, _frame):
                self.frames += 1
                if self.kind == "cpt":
                    # Model two Ed frames plus a peer's extra zero frame.
                    if self.frames <= 3:
                        return [0] * 15
                    raise ValueError("CPt rejected the transition frame")
                # Make the candidate B1d frame fail, so E's already-queued
                # output cannot mask the actual Phase 4 qualification error.
                return [0] * 27

        controller = V90AnaloguePhase4Controller(
            cpt_fields=self.fields("cpt"),
            cp_fields=self.fields("cp", ack=0), law="pcmu",
            constellation=4, baud=3428.5714285714,
            carrier=1959.1836734694)
        controller._started = True
        controller._awaiting_trn2d = False
        controller.state = controller.WAIT_ED
        controller._receiver = ScriptedReceiver(controller.cpt_profile, "pcmu")
        with patch("v90_analogue_phase4.V90DigitalPhase4Receiver",
                   ScriptedReceiver):
            # Three zero CPt frames are still Ed. The first CPt-invalid frame
            # seeds the CP descrambler but is a transition frame, not B1d.
            controller.feed_downstream(b"\x00" * (6 * 4))
            self.assertEqual(controller.state, controller.WAIT_B1D)
            self.assertEqual(controller._b1d_frames_seen, 0)
            self.assertEqual(controller._receiver.frames, 1)
            self.assertIn(("b1d-transition-frame", {
                "frame_bits": 27, "ones": 0,
                "reason": "CPt rejected boundary frame",
            }), controller.events)
            controller.feed_downstream(b"\x00" * 6)
        self.assertEqual(controller.state, controller.FAILED)
        self.assertEqual(controller._ed_zero_run, 3)
        self.assertIn(("ed-accepted", None), controller.events)
        self.assertIn(("e-sent", 20), controller.events)
        with self.assertRaisesRegex(
                RuntimeError,
                "V.90 upstream B1 unavailable after Phase 4 failure: "
                "V.90 B1d frame was not all ones"):
            controller.encode_upstream_b1()

    def test_ed_extended_zero_run_probes_each_frame_for_cp_b1d_start(self):
        class ScriptedReceiver:
            def __init__(self, profile, _law):
                self.kind = profile["kind"]
                self.frames = 0

            def decode_codewords(self, _frame):
                self.frames += 1
                if self.kind == "cpt":
                    # Three zero frames include the two-frame Ed minimum and
                    # one extra zero-coded boundary frame from the peer.
                    return [0] * 15
                return [1] * 27

        controller = V90AnaloguePhase4Controller(
            cpt_fields=self.fields("cpt"),
            cp_fields=self.fields("cp", ack=0), law="pcmu",
            constellation=4, baud=3428.5714285714,
            carrier=1959.1836734694)
        controller._started = True
        controller._awaiting_trn2d = False
        controller.state = controller.WAIT_ED
        controller._receiver = ScriptedReceiver(controller.cpt_profile, "pcmu")
        with patch("v90_analogue_phase4.V90DigitalPhase4Receiver",
                   ScriptedReceiver):
            controller.feed_downstream(b"\x00" * (6 * 3))
            self.assertEqual(controller.state, controller.WAIT_B1D)
            self.assertEqual(controller._b1d_frames_seen, 1)
            controller.feed_downstream(b"\x00" * (6 * 47))
        self.assertEqual(controller.state, controller.DATA)
        self.assertEqual(controller._b1d_frames_seen, 48)
        self.assertIn(("ed-accepted", None), controller.events)
        self.assertIn(("e-sent", 20), controller.events)
        self.assertEqual(controller.events[-1][0], "b1d-qualified")

    def test_b1d_resolves_only_unobservable_pcmu_zero_signs(self):
        cp_fields = self.fields("cp", ack=0, drn=7)
        cp_fields["tx_masks"] = (tuple(range(128)),) * 3
        controller = V90AnaloguePhase4Controller(
            cpt_fields=self.fields("cpt"), cp_fields=cp_fields,
            law="pcmu", constellation=4, baud=3428.5714285714,
            carrier=1959.1836734694)
        encoder = V90DigitalPhase4Encoder(controller.cp_profile, "pcmu")
        frame_bits = encoder.s + encoder.k
        expected_bits = [1] * (48 * frame_bits)
        wire = bytearray(encoder.encode_bits(expected_bits))
        table = g711_ucode_codewords("pcmu")
        self.assertEqual(int(g711.decode([0xff], "pcmu")[0]), 0)
        self.assertEqual(int(g711.decode([0x7f], "pcmu")[0]), 0)
        self.assertNotEqual(int(g711.decode([0xd5], "pcma")[0]),
                            int(g711.decode([0x55], "pcma")[0]))
        zero_codes = {int(table[0]), int(table[0]) ^ 0x80}
        changed = []
        for index, codeword in enumerate(wire):
            if codeword in zero_codes:
                wire[index] ^= 0x80
                changed.append(index)
        self.assertTrue(changed)

        # Exact codeword decoding sees the sign aliases as different symbols.
        raw_receiver = V90DigitalPhase4Receiver(
            controller.cp_profile, "pcmu")
        self.assertNotEqual(raw_receiver.decode_codewords(wire), expected_bits)

        controller._started = True
        controller._awaiting_trn2d = False
        controller.state = controller.WAIT_B1D
        controller._receiver = V90DigitalPhase4Receiver(
            controller.cp_profile, "pcmu")
        controller.feed_downstream(wire)
        self.assertEqual(controller.state, controller.DATA)
        self.assertEqual(controller._b1d_frames_seen, 48)
        self.assertEqual(sum(event[0] == "b1d-zero-sign-resolved"
                             for event in controller.events), len(changed))

        # A nonzero magnitude error remains fatal; the known training pattern
        # does not turn B1d into a fuzzy threshold.
        bad = bytearray(encoder.encode_bits(expected_bits)[:6])
        bad[1] = int(table[95]) ^ (bad[1] & 0x80)
        strict = V90AnaloguePhase4Controller(
            cpt_fields=self.fields("cpt"), cp_fields=cp_fields,
            law="pcmu", constellation=4, baud=3428.5714285714,
            carrier=1959.1836734694)
        strict._started = True
        strict._awaiting_trn2d = False
        strict.state = strict.WAIT_B1D
        strict._receiver = V90DigitalPhase4Receiver(
            strict.cp_profile, "pcmu")
        strict.feed_downstream(bad)
        self.assertEqual(strict.state, strict.FAILED)
        self.assertEqual(strict._b1d_frames_seen, 0)

    def test_rejected_speculative_cp_frame_is_not_reused_at_transition(self):
        cp_fields = self.fields(
            "cp", ack=0, drn=5, analog_rate_mask=0x1fff,
            codec_masks=(tuple(range(16, 32)), tuple(range(32, 64)),
                         tuple(range(80, 128))))
        controller = V90AnaloguePhase4Controller(
            cpt_fields=self.fields("cpt"),
            cp_fields=cp_fields, law="pcmu",
            constellation=4, baud=3428.5714285714,
            carrier=1959.1836734694)
        controller._started = True
        controller._awaiting_trn2d = False
        controller.state = controller.WAIT_ED
        controller._ed_zero_run = 2

        # Ucodes [15,47,127] are legal under CPt but Ucode 15 is excluded from
        # the negotiated CP map. The real CP decoder buffers this rejected
        # frame before raising, reproducing the state leak found during code review.
        table = g711_ucode_codewords("pcmu")
        boundary = bytes(table[value] for value in (15, 47, 127, 15, 47, 127))
        with self.assertRaisesRegex(ValueError, "Ucode 15 is not legal"):
            V90DigitalPhase4Receiver(controller.cp_profile, "pcmu").decode_codewords(
                boundary)

        controller.feed_downstream(boundary)
        self.assertEqual(controller.state, controller.WAIT_B1D)
        self.assertEqual(controller._b1d_frames_seen, 0)
        self.assertEqual(controller._receiver._pending, [])
        self.assertIn(("b1d-transition-frame", {
            "frame_bits": 15, "ones": 1,
            "reason": "CPt boundary frame is invalid under CP",
        }), controller.events)

        # The latched CP receiver starts with an empty buffer and decodes the
        # following CP-legal mapper frame without seeing the rejected Ucodes.
        next_frame = bytes(table[value]
                           for value in (31, 63, 127, 31, 63, 127))
        decoded = controller._receiver.decode_codewords(next_frame)
        self.assertEqual(len(decoded), round(controller.cp_profile["rate"] * 6 / 8000))
        self.assertEqual(controller._receiver._pending, [])

    def test_gpa_decisions_map_to_v34_qam_points(self):
        decisions = [(rotation, quarter)
                     for quarter in range(4) for rotation in range(4)]
        points = qam_symbols(decisions, 16)
        for (rotation, quarter), point in zip(decisions, points):
            self.assertEqual(point, complex(*v34_tx.C16[quarter])
                             * v34_tx.ROT_CW[rotation])
        with self.assertRaisesRegex(ValueError, "cannot carry a quarter"):
            qam_symbols([(0, 1)], 4)

    def test_phase4_media_tx_renders_continuous_8khz_pcm(self):
        baud = 2400.0 * 10 / 7
        controller = V90AnaloguePhase4Controller(
            cpt_fields=self.fields("cpt"),
            cp_fields=self.fields("cp", ack=0), law="pcmu",
            constellation=16, baud=baud, carrier=1959.1836734694)
        media = V90AnaloguePhase4MediaTx(
            controller, baud=baud, carrier=1959.1836734694, rms=0.1)
        media.start()
        first = media.read_pcm(160)
        second = media.read_pcm(160)
        self.assertEqual(first.shape, (160,))
        self.assertEqual(second.shape, (160,))
        self.assertTrue(all(math.isfinite(float(v))
                            for v in first))
        self.assertGreater(float((first ** 2).mean()), 0.0)
        self.assertGreater(float((second ** 2).mean()), 0.0)

    def test_cpt_restarts_ri_after_packetized_wait_without_stale_repeat_tail(self):
        """CPt response starts on its own boundary across RTP-sized reads."""
        for law in ("pcmu", "pcma"):
            for wait_packets in (1, 2, 3, 4):
                with self.subTest(law=law, wait_packets=wait_packets):
                    profile = parse_cp(build_cp_frame(
                        **self.fields("cpt", law=law, ack=0)))
                    session = V90DigitalAnswererPhase4Session(
                        uinfo=90, law=law, constellation=4,
                        mp=answerer_mp_profile(3428.5714285714))
                    session.start()
                    initial_and_wait = bytearray()
                    for _ in range(wait_packets):
                        initial_and_wait.extend(session.read_g711(160, law))
                    self.assertTrue(session.receive_profile(profile))
                    actual = session.read_g711(2200, law)

                    reference = V90DigitalAnswererPhase4Session(
                        uinfo=90, law=law, constellation=4,
                        mp=answerer_mp_profile(3428.5714285714))
                    reference.start()
                    reference_initial = reference.read_g711(192, law)
                    reference.receive_profile(profile)
                    response = reference.read_g711(2200, law)
                    initial_remaining = max(0, 192 - len(initial_and_wait))
                    expected = (reference_initial[len(initial_and_wait):]
                                if initial_remaining else b"") + response
                    expected = expected[:len(actual)]
                    self.assertEqual(actual, expected)

    def test_caller_acquires_repeating_ri_after_arbitrary_codeword_offset(self):
        """Acquire the answerer's repeated R-i when receive starts midstream."""
        for law in ("pcmu", "pcma"):
            with self.subTest(law=law):
                controller = V90AnaloguePhase4Controller(
                    cpt_fields=self.fields("cpt", law=law),
                    cp_fields=self.fields("cp", law=law, ack=0),
                    law=law, constellation=4, baud=3428.5714285714,
                    carrier=1959.1836734694)
                controller.start()
                repeated_ri = ri_codewords(90, law, duration_symbols=192)
                # Start seven codewords into the repeated signal, then split
                # its remaining symbols across uneven receive chunks.
                stream = repeated_ri[7:]
                controller.feed_downstream(stream[:11])
                controller.feed_downstream(stream[11:74])
                controller.feed_downstream(stream[74:])
                self.assertIsNotNone(controller._ri_pattern)
                self.assertTrue(controller._awaiting_trn2d)

    def test_delayed_ri_packet_does_not_consume_rbar_transition(self):
        """Keep R-i/R-bar-i aligned when a 160T packet splits a mapper frame."""
        for law in ("pcmu", "pcma"):
            with self.subTest(law=law):
                reference = ri_codewords(90, law, duration_symbols=24)
                # Initial acquisition drops one 24T R-i block. Another 160T
                # R-i packet leaves four symbols of a six-symbol mapper frame
                # in flight; the first two R-bar-i symbols would complete
                # that apparent R-i frame if it were consumed immediately.
                pending = bytearray(
                    ri_codewords(90, law, duration_symbols=192)[24:]
                    + ri_codewords(90, law, duration_symbols=192)[:160]
                    + rbar_i_codewords(90, law, duration_symbols=24))
                transition = V90AnaloguePhase4Controller._find_rbar_transition(
                    pending, reference)
                self.assertEqual(transition, (168 + 160, 3))
                self.assertIsNone(V90AnaloguePhase4Controller._find_rbar_transition(
                    bytearray(ri_codewords(90, law, duration_symbols=192)),
                    reference))

    def test_caller_answerer_phase4_and_lapm_exchange_payload(self):
        self._exercise_caller_answerer_phase4_and_lapm_exchange_payload(
            omit_ucode0=False)

    def test_caller_answerer_phase4_and_lapm_exchange_without_ucode0(self):
        self._exercise_caller_answerer_phase4_and_lapm_exchange_payload(
            omit_ucode0=True)

    def test_caller_answerer_phase4_and_lapm_exchange_without_cp_ucode0(self):
        # captured peer keeps Ucode 0 in CPt training but cannot carry its sign in
        # DATA; the CP-only omission must still train and pass LAPM.
        self._exercise_caller_answerer_phase4_and_lapm_exchange_payload(
            omit_ucode0=True, cp_only=True)

    def _exercise_caller_answerer_phase4_and_lapm_exchange_payload(
            self, *, omit_ucode0: bool, cp_only: bool = False):
        baud = 2400.0 * 10 / 7
        carrier = 1959.1836734694
        mp_profiles = (
            ("type0", answerer_mp_profile(baud), 33600),
            ("captured_peer-type1", {
                "mp_type": 1, "max_a2d_drn": 12, "trellis": 0,
                "nonlinear": 1, "shaping": 1, "rate_mask": 0x1FFF,
                "precoder": (520, -52, -119, 1, 116, 137),
            }, 28800),
        )
        for law in ("pcmu", "pcma"):
            for constellation in (4, 16):
                for downstream_drn in (5, 22):
                    for mp_name, mp_profile, expected_upstream_rate in mp_profiles:
                        with self.subTest(law=law, constellation=constellation,
                                            downstream_drn=downstream_drn,
                                            mp_profile=mp_name):
                            cpt_fields = self.fields("cpt", law=law)
                            cp_fields = self.fields("cp", law=law, ack=0,
                                                    drn=downstream_drn,
                                                    analog_rate_mask=0x1FFF)
                            if downstream_drn == 22:
                                # 56 kbit/s requires 36 modulus bits per six-octet
                                # data frame; exercise the largest permitted Ucode
                                # maps rather than silently limiting this pair to a
                                # lower-capacity test constellation.
                                cp_fields["tx_masks"] = (
                                    tuple(range(128)),) * 3
                            if omit_ucode0:
                                for fields in ((cp_fields,) if cp_only
                                               else (cpt_fields, cp_fields)):
                                    fields["tx_masks"] = tuple(
                                        tuple(value for value in mask if value != 0)
                                        for mask in fields["tx_masks"])
                            caller = V90AnaloguePhase4Controller(
                                cpt_fields=cpt_fields, cp_fields=cp_fields,
                                law=law, constellation=constellation,
                                baud=baud, carrier=carrier,
                                capture_data_bits=True)
                            media = V90AnaloguePhase4MediaTx(
                                caller, baud=baud, carrier=carrier, rms=0.1)
                            answerer = V90DigitalAnswererPhase4Session(
                                uinfo=90, law=law, constellation=constellation,
                                mp=mp_profile)

                            answerer.start()
                            # The analogue caller can identify initial R-i partway
                            # through the second media packet. It starts CPt while
                            # the tail of that packet is already in flight.
                            initial_ri = (answerer.read_g711(160, law)
                                          + answerer.read_g711(32, law))
                            self.assertEqual(len(initial_ri), 192)
                            cpt_symbols = media.start()
                            # The caller has already acquired initial R-i before
                            # it sends CPt. Feed that observed 192T through its
                            # receive path so the post-CPt R-bar-i is recognized
                            # as the transition signal, not as first acquisition.
                            self.assertEqual(media.feed_downstream(initial_ri), [])
                            # Model one RTP turnaround after CPt starts: the
                            # answerer may have another R-i packet in flight
                            # before it receives CPt and switches to R-bar-i.
                            delayed_ri = answerer.read_g711(160, law)
                            self.assertEqual(len(delayed_ri), 160)
                            self.assertEqual(media.feed_downstream(delayed_ri), [])
                            cpt_profiles = answerer.feed_symbols(cpt_symbols)
                            self.assertEqual(len(cpt_profiles), 1)
                            self.assertEqual(cpt_profiles[0]["kind"], "cpt")

                            # Advance actual answerer output in 20 ms G.711 packets.
                            # The caller's decoded MP produces CP, which is fed back
                            # immediately as the live upstream symbol event.
                            # A Type 1 MP is 192 bits instead of the 90-bit
                            # Type 0 frame, so allow its longer codeword
                            # expansion to finish before expecting CP.
                            mp_tail = 600 if mp_name == "captured_peer-type1" else 36
                            remaining = 24 + 2040 + mp_tail
                            cp_symbols = []
                            while remaining:
                                packet = answerer.read_g711(min(160, remaining), law)
                                remaining -= len(packet)
                                cp_symbols.extend(media.feed_downstream(packet))
                                if cp_symbols:
                                    break
                            self.assertTrue(cp_symbols)
                            # Model one media-packet turnaround: the answerer may
                            # emit another MP before CP reaches its parser.
                            duplicate_mp = answerer.read_g711(36, law)
                            self.assertEqual(media.feed_downstream(duplicate_mp), [])
                            self.assertEqual(caller.state, caller.WAIT_MP_PRIME)
                            cp_profiles = answerer.feed_symbols(cp_symbols)
                            self.assertEqual(len(cp_profiles), 1)
                            self.assertEqual(cp_profiles[0]["kind"], "cp")
                            self.assertEqual(cp_profiles[0]["ack"], 0)
                            self.assertEqual(caller.state, caller.WAIT_MP_PRIME)
                            self.assertEqual(caller.upstream_profile.rate,
                                             expected_upstream_rate)
                            if mp_name == "captured_peer-type1":
                                self.assertTrue(caller.upstream_profile.nonlinear)
                                self.assertTrue(caller.upstream_profile.shaping)
                                self.assertEqual(caller.upstream_profile.precoder,
                                                 mp_profile["precoder"])
                            # Let MP′, CP′, Ed and B1d advance in RTP-sized
                            # packets. The Type 1 MP is longer than Type 0 and
                            # may not align to a test read boundary; stopping
                            # on protocol state avoids consuming queued Ed/B1d
                            # using an assumed fixed MP′ byte count.
                            cp_prime_profiles = []
                            for _ in range(200):
                                if caller.state == caller.WAIT_ED:
                                    break
                                packet = answerer.read_g711(160, law)
                                response = media.feed_downstream(packet)
                                if response:
                                    cp_prime_profiles.extend(
                                        answerer.feed_symbols(response))
                            self.assertEqual(caller.state, caller.WAIT_ED)
                            self.assertEqual(len(cp_prime_profiles), 1)
                            self.assertEqual(cp_prime_profiles[0]["ack"], 1)
                            self.assertIsNone(media.upstream_b1_symbols)
                            self.assertFalse(caller.upstream_transmitter.b1_sent)
                            self.assertNotIn(("e-sent", 20), caller.events)
                            # The analogue side must hold CP′ through the return
                            # delay; E/B1 are gated on receiving the two Ed frames.
                            media.read_pcm(160)
                            self.assertEqual(caller.state, caller.WAIT_ED)
                            self.assertNotIn(("e-sent", 20), caller.events)
                            for _ in range(200):
                                if caller.state == caller.DATA:
                                    break
                                packet = answerer.read_g711(160, law)
                                media.feed_downstream(packet)
                            self.assertEqual(caller.state, caller.DATA,
                                             msg=caller._failure_reason)
                            self.assertIn(("ed-accepted", None), caller.events)
                            self.assertEqual(caller._b1d_frames_seen, 48)

                            # E already queued caller B1 on the shared media clock;
                            # it must not wait for the independent downstream B1d.
                            upstream = V90UpstreamDataReceiver(
                                caller.upstream_profile, remote_role="answer")
                            upstream_b1 = media.upstream_b1_symbols
                            self.assertIsNotNone(upstream_b1)
                            upstream.start_b1(0)
                            gain = 0.71 * np.exp(0.27j)
                            for index, symbol in enumerate(upstream_b1):
                                upstream.feed_symbol(index, gain * symbol)
                            self.assertEqual(upstream.state, upstream.DATA)
                            answerer_lapm = V90LapmLink(
                                rate=int(round(caller.cp_profile["rate"])),
                                calling_party=False)
                            self.assertEqual(caller.cp_profile["rate"],
                                             (downstream_drn + 20) * 8000 / 6)
                            caller_payload = bytes((index * 29 + 7) & 0xFF
                                                   for index in range(128))
                            answer_payload = bytes((index * 53 + 11) & 0xFF
                                                   for index in range(128))
                            caller.feed_bytes(caller_payload)
                            answerer_lapm.feed_bytes(answer_payload)
                            up_bits_per_slice = (
                                3 * caller.upstream_transmitter.bits_per_data_frame)
                            down_bits_per_slice = int(round(
                                up_bits_per_slice * caller.cp_profile["rate"] /
                                caller.upstream_profile.rate))
                            frame_bits = round(caller.cp_profile["rate"] * 6 / 8000)
                            self.assertEqual(down_bits_per_slice % frame_bits, 0)
                            down_octets = down_bits_per_slice // frame_bits * 6
                            received_caller = bytearray()
                            received_answerer = bytearray()

                            for _ in range(100):
                                for _ in range(3):
                                    upstream_symbols = media.queue_next_upstream_frame()
                                    decoded = []
                                    for offset, symbol in enumerate(
                                            upstream_symbols,
                                            start=upstream.next_symbol):
                                        decoded.extend(upstream.feed_symbol(
                                            offset, gain * symbol))
                                    if decoded:
                                        answerer_lapm.feed_line_bits(decoded)
                                received_answerer.extend(answerer_lapm.take_received())

                                downstream = answerer.read_g711(
                                    down_octets, law,
                                    data_bit_source=answerer_lapm.read_tx_bits)
                                self.assertEqual(len(downstream), down_octets)
                                media.feed_downstream(downstream)
                                received_caller.extend(caller.take_received())
                                if (bytes(received_caller) == answer_payload
                                        and bytes(received_answerer) == caller_payload):
                                    break

                            self.assertEqual((caller.lapm.v42_mode,
                                              answerer_lapm.v42_mode), ("lapm", "lapm"))
                            self.assertEqual(bytes(received_caller), answer_payload)
                            self.assertEqual(bytes(received_answerer), caller_payload)


if __name__ == "__main__":
    unittest.main()
