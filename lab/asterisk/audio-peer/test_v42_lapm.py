#!/usr/bin/env python3
"""Our V.42 (v42_lapm.V42Link): against itself and against SpanDSP's V.42 as an independent reference peer."""
import os
import random
import unittest
from unittest.mock import patch

import v42_lapm
from v42bis import Decoder, Encoder


def pump(a, b, bits, ber=0.0, burst=0, seed=1):
    rnd = random.Random(seed)
    got_a, got_b = bytearray(), bytearray()
    for i in range(bits):
        ba, bb = a.tx_bit(), b.tx_bit()
        if ber:
            if rnd.random() < ber:
                ba ^= 1
            if rnd.random() < ber:
                bb ^= 1
        if burst and i % 30000 == 9000:
            for _ in range(burst):
                a.tx_bit(), b.tx_bit()
        a.rx_bit(bb)
        b.rx_bit(ba)
        got_a += a.read()
        got_b += b.read()
    return bytes(got_a), bytes(got_b)


def drain_frames(link):
    rx = v42_lapm._Hdlc()
    frames = []
    for _ in range(len(link._tx)):
        frames.extend(rx.push(link.tx_bit()))
    for bit in v42_lapm.FLAG_BITS:  # following idle flags close an ordinary queued frame
        frames.extend(rx.push(bit))
    return frames


def pair(**kw):
    detect = kw.pop("detect", True)
    comp = kw.pop("compression", None)
    kw.setdefault("t403", None)
    return (v42_lapm.V42Link(True, detect=detect, bit_rate=9600, compression=comp, **kw),
            v42_lapm.V42Link(False, detect=detect, bit_rate=9600, compression=comp, **kw))


class T403ConnectionVerificationTests(unittest.TestCase):
    def test_idle_connected_link_is_polled_and_t401_recovers(self):
        a, b = pair(detect=False, t403=60)
        pump(a, b, 9600 * 3)
        self.assertEqual((a.state, b.state), ("lapm", "lapm"))
        for link in (a, b):
            link._t403_bits = 64
            link._start_t403()

        pump(a, b, 100)
        for link in (a, b):
            self.assertTrue(any("T403 expired" in event for event in link.events))
            self.assertEqual(link.stats["t403_expiries"], 1)
            self.assertTrue(link._poll_out)
            self.assertIsNotNone(link._t401_at)
            self.assertIsNone(link._t403_at)
            # Give the response procedure time to complete without a second
            # diagnostic timeout in this deliberately accelerated fixture.
            link._t403_bits = 10000
        pump(a, b, 128)
        for link in (a, b):
            self.assertEqual(link.state, "lapm")
            self.assertFalse(link._poll_out)
            self.assertIsNotNone(link._t403_at)
            self.assertIsNone(link._t401_at)

    def test_t403_can_be_disabled(self):
        a, b = pair(detect=False, t403=None)
        pump(a, b, 9600 * 3)
        self.assertEqual((a._t403_at, b._t403_at), (None, None))

    def test_t403_uses_rnr_when_local_receiver_is_busy(self):
        a, b = pair(detect=False, t403=60)
        pump(a, b, 9600 * 3)
        a._local_busy = True
        a._t403_bits = 64
        a._start_t403()
        pump(a, b, 300)
        self.assertIn("T403 expired; connection-verification poll sent", a.events)
        self.assertTrue(b._peer_busy)

    def test_t403_uses_rej_for_unreported_sequence_error(self):
        a, b = pair(detect=False, t403=60)
        pump(a, b, 9600 * 3)
        a._need_rej = True
        a._t403_bits = 1
        a._start_t403()
        a._tx.clear()
        a.bits_sent = a._t403_at - 1
        a.tx_bit()

        frames = [frame for frame, ok in drain_frames(a) if ok]
        self.assertEqual(frames[0], bytes((a.cmd_addr, v42_lapm.REJ, 1)))
        self.assertTrue(a._poll_out)
        self.assertTrue(a._rej_sent)
        self.assertFalse(a._need_rej)


class V42LinkParameterBoundsTests(unittest.TestCase):
    def test_zero_uses_defaults_and_supported_values_are_capped_to_local_defaults(self):
        default = v42_lapm.V42Link(True, detect=False)
        self.assertEqual((default.tx_n401, default.rx_n401,
                          default.tx_k, default.rx_k), (128, 128, 15, 15))

        configured = v42_lapm.V42Link(
            True, detect=False, frame_size=128, receive_frame_size=128,
            window=127, receive_window=127,
        )
        self.assertEqual((configured.tx_n401, configured.rx_n401,
                          configured.tx_k, configured.rx_k), (128, 128, 15, 15))

    def test_invalid_local_parameters_fail_at_construction(self):
        for name, value in (
            ("frame_size", -1), ("frame_size", 129),
            ("receive_frame_size", -1), ("receive_frame_size", 129),
            ("window", -1), ("window", 128),
            ("receive_window", -1), ("receive_window", 128),
            ("frame_size", True), ("window", 2.5),
        ):
            with self.subTest(name=name, value=value):
                with self.assertRaisesRegex(ValueError, name):
                    v42_lapm.V42Link(True, detect=False, **{name: value})


class HdlcFrameBoundaryTests(unittest.TestCase):
    def test_fcs16_matches_crc_x25_known_answer(self):
        from mnp_link import crc16_hdlc

        # RFC 1662 Appendix C defines the reflected 0x1021 FCS-16 used by
        # HDLC-like framing. This known-answer check is independent of the
        # local _stuffed/_Hdlc round trip.
        self.assertEqual(crc16_hdlc(b"123456789"), 0x906E)

    def test_hdlc_stuffing_matches_fixed_wire_vector(self):
        # Fixed vector independently computed from the HDLC FCS-16 and bit-stuffing rules:
        # address/control/N(R)=03 00 00, six FF information octets, FCS=09 9C.
        # The literal wire bits include the stuffed zero after each five ones.
        frame = bytes.fromhex("03 00 00") + b"\xff" * 6
        expected = "1100000000000000000000001111101111101111101111101111101111101111101111101111101111001000000111001"
        self.assertEqual(v42_lapm._stuffed(frame), [int(bit) for bit in expected])

        rx = v42_lapm._Hdlc()
        completed = []
        for bit in v42_lapm.FLAG_BITS + [int(bit) for bit in expected] + v42_lapm.FLAG_BITS:
            completed.extend(rx.push(bit))
        self.assertEqual(completed, [(frame, True)])

    def test_hdlc_decodes_a_valid_frame(self):
        rx = v42_lapm._Hdlc()
        frame = bytes((0x03, 0x00, 0x00)) + b"valid payload"
        completed = []
        for bit in v42_lapm.FLAG_BITS + v42_lapm._stuffed(frame) + v42_lapm.FLAG_BITS:
            completed.extend(rx.push(bit))
        self.assertEqual(completed, [(frame, True)])

    def test_unbounded_frame_is_discarded_with_bounded_memory_and_receiver_recovers(self):
        rx = v42_lapm._Hdlc()
        for bit in v42_lapm.FLAG_BITS:
            rx.push(bit)
        for bit in [0] * (rx.max_unflagged_bits + 1):
            rx.push(bit)
        self.assertFalse(rx.active)
        self.assertEqual(rx.bits, [])
        self.assertEqual(rx.overlong, 1)

        frame = bytes((0x03, 0x00, 0x00)) + b"after abort"
        completed = []
        for bit in v42_lapm.FLAG_BITS + v42_lapm._stuffed(frame) + v42_lapm.FLAG_BITS:
            completed.extend(rx.push(bit))
        self.assertEqual(completed, [(frame, True)])
        self.assertLessEqual(len(rx.bits), rx.max_unflagged_bits)

    def test_seven_one_abort_discards_partial_frame_and_next_flag_recovers(self):
        rx = v42_lapm._Hdlc()
        aborted_prefix = bytes((0x03, 0x00, 0x00))
        valid_frame = bytes((0x03, 0x00, 0x00)) + b"after abort"
        bits = (v42_lapm.FLAG_BITS + v42_lapm._stuffed(aborted_prefix)[:-8]
                + [1] * 7 + [0] + v42_lapm.FLAG_BITS
                + v42_lapm._stuffed(valid_frame) + v42_lapm.FLAG_BITS)
        completed = []
        for bit in bits:
            completed.extend(rx.push(bit))
        self.assertEqual(completed, [(valid_frame, True)])

    def test_fill_flags_and_shared_frame_boundary_decode_consecutive_frames(self):
        rx = v42_lapm._Hdlc()
        first = bytes((0x03, 0x00, 0x00)) + b"first"
        second = bytes((0x03, 0x00, 0x00)) + b"second"
        completed = []
        for bit in v42_lapm.FLAG_BITS * 4:
            completed.extend(rx.push(bit))
        self.assertGreaterEqual(rx.continuous_flags, 4)

        # One flag closes the first frame and opens the next; additional fill
        # flags before and after exercise the receiver's hunt/active boundary.
        wire = (v42_lapm._stuffed(first) + v42_lapm.FLAG_BITS
                + v42_lapm._stuffed(second) + v42_lapm.FLAG_BITS * 3)
        for bit in wire:
            completed.extend(rx.push(bit))
        self.assertEqual(completed, [(first, True), (second, True)])

    def test_shared_zero_fill_flags_start_answerer_protocol(self):
        answerer = v42_lapm.V42Link(False, detect=True)
        for bit in v42_lapm.ODP_BLOCK * 2:
            answerer.rx_bit(bit)
        self.assertTrue(answerer._answer_adp)
        self.assertEqual(answerer._phase, "detect")

        # Consecutive HDLC flags may share the zero at the boundary. After
        # the first 8-bit flag, each following flag contributes six ones and
        # its closing zero, giving a seven-bit flag interval.
        shared_flag_tail = [1] * 6 + [0]
        for bit in v42_lapm.FLAG_BITS + shared_flag_tail * 2:
            answerer.rx_bit(bit)
        self.assertEqual(answerer._hdlc.continuous_flags, 3)
        self.assertEqual(answerer._phase, "detect")
        for bit in shared_flag_tail:
            answerer.rx_bit(bit)

        self.assertEqual(answerer._hdlc.continuous_flags, 0)
        # Protocol start replaces the detector so queued XID/SABME parsing
        # begins with a clean HDLC state.
        self.assertEqual(answerer._phase, "protocol")
        self.assertEqual(answerer.state, "idle")

    def test_bad_fcs_is_reported_as_invalid_and_following_frame_recovers(self):
        rx = v42_lapm._Hdlc()
        damaged = bytearray(v42_lapm._stuffed(b"\x03\x00bad fcs"))
        # Flip a payload bit after the address/control so the frame remains
        # well-formed but its FCS no longer matches.
        damaged[8 * 3] ^= 1
        valid_frame = b"\x03\x00good"
        bits = (v42_lapm.FLAG_BITS + list(damaged) + v42_lapm.FLAG_BITS
                + v42_lapm._stuffed(valid_frame) + v42_lapm.FLAG_BITS)
        completed = []
        for bit in bits:
            completed.extend(rx.push(bit))
        self.assertEqual(len(completed), 2)
        self.assertFalse(completed[0][1])
        self.assertEqual(completed[1], (valid_frame, True))

    def test_bad_fcs_uses_optional_rej_path_but_non_octet_frame_is_discarded(self):
        receiver, peer = pair(detect=False)
        pump(receiver, peer, 9600 * 3)

        bad_fcs = bytearray(v42_lapm._stuffed(bytes((peer.cmd_addr, 0, 0)) + b"bad fcs"))
        bad_fcs[8 * 3] ^= 1
        receiver.rx_bits(v42_lapm.FLAG_BITS + list(bad_fcs) + v42_lapm.FLAG_BITS)
        self.assertEqual(receiver.stats["fcs_errors"], 1)
        self.assertTrue(receiver._need_rej)  # permitted optional V.42 §8.5.4 response

        receiver._need_rej = False
        receiver.stats["fcs_errors"] = 0
        non_octet = (v42_lapm.FLAG_BITS
                     + v42_lapm._stuffed(bytes((peer.cmd_addr, 0, 0)) + b"bits")
                     + [0] + v42_lapm.FLAG_BITS)
        receiver.rx_bits(non_octet)
        self.assertEqual(receiver.stats["fcs_errors"], 0)
        self.assertEqual(receiver.stats["hdlc_bad"], 2)
        self.assertFalse(receiver._need_rej)

    def test_v42_minimum_frame_lengths_are_enforced(self):
        def decode(frame):
            rx = v42_lapm._Hdlc()
            completed = []
            for bit in (v42_lapm.FLAG_BITS + v42_lapm._stuffed(frame)
                        + v42_lapm.FLAG_BITS):
                completed.extend(rx.push(bit))
            return completed

        # A CRC-valid frame is still invalid when it is too short for its
        # control format: I/S require 5 protected octets, U requires 4.
        self.assertEqual(decode(b"\x03"), [])
        self.assertEqual(decode(b"\x03\x00"), [])  # truncated I control
        self.assertEqual(decode(b"\x03\x63"), [(b"\x03\x63", True)])
        self.assertEqual(decode(b"\x03\x00\x00"), [(b"\x03\x00\x00", True)])


class DetectionCharacterSpacingTests(unittest.TestCase):
    @staticmethod
    def _pattern(chars, gaps):
        bits = []
        for index, char in enumerate(chars):
            bits.extend([0] + [(char >> bit) & 1 for bit in range(8)] + [1])
            if index < len(gaps):
                bits.extend([1] * gaps[index])
        return bits

    def test_answerer_requires_eight_to_sixteen_mark_bits_in_odp(self):
        chars = [v42_lapm.DC1_EVEN, v42_lapm.DC1_ODD,
                 v42_lapm.DC1_EVEN, v42_lapm.DC1_ODD]
        for gap in (8, 16):
            with self.subTest(gap=gap):
                answerer = v42_lapm.V42Link(False, detect=True)
                answerer.rx_bits(self._pattern(chars, [gap] * 3))
                self.assertTrue(answerer._answer_adp)

        for gap in (0, 7, 17, 24):
            with self.subTest(gap=gap):
                answerer = v42_lapm.V42Link(False, detect=True)
                answerer.rx_bits(self._pattern(chars, [gap] * 3))
                self.assertFalse(answerer._answer_adp)
                self.assertEqual(answerer._phase, "detect")

    def test_originator_accepts_standard_and_observed_adp_spacing(self):
        chars = [v42_lapm.CHAR_E, v42_lapm.CHAR_C,
                 v42_lapm.CHAR_E, v42_lapm.CHAR_C]
        for gap in (8, 16, 0):
            with self.subTest(gap=gap):
                caller = v42_lapm.V42Link(True, detect=True)
                for _ in range(v42_lapm.ADP_ARM_BITS):
                    caller.tx_bit()
                caller.rx_bits(self._pattern(chars, [gap] * 3))
                self.assertTrue(caller.adp_seen)
                self.assertEqual(caller._phase, "protocol")

        for gap in (7, 17, 24):
            with self.subTest(gap=gap):
                caller = v42_lapm.V42Link(True, detect=True)
                for _ in range(v42_lapm.ADP_ARM_BITS):
                    caller.tx_bit()
                caller.rx_bits(self._pattern(chars, [gap] * 3))
                self.assertFalse(caller.adp_seen)
                self.assertEqual(caller._phase, "detect")


class LinkTests(unittest.TestCase):
    def test_local_receive_busy_advertises_rnr_and_recovers_with_rr(self):
        a, b = pair(detect=False)
        pump(a, b, 9600 * 3)
        self.assertEqual((a.state, b.state), ("lapm", "lapm"))
        a.set_local_busy(True)
        pump(a, b, 9600)
        self.assertTrue(b._peer_busy)
        a.set_local_busy(False)
        pump(a, b, 9600)
        self.assertFalse(b._peer_busy)


class BreakTransferTests(unittest.TestCase):
    def connected_pair(self):
        a, b = pair(detect=False)
        pump(a, b, 9600 * 3)
        # XID's optional-function mask can change HDLC zero stuffing by a few
        # bits. Drain the final handshake frame instead of relying on the old
        # fixed budget to leave both transmit queues empty.
        for _ in range(9600):
            if not a._tx and not b._tx:
                break
            pump(a, b, 1)
        # At quiescent link idle, the scheduler may have started a partial
        # flag at the arbitrary end of the simulation budget. Break ordering
        # tests create their own explicit serialized backlog, so begin them at
        # a clean frame boundary.
        a._tx.clear()
        b._tx.clear()
        self.assertEqual((a.state, b.state), ("lapm", "lapm"))
        return a, b

    def test_break_information_field_vectors(self):
        self.assertEqual(v42_lapm.encode_brk(0), b"\x40\x00")
        self.assertEqual(v42_lapm.encode_brk(1, destructive=True, expedited=True,
                                             length_ms=1_000), b"\xc0\xc0\x64")
        self.assertEqual(v42_lapm.encode_brk(0, length_ms=10_000), b"\x40\x00\xff")
        self.assertEqual(v42_lapm.decode_brk(b"\xc0\xc0\x64"), {
            "sequence": 1, "destructive": True, "expedited": True,
            "length_ms": 1_000, "length_code": 100,
        })
        self.assertEqual(v42_lapm.decode_brk(b"\x40\x00\x00")["length_ms"], None)
        for invalid in (b"", b"\x41\x00", b"\x40", b"\x40\x01", b"\x40\x00\x00\x00"):
            self.assertIsNone(v42_lapm.decode_brk(invalid), invalid)
        with self.assertRaises(ValueError):
            v42_lapm.encode_brk(2)
        with self.assertRaises(ValueError):
            v42_lapm.encode_brk(0, length_ms=-1)
        self.assertEqual(v42_lapm.encode_brkack(0), b"\x60")
        self.assertEqual(v42_lapm.encode_brkack(1), b"\xe0")

    def test_break_request_is_indicated_once_and_acknowledged_after_control(self):
        a, b = self.connected_pair()
        a._t403_bits = 1000
        a._start_t403()
        self.assertIsNotNone(a._t403_at)
        self.assertFalse(b.acknowledge_break())
        self.assertTrue(a.request_break(length_ms=120, destructive=True, expedited=True))
        self.assertIsNone(a._t403_at)
        frames = drain_frames(a)
        self.assertEqual(len(frames), 1)
        command, ok = frames[0]
        self.assertTrue(ok)
        self.assertEqual(command, bytes((a.cmd_addr, v42_lapm.UI)) + b"\x40\xc0\x0c")
        b._on_frame(command)
        events = b.read_break_events()
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["type"], "break_indication")
        self.assertEqual(events[0]["length_ms"], 120)
        self.assertTrue(b.acknowledge_break())
        ack = drain_frames(b)
        self.assertEqual(ack, [(bytes((b.rsp_addr, v42_lapm.UI)) + b"\xe0", True)])
        a._on_frame(ack[0][0])
        self.assertEqual(a._break_send_seq, 1)
        self.assertEqual(a._t403_at, a.bits_sent + a._t403_bits)
        self.assertEqual(a.read_break_events()[0]["type"], "break_confirmed")
        self.assertEqual(b.read_break_events(), [])

    def test_duplicate_break_retransmits_ack_without_duplicate_indication(self):
        _, b = self.connected_pair()
        command = bytes((b.rsp_addr, v42_lapm.UI)) + v42_lapm.encode_brk(0)
        b._on_frame(command)
        self.assertEqual(len(b.read_break_events()), 1)
        self.assertTrue(b.acknowledge_break())
        drain_frames(b)
        b._on_frame(command)
        self.assertEqual(b.read_break_events(), [])
        self.assertEqual(drain_frames(b), [
            (bytes((b.rsp_addr, v42_lapm.UI)) + v42_lapm.encode_brkack(1), True)
        ])

    def test_duplicate_expedited_break_before_dte_response_defers_matching_acks(self):
        _, b = self.connected_pair()
        command = bytes((b.rsp_addr, v42_lapm.UI)) + v42_lapm.encode_brk(
            0, destructive=True, expedited=True)
        b._on_frame(command)
        b._on_frame(command)
        self.assertEqual(len(b.read_break_events()), 1)
        self.assertEqual(drain_frames(b), [])
        self.assertTrue(b.acknowledge_break())
        self.assertEqual(drain_frames(b), [
            (bytes((b.rsp_addr, v42_lapm.UI)) + v42_lapm.encode_brkack(1), True),
            (bytes((b.rsp_addr, v42_lapm.UI)) + v42_lapm.encode_brkack(1), True),
        ])

    def test_destructive_expedited_break_resets_data_state_but_preserves_break_state(self):
        _, b = self.connected_pair()
        b.vs, b.va, b.vr = 5, 3, 7
        b._unacked.append([3, b"outstanding"])
        b._retx.append([3, b"retry"])
        b._cq.append(b"compressed pending")
        b._txq.extend(b"unframed pending")
        b._out.extend(b"received but unread")
        b._peer_busy = b._local_busy = b._ack_pending = True
        b._rej_sent = b._need_rej = b._poll_out = True
        b._t403_bits = 1000
        b._t401_at, b._t403_at, b._tries = 12345, 23456, 2
        b._xid_pause_tx = b._peer_xid_pause = b._peer_xid_pending = True
        b._renegotiation = {"info": b"pending XID"}
        b._ua_pending_until = 34567
        b._deferred_xid_response = b"deferred XID"
        b._enc = Encoder()
        b._enc.compressed = True
        b._dec = Decoder()
        b._dec.compressed = True
        b._break_send_seq = 1
        b._post_break_txq.extend(b"data for concurrent local break")
        b._tx.extend((0, 1))  # complete the already serialized frame first

        command = bytes((b.rsp_addr, v42_lapm.UI)) + v42_lapm.encode_brk(
            0, destructive=True, expedited=True)
        b._on_frame(command)
        self.assertFalse(b.acknowledge_break())
        self.assertEqual(b.tx_bit(), 0)
        self.assertEqual(b.tx_bit(), 1)
        self.assertEqual(b.tx_bit(), 1)  # do not schedule more data before L-SIGNAL response
        self.assertTrue(b.acknowledge_break())

        self.assertEqual((b.vs, b.va, b.vr), (0, 0, 0))
        self.assertEqual((b._unacked, list(b._retx), list(b._cq)), ([], [], []))
        self.assertEqual((bytes(b._txq), bytes(b._out)), (b"", b""))
        self.assertFalse(b._peer_busy or b._local_busy or b._ack_pending
                         or b._rej_sent or b._need_rej or b._poll_out
                         or b._xid_pause_tx or b._peer_xid_pause or b._peer_xid_pending)
        self.assertIsNone(b._renegotiation)
        self.assertIsNone(b._ua_pending_until)
        self.assertIsNone(b._deferred_xid_response)
        self.assertIsNone(b._t401_at)
        self.assertEqual(b._t403_at, b.bits_sent + b._t403_bits)
        self.assertEqual(b._tries, 0)
        self.assertFalse(b._enc.compressed or b._dec.compressed)
        self.assertEqual((b._break_send_seq, b._break_recv_seq), (1, 1))
        self.assertEqual(bytes(b._post_break_txq), b"data for concurrent local break")
        self.assertEqual(drain_frames(b), [
            (bytes((b.rsp_addr, v42_lapm.UI)) + v42_lapm.encode_brkack(1), True)
        ])

    def test_transmitted_destructive_expedited_break_resets_after_wire_frame(self):
        a, _ = self.connected_pair()
        a.vs, a.va, a.vr = 4, 2, 6
        a._unacked.append([2, b"already sent"])
        a._txq.extend(b"discard before transmit")
        a._out.extend(b"decoded but unread")
        a._t403_bits = 1000
        a._t403_at = a.bits_sent + 5
        self.assertTrue(a.request_break(destructive=True, expedited=True))
        self.assertIsNone(a._t403_at)
        self.assertEqual(bytes(a._txq), b"")
        self.assertEqual((a.vs, a.va, a.vr), (4, 2, 6))
        drain_frames(a)
        self.assertEqual((a.vs, a.va, a.vr), (0, 0, 0))
        self.assertEqual(a._unacked, [])
        self.assertEqual(bytes(a._out), b"")
        self.assertIsNone(a._t403_at)  # BRK's T401 owns the transaction until BRKACK
        self.assertIsNotNone(a._break_tx_pending)
        self.assertEqual(a._break_send_seq, 0)

    def test_wrong_brkack_does_not_confirm_and_valid_brkack_does(self):
        a, _ = self.connected_pair()
        self.assertTrue(a.request_break())
        drain_frames(a)
        wrong = bytes((a.cmd_addr, v42_lapm.UI)) + v42_lapm.encode_brkack(0)
        a._on_frame(wrong)
        self.assertIsNotNone(a._break_tx_pending)
        right = bytes((a.cmd_addr, v42_lapm.UI)) + v42_lapm.encode_brkack(1)
        a._on_frame(right)
        self.assertIsNone(a._break_tx_pending)
        self.assertEqual(a.read_break_events()[0]["type"], "break_confirmed")

    def test_in_sequence_break_waits_for_prior_i_frames_and_holds_later_data(self):
        a, b = self.connected_pair()
        a.write(b"before")
        self.assertTrue(a.request_break())
        a.write(b"after")
        # Send the queued I frame, then let the peer acknowledge it.
        for _ in range(9600):
            ba, bb = a.tx_bit(), b.tx_bit()
            a.rx_bit(bb)
            b.rx_bit(ba)
        self.assertIsNotNone(a._break_tx_pending)
        self.assertIsNone(a._break_tx_waiting)
        self.assertEqual(bytes(a._txq), b"")
        self.assertEqual(bytes(a._post_break_txq), b"after")
        self.assertEqual(b.read(), b"before")
        # No I frame can overtake the unconfirmed BRK.
        self.assertEqual(a.stats["i_tx"], 1)
        b.acknowledge_break()
        for _ in range(9600):
            ba, bb = a.tx_bit(), b.tx_bit()
            a.rx_bit(bb)
            b.rx_bit(ba)
        self.assertEqual(b.read(), b"after")
        self.assertEqual(a.stats["i_tx"], 2)

    def test_non_destructive_expedited_break_precedes_and_preserves_queued_data(self):
        a, b = self.connected_pair()
        before = b"queued before expedited break"
        after = b"written while break awaits acknowledgement"
        a.write(before)
        self.assertTrue(a.request_break(destructive=False, expedited=True))
        a.write(after)

        received = bytearray()
        for _ in range(9600 * 4):
            a_bit, b_bit = a.tx_bit(), b.tx_bit()
            a.rx_bit(b_bit)
            b.rx_bit(a_bit)
            if b.read_break_events():
                self.assertEqual(a.stats["i_tx"], 0)
                self.assertTrue(b.acknowledge_break())
            received.extend(b.read())
            if (received == before + after
                    and a._break_tx_pending is None):
                break

        self.assertEqual(bytes(received), before + after)
        self.assertEqual((a.state, b.state), ("lapm", "lapm"))
        self.assertEqual(a.read_break_events()[0]["type"], "break_confirmed")
        self.assertFalse(a._break_tx_reset_after_send)
        self.assertFalse(b._break_rx_reset_applied)
        self.assertEqual(a.stats["i_tx"], 1)

    def test_break_t401_retries_same_frame_then_reports_failure(self):
        a, _ = self.connected_pair()
        a._t403_bits = 1000
        a._start_t403()
        self.assertTrue(a.request_break())
        self.assertIsNone(a._t403_at)
        original = drain_frames(a)
        self.assertEqual(len(original), 1)
        for retry in range(v42_lapm.N400):
            a.bits_sent = a._break_t401_at - 1
            a.tx_bit()
            frames = drain_frames(a)
            self.assertEqual(frames, original, retry)
        a.bits_sent = a._break_t401_at - 1
        a.tx_bit()
        self.assertIsNone(a._break_tx_pending)
        self.assertEqual(a._t403_at, a.bits_sent + a._t403_bits)
        self.assertEqual(a.read_break_events()[0]["type"], "break_failed")


class LinkRemainingTests(unittest.TestCase):
    def test_t401_poll_selects_supervisory_frame_from_receive_status(self):
        cases = (
            (False, True, False, v42_lapm.REJ, False, True),
            (False, False, True, v42_lapm.RR, False, True),
            (True, True, False, v42_lapm.RNR, True, False),
        )
        for busy, need_rej, rej_sent, control, expected_need_rej, expected_rej_sent in cases:
            with self.subTest(busy=busy, need_rej=need_rej, rej_sent=rej_sent):
                link, peer = pair(detect=False)
                pump(link, peer, 9600 * 3)
                link._local_busy = busy
                link._need_rej = need_rej
                link._rej_sent = rej_sent
                link._t401_expired()
                frames = [frame for frame, ok in drain_frames(link) if ok]
                self.assertEqual(frames[0], bytes((link.cmd_addr, control, 1)))
                self.assertTrue(link._poll_out)
                self.assertEqual(link._need_rej, expected_need_rej)
                self.assertEqual(link._rej_sent, expected_rej_sent)

    def test_originator_requires_two_adjacent_adps_for_both_outcomes(self):
        no_error_control_adp = [int(c) for c in (
            "0101000101" + "1" * 8 + "0000000001" + "1" * 8
        )]
        patterns = (
            (v42_lapm.ADP_BLOCK, "protocol", True),
            (no_error_control_adp, "detect", False),
        )
        for pattern, expected_phase, expected_adp in patterns:
            with self.subTest(expected_phase=expected_phase):
                caller = v42_lapm.V42Link(True, detect=True)
                for _ in range(v42_lapm.ADP_ARM_BITS):
                    caller.tx_bit()
                for bit in pattern:
                    caller.rx_bit(bit)
                self.assertEqual(caller._phase, "detect")
                self.assertFalse(caller.adp_seen)
                self.assertNotEqual(caller.state, "unsupported")

                for bit in pattern:
                    caller.rx_bit(bit)
                self.assertEqual(caller._phase, expected_phase)
                self.assertEqual(caller.adp_seen, expected_adp)
                if expected_phase == "detect":
                    self.assertEqual(caller.state, "unsupported")

    def test_originator_ignores_adp_before_odp_prefix_and_after_t400(self):
        pattern = v42_lapm.ADP_BLOCK * 2
        early = v42_lapm.V42Link(True, detect=True, bit_rate=1200)
        for bit in pattern:
            early.rx_bit(bit)
        self.assertFalse(early.adp_seen)
        self.assertEqual(early._phase, "detect")
        for _ in range(v42_lapm.ADP_ARM_BITS):
            early.tx_bit()
        for bit in pattern:
            early.rx_bit(bit)
        self.assertTrue(early.adp_seen)
        self.assertEqual(early._phase, "protocol")

        late = v42_lapm.V42Link(True, detect=True, bit_rate=1200)
        while late.state != "unsupported":
            late.tx_bit()
        self.assertGreater(late.bits_sent, late._t400_at)
        for bit in pattern:
            late.rx_bit(bit)
        self.assertEqual(late.state, "unsupported")
        self.assertFalse(late.adp_seen)
        self.assertEqual(late._phase, "detect")

    def test_local_busy_discards_i_information_but_processes_ack_and_poll(self):
        for ns in (0, 5):
            for pf in (0, 1):
                with self.subTest(ns=ns, pf=pf):
                    a, b = pair(detect=False)
                    pump(a, b, 9600 * 3)
                    a._local_busy = True
                    a._ack_pending = False  # RNR was already advertised.
                    a.vs = 1
                    a._unacked = [[0, b"ack this"]]
                    sent = []
                    a._s = lambda ctrl, command, final: sent.append((ctrl, command, final))

                    # Peer I command uses our response address, N(R)=1
                    # acknowledges the outstanding local frame.
                    a._on_frame(bytes((a.rsp_addr, ns << 1, (1 << 1) | pf)) + b"must be dropped")

                    self.assertEqual(a.va, 1)
                    self.assertEqual(a._unacked, [])
                    self.assertEqual(a.vr, 0)
                    self.assertEqual(a.read(), b"")
                    self.assertEqual(a.stats["i_rx"], 0)
                    self.assertEqual(sent, [(v42_lapm.RNR, False, 1)] if pf else [])

    def test_n_s_sequence_error_sends_mandatory_rej_and_recovers_exact_payload(self):
        caller, answerer = pair(detect=False)
        pump(caller, answerer, 9600 * 3)

        missing = b"missing sequence-zero frame"
        following = b"out-of-sequence frame"
        caller._unacked = [[0, missing], [1, following]]
        caller.vs = 2

        # Model loss of N(S)=0 followed by a valid-FCS N(S)=1 I-frame.
        out_of_order = bytes((answerer.rsp_addr, 1 << 1, 0)) + following
        answerer.rx_bits(v42_lapm.FLAG_BITS + v42_lapm._stuffed(out_of_order)
                         + v42_lapm.FLAG_BITS)
        self.assertEqual(answerer.vr, 0)
        self.assertEqual(answerer.read(), b"")
        self.assertEqual(answerer.stats["i_rx"], 0)

        # The receiver must not acknowledge the gap; its mandatory REJ asks
        # for N(S)=V(R)=0. Feed the actual bit-framed response to the sender.
        parser = v42_lapm._Hdlc()
        response_bits = []
        response = None
        for _ in range(2048):
            bit = answerer.tx_bit()
            response_bits.append(bit)
            frames = parser.push(bit)
            if frames:
                response = frames[0]
                break
        self.assertIsNotNone(response)
        frame, valid_fcs = response
        self.assertTrue(valid_fcs)
        self.assertEqual(frame, bytes((answerer.rsp_addr, v42_lapm.REJ, 0)))
        caller.rx_bits(response_bits)
        self.assertEqual(list(caller._retx), [(0, missing), (1, following)])

        _, received = pump(caller, answerer, 9600 * 4)
        self.assertEqual(received, missing + following)
        self.assertEqual(answerer.vr, 2)
        self.assertEqual(answerer.stats["i_rx"], 2)
        self.assertEqual(answerer.read(), b"")

    def test_polled_sequence_gap_gets_rej_final_immediately(self):
        caller, answerer = pair(detect=False)
        pump(caller, answerer, 9600 * 3)

        # The peer sends N(S)=1 while V(R)=0 and sets P=1. This is the first
        # report of the sequence error, so the solicited response must be
        # REJ(F=1), and the out-of-order payload must remain withheld.
        out_of_order = bytes((answerer.rsp_addr, 1 << 1, 1)) + b"gap data"
        answerer.rx_bits(v42_lapm.FLAG_BITS + v42_lapm._stuffed(out_of_order)
                         + v42_lapm.FLAG_BITS)
        self.assertEqual(answerer.vr, 0)
        self.assertEqual(answerer.read(), b"")
        self.assertEqual(answerer.stats["rej_tx"], 1)

        parser = v42_lapm._Hdlc()
        response = None
        for _ in range(2048):
            frames = parser.push(answerer.tx_bit())
            if frames:
                response = frames[0]
                break
        self.assertIsNotNone(response)
        frame, valid_fcs = response
        self.assertTrue(valid_fcs)
        self.assertEqual(frame, bytes((answerer.rsp_addr, v42_lapm.REJ, 1)))
        self.assertFalse(answerer._need_rej)
        self.assertTrue(answerer._rej_sent)

    def test_receiver_busy_sends_rnr_before_pending_rej_then_rej_on_clear(self):
        caller, receiver = pair(detect=False)
        pump(caller, receiver, 9600 * 3)
        gap = bytes((receiver.rsp_addr, 1 << 1, 0)) + b"out of order"
        receiver._on_frame(gap)
        self.assertTrue(receiver._need_rej)

        # If receive backpressure arrives before the pending sequence-error
        # response is sent, advertise busy with RNR and retain the REJ for
        # after busy clears (V.42 §8.4.7 and the §8.4.6 Note 2 response rule).
        receiver.set_local_busy(True)
        parser = v42_lapm._Hdlc()
        first = None
        for _ in range(2048):
            for frame, ok in parser.push(receiver.tx_bit()):
                if ok:
                    first = frame
                    break
            if first is not None:
                break
        self.assertEqual(first, bytes((receiver.rsp_addr, v42_lapm.RNR, 0)))
        self.assertTrue(receiver._need_rej)
        self.assertFalse(receiver._rej_sent)

        receiver.set_local_busy(False)
        parser = v42_lapm._Hdlc()
        next_frame = None
        for _ in range(2048):
            for frame, ok in parser.push(receiver.tx_bit()):
                if ok:
                    next_frame = frame
                    break
            if next_frame is not None:
                break
        self.assertEqual(next_frame, bytes((receiver.rsp_addr, v42_lapm.REJ, 0)))
        self.assertFalse(receiver._need_rej)
        self.assertTrue(receiver._rej_sent)

    def test_expected_frame_clears_pending_rej_before_it_is_sent(self):
        caller, answerer = pair(detect=False)
        pump(caller, answerer, 9600 * 3)

        gap = bytes((answerer.rsp_addr, 1 << 1, 0)) + b"out of order"
        answerer._on_frame(gap)
        self.assertTrue(answerer._need_rej)
        self.assertEqual(answerer.read(), b"")

        expected_payload = b"retransmitted in order"
        expected = bytes((answerer.rsp_addr, 0, 0)) + expected_payload
        answerer._on_frame(expected)
        self.assertEqual(answerer.read(), expected_payload)
        self.assertFalse(answerer._need_rej)

        parser = v42_lapm._Hdlc()
        response = None
        for _ in range(2048):
            frames = parser.push(answerer.tx_bit())
            if frames:
                response = frames[0]
                break
        self.assertIsNotNone(response)
        frame, valid_fcs = response
        self.assertTrue(valid_fcs)
        self.assertNotEqual(frame[1] & 0xEF, v42_lapm.REJ)

    def test_i_frame_in_response_role_is_rejected_without_delivering_or_acknowledging(self):
        caller, receiver = pair(detect=False)
        pump(caller, receiver, 9600 * 3)

        receiver._poll_out = True
        receiver._t401_at = 1234
        receiver.vs, receiver.va, receiver._unacked = 1, 0, [[0, b"pending"]]
        # Table 8 defines I frames as commands only. A response-address I
        # frame must not be treated as user data or as a final response.
        response_i = bytes((receiver.cmd_addr, 0, 1)) + b"response data"
        receiver._on_frame(response_i)
        self.assertEqual(receiver.state, "down")
        self.assertEqual(receiver.read(), b"")
        self.assertEqual(receiver.va, 0)
        self.assertEqual(receiver._unacked, [[0, b"pending"]])
        self.assertFalse(receiver._poll_out)
        self.assertIsNone(receiver._t401_at)
        self.assertEqual(receiver.stats["frame_reject_errors"], 1)
        self.assertIn("I frame received in response role", receiver.events[-1])

        originator = v42_lapm.V42Link(calling_party=True, detect=False)
        originator._sabme_sent = True
        originator._on_frame(bytes((originator.cmd_addr, 0, 0)) + b"not establishment")
        self.assertEqual(originator.state, "idle")
        self.assertFalse(originator._xid_done)
        self.assertIn("while disconnected; discarded", originator.events[-1])

        caller, receiver = pair(detect=False)
        pump(caller, receiver, 9600 * 3)
        # The response role is rejected before sequence-gap recovery can
        # process its N(S), N(R), or F bit.
        gap_response = bytes((receiver.cmd_addr, 1 << 1, 1)) + b"out of order"
        receiver._on_frame(gap_response)
        self.assertEqual(receiver.state, "down")
        self.assertFalse(receiver._need_rej)
        self.assertEqual(receiver.read(), b"")

        caller, receiver = pair(detect=False)
        pump(caller, receiver, 9600 * 3)
        receiver._local_busy = True  # RNR(F=0) was already advertised.
        sent = []
        receiver._s = lambda ctrl, command, final: sent.append((ctrl, command, final))
        receiver._on_frame(bytes((receiver.rsp_addr, 0, 0)) + b"discarded")
        self.assertEqual(sent, [])
        self.assertEqual(receiver.read(), b"")

    def test_test_command_is_negotiated_and_control_functions_can_loop_payload(self):
        caller, answerer = pair(detect=False)
        pump(caller, answerer, 9600 * 3)
        self.assertTrue(caller._test_negotiated)
        self.assertTrue(answerer._test_negotiated)

        probe = bytes(range(128))
        self.assertTrue(caller.request_test(probe))
        pump(caller, answerer, 9600)
        self.assertEqual(answerer.take_test_indication(), probe)
        self.assertIsNone(answerer.take_test_indication())

        # The TEST information field is delivered to the control function;
        # that function chooses to return the same bytes in its own L-TEST.
        self.assertTrue(answerer.request_test(probe))
        pump(answerer, caller, 9600)
        self.assertEqual(caller.take_test_indication(), probe)
        self.assertEqual(caller.state, "lapm")
        self.assertEqual(answerer.state, "lapm")

    def test_test_command_rejected_when_the_optional_procedure_was_not_agreed(self):
        caller, peer = pair(detect=False)
        pump(caller, peer, 9600 * 3)
        caller._test_negotiated = peer._test_negotiated = False
        self.assertFalse(caller.request_test(b"probe"))
        peer._on_frame(bytes((peer.rsp_addr, v42_lapm.TEST)) + b"probe")
        self.assertEqual(peer.state, "down")
        self.assertIn("unnegotiated", peer.events[-1])

    def test_acknowledged_payload_counts_only_valid_advancing_acknowledgements(self):
        link=v42_lapm.V42Link(True,detect=False,bit_rate=9600)
        link._unacked=[[126,b'abc'],[127,b'de'],[0,b'fghi']]
        link._ack(127)
        self.assertEqual(link.stats['wire_acked'],3)
        link._ack(127)  # Duplicate N(R), no new delivered payload.
        link._ack(4)    # Outside the outstanding sequence window.
        self.assertEqual(link.stats['wire_acked'],3)
        link._ack(1)    # Valid acknowledgement across sequence rollover.
        self.assertEqual(link.stats['wire_acked'],9)
        link._ack(1)
        self.assertEqual(link.stats['wire_acked'],9)

    def test_i_frame_ack_does_not_stop_t401_while_peer_busy(self):
        a, b = pair(detect=False)
        pump(a, b, 9600 * 3)
        a.vs, a.va, a.vr = 1, 0, 0
        a._unacked = [[0, b"outstanding"]]
        a._peer_busy = True
        a._t401_at = 1234

        # The valid incoming I frame acknowledges our final outstanding
        # frame, but the peer's busy condition still requires T401 enquiry.
        a._on_frame(bytes((a.rsp_addr, 0x00, 1 << 1)) + b"peer data")

        self.assertEqual(a.state, "lapm")
        self.assertEqual((a.va, a._unacked), (1, []))
        self.assertTrue(a._peer_busy)
        self.assertEqual(a._t401_at, 1234)
        self.assertEqual(a.read(), b"peer data")

    def test_rnr_zero_final_restarts_t401_after_acknowledging_last_frame(self):
        for role in ("command", "response"):
            with self.subTest(role=role):
                a, b = pair(detect=False)
                pump(a, b, 9600 * 3)
                address = a.rsp_addr if role == "command" else a.cmd_addr
                a.vs, a.va = 1, 0
                a._unacked = [[0, b"last frame"]]
                a._t401_at = 1234

                # A non-final RNR acknowledges the last data frame and leaves
                # the peer busy. T401 must continue so we can poll for ready.
                a._on_frame(bytes((address, v42_lapm.RNR, 1 << 1)))

                self.assertEqual(a.state, "lapm")
                self.assertEqual((a.va, a._unacked), (1, []))
                self.assertTrue(a._peer_busy)
                self.assertEqual(a._t401_at, a.bits_sent + a._t401_bits)

    def test_invalid_nr_in_information_frame_terminates_connected_link(self):
        a, b = pair(detect=False)
        pump(a, b, 9600 * 3)
        self.assertEqual(a.state, "lapm")
        # V(A) == V(S) == 0, so N(R)=1 acknowledges a frame never sent.
        a._on_frame(bytes((a.rsp_addr, 0x00, 0x02)) + b"bad ack")
        self.assertEqual(a.state, "down")
        self.assertEqual(a.stats["nr_errors"], 1)
        self.assertIn("invalid N(R) 1", a.events[-1])
        self.assertEqual(a.stats["i_rx"], 0)

    def test_invalid_nr_in_supervisory_frame_terminates_connected_link(self):
        a, b = pair(detect=False)
        pump(a, b, 9600 * 3)
        self.assertEqual(a.state, "lapm")
        # RR with N(R)=1 is outside the empty outstanding range [0, 0].
        a._on_frame(bytes((a.rsp_addr, v42_lapm.RR, 0x02)))
        self.assertEqual(a.state, "down")
        self.assertEqual(a.stats["nr_errors"], 1)
        self.assertIn("invalid N(R) 1", a.events[-1])

    def test_unpaired_final_supervisory_responses_are_ignored(self):
        for ctrl in (v42_lapm.RR, v42_lapm.RNR, v42_lapm.REJ):
            pf = 1
            with self.subTest(ctrl=hex(ctrl), pf=pf):
                a, b = pair(detect=False)
                pump(a, b, 9600 * 3)
                a.vs, a.va = 1, 0
                a._unacked = [[0, b"pending"]]
                a._peer_busy = False
                a._poll_out = False
                a._t401_at = 1234
                frame = bytes((a.cmd_addr, ctrl, (1 << 1) | pf))

                a._on_frame(frame)

                self.assertEqual(a.state, "lapm")
                self.assertEqual((a.vs, a.va), (1, 0))
                self.assertEqual(a._unacked, [[0, b"pending"]])
                self.assertFalse(a._peer_busy)
                self.assertFalse(a._poll_out)
                self.assertEqual(a._t401_at, 1234)
                self.assertEqual(a.stats["rej_rx"], 0)
                self.assertIn("unsolicited", a.events[-1])

    def test_unpaired_final_does_not_hide_malformed_or_unsupported_s_frames(self):
        cases = (
            (bytes((v42_lapm.RR, 0, 0)), "incorrect S-frame length"),
            (bytes((0x0D, 0)), "unnegotiated SREJ"),
            (bytes((0x11, 0)), "undefined S control"),
        )
        for control, expected in cases:
            with self.subTest(control=control.hex()):
                a, b = pair(detect=False)
                pump(a, b, 9600 * 3)
                # A response uses our command address and carries F=1, but
                # there is no outstanding poll. Structural/frame-rejection
                # checks still precede the unsolicited-response ignore rule.
                a._on_frame(bytes((a.cmd_addr, control[0], control[1] | 1))
                            + control[2:])
                self.assertEqual(a.state, "down")
                self.assertEqual(a.stats["frame_reject_errors"], 1)
                self.assertIn(expected, a.events[-1])

    def test_rej_f0_response_requests_sequential_retransmission(self):
        a, b = pair(detect=False)
        pump(a, b, 9600 * 3)
        a.vs, a.va = 2, 0
        a._unacked = [[0, b"first"], [1, b"second"]]

        # REJ(F=0) reports a receive sequence gap; it is not an unsolicited
        # status response and must remain effective without an outstanding P.
        a._on_frame(bytes((a.cmd_addr, v42_lapm.REJ, 0)))

        self.assertEqual(a.state, "lapm")
        self.assertEqual(list(a._retx), [(0, b"first"), (1, b"second")])
        self.assertEqual(a.stats["rej_rx"], 1)
        self.assertEqual((a.va, a.vs), (0, 0))

        # A REJ rewinds V(S) to N(R); sequential retransmissions advance it
        # again so acknowledgements for those retransmissions validate.
        a._schedule_data()
        self.assertEqual(a.vs, 1)
        first = [frame for frame, ok in drain_frames(a) if ok]
        self.assertEqual([frame[1] >> 1 for frame in first if frame[1] & 1 == 0], [0])
        a._schedule_data()
        self.assertEqual(a.vs, 2)
        second = [frame for frame, ok in drain_frames(a) if ok]
        self.assertEqual([frame[1] >> 1 for frame in second if frame[1] & 1 == 0], [1])
        self.assertTrue(a._valid_nr(2))

    def test_solicited_final_rewinds_v_s_before_retransmission(self):
        for ctrl, peer_busy in ((v42_lapm.RR, False), (v42_lapm.REJ, False),
                                (v42_lapm.RNR, True)):
            with self.subTest(ctrl=hex(ctrl)):
                a, b = pair(detect=False)
                pump(a, b, 9600 * 3)
                a.vs, a.va = 3, 0
                a._unacked = [[0, b"acked"], [1, b"retry-1"], [2, b"retry-2"]]
                a._poll_out = True
                a._tries = 4
                a._t401_at = 1234

                # N(R)=1 acknowledges the first frame. The valid F=1 closes
                # timer recovery and makes frame 1 the next sequence to send.
                a._on_frame(bytes((a.cmd_addr, ctrl, (1 << 1) | 1)))

                self.assertEqual(a.va, 1)
                self.assertEqual(a.vs, 1)
                self.assertEqual(a._unacked, [[1, b"retry-1"], [2, b"retry-2"]])
                self.assertEqual(list(a._retx), [(1, b"retry-1"), (2, b"retry-2")])
                self.assertEqual(a._peer_busy, peer_busy)
                self.assertFalse(a._poll_out)
                self.assertEqual(a._tries, 0)

                if not peer_busy:
                    a._schedule_data()
                    self.assertEqual(a.vs, 2)
                    frames = [frame for frame, ok in drain_frames(a) if ok]
                    i_frames = [frame for frame in frames if frame[1] & 1 == 0]
                    self.assertEqual([frame[1] >> 1 for frame in i_frames], [1])
                    self.assertTrue(a._valid_nr(2))

    def test_solicited_final_supervisory_response_completes_poll(self):
        a, b = pair(detect=False)
        pump(a, b, 9600 * 3)
        a.vs, a.va = 1, 0
        a._unacked = [[0, b"acknowledge"]]
        a._poll_out = True
        a._t401_at = 1234

        a._on_frame(bytes((a.cmd_addr, v42_lapm.RR, (1 << 1) | 1)))

        self.assertEqual(a.state, "lapm")
        self.assertEqual(a.va, 1)
        self.assertEqual(a.vs, 1)
        self.assertEqual(a._unacked, [])
        self.assertFalse(a._poll_out)
        self.assertIsNone(a._t401_at)

    def test_rnr_final_restarts_t401_after_acknowledging_last_frame(self):
        a, b = pair(detect=False)
        pump(a, b, 9600 * 3)
        a.vs, a.va = 1, 0
        a._unacked = [[0, b"last frame"]]
        a._poll_out = True
        a._t401_at = a.bits_sent + a._t401_bits

        # The peer's RNR(F=1) is a solicited response and acknowledges the
        # final outstanding I frame. LAPM must keep T401 alive to poll again
        # when that peer has not yet cleared its busy condition.
        a._on_frame(bytes((a.cmd_addr, v42_lapm.RNR, (1 << 1) | 1)))

        self.assertEqual(a.va, 1)
        self.assertEqual(a._unacked, [])
        self.assertTrue(a._peer_busy)
        self.assertFalse(a._poll_out)
        self.assertEqual(a._t401_at, a.bits_sent + a._t401_bits)

        a._tx.clear()
        a.bits_sent = a._t401_at
        a.tx_bit()
        self.assertTrue(a._poll_out)
        self.assertGreater(a._t401_at, a.bits_sent)
        poll_frames = [frame for frame, ok in drain_frames(a) if ok]
        self.assertTrue(any(
            (frame[1] & 0xEF) == v42_lapm.RR and (frame[2] & 1)
            for frame in poll_frames
        ), poll_frames)

    def test_rr_rnr_final_zero_during_poll_updates_status_without_ending_recovery(self):
        for ctrl, expected_busy in ((v42_lapm.RR, False), (v42_lapm.RNR, True)):
            with self.subTest(ctrl=hex(ctrl)):
                a, b = pair(detect=False)
                pump(a, b, 9600 * 3)
                a.vs, a.va = 1, 0
                a._unacked = [[0, b"acknowledge"]]
                a._peer_busy = not expected_busy
                a._poll_out = True
                a._t401_at = 1234

                # F=0 does not answer the outstanding poll, but §8.4.6 still
                # processes this status/acknowledgement during timer recovery.
                a._on_frame(bytes((a.cmd_addr, ctrl, 1 << 1)))

                self.assertEqual(a.state, "lapm")
                self.assertEqual(a.va, 1)
                self.assertEqual(a._unacked, [])
                self.assertEqual(a._peer_busy, expected_busy)
                self.assertTrue(a._poll_out)
                self.assertEqual(a._t401_at, 1234)

    def test_peer_poll_response_reflects_local_busy_and_reject_state(self):
        states = (
            ("reject pending", False, True, False, v42_lapm.REJ, True, False, 1),
            ("reject already sent", False, False, True, v42_lapm.RR, True, False, 0),
            ("local busy", True, True, False, v42_lapm.RNR, False, True, 0),
        )
        for (name, local_busy, need_rej, rej_sent, expected,
             expected_rej_sent, expected_need_rej, expected_rej_tx) in states:
            for poll in (v42_lapm.RR, v42_lapm.RNR, v42_lapm.REJ):
                with self.subTest(state=name, poll=hex(poll)):
                    a, b = pair(detect=False)
                    pump(a, b, 9600 * 3)
                    a._local_busy = local_busy
                    a._need_rej = need_rej
                    a._rej_sent = rej_sent
                    a._ack_pending = True

                    # A valid supervisory command with P=1 asks for our
                    # immediate status response. Pending reject exceptions
                    # require REJ(F=1), while an existing REJ exception uses
                    # RR(F=1) until the requested frame arrives.
                    a._on_frame(bytes((a.rsp_addr, poll, 1)))
                    replies = [frame for frame, ok in drain_frames(a) if ok]
                    status = next(frame for frame in replies
                                  if (frame[1] & 0x03) == 0x01)
                    self.assertEqual((status[0], status[1] & 0xEF, status[2] & 1),
                                     (a.rsp_addr, expected, 1))
                    self.assertEqual(a._rej_sent, expected_rej_sent)
                    self.assertEqual(a._need_rej, expected_need_rej)
                    self.assertFalse(a._ack_pending)
                    self.assertEqual(a.stats["rej_tx"], expected_rej_tx)

    def test_peer_frmr_response_terminates_connected_link(self):
        a, b = pair(detect=False)
        pump(a, b, 9600 * 3)
        # FRMR is a peer response, so it uses our command address.
        a._on_frame(bytes((a.cmd_addr, v42_lapm.FRMR, 0, 0, 0, 0, 0)))
        self.assertEqual(a.state, "down")
        self.assertEqual(a.stats["frame_reject_errors"], 1)
        self.assertIn("FRMR received", a.events[-1])

    def test_unsolicited_dm_f1_is_ignored_but_dm_f0_terminates(self):
        a, b = pair(detect=False)
        pump(a, b, 9600 * 3)
        a._on_frame(bytes((a.cmd_addr, v42_lapm.DM | 0x10)))
        self.assertEqual(a.state, "lapm")
        self.assertIn("unsolicited DM(F=1) ignored", a.events[-1])

        a, b = pair(detect=False)
        pump(a, b, 9600 * 3)
        a._poll_out = True
        a._t401_at = 1234
        a._on_frame(bytes((a.cmd_addr, v42_lapm.DM | 0x10)))
        self.assertEqual(a.state, "down")
        self.assertFalse(a._poll_out)
        self.assertIsNone(a._t401_at)
        self.assertIn("DM(F=1) during timer recovery", a.events[-1])

        a, b = pair(detect=False)
        pump(a, b, 9600 * 3)
        a._on_frame(bytes((a.cmd_addr, v42_lapm.DM)))
        self.assertEqual(a.state, "down")

    def test_unsolicited_dm_f0_does_not_abort_disconnected_answerer(self):
        answerer = v42_lapm.V42Link(False, detect=False, bit_rate=9600)
        self.assertEqual(answerer.state, "idle")

        answerer._on_frame(bytes((answerer.cmd_addr, v42_lapm.DM)))

        self.assertEqual(answerer.state, "idle")
        self.assertIn("unsolicited DM(F=0) ignored while disconnected", answerer.events[-1])

    def test_unsolicited_dm_f1_does_not_abort_disconnected_answerer(self):
        answerer = v42_lapm.V42Link(False, detect=False, bit_rate=9600)
        self.assertEqual(answerer.state, "idle")

        answerer._on_frame(bytes((answerer.cmd_addr, v42_lapm.DM | 0x10)))

        self.assertEqual(answerer.state, "idle")
        self.assertIn("unsolicited DM(F=1) ignored while disconnected", answerer.events[-1])

        # Ignoring the stray response leaves the control function able to
        # accept the peer's later, legitimate establishment command.
        answerer._on_frame(bytes((answerer.rsp_addr, v42_lapm.SABME | 0x10)))
        self.assertEqual(answerer.state, "lapm")

    def test_unsolicited_dm_f1_does_not_abort_caller_xid_phase(self):
        originator = v42_lapm.V42Link(True, detect=False, bit_rate=9600)
        originator._send_xid_command()
        xid_deadline = originator._t401_at
        self.assertFalse(originator._sabme_sent)

        originator._on_frame(bytes((originator.cmd_addr, v42_lapm.DM | 0x10)))

        self.assertEqual(originator.state, "idle")
        self.assertFalse(originator._sabme_sent)
        self.assertEqual(originator._t401_at, xid_deadline)
        self.assertIn("unsolicited DM(F=1) ignored while disconnected", originator.events[-1])

        response = (bytes((originator.cmd_addr, v42_lapm.XID))
                    + v42_lapm.build_xid(128, 15, None))
        originator._on_frame(response)
        self.assertTrue(originator._sabme_sent)
        self.assertIsNotNone(originator._t401_at)

    def test_unnegotiated_srej_is_not_silently_ignored(self):
        a, b = pair(detect=False)
        pump(a, b, 9600 * 3)
        a._on_frame(bytes((a.rsp_addr, v42_lapm.SREJ, 0)))
        self.assertEqual(a.state, "down")
        self.assertEqual(a.stats["frame_reject_errors"], 1)
        self.assertIn("unnegotiated SREJ", a.events[-1])

    def test_supervisory_frame_with_information_is_rejected(self):
        a, b = pair(detect=False)
        pump(a, b, 9600 * 3)
        a._on_frame(bytes((a.rsp_addr, v42_lapm.RR, 0, 0xAA)))
        self.assertEqual(a.state, "down")
        self.assertIn("incorrect S-frame length", a.events[-1])

    def test_undefined_supervisory_control_is_rejected(self):
        a, b = pair(detect=False)
        pump(a, b, 9600 * 3)
        a._on_frame(bytes((a.rsp_addr, 0x15, 0)))
        self.assertEqual(a.state, "down")
        self.assertIn("undefined S control", a.events[-1])

    def test_i_frame_above_negotiated_n401_is_rejected(self):
        a, b = pair(detect=False)
        pump(a, b, 9600 * 3)
        payload = b"x" * (a.n401 + 1)
        a._on_frame(bytes((a.rsp_addr, 0, 0)) + payload)
        self.assertEqual(a.state, "down")
        self.assertIn("exceeds N401", a.events[-1])

    def test_malformed_unnumbered_frame_is_rejected(self):
        a, b = pair(detect=False)
        pump(a, b, 9600 * 3)
        a._on_frame(bytes((a.rsp_addr, v42_lapm.SABME, 0)))
        self.assertEqual(a.state, "down")
        self.assertIn("incorrect U-frame length", a.events[-1])

    def test_unnumbered_controls_in_wrong_command_response_role_terminate(self):
        # Table 8/V.42 marks SABME, DISC, TEST as commands only and UA, DM,
        # FRMR as responses only. Each is a §8.5.5 frame-rejection condition
        # once the link is established.
        cases = (
            (v42_lapm.SABME, False, 2),
            (v42_lapm.DISC, False, 2),
            (v42_lapm.TEST, False, 2),
            (v42_lapm.UA, True, 2),
            (v42_lapm.DM, True, 2),
            (v42_lapm.FRMR, True, 7),
        )
        for ctrl, invalid_command, size in cases:
            with self.subTest(ctrl=hex(ctrl), invalid_command=invalid_command):
                a, b = pair(detect=False)
                pump(a, b, 9600 * 3)
                addr = a.rsp_addr if invalid_command else a.cmd_addr
                frame = bytes((addr, ctrl)) + bytes(size - 2)
                a._on_frame(frame)
                self.assertEqual(a.state, "down")
                self.assertEqual(a.stats["frame_reject_errors"], 1)
                self.assertIn("invalid", a.events[-1])

    def test_answerer_acknowledges_repeated_initial_sabme_only_in_t401_window(self):
        a, b = pair(detect=False)
        pump(a, b, 9600 * 3)
        self.assertEqual((a.state, b.state), ("lapm", "lapm"))

        # Model a lost first UA: a duplicate SABME inside the initial T401
        # response window is acknowledged without resetting the established
        # link or its sequence state.
        b._connected_at = b.bits_sent
        frame = bytes((b.rsp_addr, v42_lapm.SABME | 0x10))
        b._on_frame(frame)
        self.assertEqual(b.state, "lapm")
        self.assertIn("repeated establishment SABME acknowledged", b.events[-1])
        self.assertTrue(b._tx)
        repeated_ua = [frame for frame, ok in drain_frames(b) if ok]
        self.assertEqual(repeated_ua, [bytes((b.rsp_addr, v42_lapm.UA | 0x10))])

        # A later SABME is a termination condition, not a fresh re-establish.
        b._connected_at = b.bits_sent - b._t401_bits - 9
        b._on_frame(bytes((b.rsp_addr, v42_lapm.SABME)))
        self.assertEqual(b.state, "down")
        self.assertIn("established LAPM link: terminating", b.events[-1])

    def test_repeated_sabme_ua_precedes_rej_retransmission_request(self):
        caller, answerer = pair(detect=False)
        pump(caller, answerer, 9600 * 3)
        self.assertEqual(answerer.state, "lapm")

        # Model an outstanding answerer I frame and a duplicate initial
        # SABME whose first UA was lost. A REJ arriving before the repeated
        # UA finishes must not start retransmission ahead of that UA.
        answerer._queue_i(b"outstanding")
        answerer._connected_at = answerer.bits_sent
        answerer._on_frame(bytes((answerer.rsp_addr, v42_lapm.SABME | 0x10)))
        self.assertIsNotNone(answerer._ua_pending_until)
        answerer._on_frame(bytes((answerer.rsp_addr, v42_lapm.REJ, 0)))
        self.assertEqual(list(answerer._retx), [])
        self.assertEqual(answerer.vs, 1)
        self.assertIn("frame ignored until pending UA response is transmitted", answerer.events[-1])

        frames = [frame for frame, ok in drain_frames(answerer) if ok]
        self.assertEqual(frames, [bytes((answerer.rsp_addr, v42_lapm.UA | 0x10))])
        self.assertIsNone(answerer._ua_pending_until)

        answerer._on_frame(bytes((answerer.rsp_addr, v42_lapm.REJ, 0)))
        self.assertEqual(list(answerer._retx), [(0, b"outstanding")])
        self.assertEqual(answerer.vs, 0)

    def test_rej_completes_active_supervisory_and_final_before_retransmission(self):
        caller, sender = pair(detect=False)
        pump(caller, sender, 9600 * 3)
        payload = b"requested frame"
        sender.vs = 1
        sender._unacked = [[0, payload]]

        # Queue an RR response before the peer's REJ(P=1). The active
        # supervisory frame must finish; the solicited RR(F=1) must also be
        # sent before the requested I-frame (§8.4.4).
        sender._s(v42_lapm.RR, False, 0)
        sender._on_frame(bytes((sender.rsp_addr, v42_lapm.REJ, 1)))
        parser = v42_lapm._Hdlc()
        frames = []
        for _ in range(4096):
            frames.extend(parser.push(sender.tx_bit()))
            if sum(ok for _, ok in frames) >= 3:
                break
        valid = [frame for frame, ok in frames if ok]
        self.assertEqual(len(valid), 3)
        self.assertEqual(valid[0], bytes((sender.rsp_addr, v42_lapm.RR, 0)))
        self.assertEqual(valid[1], bytes((sender.rsp_addr, v42_lapm.RR, 1)))
        self.assertEqual(valid[2], bytes((sender.cmd_addr, 0, 0)) + payload)

    def test_originator_uses_first_i_frame_as_lost_ua_confirmation(self):
        originator, peer = pair(detect=False)
        originator._sabme_sent = True
        originator._start_t401()
        peer._connected()

        payload = b"first data after lost UA"
        peer._send_i(0, payload)
        frames = [frame for frame, ok in drain_frames(peer) if ok]
        self.assertEqual(len(frames), 1)

        # V.42 §8.3.2 permits the originator to treat the first I/S frame as
        # proof the peer accepted SABME when the UA was lost. The frame must
        # then be processed normally rather than discarded in idle state.
        originator._on_frame(frames[0])
        self.assertEqual(originator.state, "lapm")
        self.assertEqual(originator.read(), payload)
        self.assertIn("connected", originator.events)

    def test_originator_uses_first_supervisory_frame_as_lost_ua_confirmation(self):
        originator, peer = pair(detect=False)
        originator._sabme_sent = True
        originator._start_t401()
        peer._connected()
        peer._s(v42_lapm.RR, True, 0)
        frames = [frame for frame, ok in drain_frames(peer) if ok]
        self.assertEqual(len(frames), 1)

        originator._on_frame(frames[0])
        self.assertEqual(originator.state, "lapm")
        self.assertIn("connected", originator.events)

    def test_connected_disc_sends_ua_before_orderly_release(self):
        a, b = pair(detect=False)
        pump(a, b, 9600 * 3)
        a._tx.clear()
        a._on_frame(bytes((a.rsp_addr, v42_lapm.DISC | 0x10)))
        self.assertEqual(a.state, "release_pending")
        self.assertEqual(a.stats["disc_rx"], 1)
        self.assertEqual(a.stats["ua_tx"], 1)

        rx = v42_lapm._Hdlc()
        completed = []
        while a.state == "release_pending":
            completed.extend(rx.push(a.tx_bit()))
        self.assertEqual(a.state, "released")
        self.assertEqual(completed, [(bytes((a.rsp_addr, v42_lapm.UA | 0x10)), True)])

    def test_disconnected_disc_is_answered_with_dm(self):
        link = v42_lapm.V42Link(False, detect=False)
        link._on_frame(bytes((link.rsp_addr, v42_lapm.DISC | 0x10)))
        rx = v42_lapm._Hdlc()
        completed = []
        for bit in list(link._tx) + v42_lapm.FLAG_BITS:
            completed.extend(rx.push(bit))
        self.assertEqual(completed, [(bytes((link.rsp_addr, v42_lapm.DM | 0x10)), True)])
        self.assertEqual(link.state, "idle")

    def test_answerer_discards_i_frame_until_sabme_ua_is_transmitted(self):
        answerer = v42_lapm.V42Link(False, detect=False)
        answerer._tx.clear()
        premature = b"arrived before UA was transmitted"
        ua_wire = list(v42_lapm.FLAG_BITS + v42_lapm._stuffed(
            bytes((answerer.rsp_addr, v42_lapm.SABME | 0x10))) + v42_lapm.FLAG_BITS)
        early_i_wire = list(v42_lapm.FLAG_BITS + v42_lapm._stuffed(
            bytes((answerer.rsp_addr, 0, 0)) + premature) + v42_lapm.FLAG_BITS)
        answerer.rx_bits(ua_wire + early_i_wire)
        self.assertEqual(answerer.state, "lapm")
        self.assertEqual(answerer.read(), b"")
        self.assertEqual(answerer.vr, 0)

        ua_frames = [frame for frame, ok in drain_frames(answerer) if ok]
        self.assertEqual(ua_frames, [bytes((answerer.rsp_addr, v42_lapm.UA | 0x10))])

        accepted = b"arrived after UA was transmitted"
        accepted_i_wire = (v42_lapm.FLAG_BITS + v42_lapm._stuffed(
            bytes((answerer.rsp_addr, 0, 0)) + accepted) + v42_lapm.FLAG_BITS)
        answerer.rx_bits(accepted_i_wire)
        self.assertEqual(answerer.read(), accepted)

    def test_simultaneous_sabme_waits_for_peer_ua_before_connecting(self):
        link = v42_lapm.V42Link(True, detect=False)
        link._sabme_sent = True
        link._t401_at = link._t401_bits * 2
        link._on_frame(bytes((link.rsp_addr, v42_lapm.SABME | 0x10)))
        self.assertEqual(link.state, "idle")
        self.assertIn("simultaneous SABME; awaiting UA response", link.events[-1])
        link._on_frame(bytes((link.cmd_addr, v42_lapm.UA | 0x10)))
        self.assertEqual(link.state, "lapm")

        early = b"simultaneous data before our UA is sent"
        link._on_frame(bytes((link.rsp_addr, 0, 0)) + early)
        self.assertEqual(link.read(), b"")
        self.assertEqual(drain_frames(link), [(bytes((link.rsp_addr, v42_lapm.UA | 0x10)), True)])

        after = b"simultaneous data after our UA is sent"
        link._on_frame(bytes((link.rsp_addr, 0, 0)) + after)
        self.assertEqual(link.read(), after)

    def test_ua_without_final_bit_does_not_acknowledge_outstanding_sabme(self):
        link = v42_lapm.V42Link(True, detect=False)
        link._sabme_sent = True
        link._on_frame(bytes((link.cmd_addr, v42_lapm.UA)))
        self.assertEqual(link.state, "idle")

    def test_simultaneous_disc_sends_ua_then_waits_for_peer_ua(self):
        link, peer = pair(detect=False)
        pump(link, peer, 9600 * 3)
        self.assertTrue(link.release())
        deadline = link._t401_at
        link._on_frame(bytes((link.rsp_addr, v42_lapm.DISC | 0x10)))
        self.assertEqual(link.state, "releasing")
        self.assertEqual(link._t401_at, deadline)
        link._on_frame(bytes((link.cmd_addr, v42_lapm.UA | 0x10)))
        self.assertEqual(link.state, "release_pending")  # finish our UA before mode change
        frames = drain_frames(link)
        self.assertEqual(frames, [(bytes((link.rsp_addr, v42_lapm.UA | 0x10)), True)])
        self.assertEqual(link.state, "released")

    def test_disc_sabme_collision_resolves_with_dm(self):
        link, peer = pair(detect=False)
        pump(link, peer, 9600 * 3)
        self.assertTrue(link.release())
        link._on_frame(bytes((link.rsp_addr, v42_lapm.SABME | 0x10)))
        self.assertEqual(link.state, "releasing")
        self.assertIn("SABME/DISC collision; DM response sent", link.events[-1])
        link._on_frame(bytes((link.cmd_addr, v42_lapm.DM | 0x10)))
        self.assertEqual(link.state, "release_pending")
        self.assertIn((bytes((link.rsp_addr, v42_lapm.DM | 0x10)), True), drain_frames(link))
        self.assertEqual(link.state, "released")

    def test_sabme_disc_collision_resolves_with_dm(self):
        link = v42_lapm.V42Link(True, detect=False)
        link._sabme_sent = True
        link._t401_at = link._t401_bits * 2
        link._on_frame(bytes((link.rsp_addr, v42_lapm.DISC | 0x10)))
        self.assertEqual(link.state, "idle")
        link._on_frame(bytes((link.cmd_addr, v42_lapm.DM | 0x10)))
        self.assertEqual(link.state, "release_pending")
        self.assertIn((bytes((link.rsp_addr, v42_lapm.DM | 0x10)), True), drain_frames(link))
        self.assertEqual(link.state, "released")

    def test_unsolicited_dm_f0_is_ignored_during_set_mode_collisions(self):
        originator = v42_lapm.V42Link(True, detect=False)
        originator._sabme_sent = True
        originator._on_frame(bytes((originator.cmd_addr, v42_lapm.DM)))
        self.assertEqual(originator.state, "idle")

        a, b = pair(detect=False)
        pump(a, b, 9600 * 3)
        self.assertTrue(a.release())
        a._on_frame(bytes((a.cmd_addr, v42_lapm.DM)))
        self.assertEqual(a.state, "releasing")

    def test_orderly_release_retries_disc_then_enters_disconnected(self):
        link, peer = pair(detect=False)
        pump(link, peer, 9600 * 3)
        self.assertTrue(link.release())
        drain_frames(link)

        # Each T401 timeout retransmits DISC. After N400 retries and the last
        # unanswered interval, the control function receives release.
        for attempt in range(v42_lapm.N400):
            link.bits_sent = link._t401_at - 1
            link.tx_bit()
            self.assertEqual(link.state, "releasing", attempt)
            self.assertGreater(len(link._tx), 0)
            drain_frames(link)
        link.bits_sent = link._t401_at - 1
        link.tx_bit()
        self.assertEqual(link.state, "released")
        self.assertEqual(link.stats["t401_expiries"], v42_lapm.N400 + 1)
        self.assertEqual(link.stats["disc_tx"], v42_lapm.N400 + 1)

    def test_release_snapshot_counts_retransmission_copy_once(self):
        link = v42_lapm.V42Link(True, detect=False)
        link._txq.extend(b"raw queued")
        link._post_break_txq.extend(b"post-break")
        link._cq.append(b"packed")
        link._unacked.append([7, b"frame in flight"])
        link._retx.append((7, b"frame in flight"))

        snapshot = link._snapshot_release_tx()
        self.assertEqual(snapshot["raw"], b"raw queuedpost-break")
        self.assertTrue(snapshot["encoded_pending"])
        self.assertEqual(snapshot["encoded_octets"], len(b"packed") + len(b"frame in flight"))

    def test_undefined_unnumbered_control_is_rejected(self):
        a, b = pair(detect=False)
        pump(a, b, 9600 * 3)
        a._on_frame(bytes((a.rsp_addr, 0x07)))
        self.assertEqual(a.state, "down")
        self.assertIn("undefined U control", a.events[-1])

    def test_malformed_xid_in_connected_state_is_rejected(self):
        a, b = pair(detect=False)
        pump(a, b, 9600 * 3)
        a._on_frame(bytes((a.rsp_addr, v42_lapm.XID, 0x82, 0x80, 0, 4, 5, 2, 0)))
        self.assertEqual(a.state, "down")
        self.assertIn("malformed XID", a.events[-1])

    def test_xid_with_pf_set_is_rejected_without_processing(self):
        # Table 8 encodes XID only with P/F=0. A set P/F bit must not mutate
        # negotiated parameters or satisfy a pending initial XID exchange.
        connected, peer = pair(detect=False)
        pump(connected, peer, 9600 * 3)
        before = (connected.tx_n401, connected.rx_n401,
                  connected.tx_k, connected.rx_k, connected._agreed)
        connected._on_frame(bytes((connected.rsp_addr,
                                   v42_lapm.XID | 0x10))
                            + v42_lapm.build_xid(64, 4, None))
        self.assertEqual(connected.state, "down")
        self.assertEqual((connected.tx_n401, connected.rx_n401,
                          connected.tx_k, connected.rx_k,
                          connected._agreed), before)
        self.assertIn("XID P/F bit set", connected.events[-1])

        caller = v42_lapm.V42Link(True, detect=False, frame_size=32, window=4)
        caller._send_xid_command()
        caller._on_frame(bytes((caller.cmd_addr,
                                v42_lapm.XID | 0x10, 0x82)))
        self.assertFalse(caller._xid_done)
        self.assertFalse(caller._sabme_sent)
        self.assertIn("XID P/F bit set while disconnected; discarded",
                      caller.events[-1])

        answerer = v42_lapm.V42Link(False, detect=False, frame_size=32, window=4)
        before = (answerer.tx_n401, answerer.rx_n401,
                  answerer.tx_k, answerer.rx_k)
        answerer._on_frame(bytes((answerer.rsp_addr,
                                  v42_lapm.XID | 0x10, 0x82)))
        self.assertEqual((answerer.tx_n401, answerer.rx_n401,
                          answerer.tx_k, answerer.rx_k), before)
        self.assertFalse(answerer._tx)
        self.assertIn("XID P/F bit set while disconnected; discarded",
                      answerer.events[-1])

    def test_unrecognized_xid_format_is_ignored_without_dropping_lapm(self):
        a, b = pair(detect=False)
        pump(a, b, 9600 * 3)
        a._tx.clear()
        a._on_frame(bytes((a.rsp_addr, v42_lapm.XID, 0x01, 0xFF)))
        self.assertEqual(a.state, "lapm")
        self.assertIn("unrecognized XID format 0x01 ignored", a.events[-1])
        self.assertFalse(a._tx)

        caller = v42_lapm.V42Link(True, detect=False)
        caller._send_xid_command()
        caller._on_frame(bytes((caller.cmd_addr, v42_lapm.XID, 0x01)))
        self.assertFalse(caller._xid_done)
        self.assertFalse(caller._sabme_sent)
        self.assertIn("unrecognized XID format 0x01 ignored", caller.events[-1])

    def test_answerer_sends_ten_adps_before_xid_response_even_if_flags_arrive_early(self):
        answerer = v42_lapm.V42Link(False, detect=True)
        for bit in v42_lapm.ODP_BLOCK:
            answerer.rx_bit(bit)
        self.assertFalse(answerer._answer_adp)  # only two alternating DC1s
        for bit in v42_lapm.ODP_BLOCK:
            answerer.rx_bit(bit)
        self.assertTrue(answerer._answer_adp)
        self.assertIn("ODP received", answerer.events)
        self.assertEqual(answerer._adp_count, 0)

        # A caller can begin flags before the mandatory ten ADP repetitions
        # have left the answerer. Detect protocol start but preserve the ADP
        # minimum before answering its first XID.
        for bit in v42_lapm.FLAG_BITS * 4:
            answerer.rx_bit(bit)
        self.assertEqual(answerer._phase, "protocol")
        self.assertLess(answerer._adp_count, v42_lapm.ADP_MIN_REPETITIONS)

        request = bytes((answerer.rsp_addr, v42_lapm.XID)) + v42_lapm.build_xid(
            128, 15, None
        )
        for bit in v42_lapm.FLAG_BITS + v42_lapm._stuffed(request) + v42_lapm.FLAG_BITS:
            answerer.rx_bit(bit)
        self.assertIsNotNone(answerer._deferred_xid_response)

        transmitted = [answerer.tx_bit() for _ in range(
            v42_lapm.ADP_MIN_REPETITIONS * len(v42_lapm.ADP_BLOCK)
        )]
        self.assertEqual(transmitted, v42_lapm.ADP_BLOCK * v42_lapm.ADP_MIN_REPETITIONS)
        self.assertEqual(answerer._adp_count, v42_lapm.ADP_MIN_REPETITIONS)
        first_response_bit = answerer.tx_bit()
        self.assertIsNone(answerer._deferred_xid_response)

        rx = v42_lapm._Hdlc()
        frames = rx.push(first_response_bit)
        for _ in range(599):
            frames.extend(rx.push(answerer.tx_bit()))
        self.assertTrue(any(ok and frame[0] == answerer.rsp_addr
                            and frame[1] & 0xEF == v42_lapm.XID
                            for frame, ok in frames))

    def test_unnumbered_information_above_n401_is_rejected(self):
        a, b = pair(detect=False)
        pump(a, b, 9600 * 3)
        info = b"x" * (a.n401 + 1)
        a._on_frame(bytes((a.rsp_addr, v42_lapm.XID)) + info)
        self.assertEqual(a.state, "down")
        self.assertIn("exceeds N401", a.events[-1])

    def test_nr_validity_handles_modulo_128_wrap(self):
        link = v42_lapm.V42Link(True, detect=False)
        link.va, link.vs = 126, 1
        self.assertTrue(link._valid_nr(126))
        self.assertTrue(link._valid_nr(127))
        self.assertTrue(link._valid_nr(0))
        self.assertTrue(link._valid_nr(1))
        self.assertFalse(link._valid_nr(2))

    def test_acknowledged_progress_tracks_actual_one_way_payload(self):
        a,b=pair(detect=False)
        data=bytes(range(256))*3
        a.write(data)
        self.assertEqual(a.stats['wire_acked'],0)
        ga,gb=pump(a,b,9600*3)
        self.assertEqual(ga,b'')
        self.assertEqual(gb,data)
        self.assertEqual(a.stats['wire_acked'],len(data))
        self.assertEqual(b.stats['wire_acked'],0)

    def test_detection_negotiation_and_data_both_ways(self):
        a, b = pair()
        a.write(b"hello from the originator\r\n")
        b.write(b"hello from the answerer\r\n")
        ga, gb = pump(a, b, 9600 * 4)
        self.assertEqual((a.state, b.state), ("lapm", "lapm"))
        self.assertEqual(gb, b"hello from the originator\r\n")
        self.assertEqual(ga, b"hello from the answerer\r\n")

    def test_direct_start_without_detection(self):
        a, b = pair(detect=False)
        a.write(b"x" * 500)
        _, gb = pump(a, b, 9600 * 3)
        self.assertEqual(gb, b"x" * 500)

    def test_originator_gives_up_on_a_silent_peer(self):
        a = v42_lapm.V42Link(True, detect=True, bit_rate=9600)
        for _ in range(9600 * 2):
            a.tx_bit()
            a.rx_bit(1)
        self.assertEqual(a.state, "unsupported")

    def test_bulk_data_and_frame_size_negotiation(self):
        rnd = random.Random(3)
        da = bytes(rnd.randrange(256) for _ in range(6000))
        db = bytes(rnd.randrange(256) for _ in range(5000))
        a, b = pair(frame_size=32, window=4)
        a.write(da)
        b.write(db)
        ga, gb = pump(a, b, 9600 * 30)
        self.assertEqual((gb, ga), (da, db))
        self.assertEqual((a.n401, a.k), (32, 4))

    def test_bit_errors_and_lost_bursts_are_recovered(self):
        rnd = random.Random(4)
        da = bytes(rnd.randrange(256) for _ in range(4000))
        db = bytes(rnd.randrange(256) for _ in range(4000))
        a, b = pair(frame_size=64)
        a.write(da)
        b.write(db)
        ga, gb = pump(a, b, 9600 * 60, ber=1e-4, burst=400)
        self.assertEqual(a.state, "lapm")
        self.assertEqual((gb, ga), (da, db))

    def test_v42bis_is_negotiated_and_shrinks_text(self):
        text = (b"the quick brown fox jumps over the lazy dog. " * 40) + b"\r\n"
        a, b = pair(compression=(3, 1024, 8))
        a.write(text)
        b.write(text[::-1])
        ga, gb = pump(a, b, 9600 * 6)
        self.assertEqual(a.compression, (3, 1024, 8))
        self.assertEqual((gb, ga), (text, text[::-1]))
        self.assertLess(a.compressed_out, len(text) // 3)

    def test_v42bis_reset_is_ordered_between_queued_dte_data(self):
        a, b = pair(compression=(1, 1024, 8))
        pump(a, b, 9600 * 3)
        before = b"compressible prefix; " * 160
        after = b"post-reset tail"
        a.write(before)
        self.assertTrue(a.request_v42bis_reset())
        a.write(after)
        self.assertEqual(a.state, "lapm")
        self.assertEqual(b.state, "lapm")
        self.assertFalse(b.request_v42bis_reset())  # this direction was not negotiated for transmit

        _, received = pump(a, b, 9600 * 12)
        self.assertEqual(received, before + after)
        self.assertEqual(a._enc.escape, 0)
        self.assertEqual(b._dec.escape, 0)
        self.assertFalse(a._enc.compressed)

    def test_compression_is_dropped_when_the_responder_does_not_offer_it(self):
        a = v42_lapm.V42Link(True, bit_rate=9600, compression=(3, 1024, 8))
        b = v42_lapm.V42Link(False, bit_rate=9600)
        a.write(b"plain text\r\n")
        _, gb = pump(a, b, 9600 * 3)
        self.assertIsNone(a.compression)
        self.assertEqual(gb, b"plain text\r\n")

    def test_xid_round_trip(self):
        info = v42_lapm.build_xid(64, 8, (3, 1024, 8))
        self.assertEqual(v42_lapm.parse_xid(info), {"functions": 0x898A,
                                                    "tx_n401": 64, "rx_n401": 64, "tx_k": 8, "rx_k": 8,
                                                    "p0": 3, "p1": 1024, "p2": 8})

    def test_xid_n401_uses_and_accepts_minimal_binary_width(self):
        # N401 is encoded in bits. 16 octets therefore fit in one octet
        # (128 bits); the default 128 octets still requires two.
        compact = v42_lapm.build_xid(16, 8, None)
        self.assertIn(bytes((5, 1, 128, 6, 1, 128)), compact)
        self.assertEqual(
            v42_lapm.parse_xid(compact),
            {"functions": 0x898A, "tx_n401": 16, "rx_n401": 16, "tx_k": 8, "rx_k": 8,
             "p0": 0, "p1": 1024, "p2": 32},
        )
        default = v42_lapm.build_xid(128, 15, None)
        self.assertIn(bytes((5, 2, 4, 0, 6, 2, 4, 0)), default)

    def test_xid_v42bis_parameter_boundaries(self):
        for p0 in (0, 3):
            for p1 in (512, 65535):
                for p2 in (6, 250):
                    with self.subTest(p0=p0, p1=p1, p2=p2):
                        parsed = v42_lapm.parse_xid(
                            v42_lapm.build_xid(128, 15, (p0, p1, p2))
                        )
                        self.assertEqual(
                            (parsed["p0"], parsed["p1"], parsed["p2"]),
                            (p0, p1, p2),
                        )
        for invalid in ((-1, 512, 6), (4, 512, 6),
                        (0, 511, 6), (0, 65536, 6),
                        (0, 512, 5), (0, 512, 251)):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    v42_lapm.build_xid(128, 15, invalid)

    def test_asymmetric_n401_and_window_are_negotiated_per_direction(self):
        caller = v42_lapm.V42Link(
            True, detect=False, bit_rate=9600, frame_size=32, window=4,
            receive_frame_size=64, receive_window=8,
        )
        answerer = v42_lapm.V42Link(
            False, detect=False, bit_rate=9600, frame_size=64, window=8,
            receive_frame_size=32, receive_window=4,
        )
        caller_payload = bytes(range(256)) * 3
        answerer_payload = bytes(reversed(range(256))) * 3
        caller.write(caller_payload)
        answerer.write(answerer_payload)
        got_caller, got_answerer = pump(caller, answerer, 9600 * 5)

        self.assertEqual((caller.state, answerer.state), ("lapm", "lapm"))
        self.assertEqual((caller.tx_n401, caller.rx_n401, caller.tx_k, caller.rx_k), (32, 64, 4, 8))
        self.assertEqual((answerer.tx_n401, answerer.rx_n401, answerer.tx_k, answerer.rx_k), (64, 32, 8, 4))
        self.assertEqual((got_caller, got_answerer), (answerer_payload, caller_payload))

    def test_connected_n401_window_renegotiation_uses_rnr_xid_then_rr(self):
        a, b = pair(detect=False)
        pump(a, b, 9600 * 3)
        self.assertEqual((a.state, b.state), ("lapm", "lapm"))

        self.assertTrue(a.renegotiate_parameters(
            tx_n401=20, rx_n401=24, tx_k=3, rx_k=5
        ))
        queued = b"queued during XID"
        a.write(queued)
        i_tx_before = a.stats["i_tx"]
        sent = [frame for frame, ok in drain_frames(a) if ok]
        self.assertGreaterEqual(len(sent), 2)
        self.assertEqual((sent[0][1], sent[0][2] & 1), (v42_lapm.RNR, 1))
        self.assertEqual(sent[1][1] & 0xEF, v42_lapm.XID)
        self.assertEqual(sent[1][1] & 0x10, 0)
        self.assertEqual(a.stats["i_tx"], i_tx_before)
        self.assertEqual(a.pending(), len(queued))

        # The responder sees the busy poll before XID and then chooses the
        # requested directional limits.
        for frame in sent:
            b._on_frame(frame)
        self.assertTrue(b._peer_busy)
        self.assertEqual((b.rx_n401, b.tx_n401, b.rx_k, b.tx_k),
                         (20, 24, 3, 5))
        replies = [frame for frame, ok in drain_frames(b) if ok]
        for frame in replies:
            a._on_frame(frame)
        self.assertEqual((a.tx_n401, a.rx_n401, a.tx_k, a.rx_k),
                         (20, 24, 3, 5))
        self.assertIsNone(a._renegotiation)
        self.assertFalse(a._local_busy)
        self.assertFalse(a._xid_pause_tx)

        # The post-XID RR clears peer busy; data queued during the pause flows.
        clear = [frame for frame, ok in drain_frames(a) if ok]
        for frame in clear:
            b._on_frame(frame)
        self.assertFalse(b._peer_busy)
        post = b"after connected XID"
        a.write(post)
        _, received = pump(a, b, 9600)
        self.assertEqual(received, queued + post)

    def test_connected_xid_poll_final_updates_peer_busy_without_touching_xid_timer(self):
        for peer_busy, expected_control in ((False, v42_lapm.RR),
                                            (True, v42_lapm.RNR)):
            with self.subTest(peer_busy=peer_busy):
                a, b = pair(detect=False)
                pump(a, b, 9600 * 3)
                b._local_busy = peer_busy
                self.assertTrue(a.renegotiate_parameters(
                    tx_n401=32, rx_n401=32, tx_k=4, rx_k=4
                ))
                queued = b"queued behind XID poll"
                a.write(queued)
                request = [frame for frame, ok in drain_frames(a) if ok]
                for frame in request:
                    b._on_frame(frame)
                replies = [frame for frame, ok in drain_frames(b) if ok]
                poll_final = next(
                    frame for frame in replies
                    if (frame[1] & 0xEF) in (v42_lapm.RR, v42_lapm.RNR)
                )
                self.assertEqual((poll_final[1] & 0xEF, poll_final[2] & 1),
                                 (expected_control, 1))

                xid_deadline = a._t401_at
                self.assertIsNotNone(xid_deadline)
                a._on_frame(poll_final)
                self.assertEqual(a.state, "lapm")
                self.assertEqual(a._peer_busy, peer_busy)
                self.assertFalse(a._xid_poll_out)
                self.assertIsNotNone(a._renegotiation)
                self.assertEqual(a._t401_at, xid_deadline)
                self.assertFalse(any("unsolicited supervisory F=1" in event
                                     for event in a.events))
                self.assertTrue(a._xid_pause_tx)

                # The XID response, not the poll final, completes the
                # parameter transaction and releases its independent timer.
                xid_response = next(
                    frame for frame in replies
                    if (frame[1] & 0xEF) == v42_lapm.XID
                )
                a._on_frame(xid_response)
                self.assertIsNone(a._renegotiation)
                self.assertIsNone(a._t401_at)
                self.assertEqual(a._peer_busy, peer_busy)
                self.assertFalse(a._xid_pause_tx)

                if peer_busy:
                    # Complete the initiator's RNR pause with its RR command.
                    # The queued I data must remain blocked while the peer
                    # still reports busy.
                    resume = [frame for frame, ok in drain_frames(a) if ok]
                    self.assertTrue(any((frame[1] & 0xEF) == v42_lapm.RR
                                        and not (frame[2] & 1)
                                        for frame in resume))
                    self.assertFalse(any(not (frame[1] & 1) for frame in resume))
                    for frame in resume:
                        b._on_frame(frame)
                    b.set_local_busy(False)
                _, received = pump(a, b, 9600 * 2)
                self.assertFalse(a._peer_busy)
                self.assertEqual(received, queued)

    def test_connected_xid_duplicate_recovers_lost_response_until_rr(self):
        a, b = pair(detect=False)
        pump(a, b, 9600 * 3)
        self.assertTrue(a.renegotiate_parameters(
            tx_n401=32, rx_n401=32, tx_k=4, rx_k=4
        ))
        request_frames = [frame for frame, ok in drain_frames(a) if ok]
        self.assertEqual([frame[1] & 0xEF for frame in request_frames[:2]],
                         [v42_lapm.RNR, v42_lapm.XID])

        for frame in request_frames:
            b._on_frame(frame)
        first_response = [frame for frame, ok in drain_frames(b) if ok]
        self.assertTrue(any((frame[1] & 0xEF) == v42_lapm.XID
                            for frame in first_response))
        # Lose that response and deliver the XID command retransmission. The
        # responder must remember the active L-SETPARM exchange even though
        # the one-shot RNR(P=1) indication was consumed by the first XID.
        b._on_frame(request_frames[1])
        retry_response = [frame for frame, ok in drain_frames(b) if ok]
        xid_replies = [frame for frame in retry_response
                       if (frame[1] & 0xEF) == v42_lapm.XID]
        self.assertEqual(len(xid_replies), 1)
        a._on_frame(xid_replies[0])
        self.assertEqual((a.tx_n401, a.rx_n401, a.tx_k, a.rx_k), (32, 32, 4, 4))
        self.assertIsNone(a._renegotiation)
        self.assertFalse(a._local_busy)
        clear = [frame for frame, ok in drain_frames(a) if ok]
        for frame in clear:
            b._on_frame(frame)
        self.assertFalse(b._peer_busy)
        self.assertFalse(b._peer_xid_pending)

    def test_connected_xid_renegotiation_timeout_clears_rnr_and_keeps_link(self):
        a, b = pair(detect=False)
        pump(a, b, 9600 * 3)
        old = (a.tx_n401, a.rx_n401, a.tx_k, a.rx_k)
        self.assertTrue(a.renegotiate_parameters(
            tx_n401=32, rx_n401=32, tx_k=4, rx_k=4
        ))
        frames = [frame for frame, ok in drain_frames(a) if ok]
        b._on_frame(frames[0])  # peer receives RNR(P=1)
        b._on_frame(frames[1])  # peer processes XID and queues its response
        replies = [frame for frame, ok in drain_frames(b) if ok]
        # Deliver only the RNR poll response; deliberately lose XID response.
        a._on_frame(replies[0])
        for _ in range(v42_lapm.N400 + 1):
            a._t401_expired()
        self.assertEqual(a.state, "lapm")
        self.assertEqual((a.tx_n401, a.rx_n401, a.tx_k, a.rx_k), old)
        self.assertIsNone(a._renegotiation)
        self.assertFalse(a._local_busy)
        self.assertFalse(a._xid_pause_tx)
        self.assertTrue(any("renegotiation failed" in event for event in a.events))
        clear = [frame for frame, ok in drain_frames(a) if ok]
        rr = [frame for frame in clear if (frame[1] & 0xEF) == v42_lapm.RR]
        self.assertTrue(rr)
        b._on_frame(rr[-1])
        self.assertFalse(b._peer_busy)

    def test_connected_xid_renegotiation_t401_runs_on_bit_clock(self):
        a, peer = pair(detect=False)
        pump(a, peer, 9600 * 3)
        self.assertEqual((a.state, peer.state), ("lapm", "lapm"))

        self.assertTrue(a.renegotiate_parameters(
            tx_n401=32, rx_n401=32, tx_k=4, rx_k=4
        ))
        # The peer sees the poll and XID, but its XID response is deliberately
        # lost.  Thereafter only tx_bit() advances the DUT's bit clock and
        # checks T401; the timeout handler is never invoked directly.
        outbound = [frame for frame, ok in drain_frames(a) if ok]
        for frame in outbound:
            peer._on_frame(frame)
        self.assertTrue(peer._peer_busy)
        replies = [frame for frame, ok in drain_frames(peer) if ok]
        self.assertTrue(any((frame[1] & 0xEF) == v42_lapm.XID for frame in replies))

        for expected_expiries in range(1, v42_lapm.N400 + 2):
            deadline = a._t401_at
            self.assertIsNotNone(deadline)
            # T401 cannot expire before its bit deadline.
            while a.bits_sent < deadline - 1:
                a.tx_bit()
            self.assertEqual(a.stats["t401_expiries"], expected_expiries - 1)
            # The bit scheduler services the timer at the next HDLC scheduling
            # boundary, at most one flag (8 bits) after the deadline.
            for _ in range(8):
                if a.stats["t401_expiries"] == expected_expiries:
                    break
                a.tx_bit()
            self.assertEqual(a.stats["t401_expiries"], expected_expiries)

        self.assertEqual(a.state, "lapm")
        self.assertIsNone(a._renegotiation)
        self.assertFalse(a._xid_pause_tx)
        self.assertFalse(a._local_busy)
        self.assertIsNone(a._t401_at)
        self.assertTrue(any("connected XID renegotiation failed" in event
                            for event in a.events))

    def test_connected_xid_renegotiation_requires_drained_window(self):
        a, b = pair(detect=False)
        pump(a, b, 9600 * 3)
        a.write(b"outstanding frame")
        pump(a, b, 80)
        self.assertTrue(a._unacked)
        self.assertFalse(a.renegotiate_parameters(
            tx_n401=32, rx_n401=32, tx_k=4, rx_k=4
        ))
        self.assertFalse(a._local_busy)
        self.assertIsNone(a._renegotiation)

    def test_initiator_accepts_responder_selection_above_its_n401_request(self):
        caller = v42_lapm.V42Link(True, detect=False, frame_size=32, window=4)
        caller._send_xid_command()
        response = bytes((caller.cmd_addr, v42_lapm.XID)) + v42_lapm.build_xid(128, 15, None)
        self.assertGreater(len(response) - 2, caller.n401)
        caller._on_frame(response)
        self.assertTrue(caller._xid_done)
        self.assertEqual((caller.n401, caller.k), (128, 15))
        self.assertTrue(any("XID response received" in event for event in caller.events))

    def test_initiator_honors_usable_below_request_selection_and_reports_violation(self):
        caller = v42_lapm.V42Link(
            True, detect=False, frame_size=32, window=4,
            receive_frame_size=64, receive_window=8,
        )
        caller._send_xid_command()
        # In the response, the responder's RX selections apply to our TX
        # direction; its TX selections apply to our RX direction.
        caller._xid_reply({"rx_n401": 16, "tx_n401": 32, "rx_k": 2, "tx_k": 4})
        self.assertEqual((caller.tx_n401, caller.rx_n401, caller.tx_k, caller.rx_k), (16, 32, 2, 4))
        self.assertTrue(any("below request 32" in event for event in caller.events))
        self.assertTrue(any("below request 4" in event for event in caller.events))

    def test_initiator_uses_supported_defaults_for_impossible_selection(self):
        caller = v42_lapm.V42Link(True, detect=False, frame_size=32, window=4)
        caller._send_xid_command()
        caller._xid_reply({"rx_n401": 129, "tx_n401": 0, "rx_k": 16, "tx_k": 0})
        self.assertEqual((caller.tx_n401, caller.rx_n401, caller.tx_k, caller.rx_k), (128, 128, 15, 15))
        self.assertEqual(sum("invalid XID selection" in event for event in caller.events), 4)

    def test_responder_selects_at_least_the_initiator_requested_n401(self):
        responder = v42_lapm.V42Link(False, detect=False, frame_size=32, window=4)
        request = v42_lapm.parse_xid(v42_lapm.build_xid(128, 15, None))
        responder._answer_xid(request)
        self.assertEqual((responder.n401, responder.k), (128, 15))

    def test_connected_xid_still_obeys_negotiated_n401(self):
        link = v42_lapm.V42Link(False, detect=False, frame_size=32, window=4)
        link._connected()
        xid = bytes((link.rsp_addr, v42_lapm.XID)) + v42_lapm.build_xid(128, 15, None)
        link._on_frame(xid)
        self.assertEqual(link.state, "down")
        self.assertIn("exceeds N401=32", link.events[-1])

    def test_xid_parser_rejects_truncated_group_and_parameter_lengths(self):
        self.assertIsNone(v42_lapm.parse_xid(bytes((0x82, 0x80, 0, 4, 5, 2, 0))))
        self.assertIsNone(v42_lapm.parse_xid(bytes((0x82, 0x80, 0, 2, 5, 2))))

    def test_xid_parser_rejects_duplicate_or_out_of_order_groups(self):
        core = bytes((0x80, 0, 3, 5, 1, 16))
        private = bytes((0xF0, 0, 5, 0, 3)) + b"V42"
        self.assertIsNotNone(v42_lapm.parse_xid(b"\x82" + core + private))
        self.assertIsNone(v42_lapm.parse_xid(b"\x82" + private + core))
        self.assertIsNone(v42_lapm.parse_xid(b"\x82" + core + core))

        # A malformed XID received after LAPM establishment follows the
        # existing mandatory frame-rejection path rather than negotiating
        # whichever duplicate happens to occur last.
        link = v42_lapm.V42Link(False, detect=False)
        link._connected()
        link._on_frame(bytes((link.rsp_addr, v42_lapm.XID)) + b"\x82" + core + core)
        self.assertEqual(link.state, "down")
        self.assertIn("malformed XID", link.events[-1])

    def test_xid_format_identifier_without_parameter_groups_uses_defaults(self):
        """V.42 permits the general-purpose FI with zero parameter groups."""
        self.assertEqual(v42_lapm.parse_xid(b"\x82"), {})

        caller = v42_lapm.V42Link(True, detect=False, frame_size=32, window=4)
        caller._send_xid_command()
        response = bytes((caller.cmd_addr, v42_lapm.XID, 0x82))
        caller._on_frame(response)

        self.assertTrue(caller._xid_done)
        self.assertEqual((caller.tx_n401, caller.rx_n401, caller.tx_k, caller.rx_k),
                         (128, 128, 15, 15))
        self.assertIsNone(caller._agreed)

        answerer = v42_lapm.V42Link(False, detect=False, frame_size=32, window=4)
        request = bytes((answerer.rsp_addr, v42_lapm.XID, 0x82))
        answerer._on_frame(request)
        self.assertEqual((answerer.tx_n401, answerer.rx_n401, answerer.tx_k, answerer.rx_k),
                         (128, 128, 15, 15))
        self.assertIsNone(answerer._agreed)
        self.assertTrue(any("XID command received" in event for event in answerer.events))
        self.assertTrue(answerer._tx)

    def test_xid_ignores_structurally_valid_user_data_subfield(self):
        # V.42 §12.2.1.3 defines GI=0xFF as a trailing, un-length-prefixed
        # sequence of PI/PL/PV records. V.44 uses it; this V.42-only endpoint
        # must ignore the parameters without rejecting LAPM negotiation.
        base = v42_lapm.build_xid(128, 15, None)
        user_data = bytes((0xFF, 64, 3)) + b"V44" + bytes((65, 1, 1))
        info = base + user_data
        expected = v42_lapm.parse_xid(base)
        self.assertEqual(v42_lapm.parse_xid(info), expected)

        answerer = v42_lapm.V42Link(False, detect=False)
        answerer._connected()
        # Connected negotiation is valid only after the peer has paused its
        # receive direction with RNR(P=1), as required by §8.10.1.
        answerer._on_frame(bytes((answerer.rsp_addr, v42_lapm.RNR, 1)))
        answerer._on_frame(bytes((answerer.rsp_addr, v42_lapm.XID)) + info)
        self.assertEqual(answerer.state, "lapm")
        self.assertTrue(answerer._tx)

    def test_connected_xid_requires_immediately_preceding_rnr_poll(self):
        info = v42_lapm.build_xid(
            128, 15, None, include_functions=False, include_compression=False
        )
        for interposed in (None, bytes((v42_lapm.RR, 0))):
            with self.subTest(interposed=interposed):
                answerer = v42_lapm.V42Link(False, detect=False)
                answerer._connected()
                # RNR without P=1 does not authorize connected XID.
                answerer._on_frame(bytes((answerer.rsp_addr, v42_lapm.RNR, 0)))
                # P=1 is required, and a subsequent peer supervisory command
                # consumes that authorization before the XID arrives.
                if interposed is not None:
                    answerer._on_frame(bytes((answerer.rsp_addr,)) + interposed)
                    answerer._on_frame(bytes((answerer.rsp_addr, v42_lapm.RNR, 1)))
                    answerer._on_frame(bytes((answerer.rsp_addr,)) + interposed)
                answerer._tx.clear()  # discard any earlier RNR(P=1) final response
                answerer._on_frame(bytes((answerer.rsp_addr, v42_lapm.XID)) + info)
                self.assertEqual(answerer.state, "lapm")
                self.assertFalse(answerer._tx)
                self.assertTrue(any("missing RNR(P=1)" in event for event in answerer.events))

    def test_xid_rejects_truncated_user_data_parameter(self):
        info = v42_lapm.build_xid(128, 15, None) + bytes((0xFF, 65, 2, 1))
        self.assertIsNone(v42_lapm.parse_xid(info))

    def test_v42bis_private_parameters_require_first_v42_set_identifier(self):
        compression_params = bytes((1, 1, 3, 2, 2, 4, 0, 3, 1, 8))
        valid_set = bytes((0, 3)) + b"V42" + compression_params
        wrong_set = bytes((0, 3)) + b"BAD" + compression_params
        missing_set = compression_params
        reordered_set = compression_params[:7] + bytes((0, 3)) + b"V42" + compression_params[7:]
        self.assertEqual(v42_lapm.parse_xid(b"\x82\xF0" + len(valid_set).to_bytes(2, "big") + valid_set),
                         {"p0": 3, "p1": 1024, "p2": 8})

        for private in (wrong_set, missing_set, reordered_set):
            with self.subTest(private=private):
                info = b"\x82\xF0" + len(private).to_bytes(2, "big") + private
                self.assertEqual(v42_lapm.parse_xid(info), {})
                answerer = v42_lapm.V42Link(
                    False, detect=False, compression=(3, 1024, 8)
                )
                answerer._on_frame(bytes((answerer.rsp_addr, v42_lapm.XID)) + info)
                self.assertIsNone(answerer._agreed)
                answerer._connected()
                self.assertIsNone(answerer.compression)

    def test_v42bis_p0_reserved_bits_do_not_negotiate_compression(self):
        private = (bytes((0, 3)) + b"V42" + bytes((1, 1, 0x83, 2, 2, 4, 0, 3, 1, 32)))
        info = b"\x82\xF0" + len(private).to_bytes(2, "big") + private
        fields = v42_lapm.parse_xid(info)
        self.assertNotIn("p0", fields)
        self.assertEqual((fields["p1"], fields["p2"]), (1024, 32))

        caller = v42_lapm.V42Link(True, detect=False, compression=(3, 1024, 32))
        caller._send_xid_command()
        caller._on_frame(bytes((caller.cmd_addr, v42_lapm.XID)) + info)
        self.assertIsNone(caller._agreed)
        caller._connected()
        self.assertIsNone(caller.compression)

        answerer = v42_lapm.V42Link(False, detect=False, compression=(3, 1024, 32))
        answerer._on_frame(bytes((answerer.rsp_addr, v42_lapm.XID)) + info)
        self.assertIsNone(answerer._agreed)
        answerer._connected()
        self.assertIsNone(answerer.compression)

    def test_v42bis_out_of_range_p1_p2_disconnects_initiator_and_responder(self):
        # V.42bis §5.1: P1/N2 < 512 and P2/N7 outside 6..250 are
        # procedural errors requiring disconnection. In particular, P2=251
        # must not become acceptable merely because our local offer is 32.
        for p1, p2 in ((511, 32), (1024, 5), (1024, 251)):
            with self.subTest(p1=p1, p2=p2):
                private = (bytes((0, 3)) + b"V42"
                           + bytes((1, 1, 3, 2, 2)) + p1.to_bytes(2, "big")
                           + bytes((3, 1, p2)))
                info = b"\x82\xF0" + len(private).to_bytes(2, "big") + private
                parsed = v42_lapm.parse_xid(info)
                self.assertIn("_v42bis_procedural_error", parsed)

                answerer = v42_lapm.V42Link(
                    False, detect=False, compression=(3, 1024, 32)
                )
                answerer._on_frame(
                    bytes((answerer.rsp_addr, v42_lapm.XID)) + info
                )
                self.assertEqual(answerer.state, "down")
                self.assertFalse(answerer._tx)
                self.assertIn("disconnecting", answerer.events[-1])

                caller = v42_lapm.V42Link(
                    True, detect=False, compression=(3, 1024, 32)
                )
                caller._send_xid_command()
                caller._tx.clear()
                caller._on_frame(
                    bytes((caller.cmd_addr, v42_lapm.XID)) + info
                )
                self.assertEqual(caller.state, "down")
                self.assertFalse(caller._sabme_sent)
                self.assertIn("disconnecting", caller.events[-1])

    def test_v42bis_p1_p2_legal_boundaries_negotiate_with_local_caps(self):
        for p1, p2, expected in ((512, 6, (3, 512, 6)),
                                 (65535, 250, (3, 1024, 32))):
            with self.subTest(p1=p1, p2=p2):
                private = (bytes((0, 3)) + b"V42"
                           + bytes((1, 1, 3, 2, 2)) + p1.to_bytes(2, "big")
                           + bytes((3, 1, p2)))
                info = b"\x82\xF0" + len(private).to_bytes(2, "big") + private
                parsed = v42_lapm.parse_xid(info)
                self.assertNotIn("_v42bis_procedural_error", parsed)

                answerer = v42_lapm.V42Link(
                    False, detect=False, compression=(3, 1024, 32)
                )
                answerer._on_frame(
                    bytes((answerer.rsp_addr, v42_lapm.XID)) + info
                )
                self.assertEqual(answerer.state, "idle")
                self.assertEqual(answerer._agreed, expected)
                self.assertTrue(answerer._tx)

    def test_xid_parser_treats_zero_length_n401_as_absent(self):
        # PL=0 means the parameter is absent and its default applies.
        info = bytes((0x82, 0x80, 0, 2, 5, 0))
        self.assertEqual(v42_lapm.parse_xid(info), {})

    def test_xid_parser_does_not_floor_fractional_octet_n401(self):
        # N401 is encoded in bits but specifies an integral octet count.
        info = bytes((0x82, 0x80, 0, 4, 5, 2, 0, 1))
        self.assertEqual(v42_lapm.parse_xid(info), {})

    def test_no_compression_is_explicit_in_xid(self):
        fields = v42_lapm.parse_xid(v42_lapm.build_xid(128, 15, None))
        self.assertEqual(fields["p0"], 0)
        self.assertEqual((fields["p1"], fields["p2"]), (1024, 32))
        a, b = pair()
        a.write(b"uncompressed request and response")
        b.write(b"uncompressed peer payload")
        ga, gb = pump(a, b, 9600 * 5)
        self.assertEqual(ga, b"uncompressed peer payload")
        self.assertEqual(gb, b"uncompressed request and response")
        for link in (a, b):
            self.assertIsNone(link.compression)
            self.assertIsNone(link._enc)
            self.assertIsNone(link._dec)

    def test_strict_peer_accepts_xid_with_pf_zero(self):
        class StrictXidPeer(v42_lapm.V42Link):
            def _on_u(self, frame, is_command):
                if frame[1]&0xef == v42_lapm.XID and frame[1]&0x10:
                    return
                super()._on_u(frame,is_command)
        a=v42_lapm.V42Link(True,detect=False)
        b=StrictXidPeer(False,detect=False)
        a.write(b'error control without compression')
        _, received=pump(a,b,9600*5)
        self.assertEqual(a.state, "lapm")
        self.assertEqual(a.stats["xid_tx"], 1)
        self.assertFalse(a._xid_fallback)
        self.assertEqual(received,b'error control without compression')
        self.assertIsNone(a.compression)

    def test_xid_command_and_response_pf_are_zero(self):
        a=v42_lapm.V42Link(True,detect=False)
        b=v42_lapm.V42Link(False,detect=False)
        for link,send in ((a,a._send_xid_command),(b,lambda:b._answer_xid({}))):
            frames=[]
            link._send=lambda frame:frames.append(frame)
            send()
            self.assertEqual(frames[0][1],v42_lapm.XID)

    def test_xid_optional_function_mask_sets_reserved_bits_only(self):
        # V.42 Table 11a requires reserved positions 2, 4, 8, 9, 12 and 16
        # to be set for interoperability. TEST (position 14) is offered only
        # by the production link; this low-level default builder deliberately
        # leaves optional procedures clear. Extended-FCS and SREJ remain off.
        info = v42_lapm.build_xid(128, 15, None)
        self.assertEqual(info[:2], bytes((0x82, 0x80)))
        group_len = int.from_bytes(info[2:4], "big")
        params = info[4:4 + group_len]
        self.assertEqual(params[:2], bytes((3, 4)))
        mask = int.from_bytes(params[2:6], "little")
        reserved = sum(1 << (position - 1)
                       for position in (2, 4, 8, 9, 12, 16))
        unsupported = (3, 14, 17, 24)
        self.assertEqual(mask, reserved)
        for position in unsupported:
            with self.subTest(position=position):
                self.assertEqual(mask & (1 << (position - 1)), 0)

        # A peer's request for every optional function may agree to TEST, but
        # must not cause agreement to unsupported procedures.
        peer_mask = reserved | sum(1 << (p - 1) for p in unsupported)
        peer_params = (bytes((3, 4)) + peer_mask.to_bytes(4, "little")
                       + params[6:])
        peer_info = (bytes((0x82, 0x80)) + len(peer_params).to_bytes(2, "big")
                     + peer_params)
        responder = v42_lapm.V42Link(False, detect=False)
        responder._on_frame(bytes((responder.rsp_addr, v42_lapm.XID)) + peer_info)
        responses = [frame for frame, ok in drain_frames(responder)
                     if ok and len(frame) > 1
                     and (frame[1] & 0xEF) == v42_lapm.XID]
        self.assertEqual(len(responses), 1)
        response_info = responses[0][2:]
        response_len = int.from_bytes(response_info[2:4], "big")
        response_params = response_info[4:4 + response_len]
        self.assertEqual(response_params[:2], bytes((3, 4)))
        response_mask = int.from_bytes(response_params[2:6], "little")
        self.assertEqual(response_mask, reserved | (1 << 13))

    def test_response_cannot_enable_unoffered_compression(self):
        caller=v42_lapm.V42Link(True,detect=False)
        caller._xid_reply({'p0':3,'p1':1024,'p2':32})
        caller._connected()
        self.assertIsNone(caller.compression)
        self.assertIsNone(caller._enc)
        self.assertIsNone(caller._dec)


# ---- the SpanDSP implementation as a reference peer (test only) ---------------------------------------------------
def _ref_class():
    try:
        import spandsp_v42
        return spandsp_v42.V42Link if spandsp_v42.available() else None
    except Exception:
        return None


@unittest.skipUnless(_ref_class(), "libspandsp (reference implementation) not available")
class InteropTests(unittest.TestCase):
    def test_v42bis_negotiated_directions_against_reference_in_both_roles(self):
        Ref = _ref_class()
        caller_data = b"caller direction payload; " * 80
        answerer_data = b"answerer direction payload; " * 80
        for ours_calls in (True, False):
            for p0 in (1, 2, 3):
                with self.subTest(ours_calls=ours_calls, p0=p0):
                    kw = dict(detect=True, bit_rate=9600, compression=(p0, 1024, 8))
                    if ours_calls:
                        caller = v42_lapm.V42Link(True, **kw)
                        answerer = Ref(False, **kw)
                    else:
                        caller = Ref(True, **kw)
                        answerer = v42_lapm.V42Link(False, **kw)
                    pump(caller, answerer, 9600 * 3)
                    caller.write(caller_data)
                    answerer.write(answerer_data)
                    received_caller, received_answerer = pump(caller, answerer, 9600 * 8)
                    self.assertEqual(received_answerer, caller_data)
                    self.assertEqual(received_caller, answerer_data)

    def test_our_v42bis_reset_over_lapm_to_reference_peer(self):
        Ref = _ref_class()
        a = v42_lapm.V42Link(True, detect=True, bit_rate=9600, compression=(1, 1024, 8))
        b = Ref(False, detect=True, bit_rate=9600, compression=(1, 1024, 8))
        pump(a, b, 9600 * 3)
        before = b"reference interop prefix; " * 160
        after = b"post-reset reference tail"
        a.write(before)
        self.assertTrue(a.request_v42bis_reset())
        a.write(after)
        _, received = pump(a, b, 9600 * 12)
        self.assertEqual(received, before + after)

    def run_case(self, ours_calls, detect, compression):
        Ref = _ref_class()
        kw = dict(bit_rate=9600)
        if ours_calls:
            a = v42_lapm.V42Link(True, detect=detect, compression=compression, **kw)
            b = Ref(False, detect=detect, compression=compression, **kw) if compression else Ref(False, detect=detect, **kw)
        else:
            a = Ref(True, detect=detect, compression=compression, **kw) if compression else Ref(True, detect=detect, **kw)
            b = v42_lapm.V42Link(False, detect=detect, compression=compression, **kw)
        text = b"interoperability text, repeated for compression. " * 20 + bytes(range(256))
        pump(a, b, 9600 * 3)                       # connect first: the reference binding sends uncompressed what it is
        a.write(text)                              # given before the link is up (v42_layer holds user data until then)
        b.write(text[::-1])
        ga, gb = pump(a, b, 9600 * 5)
        self.assertEqual(a.state, "lapm")
        self.assertEqual(b.state, "lapm")
        self.assertEqual((gb, ga), (text, text[::-1]))
        return a, b

    def test_our_originator_with_the_reference_answerer(self):
        for detect in (True, False):
            self.run_case(True, detect, None)

    def test_the_reference_originator_with_our_answerer(self):
        for detect in (True, False):
            self.run_case(False, detect, None)

    def test_our_connected_parameter_renegotiation_with_reference_answerer(self):
        Ref = _ref_class()
        caller = v42_lapm.V42Link(True, detect=True, bit_rate=9600)
        answerer = Ref(False, detect=True, bit_rate=9600)
        try:
            pump(caller, answerer, 9600 * 4)
            self.assertEqual((caller.state, answerer.state), ("lapm", "lapm"))
            self.assertEqual((caller.tx_n401, caller.rx_n401,
                              caller.tx_k, caller.rx_k), (128, 128, 15, 15))
            self.assertTrue(hasattr(answerer._lib, "v42_set_frame_params"))
            answerer._lib.v42_set_frame_params(answerer._state_ptr, 24, 3)

            self.assertTrue(caller.renegotiate_parameters(
                tx_n401=24, rx_n401=24, tx_k=3, rx_k=3
            ))
            payload = bytes((index * 73 + 11) & 0xFF for index in range(420))
            caller.write(payload)
            received_by_caller, received_by_answerer = pump(
                caller, answerer, 9600 * 8
            )

            self.assertEqual(caller.state, "lapm")
            self.assertEqual(answerer.state, "lapm")
            self.assertEqual((caller.tx_n401, caller.rx_n401,
                              caller.tx_k, caller.rx_k), (24, 24, 3, 3))
            self.assertEqual(received_by_caller, b"")
            self.assertEqual(received_by_answerer, payload)
            self.assertTrue(any("connected XID response received" in event
                                for event in caller.events))
        finally:
            answerer.close()

    def test_connected_xid_reference_retains_larger_parameters_and_payload(self):
        """A peer may answer connected XID with its current, larger selection."""
        Ref = _ref_class()
        caller = v42_lapm.V42Link(True, detect=True, bit_rate=9600)
        answerer = Ref(False, detect=True, bit_rate=9600)
        try:
            pump(caller, answerer, 9600 * 4)
            self.assertEqual((caller.state, answerer.state), ("lapm", "lapm"))
            self.assertEqual((caller.tx_n401, caller.rx_n401,
                              caller.tx_k, caller.rx_k), (128, 128, 15, 15))

            # The unmodified reference implementation answers with its
            # existing selection instead of adopting this smaller request.
            self.assertTrue(caller.renegotiate_parameters(
                tx_n401=24, rx_n401=32, tx_k=3, rx_k=4
            ))
            caller_payload = bytes(range(256)) * 2
            answerer_payload = bytes(reversed(range(256))) * 2
            caller.write(caller_payload)
            answerer.write(answerer_payload)
            received_by_caller, received_by_answerer = pump(
                caller, answerer, 9600 * 8
            )

            self.assertEqual(caller.state, "lapm")
            self.assertEqual(answerer.state, "lapm")
            self.assertEqual((caller.tx_n401, caller.rx_n401,
                              caller.tx_k, caller.rx_k), (128, 128, 15, 15))
            self.assertEqual((received_by_caller, received_by_answerer),
                             (answerer_payload, caller_payload))
            self.assertTrue(any(
                "connected XID parameters accepted: N401 tx/rx=128/128, k tx/rx=15/15"
                in event for event in caller.events
            ), caller.events[-12:])
        finally:
            answerer.close()

    def test_v42bis_across_implementations(self):
        for ours_calls in (True, False):
            a, b = self.run_case(ours_calls, True, (3, 1024, 8))
            self.assertEqual(a.compression, (3, 1024, 8))



class XidIgnoringAnswerer(v42_lapm.V42Link):
    """An answerer that does not implement XID negotiation: it never answers an XID command but accepts SABME (V.42 7.2.2 makes XID optional)."""

    def _answer_xid(self, x):
        self.ignored_xids = getattr(self, "ignored_xids", 0) + 1


class XidResponseLostAnswerer(v42_lapm.V42Link):
    """Process/select XID parameters but lose every response on the wire."""

    def _u(self, ctrl, command, pf=0, info=b""):
        if (ctrl & 0xEF) == v42_lapm.XID and not command:
            self.lost_xid_responses = getattr(self, "lost_xid_responses", 0) + 1
            return
        return super()._u(ctrl, command, pf, info)


class CompressingAnswerer(XidIgnoringAnswerer):
    """dialup.world (main-1): ignores XID, accepts SABME, then sends V.42bis-compressed data nobody negotiated."""

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self._bis = Encoder(1024, 32)

    def write(self, data):
        super().write(self._bis.compress(data))


class AssumedBisTests(unittest.TestCase):
    TEXT = b"Welcome to dialup.world!\r\nlogin: " + b"~\xff}#\xc0!}!}!} }4}\"}&} } } } }%}&\xd23\x8f\xc7}'}\"}(}\"\x8f\xd2~" * 12

    def run_pair(self, env):
        a = v42_lapm.V42Link(True, detect=True, bit_rate=9600)
        b = CompressingAnswerer(False, detect=True, bit_rate=9600)
        b.write(self.TEXT)
        old = dict(os.environ)
        os.environ.update(env)
        try:
            ga, _ = pump(a, b, 9600 * 12)
        finally:
            os.environ.clear()
            os.environ.update(old)
        return a, ga

    def test_default_delivers_the_compressed_octets_unchanged(self):
        a, ga = self.run_pair({})
        self.assertEqual(a.state, "lapm")
        self.assertNotEqual(ga, self.TEXT)
        self.assertFalse(a._bis_assumed)

    def test_opt_in_decodes_the_unnegotiated_compressed_stream(self):
        a, ga = self.run_pair({"V42_ASSUME_BIS": "1"})
        # the decoder holds its last pending string until more data arrives, so compare all but the tail
        self.assertGreater(len(ga), len(self.TEXT) - 16)
        self.assertTrue(self.TEXT.startswith(ga))
        self.assertIsNone(a.compression)                       # still nothing negotiated
        self.assertTrue(any("V.42bis assumed" in e for e in a.events))

    def test_opt_in_is_ignored_when_the_xid_was_answered(self):
        a = v42_lapm.V42Link(True, detect=True, bit_rate=9600)
        b = v42_lapm.V42Link(False, detect=True, bit_rate=9600)
        b.write(b"plain text with a \x00\x00 inside\r\n")
        old = dict(os.environ)
        os.environ["V42_ASSUME_BIS"] = "1"
        try:
            ga, _ = pump(a, b, 9600 * 8)
        finally:
            os.environ.clear()
            os.environ.update(old)
        self.assertEqual(ga, b"plain text with a \x00\x00 inside\r\n")
        self.assertFalse(a._bis_assumed)

    def test_a_wrong_assumption_falls_back_to_raw_delivery_without_dropping_the_link(self):
        a = v42_lapm.V42Link(True, detect=True, bit_rate=9600)
        b = XidIgnoringAnswerer(False, detect=True, bit_rate=9600)
        b.write(bytes([0x00, 0x05, 0x00, 0x00, 0xFF, 0x81]) * 20)   # not V.42bis
        old = dict(os.environ)
        os.environ["V42_ASSUME_BIS"] = "1"
        try:
            pump(a, b, 9600 * 8)
        finally:
            os.environ.clear()
            os.environ.update(old)
        self.assertEqual(a.state, "lapm")


class XidFallbackTests(unittest.TestCase):
    """dialup.world (2026-10-07): ADP seen, our XID unanswered for the rest of the call, SABME never sent."""

    def run_pair(self, answerer_class, seconds=10, env=None):
        env = env or {}
        fallback = env.get("V42_XID_FALLBACK", "1") != "0"
        a = v42_lapm.V42Link(True, detect=True, bit_rate=9600,
                             xid_fallback=fallback)
        b = answerer_class(False, detect=True, bit_rate=9600)
        a.write(b"hello from the originator\r\n")
        old = dict(os.environ)
        os.environ.update(env)
        try:
            _, gb = pump(a, b, 9600 * seconds)
        finally:
            os.environ.clear()
            os.environ.update(old)
        return a, b, gb

    def test_unanswered_xid_falls_back_to_sabme_and_the_link_comes_up(self):
        a, b, gb = self.run_pair(XidIgnoringAnswerer)
        self.assertEqual((a.state, b.state), ("lapm", "lapm"))
        self.assertEqual(gb, b"hello from the originator\r\n")
        self.assertGreaterEqual(b.ignored_xids, v42_lapm.XID_ATTEMPTS)
        self.assertEqual(a.stats["xid_tx"], v42_lapm.XID_ATTEMPTS)
        self.assertTrue(any("sending SABME with default parameters" in e for e in a.events))
        self.assertIsNone(a.compression)                       # nothing was negotiated

    def test_unanswered_xid_with_nondefault_frame_request_fails_closed(self):
        caller_data = bytes(range(256)) * 3
        answer_data = b"answer" * 90
        a = v42_lapm.V42Link(True, detect=True, bit_rate=1200,
                             frame_size=32, window=4)
        b = XidIgnoringAnswerer(False, detect=True, bit_rate=1200)
        a.write(caller_data)
        b.write(answer_data)
        ga, gb = pump(a, b, 1200 * 40)
        self.assertEqual(a.state, "down")
        self.assertNotEqual(b.state, "lapm")
        self.assertEqual((ga, gb), (b"", b""))
        self.assertTrue(any("fallback suppressed" in e for e in a.events))

    def test_the_old_behaviour_is_a_deadlock_and_stays_available(self):
        a, b, gb = self.run_pair(XidIgnoringAnswerer, env={"V42_XID_FALLBACK": "0"})
        self.assertEqual(a.state, "idle")
        self.assertEqual(gb, b"")
        self.assertGreater(a.stats["xid_tx"], v42_lapm.XID_ATTEMPTS)  # still repeating the XID
        self.assertFalse(a._sabme_sent)

    def test_fallback_is_suppressed_if_peer_processed_nondefault_xid_and_reply_was_lost(self):
        # The responder selects N401=32, but all replies are lost. Proceeding
        # with default N401=128 would desynchronize the endpoints and corrupt
        # application data; the caller must exhaust XID and fail closed.
        with patch.dict(os.environ, {"V42_XID_FALLBACK": "1"}):
            a = v42_lapm.V42Link(True, detect=True, bit_rate=9600,
                                 frame_size=32, window=4)
            b = XidResponseLostAnswerer(False, detect=True, bit_rate=9600,
                                        frame_size=32, window=4)
            caller_payload = bytes(range(256))
            answer_payload = b"answer" * 40
            a.write(caller_payload)
            b.write(answer_payload)
            ga, gb = pump(a, b, 9600 * 30, seed=922)

        self.assertGreaterEqual(a.stats["xid_tx"], v42_lapm.N400 + 1)
        self.assertGreaterEqual(b.lost_xid_responses, v42_lapm.XID_ATTEMPTS)
        self.assertEqual(a.state, "down")
        self.assertNotEqual(b.state, "lapm")
        self.assertFalse(a._sabme_sent)
        self.assertEqual((ga, gb), (b"", b""))
        self.assertTrue(any("fallback suppressed" in event for event in a.events))

    def test_fallback_is_suppressed_when_v42bis_was_offered(self):
        # Even default N401/k cannot make an incomplete XID safe if the peer
        # may have activated V.42bis while its response was lost.
        with patch.dict(os.environ, {"V42_XID_FALLBACK": "1"}):
            a = v42_lapm.V42Link(True, detect=True, bit_rate=9600,
                                 compression=(3, 1024, 32))
            b = XidResponseLostAnswerer(False, detect=True, bit_rate=9600,
                                        compression=(3, 1024, 32))
            a.write(b"caller payload " * 80)
            b.write(b"answer payload " * 80)
            ga, gb = pump(a, b, 9600 * 30, seed=923)

        self.assertGreaterEqual(a.stats["xid_tx"], v42_lapm.N400 + 1)
        self.assertGreaterEqual(b.lost_xid_responses, v42_lapm.XID_ATTEMPTS)
        self.assertEqual(a.state, "down")
        self.assertNotEqual(b.state, "lapm")
        self.assertFalse(a._sabme_sent)
        self.assertEqual((ga, gb), (b"", b""))
        self.assertTrue(any("fallback suppressed" in event for event in a.events))
        self.assertEqual(gb, b"")

    def test_strict_xid_policy_does_not_establish_after_processed_response_is_lost(self):
        # V.42 8.10.3 bounds XID retries; Appendix III.3 warns that a peer may
        # have accepted the request even if every XID response is lost. Do not
        # proceed into a potentially mismatched LAPM connection.
        with patch.dict(os.environ, {"V42_XID_FALLBACK": "0"}):
            a = v42_lapm.V42Link(True, detect=True, bit_rate=9600,
                                 frame_size=32, window=4)
            b = XidResponseLostAnswerer(False, detect=True, bit_rate=9600,
                                        frame_size=32, window=4)
            a.write(bytes(range(256)) * 2)
            b.write(b"answer" * 80)
            ga, gb = pump(a, b, 9600 * 20, seed=921)

        self.assertGreaterEqual(b.lost_xid_responses, v42_lapm.N400)
        self.assertEqual(a.stats["xid_tx"], v42_lapm.N400 + 1)
        self.assertEqual(a.state, "down")
        self.assertNotEqual(b.state, "lapm")
        self.assertFalse(a._sabme_sent)
        self.assertEqual((ga, gb), (b"", b""))

    def test_explicit_link_policy_overrides_process_environment(self):
        with patch.dict(os.environ, {"V42_XID_FALLBACK": "1"}):
            a = v42_lapm.V42Link(True, detect=True, bit_rate=9600,
                                 xid_fallback=False)
        self.assertFalse(a.xid_fallback)

    def test_a_peer_that_answers_xid_never_triggers_the_fallback(self):
        a, b, gb = self.run_pair(v42_lapm.V42Link)
        self.assertEqual((a.state, b.state), ("lapm", "lapm"))
        self.assertEqual(a.stats["xid_tx"], 1)
        self.assertFalse(any("default parameters" in e for e in a.events))

    def test_a_slow_xid_response_inside_the_window_is_still_used(self):
        class SlowAnswerer(v42_lapm.V42Link):
            """Answers the XID only to the second command (first one 'lost')."""
            def _answer_xid(self, x):
                self.seen = getattr(self, "seen", 0) + 1
                if self.seen >= 2:
                    super()._answer_xid(x)
        a, b, gb = self.run_pair(SlowAnswerer)
        self.assertEqual((a.state, b.state), ("lapm", "lapm"))
        self.assertEqual(a.stats["xid_tx"], 2)
        self.assertFalse(any("default parameters" in e for e in a.events))

class TailLossWithAChattyPeerTests(unittest.TestCase):
    """A sender whose whole window is lost must still recover while the peer keeps sending I frames with an unchanged N(R).

    Before the fix an acknowledgement that covered nothing new restarted T401 on every such frame, so the timer never expired,
    no poll or retransmission happened, and the lost frames were never delivered while the peer kept talking (found in the
    V.34 retrain loopback, lab/V34_FREEZE_GUARD.md)."""

    def run_link(self, bits=380000):
        a, b = pair()
        got_b = bytearray()
        payload = bytes((i * 7 + 3) % 251 for i in range(6000))
        sent, black_from, chat = False, None, 0
        for i in range(bits):
            ba, bb = a.tx_bit(), b.tx_bit()
            if not sent and a.state == "lapm" and b.state == "lapm" and i > 40000:
                a.write(payload)
                sent, black_from = True, i + 3000
            if black_from is not None and black_from <= i < black_from + 40000:
                ba = 1                                                    # a -> b blacked out: a's whole window is lost
            a.rx_bit(bb)
            b.rx_bit(ba)
            got_b += b.read()
            a.read()
            if sent and i % 2000 == 0:
                b.write(b"chat%05d\r\n" % chat)                          # b keeps sending with an unchanged N(R)
                chat += 1
        return a, bytes(got_b), payload

    def test_lost_window_is_recovered_while_the_peer_keeps_sending(self):
        a, got, payload = self.run_link()
        self.assertEqual(a.state, "lapm")
        self.assertEqual(got, payload)
        self.assertGreaterEqual(a.stats["t401_expiries"], 1)
        self.assertGreater(a.stats["i_resent"], 0)


if __name__ == "__main__":
    unittest.main()
