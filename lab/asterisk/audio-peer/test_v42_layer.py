#!/usr/bin/env python3
"""V42Layer: two engines wired bit-for-bit, one calling and one answering."""
import unittest
import time
from collections import deque
from unittest.mock import patch

import v42_lapm
from v42_detect import ADP_ARM_BITS, adp_bits
from v42_layer import V42Layer, _Framer
from v42bis import N6


class BitEngine(V42Layer):
    def __init__(self, rate=9600):
        self.bit_rate = rate
        self._v42_setup()
        self.q = deque()
        self.fr = _Framer()
        self.out = bytearray()

    def _plain_tx_bit(self):
        return self.q.popleft() if self.q else 1

    def _plain_feed_bytes(self, data):
        for v in data:
            self.q.extend([0] + [(v >> i) & 1 for i in range(8)] + [1])

    def _plain_feed_marks(self, n):
        self.q.extend([1] * n)

    def _plain_rx_bits(self, bits):
        return self.fr.push(list(bits))


def pump(a, b, bits):
    for _ in range(bits):
        ba, bb = a.v42_tx_bit(), b.v42_tx_bit()
        a.out += a.v42_rx_bits([bb])
        b.out += b.v42_rx_bits([ba])
        a.out += a.v42_read()
        b.out += b.v42_read()


class V42LayerTests(unittest.TestCase):
    def test_opt_in_adp_trace_records_exact_async_bytes_and_mark_gaps(self):
        with patch.dict("os.environ", {"V42_CAPTURE_DETECTOR": "1"}):
            caller = BitEngine()
            caller.start_v42(True, rate=9600)
            caller.v42_tx_bits(ADP_ARM_BITS + 10)
            caller.v42_rx_bits(adp_bits(repeats=2))

        self.assertEqual(caller._v42_adp_detector.result, "v42")
        trace = caller.v42_adp_trace()
        self.assertEqual(bytes.fromhex(trace["bytes_hex"]), b"ECEC")
        self.assertEqual(trace["mark_gaps"], [None, 8, 8, 8])
        self.assertFalse(trace["truncated"])
        self.assertEqual(trace["byte_limit"], V42Layer.ADP_TRACE_LIMIT)

    def test_adp_trace_is_off_by_default(self):
        caller = BitEngine()
        caller.start_v42(True, rate=9600)
        self.assertIsNone(caller.v42_adp_trace())

    def test_response_role_i_frame_marks_layer_lost_without_payload_delivery(self):
        caller, answerer = BitEngine(), BitEngine()
        caller.start_v42(True, rate=9600)
        answerer.start_v42(False, rate=9600)
        pump(caller, answerer, 9600 * 4)
        self.assertEqual((caller.v42_mode, answerer.v42_mode), ("lapm", "lapm"))

        # A valid-FCS response-address I frame is not a legal V.42 frame type.
        # Exercise the full V42Layer receive/check path, not just V42Link.
        frame = bytes((answerer._v42.cmd_addr, 0, 0)) + b"must not reach DTE"
        wire = (v42_lapm.FLAG_BITS + v42_lapm._stuffed(frame)
                + v42_lapm.FLAG_BITS)
        answerer.v42_rx_bits(wire)

        self.assertTrue(answerer.v42_lost)
        self.assertEqual(answerer.v42_mode, "lost")
        self.assertEqual(answerer.v42_read(), b"")
        self.assertEqual(answerer._v42._out, b"")

    def test_test_loopback_primitive_is_exposed_through_control_function_layer(self):
        caller, answerer = BitEngine(), BitEngine()
        caller.start_v42(True, rate=9600)
        answerer.start_v42(False, rate=9600)
        pump(caller, answerer, 9600 * 4)
        self.assertEqual((caller.v42_mode, answerer.v42_mode), ("lapm", "lapm"))

        payload = bytes(range(96))
        self.assertTrue(caller.v42_request_test(payload))
        pump(caller, answerer, 9600)
        self.assertEqual(answerer.v42_read_test_indication(), payload)
        self.assertIsNone(answerer.v42_read_test_indication())
        # V.42's control function chooses to return the test information; the
        # error-control layer exposes the indication rather than inventing a
        # response policy itself.
        self.assertTrue(answerer.v42_request_test(payload))
        pump(caller, answerer, 9600)
        self.assertEqual(caller.v42_read_test_indication(), payload)
        self.assertEqual((caller.v42_mode, answerer.v42_mode), ("lapm", "lapm"))

    def test_v42bis_reset_is_exposed_through_layer_at_dte_boundary(self):
        caller, answerer = BitEngine(), BitEngine()
        caller.start_v42(True, rate=9600, compression=(1, 1024, 8))
        answerer.start_v42(False, rate=9600, compression=(1, 1024, 8))
        pump(caller, answerer, 9600 * 4)
        self.assertEqual((caller.v42_mode, answerer.v42_mode), ("lapm", "lapm"))

        before = b"layer reset prefix; " * 120
        after = b"layer reset suffix"
        caller.v42_feed_bytes(before)
        self.assertTrue(caller.v42_request_compression_reset())
        self.assertFalse(answerer.v42_request_compression_reset())
        caller.v42_feed_bytes(after)
        pump(caller, answerer, 9600 * 10)
        self.assertEqual(bytes(answerer.out), before + after)

    def test_v42bis_c_error_marks_layer_lost_instead_of_delivering_corrupt_data(self):
        caller, answerer = BitEngine(), BitEngine()
        caller.start_v42(True, rate=9600, compression=(1, 1024, 8))
        answerer.start_v42(False, rate=9600, compression=(1, 1024, 8))
        pump(caller, answerer, 9600 * 4)
        self.assertEqual((caller.v42_mode, answerer.v42_mode), ("lapm", "lapm"))

        answerer._v42._dec.compressed = True
        invalid_codeword = answerer._v42._dec.d.c1.to_bytes(2, "little")
        answerer._v42._on_i(bytes((answerer._v42.cmd_addr, 0, 0)) + invalid_codeword)
        answerer._v42_check()
        self.assertEqual(answerer._v42.state, "lapm")
        self.assertEqual(answerer.v42_mode, "lost")
        self.assertTrue(answerer.v42_lost)
        self.assertEqual(answerer.v42_read(), b"")

    def test_c_error_rejects_following_i_frame_in_same_rx_batch(self):
        caller, answerer = BitEngine(), BitEngine()
        caller.start_v42(True, rate=9600, compression=(3, 1024, 8),
                         compression_recovery=True)
        answerer.start_v42(False, rate=9600, compression=(3, 1024, 8),
                           compression_recovery=True)
        pump(caller, answerer, 9600 * 4)
        self.assertEqual((caller.v42_mode, answerer.v42_mode), ("lapm", "lapm"))

        decoder = answerer._v42._dec
        decoder.compressed = True
        invalid = decoder.d.c1.to_bytes(2, "little")
        valid_after_error = (N6 + ord("X")).to_bytes(2, "little")
        tx = caller._v42
        # The C-ERROR frame also polls us, so it exercises the mandatory
        # immediate RNR(F=1) response as well as the corrupted dictionary.
        tx._send(bytes((tx.cmd_addr, 0, 1)) + invalid)
        tx._send(bytes((tx.cmd_addr, 2, 0)) + valid_after_error)
        wire = list(tx._tx)
        tx._tx.clear()

        # Both valid-FCS I-frames arrive within one RTP/audio receive batch.
        answerer._v42.rx_bits(wire)
        answerer._v42_check()
        self.assertEqual(answerer.v42_mode, "recovering")
        self.assertTrue(answerer._v42._local_busy)
        self.assertEqual(answerer._v42.vr, 1)
        self.assertEqual(answerer._v42._out, b"")
        self.assertEqual(answerer._v42_recovery_stats["c_error_input_octets"],
                         len(invalid))
        parser = v42_lapm._Hdlc()
        responses = []
        for bit in list(answerer._v42._tx) + v42_lapm.FLAG_BITS:
            responses.extend(parser.push(bit))
        self.assertTrue(any(
            valid and (frame[1] & 0xEF) == v42_lapm.RNR and frame[2] & 1
            for frame, valid in responses
        ))

    def test_v42bis_c_error_reestablishes_lapm_and_resets_both_dictionaries(self):
        caller, answerer = BitEngine(), BitEngine()
        caller.start_v42(True, rate=9600, compression=(3, 1024, 8),
                         compression_recovery=True)
        answerer.start_v42(False, rate=9600, compression=(3, 1024, 8),
                           compression_recovery=True)
        pump(caller, answerer, 9600 * 4)
        self.assertEqual((caller.v42_mode, answerer.v42_mode), ("lapm", "lapm"))

        old_decoder = answerer._v42._dec
        answerer._v42._dec.compressed = True
        invalid_codeword = old_decoder.d.c1.to_bytes(2, "little")
        answerer._v42._on_i(bytes((answerer._v42.cmd_addr, 0, 0)) + invalid_codeword)
        answerer._v42_check()
        self.assertEqual(answerer.v42_mode, "recovering")
        held_outbound = b"queued while LAPM dictionaries are reset" * 4
        answerer.v42_feed_bytes(held_outbound)

        pump(caller, answerer, 9600 * 8)
        self.assertEqual((caller.v42_mode, answerer.v42_mode), ("lapm", "lapm"))
        self.assertIsNot(answerer._v42._dec, old_decoder)
        self.assertIsNotNone(caller._v42._enc)
        self.assertIsNotNone(answerer._v42._dec)

        suffix = b"payload after synchronized compression recovery" * 5
        caller.v42_feed_bytes(suffix)
        pump(caller, answerer, 9600 * 3)
        self.assertEqual(bytes(answerer.out), suffix)
        self.assertEqual(bytes(caller.out), held_outbound)

    def test_simultaneous_v42bis_c_error_reestablishes_both_directions(self):
        caller, answerer = BitEngine(), BitEngine()
        caller.start_v42(True, rate=9600, compression=(3, 1024, 8),
                         compression_recovery=True)
        answerer.start_v42(False, rate=9600, compression=(3, 1024, 8),
                           compression_recovery=True)
        pump(caller, answerer, 9600 * 4)
        self.assertEqual((caller.v42_mode, answerer.v42_mode), ("lapm", "lapm"))

        # Queue both reverse directions before C-ERROR. Each endpoint's RNR
        # then blocks the other's drain, the bilateral case that used to
        # leave both layers stuck until the overall recovery deadline.
        caller.v42_feed_bytes(bytes(range(256)) * 8)
        answerer.v42_feed_bytes(bytes(reversed(range(256))) * 8)
        old_links = (caller._v42, answerer._v42)

        # Both decoders lose compression synchronization before either control
        # function completes release. This exercises the simultaneous-DISC
        # collision and ensures only the caller starts the fresh SABME exchange.
        for engine in (caller, answerer):
            decoder = engine._v42._dec
            decoder.compressed = True
            invalid = decoder.d.c1.to_bytes(2, "little")
            link = engine._v42
            link._on_i(bytes((link.cmd_addr, 0, 0)) + invalid)
            engine._v42_check()
            self.assertEqual(engine.v42_mode, "recovering")

        pump(caller, answerer, 9600 * 20)
        self.assertEqual((caller.v42_mode, answerer.v42_mode), ("lapm", "lapm"))
        self.assertEqual(caller._v42_recovery_stats["completed"], 1)
        self.assertEqual(answerer._v42_recovery_stats["completed"], 1)
        self.assertTrue(any("peer remained busy for T401" in event
                            for link in old_links for event in link.events))

        caller_payload = b"caller after simultaneous compression error" * 3
        answerer_payload = b"answerer after simultaneous compression error" * 3
        caller_start, answerer_start = len(caller.out), len(answerer.out)
        caller.v42_feed_bytes(caller_payload)
        answerer.v42_feed_bytes(answerer_payload)
        pump(caller, answerer, 9600 * 4)
        self.assertEqual(bytes(answerer.out[answerer_start:]), caller_payload)
        self.assertEqual(bytes(caller.out[caller_start:]), answerer_payload)

    def test_c_error_peer_disc_during_reverse_drain_restarts_instead_of_failing(self):
        caller, answerer = BitEngine(), BitEngine()
        caller.start_v42(True, rate=9600, compression=(3, 1024, 8),
                         compression_recovery=True)
        answerer.start_v42(False, rate=9600, compression=(3, 1024, 8),
                           compression_recovery=True)
        pump(caller, answerer, 9600 * 4)
        self.assertEqual((caller.v42_mode, answerer.v42_mode), ("lapm", "lapm"))

        # The caller has reverse data to drain; the answerer does not. As its
        # DISC reaches the still-draining caller, that early peer release must
        # advance recovery instead of being mistaken for link failure.
        caller.v42_feed_bytes(bytes(range(256)) * 8)
        for engine in (caller, answerer):
            decoder = engine._v42._dec
            decoder.compressed = True
            link = engine._v42
            link._on_i(bytes((link.cmd_addr, 0, 0))
                       + decoder.d.c1.to_bytes(2, "little"))
            engine._v42_check()

        pump(caller, answerer, 9600 * 8)
        self.assertEqual((caller.v42_mode, answerer.v42_mode), ("lapm", "lapm"))
        self.assertEqual((caller._v42_recovery_stats["completed"],
                          answerer._v42_recovery_stats["completed"]), (1, 1))
        self.assertFalse(caller.v42_lost or answerer.v42_lost)

        caller_payload = b"after early peer DISC" * 4
        answerer_payload = b"after early peer DISC in reverse" * 4
        caller_start, answerer_start = len(caller.out), len(answerer.out)
        caller.v42_feed_bytes(caller_payload)
        answerer.v42_feed_bytes(answerer_payload)
        pump(caller, answerer, 9600 * 4)
        self.assertEqual(bytes(answerer.out[answerer_start:]), caller_payload)
        self.assertEqual(bytes(caller.out[caller_start:]), answerer_payload)

    def test_c_error_recovery_preserves_queued_data_on_both_sides(self):
        caller, answerer = BitEngine(), BitEngine()
        caller.start_v42(True, rate=9600, compression=(3, 1024, 8),
                         compression_recovery=True)
        answerer.start_v42(False, rate=9600, compression=(3, 1024, 8),
                           compression_recovery=True)
        pump(caller, answerer, 9600 * 4)
        self.assertEqual((caller.v42_mode, answerer.v42_mode), ("lapm", "lapm"))

        # Queue the detecting side's reverse-direction payload before C-ERROR;
        # its recovery procedure must drain this direction before DISC.
        answerer_pending = b"answerer data that must drain before DISC" * 4
        answerer.v42_feed_bytes(answerer_pending)

        decoder = answerer._v42._dec
        decoder.compressed = True
        invalid = decoder.d.c1.to_bytes(2, "little")
        tx = caller._v42
        tx._queue_i(invalid)
        tx._send_i(*tx._unacked[-1])
        for _ in range(9600):
            answerer.v42_rx_bits([caller.v42_tx_bit()])
            answerer.out += answerer.v42_read()
            if answerer._v42_recovery_stage == "drain":
                break
        self.assertEqual(answerer._v42_recovery_stage, "drain")

        # Deliver the busy indication before the caller accepts the queued
        # payload, so it is still raw DTE data rather than a sent I-frame.
        for _ in range(9600):
            caller_bit = caller.v42_tx_bit()
            answerer_bit = answerer.v42_tx_bit()
            caller.v42_rx_bits([answerer_bit])
            answerer.v42_rx_bits([caller_bit])
            caller.out += caller.v42_read()
            answerer.out += answerer.v42_read()
            if caller._v42._peer_busy:
                break
        self.assertTrue(caller._v42._peer_busy)

        # This DTE data arrives on the peer after RNR and before DISC. It is in
        # the old link's unsent DTE queue when the peer accepts orderly release.
        caller_pending = b"caller data held by peer RNR" * 4
        held_after_error = b"answerer data held after C-ERROR" * 4
        caller.v42_feed_bytes(caller_pending)
        caller._v42._cq.append(b"\x01\x02")  # an already-compressed frame cannot be replayed
        answerer.v42_feed_bytes(held_after_error)

        pump(caller, answerer, 9600 * 10)
        self.assertEqual((caller.v42_mode, answerer.v42_mode), ("lapm", "lapm"))
        self.assertEqual(bytes(answerer.out), caller_pending)
        self.assertEqual(bytes(caller.out), answerer_pending + held_after_error)
        self.assertEqual(caller._v42_recovery_stats["requeued_dte_bytes"],
                         len(caller_pending))
        self.assertEqual(caller._v42_recovery_stats["discarded_encoded_octets"], 2)
        self.assertEqual(answerer._v42_recovery_stats["completed"], 1)
        self.assertEqual(answerer._v42_recovery_stats["c_error_input_octets"],
                         len(invalid))

    def test_v42bis_recovery_timeout_marks_link_lost(self):
        engine = BitEngine()
        engine.start_v42(True, rate=9600, compression=(1, 1024, 8),
                         compression_recovery=True)
        engine.v42_mode = "recovering"
        engine._v42_recovery_stage = "establish"
        engine._v42_recovery_deadline = time.monotonic() - 0.001
        engine._v42_check()
        self.assertTrue(engine.v42_lost)
        self.assertIn("re-establishment timed out", engine._v42.events[-1])

    def test_connected_lapm_n400_exhaustion_marks_layer_lost_without_raw_fallback(self):
        caller, answerer = BitEngine(), BitEngine()
        caller.start_v42(True, rate=9600)
        answerer.start_v42(False, rate=9600)
        pump(caller, answerer, 9600 * 4)
        self.assertEqual((caller.v42_mode, answerer.v42_mode), ("lapm", "lapm"))

        link = caller._v42
        # Simulate each T401 expiry at the production link state boundary.
        # The LAPM suite separately drives the bit-clock timer and retry count.
        for _ in range(v42_lapm.N400 + 1):
            if link.state != "lapm":
                break
            link._t401_expired()
            caller._v42_check()

        self.assertEqual(link.state, "down")
        self.assertTrue(caller.v42_lost)
        self.assertEqual(caller.v42_mode, "lost")
        caller.v42_feed_bytes(b"must not leak into raw mode")
        self.assertEqual(list(caller.q), [])
        self.assertEqual(caller.v42_read(), b"")
        self.assertEqual(caller.v42_tx_bit(), 1)

    def test_v42bis_recovery_deadline_is_not_extended_by_link_restart(self):
        engine = BitEngine()
        engine.start_v42(True, rate=9600, compression=(1, 1024, 8),
                         compression_recovery=True)
        engine.v42_mode = "recovering"
        engine._v42_recovery_stage = "release"
        deadline = time.monotonic() + 1.0
        engine._v42_recovery_deadline = deadline
        engine._v42_bits = 1234
        engine._v42_restart_link(calling=True)
        self.assertEqual(engine._v42_recovery_deadline, deadline)
        self.assertEqual(engine._v42_bits, 0)
        engine._v42.close()

    def test_v42bis_recovery_hold_overflow_fails_closed(self):
        engine = BitEngine()
        engine.start_v42(True, rate=9600, compression=(1, 1024, 8),
                         compression_recovery=True)
        engine.v42_mode = "recovering"
        engine._v42_recovery_stage = "establish"
        engine._v42_recovery_deadline = time.monotonic() + 1.0
        engine._v42_recovery_hold_limit = 4

        engine.v42_feed_bytes(b"safe")
        self.assertEqual(engine.v42_mode, "recovering")
        self.assertEqual(bytes(engine._v42_hold), b"safe")
        engine.v42_feed_bytes(b"!")

        self.assertTrue(engine.v42_lost)
        self.assertEqual(engine._v42_recovery_stage, "failed")
        self.assertEqual(bytes(engine._v42_hold), b"")
        self.assertEqual(list(engine.q), [])  # never leak bytes into raw fallback
        self.assertEqual(engine.v42_read(), b"")
        self.assertIn("DTE data exceeded recovery hold limit", engine._v42.events[-1])
        engine._v42.close()

    def test_all_v42bis_c_error_conditions_recover_in_both_lapm_roles(self):
        import v42bis

        cases = ("stepup", "current-code", "empty-code", "reserved-command")
        for receive_role in ("caller", "answerer"):
            for case in cases:
                with self.subTest(receive_role=receive_role, condition=case):
                    caller, answerer = BitEngine(), BitEngine()
                    caller.start_v42(True, rate=9600, compression=(3, 1024, 8),
                                     compression_recovery=True)
                    answerer.start_v42(False, rate=9600, compression=(3, 1024, 8),
                                       compression_recovery=True)
                    pump(caller, answerer, 9600 * 4)
                    sender, receiver = ((answerer, caller) if receive_role == "caller"
                                        else (caller, answerer))
                    decoder = receiver._v42._dec
                    if case == "reserved-command":
                        invalid = bytes((decoder.escape, 9))
                    else:
                        decoder.compressed = True
                        if case == "stepup":
                            decoder.c2 = decoder.n1
                            codeword = v42bis.STEPUP
                        elif case == "current-code":
                            codeword = decoder.d.c1
                        else:
                            codeword = v42bis.N5
                            decoder.d.c1 = v42bis.N5 + 1
                        invalid = codeword.to_bytes(2, "little")

                    tx = sender._v42
                    tx._queue_i(invalid)
                    tx._send_i(*tx._unacked[-1])
                    pump(caller, answerer, 9600 * 8)
                    self.assertEqual((caller.v42_mode, answerer.v42_mode),
                                     ("lapm", "lapm"))
                    self.assertFalse(receiver.v42_lost)
                    self.assertEqual(bytes(receiver.out), b"")

                    suffix = (f"post-recovery {receive_role} {case} ".encode("ascii")
                              * 8)
                    sender.v42_feed_bytes(suffix)
                    pump(caller, answerer, 9600 * 3)
                    self.assertEqual(bytes(receiver.out), suffix)

    def test_connected_parameter_renegotiation_is_exposed_through_layer(self):
        caller, answerer = BitEngine(), BitEngine()
        caller.start_v42(True, rate=9600)
        answerer.start_v42(False, rate=9600)
        pump(caller, answerer, 9600 * 4)
        self.assertEqual((caller.v42_mode, answerer.v42_mode), ("lapm", "lapm"))

        self.assertTrue(caller.v42_renegotiate_parameters(
            tx_n401=32, rx_n401=24, tx_k=4, rx_k=3
        ))
        caller.v42_feed_bytes(b"queued while renegotiating")
        pump(caller, answerer, 9600 * 4)
        self.assertEqual((caller._v42.tx_n401, caller._v42.rx_n401,
                          caller._v42.tx_k, caller._v42.rx_k), (32, 24, 4, 3))
        self.assertEqual((answerer._v42.rx_n401, answerer._v42.tx_n401,
                          answerer._v42.rx_k, answerer._v42.tx_k), (32, 24, 4, 3))
        self.assertEqual(bytes(answerer.out), b"queued while renegotiating")

    def test_connected_parameter_renegotiation_answerer_initiated(self):
        caller, answerer = BitEngine(), BitEngine()
        caller.start_v42(True, rate=9600)
        answerer.start_v42(False, rate=9600)
        pump(caller, answerer, 9600 * 4)
        self.assertEqual((caller.v42_mode, answerer.v42_mode), ("lapm", "lapm"))

        caller_payload = b"caller data held behind connected XID"
        answerer_payload = b"answerer data held behind connected XID"
        self.assertTrue(answerer.v42_renegotiate_parameters(
            tx_n401=24, rx_n401=32, tx_k=3, rx_k=4
        ))
        caller.v42_feed_bytes(caller_payload)
        answerer.v42_feed_bytes(answerer_payload)
        pump(caller, answerer, 9600 * 4)

        self.assertEqual(answerer._v42.tx_n401, 24)
        self.assertGreaterEqual(answerer._v42.rx_n401, 32)
        self.assertEqual(answerer._v42.tx_n401, caller._v42.rx_n401)
        self.assertEqual(answerer._v42.rx_n401, caller._v42.tx_n401)
        self.assertEqual((answerer._v42.tx_k, answerer._v42.rx_k), (3, 4))
        self.assertEqual((answerer._v42.tx_k, answerer._v42.rx_k),
                         (caller._v42.rx_k, caller._v42.tx_k))
        self.assertEqual(bytes(caller.out), answerer_payload)
        self.assertEqual(bytes(answerer.out), caller_payload)

    def test_explicit_xid_policy_is_passed_to_lapm_link(self):
        engine = BitEngine()
        engine.start_v42(True, rate=9600, xid_fallback=False)
        self.assertFalse(engine._v42.xid_fallback)
        # The value is retained in the parameters used if ADP detection later
        # replaces the provisional link with direct V.42 establishment.
        self.assertFalse(engine._v42_params[-1])

    def test_strict_xid_attempt_does_not_fall_back_at_layer_deadline(self):
        """An unanswered XID must exhaust LAPM's N400 before raw fallback."""
        from v42_lapm import N400

        engine = BitEngine(rate=1200)
        engine.start_v42(True, rate=1200, establish_seconds=0.1,
                         detect=False, xid_fallback=False)
        # Cross the V42Layer establishment deadline, but not the first T401.
        for _ in range(1200 * 2):
            engine.v42_tx_bit()
        self.assertEqual(engine._v42.state, "idle")
        self.assertEqual(engine._v42._xid_sent, 1)
        self.assertEqual(engine.v42_mode, "detect")

        # No response: strict LAPM retries through N400, then the layer records
        # failure. It must never replace that attempt with raw asynchronous.
        for _ in range(1200 * 40):
            engine.v42_tx_bit()
            if engine.v42_mode == "failed":
                break
        self.assertEqual(engine._v42.state, "down", engine._v42.events)
        self.assertGreater(engine._v42._tries, N400)
        self.assertFalse(engine._v42._sabme_sent)
        self.assertEqual(engine.v42_mode, "failed")

    def test_invalid_v42bis_parameters_drop_call_instead_of_raw_fallback(self):
        from v42_lapm import XID

        private = (bytes((0, 3)) + b"V42"
                   + bytes((1, 1, 3, 2, 2, 4, 0, 3, 1, 251)))
        info = b"\x82\xF0" + len(private).to_bytes(2, "big") + private
        for calling in (False, True):
            with self.subTest(calling=calling):
                engine = BitEngine()
                engine.start_v42(calling, rate=9600, detect=False,
                                 compression=(3, 1024, 32))
                address = engine._v42.cmd_addr if calling else engine._v42.rsp_addr
                engine._v42._on_frame(bytes((address, XID)) + info)
                engine._v42_check()
                self.assertEqual(engine._v42.state, "down")
                self.assertEqual(engine.v42_mode, "lost")
                self.assertTrue(engine.v42_lost)

    def test_destructive_expedited_break_drops_decoded_but_uncollected_data(self):
        caller, answerer = BitEngine(), BitEngine()
        caller.start_v42(True, rate=9600)
        answerer.start_v42(False, rate=9600)
        pump(caller, answerer, 9600 * 4)
        self.assertEqual((caller.v42_mode, answerer.v42_mode), ("lapm", "lapm"))
        answerer._v42_rx.extend(b"not yet delivered")

        self.assertTrue(caller.v42_request_break(destructive=True, expedited=True))
        pump(caller, answerer, 9600 // 2)
        indication = answerer.v42_read_break_events()
        self.assertEqual(len(indication), 1)
        self.assertTrue(indication[0]["destructive"] and indication[0]["expedited"])
        self.assertTrue(answerer.v42_acknowledge_break())
        self.assertEqual(answerer.v42_read(), b"")
        pump(caller, answerer, 9600 // 2)
        self.assertEqual((caller.v42_mode, answerer.v42_mode), ("lapm", "lapm"))

    def test_l_signal_api_keeps_control_function_in_the_break_ack_path(self):
        caller, answerer = BitEngine(), BitEngine()
        caller.start_v42(True, rate=9600)
        answerer.start_v42(False, rate=9600)
        pump(caller, answerer, 9600 * 4)
        self.assertEqual((caller.v42_mode, answerer.v42_mode), ("lapm", "lapm"))

        self.assertTrue(caller.v42_request_break(length_ms=250, expedited=True))
        pump(caller, answerer, 9600 // 2)
        indication = answerer.v42_read_break_events()
        self.assertEqual(len(indication), 1)
        self.assertEqual(indication[0]["type"], "break_indication")
        self.assertEqual(indication[0]["length_ms"], 250)
        self.assertTrue(answerer.v42_acknowledge_break())
        pump(caller, answerer, 9600 // 2)
        confirmation = caller.v42_read_break_events()
        self.assertEqual(len(confirmation), 1)
        self.assertEqual(confirmation[0]["type"], "break_confirmed")
        self.assertEqual((caller.v42_mode, answerer.v42_mode), ("lapm", "lapm"))

    def test_local_orderly_release_completes_on_both_ends_without_carrier_loss(self):
        local, peer = BitEngine(), BitEngine()
        local.start_v42(False, rate=9600)
        peer.start_v42(True, rate=9600)
        pump(peer, local, 9600 * 3)
        self.assertEqual((local.v42_mode, peer.v42_mode), ("lapm", "lapm"))

        self.assertTrue(local.v42_release())
        pump(peer, local, 9600)
        self.assertEqual((local.v42_mode, peer.v42_mode), ("failed", "failed"))
        self.assertFalse(local.v42_lost)
        self.assertFalse(peer.v42_lost)

    def test_orderly_disc_falls_back_without_marking_the_physical_link_lost(self):
        from v42_lapm import DISC
        local, peer = BitEngine(), BitEngine()
        local.start_v42(False, rate=9600)
        peer.start_v42(True, rate=9600)
        pump(peer, local, 9600 * 3)
        self.assertEqual((local.v42_mode, peer.v42_mode), ("lapm", "lapm"))

        local._v42._out.extend(b"queued before DISC")
        local._v42._tx.clear()
        local._v42._on_frame(bytes((local._v42.rsp_addr, DISC | 0x10)))
        self.assertEqual(local.v42_mode, "lapm")  # wait until the UA leaves

        for _ in range(256):
            local.v42_tx_bit()
            if local.v42_mode == "failed":
                break
        self.assertEqual(local.v42_mode, "failed")  # orderly auto-reliable fallback
        self.assertFalse(local.v42_lost)
        self.assertEqual(local.v42_read(), b"queued before DISC")

    def test_answerer_direct_flags_leave_detection_timer_then_establish_lapm(self):
        from v42_lapm import FLAG_BITS
        a,b=BitEngine(),BitEngine()
        b.start_v42(False,rate=9600)
        b.v42_feed_bytes(b'held answerer payload')
        # A direct-protocol originator sends flags, not ODP. Let detection
        # expire while protocol establishment is still legitimately pending.
        for i in range(9600):
            b.v42_rx_bits([FLAG_BITS[i%8]]);b.v42_tx_bit()
        self.assertEqual(b._v42.state,'idle')
        self.assertFalse(b._v42.adp_seen)
        self.assertEqual(b.v42_mode,'detect')
        a.start_v42(True,rate=9600,detect=False)
        a.v42_feed_bytes(b'direct originator payload')
        pump(a,b,9600*5)
        self.assertEqual((a.v42_mode,b.v42_mode),('lapm','lapm'))
        self.assertEqual(bytes(a.out),b'held answerer payload')
        self.assertEqual(bytes(b.out),b'direct originator payload')

    def test_answerer_flags_without_establishment_still_have_finite_deadline(self):
        from v42_lapm import FLAG_BITS
        b=BitEngine();b.start_v42(False,rate=9600,establish_seconds=2.)
        for i in range(9600*3):
            b.v42_rx_bits([FLAG_BITS[i%8]]);b.v42_tx_bit()
        self.assertEqual(b.v42_mode,'failed')

    def test_isolated_flags_do_not_extend_raw_detection_deadline(self):
        from v42_lapm import FLAG_BITS
        b = BitEngine()
        b.start_v42(False, rate=9600)
        incoming = (FLAG_BITS + [1] * 128) * 4 + [1] * 9600
        for bit in incoming:
            b.v42_rx_bits([bit])
            b.v42_tx_bit()
        self.assertEqual(b.v42_mode, 'failed')
        self.assertEqual(b._v42._phase, 'detect')
        self.assertFalse(b._v42.adp_seen)

    def test_continuous_flags_allow_shared_zero_and_reset_across_gap(self):
        from v42_lapm import FLAG_BITS, _Hdlc
        receiver = _Hdlc()
        for bit in FLAG_BITS + FLAG_BITS[1:] * 3:
            receiver.push(bit)
        self.assertEqual(receiver.continuous_flags, 4)
        for bit in [1] * 128 + FLAG_BITS:
            receiver.push(bit)
        self.assertEqual(receiver.flags, 5)
        self.assertEqual(receiver.continuous_flags, 1)

    def test_both_ends_reach_lapm_and_exchange_data(self):
        a, b = BitEngine(), BitEngine()
        a.start_v42(True, rate=9600)
        b.start_v42(False, rate=9600)
        a.v42_feed_bytes(b"hello from the caller\r\n")
        b.v42_feed_bytes(b"hello from the answerer\r\n")
        pump(a, b, 9600 * 5)
        self.assertEqual((a.v42_mode, b.v42_mode), ("lapm", "lapm"))
        self.assertEqual(bytes(b.out), b"hello from the caller\r\n")
        self.assertEqual(bytes(a.out), b"hello from the answerer\r\n")

    def test_v42bis_is_negotiated_and_compresses_text(self):
        a, b = BitEngine(), BitEngine()
        a.start_v42(True, rate=9600, compression=(3, 1024, 8))
        b.start_v42(False, rate=9600, compression=(3, 1024, 8))
        text = (b"the quick brown fox jumps over the lazy dog. " * 40) + b"\r\n"
        a.v42_feed_bytes(text)
        b.v42_feed_bytes(text[::-1])
        pump(a, b, 9600 * 8)
        self.assertEqual((a.v42_mode, b.v42_mode), ("lapm", "lapm"))
        self.assertEqual(a.v42_compression, (3, 1024, 8))
        self.assertEqual(b.v42_compression, (3, 1024, 8))
        self.assertEqual(bytes(b.out), text)
        self.assertEqual(bytes(a.out), text[::-1])
        self.assertLess(a._v42.compressed_out, len(text) // 3)          # the line carried far less than the text

    def test_compression_is_dropped_when_the_responder_does_not_accept_it(self):
        a, b = BitEngine(), BitEngine()
        a.start_v42(True, rate=9600, compression=(3, 1024, 8))
        b.start_v42(False, rate=9600)                                  # accepts nothing
        a.v42_feed_bytes(b"plain " * 30)
        pump(a, b, 9600 * 5)
        self.assertEqual((a.v42_mode, b.v42_mode), ("lapm", "lapm"))
        self.assertIsNone(a.v42_compression)
        self.assertEqual(bytes(b.out), b"plain " * 30)

    def test_caller_falls_back_to_raw_async_when_peer_has_no_v42(self):
        a, b = BitEngine(), BitEngine()
        a.start_v42(True, rate=9600)                 # the peer never starts V.42
        a.v42_feed_bytes(b"raw data please\r\n")
        pump(a, b, 9600 * 3)
        self.assertEqual(a.v42_mode, "failed")
        self.assertIn(b"raw data please\r\n", bytes(b.out))

    def test_plain_mode_is_untouched_async(self):
        a, b = BitEngine(), BitEngine()
        a.v42_feed_bytes(b"no error control\r\n")
        pump(a, b, 400)
        self.assertEqual(bytes(b.out), b"no error control\r\n")

    def test_mnp_protocol_runs_through_the_same_layer(self):
        for level in (2, 3, 4):
            with self.subTest(level=level):
                a, b = BitEngine(), BitEngine()
                a.start_v42(True, rate=9600, protocol="mnp", mnp_level=level)
                b.start_v42(False, rate=9600, protocol="mnp", mnp_level=level)
                a.v42_feed_bytes(b"hello over MNP\r\n")
                pump(a, b, 9600 * 4)
                self.assertEqual((a.v42_mode, b.v42_mode), ("lapm", "lapm"))
                self.assertEqual(bytes(b.out), b"hello over MNP\r\n")

    def test_mnp_originator_falls_back_to_raw_when_the_peer_is_silent(self):
        a = BitEngine()
        a.start_v42(True, rate=9600, protocol="mnp", establish_seconds=3.0, detect_seconds=0)
        a.v42_feed_bytes(b"late\r\n")
        for _ in range(9600 * 5):
            a.v42_tx_bit()
            a.v42_rx_bits([1])
        self.assertEqual(a.v42_mode, "failed")



def delayed_pump(a, b, bits, delay):
    """Like pump, with `delay` bits of propagation (plus receive queue) in each direction."""
    to_a, to_b = deque([1] * delay), deque([1] * delay)
    for _ in range(bits):
        ba, bb = a.v42_tx_bit(), b.v42_tx_bit()
        to_a.append(bb)
        to_b.append(ba)
        a.out += a.v42_rx_bits([to_a.popleft()])
        b.out += b.v42_rx_bits([to_b.popleft()])
        a.out += a.v42_read()
        b.out += b.v42_read()


class DetectionWindowTests(unittest.TestCase):
    """V.42 T400 timing and late ADP handling."""

    @staticmethod
    def adp_bits(gap):
        from v42_detect import char_bits

        bits = []
        for index, char in enumerate(b"ECEC"):
            bits.extend(char_bits(char))
            if index < 3:
                bits.extend([1] * gap)
        return bits

    def run_delayed(self, one_way_seconds, extra):
        a, b = BitEngine(), BitEngine()
        a.v42_detect_extra_seconds = extra
        b.v42_detect_extra_seconds = extra
        a.start_v42(True, rate=9600)
        b.start_v42(False, rate=9600)
        a.v42_feed_bytes(b"hello from the caller\r\n")
        delayed_pump(a, b, 9600 * 9, int(one_way_seconds * 9600))
        return a, b

    def test_a_return_trip_longer_than_the_default_window_fails_without_the_extension(self):
        a, b = self.run_delayed(0.5, extra=0.0)              # ADP is back after about 1.0 s, beyond T400
        self.assertEqual(a.v42_mode, "failed")

    def test_default_detection_deadline_is_v42_t400(self):
        engine = BitEngine(rate=9600)
        engine.start_v42(True, rate=9600)
        self.assertEqual(engine._v42_detect_deadline, 750 * 9600 // 1000)

    def test_lenient_adp_path_cannot_resurrect_detection_after_t400(self):
        engine = BitEngine(rate=9600)
        engine.start_v42(True, rate=9600)
        for _ in range(72):
            engine.v42_tx_bit()
        self.assertTrue(engine._v42_adp_armed)
        engine._v42_adp_detector.feed(b"ECEC")
        engine._v42_bits = engine._v42_detect_deadline + 10
        self.assertFalse(engine._v42_lenient_adp())
        self.assertEqual(engine.v42_mode, "detect")

    def test_production_layer_accepts_back_to_back_adp_for_observed_peer(self):
        engine = BitEngine(rate=9600)
        engine.start_v42(True, rate=9600)
        for _ in range(72):
            engine.v42_tx_bit()
        self.assertTrue(engine._v42_adp_armed)

        engine.v42_rx_bits(self.adp_bits(0))

        self.assertTrue(engine._v42.adp_seen)
        self.assertEqual(engine._v42._phase, "protocol")

    def test_production_layer_does_not_accept_overlong_adp_spacing(self):
        engine = BitEngine(rate=9600)
        engine.start_v42(True, rate=9600)
        for _ in range(72):
            engine.v42_tx_bit()

        engine.v42_rx_bits(self.adp_bits(17))

        self.assertFalse(engine._v42.adp_seen)
        self.assertEqual(engine._v42._phase, "detect")

    def test_adp_returning_just_before_t400_is_not_rejected_early(self):
        a, b = self.run_delayed(0.36, extra=0.0)
        self.assertEqual((a.v42_mode, b.v42_mode), ("lapm", "lapm"))

    def test_the_extension_lets_the_same_slow_link_reach_lapm(self):
        a, b = self.run_delayed(0.5, extra=1.5)
        self.assertEqual((a.v42_mode, b.v42_mode), ("lapm", "lapm"))
        self.assertEqual(bytes(b.out), b"hello from the caller\r\n")

    def test_the_extension_does_not_change_a_normal_link(self):
        a, b = self.run_delayed(0.0, extra=1.5)
        self.assertEqual((a.v42_mode, b.v42_mode), ("lapm", "lapm"))

    def test_a_peer_without_v42_still_falls_back_after_the_longer_window(self):
        a = BitEngine()
        a.v42_detect_extra_seconds = 1.5
        a.start_v42(True, rate=9600)
        for _ in range(9600 * 6):
            a.v42_tx_bit()
            a.v42_rx_bits([1])
        self.assertEqual(a.v42_mode, "failed")
        self.assertGreater(a._v42_detect_deadline, int(2.0 * 9600))

    def test_the_v34_engine_derives_the_extension_from_its_round_trip(self):
        from unittest.mock import patch
        import os
        from v34_modem import V34Modem
        modem = V34Modem(data_rate=9600)
        self.assertAlmostEqual(modem.v42_detect_extra_seconds, V34Modem.V42_RX_LAG_MARGIN_S)
        modem.caller.rtd_samples = 3900                      # 487.5 ms
        self.assertAlmostEqual(modem.v42_detect_extra_seconds, 0.4875 + V34Modem.V42_RX_LAG_MARGIN_S)
        with patch.dict(os.environ, {"V42_DETECT_EXTRA_S": "0"}):
            self.assertEqual(modem.v42_detect_extra_seconds, 0.0)

if __name__ == "__main__":
    unittest.main()
