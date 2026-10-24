#!/usr/bin/env python3
"""Self-tests for the SpanDSP V.42 (LAPM) binding, wired bit-to-bit."""

import random
import struct
import time
import unittest

import spandsp_v42
from spandsp_v42 import V42Link
from v42_lapm import FLAG_BITS, N400, XID, _Hdlc, V42Link as ProjectV42Link
from v42_layer import V42Layer
from v22 import V22Modem


def run(caller, answerer, bits, on_bit=None, flip=None, flip_answerer=None):
    """Clock two links against each other: one bit each way per step.
    ``flip(i)`` may return True to corrupt the caller->answerer bit."""
    for i in range(bits):
        bc, ba = caller.tx_bit(), answerer.tx_bit()
        if flip and flip(i):
            bc ^= 1
        if flip_answerer and flip_answerer(i):
            ba ^= 1
        answerer.rx_bit(bc)
        caller.rx_bit(ba)
        if on_bit:
            on_bit(i)


class ProjectV42LayerPeer(V42Layer):
    """Minimal production control-function layer for the independent-peer gates."""

    def __init__(self, rate=9600):
        self.bit_rate = rate
        self._v42_setup()

    def _plain_tx_bit(self):
        return 1

    def _plain_feed_bytes(self, data):
        pass

    def _plain_feed_marks(self, count):
        pass

    def _plain_rx_bits(self, bits):
        return b""


@unittest.skipUnless(spandsp_v42.available(), "libspandsp not installed")
class V42Tests(unittest.TestCase):
    def test_c_error_recovery_times_out_when_spandsp_control_does_not_reestablish(self):
        """A LAPM peer that releases but never starts L-ESTABLISH is bounded."""
        project = ProjectV42LayerPeer()
        independent = V42Link(True, detect=False, bit_rate=9600,
                              compression=(3, 1024, 8))
        project.start_v42(False, detect=False, rate=9600,
                          compression=(3, 1024, 8), compression_recovery=True)
        try:
            def tick(count):
                for _ in range(count):
                    project_bit = project.v42_tx_bit()
                    independent_bit = independent.tx_bit()
                    project.v42_rx_bits([independent_bit])
                    independent.rx_bit(project_bit)
                    project.v42_read()
                    independent.read()

            tick(9600 * 2)
            self.assertEqual((project.v42_mode, independent.state), ("lapm", "lapm"))
            self.assertEqual(project.v42_compression, (3, 1024, 8))

            project._v42_recovery_timeout_seconds = 0.05
            decoder = project._v42._dec
            decoder.compressed = True
            invalid = decoder.d.c1.to_bytes(2, "little")
            # SpanDSP emits a valid LAPM I-frame containing invalid V.42bis
            # data. It completes the orderly release but its control function
            # is intentionally not asked to create a new LAPM instance.
            independent._tx.extend(invalid)
            for _ in range(9600 * 2):
                tick(1)
                if project._v42_recovery_stage == "establish":
                    break
            self.assertEqual(project._v42_recovery_stage, "establish",
                             project._v42.events[-12:])
            self.assertEqual(independent.state, "down", independent.events[-12:])

            time.sleep(0.06)
            project.v42_tx_bit()  # production bit clock observes the monotonic deadline
            self.assertTrue(project.v42_lost, project._v42.events[-12:])
            self.assertTrue(any("re-establishment timed out" in event
                                for event in project._v42.events), project._v42.events[-12:])
        finally:
            independent.close()
            if project._v42 is not None:
                project._v42.close()

    def test_c_error_recovery_with_spandsp_peer_in_both_roles(self):
        """SpanDSP LAPM carries each recovery case in both El-Modem roles."""
        import v42bis

        cases = ("stepup", "current-code", "empty-code", "reserved-command")
        for project_caller in (True, False):
            for condition in cases:
                with self.subTest(project_role="caller" if project_caller else "answerer",
                                  condition=condition):
                    project = ProjectV42LayerPeer()
                    independent = V42Link(not project_caller, detect=False, bit_rate=9600,
                                           compression=(3, 1024, 8))
                    independent_restarted = False
                    independent_release_bits = 0
                    project.start_v42(project_caller, detect=False, rate=9600,
                                      compression=(3, 1024, 8), compression_recovery=True)
                    received_project = bytearray()
                    received_independent = bytearray()

                    def tick(count):
                        nonlocal independent, independent_restarted, independent_release_bits
                        for _ in range(count):
                            project_bit = project.v42_tx_bit()
                            independent_bit = independent.tx_bit()
                            project.v42_rx_bits([independent_bit])
                            independent.rx_bit(project_bit)
                            received_project.extend(project.v42_read())
                            received_independent.extend(independent.read())
                            # SpanDSP supplies the LAPM entity, not V.42's
                            # control function. When El-Modem is the answerer,
                            # model the independent caller control function's
                            # L-RELEASE decision to issue a fresh L-ESTABLISH
                            # after the LAPM release response has drained.
                            # With no such peer action, El-Modem correctly
                            # remains bounded in await_establish and times out.
                            if not project_caller and independent.state == "down":
                                independent_release_bits += 1
                            if (not project_caller and independent.state == "down"
                                    and independent_release_bits >= 2000
                                    and not independent_restarted):
                                independent.close()
                                independent = V42Link(True, detect=False, bit_rate=9600,
                                                      compression=(3, 1024, 8))
                                independent_restarted = True

                    try:
                        tick(9600 * 2)
                        self.assertEqual((project.v42_mode, independent.state),
                                         ("lapm", "lapm"))
                        self.assertEqual(project.v42_compression, (3, 1024, 8))
                        self.assertEqual(independent.compression, (3, 1024, 8))

                        decoder = project._v42._dec
                        old_decoder = decoder
                        if condition == "reserved-command":
                            invalid = bytes((decoder.escape, 9))
                        else:
                            decoder.compressed = True
                            if condition == "stepup":
                                decoder.c2 = decoder.n1
                                codeword = v42bis.STEPUP
                            elif condition == "current-code":
                                codeword = decoder.d.c1
                            else:
                                codeword = v42bis.N5
                                decoder.d.c1 = v42bis.N5 + 1
                            invalid = codeword.to_bytes(2, "little")

                        # These are injected after the independent V.42bis
                        # encoder and before its LAPM framing. The peer sends
                        # the malformed octets in valid I-frames with FCS and
                        # bit stuffing; this is not a raw decoder-only vector.
                        independent._tx.extend(invalid)
                        tick(9600 * 2)
                        self.assertEqual((project.v42_mode, independent.state),
                                         ("lapm", "lapm"),
                                         (project._v42.events[-12:], independent.events[-12:],
                                          independent_restarted))
                        self.assertIsNot(project._v42._dec, old_decoder)
                        self.assertEqual(bytes(received_project), b"")

                        to_project = b"SpanDSP payload after dictionary reset; " * 30
                        to_independent = b"El-Modem payload after dictionary reset; " * 30
                        independent.write(to_project)
                        project.v42_feed_bytes(to_independent)
                        tick(9600 * 5)
                        self.assertEqual(bytes(received_project), to_project)
                        self.assertEqual(bytes(received_independent), to_independent)
                        self.assertEqual((project.v42_mode, independent.state),
                                         ("lapm", "lapm"))
                    finally:
                        if independent._bis is not None:
                            independent._lib.v42bis_free(independent._bis)
                            independent._bis = None
                        independent.close()

    def test_strict_project_xid_policy_with_span_dsp_response_loss(self):
        """An independent responder processes XID, but its replies are lost."""

        class DropXidResponses:
            def __init__(self):
                self.parser = _Hdlc()
                self.pending = []
                self.dropped = 0

            def pass_bits(self, bits, *, intercept=True):
                if not intercept:
                    self.parser = _Hdlc()
                    self.pending.clear()
                    return list(bits)
                out = []
                for bit in bits:
                    self.pending.append(bit)
                    frames = self.parser.push(bit)
                    for frame, good_fcs in frames:
                        if good_fcs and len(frame) >= 2 and frame[1] & 0xEF == XID:
                            self.dropped += 1
                            # Preserve the closing flag as idle/opening framing;
                            # discard the independently generated XID response.
                            out.extend(self.pending[-len(FLAG_BITS):])
                        else:
                            out.extend(self.pending)
                        self.pending.clear()
                return out

        project = ProjectV42Link(
            True, detect=True, bit_rate=9600, frame_size=32, window=4,
            xid_fallback=False,
        )
        independent = V42Link(False, detect=True, bit_rate=9600,
                               frame_size=32, window=4)
        dropper = DropXidResponses()
        caller_payload = bytes(range(256))
        answer_payload = b"independent answer payload" * 8
        project.write(caller_payload)
        independent.write(answer_payload)
        project_rx = bytearray()
        independent_rx = bytearray()

        try:
            for _ in range(9600 * 30):
                project_bit = project.tx_bit()
                independent.rx_bit(project_bit)
                peer_bit = independent.tx_bit()
                for delivered in dropper.pass_bits((peer_bit,), intercept=independent.adp_seen):
                    project.rx_bit(delivered)
                project_rx.extend(project.read())
                independent_rx.extend(independent.read())
                if project.state == "down":
                    break

            self.assertGreaterEqual(
                dropper.dropped, N400,
                (project.state, project.events[-12:], independent.state,
                 independent.events[-12:], dropper.parser.flags,
                 dropper.parser.overlong),
            )
            self.assertEqual(project.state, "down", project.events[-8:])
            self.assertFalse(project._sabme_sent)
            self.assertEqual(bytes(project_rx), b"")
            self.assertEqual(bytes(independent_rx), b"")
            self.assertNotEqual(independent.state, "lapm")
        finally:
            independent.close()

    def test_detection_then_lapm_and_data_both_ways(self):
        caller, answerer = V42Link(True), V42Link(False)
        a_msg = b"hello from the caller " * 40
        b_msg = bytes(range(256)) * 4
        sent = {"done": False}

        def tick(i):
            if caller.state == "lapm" == answerer.state and not sent["done"]:
                caller.write(a_msg)
                answerer.write(b_msg)
                sent["done"] = True

        run(caller, answerer, 60000, tick)
        self.assertEqual((caller.state, answerer.state), ("lapm", "lapm"),
                         (caller.events, answerer.events))
        self.assertEqual(answerer.read(), a_msg)
        self.assertEqual(caller.read(), b_msg)

    def test_data_survives_a_noisy_line(self):
        # LAPM retransmits: corrupt about 1 bit in 2000 and the payload must
        # still arrive intact (this is what the lab ARQ could not give a real
        # peer, and what raw mode cannot give at all).
        rng = random.Random(7)
        caller, answerer = V42Link(True), V42Link(False)
        payload = bytes(rng.randrange(256) for _ in range(1500))
        state = {"sent": False, "armed": False}

        def tick(i):
            if caller.state == "lapm" == answerer.state and not state["sent"]:
                caller.write(payload)
                state.update(sent=True, armed=True)

        # SpanDSP's retransmit timers are counted in bits at an assumed
        # 28800 bps, so recovering from a lost frame takes many bit-times.
        run(caller, answerer, 1500000, tick,
            flip=lambda i: state["armed"] and rng.randrange(4000) == 0)
        self.assertEqual(answerer.read(), payload)

    def test_simultaneous_duplex_payload_survives_bidirectional_bit_errors(self):
        """Exercise both independent LAPM directions under simultaneous corruption."""
        for project_calls in (True, False):
            with self.subTest(project_calls=project_calls):
                rng = random.Random(18731 + int(project_calls))
                if project_calls:
                    caller, answerer = V42Link(True), ProjectV42Link(False)
                else:
                    caller, answerer = ProjectV42Link(True), V42Link(False)
                caller_payload = bytes((i * 37 + 11) & 0xFF for i in range(900))
                answerer_payload = bytes((i * 53 + 7) & 0xFF for i in range(900))
                armed = {"value": False}

                def start(_):
                    if (not armed["value"] and caller.state == "lapm"
                            and answerer.state == "lapm"):
                        caller.write(caller_payload)
                        answerer.write(answerer_payload)
                        armed["value"] = True

                def damage(_):
                    return armed["value"] and rng.randrange(4000) == 0

                run(caller, answerer, 2_400_000, on_bit=start,
                    flip=damage, flip_answerer=damage)
                self.assertEqual((caller.state, answerer.state), ("lapm", "lapm"),
                                 (caller.events[-12:], answerer.events[-12:]))
                self.assertEqual(answerer.read(), caller_payload)
                self.assertEqual(caller.read(), answerer_payload)

    def test_non_v42_peer_is_reported_unsupported(self):
        caller = V42Link(True)
        # A peer that sends only marks (idle line) never answers the ODP.
        for _ in range(30000):
            caller.tx_bit()
            caller.rx_bit(1)
        self.assertEqual(caller.state, "unsupported", caller.events)

    def test_project_lapm_orderly_release_is_acknowledged_by_spandsp(self):
        """Check the project's DISC/UA release against a separate LAPM stack."""
        project = ProjectV42Link(True, detect=True, bit_rate=2400)
        independent = V42Link(False, bit_rate=2400)
        project_payload = bytes(range(256)) * 2
        independent_payload = bytes(reversed(range(256))) * 2
        received_project = bytearray()
        received_independent = bytearray()
        payloads_queued = payloads_complete = release_requested = False
        release_after = None

        try:
            for bit_index in range(160_000):
                project_bit = project.tx_bit()
                independent_bit = independent.tx_bit()
                independent.rx_bit(project_bit)
                project.rx_bit(independent_bit)

                if (not payloads_queued and project.state == "lapm"
                        and independent.state == "lapm"):
                    project.write(project_payload)
                    independent.write(independent_payload)
                    payloads_queued = True

                received_project.extend(project.read())
                received_independent.extend(independent.read())
                if (payloads_queued and not payloads_complete
                        and received_project == independent_payload
                        and received_independent == project_payload):
                    # Allow the final I-frame acknowledgements to cross before
                    # initiating DISC; the test is specifically about release.
                    payloads_complete = True
                    release_after = bit_index + 480
                if (release_after is not None and not release_requested
                        and bit_index >= release_after):
                    release_requested = project.release()
                if release_requested and project.state == "released":
                    break

            self.assertTrue(payloads_queued, (project.state, independent.state))
            self.assertEqual(bytes(received_project), independent_payload)
            self.assertEqual(bytes(received_independent), project_payload)
            self.assertTrue(payloads_complete)
            self.assertTrue(release_requested)
            self.assertEqual(project.state, "released", project.events)
            self.assertTrue(any("DISC acknowledged by UA" in event
                                for event in project.events), project.events)
            self.assertIn("LINK_DISCONNECTED", independent.events)
        finally:
            independent.close()

    def test_bell212a_audio_lapm_release_round_trip_with_spandsp(self):
        """Exercise Bell audio demodulation across an independent LAPM release."""

        class SpanDSPBellAnswerer(V22Modem):
            def __init__(self):
                super().__init__(False, 1200, profile="bell212a")
                self.span_link = None

            def start_span_link(self):
                self.span_link = V42Link(False, detect=True, bit_rate=1200)
                self.v42_mode = "detect"

            def _sync_span_mode(self):
                if self.span_link is None:
                    return
                self.v42_mode = {
                    "detect": "detect", "lapm": "lapm",
                    "unsupported": "failed", "down": "off",
                }[self.span_link.state]

            def v42_tx_bit(self):
                if self.span_link is None:
                    return 1
                bit = self.span_link.tx_bit()
                self._sync_span_mode()
                return bit

            def v42_rx_bits(self, bits):
                if self.span_link is not None:
                    for bit in bits:
                        self.span_link.rx_bit(bit)
                    self._sync_span_mode()
                return b""

            def v42_feed_bytes(self, data):
                if self.span_link is not None:
                    self.span_link.write(data)

            def v42_read(self):
                return self.span_link.read() if self.span_link is not None else b""

        caller = V22Modem(True, 1200, profile="bell212a")
        answerer = SpanDSPBellAnswerer()
        caller_payload = bytes(range(256)) * 2
        answerer_payload = bytes(reversed(range(256))) * 2
        got_caller, got_answerer = bytearray(), bytearray()
        v42_started = payloads_queued = payloads_complete = False
        release_requested = False
        release_after_tick = None
        received_payloads = None
        try:
            for tick in range(50 * 35):
                caller_pcm = caller.tx_samples(160)
                answerer_pcm = answerer.tx_samples(160)
                got_answerer.extend(answerer.feed_pcm(
                    struct.pack("<160h", *caller_pcm)))
                got_caller.extend(caller.feed_pcm(
                    struct.pack("<160h", *answerer_pcm)))

                # Bell 212A reports carrier before its post-training mark
                # interval ends. Start each LAPM detector when that modem's
                # physical synchronous data stream is ready, rather than
                # spending the peer's T400 while only marks are on the line.
                if (not v42_started and caller.v42_data_ready
                        and answerer.v42_data_ready):
                    caller.start_v42(True, detect=True, rate=1200)
                    answerer.start_span_link()
                    v42_started = True

                if (v42_started and not payloads_queued
                        and caller.v42_mode == "lapm" and answerer.v42_mode == "lapm"):
                    caller.feed_bytes(caller_payload)
                    answerer.feed_bytes(answerer_payload)
                    payloads_queued = True

                if (payloads_queued and not payloads_complete
                        and got_caller == answerer_payload
                        and got_answerer == caller_payload):
                    payloads_complete = True
                    release_after_tick = tick + 20  # allow final I-frame ACKs to cross
                    received_payloads = (bytes(got_caller), bytes(got_answerer))

                if (payloads_complete and not release_requested
                        and tick >= release_after_tick):
                    release_requested = caller.v42_release()

                if release_requested and caller._v42.state == "released":
                    break

            self.assertTrue(v42_started, (caller.events, answerer.events))
            self.assertTrue(payloads_queued)
            self.assertTrue(payloads_complete)
            self.assertEqual(received_payloads[0], answerer_payload)
            self.assertEqual(received_payloads[1], caller_payload)
            self.assertTrue(release_requested)
            self.assertEqual(caller._v42.state, "released", caller._v42.events)
            self.assertTrue(any("DISC acknowledged by UA" in event
                                for event in caller._v42.events), caller._v42.events)
            self.assertIn("LINK_DISCONNECTED", answerer.span_link.events)
        finally:
            caller.close()
            answerer.close()


@unittest.skipUnless(spandsp_v42.available(), "libspandsp not installed")
class V42TimerTests(unittest.TestCase):
    def test_corrected_timers_keep_a_multi_window_transfer_near_wire_speed(self):
        # With SpanDSP's default 28800 bps timer assumption a 2400 bps link
        # stalls waiting on timers: a 6000-byte transfer took 361 s against a
        # ~25 s wire minimum. With the line rate set it takes ~31 s.
        rate, nbytes = 2400, 4000  # bigger than the 15 x 128 byte send window
        caller = V42Link(True, bit_rate=rate)
        answerer = V42Link(False, bit_rate=rate)
        if not caller.timers_scaled:
            self.skipTest("libspandsp lacks v42_set_bit_rate (unpatched build)")
        payload = bytes(range(256)) * (nbytes // 256) + bytes(nbytes % 256)
        got, start = bytearray(), None
        for i in range(2_000_000):
            if start is None and caller.state == "lapm" == answerer.state:
                caller.write(payload)
                start = i
            bc, ba = caller.tx_bit(), answerer.tx_bit()
            answerer.rx_bit(bc)
            caller.rx_bit(ba)
            got += answerer.read()
            if len(got) >= nbytes:
                break
        self.assertEqual(bytes(got), payload)
        seconds = (i - start) / rate
        wire_minimum = nbytes * 10 / rate
        self.assertLess(seconds, 2.0 * wire_minimum,
                        f"{seconds:.0f}s for a {wire_minimum:.0f}s transfer")


@unittest.skipUnless(spandsp_v42.available(), "libspandsp not installed")
class V42AckStallTests(unittest.TestCase):
    def test_a_window_full_of_small_frames_does_not_stall_for_the_poll_timer(self):
        # SpanDSP built every ack with the F bit set, and the sender discards
        # such an ack unless a poll is outstanding, so a one-way sender waited
        # for the 10 s T403 timer each time its window filled: 800 bytes in
        # 32-byte frames (25 frames, window 15) took 11 s against a 4 s wire
        # time. spandsp-v42-tuning.patch echoes the P bit instead.
        rate, nbytes = 2400, 800
        caller = V42Link(True, bit_rate=rate, frame_size=32)
        answerer = V42Link(False, bit_rate=rate, frame_size=32)
        if not caller.frame_params_set:
            self.skipTest("libspandsp lacks v42_set_frame_params (unpatched build)")
        payload = bytes(range(256)) * 4
        payload = payload[:nbytes]
        got, start = bytearray(), None
        for i in range(1_000_000):
            if start is None and caller.state == "lapm" == answerer.state:
                caller.write(payload)
                start = i
            bc, ba = caller.tx_bit(), answerer.tx_bit()
            answerer.rx_bit(bc)
            caller.rx_bit(ba)
            got += answerer.read()
            if len(got) >= nbytes:
                break
        self.assertEqual(bytes(got), payload)
        seconds = (i - start) / rate
        self.assertLess(seconds, 2.0 * (nbytes * 10 / rate), f"{seconds:.1f}s")


if __name__ == "__main__":
    unittest.main()
