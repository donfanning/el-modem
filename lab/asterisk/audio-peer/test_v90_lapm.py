"""Paired bit-pipe tests for the V.90 V.42/LAPM adapter."""
import unittest

import numpy as np
import v34_data
import v34_negotiation
import v34_tx
from v90_digital_phase4 import V90DigitalPhase4Encoder
from v90_digital_phase4_rx import V90DigitalPhase4Receiver
from v90_lapm import V90LapmLink
from v90_upstream_data import V90UpstreamDataReceiver
from test_v90_cp import _frame
from v90_cp import parse_cp
from v90_digital_phase3_session import V90DigitalAnswererPhase3Session
from v90_digital_phase4_session import V90DigitalAnswererPhase4Session


def _profile(law):
    maps = tuple(tuple(range(16 + 16 * index, 32 + 16 * index))
                 for index in range(6))
    return {
        "kind": "cp", "codec": law, "ack": 0,
        "rate": 21 * 8000 / 6,
        "spectral_sign_bits": 0, "lookahead": 0,
        "interval_constellations": tuple(range(6)),
        "constellations": maps,
        "codec_constellations": maps,
    }


class V90LapmLinkTests(unittest.TestCase):
    def test_plain_fallback_supplies_mark_bits(self):
        link = V90LapmLink(rate=9600, calling_party=False)
        link._plain_feed_marks(5)
        self.assertEqual([link._plain_tx_bit() for _ in range(5)], [1] * 5)

    def test_paired_lapm_links_exchange_exact_dte_payload(self):
        for law in ("pcmu", "pcma"):
            with self.subTest(law=law):
                profile = _profile(law)
                downstream_tx = V90DigitalPhase4Encoder(profile, law)
                downstream_rx = V90DigitalPhase4Receiver(profile, law)
                frame_bits = downstream_tx.s + downstream_tx.k
                b1d = downstream_tx.encode_bits([1] * (48 * frame_bits))
                self.assertEqual(downstream_rx.decode_codewords(b1d),
                                 [1] * (48 * frame_bits))

                # 28 kbit/s maps to 21 user bits in each six-octet V.90 frame.
                caller = V90LapmLink(rate=28000, calling_party=True)
                answerer = V90LapmLink(rate=28000, calling_party=False)
                caller_payload = bytes((index * 73 + 19) & 0xFF
                                       for index in range(1536))
                answer_payload = bytes((index * 41 + 203) & 0xFF
                                       for index in range(1024))
                caller_received = bytearray()
                answer_received = bytearray()

                for _ in range(3000):
                    up_bits = caller.read_tx_bits(4 * frame_bits)
                    down_bits = answerer.read_tx_bits(4 * frame_bits)
                    down_octets = downstream_tx.encode_bits(down_bits)
                    down_recovered = downstream_rx.decode_codewords(down_octets)
                    answerer.feed_line_bits(up_bits)
                    caller.feed_line_bits(down_recovered)
                    caller.take_received()
                    answerer.take_received()
                    if caller.v42_mode == answerer.v42_mode == "lapm":
                        break
                else:
                    self.fail(
                        f"LAPM did not establish: {caller.v42_mode}/"
                        f"{answerer.v42_mode}")

                caller.feed_bytes(caller_payload)
                answerer.feed_bytes(answer_payload)
                for _ in range(12000):
                    up_bits = caller.read_tx_bits(4 * frame_bits)
                    down_bits = answerer.read_tx_bits(4 * frame_bits)
                    down_octets = downstream_tx.encode_bits(down_bits)
                    down_recovered = downstream_rx.decode_codewords(down_octets)
                    answerer.feed_line_bits(up_bits)
                    caller.feed_line_bits(down_recovered)
                    caller_received.extend(caller.take_received())
                    answer_received.extend(answerer.take_received())
                    if (bytes(caller_received) == answer_payload
                            and bytes(answer_received) == caller_payload):
                        break
                self.assertEqual(bytes(answer_received), caller_payload)
                self.assertEqual(bytes(caller_received), answer_payload)
                self.assertEqual((caller.v42_mode, answerer.v42_mode),
                                 ("lapm", "lapm"))

    def test_paired_startup_mode_matrix(self):
        """Compare V.42 detection and direct startup for native V.90 roles."""
        cases = (
            # Legacy V.90 control: without a V.92 agreement, retain the
            # ODP/ADP-detect-capable startup while still delivering payload.
            ("legacy-v90-detect-both", True, True, True, True, True),
            ("direct-both", False, False, True, False, False),
            # The V.42 originator starts directly; the answerer detects ODP.
            ("direct-originator", False, True, True, False, False),
            # Negative control: the originator waits for ADP while the
            # answerer starts LAPM directly, so the two startup procedures
            # cannot synchronize.
            ("detect-originator", True, False, False, False, False),
        )
        for (name, caller_detect, answerer_detect, should_establish,
             expect_odp, expect_adp) in cases:
            with self.subTest(startup=name):
                caller = V90LapmLink(
                    rate=28000, calling_party=True, detect=caller_detect)
                answerer = V90LapmLink(
                    rate=28000, calling_party=False, detect=answerer_detect)
                caller_payload = bytes((i * 37 + 11) & 0xFF for i in range(192))
                answer_payload = bytes((i * 91 + 3) & 0xFF for i in range(160))
                caller_rx = bytearray()
                answer_rx = bytearray()
                for _ in range(5000):
                    up = caller.read_tx_bits(64)
                    down = answerer.read_tx_bits(64)
                    answerer.feed_line_bits(up)
                    caller.feed_line_bits(down)
                    caller.take_received()
                    answerer.take_received()
                    if caller.v42_mode == answerer.v42_mode == "lapm":
                        break
                established = (
                    caller.v42_mode == answerer.v42_mode == "lapm")
                self.assertEqual(established, should_establish)
                self.assertEqual(
                    "ODP received" in answerer._v42.events, expect_odp)
                self.assertEqual(
                    "ADP received" in caller._v42.events, expect_adp)
                if not established:
                    self.assertEqual(caller._v42.stats["xid_tx"], 0)
                    self.assertEqual(answerer._v42.stats["xid_tx"], 0)
                    continue

                caller.feed_bytes(caller_payload)
                answerer.feed_bytes(answer_payload)
                for _ in range(12000):
                    up = caller.read_tx_bits(64)
                    down = answerer.read_tx_bits(64)
                    answerer.feed_line_bits(up)
                    caller.feed_line_bits(down)
                    answer_rx.extend(answerer.take_received())
                    caller_rx.extend(caller.take_received())
                    if (bytes(answer_rx) == caller_payload
                            and bytes(caller_rx) == answer_payload):
                        break
                self.assertEqual(bytes(answer_rx), caller_payload)
                self.assertEqual(bytes(caller_rx), answer_payload)
                self.assertEqual((caller.v42_mode, answerer.v42_mode),
                                 ("lapm", "lapm"))
                self.assertGreater(caller._v42.stats["xid_tx"], 0)
                self.assertGreater(caller._v42.stats["ua_rx"], 0)
                self.assertGreater(answerer._v42.stats["ua_tx"], 0)

    def test_joined_v34_upstream_b1_and_v90_downstream_deliver_exact_payload(self):
        baud = 3428.5714285714284
        carrier = 1959.1836734693877
        session = V90DigitalAnswererPhase3Session(
            baud=baud, carrier=carrier, law="pcmu", uinfo=67,
            rate_mask=0xFFFF, enable_lapm=True)
        phase4 = V90DigitalAnswererPhase4Session(
            uinfo=67, law="pcmu", constellation=4,
            mp=session.phase4_mp_profile)
        phase4.start()
        session.phase4 = phase4
        cpt = parse_cp(_frame(
            kind="cpt", drn=7, selectors=(0, 1, 2, 0, 1, 2),
            tx_masks=(tuple(range(128)),) * 3, law=0, ack=0,
            spectral_sign_bits=0))
        cp = parse_cp(_frame(
            kind="cp", drn=7, selectors=(0, 1, 2, 0, 1, 2),
            tx_masks=(tuple(range(128)),) * 3, law=0, ack=0,
            spectral_sign_bits=0))
        self.assertTrue(phase4.receive_profile(cpt))
        phase4.read_g711(50000)  # TRN2d and MP precede the selected CP.
        if phase4._output:
            phase4.read_g711(len(phase4._output))
        self.assertTrue(session.receive_phase4_profile(cp))
        phase4.read_g711(1)  # Start a complete MP′ before the caller's CP′.
        mp_prime_tail = len(phase4._output)
        self.assertGreater(mp_prime_tail, 0)
        phase4.read_g711(mp_prime_tail)
        cp_prime = {**cp, "ack": 1}
        self.assertTrue(session.receive_phase4_profile(cp_prime))
        ed = phase4.read_g711(12)
        self.assertEqual(len(ed), 12)  # Two six-octet Ed frames precede B1d.
        b1d = phase4.read_g711(288)
        self.assertEqual(len(b1d), 288)  # 48 six-octet data frames.
        self.assertEqual(phase4.state, phase4.transmitter.DATA_READY)

        upstream_profile = session.analog_upstream_profile
        self.assertEqual(upstream_profile.rate, 33600)
        upstream_rx = session.upstream_data
        upstream_tx = v34_data.DataTx(upstream_profile.rate,
                                      upstream_profile.baud)
        # V.90 §6.5 assigns the analogue upstream transmitter the GPA
        # scrambler convention used by the V.34 answerer role in our tables.
        upstream_scrambler = v34_tx.Scrambler("answer")
        caller = V90LapmLink(rate=upstream_profile.rate, calling_party=True)
        answerer = session.lapm
        self.assertIsNotNone(answerer)

        # Train the production V.34 receiver from the peer's actual B1 mapper
        # sequence, preserving its complex equalizer-output coordinate.
        b1_bits = [upstream_scrambler.step(1)
                   for _ in range(upstream_tx.bits_per_data_frame)]
        self.assertEqual(b1_bits, upstream_rx.receiver.expected)
        b1 = upstream_tx.data_frame(b1_bits)
        upstream_rx.start_b1(0)
        phase_gain = 0.71 * np.exp(0.27j)
        for index, symbol in enumerate(b1):
            upstream_rx.feed_symbol(index, phase_gain * symbol)
        self.assertEqual(upstream_rx.state, upstream_rx.DATA)

        downstream_rx = V90DigitalPhase4Receiver(cp, "pcmu")
        up_bits_per_slice = 3 * upstream_tx.bits_per_data_frame
        down_bits_per_slice = (up_bits_per_slice * 15) // 14
        self.assertEqual(up_bits_per_slice / upstream_profile.rate,
                         down_bits_per_slice / cp["rate"])
        self.assertEqual(down_bits_per_slice % (phase4.transmitter.data_encoder.s
                                                + phase4.transmitter.data_encoder.k), 0)
        frame_bits = (phase4.transmitter.data_encoder.s
                      + phase4.transmitter.data_encoder.k)
        # Ed is emitted with the continuing CPt mapper. The data receiver is
        # reset at B1d, whose CP mapper begins with fresh state.
        self.assertEqual(downstream_rx.decode_codewords(b1d), [1] * (48 * frame_bits))

        sent_upstream = []
        decoded_upstream_all = []

        def exchange(caller_received, answer_received):
            up_bits = caller.read_tx_bits(up_bits_per_slice)
            sent_upstream.extend(up_bits)
            decoded_upstream = []
            for offset in range(0, len(up_bits), upstream_tx.bits_per_data_frame):
                scrambled = [upstream_scrambler.step(bit)
                             for bit in up_bits[offset:offset + upstream_tx.bits_per_data_frame]]
                symbols = upstream_tx.data_frame(
                    scrambled)
                for symbol in symbols:
                    decoded_upstream.extend(upstream_rx.feed_symbol(
                        upstream_rx.next_symbol, phase_gain * symbol))
            decoded_upstream_all.extend(decoded_upstream)
            if decoded_upstream:
                answerer.feed_line_bits(decoded_upstream)
            answer_received.extend(answerer.take_received())

            codewords = phase4.read_g711(
                (down_bits_per_slice // frame_bits) * 6,
                data_bit_source=answerer.read_tx_bits)
            self.assertEqual(len(codewords), (down_bits_per_slice // frame_bits) * 6)
            caller.feed_line_bits(downstream_rx.decode_codewords(codewords))
            caller_received.extend(caller.take_received())

        # The link negotiation must complete through both real asymmetric
        # modem bit paths before any DTE payload is accepted.
        scratch_caller = bytearray()
        scratch_answerer = bytearray()
        for _ in range(100):
            exchange(scratch_caller, scratch_answerer)
            if caller.v42_mode == answerer.v42_mode == "lapm":
                break
        self.assertEqual((caller.v42_mode, answerer.v42_mode),
                         ("lapm", "lapm"),
                         f"caller={caller._v42.events[-12:]}, "
                         f"answerer={answerer._v42.events[-12:]}; "
                         f"upstream={upstream_rx.events[-4:]}; "
                         f"decoded {len(decoded_upstream_all)}/{len(sent_upstream)}, "
                         f"bit errors={sum(a != b for a, b in zip(sent_upstream, decoded_upstream_all))}, "
                         f"prefix sent/decoded={sent_upstream[:48]}/{decoded_upstream_all[:48]}")

        caller_payload = bytes((index * 29 + 7) & 0xFF for index in range(384))
        answer_payload = bytes((index * 53 + 11) & 0xFF for index in range(512))
        session.feed_bytes(answer_payload)
        caller_received = bytearray()
        answer_received = bytearray()
        caller_request_queued = False
        for _ in range(300):
            exchange(caller_received, answer_received)
            if not caller_request_queued and bytes(caller_received) == answer_payload:
                # Application response is causally gated by bytes received
                # through Phase 4 G.711 decode and caller LAPM.
                caller.feed_bytes(caller_payload)
                caller_request_queued = True
            if (bytes(caller_received) == answer_payload
                    and bytes(answer_received) == caller_payload):
                break
        self.assertTrue(caller_request_queued,
                        "caller response was never released by received payload")
        self.assertEqual(bytes(answer_received), caller_payload)
        self.assertEqual(bytes(caller_received), answer_payload)


if __name__ == "__main__":
    unittest.main()
