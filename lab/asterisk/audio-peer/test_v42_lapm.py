#!/usr/bin/env python3
"""Our V.42 (v42_lapm.V42Link): against itself and against SpanDSP's V.42 as an independent reference peer."""
import os
import random
import unittest

import v42_lapm
from v42bis import Encoder


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


def pair(**kw):
    detect = kw.pop("detect", True)
    comp = kw.pop("compression", None)
    return (v42_lapm.V42Link(True, detect=detect, bit_rate=9600, compression=comp, **kw),
            v42_lapm.V42Link(False, detect=detect, bit_rate=9600, compression=comp, **kw))


class LinkTests(unittest.TestCase):
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

    def test_compression_is_dropped_when_the_responder_does_not_offer_it(self):
        a = v42_lapm.V42Link(True, bit_rate=9600, compression=(3, 1024, 8))
        b = v42_lapm.V42Link(False, bit_rate=9600)
        a.write(b"plain text\r\n")
        _, gb = pump(a, b, 9600 * 3)
        self.assertIsNone(a.compression)
        self.assertEqual(gb, b"plain text\r\n")

    def test_xid_round_trip(self):
        info = v42_lapm.build_xid(64, 8, (3, 1024, 8))
        self.assertEqual(v42_lapm.parse_xid(info), {"tx_n401": 64, "rx_n401": 64, "tx_k": 8, "rx_k": 8,
                                                    "p0": 3, "p1": 1024, "p2": 8})

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

    def test_v42bis_across_implementations(self):
        for ours_calls in (True, False):
            a, b = self.run_case(ours_calls, True, (3, 1024, 8))
            self.assertEqual(a.compression, (3, 1024, 8))



class XidIgnoringAnswerer(v42_lapm.V42Link):
    """An answerer that does not implement XID negotiation: it never answers an XID command but accepts SABME (V.42 7.2.2 makes XID optional)."""

    def _answer_xid(self, x):
        self.ignored_xids = getattr(self, "ignored_xids", 0) + 1


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
        a = v42_lapm.V42Link(True, detect=True, bit_rate=9600)
        b = answerer_class(False, detect=True, bit_rate=9600)
        a.write(b"hello from the originator\r\n")
        old = dict(os.environ)
        os.environ.update(env or {})
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

    def test_the_old_behaviour_is_a_deadlock_and_stays_available(self):
        a, b, gb = self.run_pair(XidIgnoringAnswerer, env={"V42_XID_FALLBACK": "0"})
        self.assertEqual(a.state, "idle")
        self.assertEqual(gb, b"")
        self.assertGreater(a.stats["xid_tx"], v42_lapm.XID_ATTEMPTS)  # still repeating the XID
        self.assertFalse(a._sabme_sent)

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
