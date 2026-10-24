#!/usr/bin/env python3
"""Self-tests for the SpanDSP V.22/V.22bis binding (skipped when
libspandsp is not installed)."""

import struct
import unittest

import spandsp_v22
from spandsp_v22 import V22bisModem, _AsyncFramer


def run_pair(caller, answerer, seconds, on_tick=None):
    """Exchange 20 ms audio frames between two modems for ``seconds``."""
    rx_caller, rx_answerer = bytearray(), bytearray()
    for tick in range(int(seconds * 50)):
        a = caller.tx_samples(160)
        b = answerer.tx_samples(160)
        rx_answerer += answerer.feed_pcm(struct.pack("<160h", *a))
        rx_caller += caller.feed_pcm(struct.pack("<160h", *b))
        if on_tick:
            on_tick(tick)
    return bytes(rx_caller), bytes(rx_answerer)


class FramerTests(unittest.TestCase):
    def feed(self, bits):
        framer, out = _AsyncFramer(), []
        for bit in bits:
            out.extend(framer.push(bit))
        return out

    @staticmethod
    def char(value, stop=1):
        return [0] + [(value >> i) & 1 for i in range(8)] + ([stop] if stop is not None else [])

    def test_idle_and_back_to_back_characters(self):
        bits = [1] * 7 + self.char(0x48) + self.char(0x69) + [1] * 5 + self.char(0x00)
        self.assertEqual(self.feed(bits), [0x48, 0x69, 0x00])

    def test_deleted_stop_bit_is_accepted(self):
        bits = self.char(0x41, stop=None) + self.char(0x42) + [1, 1]
        self.assertEqual(self.feed(bits), [0x41, 0x42])


@unittest.skipUnless(spandsp_v22.available(), "libspandsp not installed")
class V22bisLoopbackTests(unittest.TestCase):
    def test_default_pair_trains_at_1200_and_exchanges_data(self):
        caller = V22bisModem(calling_party=True)
        answerer = V22bisModem(calling_party=False)
        sent = {"done": False}

        def tick(n):
            if caller.connected and answerer.connected and not sent["done"]:
                caller.feed_bytes(b"HELLO FROM CALLER\r\n")
                answerer.feed_bytes(b"HELLO FROM ANSWERER\r\n")
                sent["done"] = True

        rx_caller, rx_answerer = run_pair(caller, answerer, 12, tick)
        self.assertTrue(caller.connected and answerer.connected,
                        (caller.events, answerer.events))
        self.assertEqual((caller.bit_rate, answerer.bit_rate), (1200, 1200))
        self.assertEqual(rx_answerer, b"HELLO FROM CALLER\r\n")
        self.assertEqual(rx_caller, b"HELLO FROM ANSWERER\r\n")


@unittest.skipUnless(spandsp_v22.available(), "libspandsp not installed")
class V22bisRateTests(unittest.TestCase):
    def _run(self, caller_rate, answerer_rate):
        caller = V22bisModem(True, bit_rate=caller_rate, bell212a=False)
        answerer = V22bisModem(False, bit_rate=answerer_rate, bell212a=False)
        run_pair(caller, answerer, 15)
        return caller, answerer

    def test_2400_both_ends_settles_at_2400(self):
        caller, answerer = self._run(2400, 2400)
        self.assertEqual((caller.bit_rate, answerer.bit_rate), (2400, 2400))
        self.assertTrue(caller.is_settled() and answerer.is_settled())

    def test_2400_offer_to_1200_peer_falls_back_and_settles_at_1200(self):
        caller, answerer = self._run(2400, 1200)
        self.assertEqual(caller.bit_rate, 1200)
        # Provisional at 1200 until the retrain window has passed, then settled.
        self.assertTrue(caller.is_settled())


@unittest.skipUnless(spandsp_v22.available(), "libspandsp not installed")
class V22bisV42Tests(unittest.TestCase):
    def _pair(self):
        caller = V22bisModem(True, bit_rate=2400, bell212a=False)
        answerer = V22bisModem(False, bit_rate=2400, bell212a=False)
        return caller, answerer

    def test_default_detection_deadline_matches_v42_t400(self):
        caller, answerer = self._pair()
        # V.42 detection starts after the line modem has settled, so its
        # bit-count deadline must use the negotiated bearer rate rather than
        # the provisional 1200-bit/s rate present during V.22bis startup.
        run_pair(caller, answerer, 15)
        self.assertEqual((caller.bit_rate, answerer.bit_rate), (2400, 2400))
        caller.start_v42(True)
        self.assertEqual(caller._v42_detect_deadline, 750 * caller.bit_rate // 1000)

    def test_v42_lapm_over_a_v22bis_link_carries_data_both_ways(self):
        caller, answerer = self._pair()
        a_msg, b_msg = b"caller data " * 50, bytes(range(256)) * 2
        state = {"v42": False, "sent": False}
        got_a, got_c = bytearray(), bytearray()

        def tick(n):
            if not state["v42"] and caller.is_settled() and answerer.is_settled():
                caller.start_v42(True)
                answerer.start_v42(False)
                state["v42"] = True
            if (not state["sent"] and caller.v42_mode == "lapm"
                    and answerer.v42_mode == "lapm"):
                caller.feed_bytes(a_msg)
                answerer.feed_bytes(b_msg)
                state["sent"] = True

        rx_c, rx_a = bytearray(), bytearray()
        for t in range(50 * 40):
            fa, fb = caller.tx_samples(160), answerer.tx_samples(160)
            rx_a += answerer.feed_pcm(struct.pack("<160h", *fa))
            rx_c += caller.feed_pcm(struct.pack("<160h", *fb))
            tick(t)
        self.assertEqual((caller.v42_mode, answerer.v42_mode), ("lapm", "lapm"))
        self.assertEqual(bytes(rx_a), a_msg)
        self.assertEqual(bytes(rx_c), b_msg)

    def _answerer_with_adp(self, adp_chars, then_lapm):
        """A stand-in for a real peer: waits for our ODP, answers with the given
        ADP characters sent *back to back* (no mark bits between them, as NIST's
        modem does), and then optionally speaks LAPM. Returns (answerer, step)."""
        from spandsp_v42 import V42Link
        from v42_detect import OdpDetector, char_bits

        answerer = V22bisModem(False, bit_rate=2400, bell212a=False)
        odp = OdpDetector()
        state = {"sent": False}

        def step(decoded):
            if not state["sent"] and odp.feed(decoded):
                bits = []
                for _ in range(12):
                    for value in adp_chars:
                        bits += char_bits(value)  # contiguous: 10 bits per char
                answerer.feed_bits(bits)
                state["sent"] = True
            elif state["sent"] and then_lapm and answerer.v42_mode == "off" \
                    and answerer.pending_bits() == 0:
                answerer._v42 = V42Link(False, detect=False, bit_rate=2400)
                answerer._v42_detect_deadline = 10 ** 9      # the stand-in peer is
                answerer._v42_establish_deadline = 10 ** 9   # not subject to ours
                answerer.v42_mode = "lapm"
        return answerer, step

    def test_concurrent_transmit_and_receive_threads_do_not_corrupt_the_engine(self):
        # The session transmits from the RTP send thread and receives from the
        # main thread, and ctypes drops the GIL during C calls, so both really
        # run inside SpanDSP at once. V.42 shares one state between the two
        # sides; unlocked this segfaulted outright and, in the lab, silently
        # dropped a whole LAPM frame. Hammer both sides from separate threads.
        import threading
        import queue

        caller, answerer = self._pair()
        qa, qb = queue.Queue(), queue.Queue()
        stop, errors = threading.Event(), []
        started = {"a": False, "b": False}

        def tx(m, out):
            try:
                while not stop.is_set():
                    out.put(struct.pack("<160h", *m.tx_samples(160)))
            except Exception as exc:  # pragma: no cover
                errors.append(exc)

        def rx(m, inq):
            try:
                while not stop.is_set():
                    try:
                        m.feed_pcm(inq.get(timeout=0.1))
                    except queue.Empty:
                        pass
            except Exception as exc:  # pragma: no cover
                errors.append(exc)

        threads = [threading.Thread(target=tx, args=(caller, qb)),
                   threading.Thread(target=tx, args=(answerer, qa)),
                   threading.Thread(target=rx, args=(caller, qa)),
                   threading.Thread(target=rx, args=(answerer, qb))]
        for t in threads:
            t.start()
        import time as _time
        end = _time.time() + 4
        while _time.time() < end:
            for who, m, calling in (("a", caller, True), ("b", answerer, False)):
                if not started[who] and m.is_settled():
                    m.start_v42(calling, frame_size=32)
                    started[who] = True
            if caller.v42_mode == "lapm":
                caller.feed_bytes(b"x" * 64)
            _time.sleep(0.01)
        stop.set()
        for t in threads:
            t.join()
        self.assertEqual(errors, [])

    def test_caller_accepts_a_back_to_back_adp_and_goes_straight_to_lapm(self):
        # SpanDSP's detector wanted 8-16 marks between the ADP characters and
        # never recognised NIST's contiguous "E C E C"; the caller then gave up
        # and the peer hung up. Our own detector decodes the characters as plain
        # async, so it accepts the ADP itself and the lenient fallback in the
        # layer is no longer needed.
        caller = V22bisModem(True, bit_rate=2400, bell212a=False)
        answerer, step = self._answerer_with_adp((0x45, 0x43), then_lapm=True)
        state = {"v42": False, "sent": False}
        rx_a, rx_c = bytearray(), bytearray()
        for t in range(50 * 40):
            fa, fb = caller.tx_samples(160), answerer.tx_samples(160)
            got_a = answerer.feed_pcm(struct.pack("<160h", *fa))
            rx_c += caller.feed_pcm(struct.pack("<160h", *fb))
            step(got_a)
            if not state["v42"] and caller.is_settled() and answerer.is_settled():
                caller.start_v42(True)
                state["v42"] = True
            if not state["sent"] and caller.v42_mode == "lapm" and answerer.v42_mode == "lapm":
                caller.feed_bytes(b"hello over lapm " * 20)
                answerer.feed_bytes(b"banner from the peer\r\n")
                state["sent"] = True
            if state["sent"]:
                rx_a += got_a
        self.assertEqual(caller.v42_mode, "lapm", caller.v42_mode)
        self.assertTrue(caller._v42.adp_seen, "our detector must accept contiguous ADP characters")
        self.assertEqual(bytes(rx_a), b"hello over lapm " * 20)
        self.assertEqual(bytes(rx_c), b"banner from the peer\r\n")

    def test_caller_stops_promptly_when_the_peer_answers_no_v42(self):
        caller = V22bisModem(True, bit_rate=2400, bell212a=False)
        answerer, step = self._answerer_with_adp((0x45, 0x00), then_lapm=False)
        started_at = None
        for t in range(50 * 30):
            fa, fb = caller.tx_samples(160), answerer.tx_samples(160)
            got_a = answerer.feed_pcm(struct.pack("<160h", *fa))
            caller.feed_pcm(struct.pack("<160h", *fb))
            step(got_a)
            if started_at is None and caller.is_settled() and answerer.is_settled():
                caller.start_v42(True)
                started_at = t
            if started_at is not None and caller.v42_mode == "failed":
                self.assertLess((t - started_at) / 50, 0.8)  # well under the T400 window
                return
        self.fail(f"caller never fell back, mode={caller.v42_mode}")

    def test_a_failed_lapm_link_is_reported_lost_not_silently_raw(self):
        # If an established LAPM link dies (retries exhausted) the peer is still
        # in LAPM. Reverting to raw data would corrupt the stream silently, so
        # the modem reports the link lost and the call is dropped.
        caller, answerer = self._pair()
        caller.start_v42(True)
        caller.v42_mode = "lapm"          # as if LAPM had come up ...
        caller._v42.state = "down"        # ... and then SpanDSP reported it down
        caller._v42_check()
        self.assertEqual(caller.v42_mode, "lost")
        self.assertTrue(caller.v42_lost)
        # Nothing further is accepted for transmission, and the line idles.
        caller.feed_bytes(b"data after the link died")
        self.assertEqual(caller.pending_bits(), 0)

    def test_caller_falls_back_to_raw_when_the_peer_has_no_v42(self):
        caller, answerer = self._pair()
        started = {"v": False, "sent": False}
        rx_a = bytearray()
        for t in range(50 * 30):
            fa, fb = caller.tx_samples(160), answerer.tx_samples(160)
            rx_a += answerer.feed_pcm(struct.pack("<160h", *fa))
            caller.feed_pcm(struct.pack("<160h", *fb))
            if not started["v"] and caller.is_settled() and answerer.is_settled():
                caller.start_v42(True)  # the answerer never does
                caller.feed_bytes(b"hello raw peer\r\n")  # held during detection
                started["v"] = True
        self.assertEqual(caller.v42_mode, "failed")
        # A raw peer sees the ODP as DC1 (0x11/0x91) characters -- harmless by
        # design -- and then our data, cleanly framed.
        self.assertTrue(bytes(rx_a).endswith(b"hello raw peer\r\n"), bytes(rx_a))
        self.assertTrue(set(bytes(rx_a)[:-16]) <= {0x11, 0x91}, bytes(rx_a)[:-16])


if __name__ == "__main__":
    unittest.main()
