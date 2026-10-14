#!/usr/bin/env python3
"""V.22 / V.22bis (v22.V22Modem): our modem against itself and against SpanDSP's as an independent reference."""
import random
import struct
import time
import unittest

try:
    import numpy  # noqa: F401
except ImportError:
    numpy = None

if numpy is not None:                     # v22.py's signal chain is built on numpy
    import v22


def link(a, b, seconds=12.0, delay=40, noise=0.0, ppm=0.0, seed=1, until=None):
    # random.Random rather than numpy: this suite must run on a bare checkout, and the noise
    # source does not need numpy's determinism guarantees to be reproducible.
    rng = random.Random(seed)
    to_a, to_b = [0] * delay, [0] * delay
    got_a, got_b = bytearray(), bytearray()
    for tick in range(int(seconds * 50)):
        sa, sb = a.tx_samples(160), b.tx_samples(160)
        to_a += sb
        to_b += sa
        xa, xb = to_a[:160], to_b[:160]
        del to_a[:160], to_b[:160]
        if noise:
            xa = [v + rng.gauss(0, noise) for v in xa]
            xb = [v + rng.gauss(0, noise) for v in xb]
        got_a += a.feed_pcm(struct.pack("<160h", *[int(max(-32768, min(32767, v))) for v in xa]))
        got_b += b.feed_pcm(struct.pack("<160h", *[int(max(-32768, min(32767, v))) for v in xb]))
        if until and until(a, b, tick):
            break
    return bytes(got_a), bytes(got_b)


@unittest.skipIf(numpy is None, "numpy not installed")
class SelfTests(unittest.TestCase):
    def test_transmitted_dbm0_matches_measured_pcm_power(self):
        # Independent physical reference: full-scale sine = +3.14 dBm0.
        # Measure generated random-data audio, rather than just its gain field.
        import math
        import numpy as np
        for calling in (True, False):
            for requested in (-20.0, -10.0):
                with self.subTest(calling=calling, requested=requested):
                    m = v22.V22Modem(calling, 2400, tx_power_dbm0=requested)
                    try:
                        m._actions.clear()
                        m._set_tx("data", 2400)
                        rng = random.Random(882)
                        m.feed_bits([rng.randrange(2) for _ in range(24000)])
                        pcm = np.asarray(m.tx_samples(80000), dtype=float)[8000:72000]
                        measured = 10 * math.log10(float(np.mean(pcm ** 2)) /
                                                   (32768 ** 2 / 2)) + 3.14
                        self.assertAlmostEqual(measured, requested, delta=0.12)
                        self.assertLess(float(np.max(np.abs(pcm))), 32767)
                    finally:
                        m.close()

    def test_default_transmit_level_meets_v2_one_minute_mean_cap(self):
        # V.2 section 2.3(i): one-minute mean power shall not exceed -13 dBm0 (switched network).
        import math
        m = v22.V22Modem(True, 2400)
        try:
            default_dbm0 = 20 * math.log10(m.rms / v22._ZERO_DBM0_RMS)
            self.assertLessEqual(default_dbm0, -13.0)
        finally:
            m.close()

    def test_receive_dbm0_uses_pcm_reference_and_explicit_rms_is_preserved(self):
        m = v22.V22Modem(True, 2400, rms=1234.0)
        try:
            self.assertEqual(m.rms, 1234.0)
            m._rb.power = (32768 ** 2 / 2) * 10 ** ((-18.0 - 3.14) / 10)
            self.assertAlmostEqual(m.rx_signal_power_dbm0, -18.0)
            m._rb.power = 0
            self.assertEqual(m.rx_signal_power_dbm0, -99.0)
        finally:
            m.close()

    def test_scrambler_roundtrip_and_constellation(self):
        s, d = v22.Scrambler(), v22.Descrambler()
        bits = [(i * 7 + (i >> 3)) & 1 for i in range(500)]
        out = [d.step(s.step(b)) for b in bits]
        self.assertEqual(out[20:], bits[20:])
        self.assertEqual(len(v22.POINTS), 16)
        self.assertAlmostEqual(sum(abs(p[2]) ** 2 for p in v22.POINTS) / 16, 10)

    def test_handshake_and_data_at_2400(self):
        a = v22.V22Modem(True, 2400)
        b = v22.V22Modem(False, 2400)
        sent = {"done": False}

        def until(a, b, tick):
            if a.is_settled() and b.is_settled() and not sent["done"]:
                a.feed_bytes(b"hello from the caller\r\n")
                b.feed_bytes(b"hello from the answerer\r\n")
                sent["done"] = True
            return False

        ga, gb = link(a, b, seconds=14, until=until)
        self.assertEqual((a.bit_rate, b.bit_rate), (2400, 2400))
        self.assertEqual(gb, b"hello from the caller\r\n")
        self.assertEqual(ga, b"hello from the answerer\r\n")


def _exchange(a, b, seconds=16):
    sent = {"done": False}

    def until(a, b, tick):
        if a.is_settled() and b.is_settled() and not sent["done"]:
            a.feed_bytes(b"from caller\r\n")
            b.feed_bytes(b"from answerer\r\n")
            sent["done"] = True
        return False

    return link(a, b, seconds=seconds, until=until)


@unittest.skipIf(numpy is None, "numpy not installed")
class SpandspInterop(unittest.TestCase):
    """SpanDSP is only a test reference; skipped where its library is not built."""

    def setUp(self):
        try:
            import spandsp_v22
            self.ref = spandsp_v22.V22bisModem
            self.ref(True, 1200)
        except Exception as exc:                                  # pragma: no cover
            self.skipTest(f"SpanDSP reference unavailable: {exc}")

    def _run(self, ours_calls, rate):
        ours = v22.V22Modem(ours_calls, rate)
        ref = self.ref(not ours_calls, rate)
        a, b = (ours, ref) if ours_calls else (ref, ours)
        ga, gb = _exchange(a, b)
        self.assertEqual((ours.bit_rate, ref.bit_rate), (rate, rate))
        got_ours, got_ref = (ga, gb) if ours_calls else (gb, ga)
        self.assertEqual(got_ours, b"from answerer\r\n" if ours_calls else b"from caller\r\n")
        self.assertEqual(got_ref, b"from caller\r\n" if ours_calls else b"from answerer\r\n")

    def test_ours_calls_2400(self): self._run(True, 2400)
    def test_ours_answers_2400(self): self._run(False, 2400)
    def test_ours_calls_1200(self): self._run(True, 1200)
    def test_ours_answers_1200(self): self._run(False, 1200)


if __name__ == "__main__":
    unittest.main()
