#!/usr/bin/env python3
"""Tests for the streaming V.32 modem (v32_stream.py), driven by the offline simulator in
v32_sim_rx.py: our own transmitter plays the answerer's second transmission through a channel
(clock offset, carrier offset, noise) into the receive state machine."""

import math
import unittest

import v32_sim_rx as sim
from v32_stream import Baseband, EchoCanceller, _AsyncDeframer, find_echo_delay


def _decoded(m, got, text):
    return m.connected and got.count(text[:50]) >= 1


class ChannelTests(unittest.TestCase):
    def test_clean_9600(self):
        m, got, text = sim.run(quiet=True, debug=False)
        self.assertTrue(_decoded(m, got, text))
        self.assertEqual(m.bit_rate, 9600)

    def test_all_rates_down_to_4800(self):
        for rate in (14400, 12000, 7200, 4800):
            m, got, text = sim.run(rate=rate, quiet=True, debug=False)
            self.assertTrue(_decoded(m, got, text), rate)
            self.assertEqual(m.bit_rate, rate)

    def test_clock_offset_100ppm_both_ways(self):
        for ppm in (100, -100):
            m, got, text = sim.run(ppm=ppm, quiet=True, debug=False)
            self.assertTrue(_decoded(m, got, text), ppm)

    def test_carrier_offset_up_to_10hz(self):
        for df in (-10.0, -2.0, 5.0, 10.0):
            m, got, text = sim.run(df=df, quiet=True, debug=False)
            self.assertTrue(_decoded(m, got, text), df)

    def test_noise_about_26db_snr(self):
        m, got, text = sim.run(noise=100, quiet=True, debug=False)
        self.assertTrue(_decoded(m, got, text))

    def test_long_call_keeps_lock_with_clock_offset(self):
        # a second data burst 25 s after the first (idle marks between) must still decode
        m, got, text = sim.run(ppm=100, quiet=True, debug=False, idle_s=25)
        self.assertEqual(got.count(text[:50]), 8)


class PartsTests(unittest.TestCase):
    def test_trn_locator_matches_scalar_candidates_after_buffer_trimming(self):
        import cmath
        import random
        from v32_stream import V32Modem, BAUD
        from v32_rx import expected_trn
        from v32_caller import GPA, GPC
        rnd = random.Random(807)
        for base, tap, phase in ((0, GPA, 0), (901, GPC, 1)):
            m = V32Modem(use_v8=False, echo=False)
            m.ubase = base
            m.u = [complex(rnd.gauss(0, 1), rnd.gauss(0, 1)) for _ in range(2800)]
            ref = expected_trn(300, tap)
            k = (base+1200)//2
            for j, r in enumerate(ref):
                m.u[2*(k+j)+phase-base] += 20*r*cmath.exp(2j*math.pi*5*j/BAUD)
            predicted = k/BAUD
            window = .1
            got = m._locate(predicted, window, tap)
            best = None
            lo = max((base+4)//2, int(predicted*2*BAUD)//2-int(window*BAUD))
            hi = int(predicted*2*BAUD)//2+int(window*BAUD)
            for ph in (0, 1):
                for start in range(lo, hi):
                    i = 2*start+ph-base
                    if i < 0 or i+600 > len(m.u):continue
                    cs, energy = [], 0.
                    for b in range(6):
                        corr = 0j
                        for r, z in zip(ref[b*50:(b+1)*50], m.u[i+100*b:i+100*(b+1):2]):
                            corr += r.conjugate()*z
                            energy += z.real*z.real+z.imag*z.imag
                        cs.append(corr)
                    score = sum(abs(c) for c in cs)/(math.sqrt(energy*3000.)+1e-9)
                    if best is None or score > best[0]:best = (score, start, ph, cs)
            acc = sum(best[3][j+1]*best[3][j].conjugate() for j in range(5))
            self.assertEqual(got[1:3], best[1:3])
            self.assertAlmostEqual(got[0], best[0], places=13)
            self.assertAlmostEqual(got[3], cmath.phase(acc)*BAUD/(2*math.pi*50), places=11)

    def test_baseband_recovers_a_carrier_tone_at_two_samples_per_symbol(self):
        fs = 8000
        x = [round(1000 * math.cos(2 * math.pi * 1800 * n / fs + 0.7)) for n in range(4000)]
        bb = Baseband(fs)
        outs = [v for _, v in bb.feed(x)]
        steady = outs[40:-40]
        mag = [abs(v) for v in steady]
        self.assertAlmostEqual(sum(mag) / len(mag), 1000, delta=10)
        self.assertLess(max(mag) - min(mag), 10)

    def test_echo_canceller_removes_a_delayed_copy(self):
        import random
        rnd = random.Random(1)
        ec = EchoCanceller(taps=16)
        ec.configure(50)
        tx = [complex(rnd.gauss(0, 1000), rnd.gauss(0, 1000)) for _ in range(6000)]
        res_pow = ref_pow = 0.0
        for m in range(6000):
            ec.push_tx(tx[m])
            echo = 0.3 * tx[m - 52] if m >= 52 else 0j
            out = ec.process(echo, m, 0.2)
            if m > 5000:
                res_pow += abs(out) ** 2
                ref_pow += abs(echo) ** 2
        self.assertLess(res_pow / ref_pow, 0.01)          # at least 20 dB

    def test_echo_delay_search(self):
        import random
        rnd = random.Random(2)
        tx = [complex(rnd.gauss(0, 1), rnd.gauss(0, 1)) for _ in range(3000)]
        rx = [0j] * 230 + [0.2 * v for v in tx]
        rx = [a + complex(rnd.gauss(0, 0.2), rnd.gauss(0, 0.2)) for a in rx]
        lag, ratio = find_echo_delay(tx[:1400], rx, 400)
        self.assertEqual(lag, 230)

    def test_async_deframer(self):
        d = _AsyncDeframer()
        bits = []
        for b in b"Hi!":
            bits += [0] + [(b >> i) & 1 for i in range(8)] + [1] + [1] * 3
        out = d.push([1] * 7 + bits[:11]) + d.push(bits[11:])
        self.assertEqual(out, b"Hi!")


@unittest.skipUnless(__import__("spandsp_v8").available(), "libspandsp not installed")
class V8StageTests(unittest.TestCase):
    def test_engine_answers_an_ansam_with_cm_and_completes_v8(self):
        import struct

        import spandsp_v8 as v8
        from v32_stream import V32Modem

        answerer = v8.V8Negotiator(False, modulations=v8.V8_MOD_V32 | v8.V8_MOD_V34 | v8.V8_MOD_V22)
        m = V32Modem(fs=8000, debug=True)
        for _ in range(1500):
            m.feed_pcm(struct.pack("<160h", *answerer.tx(160)))
            answerer.rx(m.tx_samples(160))
            if answerer.status == 2:
                for _ in range(20):                      # our caller still owes its own 75 ms of silence after CJ
                    m.feed_pcm(struct.pack("<160h", *answerer.tx(160)))
                    answerer.rx(m.tx_samples(160))
                break
        text = " | ".join(e for _, e in m.events)
        self.assertEqual(answerer.status_name, "v8-call")
        self.assertIn("V.8: v8-call", text)
        self.assertIn("V.8 joint menu", text)
        self.assertNotEqual(m.state, "failed")

    def test_without_v8_offer_the_engine_stays_silent(self):
        import struct

        from v32_stream import V32Modem

        m = V32Modem(fs=8000)
        out = []
        for _ in range(100):
            m.feed_pcm(struct.pack("<160h", *([0] * 160)))
            out += m.tx_samples(160)
        self.assertFalse(any(out))


if __name__ == "__main__":
    unittest.main()
