#!/usr/bin/env python3
"""Self-tests for the V.32 caller transmitter (pure Python, no numpy)."""

import cmath
import math
import unittest

from v32_caller import (
    GPC, NR_DIFF, POINTS, SPAN, SPS24, V32CallerTx, Scrambler, rate_word, rrc_pulse,
)


def decode_dibits(symbols):
    """Fisher's 4800 bps decoder: quadrant slicer + differential table."""
    q, bits = 0, []
    for v in symbols:
        q = ((q << 1) | int(2.0 * v.imag > -v.real)) & 0xFF
        q = ((q << 1) | int(v.real > v.imag)) & 0xFF
        d = NR_DIFF[q & 0xF]
        bits += [(d >> 1) & 1, d & 1]
    return bits


def _reference_tx_samples(tx, count):
    """Original direct pulse sum, retained as a sample-level parity oracle."""
    out = []
    gs, gy = tx._g_sample, tx._g_symbol
    for _ in range(count):
        n = tx._n
        lo = -(-(gs*n - tx._span*gy)//gy)
        hi = (gs*n + tx._span*gy)//gy
        acc = 0j
        for k in range(max(lo, 0), hi+1):
            v = tx._symbol(k)
            if v:
                acc += v * tx._pulse[tx._centre + gs*n - gy*k]
        theta = 2*math.pi*tx.carrier*n/tx.fs
        out.append(round(acc.real*math.cos(theta) - acc.imag*math.sin(theta)))
        tx._n += 1
    horizon = gs*tx._n//gy - tx._span - 2
    for k in [k for k in tx._sym if k < horizon]:
        del tx._sym[k]
    return out


class PolyphaseParityTests(unittest.TestCase):
    def test_exact_samples_state_and_payload_across_all_rates_and_grids(self):
        for fs in (8000, 9600):
            for rate in (4800, 7200, 9600, 12000, 14400):
                with self.subTest(fs=fs, rate=rate):
                    a, b = [V32CallerTx(sample_rate=fs, my_rates=(rate,), trn_length=320)
                            for _ in range(2)]
                    def compare(count):
                        self.assertEqual(a.tx_samples(count), _reference_tx_samples(b, count))
                        self.assertEqual(a._k_next, b._k_next)
                        self.assertEqual(a.state, b.state)
                        self.assertEqual(a._tx_bytes, b._tx_bytes)
                    for obj in (a, b):
                        obj.start_ac()
                    compare(319)
                    for obj in (a, b):
                        obj.reverse_ac()
                    compare(160)
                    for obj in (a, b):
                        obj.stop()
                    compare(63)
                    for obj in (a, b):
                        obj.start_training_now()
                        obj.feed_bytes(bytes(range(256)))
                        obj.request_e(rate)
                    for chunk in (1, 63, 160, 319, 160, 8000):
                        compare(chunk)


class ConstellationTests(unittest.TestCase):
    def test_v32bis_tables_are_complete_symmetric_partitioned_constellations(self):
        import itertools
        import math

        from v32_caller import TCM_BITS, V32_TCM
        for rate, nbits in TCM_BITS.items():
            table = V32_TCM[rate]
            self.assertEqual(len(table), 2 ** (nbits + 1))                   # + the redundant bit
            self.assertEqual(len(set(table.values())), len(table))
            power = sum(abs(z) ** 2 for z in table.values()) / len(table)
            self.assertAlmostEqual(power, 10.0, places=6)                    # same power as the training
            pts = set((round(z.real, 6), round(z.imag, 6)) for z in table.values())
            self.assertTrue(all((round(-y, 6), round(x, 6)) in pts for x, y in pts))      # 90 deg symmetric
            subsets = {}
            for key, z in table.items():
                subsets.setdefault(key[:3], []).append(z)
            self.assertEqual({len(v) for v in subsets.values()}, {2 ** (nbits - 2)})
            dmin = min(abs(a - b) for a, b in itertools.combinations(table.values(), 2))
            dsub = min(abs(a - b) for v in subsets.values() for a, b in itertools.combinations(v, 2))
            self.assertAlmostEqual(dsub / dmin, 2 * math.sqrt(2), places=3)            # set partitioning


class PulseTests(unittest.TestCase):
    def test_rrc_is_symmetric_and_roughly_unit_energy_at_symbol_spacing(self):
        p = rrc_pulse(0.25)
        self.assertEqual(len(p), 2 * SPAN * SPS24 + 1)
        for j in range(len(p)):
            self.assertAlmostEqual(p[j], p[-1 - j], places=9)
        # unit energy as an integral (an RRC is not Nyquist, so symbol-spaced
        # samples do not sum to 1): sum over the 24 kHz grid, 10 points per symbol
        energy = sum(v * v for v in p) / SPS24
        self.assertAlmostEqual(energy, 1.0, delta=0.03)


class SymbolStreamTests(unittest.TestCase):
    def test_training_layout_and_rate_signal_round_trip(self):
        tx = V32CallerTx(my_rates=(9600,))
        tx.start_training()
        gen = tx._training()
        syms = [next(gen) for _ in range(256 + 16 + tx.trn_length + 3000)]
        # S: ABAB..., S-bar: CDCD...
        self.assertEqual(syms[:4], [POINTS[0], POINTS[1], POINTS[0], POINTS[1]])
        self.assertEqual(syms[256:260], [POINTS[3], POINTS[2], POINTS[3], POINTS[2]])
        # first 256 TRN symbols use only A/C
        self.assertTrue(all(s in (POINTS[0], POINTS[3]) for s in syms[272:528]))
        self.assertEqual({s for s in syms[528:2000]}, set(POINTS))
        # the rate signal after TRN decodes (GPC descrambler) to the repeated rate word
        rate_syms = syms[256 + 16 + tx.trn_length:]
        scr, bits = Scrambler(GPC), []
        # the receiver's descrambler has been running since the start of TRN in
        # real use; here resynchronise it on the received bits (it self-syncs in 23)
        bits = [scr.rev(b) for b in decode_dibits(rate_syms)]
        word = rate_word(9600)
        self.assertEqual(word, 0x0B91)
        found = False
        for align in range(16):
            words = [int("".join(map(str, bits[align + 16 * i: align + 16 * i + 16])), 2)
                     for i in range((len(bits) - align) // 16 - 1)]
            tail = words[10:]
            if tail and all(w == word for w in tail[:100]):
                found = True
                break
        self.assertTrue(found, "decoded rate signal is not the repeated rate word")

    def test_rate_word_follows_the_trn_immediately(self):
        # measured from the Smart Link caller: no sync bit, word aligned to the TRN end
        tx = V32CallerTx(my_rates=(9600,), s_length=336, trn_length=8192)
        tx.start_training()
        gen = tx._training()
        syms = [next(gen) for _ in range(336 + 16 + 8192 + 600)]
        scr = Scrambler(GPC)
        # descramble everything from the start: TRN symbols are raw dibits (index = (b1<<1)|b2)
        for s in syms[336 + 16: 336 + 16 + 8192]:
            k = POINTS.index(s)
            scr.rev((k >> 1) & 1), scr.rev(k & 1)
        q, bits = POINTS.index(syms[336 + 16 + 8192 - 1]), []
        for s in syms[336 + 16 + 8192:]:
            k = POINTS.index(s)
            d = NR_DIFF[(q << 2) | k]
            q = k
            bits += [scr.rev((d >> 1) & 1), scr.rev(d & 1)]
        self.assertEqual(int("".join(map(str, bits[:16])), 2), 0x0B91)
        self.assertEqual(int("".join(map(str, bits[16:32])), 2), 0x0B91)

    def test_e_word_is_sent_once_when_requested(self):
        tx = V32CallerTx(my_rates=(4800,), trn_length=100)
        tx.start_training()
        gen = tx._training()
        first = [next(gen) for _ in range(256 + 16 + 100 + 200)]
        self.assertEqual(tx.state, "rate")
        tx.request_e()
        more = [next(gen) for _ in range(400)]
        self.assertEqual(tx.state, "e")
        scr = Scrambler(GPC)
        for s in first[256 + 16: 256 + 16 + 100]:
            k = POINTS.index(s)
            scr.rev((k >> 1) & 1), scr.rev(k & 1)
        q, bits = POINTS.index(first[256 + 16 + 100 - 1]), []
        for s in first[256 + 16 + 100:] + more:
            k = POINTS.index(s)
            d = NR_DIFF[(q << 2) | k]
            q = k
            bits += [scr.rev((d >> 1) & 1), scr.rev(d & 1)]
        words = [int("".join(map(str, bits[16 * i:16 * i + 16])), 2) for i in range(len(bits) // 16)]
        self.assertEqual(words.count(0xF000 | rate_word(4800)), 1)
        after = words[words.index(0xF000 | rate_word(4800)) + 1:]
        self.assertTrue(after and all(w == 0xFFFF for w in after[:5]), "ones follow E")


class AudioTests(unittest.TestCase):
    def test_audio_demodulates_to_the_intended_symbols(self):
        tx = V32CallerTx(amplitude=900.0)
        tx.start_training()
        audio = tx.tx_samples(8000 // 2)          # 0.5 s = 1200 symbols
        n_sym = 1100
        pulse = tx._pulse
        centre = tx._centre

        def matched(k):
            acc = 0j
            for n in range(max(0, (10 * k - SPAN * SPS24) // 3), min(len(audio), (10 * k + SPAN * SPS24) // 3 + 1)):
                j = 3 * n - 10 * k
                if -SPAN * SPS24 <= j <= SPAN * SPS24:
                    theta = 2.0 * math.pi * 1800.0 * n / 8000
                    acc += audio[n] * 2 * cmath.exp(-1j * theta) * pulse[centre + j]
            return acc * 3.0 / 10 / 900.0

        expected = []
        gen = V32CallerTx(amplitude=1.0)
        gen.start_training()
        g = gen._training()
        expected = [next(g) for _ in range(n_sym)]
        got = [matched(k) for k in range(40, n_sym - 40)]
        errs = [abs(got[i] - expected[40 + i]) for i in range(len(got))]
        self.assertLess(sum(errs) / len(errs), 0.15, "demodulated symbols are not close to the intended ones")
        decisions = [min(range(4), key=lambda i: abs(v - POINTS[i])) for v in got]
        want = [min(range(4), key=lambda i: abs(v - POINTS[i])) for v in expected[40:n_sym - 40]]
        self.assertEqual(decisions, want)

    def test_output_level_is_close_to_the_answerers(self):
        tx = V32CallerTx(amplitude=900.0)
        tx.start_training()
        audio = tx.tx_samples(8000)
        # skip the S segment (alternating symbols are not the typical power); use TRN
        tx2 = V32CallerTx(amplitude=900.0)
        tx2.start_training()
        tx2.tx_samples(8000 * 5 // 10 + 400)       # run past S/S-bar
        trn = tx2.tx_samples(4000)
        rms = math.sqrt(sum(x * x for x in trn) / len(trn))
        self.assertAlmostEqual(rms, 2027, delta=250)

    def test_silent_until_training_starts(self):
        tx = V32CallerTx()
        self.assertEqual(set(tx.tx_samples(800)), {0})
        tx.start_training()
        self.assertNotEqual(set(tx.tx_samples(800)), {0})


if __name__ == "__main__":
    unittest.main()
