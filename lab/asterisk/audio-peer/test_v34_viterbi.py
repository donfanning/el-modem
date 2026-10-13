#!/usr/bin/env python3
"""Viterbi detector: must reproduce the transmitted symbols, with noise, and beat symbol-by-symbol slicing."""
import random
import unittest

try:
    import numpy as np
except ImportError:
    np = None

if np is not None:
    import v34_data as d
    import v34_viterbi as vv


def run(rate, baud, noise, seed=1, frames=6):
    tx = d.DataTx(rate, baud, True)
    rnd = random.Random(seed)
    sy = []
    for _ in range(frames):
        sy += tx.data_frame([rnd.randrange(2) for _ in range(tx.bits_per_data_frame)])
    L = 4 * tx.M * 2 ** tx.q
    det = vv.TrellisDetector(L, tx.J, tx.P)
    g = np.random.default_rng(seed)
    rxs = [s + complex(g.normal(0, noise), g.normal(0, noise)) for s in sy]
    for i in range(0, len(rxs), 2):
        det.feed(rxs[i], rxs[i + 1])
    det.flush()
    out = [p for pair in det.decided for p in pair]
    errs_v = sum(1 for a, b in zip(out, sy) if abs(a - b) > 1e-9)
    pts = det.pts
    errs_h = sum(1 for r, s in zip(rxs, sy) if abs(pts[int(np.argmin(np.abs(pts - r)))] - s) > 1e-9)
    return errs_v, errs_h, len(sy)


@unittest.skipIf(np is None, "numpy not installed")
class ViterbiTests(unittest.TestCase):
    def test_trace_survives_flush_without_changing_decisions(self):
        tx=d.DataTx(9600,3200,True)
        symbols=tx.data_frame([i%2 for i in range(tx.bits_per_data_frame)])
        ordinary=vv.TrellisDetector(4*tx.M*2**tx.q,tx.J,tx.P,depth=2)
        traced=vv.TrellisDetector(4*tx.M*2**tx.q,tx.J,tx.P,depth=2,trace=True)
        for i in range(0,len(symbols),2):
            for detector in (ordinary,traced):detector.feed(*symbols[i:i+2])
        ordinary.flush();traced.flush()
        self.assertEqual(ordinary.decided,traced.decided)
        self.assertEqual(traced.hist,[])
        self.assertEqual(len(traced.trace),len(symbols)//2)
        self.assertIsNone(ordinary.trace)
        for i,snapshot in enumerate(traced.snapshots):
            self.assertEqual(snapshot['interval'],i)
            np.testing.assert_array_equal(snapshot['metric'],traced.trace[i]['metric_after'])
            self.assertGreaterEqual(snapshot['decision_interval'],i)

    def test_trace_metrics_match_exhaustive_point_pair_search(self):
        # Exhaustive point pairs avoid the production class/parity compression
        # and incoming-transition indexing. Use the actual feedback domain.
        rng=np.random.default_rng(260)
        det=vv.TrellisDetector(16,8,12,precoder=[.2+.1j,-.1j,.05],
                              nonlinear=True,energy=20,trace=True)
        for _ in range(12):
            det.feed(complex(*rng.normal(0,4,2)),complex(*rng.normal(0,4,2)))
            h=det.trace[-1];expected=np.full(16,np.inf)
            labels=[[d.subset_label(u+h['c'+str(i)]) for u in det.pts] for i in (0,1)]
            distances=[]
            for i in (0,1):
                x=det.pts+h['delta'+str(i)];z=.3125*abs(x)**2/20
                distances.append(abs(x*(1+z/6+z*z/120)-h['wave'+str(i)])**2)
            for st in range(16):
                a,b,c,e=[(st>>k)&1 for k in (3,2,1,0)]
                U=e^h['inversion_bit']^d.modulo_bit(h['c0'],h['c1'])
                for i in range(16):
                    for j in range(16):
                        if ((i%4)&1)^((j%4)&1)!=U:continue
                        nib=d.TABLE13[labels[0][i]][labels[1][j]]
                        y2,y1=(nib>>1)&1,nib&1
                        nxt=(e<<3)|((a^y2^e)<<2)|((b^y2)<<1)|(c^y1)
                        cost=h['metric_before'][st]+distances[0][i]+distances[1][j]
                        expected[nxt]=min(expected[nxt],cost)
            expected-=expected.min()
            np.testing.assert_allclose(h['metric_after'],expected,atol=1e-12,rtol=1e-12)

    def test_noise_free_is_exact(self):
        for rate, baud in ((4800, 2400), (9600, 3200), (14400, 3429)):
            ev, eh, n = run(rate, baud, 0.0)
            self.assertEqual((ev, eh), (0, 0), (rate, baud))

    def test_viterbi_makes_fewer_errors_than_slicing_in_noise(self):
        ev, eh, n = run(9600, 3200, 0.55)
        self.assertGreater(eh, 10)
        self.assertLess(ev, eh * 0.6)


if __name__ == "__main__":
    unittest.main()
