"""Cross-check production encoding against an independent ITU-derived pipeline."""
import unittest
import numpy as np
from v34_b1_reference import build, quantize, shell_unrank, subset, scrambled_ones
from v34_data import DataTx, ShellMapper, precoder_from_mp
from v34_tx import Scrambler

class B1OracleTests(unittest.TestCase):
    def test_literal_scrambler_prefixes_and_switching_pattern(self):
        self.assertEqual(scrambled_ones(24,'call'),[int(c) for c in '1'*18+'0'*5+'1'])
        self.assertEqual(scrambled_ones(24,'answer'),[int(c) for c in '111110000011111000001110'])
        # The worked example printed in 8.2: 19200 at 3000 baud, SWP=0x0421.
        ref=build(19200,3000,'call',trace=True)
        pattern=[int(ref['alignment'][4*i]['frame_high']) for i in range(15)]
        self.assertEqual(''.join(map(str,pattern)),'000010000100001')
        for baud,rate in ((2400,9600),(24000/7,33600)):
            tx=DataTx(rate,baud)
            self.assertEqual([tx._v0(m) for m in (0,2*tx.P,4*tx.P,6*tx.P)],[1,0,0,1])

    def test_literal_figure_nine_grid(self):
        # Read directly from Figure 9, printed p.23; rows Im=3,1,-1,-3,
        # columns Re=-3,-1,1,3. This does not reproduce a label formula.
        rows=((1,6,5,2),(4,3,0,7),(5,2,1,6),(0,7,4,3))
        for imaginary,row in zip((3,1,-1,-3),rows):
            for real,label in zip((-3,-1,1,3),row):
                self.assertEqual(subset(complex(real,imaginary)),label)

    def test_worked_first_four_intervals(self):
        # 2400 bit/s at 2400 baud: N=96, P=12, b=8, K=q=0.
        # First eight call-scrambler bits are 1. Figure 5 gives v=1+j.
        # Equation 9-25 gives Z=1,2,3,0; last-frame Table 12 starts V0=1.
        # Figure 9/Table 13 give labels 3,5,7,1. Figure 10 gives these
        # delay outputs, worked directly from its feedback/XOR diagram.
        ref=build(2400,2400,'call',trace=True)
        rows=ref['alignment'][:4]
        self.assertEqual([r['Z'] for r in rows],[1,2,3,0])
        self.assertEqual([r['V0'] for r in rows],[1,0,0,0])
        self.assertEqual([r['Y4Y3Y2Y1'] for r in rows],[3,5,7,1])
        self.assertEqual([r['state_after'] for r in rows],
                         [[0,1,1,1],[1,1,1,0],[0,0,0,0],[0,0,0,1]])
        self.assertEqual(ref['u'][:8],[1-1j,1+1j,-1-1j,1-1j,-1+1j,1-1j,1+1j,-1-1j])
        self.assertEqual(rows[0]['pair_bit_span'],[0,2])
        self.assertEqual(rows[0]['pair_bits'],[1,1])

    def test_trace_preserves_domains_and_inversion_placement(self):
        for baud,rate,j,p in ((2400,9600,7,12),(24000/7,33600,8,15)):
            for role in ('call','answer'):
                ref=build(rate,baud,role,trace=True)
                plain=build(rate,baud,role)
                for key in plain:self.assertEqual(ref[key],plain[key])
                rows=ref['alignment']
                self.assertEqual([(r['interval_4d'],r['V0']) for r in rows if r['inversion_index'] is not None],[(0,1),(2*p,0)])
                self.assertEqual([rows[0]['inversion_index'],rows[2*p]['inversion_index']],[2*j-2,2*j-1])
                for row in rows:
                    self.assertEqual(row['U0'],row['Y0']^row['C0']^row['V0'])
                    self.assertEqual(row['x'],[[y[0]-a[0],y[1]-a[1]] for y,a in zip(row['y'],row['predictor'])])

    def test_shell_order(self):
        for m in (2,3,5):
            production=ShellMapper(m)
            for rank in range(min(m**8,1024)):
                self.assertEqual(shell_unrank(m,rank),production.indices(rank))

    def test_quantization_ties(self):
        self.assertEqual(quantize(3+5j,2),2+4j)
        self.assertEqual(quantize(-3-5j,2),-2-4j)

    def test_state_search_at_lower_rate_uses_only_calibration_half(self):
        from v34_receive import calibration_scores, calibrate
        states=[tuple((s>>k)&1 for k in range(4)) for s in range(16)]
        refs=np.array([build(24000,24000/7,'answer',True,initial_state=state)['x'] for state in states])
        x=refs[2];energy=750.;z=.3125*abs(x)**2/energy
        wave=.7*np.exp(.3j)*x*(1+z/6+z*z/120)
        observed=np.array([np.roll(wave,1),wave,np.roll(wave,-1)])
        scores=calibration_scores(observed,refs,True)
        self.assertEqual(np.unravel_index(np.argmin(scores),scores.shape),(2,1))
        gain,_,_,ref=calibrate(wave,x,True)
        n=len(x)//2
        scalar=np.mean(abs(wave[:n]-gain*ref[:n])**2)/np.mean(abs(wave[:n])**2)
        self.assertAlmostEqual(scores[2,1],scalar,places=12)
        damaged=observed.copy();damaged[:,n:]*=-3j
        np.testing.assert_array_equal(calibration_scores(damaged,refs,True),scores)

    def test_b1_domains(self):
        coefficients=(632,-46,-594,65,499,-74)
        for baud,rates in [(2400,(4800,9600,14400)),(19200/7,(9600,19200)),
                           (2800,(9600,19200)),(3000,(9600,24000)),
                           (3200,(9600,24000,28800)),(24000/7,(31200,33600))]:
            for rate in rates:
                for role in ('call','answer'):
                    for shaping in (False,True):
                        for h in ((0,)*6,coefficients):
                            with self.subTest(rate=rate,baud=baud,role=role,shaping=shaping,h=h):
                                ref=build(rate,baud,role,shaping,h)
                                tx=DataTx(rate,baud,expanded=shaping,precoder=precoder_from_mp(h))
                                sc=Scrambler(role)
                                bits=[sc.step(1) for _ in range(tx.bits_per_data_frame)]
                                self.assertEqual(ref['bits'],bits)
                                np.testing.assert_array_equal(ref['x'],tx.data_frame(bits))

if __name__=='__main__':unittest.main()
