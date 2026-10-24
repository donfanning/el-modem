"""Independent NumPy feedback parity and complete equalizer trajectory checks."""
import copy
import math
import unittest
import numpy as np
from v34_feedback_native import feedback,window_energy,centroid,warmup
from v34_rx import Phase3Rx,SymbolEq,decide4


class FeedbackTests(unittest.TestCase):
    def test_feedback_matches_original_expression(self):
        rng=np.random.default_rng(340400)
        warmup()
        for length in (24,48,96):
            for d in (0j,1+1j,-3-1j):
                h=rng.normal(size=length)+1j*rng.normal(size=length)
                w=10000*(rng.normal(size=length)+1j*rng.normal(size=length))
                yr=1.4-.2j;phase=.37;freq=.001;mu=.03;kp=.04;ki=.0001
                expected=h.copy()
                expected+=mu*np.conj(w)*((d-yr)*complex(math.cos(phase),math.sin(phase)))/(float(np.vdot(w,w).real)+1e-9)
                efreq=freq;ephase=phase
                if abs(d)>0:
                    perr=math.atan2((yr*d.conjugate()).imag,(yr*d.conjugate()).real)
                    efreq+=ki*perr;ephase+=kp*perr+efreq
                actual=h.copy();ap,af=feedback(actual,w,yr,d,mu,kp,ki,phase,freq)
                np.testing.assert_allclose(actual,expected,rtol=2e-15,atol=2e-15)
                self.assertAlmostEqual(ap,ephase,places=15)
                self.assertAlmostEqual(af,efreq,places=15)
                known=h.copy();energy=window_energy(w)
                kp2,kf2=feedback(known,w,yr,d,mu,kp,ki,phase,freq,energy)
                np.testing.assert_allclose(known,expected,rtol=2e-15,atol=2e-15)
                self.assertAlmostEqual(kp2,ephase,places=15)
                self.assertAlmostEqual(kf2,efreq,places=15)
                np.testing.assert_allclose(energy,float(np.vdot(w,w).real),rtol=1e-15,atol=1e-12)
                oracle=float(np.sum(np.arange(length)*abs(h)**2)/(np.sum(abs(h)**2)+1e-12))-length//2
                self.assertAlmostEqual(centroid(h,length//2),oracle,places=12)
        self.assertEqual(centroid(np.zeros(24,complex),12),-12.)

    def test_complete_trajectory_preserves_callbacks_and_frozen_modes(self):
        rng=np.random.default_rng(340401)
        for mode,adapt,nlms in (('centroid',True,True),('gradient',True,True),
                                ('centroid',False,True),('centroid',True,False)):
            R=Phase3Rx(3200,1800,'answer');R._z=rng.normal(size=3000)+1j*rng.normal(size=3000)
            R._span=12;R.taps=np.zeros(24,complex);R.taps[12]=1;R.lag=40;R.delay=0
            slow=SymbolEq(R,20);slow.adapt=adapt;slow.adapt_nlms=nlms;slow.timing_mode=mode
            slow.ktp=.0001;slow.kti=.000001;slow.cdead=.01;slow.calpha=.1
            slow._native_feedback=False;fast=copy.deepcopy(slow);fast._native_feedback=True
            counts=[0,0];observed=[[],[]]
            for index,eq in enumerate((slow,fast)):
                def decide(y):
                    counts[index]+=1
                    return decide4(y)
                for _ in range(400):observed[index].append(eq.step(decide=decide))
            self.assertEqual(counts,[400,400]);self.assertEqual(slow.slips,fast.slips)
            np.testing.assert_allclose(observed[0],observed[1],rtol=1e-10,atol=1e-10)
            np.testing.assert_allclose(slow.h,fast.h,rtol=1e-10,atol=1e-10)
            for key in ('dt','vel','phase','freq','cf'):
                self.assertAlmostEqual(getattr(slow,key),getattr(fast,key),places=10)


if __name__=='__main__':unittest.main()
