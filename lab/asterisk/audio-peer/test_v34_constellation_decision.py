"""Decision parity at dense-constellation boundaries and with feedback state."""
import unittest
import numpy as np
import v34_data as vd
from v34_datarx import DataRx
from v34_negotiation import DirectionParameters
from v34_receive import Receiver, inverse_projection, projection


class ConstellationDecisionTests(unittest.TestCase):
    @staticmethod
    def reference(rx, y):
        return complex(rx._pts[int(np.argmin(np.abs(rx._pts-y)))])

    def test_interior_holes_and_outside_match_at_all_data_rates(self):
        rng=np.random.default_rng(336021)
        for rate in range(4800,33601,2400):
            for expanded in (False,True):
                rx=DataRx(rate,3429,expanded)
                samples=rx._pts[rng.integers(len(rx._pts),size=800)]
                samples=samples+.6*(rng.standard_normal(800)+1j*rng.standard_normal(800))
                extent=max(abs(rx._pts.real).max(),abs(rx._pts.imag).max())+8
                samples=np.r_[samples,rng.uniform(-extent,extent,800)+1j*rng.uniform(-extent,extent,800)]
                for y in samples:
                    self.assertEqual(rx.nearest_point(complex(y)),self.reference(rx,y),
                                     (rate,expanded,y))

    def test_exact_and_ulp_near_ties_preserve_first_point(self):
        for rate in (4800,24000,31200,33600):
            rx=DataRx(rate,3429)
            extent=int(max(abs(rx._pts.real).max(),abs(rx._pts.imag).max()))+2
            extent=(extent+1)//2*2
            for real in range(-extent,extent+1,2):
                for imag in range(-extent,extent+1):
                    for delta in (0.,np.spacing(float(real)),1e-11,-1e-11):
                        y=complex(real+delta,imag)
                        self.assertEqual(rx.nearest_point(y),self.reference(rx,y))
                        y=complex(imag,real+delta)
                        self.assertEqual(rx.nearest_point(y),self.reference(rx,y))

    def test_forward_target_and_precoder_history_match_reference(self):
        rng=np.random.default_rng(336022)
        for nonlinear in (False,True):
            for precoder in ((0,)*6,(129,188,-297,-160,222,-69)):
                profile=DirectionParameters(33600,3429,1959.,nonlinear=nonlinear,precoder=precoder)
                rx=Receiver(profile,'answer');rx.energy=675.
                history=[0j]*3
                for y in rng.normal(0,22,1000)+1j*rng.normal(0,22,1000):
                    x=inverse_projection(y,rx.energy) if nonlinear else y
                    p=vd._round_complex(sum(a*h for a,h in zip(history,rx.h)),2**-7)
                    c=vd._round_complex(p,2**rx.tx.w)
                    u=self.reference(rx.drx,x+p-c);target=u+c-p
                    history=[target]+history[:2]
                    expected=complex(projection(target,rx.energy)) if nonlinear else target
                    self.assertEqual(rx.decide(y),expected)
                    self.assertEqual(rx.slicer_hist,history)


if __name__=='__main__':unittest.main()
