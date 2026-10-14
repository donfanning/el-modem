import os
import unittest
from unittest.mock import patch
import numpy as np
from v32_stream import Baseband, EchoCanceller, _find_delay_gen


def result(generator):
    while True:
        try:next(generator)
        except StopIteration as completed:return completed.value


class SafeEchoTests(unittest.TestCase):
    def test_batched_fir_preserves_scalar_samples_and_clock_positions(self):
        rng=np.random.default_rng(3813)
        for fs in (8000,9600):
            with patch.dict(os.environ,{'V32_ECHO_SAFE':'0'}):old=Baseband(fs)
            with patch.dict(os.environ,{'V32_ECHO_SAFE':'1'}):new=Baseband(fs)
            for block in range(90):
                samples=rng.integers(-12000,12000,size=(block*17)%200+1).tolist()
                if block==30:
                    old.step/=1.000129;new.step/=1.000129
                if block==50:
                    old.shift(.125);new.shift(.125)
                self.assertEqual(old.feed(samples),new.feed(samples))
                self.assertEqual(old.pos,new.pos)
                self.assertEqual(old.base,new.base)

    def test_delay_search_matches_scalar_peak_ratio_and_null(self):
        rng=np.random.default_rng(3810)
        tx=rng.normal(size=1800)+1j*rng.normal(size=1800)
        for lag in (0,173,399):
            rx=np.r_[np.zeros(lag),.25*tx,np.zeros(450-lag)]
            rx+=.02*(rng.normal(size=len(rx))+1j*rng.normal(size=len(rx)))
            with patch.dict(os.environ,{'V32_ECHO_SAFE':'0'}):old=result(_find_delay_gen(tx[:1400].tolist(),rx.tolist(),400))
            with patch.dict(os.environ,{'V32_ECHO_SAFE':'1'}):new=result(_find_delay_gen(tx[:1400].tolist(),rx.tolist(),400))
            self.assertEqual(old[0],lag)
            self.assertEqual(new[0],lag)
            self.assertAlmostEqual(old[1],new[1],delta=1e-9)
        with patch.dict(os.environ,{'V32_ECHO_SAFE':'1'}):
            self.assertIsNone(result(_find_delay_gen(tx[:1400],np.zeros(2000),400)))

    def test_frozen_dot_matches_scalar_across_ring_and_history_wraps(self):
        rng=np.random.default_rng(3811)
        with patch.dict(os.environ,{'V32_ECHO_SAFE':'0'}):old=EchoCanceller(taps=16,history=128)
        with patch.dict(os.environ,{'V32_ECHO_SAFE':'1'}):new=EchoCanceller(taps=16,history=128)
        old.configure(40);new.configure(40)
        for m in range(1500):
            tx=complex(*rng.normal(size=2))*1000
            rx=complex(*rng.normal(size=2))*1000
            old.push_tx(tx);new.push_tx(tx)
            # Training -> frozen -> training -> frozen checks cache invalidation.
            mu=.2 if m<250 or 700<=m<800 else 0.
            a=old.process(rx,m,mu);b=new.process(rx,m,mu)
            self.assertAlmostEqual(abs(a-b),0,delta=1e-9)
            np.testing.assert_allclose(old.c,new.c,rtol=1e-12,atol=1e-12)
            if m%256==0:old.recenter();new.recenter()
        self.assertAlmostEqual(old.p_in,new.p_in,delta=1e-8)
        self.assertAlmostEqual(old.p_out,new.p_out,delta=1e-8)

    def test_precursor_and_postcursor_echo_fit_with_centered_support(self):
        rng=np.random.default_rng(3812)
        tx=rng.normal(size=10000)+1j*rng.normal(size=10000)
        # Peak at delay 64, with real precursor support at delay 62.
        with patch.dict(os.environ,{'V32_ECHO_SAFE':'1'}):ec=EchoCanceller(taps=16)
        ec.configure(56)
        residual=power=0.
        for m,x in enumerate(tx):
            ec.push_tx(x)
            echo=sum(g*tx[m-d] for d,g in [(62,.04),(64,.25),(67,-.025)] if m>=d)
            clean=ec.process(echo,m,.2 if m<7000 else 0.)
            if m>7500:residual+=abs(clean)**2;power+=abs(echo)**2
        self.assertLess(residual/power,1e-6)
        self.assertEqual(int(np.argmax(np.abs(ec.c))),8)


if __name__=='__main__':unittest.main()
