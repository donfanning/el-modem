"""Streaming storage must preserve physical interpolation and clock epochs."""
import unittest
from types import SimpleNamespace
import numpy as np
import v34_rx as rx
from v34_streaming import trim_data_pcm,update_data_baseband


class StreamingTests(unittest.TestCase):
    def test_data_input_is_identical_at_different_packet_cadences(self):
        rng=np.random.default_rng(789);pcm=rng.normal(size=16000)
        outputs=[]
        for batch in (160,480,800):
            R=rx.Phase3Rx(3428.5714285714,1959.1836734694)
            R.lag=20;R.delay=.2;R._span=12
            R.taps=np.zeros(24,complex);R.taps[12]=1
            eq=rx.SymbolEq(R,100);eq.begin_data_tracking(False)
            result=[]
            for end in range(batch,len(pcm)+1,batch):
                R._z=R.update_baseband(pcm[:end])
                while eq.k<eq.available():result.append(eq.step(decide=lambda y:y))
            R._z=R.update_baseband(pcm)
            while eq.k<eq.available():result.append(eq.step(decide=lambda y:y))
            outputs.append(result)
        for result in outputs[1:]:np.testing.assert_allclose(result,outputs[0],rtol=1e-10,atol=1e-10)

    def test_trimmed_filter_and_equalizer_match_full_history(self):
        rng=np.random.default_rng(456)
        pcm=rng.normal(size=50000)
        full=rx.Phase3Rx(3428.5714285714,1959.1836734694)
        bounded=rx.Phase3Rx(full.baud,full.fc)
        for R in (full,bounded):
            R.lag=20;R.delay=.2;R._span=12
            R.taps=np.zeros(24,complex);R.taps[12]=1
            R._z=R.update_baseband(pcm[:16000])
        a,b=rx.SymbolEq(full,6000),rx.SymbolEq(bounded,6000)
        for e in (a,b):e.adapt=False;e.freq=.0002;e.vel=1e-5
        engine=SimpleNamespace(_rx=pcm[:16000].copy(),rx_t=16000,_rx_state={'qualified':True})
        for end in range(16160,50001,160):
            full._z=full.update_baseband(pcm[:end])
            engine._rx=np.concatenate([engine._rx,pcm[end-160:end]])
            engine.rx_t=end
            update_data_baseband(engine,bounded,0)
            self.assertEqual(a.available(),b.available())
            while a.k<a.available():
                self.assertAlmostEqual(abs(a.step(decide=lambda y:y)-b.step(decide=lambda y:y)),0,places=10)
            self.assertEqual(a.timing_snapshot(),b.timing_snapshot())
            trim_data_pcm(engine,b,0)
            self.assertLess(len(engine._rx),21000)
        self.assertGreater(engine._rx_origin,0)
        self.assertLess(len(bounded._bb_storage),42000)

    def test_trim_preserves_future_and_does_not_run_before_qualification(self):
        e=SimpleNamespace(R=SimpleNamespace(lag=0,LEAD=8,delay=0,spl=2),k=25000,dt=.1)
        engine=SimpleNamespace(_rx=np.arange(51000),_rx_state=None)
        trim_data_pcm(engine,e,0)
        self.assertEqual(len(engine._rx),51000)
        engine._rx_state={'vit':True}
        trim_data_pcm(engine,e,0)
        np.testing.assert_array_equal(engine._rx,np.arange(engine._rx_origin,51000))
        self.assertLess(engine._rx_origin,2*e.k)

    def test_connected_mp_tracking_retains_absolute_input_and_b1_support(self):
        e=SimpleNamespace(R=SimpleNamespace(lag=0,LEAD=8,delay=0,spl=2),k=25000,dt=.1)
        engine=SimpleNamespace(_rx=np.arange(51000),_rx_state=None)
        trim_data_pcm(engine,e,0,tracking=True)
        self.assertGreater(engine._rx_origin,0)
        np.testing.assert_array_equal(engine._rx,np.arange(engine._rx_origin,51000))
        self.assertLess(len(engine._rx),21000)
        # Transition to B1: the initializer may need past/current burst input.
        before=engine._rx.copy()
        e.k+=1000
        trim_data_pcm(engine,e,0)
        np.testing.assert_array_equal(engine._rx,before)


if __name__=='__main__':unittest.main()
