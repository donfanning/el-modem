"""Independent exact and causal phase-feedback gates."""
import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent/'tools'))
from b1_phase_probe import decode,PhaseTracker
from v34_receive import projection,Receiver
from v34_negotiation import DirectionParameters
import v34_b1_reference as independent


class PhaseProbeTests(unittest.TestCase):
    def signal(self,rate):
        p=dict(rate=rate,baud=24000/7,carrier=96000/49,trellis=0,shaping=True,
               nonlinear=True,precoder=[129,188,-297,-160,222,-69])
        ref=independent.build(rate,p['baud'],'answer',True,p['precoder'])
        wanted=np.random.default_rng(991).integers(0,2,len(ref['bits']))
        flips=tuple(i for i,(a,b) in enumerate(zip(ref['bits'],wanted)) if a!=b)
        ref=independent.diagnostic_build(rate,p['baud'],'answer',True,p['precoder'],flip_bits=flips)
        x=np.asarray(ref['x']);energy=float(np.mean(abs(x)**2))
        return p,ref,projection(x,energy),energy

    def test_exact_random_bits_and_disabled_parity_at_three_rates(self):
        for rate in (24000,31200,33600):
            p,ref,wave,energy=self.signal(rate)
            for enabled in (False,True):
                result=decode(wave,p,energy,enabled=enabled)
                np.testing.assert_array_equal(result['bits'],ref['bits'])
                np.testing.assert_array_equal(result['mapper'],ref['u'])
                np.testing.assert_allclose([u['next_phase'] for u in result['updates']],0,atol=1e-12)

    def test_phase_step_has_correct_sign_and_only_changes_next_interval(self):
        p,ref,wave,energy=self.signal(24000)
        receiver=Receiver(DirectionParameters(**p),'answer');receiver.energy=energy
        tracker=PhaseTracker(receiver.detector())
        for pair in (wave*np.exp(.01j)).reshape(-1,2):tracker.feed(*pair)
        self.assertEqual(tracker.updates[0]['applied_phase'],0)
        self.assertAlmostEqual(tracker.updates[1]['applied_phase'],tracker.updates[0]['next_phase'])
        self.assertAlmostEqual(tracker.phase,.01,places=7)


if __name__=='__main__':unittest.main()
