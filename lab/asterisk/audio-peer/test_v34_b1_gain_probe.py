"""Causality, waveform coordinates and independent known-bit gain gates."""
import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent/'tools'))
from b1_gain_probe import early_decision, GainTracker, gates, decode
from v34_receive import Receiver, projection
from v34_negotiation import DirectionParameters
import v34_b1_reference as independent


class GainProbeTests(unittest.TestCase):
    def profile(self,rate=33600,nonlinear=True):
        return dict(rate=rate,baud=24000/7,carrier=96000/49,trellis=0,
            nonlinear=nonlinear,shaping=True,precoder=[129,188,-297,-160,222,-69])

    def test_independent_b1_and_random_bits_both_roles(self):
        for rate in (24000,31200,33600):
            self.assertEqual(len(gates(self.profile(rate))),4)

    def test_early_decision_preserves_state_and_projects_waveform(self):
        profile=self.profile(); ref=independent.build(33600,24000/7,'answer',True,profile['precoder'])
        x=np.asarray(ref['x']); energy=float(np.mean(abs(x)**2)); wave=projection(x,energy)
        receiver=Receiver(DirectionParameters(**profile),'answer');receiver.energy=energy
        detector=receiver.detector();detector.feed(*wave[:2])
        metric=detector.metric.copy();history=list(detector.x_hist)
        early=early_decision(detector)
        np.testing.assert_array_equal(early['mapper'],ref['u'][:2])
        np.testing.assert_allclose(early['waveform'],wave[:2],atol=1e-12)
        np.testing.assert_array_equal(detector.metric,metric)
        self.assertEqual(detector.x_hist,history)
        self.assertEqual(detector.m4,1);self.assertEqual(len(detector.hist),1)
        self.assertEqual(detector.decided,[])
        detector.flush()
        with self.assertRaisesRegex(ValueError,'no current interval'):early_decision(detector)

    def test_update_is_for_next_interval_and_recovers_injected_gain(self):
        profile=self.profile(14400,False)
        ref=independent.build(14400,24000/7,'answer',True,profile['precoder'])
        wave=np.asarray(ref['x']); receiver=Receiver(DirectionParameters(**profile),'answer')
        tracker=GainTracker(receiver.detector())
        for pair in (wave*1.08).reshape(-1,2):tracker.feed(*pair)
        self.assertEqual(tracker.updates[0]['applied_scale'],1.)
        self.assertAlmostEqual(tracker.updates[1]['applied_scale'],tracker.updates[0]['next_scale'])
        self.assertAlmostEqual(tracker.scale,1.08,places=4)
        self.assertTrue(all(abs(np.log(u['next_scale']/u['applied_scale']))<=.02000000001 for u in tracker.updates))

    def test_early_waveform_targets_with_strong_precoder_and_nonzero_state(self):
        profile=self.profile();profile['precoder']=[8192,4096,-4096,2048,2048,-1024]
        state=(1,0,1,1)
        ref=independent.build(33600,24000/7,'answer',True,profile['precoder'],state)
        x=np.asarray(ref['x']); energy=float(np.mean(abs(x)**2)); wave=projection(x,energy)
        receiver=Receiver(DirectionParameters(**profile),'answer',state);receiver.energy=energy
        detector=receiver.detector()
        for i in range(0,len(wave),2):
            detector.feed(*wave[i:i+2])
            early=early_decision(detector)
            np.testing.assert_array_equal(early['mapper'],ref['u'][i:i+2])
            np.testing.assert_allclose(early['waveform'],wave[i:i+2],atol=1e-12)


if __name__=='__main__':unittest.main()
