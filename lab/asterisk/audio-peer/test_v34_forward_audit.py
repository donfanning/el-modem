"""Forward audit controls from an analytic transmitter, not fitted pulse knots."""
import unittest
import numpy as np
from v34_tx import rrc,trn_symbols
from v34_b1_reference import build,diagnostic_build
from v34_forward_audit import fit_pulse,compare_b1,inverse_system,solve_inverse,select_inverse


def transmit(symbols,origin,spl,carrier,n,filtered=False):
    samples=np.arange(n);out=np.zeros(n,complex)
    for k,z in enumerate(symbols):
        t=(samples-origin)/spl-k
        mask=abs(t)<=8
        out[mask]+=z*rrc(t[mask],1,.15)
        if filtered:
            t=t-.63;mask=abs(t)<=8
            out[mask]+=(.18+.09j)*z*rrc(t[mask],1,.15)
    return (out*np.exp(2j*np.pi*carrier*samples/8000)).real


class ForwardAuditTests(unittest.TestCase):
    def test_filtered_offset_clock_and_carrier_control(self):
        symbols,_=trn_symbols(950,'call',16)
        origin=61.7;spl=8000/(24000/7)*(1+140e-6);carrier=1959.183673469+1.2
        pcm=transmit(symbols,origin,spl,carrier,2400,True)
        model,_=fit_pulse(pcm,symbols,60,8000/(24000/7),1959.183673469)
        self.assertLess(model.held_out_training_nmse,.006)
        self.assertLess(abs(model.carrier_offset_hz-1.2),.15)
        self.assertLess(abs(model.clock_ppm-140),60)

    def test_reset_discrimination_and_held_out_isolation(self):
        trn,_=trn_symbols(1000,'call',16);spl=8000/(24000/7);carrier=1959.183673469
        pcm=transmit(trn,40,spl,carrier,2700)
        model,_=fit_pulse(pcm,trn,40,spl,carrier)
        good=build(24000,24000/7,'call',True,initial_state=(0,1,0,0))['x']
        wrong=build(24000,24000/7,'call',True)['x']
        b1=transmit(good,40,spl,carrier,400)
        a,trace=compare_b1(b1,model,good,40)
        b,_=compare_b1(b1,model,wrong,40)
        self.assertLess(a['held_out_nmse'],.005)
        self.assertGreater(b['held_out_nmse'],a['held_out_nmse']*20)
        changed=b1.copy();changed[trace['samples'][trace['held_out']]]+=10
        c,_=compare_b1(changed,model,good,40)
        self.assertEqual(a['gain_real'],c['gain_real'])
        self.assertEqual(a['gain_imag'],c['gain_imag'])
        self.assertGreater(c['held_out_nmse'],a['held_out_nmse'])

    def test_e_tail_and_wrong_boundary_control(self):
        trn,_=trn_symbols(1000,'answer',16);spl=8000/(24000/7);carrier=1959.183673469
        model,_=fit_pulse(transmit(trn,40,spl,carrier,2700),trn,40,spl,carrier)
        context=trn[-model.span:]
        symbols=build(24000,24000/7,'answer',True)['x']
        pcm=transmit(np.r_[context,symbols],40-model.span*spl,spl,carrier,400)
        correct,_=compare_b1(pcm,model,symbols,40,context=context)
        shifted,_=compare_b1(pcm,model,symbols,40+spl,context=context)
        self.assertLess(correct['first_eight_nmse'],.005)
        self.assertLess(correct['held_out_nmse'],.005)
        self.assertGreater(shifted['held_out_nmse'],.1)

    def test_final_trn_and_early_b1_do_not_select_parameters(self):
        trn,_=trn_symbols(1000,'call',16);spl=8000/(24000/7);carrier=1959.183673469
        pcm=transmit(trn,40,spl,carrier,2700);model,trace=fit_pulse(pcm,trn,40,spl,carrier)
        changed=pcm.copy();changed[trace['validation_samples'][-100:]]+=100
        other,_=fit_pulse(changed,trn,40,spl,carrier)
        np.testing.assert_array_equal(model.pulse,other.pulse)
        self.assertEqual(model.selection,other.selection)
        self.assertGreater(other.held_out_training_nmse,model.held_out_training_nmse)
        symbols=build(24000,24000/7,'call',True)['x'];context=trn[-16:]
        b1=transmit(np.r_[context,symbols],40-16*spl,spl,carrier,400)
        a,t=compare_b1(b1,model,symbols,40,context=context)
        changed=b1.copy();changed[t['samples'][(t['samples']-40)/spl<8]]+=10
        b,_=compare_b1(changed,model,symbols,40,context=context)
        self.assertEqual(a['gain_real'],b['gain_real']);self.assertEqual(a['gain_imag'],b['gain_imag'])
        self.assertGreater(b['first_eight_nmse'],a['first_eight_nmse'])

    def test_raw_inverse_recovers_unknown_symbols_without_expected_b1(self):
        trn,_=trn_symbols(1000,'answer',16);spl=8000/(24000/7);carrier=1959.183673469
        pcm=transmit(trn,40,spl,carrier,2700);model,_=fit_pulse(pcm,trn,40,spl,carrier)
        ridge,evidence=select_inverse(pcm,model,trn);self.assertFalse(evidence['selection_uses_b1'])
        rng=np.random.default_rng(17);unknown=rng.normal(size=120)+1j*rng.normal(size=120)
        context=trn[-16:];suffix=trn[:32]
        wave=transmit(np.r_[context,unknown,suffix],60-16*spl,spl,carrier,500)
        system=inverse_system(wave,model,60,len(unknown),context);recovered,info=solve_inverse(system,ridge)
        self.assertFalse(info['expected_b1_used_for_inversion'])
        self.assertLess(np.mean(abs(recovered[:-16]-unknown[:-16])**2),.01)

    def test_diagnostic_variants_preserve_the_standard_constructor(self):
        standard=build(24000,24000/7,'call',True,trace=True)
        same=diagnostic_build(24000,24000/7,'call',True)
        for key in ('bits','u','x','states'):self.assertEqual(standard[key],same[key])
        variant=diagnostic_build(24000,24000/7,'call',True,flip_bits=(35,40,41))
        self.assertEqual([i for i,(a,b) in enumerate(zip(standard['bits'],variant['bits'])) if a!=b],[35,40,41])
        for offset in (-2,-1,1,2):
            other=diagnostic_build(24000,24000/7,'call',True,inversion_offset_4d=offset)
            self.assertFalse(other['standard_qualification']);self.assertNotEqual(standard['u'],other['u'])

    def test_control_noise_and_cache_probe_are_reproducible(self):
        from v34_channel_controls import impair,timing_cache_probe
        from v34_forward_audit import PulseModel
        model=PulseModel(80,2.5,1900,np.ones(65,complex),0,0,0,0)
        pcm=np.sin(np.arange(2000)*.4)*100
        residual=np.random.default_rng(8).normal(size=1000)
        a=impair(pcm,model,model,residual,100,'noise',3)
        np.testing.assert_array_equal(a,impair(pcm,model,model,residual,100,'noise',3))
        self.assertFalse(np.array_equal(a,impair(pcm,model,model,residual,100,'noise',4)))
        probe=timing_cache_probe();self.assertLess(probe['cached_nmse'],1e-20)
        self.assertLess(probe['rebuilt_nmse'],1e-20)

    def test_clock_resampling_preserves_high_frequency_amplitude(self):
        from v34_channel_controls import bandlimited_sample
        n=np.arange(4096);pcm=np.cos(2*np.pi*3400*n/8000)
        positions=np.arange(100,3900)+.37
        actual=bandlimited_sample(pcm,positions)
        expected=np.cos(2*np.pi*3400*positions/8000)
        self.assertLess(np.mean((actual-expected)**2),1e-6)


if __name__=='__main__':unittest.main()
