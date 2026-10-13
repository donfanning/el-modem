"""Physical PCM superposition and pre-B1 selection invariants."""
import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent/'tools'))
from b1_duplex_probe import pcm_filter, echo_features, operator, fit_real_echo, train_model, FIT, HOLD
import v34_rx as rx


class DuplexProbeTests(unittest.TestCase):
    def test_collapsed_integer_delay_operator_matches_direct_sinc_at_edges(self):
        rng=np.random.default_rng(3405)
        z=rng.normal(size=4096)+1j*rng.normal(size=4096)
        centers=np.concatenate((np.array([-20.,0.,.001,4095.,4110.]),rng.uniform(0,4095,100)))
        h=rng.normal(size=48)+1j*rng.normal(size=48)
        rotation=np.exp(1j*rng.normal(size=len(centers)))
        fc=1959.1836734693877;spl=2.333335
        for delay in (0,173,1600):
            expected=np.column_stack([operator(z,centers-delay-j,h,spl,rotation)
                *np.exp(-2j*np.pi*fc*(delay+j)/8000) for j in range(32)])
            actual=echo_features(z,centers,h,spl,rotation,fc,delay)
            np.testing.assert_allclose(actual,expected,atol=1e-9,rtol=1e-10)

    def test_passband_delays_and_real_taps_match_actual_receiver_operator(self):
        rng=np.random.default_rng(3401)
        own=rng.normal(size=12000);source=rng.normal(size=12000)
        h=rng.normal(size=48)+1j*rng.normal(size=48)
        centers=np.arange(1800, 11000, 2.333335)
        rotation=np.exp(-1j*np.arange(len(centers))*.0002)
        fc=1959.1836734693877;spl=8000/(24000/7)
        taps=np.array([.12, -.03, .02, 0., -.01]);delay=173
        baseline=operator(rx.to_baseband(own, fc, 2125.714), centers, h, spl, rotation)
        injected=operator(rx.to_baseband(own+pcm_filter(source, delay, taps), fc, 2125.714),
                          centers, h, spl, rotation)
        features=echo_features(rx.to_baseband(source, fc, 2125.714), centers, h, spl,
                               rotation, fc, delay, len(taps))
        np.testing.assert_allclose(injected-baseline, features@taps, atol=3e-11, rtol=1e-10)

    def test_echo_fit_removes_own_channel_nuisance_and_uses_real_coefficients(self):
        rng=np.random.default_rng(3402)
        nuisance=rng.normal(size=(1500, 3))+1j*rng.normal(size=(1500, 3))
        features=rng.normal(size=(1500, 8))+1j*rng.normal(size=(1500, 8))
        features+=nuisance@rng.normal(size=(3, 8))
        truth=rng.normal(size=8)*.1
        y=nuisance@np.array([1+.2j, .3j, -.02])+features@truth
        estimated, _, _=fit_real_echo(nuisance, features, y)
        self.assertFalse(np.iscomplexobj(estimated))
        np.testing.assert_allclose(estimated, truth, rtol=.002, atol=.0003)

    def test_no_echo_or_zero_source_yields_no_spurious_cancellation(self):
        rng=np.random.default_rng(3403)
        nuisance=rng.normal(size=(1200, 3))+1j*rng.normal(size=(1200, 3))
        features=rng.normal(size=(1200, 8))+1j*rng.normal(size=(1200, 8))
        y=nuisance@np.array([1+.1j, .01j, -.01])
        taps, _, _=fit_real_echo(nuisance, features, y)
        np.testing.assert_allclose(taps, 0, atol=1e-14)
        taps, _, _=fit_real_echo(nuisance, np.zeros_like(features), y)
        np.testing.assert_array_equal(taps, np.zeros(8))

    def test_validation_changes_gate_not_fit_delay_or_coefficients(self):
        rng=np.random.default_rng(3404)
        source=rng.normal(size=18000);fc=1959.1836734693877;spl=2.3333333
        h=np.array([1+0j, .2+.1j]);indices=np.arange(1024, 4500)
        centers=2500+indices*spl;known=rng.normal(size=5500)+1j*rng.normal(size=5500)
        rotation=np.ones(len(indices), complex)
        z=rx.to_baseband(source, fc, max(1700., .62*8000/spl))
        truth=echo_features(z, centers, h, spl, rotation, fc, 173, 3)@np.array([.25, .04, -.025])
        trace=dict(symbol=indices, center=centers, rotation=rotation,
                   observed=known[indices]+truth)
        first=train_model(source, trace, known, fc, spl, h, 0)
        changed=trace['observed'].copy()
        hold=(indices>=HOLD[0]) & (indices<HOLD[1])
        changed[hold]=known[indices[hold]]
        second=train_model(source, {**trace, 'observed':changed}, known, fc, spl, h, 0)
        self.assertTrue(first['accepted'])
        self.assertFalse(second['accepted'])
        self.assertEqual(first['delay_samples'], second['delay_samples'])
        np.testing.assert_array_equal(first['taps'], second['taps'])

    def test_support_overlap_aborts_before_fitting(self):
        indices=np.arange(1024, 4500);centers=indices*.2
        trace=dict(symbol=indices, center=centers, rotation=np.ones(len(indices), complex),
                   observed=np.ones(len(indices), complex))
        with self.assertRaisesRegex(ValueError, 'PCM support overlap'):
            train_model(np.zeros(18000), trace, np.ones(5500, complex),
                        1959., .2, np.ones(48), 0)

    def test_available_history_excludes_future_delays_without_changing_fit_blocks(self):
        rng = np.random.default_rng(3406)
        full = rng.normal(size=18000)
        source = full[:12200]
        fc, spl = 1959.1836734693877, 2.3333333
        h = np.array([1+0j, .2+.1j])
        indices = np.arange(1024, 4500)
        centers = 2500 + indices * spl
        known = rng.normal(size=5500) + 1j*rng.normal(size=5500)
        rotation = np.ones(len(indices), complex)
        z = rx.to_baseband(full, fc, max(1700., .62*8000/spl))
        truth = echo_features(z, centers, h, spl, rotation, fc, 901, 3) @ np.array([.25, .04, -.025])
        hold = (indices >= HOLD[0]) & (indices < HOLD[1])
        support = len(h)*spl/4 + 46
        lower = int(np.floor(centers[hold].max() + support - len(source))) + 1
        trace = dict(symbol=indices, center=centers, rotation=rotation,
                     observed=known[indices]+truth)
        with self.assertRaisesRegex(ValueError, 'opposite PCM context'):
            train_model(source, trace, known, fc, spl, h, 0)
        result = train_model(source, trace, known, fc, spl, h, 0, minimum_delay=lower)
        self.assertTrue(result['accepted'])
        self.assertGreater(result['improvement_fraction'], .99)
        self.assertEqual(result['fit_symbols'], list(FIT))
        self.assertEqual(result['validation_symbols'], list(HOLD))
        self.assertGreaterEqual(result['delay_samples'], lower)
        self.assertLess(centers[hold].max()+support-result['delay_samples'], len(source))
        self.assertTrue(all(delay >= lower for delay in result['screening_shortlist']))
        changed = trace['observed'].copy()
        changed[hold] = known[indices[hold]]
        rejected = train_model(source, {**trace, 'observed': changed}, known,
            fc, spl, h, 0, minimum_delay=lower)
        self.assertFalse(rejected['accepted'])
        np.testing.assert_array_equal(rejected['taps'], result['taps'])
        for invalid in (-1, 1601, .5):
            with self.assertRaisesRegex(ValueError, 'causal delay range'):
                train_model(source, trace, known, fc, spl, h, 0, minimum_delay=invalid)


if __name__=='__main__':unittest.main()
