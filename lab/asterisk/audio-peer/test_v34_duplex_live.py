"""Causal local-TX history and state-replay coordinate invariants."""
import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent/'tools'))
import v34_rx as rx
from duplex_live_probe import fir_segment,Canceller
from b1_duplex_probe import pcm_filter


class DuplexLiveTests(unittest.TestCase):
    def test_chunked_fir_uses_committed_tx_and_matches_full_pcm_convolution(self):
        rng=np.random.default_rng(3481);source=rng.normal(size=4000);taps=rng.normal(size=32)*.01
        expected=pcm_filter(source,173,taps)
        actual=np.concatenate([fir_segment(source[:min(start+160,len(source))],start,
            min(160,len(source)-start),173,taps) for start in range(0,len(source),160)])
        np.testing.assert_allclose(actual,expected,atol=1e-15)
        with self.assertRaisesRegex(ValueError,'future TX'):
            fir_segment(source[:100],100,160,0,taps)

    def test_startup_padding_is_explicit_and_later_clock_changes_abort(self):
        originals=(rx.Phase4Receiver.feed,rx.SymbolEq.step)
        canceller=Canceller()
        try:
            canceller.append_tx(np.arange(160),185)
            canceller.append_tx(np.arange(160,320),160)
            self.assertEqual(canceller.padding,25)
            np.testing.assert_array_equal(canceller.source[:345],np.r_[np.zeros(25),np.arange(320)])
            with self.assertRaisesRegex(ValueError,'clock changes'):
                canceller.append_tx(np.arange(160),161)
        finally:canceller.close()
        self.assertIs(rx.Phase4Receiver.feed,originals[0])
        self.assertIs(rx.SymbolEq.step,originals[1])

    def test_unaccepted_model_is_exact_no_cancellation_parity(self):
        canceller=Canceller(enabled=True)
        try:
            pcm=np.arange(-80,80,dtype='<i2');canceller.append_tx(pcm,160)
            canceller.model=dict(accepted=False)
            np.testing.assert_array_equal(canceller.process(pcm,0),pcm)
            self.assertIsNone(canceller.activation)
        finally:canceller.close()


if __name__=='__main__':unittest.main()
