"""Per-call isolation and causal cancellation fallback, independent of labels."""
import unittest
from types import SimpleNamespace
import numpy as np
import v34_rx as rx
from v34_receive_dsp import ReceiveDsp


class ReceiveDspTests(unittest.TestCase):
    def test_instances_do_not_change_receiver_classes(self):
        feed,step=rx.Phase4Receiver.feed,rx.SymbolEq.step
        left,right=SimpleNamespace(),SimpleNamespace()
        a,b=ReceiveDsp(left,cancel=True),ReceiveDsp(right,predictive=True)
        a.commit_tx(np.ones(160));b.commit_tx(np.ones(160)*7)
        self.assertIs(left.receive_dsp,a);self.assertIs(right.receive_dsp,b)
        self.assertIs(rx.Phase4Receiver.feed,feed);self.assertIs(rx.SymbolEq.step,step)
        self.assertEqual(a.source[0],1);self.assertEqual(b.source[0],7)

    def test_rejection_preserves_pcm_and_future_source_falls_back(self):
        d=ReceiveDsp(SimpleNamespace(),cancel=True)
        pcm=np.arange(160,dtype='<i2')
        d.model=dict(accepted=False)
        np.testing.assert_array_equal(d.process(pcm),pcm)
        d.model=dict(accepted=True,delay_samples=0,taps=[.3])
        np.testing.assert_array_equal(d.process(pcm),pcm)
        self.assertFalse(d.model['accepted']);self.assertEqual(len(d.fallbacks),1)

    def test_known_channel_subtraction_and_history_bound(self):
        d=ReceiveDsp(SimpleNamespace(),cancel=True)
        d.commit_tx(np.full(512,1000.))
        d.model=dict(accepted=True,delay_samples=173,taps=[.25])
        d.process(np.zeros(160,dtype='<i2'))
        # The activation fade is finished at sample 160; echo arrives at 173.
        result=d.process(np.full(160,250,dtype='<i2'))
        np.testing.assert_array_equal(result[13:],np.zeros(147,dtype='<i2'))
        for _ in range(200):d.commit_tx(np.zeros(160))
        self.assertGreater(d.source_origin,0)
        self.assertLessEqual(d.length-d.source_origin,20096)


if __name__=='__main__':unittest.main()
