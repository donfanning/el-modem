"""Channel timing must be independent of packet boundaries and decoded content."""
import unittest
import numpy as np
from v34_data_fade import DataFade


class DataFadeTests(unittest.TestCase):
    def test_sample_preserving_and_packet_partition_independent(self):
        x=np.arange(2000,dtype=np.int16)+100
        one=DataFade(True,delay=100,edge=160,silence=640)
        chunks=DataFade(True,delay=100,edge=160,silence=640)
        one.arm(0);chunks.arm(0);chunks.arm(900)
        expected=one.process(x,0)
        pieces=[]
        for i in range(0,len(x),73):pieces.append(chunks.process(x[i:i+73],i))
        np.testing.assert_array_equal(np.concatenate(pieces),expected)
        np.testing.assert_array_equal(expected[:100],x[:100])
        np.testing.assert_array_equal(expected[1060:],x[1060:])
        self.assertTrue(np.all(expected[260:900]==0))
        self.assertEqual(len(expected),len(x))
        self.assertEqual(one.report(),chunks.report())
        self.assertEqual(one.report()['zero_gain_samples'],642)

    def test_payload_semantics_and_disabled_identity(self):
        f=DataFade(True);f.arm(80)
        self.assertTrue(f.ready_for_payload(80,'lapm'))
        self.assertFalse(f.ready_for_payload(f.end+3999,'raw'))
        self.assertTrue(f.ready_for_payload(f.end+4000,'raw'))
        off=DataFade(False);off.arm(80)
        x=np.array([-32000,0,32000],dtype=np.int16)
        np.testing.assert_array_equal(off.process(x,off.start),x)
        self.assertTrue(off.ready_for_payload(0,'raw'))
        self.assertEqual(off.samples_affected,0)
