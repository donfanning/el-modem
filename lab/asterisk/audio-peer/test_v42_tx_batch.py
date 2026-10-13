"""Bulk line bits match bit-at-a-time clocks, framing, timers and fallback."""
import unittest
import numpy as np
from test_v42_layer import BitEngine
from v42_lapm import N400
from mnp_link import MnpLink


class BatchTransmitTests(unittest.TestCase):
    def compare(self, a, b, count):
        self.assertEqual([a.v42_tx_bit() for _ in range(count)],b.v42_tx_bits(count))
        self.assertEqual(a.v42_mode,b.v42_mode)
        self.assertEqual(a._v42_bits,b._v42_bits)
        self.assertEqual(list(a.q),list(b.q))
        if a._v42 is not None:
            clocks=(('_bits','_resp_t401','_lr_attempts') if isinstance(a._v42,MnpLink)
                    else ('bits_sent','_t401_at','_tries','stats'))
            for key in ('state','events')+clocks:
                self.assertEqual(getattr(a._v42,key),getattr(b._v42,key),key)
            self.assertEqual(list(a._v42._tx),list(b._v42._tx))

    def test_detection_fallback_and_raw_preserve_every_clock(self):
        for protocol in ('raw','lapm','mnp'):
            a,b=BitEngine(33600),BitEngine(33600)
            if protocol!='raw':
                for x in (a,b):x.start_v42(True,protocol=protocol,rate=33600,establish_seconds=2.)
            for x in (a,b):x.v42_feed_bytes(b'\x00\xff\x7e\x7d'*300)
            for _ in range(100):self.compare(a,b,1136)
            if protocol!='raw':self.assertEqual(a.v42_mode,'failed')

    def test_both_directions_match_through_establishment_payload_and_flags(self):
        rng=np.random.default_rng(336023)
        for compression in (None,(3,1024,8)):
            ref=[BitEngine(33600),BitEngine(33600)]
            bat=[BitEngine(33600),BitEngine(33600)]
            messages=[rng.integers(0,256,size=2000,dtype=np.uint8).tobytes(),
                      b'The quick brown fox\x00\xff\x7e'*100]
            for pair in (ref,bat):
                for i,x in enumerate(pair):
                    x.start_v42(i==0,rate=33600,compression=compression)
                    x.v42_feed_bytes(messages[i])
            for frame in range(200):
                count=(17,1136,79)[frame%3]
                streams=[]
                for a,b in zip(ref,bat):
                    bits=[a.v42_tx_bit() for _ in range(count)]
                    self.assertEqual(bits,b.v42_tx_bits(count));streams.append(bits)
                    self.assertEqual(a._v42.bits_sent,b._v42.bits_sent)
                    self.assertEqual(a._v42_bits,b._v42_bits)
                    self.assertEqual(a._v42.stats,b._v42.stats)
                    self.assertEqual(a._v42._t401_at,b._v42._t401_at)
                for pair in (ref,bat):
                    for i,x in enumerate(pair):
                        x.out+=x.v42_rx_bits(streams[1-i]);x.out+=x.v42_read()
            for i,(a,b) in enumerate(zip(ref,bat)):
                self.assertEqual(a.v42_mode,'lapm');self.assertEqual(b.v42_mode,'lapm')
                self.assertEqual(a.out,b.out);self.assertEqual(a.out,messages[1-i])

    def test_queued_bits_do_not_delay_timer_expiry_or_lost_link(self):
        a,b=BitEngine(33600),BitEngine(33600)
        for x in (a,b):
            x.start_v42(True,rate=33600,detect=False)
            x.v42_mode=x._v42.state='lapm';x._v42._started=True
            x._v42._tx.clear();x._v42._tx.extend([0,1]*19)
            x._v42._tries=N400;x._v42._t401_at=5
        self.compare(a,b,1136)
        self.assertEqual(a.v42_mode,'lost')
        self.assertEqual(a._v42.bits_sent,39)
        self.assertEqual(a._v42.stats['t401_expiries'],1)

    def test_single_bit_override_is_honoured(self):
        class CustomBits(BitEngine):
            def v42_tx_bit(self):return self.q.popleft() if self.q else 0
        a=CustomBits();a.q.extend([1,0,1])
        self.assertEqual(a.v42_tx_bits(6),[1,0,1,0,0,0])


if __name__=='__main__':unittest.main()
