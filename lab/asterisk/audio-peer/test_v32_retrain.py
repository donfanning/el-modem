"""Two-ended line recovery, original profile ceilings and retained application link."""
import struct
import random
import unittest
from collections import deque
from v32_stream import V32Modem
from v32_answer import V32Answerer


class Pair:
    def __init__(self):
        self.c=V32Modem(use_v8=False,echo=False)
        self.a=V32Answerer(use_v8=False,echo=False)
        self.to_c=deque([0]*320);self.to_a=deque([0]*320)
        self.got_c=bytearray();self.got_a=bytearray()

    def step(self):
        self.to_a.extend(self.c.tx_samples(160));self.to_c.extend(self.a.tx_samples(160))
        self.got_c.extend(self.c.feed_pcm(struct.pack('<160h',*(self.to_c.popleft() for _ in range(160)))))
        self.got_a.extend(self.a.feed_pcm(struct.pack('<160h',*(self.to_a.popleft() for _ in range(160)))))

    def until(self,predicate,limit=2300,carrier=False):
        for _ in range(limit):
            if predicate():return
            self.step()
            if carrier and not (self.c.connected and self.a.connected):
                raise AssertionError('logical carrier dropped during recovery')
            if self.c.state=='failed' or self.a.state=='failed':
                raise AssertionError(str((self.c.failed_reason,self.a.failed_reason)))
        raise AssertionError(str((self.c.state,self.a.state,self.c.retrain_stats,self.a.retrain_stats)))

    def ready(self):
        return all(m.state=='data' and not m._retraining for m in (self.c,self.a))


class RetrainTests(unittest.TestCase):
    def test_same_initiator_retrains_three_times_with_verified_lapm_and_budget_renewal(self):
        for role in ('call', 'answer'):
            with self.subTest(role=role):
                p=Pair();p.until(p.ready)
                p.c.start_v42(True,detect=False)
                p.a.start_v42(False,detect=False)
                p.until(lambda:p.c.v42_mode==p.a.v42_mode=='lapm')
                links=(p.c._v42,p.a._v42)
                initiator=p.c if role=='call' else p.a
                expected_c=bytearray();expected_a=bytearray()
                for epoch in range(4):
                    cbytes=random.Random(301+epoch).randbytes(16384)
                    abytes=random.Random(401+epoch).randbytes(16384)
                    expected_c.extend(cbytes);expected_a.extend(abytes)
                    p.c.feed_bytes(cbytes);p.a.feed_bytes(abytes)
                    p.until(lambda:len(p.got_a)>=len(expected_c) and len(p.got_c)>=len(expected_a),carrier=True)
                    self.assertEqual(p.got_a,expected_c)
                    self.assertEqual(p.got_c,expected_a)
                    if epoch<3:
                        self.assertTrue(initiator.request_retrain(9600))
                        p.until(p.ready,carrier=True)
                        self.assertEqual((p.c.bit_rate,p.a.bit_rate),(9600,9600))
                self.assertIs(p.c._v42,links[0]);self.assertIs(p.a._v42,links[1])
                for m in (p.c,p.a):
                    self.assertEqual(m.retrain_stats['attempts'],3)
                    self.assertEqual(m.retrain_stats['budget']['total'],3)
                    self.assertGreaterEqual(m.retrain_stats['budget']['rearms'],1)
                    self.assertTrue(all('completed_seconds' in e for e in m.retrain_stats['events']))

    def test_completed_raw_character_is_not_replayed_during_following_marks(self):
        m=V32Modem(use_v8=False,echo=False)
        m.feed_bytes(b'ab')
        for _ in range(5):m.tx._frame_bit()
        self.assertEqual(m.tx._current_byte,ord('a'))
        m.tx.feed_marks(50)
        for _ in range(5):m.tx._frame_bit()
        self.assertIsNone(m.tx._current_byte)
        m.state='data';m.connected=True;m.bit_rate=9600
        self.assertTrue(m.request_retrain())
        self.assertEqual(list(m.tx._tx_bytes),list(b'b'))

    def test_both_roles_recover_down_then_up_and_preserve_pending_raw_character(self):
        p=Pair();p.until(p.ready)
        self.assertEqual((p.c.bit_rate,p.a.bit_rate),(14400,14400))
        p.c.feed_bytes(b'ab');p.c.tx._frame_bit()
        self.assertTrue(p.c.request_retrain(4800))
        self.assertEqual(list(p.c.tx._tx_bytes),list(b'ab'))
        # The partial old raw character may already have reached the remote DTE.
        # Prove its retention, then isolate the fresh post-recovery payload gate.
        p.c.tx._tx_bytes.clear()
        p.until(p.ready,carrier=True)
        self.assertEqual((p.c.bit_rate,p.a.bit_rate),(4800,4800))
        self.assertTrue(p.a.request_retrain())
        p.until(p.ready,carrier=True)
        self.assertEqual((p.c.bit_rate,p.a.bit_rate),(14400,14400))
        p.got_c.clear();p.got_a.clear()
        cbytes=bytes(range(256))*2;abytes=bytes(reversed(range(256)))*2
        p.c.feed_bytes(cbytes);p.a.feed_bytes(abytes)
        p.until(lambda:len(p.got_a)>=len(cbytes) and len(p.got_c)>=len(abytes),carrier=True)
        self.assertEqual(bytes(p.got_a),cbytes);self.assertEqual(bytes(p.got_c),abytes)
        self.assertEqual([e['negotiated_rate'] for e in p.c.retrain_stats['events']],[4800,14400])
        self.assertEqual([e['negotiated_rate'] for e in p.a.retrain_stats['events']],[4800,14400])
        self.assertFalse(p.c.use_echo or p.a.use_echo)

    def test_lapm_payload_crosses_both_role_retrains_without_new_link(self):
        self.lapm_crossing(None)

    def test_compression_dictionaries_survive_both_role_retrains(self):
        self.lapm_crossing((3,512,8))

    def lapm_crossing(self, compression):
        p=Pair();p.until(p.ready)
        p.c.start_v42(True,detect=False,compression=compression)
        p.a.start_v42(False,detect=False,compression=compression)
        p.until(lambda:p.c.v42_mode==p.a.v42_mode=='lapm')
        links=(p.c._v42,p.a._v42)
        dictionaries=[(m._v42._enc,m._v42._dec) for m in (p.c,p.a)]
        if compression:
            self.assertTrue(all(enc is not None and dec is not None for enc,dec in dictionaries))
        cbytes=random.Random(101).randbytes(8192);abytes=random.Random(102).randbytes(8192)
        p.c.feed_bytes(cbytes);p.a.feed_bytes(abytes)
        p.until(lambda:len(p.got_a)>=300 and len(p.got_c)>=300)
        self.assertTrue(p.c.request_retrain(12000))
        p.until(p.ready,carrier=True)
        self.assertEqual((p.c.bit_rate,p.a.bit_rate),(12000,12000))
        self.assertEqual((p.c._v42.rate,p.a._v42.rate),(12000,12000))
        self.assertTrue(p.a.request_retrain())
        p.until(p.ready,carrier=True)
        self.assertEqual((p.c.bit_rate,p.a.bit_rate),(14400,14400))
        self.assertIs(p.c._v42,links[0]);self.assertIs(p.a._v42,links[1])
        for m,(enc,dec) in zip((p.c,p.a),dictionaries):
            self.assertIs(m._v42._enc,enc);self.assertIs(m._v42._dec,dec)
        p.until(lambda:len(p.got_a)>=len(cbytes) and len(p.got_c)>=len(abytes),carrier=True)
        self.assertEqual(bytes(p.got_a),cbytes);self.assertEqual(bytes(p.got_c),abytes)

    def test_timeout_budget_and_ineligible_cap_do_not_restart_link(self):
        m=V32Modem(use_v8=False,echo=False)
        m.connected=True;m.state='data';m.bit_rate=9600
        self.assertFalse(m.request_retrain(300))
        self.assertEqual(m.retrain_stats['attempts'],0)
        self.assertTrue(m.request_retrain())
        m._t=45*m.fs;m._retrain_block([0]*160)
        self.assertEqual(m.state,'failed');self.assertFalse(m.connected)
        self.assertFalse(m.retrain_stats['active'])
        m=V32Modem(use_v8=False,echo=False)
        m.connected=True;m.state='data';m._retrain_attempts=2
        m._retrain_budget.take();m._retrain_budget.take()
        self.assertFalse(m.request_retrain())
        self.assertEqual(m.state,'failed');self.assertFalse(m.connected)
