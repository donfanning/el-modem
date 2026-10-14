"""Real PCM/DTE online rate commands preserve pending LAPM application data."""
import random
import unittest
from types import SimpleNamespace
from modem_dce import AtDce
from test_v32_retrain import Pair
from v34_online_control import OnlineControl


class OnlineRateTests(unittest.TestCase):
    def test_commands_in_both_roles_wait_for_exchange_and_preserve_duplex_queue(self):
        for role in ('call', 'answer'):
            with self.subTest(role=role):
                p = Pair()
                p.until(p.ready)
                p.c.start_v42(True, detect=False)
                p.a.start_v42(False, detect=False)
                p.until(lambda: p.c.v42_mode == p.a.v42_mode == 'lapm')
                owners = [OnlineControl(lambda m=m: m, allow_v32_rate=True) for m in (p.c, p.a)]
                for owner in owners:
                    owner.activate()
                initiator = p.c if role == 'call' else p.a
                owner = owners[0 if role == 'call' else 1]
                dce = AtDce()
                dce.feed(b'ATE0V1X4\r')
                dce.online = True
                dce.attach_online_control(owner.request)
                dce.attach_resume_online(lambda: None)
                links = (p.c._v42, p.a._v42)
                taps = (p.c.eq, p.a.eq)
                expected_c, expected_a = bytearray(), bytearray()
                for command, rate in ((3, 14400), (4, 12000), (5, 14400)):
                    a = random.Random(7100+command).randbytes(4096)
                    b = random.Random(8100+command).randbytes(4096)
                    expected_a.extend(a); expected_c.extend(b)
                    p.c.feed_bytes(a); p.a.feed_bytes(b)
                    self.assertEqual(dce.feed(f'ATO{command}\r'.encode()), b'')
                    self.assertIsNone(owner.poll())
                    self.assertTrue(dce.online_return_pending)
                    self.assertEqual(dce.feed(b'ATO4\r'), b'\r\nERROR\r\n')
                    records = []
                    def finished():
                        r = owner.poll()
                        if r is not None:
                            records.append(r)
                        return bool(records)
                    p.until(finished, carrier=True)
                    result, = records
                    self.assertEqual(result['status'], 'rate-exchanged')
                    self.assertEqual(result['procedure'], 'V32bis-section-8')
                    self.assertEqual((result['tx_rate'], result['rx_rate']), (rate, rate))
                    self.assertEqual(result['exchange']['status'], 'complete')
                    self.assertNotIn('_transaction', result)
                    dce.finish_online_control(result['tx_rate'], '/LAPM')
                    self.assertEqual(dce.take_notifications(), f'\r\nCONNECT {rate}/LAPM\r\n'.encode())
                    p.until(lambda: len(p.got_a) >= len(expected_a) and len(p.got_c) >= len(expected_c), carrier=True)
                    self.assertEqual(p.got_a, expected_a)
                    self.assertEqual(p.got_c, expected_c)
                    self.assertIsNone(owner.poll())
                self.assertEqual((p.c._v42, p.a._v42), links)
                self.assertEqual((p.c.eq, p.a.eq), taps)
                self.assertEqual((p.c.retrain_stats['attempts'], p.a.retrain_stats['attempts']), (0, 0))

    def fake(self):
        e = SimpleNamespace(is_v32=True, state='data', connected=True, v42_mode='lapm',
                            bit_rate=9600, tx=SimpleNamespace(data_rate=9600),
                            _configured_rates=(14400, 9600, 4800), _rrp=None,
                            retrain_stats=dict(attempts=0, active=False, events=[]), failed_reason='')
        def enable():
            if e._rrp is None: e._rrp = SimpleNamespace(active=False, events=[])
            return True
        def request(rate):
            e._rrp.active = True
            e._rrp.events.append(dict(status='pending', requested_rate=rate))
            return True
        e.enable_rate_changes = enable
        e.request_rate_change = request
        return e

    def test_configured_gaps_edges_raw_and_opt_in(self):
        for parameter, target in ((3,9600),(4,4800),(5,14400)):
            e=self.fake();o=OnlineControl(lambda:e,allow_v32_rate=True)
            self.assertTrue(o.request(parameter))
            self.assertEqual(o.report()['pending']['requested_cap'],target)
            self.assertNotIn('_transaction',o.report()['pending'])
        for mode,rate in (('raw',9600),('lapm',4800),('lapm',14400)):
            e=self.fake();e.v42_mode=mode;e.bit_rate=rate
            o=OnlineControl(lambda:e,allow_v32_rate=True)
            self.assertFalse(o.request(4 if rate==4800 else 5))
        e=self.fake();self.assertFalse(OnlineControl(lambda:e).request(3))

    def test_o1_snapshot_precedes_line_restart_and_is_detached(self):
        from v42_lapm import V42Link
        e=self.fake();e._v42=V42Link(True,detect=False)
        e._v42._unacked.append([7,b'pending'])
        e._v42._txq.extend(b'queued')
        e._v42._tx.extend([0,1,0])
        e._txn=12345
        e.bb=SimpleNamespace(n=12000,pos=11982.5,step=5/3)
        e._k=7180;e._ph_acc=.125;e._freq=.0003;e._cofs=2
        def restart():
            e._v42._unacked.clear();e._txn=0
            return True
        e.request_retrain=restart
        owner=OnlineControl(lambda:e)
        self.assertTrue(owner.request(1))
        snapshot=owner.report()['pending']['request_link_snapshot']
        self.assertEqual(snapshot['tx_samples'],12345)
        self.assertEqual(snapshot['rx_samples'],12000)
        self.assertEqual(snapshot['v32_receive']['next_half_symbol_sample'],11982.5)
        self.assertEqual(snapshot['v32_receive']['next_symbol_index'],7180)
        self.assertEqual(snapshot['lapm']['outstanding_sequences'],[7])
        self.assertEqual(snapshot['lapm']['queued_application_bytes'],6)
        self.assertEqual(snapshot['lapm']['queued_wire_bits'],3)
        self.assertEqual(e._v42._unacked,[])
        snapshot['lapm']['outstanding_sequences'].clear()
        self.assertEqual(owner.report()['pending']['request_link_snapshot']['lapm']['outstanding_sequences'],[7])

    def test_superseding_full_retrain_waits_and_reports_fallback_not_fast_success(self):
        e=self.fake();o=OnlineControl(lambda:e,allow_v32_rate=True)
        self.assertTrue(o.request(4))
        e.retrain_stats=dict(attempts=1,active=True,events=[dict(reason='peer request')])
        e._rrp=None
        self.assertIsNone(o.poll())
        e.retrain_stats['active']=False
        e.retrain_stats['events'][-1]['completed_seconds']=7
        e.bit_rate=e.tx.data_rate=4800
        self.assertEqual(o.poll()['status'],'fallback-retrained')
        self.assertIsNone(o.poll())

    def test_carrier_failure_is_terminal_once(self):
        e=self.fake();o=OnlineControl(lambda:e,allow_v32_rate=True)
        self.assertTrue(o.request(3))
        e.state='failed';e.connected=False;e.failed_reason='rate timeout'
        result=o.poll()
        self.assertEqual((result['status'],result['reason']),('failed','rate timeout'))
        self.assertIsNone(o.poll())

    def test_failed_superseding_recovery_is_not_reported_as_fallback_success(self):
        e=self.fake();o=OnlineControl(lambda:e,allow_v32_rate=True)
        self.assertTrue(o.request(3))
        e.retrain_stats=dict(attempts=1,active=False,
                            events=[dict(completed_seconds=7,failure='training failed')])
        self.assertEqual(o.poll()['status'],'failed')
        self.assertIsNone(o.poll())

    def test_following_peer_exchange_must_settle_before_connect(self):
        e=self.fake();o=OnlineControl(lambda:e,allow_v32_rate=True)
        self.assertTrue(o.request(3))
        e._rrp.events[-1]['status']='complete'
        e._rrp.events.append(dict(origin='peer',status='pending'))
        self.assertIsNone(o.poll())
        e._rrp.active=False
        e._rrp.events[-1].update(status='complete',rate=4800)
        e.bit_rate=e.tx.data_rate=4800
        result=o.poll()
        self.assertEqual(result['tx_rate'],4800)
        self.assertEqual(result['following_exchange']['rate'],4800)
