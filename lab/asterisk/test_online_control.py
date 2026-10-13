"""Online commands must represent DSP completion, preserve the call, and reject ambiguity."""
import unittest
from modem_dce import AtDce

class OnlineCommands(unittest.TestCase):
    def test_automatic_modes_query_store_and_reset(self):
        dce=AtDce();dce.feed(b'ATE0V1\r')
        self.assertEqual(dce.auto_rate_mode,0)
        self.assertIn(b'OK',dce.feed(b'AT%E2&W0\r'))
        dce.feed(b'AT%E0\r');self.assertEqual(dce.auto_rate_mode,0)
        dce.feed(b'ATZ\r');self.assertEqual(dce.auto_rate_mode,2)
        self.assertEqual(dce.feed(b'AT%E?\r'),b'\r\n2\r\nOK\r\n')
        self.assertEqual(dce.feed(b'AT%E3\r'),b'\r\nERROR\r\n')
        dce.feed(b'AT&F\r');self.assertEqual(dce.auto_rate_mode,0)

    def setUp(self):
        self.dce=AtDce()
        self.dce.feed(b'ATE0V1X4\r')
        self.dce.online=True
        self.actions=[];self.resumed=[]
        self.dce.attach_online_control(lambda n:self.actions.append(n) or True)
        self.dce.attach_resume_online(lambda:self.resumed.append(True))

    def test_deferred_connect_uses_actual_rate_and_preserves_call_accounting(self):
        self.dce.notify_call_progress('CONNECT');self.dce.take_notifications()
        call=self.dce._call
        self.dce.count_chars(sent=19,received=23)
        self.assertEqual(self.dce.feed(b'ATO4\r'),b'')
        self.assertEqual(self.actions,[4]);self.assertEqual(self.resumed,[])
        self.assertEqual(self.dce.feed(b'ATO5\r'),b'\r\nERROR\r\n')
        self.dce.finish_online_control(31200,'/LAPM')
        self.assertEqual(self.dce.take_notifications(),b'\r\nCONNECT 31200/LAPM\r\n')
        self.assertEqual(self.resumed,[True]);self.assertIs(self.dce._call,call)
        self.assertEqual((call['chars_sent'],call['chars_received']),(19,23))

    def test_all_supported_actions_are_parsed_and_dispatched(self):
        for n in (1,3,4,5):
            self.assertEqual(self.dce.feed(f'ATO{n}\r'.encode()),b'')
            self.assertEqual(self.actions[-1],n)
            self.dce.finish_online_control(33600,'');self.dce.take_notifications()
        for cmd in (b'ATO2\r',b'ATO6\r',b'ATO10\r',b'ATO3garbage\r',b'ATO3H\r'):
            self.assertEqual(self.dce.feed(cmd),b'\r\nERROR\r\n')
        self.assertEqual(self.actions,[1,3,4,5])

    def test_resume_only_and_unsupported_or_disconnected_engine(self):
        self.assertIn(b'CONNECT',self.dce.feed(b'ATO0\r'))
        self.assertEqual(self.actions,[])
        self.dce.attach_online_control(lambda n:False)
        self.assertEqual(self.dce.feed(b'ATO1\r'),b'\r\nERROR\r\n')
        self.assertFalse(self.dce.online_return_pending)
        self.dce.online=False
        self.assertEqual(self.dce.feed(b'ATO4\r'),b'\r\nNO CARRIER\r\n')


class V32OnlineControl(unittest.TestCase):
    def engine(self):
        from types import SimpleNamespace
        e=SimpleNamespace(is_v32=True, connected=True, state='data', bit_rate=9600,
                          tx=SimpleNamespace(data_rate=9600), _retraining=False,
                          retrain_stats=dict(attempts=0,active=False,events=[]),failed_reason='')
        def request():
            e._retraining=True;e.state='aa'
            e.retrain_stats=dict(attempts=1,active=True,events=[dict(reason='local request',offered_rates=[9600])])
            return True
        e.request_retrain=request
        return e

    def test_o1_waits_for_physical_completion_and_preserves_command_accounting(self):
        from v34_online_control import OnlineControl
        e=self.engine();owner=OnlineControl(lambda:e)
        d=AtDce();d.feed(b'ATE0V1X4\r');d.online=True
        d.attach_online_control(owner.request);d.attach_resume_online(lambda:None)
        d.notify_call_progress('CONNECT');d.take_notifications();call=d._call
        d.count_chars(sent=123,received=456)
        self.assertEqual(d.feed(b'ATO1\r'),b'')
        self.assertIsNone(owner.poll())
        self.assertEqual(d.feed(b'ATO1\r'),b'\r\nERROR\r\n')
        e._retraining=False;e.state='data';e.retrain_stats['active']=False
        e.retrain_stats['events'][-1].update(completed_seconds=7,negotiated_rate=9600)
        result=owner.poll();self.assertEqual(result['status'],'retrained')
        self.assertEqual(result['procedure'],'V32bis-section-7')
        d.finish_online_control(result['tx_rate'],'/LAPM')
        self.assertEqual(d.take_notifications(),b'\r\nCONNECT 9600/LAPM\r\n')
        self.assertIs(d._call,call);self.assertEqual((call['chars_sent'],call['chars_received']),(123,456))
        self.assertIsNone(owner.poll())

    def test_v32_o1_does_not_enable_experimental_v34_controls(self):
        from v34_online_control import OnlineControl
        from types import SimpleNamespace
        e=self.engine();owner=OnlineControl(lambda:e,allow_v34=False)
        self.assertTrue(owner.request(1))
        e2=SimpleNamespace(connected=True,is_v32=False)
        other=OnlineControl(lambda:e2,allow_v34=False)
        for n in (1,3,4,5):self.assertFalse(other.request(n))

    def test_no_silent_substitution_for_short_rate_procedures_and_failed_requests(self):
        from v34_online_control import OnlineControl
        e=self.engine();owner=OnlineControl(lambda:e)
        for n in (2,3,4,5):self.assertFalse(owner.request(n))
        self.assertFalse(owner.request(1,origin='automatic'))
        self.assertFalse(owner.request(1,rates=[9600]))
        e.request_retrain=lambda:False
        self.assertFalse(owner.request(1));self.assertIsNone(owner.pending)

    def test_failure_is_terminal_once_and_new_exchange_cannot_complete_old_command(self):
        from v34_online_control import OnlineControl
        e=self.engine();owner=OnlineControl(lambda:e);self.assertTrue(owner.request(1))
        e.retrain_stats['attempts']=2;e.retrain_stats['active']=False
        e.retrain_stats['events'][-1]['completed_seconds']=7
        r=owner.poll();self.assertEqual(r['status'],'failed');self.assertIn('superseded',r['reason'])
        self.assertIsNone(owner.poll())
        e=self.engine();owner=OnlineControl(lambda:e);self.assertTrue(owner.request(1))
        e.state='failed';e.connected=False;e.failed_reason='training timeout'
        r=owner.poll();self.assertEqual(r['status'],'failed');self.assertEqual(r['reason'],'training timeout')
        self.assertIsNone(owner.poll())
