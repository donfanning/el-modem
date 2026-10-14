"""Gate regressions: missing call-time modules, network isolation and exact data."""
import builtins
import os
import unittest
from unittest.mock import patch
from modem_dce import AtDce, OnlineAudioSession, line_from_ms
from modem_selftest import profile_options, run, make_session


class ArtifactGateTests(unittest.TestCase):
    def test_automatic_v34_capture_attaches_before_engine_pcm(self):
        with patch.dict(os.environ, {'EL_MODEM_DSP_CAPTURE': '/unused/test'}):
            options = profile_options(AtDce, line_from_ms, 'V34,1,300,33600')
            session = make_session(OnlineAudioSession, options, True)
        session._v22._choose('v34')
        with patch('v34_diagnostic.configured') as configured:
            configured.return_value.feed.return_value = b''
            session._receive_pcm(bytes(320))
            configured.assert_called_once_with(session._v22.engine)
            configured.return_value.feed.assert_called_once_with(bytes(320), session.receiver)
        # The second block reuses the capture rather than reopening its files.
        with patch('v34_diagnostic.configured') as configured:
            session._receive_pcm(bytes(320))
            configured.assert_not_called()
        session.stop()

    def test_all_deferred_factories_without_network_or_capture(self):
        with patch.dict(os.environ, {'MODEM_CAPTURE': '/unwritable/offline-test', 'V34_RATE': '9600'}):
            with patch('socket.socket', side_effect=AssertionError('network socket opened')):
                report = run(OnlineAudioSession, line_from_ms, AtDce)
            self.assertEqual(os.environ['MODEM_CAPTURE'], '/unwritable/offline-test')
            self.assertEqual(os.environ['V34_RATE'], '9600')
        self.assertEqual(report['status'], 'PASS', report)
        self.assertEqual(len(report['gates']), 22)
        self.assertEqual({g['selected'] for g in report['gates'] if 'selected' in g}, {'v22', 'v32', 'v34'})

    def test_missing_call_time_module_fails_gate(self):
        original = builtins.__import__
        def unavailable(name, *args, **kwargs):
            if name == 'v34_realtime':
                raise ModuleNotFoundError('No module named v34_realtime')
            return original(name, *args, **kwargs)
        with patch('builtins.__import__', side_effect=unavailable):
            report = run(OnlineAudioSession, line_from_ms, AtDce)
        self.assertEqual(report['status'], 'FAIL')
        self.assertIn('v34_realtime', report['error'])

    def test_invalid_at_profile_cannot_enter_factory(self):
        for profile in ('V34,9,300,33600', 'V34,1,33600,300', 'V34,1,299,33600',
                        'V34,1,300,33601', 'V34,1,300,33600,7'):
            with self.subTest(profile=profile), self.assertRaisesRegex(AssertionError, 'AT profile rejected'):
                profile_options(AtDce, line_from_ms, profile)
        dce = AtDce(); previous = dce.ms.copy()
        self.assertIn(b'ERROR', dce.feed(b'AT+MS=V34,9,300,33600\r'))
        self.assertEqual(dce.ms, previous)

    def test_modulation_reset_and_saved_profile_recall(self):
        dce = AtDce()
        factory = dce.ms.copy()
        dce.feed(b'AT+MS=V34,1,300,33600\rATZ\r')
        self.assertEqual(dce.ms, factory)
        dce.feed(b'AT+MS=V34,1,300,33600\rAT&W0\r')
        saved = dce.ms.copy()
        dce.feed(b'AT+MS=V32,0,4800,9600\rATZ\r')
        self.assertEqual(dce.ms, saved)
        dce.feed(b'AT&F\r')
        self.assertEqual(dce.ms, factory)
        dce.feed(b'ATZ\r')
        self.assertEqual(dce.ms, saved)

    def test_full_profile_payload_and_error_control_without_network(self):
        with patch('socket.socket', side_effect=AssertionError('network socket opened')):
            report = run(OnlineAudioSession, line_from_ms, AtDce, with_payload=True)
        self.assertEqual(report['status'], 'PASS', report.get('error'))
        payload = [g for g in report['gates'] if g['gate'] == 'pcm_payload']
        self.assertEqual(len(payload), 11)
        self.assertEqual({g['protocol'] for g in payload if 'protocol' in g}, {'lapm', 'v42bis', 'mnp'})
        automatic = [g for g in payload if g.get('peer_menu') == 'v34'][0]
        self.assertEqual(automatic['rates'], [33600, 33600])


if __name__ == '__main__':
    unittest.main()
