"""Offline startup ordering checks; no transport or timing qualification."""
import contextlib
import io
import os
import sys
import unittest
from unittest import mock

import modem_dce


class ResidentPreparationTests(unittest.TestCase):
    def run_main(self, args, preimport, serve):
        with mock.patch.dict(os.environ, {}, clear=True), \
                mock.patch.object(sys, 'argv', ['modem_dce.py', *args]), \
                mock.patch.object(modem_dce, 'preimport_v34_modules', preimport), \
                mock.patch.object(modem_dce, 'run_pty', serve), \
                mock.patch.object(modem_dce.signal, 'signal'):
            modem_dce.main()

    def test_preparation_finishes_before_serving_dte(self):
        events = []
        self.run_main(['--v34-preimport'],
                      lambda: events.append(('prepared', os.getpid())),
                      lambda *a, **kw: events.append(('serve', os.getpid())))
        self.assertEqual(events, [('prepared', os.getpid()), ('serve', os.getpid())])

    def test_failed_preparation_never_serves_dte(self):
        serve = mock.Mock()
        with self.assertRaisesRegex(ImportError, 'missing dependency'):
            self.run_main(['--v34-preimport'],
                          mock.Mock(side_effect=ImportError('missing dependency')), serve)
        serve.assert_not_called()

    def test_runner_environment_prepares_resident_process(self):
        events = []
        with mock.patch.dict(os.environ, {'V34_PREIMPORT': '1'}, clear=True), \
                mock.patch.object(sys, 'argv', ['modem_dce.py']), \
                mock.patch.object(modem_dce, 'preimport_v34_modules',
                                  lambda: events.append('prepared')), \
                mock.patch.object(modem_dce, 'run_pty',
                                  lambda *a, **kw: events.append('serve')), \
                mock.patch.object(modem_dce.signal, 'signal'):
            modem_dce.main()
        self.assertEqual(events, ['prepared', 'serve'])

    def test_missing_requested_arithmetic_backend_never_opens_dte(self):
        serve=mock.Mock()
        with mock.patch.dict(os.environ, {'V34_EQUALIZER_BACKEND':'rust'}, clear=True), \
                mock.patch.dict(sys.modules, {'el_modem_dsp':None}), \
                mock.patch.object(sys,'argv',['modem_dce.py']), \
                mock.patch.object(modem_dce,'run_pty',serve):
            with self.assertRaises(ImportError):modem_dce.main()
        serve.assert_not_called()

    def test_arithmetic_warmup_precedes_dte(self):
        events=[]
        with mock.patch('v34_equalizer_backend.warmup',side_effect=lambda:events.append('arithmetic')):
            self.run_main([],lambda:None,lambda *a,**kw:events.append('serve'))
        self.assertEqual(events,['arithmetic','serve'])

    def test_separate_self_test_does_not_run_resident_preparation(self):
        prepare, serve = mock.Mock(), mock.Mock()
        with mock.patch('modem_selftest.run', return_value={'status': 'PASS'}), \
                contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as exit_result:
                self.run_main(['--self-test', '--v34-preimport'], prepare, serve)
        self.assertEqual(exit_result.exception.code, 0)
        prepare.assert_not_called()
        serve.assert_not_called()


if __name__ == '__main__':
    unittest.main()
