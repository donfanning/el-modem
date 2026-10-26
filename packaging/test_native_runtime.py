import unittest
from native_runtime import NATIVE_ENV, override


class NativeRuntimeTests(unittest.TestCase):
    def test_compose_entrypoint_and_capture_identity(self):
        service = override('sha256:example', '/tmp/cap/trial',
                           'media.example.test', 'sip.example.test')['services']['modem']
        self.assertEqual(service['entrypoint'], [])
        self.assertEqual(service['pull_policy'], 'never')
        self.assertEqual(service['command'][:3], ['python3', '-u', '/app/modem_dce.py'])
        for flag, suffix in [('--capture', '.pcm'), ('--dsp-capture', '.dsp'),
                             ('--metrics-file', '.metrics.json')]:
            self.assertEqual(service['command'][service['command'].index(flag)+1],
                             '/tmp/cap/trial'+suffix)
        self.assertEqual(service['environment']['V34_CLOCK_FIT_DUMP'], '/tmp/cap/trial')
        self.assertEqual(service['environment']['V34_TIMING_MODE'], 'centroid')
        self.assertEqual(service['environment']['V42_CERROR_RECOVERY'], '0')

    def test_profiles_do_not_share_mutable_settings(self):
        first = override('one', '/tmp/cap/one', 'media.test', 'sip.test')['services']['modem']
        first['environment']['OPENBLAS_NUM_THREADS'] = '99'
        second = override('two', '/tmp/cap/two', 'media.test', 'sip.test')['services']['modem']
        self.assertEqual(second['environment']['OPENBLAS_NUM_THREADS'], '1')
        self.assertEqual(NATIVE_ENV['OPENBLAS_NUM_THREADS'], '1')

    def test_refuse_prefix_that_cannot_be_archived(self):
        for prefix in ['relative', '/tmp/elsewhere/name', '/tmp/cap', '/tmp/cap/../lost']:
            with self.subTest(prefix=prefix), self.assertRaises(ValueError):
                override('candidate', prefix, 'media.test', 'sip.test')


if __name__ == '__main__':
    unittest.main()
