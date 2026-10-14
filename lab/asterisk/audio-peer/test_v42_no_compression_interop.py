"""Independent LAPM reference offering compression to our compression-off link.

Run with a SpanDSP library built by the lab audio-peer Dockerfile. These
bit-stream controls are not DSP, RTP, timing or remote-service qualification.
"""
import os
import unittest
from unittest.mock import patch

import v42_lapm
from test_v42_lapm import _ref_class, pump


@unittest.skipUnless(_ref_class(), 'SpanDSP reference library unavailable')
class NoCompressionInteropTests(unittest.TestCase):
    def exercise(self, calling, default_parameters, detect):
        with patch.dict(os.environ, {'V42_DEFAULT_PARAMETERS': str(int(default_parameters)),
                                     'V42_ASSUME_BIS': '0', 'V42_XID_FALLBACK': '1'}):
            ours = v42_lapm.V42Link(calling, detect=detect, bit_rate=9600,
                                    frame_size=32, window=4, compression=None)
            reference = _ref_class()(not calling, detect=detect, bit_rate=9600,
                                     compression=(3, 1024, 8))
            self.assertTrue(hasattr(reference._lib, 'v42_set_compression'),
                            'reference must actually support offering compression')
            pump(ours, reference, 9600 * 3)
            forward = bytes(range(256)) * 5 + b'caller-independent binary tail'
            reverse = b'Login:\r\nPassword:\r\n' * 70 + bytes(range(255, -1, -1))
            ours.write(forward)
            reference.write(reverse)
            received, reference_received = pump(ours, reference, 9600 * 6)
        self.assertEqual((ours.state, reference.state), ('lapm', 'lapm'))
        self.assertEqual((received, reference_received), (reverse, forward))
        self.assertIsNone(ours.compression)
        self.assertIsNone(reference.compression)
        self.assertIsNone(ours._enc)
        self.assertIsNone(ours._dec)
        self.assertFalse(ours._xid_fallback)
        if calling:
            if default_parameters:
                self.assertEqual(ours.stats['xid_tx'], 0)
            else:
                self.assertGreaterEqual(ours.stats['xid_tx'], 1)
                self.assertLessEqual(ours.stats['xid_tx'], v42_lapm.XID_ATTEMPTS)
            self.assertEqual(ours.stats['default_establishment'], int(default_parameters))

    def test_originator_declines_reference_compression_offer(self):
        for detect in (False, True):
            with self.subTest(detect=detect):
                self.exercise(True, False, detect)

    def test_answerer_declines_reference_compression_offer(self):
        for detect in (False, True):
            with self.subTest(detect=detect):
                self.exercise(False, False, detect)

    def test_default_parameter_originator_keeps_reference_uncompressed(self):
        for detect in (False, True):
            with self.subTest(detect=detect):
                self.exercise(True, True, detect)

    def test_default_parameter_answerer_declines_reference_xid_offer(self):
        for detect in (False, True):
            with self.subTest(detect=detect):
                self.exercise(False, True, detect)


if __name__ == '__main__':
    unittest.main()
