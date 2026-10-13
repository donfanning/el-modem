"""Default-parameter establishment: wire sequence, loss and payload controls."""
import os
import unittest
from unittest.mock import patch

from v42_lapm import DEFAULT_K, DEFAULT_N401, N400, SABME, UA, XID, V42Link, build_xid
from test_v42_lapm import pump


class RecordingLink(V42Link):
    def __init__(self, *args, **kwargs):
        self.frames = []
        super().__init__(*args, **kwargs)

    def _send(self, frame):
        self.frames.append(frame)
        super()._send(frame)


class LoseFirstUa(RecordingLink):
    def __init__(self, *args, **kwargs):
        self.ua_count = 0
        super().__init__(*args, **kwargs)

    def _u(self, ctrl, command, pf=0, info=b''):
        if ctrl == UA and not command:
            self.ua_count += 1
            if self.ua_count == 1:
                return
        super()._u(ctrl, command, pf, info)


class DefaultParameterTests(unittest.TestCase):
    def test_defaults_establish_and_deliver_both_ways_with_and_without_detection(self):
        for detect in (False, True):
            with self.subTest(detect=detect), patch.dict(os.environ, {
                    'V42_DEFAULT_PARAMETERS': '1', 'V42_ASSUME_BIS': '0'}):
                a = RecordingLink(True, detect=detect, frame_size=32, window=4)
                b = RecordingLink(False, detect=detect, frame_size=32, window=4)
                a.write(bytes(range(256)) * 3)
                b.write(bytes(reversed(range(256))) * 3)
                got_a, got_b = pump(a, b, 9600 * 4)
                self.assertEqual(got_a, bytes(reversed(range(256))) * 3)
                self.assertEqual(got_b, bytes(range(256)) * 3)
                self.assertEqual(a.frames[0], bytes((a.cmd_addr, SABME | 0x10)))
                self.assertEqual(a.stats['default_establishment'], 1)
                for link in (a, b):
                    self.assertEqual(link.state, 'lapm')
                    self.assertEqual((link.n401, link.k), (DEFAULT_N401, DEFAULT_K))
                    self.assertFalse(any(f[1] & 0xEF == XID for f in link.frames))
                    self.assertFalse(link._xid_fallback)
                    self.assertIsNone(link.compression)
                    self.assertIsNone(link._enc)
                    self.assertIsNone(link._dec)

    def test_first_ua_loss_retries_sabme_and_recovers_payload(self):
        with patch.dict(os.environ, {'V42_DEFAULT_PARAMETERS': '1'}):
            a, b = RecordingLink(True, detect=False), LoseFirstUa(False, detect=False)
            a.write(b'caller payload' * 30)
            b.write(b'answerer payload' * 30)
            got_a, got_b = pump(a, b, 9600 * 5)
        self.assertEqual((got_a, got_b), (b'answerer payload' * 30, b'caller payload' * 30))
        self.assertGreaterEqual(b.ua_count, 2)
        self.assertEqual(a.stats['xid_tx'], 0)
        self.assertGreaterEqual(a.stats['t401_expiries'], 1)

    def test_unanswered_sabme_terminates_without_data_or_xid(self):
        with patch.dict(os.environ, {'V42_DEFAULT_PARAMETERS': '1'}):
            link = RecordingLink(True, detect=False)
            link.write(b'must not precede establishment')
            for _ in range(9600 * 20):
                link.tx_bit()
                link.rx_bit(1)
        self.assertEqual(link.state, 'down')
        self.assertEqual(link.stats['t401_expiries'], N400 + 1)
        self.assertEqual(link.stats['i_tx'], 0)
        self.assertEqual(link.stats['xid_tx'], 0)

    def test_compression_offer_is_rejected_in_default_mode(self):
        with patch.dict(os.environ, {'V42_DEFAULT_PARAMETERS': '1'}):
            for calling in (True, False):
                with self.subTest(calling=calling), self.assertRaisesRegex(ValueError, 'compression disabled'):
                    V42Link(calling, compression=(3, 1024, 32))

    def test_unsolicited_xid_response_does_not_change_defaults(self):
        with patch.dict(os.environ, {'V42_DEFAULT_PARAMETERS': '1'}):
            link = RecordingLink(True, detect=False)
            link._on_frame(bytes((link.cmd_addr, XID)) + build_xid(32, 4, (3, 1024, 32)))
        self.assertEqual((link.n401, link.k), (DEFAULT_N401, DEFAULT_K))
        self.assertEqual(link.frames, [])
        self.assertFalse(link._xid_done)

    def test_peer_xid_command_is_answered_without_compression(self):
        with patch.dict(os.environ, {'V42_DEFAULT_PARAMETERS': '1'}):
            link = RecordingLink(True, detect=False)
            link._on_frame(bytes((link.rsp_addr, XID)) + build_xid(32, 4, (3, 1024, 32)))
        self.assertEqual((link.n401, link.k), (32, 4))
        self.assertEqual(link.frames[0][1], XID)
        self.assertIsNone(link._agreed)

    def test_disabled_option_preserves_requested_parameters_and_xid(self):
        with patch.dict(os.environ, {'V42_DEFAULT_PARAMETERS': '0'}):
            a, b = RecordingLink(True, detect=False, frame_size=32, window=4), RecordingLink(False, detect=False)
            pump(a, b, 9600 * 2)
        self.assertEqual((a.n401, a.k), (32, 4))
        self.assertEqual(a.frames[0][1], XID)
        self.assertEqual(a.stats['default_establishment'], 0)


if __name__ == '__main__':
    unittest.main()
