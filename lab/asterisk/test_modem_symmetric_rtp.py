"""Symmetric RTP destination latching; no network or call needed."""
import socket
import unittest

from modem_dce import OnlineAudioSession


def make_session():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", 0))
    session = OnlineAudioSession(
        sock, ("10.9.9.9", 10000), "originate", "answer",
        negotiate=False, line_mode="bell103",
    )
    session._stop.set()  # never started; make doubly sure no thread runs
    return session, sock


class SymmetricRtpTests(unittest.TestCase):
    def test_initial_destination_is_sdp_address(self):
        session, sock = make_session()
        try:
            self.assertEqual(session._send_addr, ("10.9.9.9", 10000))
        finally:
            sock.close()

    def test_latches_to_observed_source(self):
        session, sock = make_session()
        try:
            session.note_rtp_source(("192.168.192.21", 10004))
            self.assertEqual(session._send_addr, ("192.168.192.21", 10004))
            # NAT rebinding: latest valid source wins
            session.note_rtp_source(("192.168.192.21", 10008))
            self.assertEqual(session._send_addr, ("192.168.192.21", 10008))
        finally:
            sock.close()

    def test_rejects_garbage(self):
        session, sock = make_session()
        try:
            for bad in (None, (), ("host",), ("", 10000), ("0.0.0.0", 10000),
                        ("host", 0), ("host", 65536), ("host", -1),
                        ("host", "notaport"), (12345, 10000)):
                session.note_rtp_source(bad)
                self.assertEqual(session._send_addr, ("10.9.9.9", 10000),
                                 "accepted %r" % (bad,))
        finally:
            sock.close()


if __name__ == "__main__":
    unittest.main()
