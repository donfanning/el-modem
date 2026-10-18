"""Symmetric RTP destination latching; no network or call needed."""
import socket
import struct
import unittest
from unittest.mock import Mock

from modem_dce import OnlineAudioSession


def make_session():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", 0))
    session = OnlineAudioSession(
        sock, ("192.0.2.9", 10000), "originate", "answer",
        negotiate=False, line_mode="bell103",
    )
    session._stop.set()  # never started; make doubly sure no thread runs
    return session, sock


class SymmetricRtpTests(unittest.TestCase):
    @staticmethod
    def packet(seq, ssrc=1234):
        return struct.pack('!BBHII',0x80,0,seq,(seq*160)&0xffffffff,ssrc)+bytes(160)

    def test_initial_destination_is_sdp_address(self):
        session, sock = make_session()
        try:
            self.assertEqual(session._send_addr, ("192.0.2.9", 10000))
        finally:
            sock.close()

    def test_latches_to_observed_source(self):
        session, sock = make_session()
        try:
            # Documentation-only TEST-NET address; never expose lab topology.
            addr=("192.0.2.21",10004)
            for seq in range(2):
                self.assertEqual(session.note_rtp_source(addr,self.packet(seq)),[])
                self.assertEqual(session._send_addr,("192.0.2.9",10000))
            accepted=session.note_rtp_source(addr,self.packet(2))
            self.assertEqual([x[0] for x in accepted],[self.packet(x) for x in range(3)])
            self.assertEqual(session._send_addr, ("192.0.2.21", 10004))
            # A later source cannot hijack the media destination or DSP.
            for seq in range(3,6):
                self.assertEqual(session.note_rtp_source((addr[0],10008),self.packet(seq)),[])
                self.assertEqual(session.note_rtp_source(addr,self.packet(seq,5678)),[])
            self.assertEqual(session._send_addr, addr)
        finally:
            sock.close()

    def test_rejects_garbage(self):
        session, sock = make_session()
        try:
            for bad in (None, (), ("host",), ("", 10000), ("0.0.0.0", 10000),
                        ("host", 0), ("host", 65536), ("host", -1),
                        ("host", "notaport"), (12345, 10000), ('224.0.0.1',1234)):
                self.assertEqual(session.note_rtp_source(bad,self.packet(0)),[])
                self.assertEqual(session._send_addr, ("192.0.2.9", 10000),
                                 "accepted %r" % (bad,))
        finally:
            sock.close()

    def test_probation_rejects_gaps_duplicates_and_stale_streams(self):
        session,sock=make_session();addr=('127.0.0.1',1234)
        try:
            for seq,at in ((0,1.),(0,1.02),(3,1.04),(4,3.)):
                self.assertEqual(session.note_rtp_source(addr,self.packet(seq),at),[])
            self.assertIsNone(session._rtp_peer)
            self.assertEqual(session.note_rtp_source(addr,self.packet(5),3.02),[])
            self.assertEqual(len(session.note_rtp_source(addr,self.packet(6),3.04)),3)
        finally:sock.close()

    def test_wrap_and_buffered_metadata_reach_decoder_once(self):
        session,sock=make_session();addr=('127.0.0.1',1234)
        session.handle_rtp_packet=Mock(return_value=b'output')
        try:
            for index,seq in enumerate((65535,0,1)):
                result=session.handle_rtp_datagram(addr,self.packet(seq),10.+index*.02,{'index':index})
                self.assertEqual(result,b'' if index<2 else b'output'*3)
            self.assertEqual(session.handle_rtp_packet.call_count,3)
            for index,call in enumerate(session.handle_rtp_packet.call_args_list):
                self.assertEqual(call.kwargs,dict(received_at=10.+index*.02,clock_context={'index':index}))
            session.handle_rtp_datagram(('127.0.0.2',1234),self.packet(2))
            self.assertEqual(session.handle_rtp_packet.call_count,3)
            session.handle_rtp_datagram(addr,self.packet(2))
            self.assertEqual(session.handle_rtp_packet.call_count,4)
        finally:sock.close()

    def test_unsupported_headers_and_empty_payload_do_not_learn(self):
        session,sock=make_session();addr=('127.0.0.1',1234)
        try:
            for seq in range(5):
                packet=self.packet(seq)
                for bad in (b'',packet[:12],bytes([0x81])+packet[1:],
                            bytes([0x90])+packet[1:],bytes([0xa0])+packet[1:],
                            packet[:1]+bytes([8])+packet[2:]):
                    self.assertEqual(session.note_rtp_source(addr,bad),[])
            self.assertIsNone(session._rtp_peer)
        finally:sock.close()


if __name__ == "__main__":
    unittest.main()
