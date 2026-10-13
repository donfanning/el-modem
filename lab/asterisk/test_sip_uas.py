#!/usr/bin/env python3
"""Exercise incoming SIP call signaling through the modem DCE."""

import socket
import threading
import unittest
import uuid

from modem_dce import AtDce
from sip_uas import SipIncomingCallServer, parse_message


class SipIncomingCallTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.client.bind(("127.0.0.1", 0))
        self.client.settimeout(3)
        self.client_ip, self.client_port = self.client.getsockname()
        self.call_id = f"{uuid.uuid4()}@test"
        self.from_tag = uuid.uuid4().hex[:12]
        self.cseq = 1
        self.dce = AtDce()
        self.dce.feed(b"ATE0\r")
        self.server = SipIncomingCallServer(
            "127.0.0.1",
            0,
            "127.0.0.1",
            16000,
            self.dce.begin_incoming_call,
            self.dce.notify_incoming_ring,
            lambda: self.dce.notify_call_progress("NO CARRIER"),
            ring_interval=0.2,
        )
        self.server.start()
        self.server.port = self.server._socket.getsockname()[1]
        self.dce.attach_call_control(self.server.answer, self.server.hangup)
        self.target = ("127.0.0.1", self.server.port)

    def tearDown(self) -> None:
        self.server.close()
        self.client.close()

    def _invite(self) -> bytes:
        branch = "z9hG4bK" + uuid.uuid4().hex
        sdp = (
            "v=0\r\n"
            f"o=test 1 1 IN IP4 {self.client_ip}\r\n"
            "s=test\r\n"
            f"c=IN IP4 {self.client_ip}\r\n"
            "t=0 0\r\n"
            f"m=audio {self.client_port} RTP/AVP 0\r\n"
            "a=rtpmap:0 PCMU/8000\r\n"
        )
        return (
            f"INVITE sip:modem@127.0.0.1:{self.server.port} SIP/2.0\r\n"
            f"Via: SIP/2.0/UDP {self.client_ip}:{self.client_port};branch={branch};rport\r\n"
            f"From: <sip:caller@127.0.0.1>;tag={self.from_tag}\r\n"
            f"To: <sip:modem@127.0.0.1:{self.server.port}>\r\n"
            f"Call-ID: {self.call_id}\r\n"
            f"CSeq: {self.cseq} INVITE\r\n"
            f"Contact: <sip:caller@{self.client_ip}:{self.client_port}>\r\n"
            "Content-Type: application/sdp\r\n"
            f"Content-Length: {len(sdp.encode())}\r\n\r\n{sdp}"
        ).encode()

    def _recv(self) -> tuple[dict[str, str], str]:
        data, _ = self.client.recvfrom(65535)
        start_line, headers, body = parse_message(data)
        return headers, start_line

    def test_incoming_invite_rings_until_serial_ata_then_hangup(self) -> None:
        self.client.sendto(self._invite(), self.target)
        statuses = []
        while not any("180 Ringing" in status for status in statuses):
            headers, start = self._recv()
            if start.startswith("SIP/2.0"):
                statuses.append(start)
        self.assertTrue(any("100 Trying" in status for status in statuses))
        self.assertTrue(self.dce.incoming_ringing)
        self.assertEqual(self.dce.take_notifications(), b"\r\nRING\r\n")

        answer_result: list[bytes] = []
        answer_thread = threading.Thread(
            target=lambda: answer_result.append(self.dce.feed(b"ATA\r"))
        )
        answer_thread.start()
        response = b""
        response_headers: dict[str, str] = {}
        while not response.startswith(b"SIP/2.0 200"):
            response, _ = self.client.recvfrom(65535)
            _, response_headers, _ = parse_message(response)
        self.assertIn("m=audio 16000 RTP/AVP 0", response.decode())
        to_header = response_headers["to"]
        ack = (
            f"ACK sip:modem@127.0.0.1:{self.server.port} SIP/2.0\r\n"
            f"Via: SIP/2.0/UDP {self.client_ip}:{self.client_port};branch=z9hG4bKack\r\n"
            f"From: <sip:caller@127.0.0.1>;tag={self.from_tag}\r\n"
            f"To: {to_header}\r\n"
            f"Call-ID: {self.call_id}\r\n"
            f"CSeq: {self.cseq} ACK\r\n"
            "Content-Length: 0\r\n\r\n"
        )
        self.client.sendto(ack.encode(), self.target)
        answer_thread.join(timeout=3)
        self.assertFalse(answer_thread.is_alive())
        self.assertEqual(answer_result, [b""])
        self.assertTrue(self.dce.online)
        self.assertEqual(self.dce.take_notifications(), b"\r\nCONNECT 300\r\n")

        bye = (
            f"BYE sip:modem@127.0.0.1:{self.server.port} SIP/2.0\r\n"
            f"Via: SIP/2.0/UDP {self.client_ip}:{self.client_port};branch=z9hG4bKbye\r\n"
            f"From: <sip:caller@127.0.0.1>;tag={self.from_tag}\r\n"
            f"To: {to_header}\r\n"
            f"Call-ID: {self.call_id}\r\n"
            "CSeq: 2 BYE\r\n"
            "Content-Length: 0\r\n\r\n"
        )
        self.client.sendto(bye.encode(), self.target)
        _, bye_status = self._recv()
        self.assertEqual(bye_status, "SIP/2.0 200 OK")
        self.assertEqual(self.dce.take_notifications(), b"\r\nNO CARRIER\r\n")
        self.assertFalse(self.dce.online)


if __name__ == "__main__":
    unittest.main()
