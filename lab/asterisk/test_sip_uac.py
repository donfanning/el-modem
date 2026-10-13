#!/usr/bin/env python3
"""Direct signaling tests for the outbound SIP user agent (SipOriginateCall)."""

import socket
import threading
import unittest
from unittest.mock import Mock,patch

from sip_uac import SipOriginateCall
from sip_uas import parse_message


class FakeUas:
    """A tiny scripted SIP UAS used only to drive SipOriginateCall's states."""

    def __init__(self) -> None:
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("127.0.0.1", 0))
        self.sock.settimeout(3)
        self.host, self.port = self.sock.getsockname()

    def close(self) -> None:
        self.sock.close()

    def recv_invite(self) -> tuple[str, dict[str, str], tuple[str, int]]:
        data, source = self.sock.recvfrom(65535)
        start, headers, _ = parse_message(data)
        assert start.startswith("INVITE"), start
        return start, headers, source

    def send(self, status: str, headers: dict[str, str], source: tuple[str, int], body: str = "") -> None:
        lines = [
            f"SIP/2.0 {status}",
            f"Via: {headers.get('via', '')}",
            f"From: {headers.get('from', '')}",
            f"To: {headers.get('to', '') + (';tag=uas' if 'tag=' not in headers.get('to', '') else '')}",
            f"Call-ID: {headers.get('call-id', '')}",
            f"CSeq: {headers.get('cseq', '')}",
        ]
        if body:
            lines.append("Content-Type: application/sdp")
        lines.extend((f"Content-Length: {len(body.encode())}", "", body))
        self.sock.sendto("\r\n".join(lines).encode(), source)

    def recv_any(self) -> tuple[str, dict[str, str], tuple[str, int]]:
        data, source = self.sock.recvfrom(65535)
        start, headers, _ = parse_message(data)
        return start, headers, source

    def send_bye(self, headers: dict[str, str], client_addr: tuple[str, int]) -> None:
        lines = [
            "BYE sip:modem@test SIP/2.0",
            f"Via: SIP/2.0/UDP {self.host}:{self.port};branch=z9hG4bKuasbye",
            f"From: {headers.get('to', '')}",
            f"To: {headers.get('from', '')}",
            f"Call-ID: {headers.get('call-id', '')}",
            "CSeq: 2 BYE",
            "Content-Length: 0",
            "",
            "",
        ]
        self.sock.sendto("\r\n".join(lines).encode(), client_addr)


class SipOriginateCallTests(unittest.TestCase):
    def setUp(self) -> None:
        self.uas = FakeUas()

    def tearDown(self) -> None:
        self.uas.close()

    def _make_call(self, on_connected=None, on_failed=None, on_hangup=None) -> SipOriginateCall:
        return SipOriginateCall(
            host="127.0.0.1",
            port=0,
            target_host=self.uas.host,
            target_port=self.uas.port,
            media_host="127.0.0.1",
            media_port=17500,
            on_connected=on_connected or (lambda addr: None),
            on_failed=on_failed or (lambda reason: None),
            on_hangup=on_hangup or (lambda: None),
        )

    def test_ringing_then_answer_then_remote_bye(self) -> None:
        connected: list[tuple[str, int]] = []
        hung_up = threading.Event()

        call = self._make_call(
            on_connected=connected.append,
            on_hangup=hung_up.set,
        )
        thread = threading.Thread(target=call.dial, args=("105",))
        thread.start()

        _, headers, source = self.uas.recv_invite()
        self.uas.send("180 Ringing", headers, source)
        sdp = (
            "v=0\r\no=uas 1 1 IN IP4 127.0.0.1\r\ns=uas\r\n"
            "c=IN IP4 127.0.0.1\r\nt=0 0\r\n"
            "m=audio 17777 RTP/AVP 0\r\na=rtpmap:0 PCMU/8000\r\n"
        )
        self.uas.send("200 OK", headers, source, body=sdp)

        ack_start, ack_headers, _ = self.uas.recv_any()
        self.assertTrue(ack_start.startswith("ACK"), ack_start)
        self.assertEqual(ack_headers.get("call-id"), headers.get("call-id"))

        for _ in range(50):
            if connected:
                break
            thread.join(timeout=0.1)
        self.assertEqual(connected, [("127.0.0.1", 17777)])

        self.uas.send_bye(headers, source)
        thread.join(timeout=3)
        self.assertFalse(thread.is_alive())
        self.assertTrue(hung_up.wait(1))

    def test_busy_final_response_reports_failure(self) -> None:
        failures: list[str] = []
        call = self._make_call(on_failed=failures.append)
        thread = threading.Thread(target=call.dial, args=("105",))
        thread.start()

        _, headers, source = self.uas.recv_invite()
        self.uas.send("486 Busy Here", headers, source)

        thread.join(timeout=3)
        self.assertFalse(thread.is_alive())
        self.assertEqual(failures, ["sip 486"])

    def test_timeout_without_any_response(self) -> None:
        failures: list[str] = []
        call = self._make_call(on_failed=failures.append)
        thread = threading.Thread(target=call.dial, args=("105", 0.5))
        thread.start()
        self.uas.recv_invite()
        thread.join(timeout=3)
        self.assertFalse(thread.is_alive())
        self.assertEqual(failures, ["no response"])            # not even a 100 Trying: there is no line

    def test_ringing_but_unanswered_is_a_timeout(self) -> None:
        failures: list[str] = []
        call = self._make_call(on_failed=failures.append)
        thread = threading.Thread(target=call.dial, args=("105", 1.0))
        thread.start()
        _, headers, source = self.uas.recv_invite()
        self.uas.send("180 Ringing", headers, source)
        thread.join(timeout=4)
        self.assertFalse(thread.is_alive())
        self.assertEqual(failures, ["timeout"])                # it rang and nobody answered: NO ANSWER

    def test_local_hangup_before_answer_sends_cancel(self) -> None:
        call = self._make_call()
        thread = threading.Thread(target=call.dial, args=("105", 5))
        thread.start()
        _, headers, source = self.uas.recv_invite()
        self.uas.send("180 Ringing", headers, source)

        call.hangup()
        cancel_start, cancel_headers, _ = self.uas.recv_any()
        self.assertTrue(cancel_start.startswith("CANCEL"), cancel_start)
        self.assertEqual(cancel_headers.get("call-id"), headers.get("call-id"))

        self.uas.send("487 Request Terminated", headers, source)
        thread.join(timeout=3)
        self.assertFalse(thread.is_alive())

    def test_local_hangup_after_answer_sends_bye(self) -> None:
        call = self._make_call()
        thread = threading.Thread(target=call.dial, args=("105", 5))
        thread.start()
        _, headers, source = self.uas.recv_invite()
        sdp = (
            "v=0\r\no=uas 1 1 IN IP4 127.0.0.1\r\ns=uas\r\n"
            "c=IN IP4 127.0.0.1\r\nt=0 0\r\n"
            "m=audio 17778 RTP/AVP 0\r\na=rtpmap:0 PCMU/8000\r\n"
        )
        self.uas.send("200 OK", headers, source, body=sdp)
        ack_start, _, _ = self.uas.recv_any()
        self.assertTrue(ack_start.startswith("ACK"))

        call.hangup()
        bye_start, bye_headers, _ = self.uas.recv_any()
        self.assertTrue(bye_start.startswith("BYE"), bye_start)
        self.assertEqual(bye_headers.get("call-id"), headers.get("call-id"))
        self.uas.send("200 OK", bye_headers, source)
        thread.join(timeout=3)
        self.assertFalse(thread.is_alive())


class WindowsUdpFailureTests(unittest.TestCase):
    def test_udp_reset_reports_no_line_once(self):
        endpoint=Mock();endpoint.getsockname.return_value=('127.0.0.1',15000)
        endpoint.recvfrom.side_effect=ConnectionResetError('ICMP unreachable')
        outcomes=[]
        call=SipOriginateCall('127.0.0.1',0,'127.0.0.1',5060,'127.0.0.1',16000,
            on_connected=lambda address:outcomes.append('connected'),on_failed=outcomes.append,on_hangup=lambda:None)
        with patch('sip_uac.socket.socket',return_value=endpoint):call.dial('test',2)
        self.assertEqual(outcomes,['no line'])
        endpoint.close.assert_called()

if __name__ == "__main__":
    unittest.main()
