#!/usr/bin/env python3
"""Minimal outbound SIP user agent (originate) for the isolated modem lab.

This intentionally mirrors the style of ``sip_uas.py``: it is a lab-scale
implementation of just enough SIP to place one call through the Asterisk
test PBX and track its progress, not a production SIP stack. It has no
authentication, retransmission timers, or multi-dialog support, matching the
single-line modem this project is building.
"""

import logging
import socket
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Callable

from sip_uas import parse_message, parse_remote_rtp

logger = logging.getLogger(__name__)


@dataclass
class _Dialog:
    call_id: str
    from_tag: str
    invite_branch: str
    cseq: int = 1
    to_header: str = ""
    from_uri: str = ""


class SipOriginateCall:
    """Places one outbound call and reports ringing/answer/failure/hangup."""

    def __init__(
        self,
        host: str,
        port: int,
        target_host: str,
        target_port: int,
        media_host: str,
        media_port: int,
        on_connected: Callable[[tuple[str, int]], None],
        on_failed: Callable[[str], None],
        on_hangup: Callable[[], None],
        username: str | None = None,
        password: str | None = None,
        caller_number: str | None = None,
    ):
        # Optional digest credentials: with them a 401/407 challenge is answered (SIP trunk providers).
        self.username, self._password, self.caller_number = username, password, caller_number
        self.host = host
        self.port = port
        self.target_host = target_host
        self.target_port = target_port
        self.media_host = media_host
        self.media_port = media_port
        self.on_connected = on_connected
        self.on_failed = on_failed
        self.on_hangup = on_hangup
        self._socket: socket.socket | None = None
        self._lock = threading.RLock()
        self._dialog: _Dialog | None = None
        self._answered = False
        self._finished = False

    def _invite_sdp(self) -> str:
        return (
            "v=0\r\n"
            f"o=modem-lab 1 1 IN IP4 {self.media_host}\r\n"
            "s=Bell103Modem\r\n"
            f"c=IN IP4 {self.media_host}\r\n"
            "t=0 0\r\n"
            f"m=audio {self.media_port} RTP/AVP 0\r\n"
            "a=rtpmap:0 PCMU/8000\r\n"
            "a=ptime:20\r\n"
            "a=sendrecv\r\n"
        )

    def dial(self, number: str, timeout: float = 30.0) -> None:
        """Send the INVITE and run this call's signaling to completion.

        Intended to run in its own thread; reports outcomes only through the
        on_connected/on_failed/on_hangup callbacks.
        """
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((self.host, self.port))
        sock.settimeout(0.5)
        with self._lock:
            self._socket = sock
        call_id = f"{uuid.uuid4()}@modem-lab"
        from_tag = uuid.uuid4().hex[:12]
        branch = "z9hG4bK" + uuid.uuid4().hex
        dialog = _Dialog(call_id=call_id, from_tag=from_tag, invite_branch=branch)
        with self._lock:
            self._dialog = dialog
        local_port = sock.getsockname()[1]
        sdp = self._invite_sdp()
        from_user = self.caller_number or self.username or "modem"
        dialog.from_uri = f"sip:{from_user}@{self.target_host if self.username else self.host}"
        uri = f"sip:{number}@{self.target_host}"

        def build_invite(auth_header: str | None = None) -> str:
            lines = (
                f"INVITE {uri} SIP/2.0\r\n"
                f"Via: SIP/2.0/UDP {self.host}:{local_port};branch={dialog.invite_branch};rport\r\n"
                f"From: <{dialog.from_uri}>;tag={from_tag}\r\n"
                f"To: <sip:{number}@{self.target_host}>\r\n"
                f"Call-ID: {call_id}\r\n"
                f"CSeq: {dialog.cseq} INVITE\r\n"
                f"Contact: <sip:{from_user}@{self.media_host}:{local_port}>\r\n"
                "Max-Forwards: 70\r\n"
            )
            if auth_header:
                lines += auth_header + "\r\n"
            return lines + f"Content-Type: application/sdp\r\nContent-Length: {len(sdp.encode())}\r\n\r\n{sdp}"

        try:
            sock.sendto(build_invite().encode(), (self.target_host, self.target_port))
        except OSError:                                    # network unreachable, no route: there is no line
            self._fail_once("no line")
            sock.close()
            return
        challenged = False
        heard = False
        first_deadline = time.monotonic() + 8.0            # no reply at all to the INVITE in this time: no line

        deadline = time.monotonic() + timeout
        try:
            while time.monotonic() < deadline:
                with self._lock:
                    if self._finished:
                        return
                try:
                    data, _ = sock.recvfrom(65535)
                except socket.timeout:
                    if not heard and time.monotonic() > first_deadline:
                        self._fail_once("no response")
                        return
                    continue
                except (ConnectionRefusedError, ConnectionResetError): # Linux/Windows ICMP port unreachable
                    self._fail_once("no line")
                    return
                except OSError:
                    return
                heard = True
                start_line, headers, body = parse_message(data)
                if not start_line.startswith("SIP/2.0") or headers.get("call-id") != call_id:
                    continue
                fields = start_line.split(" ", 2)
                if len(fields) < 2:
                    continue
                code = fields[1]
                to_header = headers.get("to", "")
                if code in ("401", "407") and self.username and not challenged:
                    challenged = True
                    from sip_auth import digest_response, parse_challenge  # noqa: PLC0415

                    self._send_ack(sock, dialog, with_route=False)
                    chal = headers.get("www-authenticate") or headers.get("proxy-authenticate", "")
                    name = "Authorization" if code == "401" else "Proxy-Authorization"
                    dialog.cseq += 1
                    dialog.invite_branch = "z9hG4bK" + uuid.uuid4().hex
                    sock.sendto(build_invite(f"{name}: " + digest_response(
                        self.username, self._password or "", "INVITE", uri, parse_challenge(chal))).encode(),
                        (self.target_host, self.target_port))
                    continue
                if code.startswith("1"):
                    if "tag=" in to_header:
                        dialog.to_header = to_header
                    continue
                if code.startswith("2"):
                    dialog.to_header = to_header
                    self._send_ack(sock, dialog, with_route=False)
                    remote_addr = parse_remote_rtp(body)
                    with self._lock:
                        if self._finished:
                            return
                        self._answered = True
                    self.on_connected(remote_addr or (self.target_host, self.target_port))
                    self._wait_for_hangup(sock, dialog)
                    return
                # Any other final response (4xx/5xx/6xx) ends the attempt.
                if "tag=" in to_header:
                    dialog.to_header = to_header
                self._send_ack(sock, dialog, with_route=False)
                with self._lock:
                    if self._finished:
                        return
                    self._finished = True
                self.on_failed(f"sip {code}")
                return
            self._fail_once("timeout" if heard else "no response")
        finally:
            with self._lock:
                self._socket = None
            sock.close()

    def _fail_once(self, reason: str) -> None:
        with self._lock:
            if self._finished:
                return
            self._finished = True
        self.on_failed(reason)

    def _send_ack(self, sock: socket.socket, dialog: _Dialog, with_route: bool) -> None:
        local_port = sock.getsockname()[1]
        ack = (
            f"ACK sip:{self.target_host} SIP/2.0\r\n"
            f"Via: SIP/2.0/UDP {self.host}:{local_port};branch={dialog.invite_branch};rport\r\n"
            f"From: <{dialog.from_uri}>;tag={dialog.from_tag}\r\n"
            f"To: {dialog.to_header}\r\n"
            f"Call-ID: {dialog.call_id}\r\n"
            f"CSeq: {dialog.cseq} ACK\r\n"
            "Max-Forwards: 70\r\nContent-Length: 0\r\n\r\n"
        )
        sock.sendto(ack.encode(), (self.target_host, self.target_port))

    def _wait_for_hangup(self, sock: socket.socket, dialog: _Dialog) -> None:
        sock.settimeout(0.5)
        while True:
            with self._lock:
                if self._finished:
                    return
            try:
                data, source = sock.recvfrom(65535)
            except socket.timeout:
                continue
            except OSError:
                return
            request_line, headers, _ = parse_message(data)
            if headers.get("call-id") != dialog.call_id:
                continue
            method = request_line.split(" ", 1)[0].upper()
            if method == "BYE":
                self._send_stateless_ok(sock, source, headers)
                with self._lock:
                    if self._finished:
                        return
                    self._finished = True
                self.on_hangup()
                return

    def _send_stateless_ok(
        self,
        sock: socket.socket,
        source: tuple[str, int],
        headers: dict[str, str],
    ) -> None:
        lines = [
            "SIP/2.0 200 OK",
            f"Via: {headers.get('via', '')}",
            f"From: {headers.get('from', '')}",
            f"To: {headers.get('to', '')}",
            f"Call-ID: {headers.get('call-id', '')}",
            f"CSeq: {headers.get('cseq', '')}",
            "Content-Length: 0",
            "",
            "",
        ]
        sock.sendto("\r\n".join(lines).encode(), source)

    def hangup(self) -> None:
        """Locally end the call: CANCEL if still ringing, BYE if answered."""
        with self._lock:
            if self._finished:
                return
            answered = self._answered
            dialog = self._dialog
            sock = self._socket
            # Keep socket ownership until the terminating request is sent.
            # dial() observes _finished under this same lock, then closes
            # its socket in finally; publishing it earlier races CANCEL/BYE.
            if sock is None or dialog is None:
                self._finished = True
                return
            local_port = sock.getsockname()[1]
            if answered:
                dialog.cseq += 1
                branch = "z9hG4bK" + uuid.uuid4().hex
                request = (
                    f"BYE sip:{self.target_host} SIP/2.0\r\n"
                    f"Via: SIP/2.0/UDP {self.host}:{local_port};branch={branch};rport\r\n"
                    f"From: <{dialog.from_uri}>;tag={dialog.from_tag}\r\n"
                    f"To: {dialog.to_header}\r\n"
                    f"Call-ID: {dialog.call_id}\r\n"
                    f"CSeq: {dialog.cseq} BYE\r\n"
                    "Max-Forwards: 70\r\nContent-Length: 0\r\n\r\n"
                )
            else:
                request = (
                    f"CANCEL sip:{self.target_host} SIP/2.0\r\n"
                    f"Via: SIP/2.0/UDP {self.host}:{local_port};branch={dialog.invite_branch};rport\r\n"
                    f"From: <{dialog.from_uri}>;tag={dialog.from_tag}\r\n"
                    f"To: <sip:{self.target_host}>\r\n"
                    f"Call-ID: {dialog.call_id}\r\n"
                    f"CSeq: {dialog.cseq} CANCEL\r\n"
                    "Max-Forwards: 70\r\nContent-Length: 0\r\n\r\n"
                )
            try:
                sock.sendto(request.encode(), (self.target_host, self.target_port))
            except OSError:
                pass
            finally:
                self._finished = True
