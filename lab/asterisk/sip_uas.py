#!/usr/bin/env python3
"""Minimal inbound SIP user agent for the isolated modem lab."""

import logging
import os
import re
import socket
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable

logger = logging.getLogger(__name__)


def _sip_trace(event: str, **fields: object) -> None:
    """Emit opt-in SIP setup metadata without headers, SDP, or credentials."""
    if os.environ.get("EL_MODEM_SIP_TRACE") != "1":
        return
    details = " ".join(f"{key}={value}" for key, value in fields.items())
    print(f"SIPTRACE side=answerer event={event} {details}".rstrip(), file=sys.stderr, flush=True)


def parse_caller(from_header: str, *, privacy: str = "") -> dict:
    """Caller ID from a SIP From header: ``"Name" <sip:number@host>;tag=x`` -> {'name': 'Name', 'number': 'number'}."""
    import re  # noqa: PLC0415

    m = re.search(r'sips?:([^@>;\s]+)', from_header)
    name = re.match(r'\s*"([^"]*)"', from_header)
    if not name:
        name = re.match(r'\s*([^<"]+?)\s*<', from_header)
    number = m.group(1) if m else ""
    display = name.group(1).strip() if name else ""
    restricted = "id" in {token.strip().lower() for token in re.split(r"[;,]", privacy)}
    restricted = restricted or number.lower() in ("anonymous", "restricted", "unavailable")
    if restricted:
        return {"number": "", "name": "", "presentation": "restricted", "technology": "SIP"}
    return {"number": number, "name": display, "presentation": "allowed", "technology": "SIP"}


@dataclass
class SipInvite:
    source: tuple[str, int]
    headers: dict[str, str]
    request_uri: str
    body: str
    to_tag: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    acked: threading.Event = field(default_factory=threading.Event)
    answered: bool = False
    cancelled: bool = False
    ring_deadline: float = 0.0


def parse_message(data: bytes) -> tuple[str, dict[str, str], str]:
    text = data.decode("utf-8", "replace")
    head, _, body = text.partition("\r\n\r\n")
    lines = head.split("\r\n")
    headers: dict[str, str] = {}
    for line in lines[1:]:
        key, separator, value = line.partition(":")
        if separator:
            normalized = key.strip().lower()
            if normalized in headers:
                headers[normalized] += "," + value.strip()
            else:
                headers[normalized] = value.strip()
    return lines[0] if lines else "", headers, body


def parse_remote_rtp(body: str) -> tuple[str, int] | None:
    remote_ip = None
    remote_port = None
    for line in body.splitlines():
        line = line.strip()
        if line.startswith("c=IN IP4 "):
            remote_ip = line.rsplit(" ", 1)[-1]
        elif line.startswith("m=audio "):
            parts = line.split()
            if len(parts) >= 2 and parts[1].isdigit():
                remote_port = int(parts[1])
    if remote_ip and remote_port:
        return remote_ip, remote_port
    return None


class SipIncomingCallServer:
    def __init__(
        self,
        host: str,
        port: int,
        media_host: str,
        media_port: int,
        on_incoming: Callable[[], None],
        on_ring: Callable[[], None],
        on_hangup: Callable[[], None],
        ring_interval: float = 6.0,
        packetization_ms: int = 20,
    ):
        if packetization_ms not in (10, 20):
            raise ValueError("SIP audio packetization must be 10 or 20 ms")
        self.host = host
        self.port = port
        self.media_host = media_host
        self.media_port = media_port
        self.packetization_ms = packetization_ms
        self.on_incoming = on_incoming
        self.on_ring = on_ring
        self.on_hangup = on_hangup
        self.ring_interval = ring_interval
        self._socket: socket.socket | None = None
        self.last_caller: dict = {}              # {'number','name'} of the most recent INVITE's From header
        self.registration = None                # a SipRegistration sharing this socket (see sip_auth.py)
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.RLock()
        self._invite: SipInvite | None = None

    def start(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((self.host, self.port))
        sock.settimeout(0.2)
        self._socket = sock
        _sip_trace("bound", local=f"{sock.getsockname()[0]}:{sock.getsockname()[1]}")
        self._thread = threading.Thread(target=self._serve, name="sip-uas", daemon=True)
        self._thread.start()

    def _send_response(
        self,
        invite: SipInvite,
        status: str,
        body: str = "",
        extra_headers: tuple[tuple[str, str], ...] = (),
    ) -> None:
        if self._socket is None:
            return
        lines = [
            f"SIP/2.0 {status}",
            f"Via: {invite.headers.get('via', '')}",
            f"From: {invite.headers.get('from', '')}",
            f"To: {self._tagged_to(invite)}",
            f"Call-ID: {invite.headers.get('call-id', '')}",
            f"CSeq: {invite.headers.get('cseq', '')}",
        ]
        lines.extend(f"{key}: {value}" for key, value in extra_headers)
        if body:
            lines.extend(
                (
                    "Contact: <sip:modem@{}:{}>".format(self.media_host, self.port),
                    "Content-Type: application/sdp",
                )
            )
        lines.extend((f"Content-Length: {len(body.encode())}", "", body))
        self._socket.sendto("\r\n".join(lines).encode(), invite.source)
        _sip_trace("response_sent", status=status.split(" ", 1)[0], target=f"{invite.source[0]}:{invite.source[1]}")

    @staticmethod
    def _tagged_to(invite: SipInvite) -> str:
        to_value = invite.headers.get("to", "")
        if "tag=" in to_value:
            return to_value
        return f"{to_value};tag={invite.to_tag}"

    def _answer_sdp(self) -> str:
        return (
            "v=0\r\n"
            f"o=modem-lab 1 1 IN IP4 {self.media_host}\r\n"
            "s=Bell103Modem\r\n"
            f"c=IN IP4 {self.media_host}\r\n"
            "t=0 0\r\n"
            f"m=audio {self.media_port} RTP/AVP 0\r\n"
            "a=rtpmap:0 PCMU/8000\r\n"
            f"a=ptime:{self.packetization_ms}\r\n"
            "a=sendrecv\r\n"
        )

    def answer(self, timeout: float = 8.0) -> bool:
        with self._lock:
            invite = self._invite
            if invite is None or invite.cancelled or invite.answered:
                return False
            invite.answered = True
            self._send_response(invite, "200 OK", self._answer_sdp())
        if not invite.acked.wait(timeout):
            _sip_trace("ack_timeout", source=f"{invite.source[0]}:{invite.source[1]}")
            logger.warning("SIP ACK timed out for incoming modem call")
            with self._lock:
                invite.answered = False
            return False
        return True

    def remote_rtp(self) -> tuple[str, int] | None:
        """RTP address the current/last answered caller advertised in its
        INVITE SDP, i.e. where this side should send outgoing media."""
        with self._lock:
            invite = self._invite
            if invite is None:
                return None
            return parse_remote_rtp(invite.body)

    def hangup(self) -> None:
        with self._lock:
            invite = self._invite
            if invite is None:
                return
            if invite.answered:
                self._send_in_dialog(invite, "BYE", 2)
            else:
                self._send_response(invite, "487 Request Terminated")
                invite.cancelled = True
            self._invite = None
        self.on_hangup()

    def _send_in_dialog(self, invite: SipInvite, method: str, cseq: int) -> None:
        if self._socket is None:
            return
        branch = "z9hG4bK" + uuid.uuid4().hex
        request = (
            f"{method} {invite.request_uri} SIP/2.0\r\n"
            f"Via: SIP/2.0/UDP {self.host}:{self.port};branch={branch};rport\r\n"
            f"From: {invite.headers.get('to', '')};tag={invite.to_tag}\r\n"
            f"To: {invite.headers.get('from', '')}\r\n"
            f"Call-ID: {invite.headers.get('call-id', '')}\r\n"
            f"CSeq: {cseq} {method}\r\n"
            "Max-Forwards: 70\r\n"
            "Content-Length: 0\r\n\r\n"
        )
        self._socket.sendto(request.encode(), invite.source)

    def _serve(self) -> None:
        assert self._socket is not None
        while not self._stop.is_set():
            try:
                data, source = self._socket.recvfrom(65535)
            except socket.timeout:
                self._emit_periodic_ring()
                continue
            except ConnectionResetError:
                # Winsock surfaces a prior UDP send's ICMP error here. Keep
                # the listener available to new callers rather than killing it.
                continue
            except OSError:
                if not self._stop.is_set():
                    logger.exception("SIP modem socket failed")
                return
            request_line, headers, body = parse_message(data)
            method = request_line.split(" ", 1)[0].upper()
            if method in {"INVITE", "ACK", "CANCEL", "BYE"}:
                _sip_trace("request_received", method=method, source=f"{source[0]}:{source[1]}")
            if request_line.startswith("SIP/2.0"):
                if self.registration is not None:
                    self.registration.deliver(data)         # responses to our REGISTERs
                continue
            if method == "INVITE":
                self._handle_invite(request_line, headers, body, source)
            elif method == "ACK":
                self._handle_ack(headers)
            elif method == "CANCEL":
                self._handle_cancel(headers, source)
            elif method == "BYE":
                self._handle_bye(headers, source)
            else:
                self._send_stateless(source, request_line, headers, "405 Method Not Allowed")
            self._emit_periodic_ring()

    def _emit_periodic_ring(self) -> None:
        now = time.monotonic()
        with self._lock:
            invite = self._invite
            if (
                invite is None
                or invite.answered
                or invite.cancelled
                or now < invite.ring_deadline
            ):
                return
            invite.ring_deadline = now + self.ring_interval
        self.on_ring()
        with self._lock:
            if self._invite is invite and not invite.answered and not invite.cancelled:
                self._send_response(invite, "180 Ringing")

    def _handle_invite(
        self,
        request_line: str,
        headers: dict[str, str],
        body: str,
        source: tuple[str, int],
    ) -> None:
        request_uri = request_line.split(" ", 2)[1]
        with self._lock:
            current = self._invite
            if current is not None and headers.get("call-id") == current.headers.get("call-id"):
                self._send_response(
                    current,
                    "200 OK" if current.answered else "180 Ringing",
                    self._answer_sdp() if current.answered else "",
                )
                return
            if current is not None:
                temporary = SipInvite(source, headers, request_uri, body)
                self._send_response(temporary, "486 Busy Here")
                return
            invite = SipInvite(
                source=source,
                headers=headers,
                request_uri=request_uri,
                body=body,
                ring_deadline=time.monotonic() + self.ring_interval,
            )
            self._invite = invite
        self.last_caller = parse_caller(headers.get("from", ""), privacy=headers.get("privacy", ""))
        self.on_incoming()
        self.on_ring()
        self._send_response(invite, "100 Trying")
        self._send_response(invite, "180 Ringing")

    def _find_invite(self, headers: dict[str, str]) -> SipInvite | None:
        with self._lock:
            invite = self._invite
            if invite is None:
                return None
            if headers.get("call-id") != invite.headers.get("call-id"):
                return None
            return invite

    def _handle_ack(self, headers: dict[str, str]) -> None:
        invite = self._find_invite(headers)
        if invite is not None:
            invite.acked.set()
            _sip_trace("ack_matched")

    def _handle_cancel(self, headers: dict[str, str], source: tuple[str, int]) -> None:
        invite = self._find_invite(headers)
        if invite is None:
            self._send_stateless(source, "CANCEL sip:modem SIP/2.0", headers, "481 Call/Transaction Does Not Exist")
            return
        self._send_stateless(source, "CANCEL sip:modem SIP/2.0", headers, "200 OK")
        if not invite.answered:
            self._send_response(invite, "487 Request Terminated")
            invite.cancelled = True
            with self._lock:
                if self._invite is invite:
                    self._invite = None
            self.on_hangup()

    def _handle_bye(self, headers: dict[str, str], source: tuple[str, int]) -> None:
        invite = self._find_invite(headers)
        self._send_stateless(source, "BYE sip:modem SIP/2.0", headers, "200 OK")
        if invite is not None:
            with self._lock:
                if self._invite is invite:
                    self._invite = None
            self.on_hangup()

    def _send_stateless(
        self,
        source: tuple[str, int],
        request_line: str,
        headers: dict[str, str],
        status: str,
    ) -> None:
        if self._socket is None:
            return
        to_value = headers.get("to", "")
        if request_line.startswith("CANCEL") or request_line.startswith("BYE"):
            current = self._find_invite(headers)
            if current is not None:
                to_value = self._tagged_to(current)
        lines = [
            f"SIP/2.0 {status}",
            f"Via: {headers.get('via', '')}",
            f"From: {headers.get('from', '')}",
            f"To: {to_value}",
            f"Call-ID: {headers.get('call-id', '')}",
            f"CSeq: {headers.get('cseq', '')}",
            "Content-Length: 0",
            "",
            "",
        ]
        self._socket.sendto("\r\n".join(lines).encode(), source)

    def close(self) -> None:
        self._stop.set()
        if self._socket is not None:
            self._socket.close()
        if self._thread is not None:
            self._thread.join(timeout=2)
