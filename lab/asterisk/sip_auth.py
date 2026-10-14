"""SIP digest authentication (RFC 2617 / RFC 8760-style MD5 only) and a REGISTER client -- our own code.

Lets the built-in SIP user agent talk to an ordinary SIP trunk provider directly, with no PBX in between.
Credentials are passed in by the caller; this module never logs or stores them.
"""
import hashlib
import queue
import re
import socket
import threading
import time
import uuid

from sip_uas import parse_message


def parse_challenge(header: str) -> dict[str, str]:
    """``Digest realm="x", nonce="y", qop="auth"`` -> {'realm': 'x', 'nonce': 'y', 'qop': 'auth'}."""
    body = header.split(None, 1)[1] if " " in header else ""
    return {k.lower(): (q if q else v) for k, q, v in re.findall(r'(\w+)=(?:"([^"]*)"|([^,\s]+))', body)}


def digest_response(user: str, password: str, method: str, uri: str, ch: dict[str, str],
                    nc: int = 1, cnonce: str | None = None) -> str:
    """The value of an Authorization/Proxy-Authorization header answering challenge ``ch``."""
    md5 = lambda s: hashlib.md5(s.encode()).hexdigest()          # noqa: E731 -- MD5 is what SIP digest specifies
    realm, nonce = ch.get("realm", ""), ch.get("nonce", "")
    ha1 = md5(f"{user}:{realm}:{password}")
    ha2 = md5(f"{method}:{uri}")
    qop = "auth" if "auth" in [q.strip() for q in ch.get("qop", "").split(",")] else ""
    if qop:
        cnonce = cnonce or uuid.uuid4().hex[:16]
        response = md5(f"{ha1}:{nonce}:{nc:08x}:{cnonce}:{qop}:{ha2}")
    else:
        response = md5(f"{ha1}:{nonce}:{ha2}")
    out = f'Digest username="{user}", realm="{realm}", nonce="{nonce}", uri="{uri}", response="{response}", algorithm=MD5'
    if qop:
        out += f', qop={qop}, nc={nc:08x}, cnonce="{cnonce}"'
    if "opaque" in ch:
        out += f', opaque="{ch["opaque"]}"'
    return out


class SipRegistration:
    """Keeps one REGISTER binding alive with a SIP registrar (digest-authenticated), from a background thread."""

    def __init__(self, registrar_host: str, registrar_port: int, user: str, password: str, local_host: str,
                 local_port: int, expires: int = 300, domain: str | None = None):
        self.registrar = (registrar_host, registrar_port)
        self.user, self._password = user, password
        self.local = (local_host, local_port)
        self.domain = domain or registrar_host
        self.expires = expires
        self.registered = False
        self.last_status = ""
        self._stop = threading.Event()
        self._call_id = f"{uuid.uuid4()}@el-modem"
        self._cseq = 0
        self._thread: threading.Thread | None = None
        self._inbox: queue.Queue = queue.Queue()

    def _request(self, sock: socket.socket, auth: str | None, expires: int) -> tuple[str, dict[str, str]]:
        self._cseq += 1
        branch = "z9hG4bK" + uuid.uuid4().hex
        uri = f"sip:{self.domain}"
        lines = [
            f"REGISTER {uri} SIP/2.0",
            f"Via: SIP/2.0/UDP {self.local[0]}:{self.local[1]};branch={branch};rport",
            f"From: <sip:{self.user}@{self.domain}>;tag={self._call_id[:8]}",
            f"To: <sip:{self.user}@{self.domain}>",
            f"Call-ID: {self._call_id}",
            f"CSeq: {self._cseq} REGISTER",
            f"Contact: <sip:{self.user}@{self.local[0]}:{self.local[1]}>;expires={expires}",
            "Max-Forwards: 70", f"Expires: {expires}", "Content-Length: 0",
        ]
        if auth:
            lines.insert(-1, f"Authorization: {auth}")
        sock.sendto(("\r\n".join(lines) + "\r\n\r\n").encode(), self.registrar)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            try:
                data, _ = sock.recvfrom(65535)
            except (socket.timeout, queue.Empty):
                continue
            start, headers, _ = parse_message(data)
            if start.startswith("SIP/2.0") and headers.get("call-id") == self._call_id:
                code = start.split(" ", 2)[1]
                if code.startswith("1"):
                    continue
                return code, headers
        return "timeout", {}

    def register_once(self, sock: socket.socket, expires: int | None = None) -> bool:
        expires = self.expires if expires is None else expires
        code, headers = self._request(sock, None, expires)
        if code in ("401", "407"):
            ch = parse_challenge(headers.get("www-authenticate") or headers.get("proxy-authenticate", ""))
            code, headers = self._request(sock, digest_response(self.user, self._password, "REGISTER",
                                                                f"sip:{self.domain}", ch), expires)
        self.last_status = code
        self.registered = code == "200" and expires > 0
        return code == "200"

    def start(self, shared_with=None) -> None:
        """Run the registration loop. With ``shared_with`` (a SipIncomingCallServer) the REGISTER requests go out of that
        server's socket -- so the provider's INVITEs, sent to the registered contact, arrive where the server listens --
        and the server hands the responses back through ``deliver``."""
        def run() -> None:
            if shared_with is not None:
                sock = _SharedSocket(shared_with, self)
            else:
                sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                sock.settimeout(0.5)
            try:
                if shared_with is None:
                    sock.bind(self.local)
                while not self._stop.is_set():
                    self.register_once(sock)
                    self._stop.wait(max(5, self.expires * 0.8) if self.registered else 15)
                self.register_once(sock, 0)                 # unregister politely
            finally:
                sock.close()

        self._thread = threading.Thread(target=run, name="sip-register", daemon=True)
        self._thread.start()

    def deliver(self, data: bytes) -> None:
        self._inbox.put((data, None))

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(3)


class _SharedSocket:
    """Just enough socket for SipRegistration: sends through the UAS socket, receives what the UAS routes to us."""

    def __init__(self, uas, reg: SipRegistration):
        self._uas, self._reg = uas, reg
        uas.registration = reg

    def sendto(self, data: bytes, addr) -> None:
        sock = self._uas._socket
        if sock is not None:
            try:
                sock.sendto(data, addr)
            except OSError:
                pass                                    # server already closed

    def recvfrom(self, _n: int):
        return self._reg._inbox.get(timeout=0.5)

    def close(self) -> None:
        pass
