"""Known-payload load test over a telnet gateway: the shared protocol, the server and the DTE client.

A modem call reaches a telnet gateway (e.g. a Patton RAS pool's host prompt), which opens TCP to
``telnet_load_server``. Both directions then carry self-checking ASCII records, so the data segment
can be driven as hard as the link allows and every loss or corruption is located by sequence number.

Records are 7-bit text so they survive a raw relay, a real telnet client (NVT CR/LF rewriting, IAC
negotiation) and any 8-bit-stripping hop alike::

    D00000042 1a2b3c4d <60 base32 chars>\\n      data record (direction D=down, U=up)
    A00000042\\n                                  server -> client: highest in-order up record seen
    K00000042\\n                                  client -> server: next down record expected (window)
    E<client text>\\n                             echo request / reply (mode ``echo``)

Session: server sends ``EL-LOAD 1 token?``; client sends the token line, then ``RUN <mode> <seconds>
<seed> [W<bytes>]``; mode is ``down``, ``up``, ``both`` or ``echo``. The server answers ``GO`` and, after
the run, ``RESULT <json>`` and closes.

Flow control. A dial-in gateway can buffer far more than its modem line carries (a Patton dial-in pool
took 262 KB in 14 s at 2400 bit/s), so TCP backpressure alone lets minutes of data queue in front of
the acknowledgements and the RESULT. Both directions are therefore windowed end to end, as modem file
transfers (ZMODEM, Kermit) are: up by the server's ``A`` lines, down (with ``W<bytes>``) by the
client's ``K`` lines. The client also honours XON/XOFF from its DCE (``AT&K4``); the records are 7-bit
text, so DC1/DC3 never occur in data.
"""
from __future__ import annotations

import base64
import hashlib
import json
import re
import select
import time
import zlib

BANNER = b"EL-LOAD 1 token?\r\n"
RECORD_DATA_CHARS = 60
RECORD_BYTES = 1 + 8 + 1 + 8 + 1 + RECORD_DATA_CHARS + 1
MODES = ("down", "up", "both", "echo")
IAC, DONT, DO, WONT, WILL, SB, SE = 255, 254, 253, 252, 251, 250, 240
_RECORD = re.compile(rb"([DU])(\d{8}) ([0-9a-f]{8}) ([A-Z2-7]{%d})" % RECORD_DATA_CHARS)
_ACK = re.compile(rb"A(\d{8})")
_DOWN_ACK = re.compile(rb"K(\d{8})")
XON, XOFF = 0x11, 0x13
DOWN_ACK_EVERY = 4                    # records between K lines (each is 11 bytes upstream)


def record_data(seed: int, direction: str, seq: int) -> bytes:
    """Deterministic 60-character body, so the receiver regenerates what it should have got."""
    digest = hashlib.sha256(f"{seed}:{direction}:{seq}".encode()).digest()
    digest += hashlib.sha256(digest).digest()
    return base64.b32encode(digest)[:RECORD_DATA_CHARS]


def record(seed: int, direction: str, seq: int) -> bytes:
    head = f"{direction}{seq:08d}".encode()
    data = record_data(seed, direction, seq)
    return head + b" %08x " % zlib.crc32(head + data) + data + b"\n"


class TelnetFilter:
    """Strip telnet commands from a byte stream and queue refusals for any option request.

    ASCII records never contain 0xFF, so any IAC is a command whether the far end is a raw relay or a
    telnet client. CR and NUL are dropped: NVT may turn LF into CR LF or CR NUL."""

    def __init__(self):
        self._state = 0
        self._verb = 0
        self.replies = bytearray()
        self.commands = 0

    def feed(self, data: bytes) -> bytes:
        out = bytearray()
        for byte in data:
            if self._state == 0:
                if byte == IAC:
                    self._state = 1
                elif byte not in (0, 13):
                    out.append(byte)
            elif self._state == 1:
                self.commands += 1
                if byte in (DO, DONT, WILL, WONT):
                    self._verb, self._state = byte, 2
                elif byte == SB:
                    self._state = 3
                elif byte == IAC:              # escaped 0xFF: never valid in a record
                    out.append(byte); self._state = 0
                else:
                    self._state = 0
            elif self._state == 2:
                if self._verb == DO:
                    self.replies += bytes((IAC, WONT, byte))
                elif self._verb == WILL:
                    self.replies += bytes((IAC, DONT, byte))
                self._state = 0
            elif self._state == 3:
                if byte == IAC:
                    self._state = 4
            elif self._state == 4:
                self._state = 0 if byte == SE else 3
        return bytes(out)


class RecordChecker:
    """Verify one direction's records: in order, CRC-valid and matching the regenerated body."""

    def __init__(self, seed: int, direction: str):
        self.seed, self.direction = seed, direction
        self.expected = 0
        self.good = self.bad_crc = self.bad_body = self.missing = self.duplicates = 0
        self.first_problems: list[dict] = []
        self.good_bytes = 0

    def _problem(self, kind, seq):
        if len(self.first_problems) < 20:
            self.first_problems.append(dict(kind=kind, seq=seq, expected=self.expected))

    def line(self, line: bytes) -> bool:
        match = _RECORD.fullmatch(line)
        if match is None or match[1].decode() != self.direction:
            return False
        seq = int(match[2])
        head = match[1] + match[2]
        if int(match[3], 16) != zlib.crc32(head + match[4]):
            self.bad_crc += 1; self._problem("crc", seq); return True
        if match[4] != record_data(self.seed, self.direction, seq):
            self.bad_body += 1; self._problem("body", seq); return True
        if seq < self.expected:
            self.duplicates += 1; self._problem("duplicate", seq); return True
        if seq > self.expected:
            self.missing += seq - self.expected; self._problem("gap", seq)
        self.expected = seq + 1
        self.good += 1
        self.good_bytes += RECORD_BYTES
        return True

    def report(self) -> dict:
        return dict(direction=self.direction, good=self.good, good_bytes=self.good_bytes,
                    bad_crc=self.bad_crc, bad_body=self.bad_body, missing=self.missing,
                    duplicates=self.duplicates, next_expected=self.expected,
                    first_problems=self.first_problems)


class Throughput:
    """Bytes per whole second since start, for a throughput-over-time trace."""

    def __init__(self, start: float):
        self.start = start
        self.buckets: dict[int, int] = {}

    def add(self, count: int, now: float):
        if count:
            second = int(now - self.start)
            self.buckets[second] = self.buckets.get(second, 0) + count

    def report(self) -> dict:
        if not self.buckets:
            return dict(bytes=0, per_second=[])
        last = max(self.buckets)
        series = [self.buckets.get(s, 0) for s in range(last + 1)]
        return dict(bytes=sum(series), per_second=series)


def _lines(buffer: bytearray):
    """Pop complete LF-terminated lines from ``buffer``."""
    while True:
        end = buffer.find(b"\n")
        if end < 0:
            return
        line = bytes(buffer[:end]).strip(b"\r\0 ")
        del buffer[:end + 1]
        yield line


def run_client(connection, token: str, mode: str, seconds: float, seed: int,
               banner_timeout: float = 60.0, window_bytes: int = 4096,
               drain_seconds: float = 20.0, initial: bytes = b"",
               down_window_bytes: int | None = None) -> dict:
    """Drive one load session from the modem side over ``connection`` (the DTE socket, online).

    Upstream is windowed by the server's ``A`` acknowledgements so at most ``window_bytes`` of
    unacknowledged records sit in DCE and gateway buffers; the measured rate is the link's.
    ``down_window_bytes`` asks the server to keep at most that much of the down stream unacknowledged
    (our ``K`` lines), so a gateway's deep buffer never fills. ``initial`` is data already read from the
    connection (e.g. the banner seen by a prompt loop)."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    result = dict(mode=mode, seconds=seconds, seed=seed, banner_seen=False, go=False)
    telnet = TelnetFilter()
    flow = dict(paused=False, xoff=0)

    def strip_flow(data: bytes) -> bytes:
        if XON not in data and XOFF not in data:
            return data
        for value in data:
            if value == XOFF:
                flow["paused"] = True; flow["xoff"] += 1
            elif value == XON:
                flow["paused"] = False
        return data.replace(bytes((XON,)), b"").replace(bytes((XOFF,)), b"")

    buffer = bytearray(telnet.feed(strip_flow(initial)))
    started = time.monotonic()

    def receive() -> bytes:
        if not select.select([connection], [], [], 0.05)[0]:
            return b""
        data = connection.recv(65536)
        if not data:
            raise ConnectionError("DTE socket closed")
        clean = strip_flow(telnet.feed(data))
        if telnet.replies:
            connection.sendall(bytes(telnet.replies)); telnet.replies.clear()
        return clean

    deadline = started + banner_timeout
    while BANNER.strip() not in bytes(buffer).replace(b"\r", b""):
        if time.monotonic() > deadline or b"NO CARRIER" in buffer:
            result["stop_reason"] = "no_load_banner"
            result["received_before_banner"] = bytes(buffer[-400:]).decode("latin1")
            return result
        buffer += receive()
    result["banner_seen"] = True
    result["banner_seconds"] = round(time.monotonic() - started, 2)
    buffer.clear()
    connection.sendall(token.encode() + b"\r\n")
    window = f" W{int(down_window_bytes)}" if down_window_bytes else ""
    connection.sendall(f"RUN {mode} {seconds:g} {seed}{window}\r\n".encode())
    result["down_window_bytes"] = down_window_bytes
    deadline = time.monotonic() + 30
    go = False
    while not go:
        # One line at a time: records that arrive in the same read as GO stay in the buffer.
        refused = []
        for line in _lines(buffer):
            if line == b"GO":
                go = True
                break
            if line.startswith((b"DENIED", b"BUSY", b"ERROR")):
                refused.append(line)
        if go:
            break
        if refused or time.monotonic() > deadline:
            result["stop_reason"] = "refused" if refused else "no_go"
            result["server_said"] = [line.decode("latin1") for line in refused]
            return result
        buffer += receive()
    result["go"] = True
    run_start = time.monotonic()
    down = RecordChecker(seed, "D")
    down_rate = Throughput(run_start)
    up_rate = Throughput(run_start)
    up_seq = acked = 0
    down_acked = 0
    unrecognised = 0
    rtts = []
    echo_sent: dict[int, float] = {}
    server_result = None
    send_up = mode in ("up", "both")
    end_at = run_start + seconds
    final_deadline = end_at + drain_seconds
    ended = False
    while time.monotonic() < final_deadline:
        now = time.monotonic()
        if not ended and now >= end_at and not flow["paused"]:
            connection.sendall(f"END {up_seq}\r\n".encode()); ended = True
        if down_window_bytes and down.expected - down_acked >= DOWN_ACK_EVERY and not flow["paused"]:
            down_acked = down.expected
            connection.sendall(b"K%08d\r\n" % down_acked)
        if not ended and send_up and not flow["paused"]:
            while (up_seq - acked) * RECORD_BYTES < window_bytes:
                connection.sendall(record(seed, "U", up_seq)); up_seq += 1
                up_rate.add(RECORD_BYTES, now)
        if not ended and mode == "echo" and not echo_sent:
            echo_sent[len(rtts)] = now
            connection.sendall(b"E%08d\r\n" % len(rtts))
        data = receive()
        now = time.monotonic()
        buffer += data
        for line in _lines(buffer):
            if not line:
                continue
            if down.line(line):
                down_rate.add(RECORD_BYTES, now); continue
            ack = _ACK.fullmatch(line)
            if ack:
                acked = max(acked, int(ack[1])); continue
            if line.startswith(b"E") and line[1:].isdigit():
                sent = echo_sent.pop(int(line[1:]), None)
                if sent is not None:
                    rtts.append(round(now - sent, 4))
                continue
            if line.startswith(b"RESULT "):
                try:
                    server_result = json.loads(line[7:])
                except ValueError:
                    server_result = dict(unparsed=line[7:200].decode("latin1"))
                continue
            if b"NO CARRIER" in line:
                result["stop_reason"] = "no_carrier"
                break
            unrecognised += 1
        if server_result is not None or result.get("stop_reason"):
            break
    result.update(
        stop_reason=result.get("stop_reason") or ("server_result" if server_result else "drain_timeout"),
        run_seconds=round(time.monotonic() - run_start, 2),
        down=dict(down.report(), **down_rate.report()) if mode in ("down", "both") else None,
        up=dict(sent_records=up_seq, acked_records=acked, **up_rate.report()) if send_up else None,
        echo_rtt_seconds=rtts if mode == "echo" else None,
        unrecognised_lines=unrecognised, telnet_commands=telnet.commands, xoff_count=flow["xoff"],
        server=server_result)
    down_ok = mode not in ("down", "both") or (
        down.good > 0 and not (down.bad_crc or down.bad_body or down.missing or down.duplicates))
    up_ok = not send_up or bool(server_result and server_result.get("up", {}).get("clean"))
    result["payload_verified"] = bool(server_result) and down_ok and up_ok
    return result
