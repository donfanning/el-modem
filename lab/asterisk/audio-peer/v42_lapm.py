"""ITU-T V.42 error control (LAPM) over a raw bit stream, written from the Recommendation (03/2002 text).

Covers the detection phase (ODP / ADP, §7.2.1), HDLC framing with zero insertion and the 16-bit FCS (§8.1), the
modulo-128 LAPM frames and procedures (§8.2-8.9: SABME/UA/DM/DISC, I, RR, RNR, REJ, XID), parameter negotiation by XID
(§8.10, §12.2, N401 and k), and V.42bis data compression negotiated in the XID private parameter group (V.42bis
Annex A) using our own v42bis.py. Not implemented: SREJ, TEST, break signalling (BRK), 32-bit FCS, timer T403.

``V42Link`` has the same surface as the SpanDSP-backed class it replaces (spandsp_v42.V42Link): feed line bits with
``rx_bit``, take line bits with ``tx_bit``, user data in with ``write`` and out with ``read``.

Credits: the frame formats and procedures are ITU-T's; the interoperability details that real peers need (address octets
for originator/answerer commands and responses, the XID parameter encoding, tolerance for peers that send the ADP
characters back to back) were checked against SpanDSP's v42.c (LGPL, Steve Underwood) and the Smart Link modem as
black-box peers; no code was copied from either.
"""

from __future__ import annotations

import os
import math

from collections import deque

from mnp_link import crc16_hdlc
from v42bis import Decoder, Encoder, V42bisError

FLAG = 0x7E
DEFAULT_N401, DEFAULT_K = 128, 15
N400 = 10
XID_ATTEMPTS = 2                  # unanswered XID commands before the originator skips negotiation and sends SABME (V.42 7.2.2)
# unnumbered frame control values (P/F bit 0x10)
SABME, UA, DM, DISC, FRMR, XID, UI, TEST = 0x6F, 0x63, 0x0F, 0x43, 0x87, 0xAF, 0x03, 0xE3
RR, RNR, REJ, SREJ = 0x01, 0x05, 0x09, 0x0D
ODP_BLOCK = [int(c) for c in "0100010001" + "1" * 8 + "0100010011" + "1" * 8]       # DC1 even parity, DC1 odd parity
ADP_BLOCK = [int(c) for c in "0101000101" + "1" * 8 + "0110000101" + "1" * 8]        # E, C: V.42 supported
FLAG_BITS = [0, 1, 1, 1, 1, 1, 1, 0]
CHAR_E, CHAR_C, CHAR_NULL, DC1_EVEN, DC1_ODD = 0x45, 0x43, 0x00, 0x11, 0x91


class _Hdlc:
    """Bit-level HDLC receiver: flag hunt, zero deletion, FCS-16. ``push`` returns completed (frame, fcs_ok) pairs."""

    def __init__(self) -> None:
        self.ones = 0
        self.bits: list[int] = []
        self.active = False
        self.flags = 0                         # cumulative flags seen since reset
        self.continuous_flags = 0
        self._flag_gap = 9

    def push(self, bit: int) -> list[tuple[bytes, bool]]:
        out: list[tuple[bytes, bool]] = []
        # Adjacent flags end seven (shared zero) or eight bits apart.
        if self._flag_gap < 9:
            self._flag_gap += 1
        if bit:
            self.ones += 1
            if self.active and self.ones <= 6:
                self.bits.append(1)
            if self.ones >= 7:                 # abort / idle marks
                self.active, self.bits = False, []
            return out
        ones, self.ones = self.ones, 0
        if ones == 5:
            return out                         # stuffed zero
        if ones == 6:                          # flag: 0 111111 0 (its leading zero and six ones are already stored)
            self.flags += 1
            self.continuous_flags = (self.continuous_flags + 1
                                     if self._flag_gap in (7, 8) else 1)
            self._flag_gap = 0
            if self.active and len(self.bits) >= 7 + 24:
                data = self.bits[:-7]
                if len(data) % 8 == 0:
                    raw = bytes(sum(data[i + j] << j for j in range(8)) for i in range(0, len(data), 8))
                    out.append((raw[:-2], crc16_hdlc(raw[:-2]) == int.from_bytes(raw[-2:], "little")))
                else:
                    out.append((b"", False))
            self.active, self.bits = True, []
            return out
        if self.active:
            self.bits.append(0)
        return out


def _stuffed(frame: bytes) -> list[int]:
    body = frame + crc16_hdlc(frame).to_bytes(2, "little")
    bits: list[int] = []
    ones = 0
    for byte in body:
        for i in range(8):
            b = (byte >> i) & 1
            bits.append(b)
            ones = ones + 1 if b else 0
            if ones == 5:
                bits.append(0)
                ones = 0
    return bits


def build_xid(n401: int, k: int, compression: tuple[int, int, int] | None) -> bytes:
    """XID information field (§12.2.2): general-purpose FI, parameter group, optional V.42bis private group."""
    params = bytes((3, 4, 0x8A, 0x89, 0, 0))                                        # HDLC optional functions
    params += bytes((5, 2)) + (n401 * 8).to_bytes(2, "big") + bytes((6, 2)) + (n401 * 8).to_bytes(2, "big")
    params += bytes((7, 1, k, 8, 1, k))
    out = bytes((0x82, 0x80)) + len(params).to_bytes(2, "big") + params
    # V.42bis Annex A: P0=0 explicitly requests neither direction.
    # Send the negative request too, rather than relying on omission being
    # interpreted as the default by legacy peers. This never enables BIS.
    p0, p1, p2 = compression if compression is not None else (0, 1024, 32)
    priv = bytes((0, 3)) + b"V42" + bytes((1, 1, p0, 2, 2)) + p1.to_bytes(2, "big") + bytes((3, 1, p2))
    out += bytes((0xF0,)) + len(priv).to_bytes(2, "big") + priv
    return out


def parse_xid(info: bytes) -> dict | None:
    """Parameters found in an XID information field (unknown groups and parameters are ignored)."""
    if len(info) < 3 or info[0] != 0x82:
        return None
    out: dict = {}
    i = 1
    while i + 3 <= len(info):
        gid, glen = info[i], int.from_bytes(info[i + 1:i + 3], "big")
        body = info[i + 3:i + 3 + glen]
        i += 3 + glen
        j = 0
        while j + 2 <= len(body):
            pid, plen = body[j], body[j + 1]
            val = int.from_bytes(body[j + 2:j + 2 + plen], "big")
            j += 2 + plen
            if gid == 0x80:
                if pid == 5:
                    out["tx_n401"] = val // 8
                elif pid == 6:
                    out["rx_n401"] = val // 8
                elif pid == 7:
                    out["tx_k"] = val
                elif pid == 8:
                    out["rx_k"] = val
            elif gid == 0xF0:
                if pid == 1:
                    out["p0"] = val
                elif pid == 2:
                    out["p1"] = val
                elif pid == 3:
                    out["p2"] = val
    return out


class V42Link:
    def __init__(self, calling_party: bool, detect: bool = True, bit_rate: int = 9600, frame_size: int = 0,
                 window: int = 0, compression: tuple[int, int, int] | None = None, t400: float = 0.75):
        self.calling = calling_party
        self.rate = max(bit_rate, 300)
        self.state = "detect" if detect else "idle"
        self.adp_seen = False
        self.events: list[str] = []
        self.compression: tuple[int, int, int] | None = None      # agreed (p0, p1, p2) once connected
        self.compressed_out = 0
        # Link statistics for the modem's diagnostic screens (ATI6/ATI11)
        self.stats = dict(xid_tx=0, i_tx=0, i_rx=0, i_resent=0, rej_tx=0, rej_rx=0, fcs_errors=0, t401_expiries=0,
                          bytes_in=0, bytes_out=0, wire_in=0, wire_out=0, wire_acked=0,
                          hdlc_good=0, hdlc_bad=0)
        self.bits_sent = 0
        self.cmd_addr = 0x03 if calling_party else 0x01
        self.rsp_addr = 0x01 if calling_party else 0x03
        # Opt-in diagnostic: V.42 7.2.2 permits establishment without
        # negotiation when the standard defaults suffice. This is distinct
        # from abandoning an attempted XID exchange after response loss.
        self._default_parameters = os.environ.get("V42_DEFAULT_PARAMETERS", "0") == "1"
        if self._default_parameters and compression is not None:
            raise ValueError("default-parameter LAPM requires compression disabled")
        self.n401 = DEFAULT_N401 if self._default_parameters else min(frame_size or DEFAULT_N401, DEFAULT_N401)
        self.k = DEFAULT_K if self._default_parameters else min(window or DEFAULT_K, DEFAULT_K)
        self.stats['default_establishment'] = 0
        self._offer = compression
        self._t400_bits = int(t400 * self.rate)
        self._t400_at = self._t400_bits
        self._answer_detect_at = self._t400_bits * 2
        self._t401_bits = int(max(1.5, 2.2 * (self.n401 + 8) * 8 / self.rate + 1.0) * self.rate)
        self._tx: deque[int] = deque()
        self._hdlc = _Hdlc()
        self._async: list[int] = []
        self._chars: list[int] = []                                # characters seen during detection
        self._dc1: list[int] = []
        self._in = bytearray()
        self._out = bytearray()
        self._txq = bytearray()                                    # user data not yet in an I frame
        self._adp_count = 0
        self._xid_fallback = False                                  # set when SABME was sent without an XID answer (kept across _reset_link)
        self._bis_assumed = False
        # LAPM variables
        self._reset_link()
        self._started = False
        self._establish_at = 0                                     # originator: bit count to begin (>= 16 flags)
        self._phase = "detect" if detect else "protocol"
        if not detect:
            self._begin_protocol()

    # -- public surface --------------------------------------------------------------------------------------
    def set_bit_rate(self, bit_rate: int) -> None:
        """Preserve the link and remaining timer seconds across physical retraining.

        The clock counts transmitted bits. Rescale deadlines relative to that
        clock, not relative to zero; already-expired timers stay expired.
        Queued frames and the minimum flag count before XID are unchanged.
        """
        rate = max(int(bit_rate), 300)
        if rate == self.rate:
            return
        ratio = rate / self.rate
        now = self.bits_sent
        for name in ('_t401_at', '_t400_at', '_answer_detect_at'):
            deadline = getattr(self, name)
            if deadline is not None:
                setattr(self, name, now + math.ceil(max(0, deadline-now) * ratio))
        self._t400_bits = math.ceil(self._t400_bits * ratio)
        self.rate = rate
        self._t401_bits = int(max(1.5, 2.2 * (self.n401 + 8) * 8 / rate + 1.0) * rate)

    def close(self) -> None:
        self._tx.clear()

    def write(self, data: bytes) -> None:
        self.stats["bytes_out"] += len(data)
        self._txq += data

    def read(self) -> bytes:
        out, self._out = bytes(self._out), bytearray()
        return out

    def pending(self) -> int:
        return len(self._txq) + sum(len(c) for c in self._cq) + sum(len(f[1]) for f in self._unacked) + len(self._retx)

    def tx_bit(self) -> int:
        self.bits_sent += 1
        if not self._tx:
            self._schedule()
        if self._tx:
            return self._tx.popleft()
        return 1

    def take_queued_tx(self, limit: int) -> list[int]:
        """Drain existing LAPM bits without moving a scheduling boundary.

        ``tx_bit`` schedules only when the queue is empty, after incrementing
        its bit clock. This operation never schedules: it advances that same
        clock for bits already generated. The owning V42Layer holds its lock.
        """
        count=min(limit,len(self._tx))
        if count<=0 or self.state!='lapm':return []
        if count==len(self._tx):
            bits=list(self._tx);self._tx.clear()
        else:
            bits=[self._tx.popleft() for _ in range(count)]
        self.bits_sent+=count
        return bits

    def rx_bit(self, bit: int) -> None:
        if self.state in ("down", "unsupported"):
            return
        if self._phase == "detect":
            self._detect_rx(bit)
            if self._phase == "detect":
                self._hdlc_feed_for_start(bit)
                return
        for frame, ok in self._hdlc.push(bit):
            self._received_frame(frame, ok)

    def rx_bits(self, bits) -> None:
        """Batch established LAPM bits without delaying frame side effects.

        Dispatch each completed frame before consuming its next bit: DISC,
        SABME, compression and window updates retain scalar ordering. Other
        states still use the detection/protocol path of ``rx_bit``.
        """
        push = self._hdlc.push
        received = self._received_frame
        single = self.rx_bit
        for bit in bits:
            if self.state != 'lapm':
                single(bit)
                push = self._hdlc.push
                continue
            for frame, ok in push(bit):
                received(frame, ok)

    def _received_frame(self, frame: bytes, ok: bool) -> None:
        # Count established opportunities including control frames, before a
        # frame's side effects can change link state.
        if self.state == 'lapm':
            self.stats['hdlc_good' if ok else 'hdlc_bad'] += 1
        if ok:
            self._on_frame(frame)
        elif self.state == "lapm":
            self.stats["fcs_errors"] += 1
            self._need_rej = True

    # -- link state -------------------------------------------------------------------------------------------
    def _reset_link(self) -> None:
        self.vs = self.va = self.vr = 0
        self._unacked: list[list] = []                              # [ns, payload]
        self._retx: deque = deque()
        self._cq: deque = deque()                                   # compressed octets waiting for a window slot
        self._peer_busy = False
        self._ack_pending = False
        self._rej_sent = False
        self._need_rej = False
        self._t401_at: int | None = None
        self._tries = 0
        self._poll_out = False
        self._xid_done = False
        self._xid_sent = 0
        self._sabme_sent = False
        self._enc: Encoder | None = None
        self._dec: Decoder | None = None

    def _event(self, text: str) -> None:
        self.events.append(text)

    # -- detection phase (§7.2.1) ---------------------------------------------------------------------------
    def _detect_rx(self, bit: int) -> None:
        a = self._async
        a.append(bit)
        while len(a) >= 10:
            if a[0] == 0 and a[9] == 1:
                ch = sum(a[1 + j] << j for j in range(8))
                del a[:10]
                self._chars.append(ch)
                self._on_char(ch)
            else:
                del a[0]
            if self._phase != "detect":
                return

    def _on_char(self, ch: int) -> None:
        if self.calling:
            if ch == CHAR_NULL and len(self._chars) >= 2 and self._chars[-2] == CHAR_E:
                self._event("ADP: no error control desired")
                self.state = "unsupported"
            elif ch == CHAR_C and len(self._chars) >= 2 and self._chars[-2] == CHAR_E:
                self._event("ADP received")
                self.adp_seen = True
                self._phase = "protocol"
                self._begin_protocol()
        else:
            if ch in (DC1_EVEN, DC1_ODD):
                self._dc1.append(ch)
                tail = self._dc1[-4:]
                if len(tail) == 4 and all(tail[i] != tail[i + 1] for i in range(3)):
                    self._event("ODP received")
                    self.adp_seen = True                            # the originator is V.42 capable
                    self._adp_count = 0
                    self._answer_adp = True
            else:
                self._dc1.clear()

    def _hdlc_feed_for_start(self, bit: int) -> None:
        """Answerer: continuous flags or a frame also mean the protocol phase has started (§7.2.1.3)."""
        for frame, ok in self._hdlc.push(bit):
            if ok:
                self._phase = "protocol"
                self._begin_protocol()
                self._on_frame(frame)
                return
        if not self.calling and self._hdlc.continuous_flags >= 4:
            self._phase = "protocol"
            self._begin_protocol()

    _answer_adp = False

    def _begin_protocol(self) -> None:
        self.state = "idle"
        self._phase = "protocol"
        if self.calling:
            self._establish_at = self.bits_sent + 20 * 8          # at least 16 flags first (§8.10.2 note)
        self._hdlc = _Hdlc()

    # -- transmit scheduler -----------------------------------------------------------------------------------
    def _schedule(self) -> None:
        if self.state in ("down", "unsupported"):
            return
        if self._phase == "detect":
            if self.calling:
                if self.bits_sent > self._t400_at and (self.bits_sent - 1) % 36 == 0:
                    self._event("T400 expired without an ADP")
                    self.state = "unsupported"
                    return
                self._tx.extend(ODP_BLOCK)
            elif self._answer_adp:
                self._tx.extend(ADP_BLOCK)
                self._adp_count += 1
            else:
                if self.bits_sent > self._answer_detect_at and not self.adp_seen:
                    self._event("no ODP: originator is not V.42 capable")
                    self.state = "unsupported"
                    return
            return
        now = self.bits_sent
        if self.calling and not self._started:
            if now < self._establish_at:
                self._tx.extend(FLAG_BITS)
                return
            self._started = True
            if self._default_parameters:
                self._agreed = None
                self._sabme_sent = True
                self.stats['default_establishment'] += 1
                self._event("SABME with standard defaults; no XID initiated")
                self._u(SABME, True, 1)
                self._start_t401()
            else:
                self._send_xid_command()
            return
        # timer T401 (§8.4.8): acknowledgement overdue
        if self._t401_at is not None and now >= self._t401_at:
            self._t401_expired()
        if self.state == "idle":
            self._tx.extend(FLAG_BITS)
            return
        if self.state == "lapm":
            self._schedule_data()
            return
        self._tx.extend(FLAG_BITS)

    def _send(self, frame: bytes) -> None:
        self._tx.extend(FLAG_BITS)
        self._tx.extend(_stuffed(frame))

    def _u(self, ctrl: int, command: bool, pf: int = 0, info: bytes = b"") -> None:
        addr = self.cmd_addr if command else self.rsp_addr
        self._send(bytes((addr, ctrl | (0x10 if pf else 0))) + info)

    def _s(self, ctrl: int, command: bool, pf: int) -> None:
        addr = self.cmd_addr if command else self.rsp_addr
        self._send(bytes((addr, ctrl, (self.vr << 1) | (1 if pf else 0))))

    def _start_t401(self) -> None:
        self._t401_at = self.bits_sent + self._t401_bits

    def _send_xid_command(self) -> None:
        self._xid_sent += 1
        self.stats["xid_tx"] += 1
        self._u(XID, True, 0, build_xid(self.n401, self.k, self._offer))
        self._start_t401()

    def _t401_expired(self) -> None:
        self._tries += 1
        self.stats["t401_expiries"] += 1
        if self._tries > N400:
            self._event("N400 retransmissions exhausted")
            self.state = "down"
            return
        if self.state == "idle":
            if self.calling:
                if self._sabme_sent:
                    self._u(SABME, True, 1)
                elif self._xid_sent >= XID_ATTEMPTS and os.environ.get("V42_XID_FALLBACK", "1") != "0":
                    # Parameter negotiation is optional (V.42 7.2.2); a peer that never answers the XID may not implement it
                    # and wait for the SABME, so ask for the connection with the default parameters instead of repeating
                    # the XID for N400 * T401. V42_XID_FALLBACK=0 restores the XID-only behaviour.
                    self._event(f"XID unanswered after {self._xid_sent} commands: sending SABME with default parameters")
                    self._agreed = None
                    self._tries = 0
                    self._sabme_sent = True
                    self._xid_fallback = True
                    self._u(SABME, True, 1)
                else:
                    self._send_xid_command()
                self._start_t401()
            else:
                self._t401_at = None
            return
        # connected: poll the peer (timer recovery) and resend what is outstanding when it answers
        self._s(RR, True, 1)
        self._poll_out = True
        self._start_t401()

    # -- data transfer (§8.4) ---------------------------------------------------------------------------------
    def _window_open(self) -> bool:
        return len(self._unacked) < self.k and not self._peer_busy and not self._poll_out

    def _schedule_data(self) -> None:
        if self._need_rej and not self._rej_sent:
            self._s(REJ, False, 0)
            self.stats["rej_tx"] += 1
            self._rej_sent = True
            self._need_rej = False
            self._ack_pending = False
            return
        if self._retx:
            ns, payload = self._retx.popleft()
            self.stats["i_resent"] += 1
            self._send_i(ns, payload)
            return
        if self._enc is not None and not self._cq and self._txq:
            # compress a slice small enough that the result normally fits one frame; the byte stream may be cut anywhere
            raw = bytes(self._txq[:max(1, self.n401 * 3 // 4)])
            del self._txq[:len(raw)]
            packed = self._enc.compress(raw) + self._enc.flush()
            self.compressed_out += len(packed)
            for i in range(0, len(packed), self.n401):
                self._cq.append(packed[i:i + self.n401])
        if self._window_open():
            payload = None
            if self._enc is not None:
                if self._cq:
                    payload = self._cq.popleft()
            elif self._txq:
                payload = bytes(self._txq[:self.n401])
                del self._txq[:self.n401]
            if payload:
                self._queue_i(payload)
                self._send_i(*self._unacked[-1])
                return
        if self._ack_pending:
            self._s(RR, False, 0)
            self._ack_pending = False
            return
        self._tx.extend(FLAG_BITS)

    def _queue_i(self, payload: bytes) -> None:
        self._unacked.append([self.vs, payload])
        self.vs = (self.vs + 1) & 0x7F

    def _send_i(self, ns: int, payload: bytes) -> None:
        self.stats["i_tx"] += 1
        self.stats["wire_out"] += len(payload)
        self._send(bytes((self.cmd_addr, ns << 1, (self.vr << 1))) + payload)
        self._ack_pending = False
        if self._t401_at is None:
            self._start_t401()

    # -- reception -------------------------------------------------------------------------------------------------
    def _on_frame(self, f: bytes) -> None:
        if len(f) < 2:
            return
        addr, ctrl = f[0], f[1]
        if addr not in (self.cmd_addr, self.rsp_addr):
            return
        is_command = addr == self.rsp_addr                           # the peer's commands carry my response address
        if ctrl & 0x01 == 0:
            self._on_i(f)
        elif ctrl & 0x03 == 0x01:
            self._on_s(f, is_command)
        else:
            self._on_u(f, is_command)

    def _on_u(self, f: bytes, is_command: bool) -> None:
        ctrl, pf = f[1] & 0xEF, (f[1] >> 4) & 1
        if ctrl == XID:
            xid = parse_xid(f[2:])
            if is_command:
                self._answer_xid(xid or {})
            elif (xid is not None and self.state == "idle" and self.calling
                  and (not self._default_parameters or self._xid_sent)):
                self._xid_reply(xid)
        elif ctrl == SABME and is_command:
            if self.state in ("idle", "lapm"):
                self._u(UA, False, pf)
                self._connected()
        elif ctrl == UA and not is_command and self.state == "idle" and self.calling and self._sabme_sent:
            self._connected()
        elif ctrl == DM and not is_command:
            if self.state == "idle" and self.calling and not pf:
                return                                                # unsolicited DM: peer is ready, keep going
            self._event("DM received")
            self.state = "down"
        elif ctrl == DISC and is_command:
            self._u(UA, False, pf)
            self.state = "down"
            self._event("DISC received")
        elif ctrl == UA and self.state == "lapm":
            pass

    def _answer_xid(self, x: dict) -> None:
        """Responder side of the XID negotiation: agree on the lower values, accept V.42bis if we can (§8.10, Annex A)."""
        n401 = min(self.n401, x.get("tx_n401", DEFAULT_N401), x.get("rx_n401", DEFAULT_N401))
        k = min(self.k, x.get("tx_k", DEFAULT_K), x.get("rx_k", DEFAULT_K))
        self.n401, self.k = max(n401, 1), max(k, 1)
        comp = None
        if "p0" in x and self._offer:
            p0 = x["p0"] & self._offer[0]
            p1 = min(x.get("p1", 512), self._offer[1])
            p2 = min(x.get("p2", 6), self._offer[2])
            comp = (p0, p1, p2)
        self._agreed = comp
        self._u(XID, False, 0, build_xid(self.n401, self.k, comp))

    def _xid_reply(self, x: dict) -> None:
        """Originator: the XID response carries the agreed values; then request the connection (SABME)."""
        self.n401 = max(min(self.n401, x.get("tx_n401", self.n401), x.get("rx_n401", self.n401)), 1)
        self.k = max(min(self.k, x.get("tx_k", self.k), x.get("rx_k", self.k)), 1)
        # A response cannot enable a direction or size we did not offer.
        # In particular, %C0 must never create an encoder/decoder merely
        # because the peer replies with a nonzero P0.
        if self._offer and "p0" in x:
            self._agreed = (x['p0'] & self._offer[0],
                            min(x.get('p1',512),self._offer[1]),
                            min(x.get('p2',6),self._offer[2]))
        else:
            self._agreed = None
        self._xid_done = True
        self._tries = 0
        self._sabme_sent = True
        self._u(SABME, True, 1)
        self._start_t401()

    _agreed: tuple[int, int, int] | None = None

    def _connected(self) -> None:
        if self.state == "lapm":
            return
        self._reset_link()
        self._t401_at = None
        self.state = "lapm"
        self._event("connected")
        a = self._agreed
        if a and a[0] in (1, 2, 3) and a[1] >= 512 and 6 <= a[2] <= 250:
            self.compression = a
            # P0: bit 0 = initiator -> responder, bit 1 = responder -> initiator
            tx_dir = 1 if self.calling else 2
            rx_dir = 2 if self.calling else 1
            self._enc = Encoder(a[1], a[2]) if a[0] & tx_dir else None
            self._dec = Decoder(a[1], a[2]) if a[0] & rx_dir else None

    def _ack(self, nr: int) -> None:
        """N(R) acknowledges everything before it (§8.4.6)."""
        if self._unacked:
            n = (nr - self._unacked[0][0]) & 0x7F
            # Only an acknowledgement that covers outstanding frames stops and restarts T401 (§8.4.8). One that acknowledges
            # nothing new (a peer's I frames repeating the same N(R) because our frames were lost) must leave the timer
            # running; restarting it on every such frame kept T401 from ever expiring while the peer kept sending, and the
            # lost frames were never polled for or resent (lab/V34_FREEZE_GUARD.md, LAPM stall after a retrain).
            if 0 < n <= len(self._unacked):
                # Payload octets covered by a valid advancing N(R), rather
                # than queued bytes or repeated transmissions. This remains
                # meaningful when application traffic is outbound only.
                self.stats['wire_acked'] += sum(len(frame[1]) for frame in self._unacked[:n])
                del self._unacked[:n]
                # During timer recovery, an advancing N(R) does not answer
                # the outstanding poll. Preserve its deadline/retry count
                # until a supervisory response with F=1 arrives (§8.4.8).
                # Otherwise a delayed I/RR acknowledgement strands the link
                # with _poll_out=True and no timer to retry the lost final.
                if not self._poll_out:
                    self._tries = 0
                    self._t401_at = None
                    if self._unacked:
                        self._start_t401()

    def _on_s(self, f: bytes, is_command: bool) -> None:
        if self.state != "lapm" or len(f) < 3:
            return
        ctrl, nr, pf = f[1], f[2] >> 1, f[2] & 1
        if ctrl == RR or ctrl == RNR:
            self._peer_busy = ctrl == RNR
            self._ack(nr)
            if not is_command and pf and self._poll_out:
                self._poll_out = False
                self._t401_at = None
                self._retx = deque((u[0], u[1]) for u in self._unacked)      # the poll answered: go back to V(A)
                if self._unacked:
                    self._start_t401()
        elif ctrl == REJ:
            self.stats["rej_rx"] += 1
            self._peer_busy = False
            self._ack(nr)
            # A REJ received during timer recovery updates V(A), but only
            # a response with F=1 answers our outstanding poll (§8.4.4).
            # A peer command with P=1 must still be answered below without
            # cancelling recovery in this direction.
            if not self._poll_out or (not is_command and pf):
                self._poll_out = False
                self._retx = deque((u[0], u[1]) for u in self._unacked)
                self._t401_at = None
                if self._unacked:
                    self._start_t401()
        if is_command and pf:
            self._s(RR, False, 1)                                         # answer a poll at once
            self._ack_pending = False

    def _maybe_assume_bis(self) -> None:
        """Opt-in (``V42_ASSUME_BIS``): after a SABME without XID, decode the peer's data as V.42bis from its first I frame.
        Some answerers compress by default although nothing was negotiated (dialup.world, main-1 call); plain text passes through
        a decoder in transparent mode and the 00 00 (ESC+ECM) pair switches it to compressed mode. ``V42_ASSUME_BIS=1`` uses
        N2=1024, N7=32 (N7=32 is what that stream decodes with); ``V42_ASSUME_BIS=N2,N7`` sets them. Receive direction only; a peer
        that sends raw 00 octets would be misread, which is why this is off by default. A decode error drops back to raw delivery."""
        spec = os.environ.get("V42_ASSUME_BIS", "")
        if spec in ("", "0"):
            return
        try:
            n2, n7 = (int(v) for v in spec.split(",")) if "," in spec else (1024, 32)
        except ValueError:
            return
        self._dec = Decoder(n2, n7)
        self._bis_assumed = True
        self._event(f"V.42bis assumed on receive after SABME without XID (N2={n2}, N7={n7})")

    def _on_i(self, f: bytes) -> None:
        if self.state != "lapm" or len(f) < 3:
            return
        ns, nr, pf = f[1] >> 1, f[2] >> 1, f[2] & 1
        self._ack(nr)
        if ns == self.vr:
            self.vr = (self.vr + 1) & 0x7F
            self._rej_sent = False
            self._ack_pending = True
            payload = bytes(f[3:])
            self.stats["i_rx"] += 1
            self.stats["wire_in"] += len(payload)
            if self._dec is None and self._xid_fallback and self.compression is None and not self._bis_assumed:
                self._maybe_assume_bis()
            if self._dec is not None:
                try:
                    payload = self._dec.decompress(payload)
                except V42bisError as exc:
                    self._event(f"V.42bis error: {exc}")
                    if not self._bis_assumed:
                        self.state = "down"
                        return
                    self._dec = None                                    # assumed compression was wrong: deliver raw from here
            self.stats["bytes_in"] += len(payload)
            self._out += payload
            if pf:
                self._s(RR, False, 1)
                self._ack_pending = False
        else:
            if not self._rej_sent:
                self._need_rej = True
            elif pf:
                self._s(RR, False, 1)
