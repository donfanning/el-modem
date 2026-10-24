"""ITU-T V.42 error control (LAPM) over a raw bit stream, written from the Recommendation (03/2002 text).

It covers the detection phase (ODP / ADP, §7.2.1), HDLC framing with zero insertion and the 16-bit FCS (§8.1), the
modulo-128 LAPM frames and procedures (§8.2-8.9: SABME/UA/DM/DISC, I, RR, RNR, REJ, XID, TEST), parameter negotiation by XID
(§8.10, §12.2, N401 and k), and V.42bis data compression negotiated in the XID private parameter group (V.42bis
Annex A) using our own v42bis.py. LAPM BRK/BRKACK sequencing and retry is
exposed as an L-SIGNAL API and bridged to native serial BREAK by the DCE;
expedited ordering is supported only by native serial backends that can
reorder queued output. The optional TEST procedure is negotiated and exposed
through an L-TEST indication/request API. Optional connection monitoring uses
timer T403 (60 seconds by default). Not implemented: selective retransmission
(SREJ) and 32-bit FCS; these remain unrequested in our XID. The receiver bounds
an unclosed HDLC frame per §8.5.5, validates N401 and control-frame lengths, and terminates on invalid N(R),
unsupported controls, or received FRMR as required by §§8.5.2 and 8.5.5. An unnegotiated SREJ is an unrecognized
control field.

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
from v42_detect import ADP_ARM_BITS

FLAG = 0x7E
DEFAULT_N401, DEFAULT_K = 128, 15
N400 = 10
XID_ATTEMPTS = 2                  # unanswered XID commands before the originator skips negotiation and sends SABME (V.42 7.2.2)
_XID_V42BIS_PROCEDURAL_ERROR = "_v42bis_procedural_error"
# unnumbered frame control values (P/F bit 0x10)
SABME, UA, DM, DISC, FRMR, XID, UI, TEST = 0x6F, 0x63, 0x0F, 0x43, 0x87, 0xAF, 0x03, 0xE3
RR, RNR, REJ, SREJ = 0x01, 0x05, 0x09, 0x0D
ODP_BLOCK = [int(c) for c in "0100010001" + "1" * 8 + "0100010011" + "1" * 8]       # DC1 even parity, DC1 odd parity
ADP_BLOCK = [int(c) for c in "0101000101" + "1" * 8 + "0110000101" + "1" * 8]        # E, C: V.42 supported
FLAG_BITS = [0, 1, 1, 1, 1, 1, 1, 0]
CHAR_E, CHAR_C, CHAR_NULL, DC1_EVEN, DC1_ODD = 0x45, 0x43, 0x00, 0x11, 0x91
ADP_MIN_REPETITIONS = 10         # answerer sends ADP at least ten times after ODP (V.42 §7.2.1.3)
DETECTION_MIN_MARK_BITS = 8      # V.42 §7.2.1 ODP/ADP character spacing
DETECTION_MAX_MARK_BITS = 16


class _Hdlc:
    """Bit-level HDLC receiver: flag hunt, zero deletion, FCS-16.

    ``push`` returns completed ``(frame, fcs_ok)`` pairs. ``frame is None``
    means the delimited bits could not form an octet-aligned frame; a byte
    string with ``fcs_ok=False`` is a well-formed frame with a bad FCS.
    Keeping those cases distinct is required by V.42 §8.5.4: only the optional
    FCS-error response procedure may send REJ; other invalid frames are
    discarded without action.
    """

    def __init__(self) -> None:
        self.ones = 0
        self.bits: list[int] = []
        self.active = False
        # V.42 §8.5.5 permits an unbounded frame to be discarded after twice
        # the largest permitted frame plus two octets without a closing flag.
        # Count destuffed bits here; the limit uses the maximum default N401
        # so it remains safe even before XID establishes a smaller value.
        self.max_unflagged_bits = (2 * (DEFAULT_N401 + 5) + 2) * 8
        self.overlong = 0
        self.flags = 0                         # cumulative flags seen since reset
        self.continuous_flags = 0
        self._flag_gap = 9

    def _append(self, bit: int) -> None:
        if not self.active:
            return
        self.bits.append(bit)
        if len(self.bits) > self.max_unflagged_bits:
            self.bits.clear()
            self.active = False
            self.overlong += 1

    def push(self, bit: int) -> list[tuple[bytes | None, bool]]:
        out: list[tuple[bytes | None, bool]] = []
        # Adjacent flags end seven (shared zero) or eight bits apart.
        if self._flag_gap < 9:
            self._flag_gap += 1
        if bit:
            self.ones += 1
            if self.active and self.ones <= 6:
                self._append(1)
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
                    frame = raw[:-2]
                    # V.42 §8.1.3 sets the minimum protected length to five
                    # octets for I/S frames (address, two-octet control,
                    # FCS-16) and four for U frames (address, one-octet
                    # control, FCS-16). Discard these malformed frames here,
                    # before they can affect link counters or T403 state.
                    if len(raw) >= 4 and len(frame) >= 2:
                        control = frame[1]
                        sequenced = (control & 1) == 0 or (control & 3) == 1
                        minimum = 5 if sequenced else 4
                        if len(raw) >= minimum:
                            out.append((frame, crc16_hdlc(frame) == int.from_bytes(raw[-2:], "little")))
                else:
                    out.append((None, False))
            self.active, self.bits = True, []
            return out
        if self.active:
            self._append(0)
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


def build_xid(n401: int, k: int, compression: tuple[int, int, int] | None,
              *, rx_n401: int | None = None, rx_k: int | None = None,
              include_functions: bool = True,
              include_test: bool = False,
              include_compression: bool = True) -> bytes:
    """XID information field (§12.2.2): general-purpose FI, parameter group, optional TEST and V.42bis."""
    rx_n401 = n401 if rx_n401 is None else rx_n401
    rx_k = k if rx_k is None else rx_k
    function_mask = 0x0000898A | (0x00002000 if include_test else 0)
    params = (bytes((3, 4)) + function_mask.to_bytes(4, "little")
              if include_functions else b"")     # HDLC optional functions
    def xid_integer(value: int) -> bytes:
        if not isinstance(value, int) or value < 0:
            raise ValueError("XID integer parameters must be non-negative integers")
        width = max(1, (value.bit_length() + 7) // 8)
        return value.to_bytes(width, "big")

    tx_n401_bits = xid_integer(n401 * 8)
    rx_n401_bits = xid_integer(rx_n401 * 8)
    params += bytes((5, len(tx_n401_bits))) + tx_n401_bits
    params += bytes((6, len(rx_n401_bits))) + rx_n401_bits
    params += bytes((7, 1, k, 8, 1, rx_k))
    out = bytes((0x82, 0x80)) + len(params).to_bytes(2, "big") + params
    if include_compression:
        # V.42bis Annex A: P0=0 explicitly requests neither direction.
        # Send the negative request too, rather than relying on omission being
        # interpreted as the default by legacy peers. This never enables BIS.
        p0, p1, p2 = compression if compression is not None else (0, 1024, 32)
        if not 0 <= p0 <= 3:
            raise ValueError("V.42bis P0 must be 0..3")
        if not 512 <= p1 <= 65535:
            raise ValueError("V.42bis P1 must be 512..65535")
        if not 6 <= p2 <= 250:
            raise ValueError("V.42bis P2 must be 6..250")
        priv = bytes((0, 3)) + b"V42" + bytes((1, 1, p0, 2, 2)) + p1.to_bytes(2, "big") + bytes((3, 1, p2))
        out += bytes((0xF0,)) + len(priv).to_bytes(2, "big") + priv
    return out


def parse_xid(info: bytes) -> dict | None:
    """Parameters found in an XID information field (unknown groups and parameters are ignored)."""
    # V.42 §12.2.1 allows the general-purpose FI followed by zero or more
    # data-link groups. An FI-only field is therefore valid and means that no
    # parameter values were supplied; callers use the standard defaults.
    if not info or info[0] != 0x82:
        return None
    out: dict = {}
    v42bis_parameters_seen = False
    last_group_id = -1
    i = 1
    while i < len(info):
        if info[i] == 0xFF:
            # V.42 §12.2.1.3's optional user-data subfield follows the
            # length-prefixed data-link groups but has no GL. Parse its
            # PI/PL/PV sequence for structural integrity, then ignore values
            # we do not implement (for example, V.44 negotiation).
            body = info[i + 1:]
            j = 0
            while j < len(body):
                if j + 2 > len(body):
                    return None
                plen = body[j + 1]
                j += 2
                if j + plen > len(body):
                    return None
                j += plen
            break
        if i + 3 > len(info):
            return None
        gid, glen = info[i], int.from_bytes(info[i + 1:i + 3], "big")
        # V.42 §12.2.1.2 requires data-link subfields to appear in ascending
        # GI order. Reject duplicates as well: accepting them would make a
        # peer's effective parameter values depend on field order.
        if gid <= last_group_id:
            return None
        last_group_id = gid
        i += 3
        if i + glen > len(info):
            return None
        body = info[i:i + glen]
        i += glen
        parameters = []
        j = 0
        while j < len(body):
            if j + 2 > len(body):
                return None
            pid, plen = body[j], body[j + 1]
            j += 2
            if j + plen > len(body):
                return None
            raw = body[j:j + plen]
            j += plen
            parameters.append((pid, plen, raw))
        if gid == 0x80:
            for pid, plen, raw in parameters:
                val = int.from_bytes(raw, "big")
                if pid == 3 and plen == 4:
                    out["functions"] = int.from_bytes(raw, "little")
                elif pid == 5 and plen > 0:
                    # N401 is carried in bits for XID but its negotiated value
                    # is an integral number of octets (V.42 §12.2.2, Note 3).
                    # Do not silently floor a malformed fractional-octet value.
                    if val % 8 == 0:
                        out["tx_n401"] = val // 8
                elif pid == 6 and plen > 0:
                    if val % 8 == 0:
                        out["rx_n401"] = val // 8
                elif pid == 7 and plen == 1:
                    out["tx_k"] = val
                elif pid == 8 and plen == 1:
                    out["rx_k"] = val
        elif gid == 0xF0:
            # V.42 §12.2.2, Table 11b requires PI=0 / "V42" to be first.
            # A missing/foreign parameter-set identifier means these are not
            # V.42bis parameters; do not let them negotiate compression.
            if not parameters or parameters[0] != (0, 3, b"V42"):
                continue
            v42bis_parameters_seen = True
            for pid, plen, raw in parameters[1:]:
                val = int.from_bytes(raw, "big")
                if pid == 1 and plen == 1:
                    # V.42 bis Annex A encodes P0 as 000000nn. Do not mask
                    # reserved high bits later: doing so could turn a malformed
                    # value into a request for valid compression directions.
                    if val <= 3:
                        out["p0"] = val
                elif pid == 2 and plen == 2:
                    out["p1"] = val
                elif pid == 3 and plen == 1:
                    out["p2"] = val
    if v42bis_parameters_seen:
        # V.42bis §5.1 makes these bounds normative: P1 (N2) cannot be
        # below 512, and P2 (N7) must be in 6..250. Local capability limits
        # are applied later; they must not conceal an invalid peer value.
        p1 = out.get("p1", 512)
        p2 = out.get("p2", 6)
        if p1 < 512 or not 6 <= p2 <= 250:
            out[_XID_V42BIS_PROCEDURAL_ERROR] = (
                f"invalid V.42bis parameters P1={p1} P2={p2}"
            )
    return out


def encode_brk(sequence: int, *, destructive: bool = False,
               expedited: bool = False, length_ms: int | None = None) -> bytes:
    """Encode a V.42 §12.3.1 BRK message (bit 1 is the least-significant bit)."""
    if sequence not in (0, 1):
        raise ValueError("BRK sequence must be 0 or 1")
    if length_ms is not None and (not isinstance(length_ms, int) or length_ms < 0):
        raise ValueError("break length must be a non-negative integer number of milliseconds")
    first = 0x40 | (sequence << 7)  # BRK type and N(SB)
    option = (0x80 if destructive else 0) | (0x40 if expedited else 0)  # D,S
    info = bytes((first, option))
    if length_ms is not None:
        length = (length_ms + 9) // 10
        info += bytes((min(length, 255),))
    return info


def decode_brk(info: bytes) -> dict | None:
    """Decode a well-formed BRK message; return None for reserved/malformed UI data."""
    if len(info) not in (2, 3) or (info[0] & 0x7F) != 0x40 or (info[1] & 0x3F):
        return None
    length = info[2] if len(info) == 3 else None
    return dict(sequence=(info[0] >> 7) & 1,
                destructive=bool(info[1] & 0x80),
                expedited=bool(info[1] & 0x40),
                length_ms=(None if length in (None, 0) else length * 10),
                length_code=length)


def encode_brkack(sequence: int) -> bytes:
    """Encode a V.42 §12.3.2 BRKACK message."""
    if sequence not in (0, 1):
        raise ValueError("BRKACK sequence must be 0 or 1")
    return bytes((0x60 | (sequence << 7),))


class V42Link:
    def __init__(self, calling_party: bool, detect: bool = True, bit_rate: int = 9600, frame_size: int = 0,
                 window: int = 0, compression: tuple[int, int, int] | None = None, t400: float = 0.75,
                 receive_frame_size: int | None = None, receive_window: int | None = None,
                 xid_fallback: bool | None = None, t403: float | None = 60.0):
        # Zero is the internal "use V.42 default" sentinel. Explicit values
        # must be valid protocol parameters; silently retaining a negative
        # value eventually produces an invalid XID or broken frame limits.
        for name, value, maximum in (
            ("frame_size", frame_size, 128),
            ("receive_frame_size", frame_size if receive_frame_size is None else receive_frame_size, 128),
            ("window", window, 127),
            ("receive_window", window if receive_window is None else receive_window, 127),
        ):
            if (not isinstance(value, int) or isinstance(value, bool)
                    or not 0 <= value <= maximum):
                raise ValueError(f"{name} must be 0 (default) or in 1..{maximum}")
        self.calling = calling_party
        self.rate = max(bit_rate, 300)
        self.state = "detect" if detect else "idle"
        self.adp_seen = False
        self.events: list[str] = []
        self.compression: tuple[int, int, int] | None = None      # agreed (p0, p1, p2) once connected
        self.compressed_out = 0
        # Link statistics for the modem's diagnostic screens (ATI6/ATI11)
        self.stats = dict(xid_tx=0, i_tx=0, i_rx=0, i_resent=0, rej_tx=0, rej_rx=0, nr_errors=0,
                          frame_reject_errors=0, fcs_errors=0, t401_expiries=0,
                          disc_tx=0, disc_rx=0, ua_tx=0, ua_rx=0, dm_tx=0, dm_rx=0,
                          bytes_in=0, bytes_out=0, wire_in=0, wire_out=0, wire_acked=0,
                          hdlc_good=0, hdlc_bad=0, t403_expiries=0)
        self.bits_sent = 0
        self.cmd_addr = 0x03 if calling_party else 0x01
        self.rsp_addr = 0x01 if calling_party else 0x03
        # Opt-in diagnostic: V.42 7.2.2 permits establishment without
        # negotiation when the standard defaults suffice. This is distinct
        # from abandoning an attempted XID exchange after response loss.
        self._default_parameters = os.environ.get("V42_DEFAULT_PARAMETERS", "0") == "1"
        # If a caller's XID response never arrives, a compatibility policy may
        # proceed with defaults for peers that do not implement XID. Strict
        # policy keeps retrying through N400, then fails the attempted V.42
        # establishment. DCE callers pass an explicit per-modem setting;
        # the environment variable remains only as a default for direct users.
        self.xid_fallback = (os.environ.get("V42_XID_FALLBACK", "1") != "0"
                             if xid_fallback is None else bool(xid_fallback))
        self._xid_fallback_suppressed = False
        if self._default_parameters and compression is not None:
            raise ValueError("default-parameter LAPM requires compression disabled")
        self.tx_n401 = DEFAULT_N401 if self._default_parameters else min(frame_size or DEFAULT_N401, DEFAULT_N401)
        self.rx_n401 = (DEFAULT_N401 if self._default_parameters else
                        min((receive_frame_size if receive_frame_size is not None else frame_size)
                            or DEFAULT_N401, DEFAULT_N401))
        self.tx_k = DEFAULT_K if self._default_parameters else min(window or DEFAULT_K, DEFAULT_K)
        rx_window = receive_window if receive_window is not None else window
        self.rx_k = DEFAULT_K if self._default_parameters else min(rx_window or DEFAULT_K, DEFAULT_K)
        self.stats['default_establishment'] = 0
        self._offer = compression
        self._fatal_protocol_error = False
        # Set by a negotiated V.42bis decoder when its dictionary loses
        # synchronization. The V.42 control-function layer decides whether
        # to attempt L-RELEASE/L-ESTABLISH recovery or fail the call.
        self._compression_error: str | None = None
        self._compression_error_octets = 0
        self._t400_bits = int(t400 * self.rate)
        self._t400_at = self._t400_bits
        self._answer_detect_at = self._t400_bits * 2
        if t403 is not None and float(t403) <= 0:
            raise ValueError("T403 must be positive or None to disable")
        self._t403_seconds = None if t403 is None else float(t403)
        self._t403_bits = (None if self._t403_seconds is None
                           else max(1, int(self._t403_seconds * self.rate)))
        self._t401_bits = int(max(1.5, 2.2 * (self.tx_n401 + 8) * 8 / self.rate + 1.0) * self.rate)
        self._tx: deque[int] = deque()
        self._hdlc = _Hdlc()
        self._async: list[int] = []
        self._detect_bit_count = 0
        self._last_detect_char_end: int | None = None
        self._chars: list[int] = []                                # characters seen during detection
        self._char_gaps: list[int | None] = []                     # mark bits after prior async stop bit
        self._dc1: list[int] = []
        self._in = bytearray()
        self._out = bytearray()
        self._txq = bytearray()                                    # user data not yet in an I frame
        self._adp_count = 0
        self._deferred_xid_response: bytes | None = None
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
    @property
    def n401(self) -> int:
        """Compatibility alias for this endpoint's transmit N401."""
        return self.tx_n401

    @n401.setter
    def n401(self, value: int) -> None:
        self.tx_n401 = value

    @property
    def k(self) -> int:
        """Compatibility alias for this endpoint's transmit window."""
        return self.tx_k

    @k.setter
    def k(self, value: int) -> None:
        self.tx_k = value

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
        for name in ('_t401_at', '_t400_at', '_answer_detect_at', '_break_t401_at', '_t403_at'):
            deadline = getattr(self, name)
            if deadline is not None:
                setattr(self, name, now + math.ceil(max(0, deadline-now) * ratio))
        self._t400_bits = math.ceil(self._t400_bits * ratio)
        if self._t403_seconds is not None:
            self._t403_bits = max(1, math.ceil(self._t403_seconds * rate))
        self.rate = rate
        self._t401_bits = int(max(1.5, 2.2 * (self.tx_n401 + 8) * 8 / rate + 1.0) * rate)

    def close(self) -> None:
        self._tx.clear()

    def set_local_busy(self, busy: bool) -> None:
        """Advertise receive-side DTE backpressure with LAPM RNR/RR."""
        busy = bool(busy)
        if self._renegotiation is not None and not busy:
            return
        if self.state != "lapm" or busy == self._local_busy:
            return
        self._local_busy = busy
        self._ack_pending = True

    def renegotiate_parameters(self, *, tx_n401: int, rx_n401: int,
                               tx_k: int, rx_k: int) -> bool:
        """Start N401/window renegotiation on an established LAPM link.

        The initiator must have no outstanding I frames before beginning. The
        procedure sends RNR(P=1), pauses new I-frame transmission, and then
        negotiates directional N401/k values with XID. Compression remains in
        its current state; this API never changes V.42bis dictionaries.
        """
        values = (tx_n401, rx_n401, tx_k, rx_k)
        if any(not isinstance(value, int) or isinstance(value, bool)
               for value in values):
            return False
        if (not 18 <= tx_n401 <= DEFAULT_N401
                or not 18 <= rx_n401 <= DEFAULT_N401
                or not 1 <= tx_k <= DEFAULT_K
                or not 1 <= rx_k <= DEFAULT_K):
            return False
        if (self.state != "lapm" or self._renegotiation is not None
                or self.compression is not None
                or self._local_busy or self._peer_busy or self._poll_out
                or self._unacked or self._retx or self._cq
                or self._ack_pending or self._need_rej
                or self._break_tx_pending is not None
                or self._break_tx_waiting is not None
                or self._break_rx_confirmations):
            return False
        info = build_xid(tx_n401, tx_k, None, rx_n401=rx_n401, rx_k=rx_k,
                         include_functions=False, include_compression=False)
        if len(info) > min(self.tx_n401, self.rx_n401):
            return False
        self._renegotiation = dict(info=info, tx_n401=tx_n401,
                                   rx_n401=rx_n401, tx_k=tx_k, rx_k=rx_k)
        self._xid_pause_tx = True
        self._local_busy = True
        self._ack_pending = False
        self._tries = 0
        self._xid_poll_out = True
        self._s(RNR, True, 1)
        self._send_renegotiation_xid()
        self._event("connected N401/window renegotiation started; RNR(P=1) sent")
        return True

    def _send_renegotiation_xid(self) -> None:
        assert self._renegotiation is not None
        self.stats["xid_tx"] += 1
        self._u(XID, True, 0, self._renegotiation["info"])
        self._start_t401()

    def release(self) -> bool:
        """Request orderly LAPM release with DISC (§8.7); return False if inactive."""
        if self.state != "lapm":
            return False
        # §8.7.1 discards queued data and associated outstanding frames.
        self._release_tx_snapshot = self._snapshot_release_tx()
        self._tx.clear()
        self._tx.extend([1] * 8)
        self._txq.clear()
        self._cq.clear()
        self._unacked.clear()
        self._retx.clear()
        self._clear_break_state()
        self._ack_pending = self._need_rej = self._poll_out = False
        self._xid_poll_out = False
        self._renegotiation = None
        self._xid_pause_tx = False
        self._peer_xid_pause = False
        self._peer_xid_pending = False
        self.vs = self.va = self.vr = 0
        self._tries = 0
        self._u(DISC, True, 1)
        self.state = "releasing"
        self._start_t401()
        self._event("LAPM orderly release requested; DISC sent")
        return True

    def request_test(self, information: bytes) -> bool:
        """Issue an L-TEST request after the optional TEST procedure is agreed."""
        information = bytes(information)
        if (self.state != "lapm" or not self._test_negotiated
                or len(information) > self.tx_n401):
            return False
        self._u(TEST, True, 0, information)
        self._event(f"TEST request sent ({len(information)} information octets)")
        return True

    def take_test_indication(self) -> bytes | None:
        """Deliver one received TEST information field to the control function."""
        return self._test_indications.popleft() if self._test_indications else None

    def request_break(self, *, length_ms: int | None = None,
                      destructive: bool = False, expedited: bool = False) -> bool:
        """Request V.42 break transfer; expedited selects S=1, otherwise wait in sequence."""
        if (self.state != "lapm" or self._break_tx_pending is not None
                or self._break_tx_waiting is not None):
            return False
        info = encode_brk(self._break_send_seq, destructive=destructive,
                          expedited=expedited, length_ms=length_ms)
        if len(info) > self.n401:
            return False
        self._break_tx_tries = 0
        self._break_tx_expedited = bool(expedited)
        self._break_tx_destructive = bool(destructive)
        if destructive:
            # §7.4 destructive handling discards queued (not-yet-transmitted)
            # application data. Preserve already-transmitted frames awaiting
            # acknowledgement/retransmission and any current serialized bits.
            self._txq.clear()
            self._cq.clear()
        self._break_tx_reset_after_send = bool(destructive and expedited)
        if expedited:
            self._break_tx_pending = info
            self._send_break_frame()
        else:
            self._break_tx_waiting = info
            self._event("BRK queued in sequence behind prior I-frame data")
            self._maybe_send_ordered_break()
        return True

    def read_break_events(self) -> list[dict]:
        """Return received BRK indications and local BRK confirmation/failure events."""
        events, self._break_events = list(self._break_events), deque()
        return events

    def acknowledge_break(self) -> bool:
        """Complete one accepted L-SIGNAL indication and transmit its BRKACK."""
        if not self._break_rx_confirmations or self.state != "lapm":
            return False
        confirmation = self._break_rx_confirmations[0]
        message = confirmation["message"]
        reset = message["destructive"] and message["expedited"]
        # Finish an already serialized frame before resetting sequence state.
        # While waiting, tx_bit() idles rather than scheduling another data
        # frame. This preserves HDLC framing and prevents stale queued data
        # from crossing the break boundary.
        if reset and self._tx:
            return False
        self._break_rx_confirmations.popleft()
        if reset:
            self._reset_data_state_for_break()
            self._break_rx_reset_applied = True
        info = confirmation["info"]
        for pf in confirmation["pfs"]:
            self._u(UI, False, pf, info)
        self._break_rx_last_ack = info
        self._event("BRKACK sent after control-function confirmation")
        self._start_t403()
        return True

    def _reset_data_state_for_break(self) -> None:
        """Reset non-BRK protocol/data state for destructive expedited BREAK.

        V.42 §7.5 Table 5 resets control/error-control state except state
        pertaining to BREAK transfer. The negotiated link and compression
        parameters remain selected; their sequence and dictionary state reset.
        The BREAK transaction restarts T403 when its acknowledgement completes.
        """
        self.vs = self.va = self.vr = 0
        self._unacked.clear()
        self._retx.clear()
        self._cq.clear()
        self._txq.clear()
        self._out.clear()
        self._peer_busy = False
        self._peer_xid_pause = False
        self._peer_xid_pending = False
        self._local_busy = False
        self._ack_pending = False
        self._rej_sent = False
        self._need_rej = False
        self._t401_at = None
        self._tries = 0
        self._poll_out = False
        self._xid_poll_out = False
        self._xid_pause_tx = False
        self._renegotiation = None
        self._ua_pending_until = None
        self._deferred_xid_response = None
        if self._enc is not None:
            self._enc.init()
        if self._dec is not None:
            self._dec.init()

    def _clear_break_state(self) -> None:
        self._break_send_seq = self._break_recv_seq = 0
        self._break_tx_pending: bytes | None = None
        self._break_tx_waiting: bytes | None = None
        self._break_tx_expedited = False
        self._break_tx_destructive = False
        self._break_tx_tries = 0
        self._break_tx_reset_after_send = False
        self._break_t401_at: int | None = None
        self._break_rx_confirmations: deque = deque()
        self._break_events: deque = deque()
        self._break_rx_last_ack: bytes | None = None
        self._break_rx_reset_applied = False
        self._post_break_txq = bytearray()

    def _send_break_frame(self) -> None:
        if self._break_tx_pending is None:
            return
        # BRK has its own T401 transaction. T403 must not run concurrently
        # with an outstanding acknowledgement timer (§8.12.3.2).
        self._t403_at = None
        self._u(UI, True, 0, self._break_tx_pending)
        self._break_t401_at = self.bits_sent + self._t401_bits
        self._event(f"BRK sent with N(SB)={self._break_send_seq}")

    def _maybe_send_ordered_break(self) -> bool:
        if (self._break_tx_waiting is None or self._txq or self._cq or self._retx
                or self._unacked or self._ack_pending or self._need_rej):
            return False
        self._break_tx_pending = self._break_tx_waiting
        self._break_tx_waiting = None
        self._send_break_frame()
        return True

    def _handle_break_frame(self, f: bytes, is_command: bool, pf: int) -> bool:
        """Handle V.42 UI break user-control messages; return if recognized."""
        if len(f) <= 2:
            return False
        info = f[2:]
        message_type = info[0] & 0x7F
        if message_type == 0x40:
            message = decode_brk(info)
            if message is None or not is_command:
                return True
            sequence = message["sequence"]
            if sequence == self._break_recv_seq:
                self._break_recv_seq ^= 1
                response = encode_brkack(self._break_recv_seq)
                self._break_rx_confirmations.append({"pfs": [pf], "info": response,
                                                     "message": message})
                self._break_events.append(dict(type="break_indication", **message))
                self._event(f"BRK indication queued; N(SB)={sequence}")
            elif (self._break_rx_confirmations
                  and sequence == (self._break_recv_seq ^ 1)):
                # A retry may arrive before the DTE action lets us send the
                # first BRKACK. Defer its matching F-bit response too.
                self._break_rx_confirmations[-1]["pfs"].append(pf)
            elif self._break_rx_last_ack is not None:
                # A duplicate must not repeat the terminal-side BREAK.
                # Its response F bit mirrors this command's P bit.
                self._u(UI, False, pf, self._break_rx_last_ack)
                self._event(f"duplicate BRK N(SB)={sequence}; prior BRKACK retransmitted")
            return True
        if message_type == 0x60:
            if len(info) != 1 or is_command:
                return True
            sequence = (info[0] >> 7) & 1
            if (self._break_tx_pending is not None
                    and sequence == (self._break_send_seq ^ 1)):
                self._finish_break_tx(True, f"BRKACK received; N(RB)={sequence}")
            return True
        return False

    def _finish_break_tx(self, success: bool, reason: str) -> None:
        info = self._break_tx_pending
        if info is None:
            return
        sequence = (info[0] >> 7) & 1
        if success:
            self._break_send_seq ^= 1
        self._break_events.append(dict(type="break_confirmed" if success else "break_failed",
                                       sequence=sequence, reason=reason))
        self._event(reason)
        self._break_tx_pending = None
        self._break_tx_waiting = None
        self._break_t401_at = None
        self._break_tx_tries = 0
        self._break_tx_expedited = self._break_tx_destructive = False
        self._break_tx_reset_after_send = False
        if self._post_break_txq:
            self._txq.extend(self._post_break_txq)
            self._post_break_txq.clear()
        self._start_t403()

    def write(self, data: bytes) -> None:
        self.stats["bytes_out"] += len(data)
        if self._break_tx_pending is not None or self._break_tx_waiting is not None:
            self._post_break_txq += data
        else:
            self._txq += data

    def request_v42bis_reset(self) -> bool:
        """Queue a V.42bis RESET at the current transmit-stream boundary.

        Already queued DTE bytes are encoded before the command; later writes
        remain behind it. The RESET travels in ordinary LAPM I frames, so
        outstanding frame acknowledgement is not required. Return False when
        V.42bis transmit compression is not active or another control procedure
        makes inserting a stream boundary unsafe.
        """
        if (self.state != "lapm" or self._enc is None
                or self._break_tx_pending is not None or self._break_tx_waiting is not None
                or self._break_tx_reset_after_send or self._renegotiation is not None):
            return False

        # Preserve the byte ordering around the reset even when DTE data has
        # not yet reached the compression queue. Mirror the regular scheduler's
        # bounded chunk size and flush each chunk into the ordered wire queue.
        while self._txq:
            raw = bytes(self._txq[:max(1, self.tx_n401 * 3 // 4)])
            del self._txq[:len(raw)]
            packed = self._enc.compress(raw) + self._enc.flush()
            self.compressed_out += len(packed)
            for i in range(0, len(packed), self.tx_n401):
                self._cq.append(packed[i:i + self.tx_n401])

        reset = self._enc.request_reset()
        self.compressed_out += len(reset)
        for i in range(0, len(reset), self.tx_n401):
            self._cq.append(reset[i:i + self.tx_n401])
        self._event("V.42bis RESET queued at transmit stream boundary")
        return True

    def read(self) -> bytes:
        out, self._out = bytes(self._out), bytearray()
        return out

    def pending(self) -> int:
        return (len(self._txq) + len(self._post_break_txq) + sum(len(c) for c in self._cq)
                + sum(len(f[1]) for f in self._unacked) + len(self._retx))

    def _snapshot_release_tx(self) -> dict:
        """Capture unsent DTE bytes before LAPM release discards its queues.

        Raw bytes still in ``_txq`` can be safely placed behind a fresh
        V.42bis dictionary. Compressed chunks and unacknowledged frames cannot
        be replayed safely: the peer may already have consumed them.
        """
        encoded_pending = bool(self._cq or self._unacked or self._retx)
        # Retransmission entries reference frames that remain in _unacked;
        # count each N(S) once so recovery metrics do not inflate possible
        # loss when a frame is queued for retransmission.
        pending_frames = {frame[0]: frame[1] for frame in self._unacked}
        for ns, payload in self._retx:
            pending_frames.setdefault(ns, payload)
        encoded_octets = (sum(len(chunk) for chunk in self._cq)
                          + sum(len(payload) for payload in pending_frames.values()))
        return {
            "raw": bytes(self._txq) + bytes(self._post_break_txq),
            "encoded_pending": encoded_pending,
            "encoded_octets": encoded_octets,
        }

    def take_release_tx_snapshot(self) -> dict | None:
        """Take the DTE queue snapshot captured by the orderly release."""
        snapshot = self._release_tx_snapshot
        self._release_tx_snapshot = None
        return snapshot

    def tx_bit(self) -> int:
        self.bits_sent += 1
        if not self._tx:
            if self._break_rx_reset_waiting():
                return 1
            if self.state == "release_pending":
                self.state = "released"
                self._event("LAPM release response transmitted")
                return 1
            self._schedule()
        if self._tx:
            bit = self._tx.popleft()
            self._finish_pending_ua_if_sent()
            if not self._tx and self._break_tx_reset_after_send:
                self._reset_data_state_for_break()
                self._break_tx_reset_after_send = False
            if self.state == "release_pending" and not self._tx:
                self.state = "released"
                self._event("LAPM release response transmitted")
            return bit
        return 1

    def _break_rx_reset_waiting(self) -> bool:
        return bool(self._break_rx_confirmations
                    and self._break_rx_confirmations[0]["message"]["destructive"]
                    and self._break_rx_confirmations[0]["message"]["expedited"])

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
            if self._break_tx_reset_after_send:
                self._reset_data_state_for_break()
                self._break_tx_reset_after_send = False
        else:
            bits=[self._tx.popleft() for _ in range(count)]
        self.bits_sent+=count
        self._finish_pending_ua_if_sent()
        return bits

    def _finish_pending_ua_if_sent(self) -> None:
        if (self._ua_pending_until is not None
                and self.bits_sent >= self._ua_pending_until):
            self._ua_pending_until = None

    def rx_bit(self, bit: int) -> None:
        if self.state in ("down", "unsupported", "release_pending", "released"):
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

    def _received_frame(self, frame: bytes | None, ok: bool) -> None:
        # Count established opportunities including control frames, before a
        # frame's side effects can change link state.
        if self.state == 'lapm':
            self.stats['hdlc_good' if ok else 'hdlc_bad'] += 1
        if frame is None:
            # The invalid bit sequence is not an FCS error. §8.5.4 requires
            # discarding it without taking action; in particular do not
            # trigger the optional REJ response reserved for a bad FCS.
            return
        addressed = ok and len(frame) >= 2 and frame[0] in (self.cmd_addr, self.rsp_addr)
        if ok:
            self._on_frame(frame)
        elif self.state == "lapm":
            # V.42 §8.5.4 permits (but does not require) a REJ response to a
            # completed frame whose FCS fails. Other invalid frames do not
            # enter this branch.
            self.stats["fcs_errors"] += 1
            self._need_rej = True
        if addressed and self.state == "lapm" and self._t401_at is None and not self._poll_out:
            self._start_t403()

    # -- link state -------------------------------------------------------------------------------------------
    def _reset_link(self) -> None:
        self.vs = self.va = self.vr = 0
        self._unacked: list[list] = []                              # [ns, payload]
        self._retx: deque = deque()
        self._cq: deque = deque()                                   # compressed octets waiting for a window slot
        self._peer_busy = False
        self._peer_xid_pause = False
        self._peer_xid_pending = False
        self._local_busy = False
        self._ack_pending = False
        self._rej_sent = False
        self._need_rej = False
        self._clear_break_state()
        self._t401_at: int | None = None
        self._t403_at: int | None = None
        self._tries = 0
        self._poll_out = False
        self._xid_poll_out = False
        self._xid_done = False
        self._xid_sent = 0
        self._xid_fallback_suppressed = False
        self._renegotiation: dict | None = None
        self._release_tx_snapshot: dict | None = None
        self._test_negotiated = False
        self._test_indications: deque[bytes] = deque()
        self._ua_pending_until: int | None = None
        self._xid_pause_tx = False
        self._sabme_sent = False
        self._deferred_xid_response = None
        self._connected_at: int | None = None
        self._enc: Encoder | None = None
        self._dec: Decoder | None = None

    def _event(self, text: str) -> None:
        self.events.append(text)

    # -- detection phase (§7.2.1) ---------------------------------------------------------------------------
    def _detect_rx(self, bit: int) -> None:
        self._detect_bit_count += 1
        a = self._async
        a.append(bit)
        while len(a) >= 10:
            if a[0] == 0 and a[9] == 1:
                ch = sum(a[1 + j] << j for j in range(8))
                start = self._detect_bit_count - len(a)
                gap = (None if self._last_detect_char_end is None else
                       start - self._last_detect_char_end)
                self._last_detect_char_end = start + 10
                del a[:10]
                self._chars.append(ch)
                self._char_gaps.append(gap)
                if len(self._chars) > 64:
                    del self._chars[:-64]
                    del self._char_gaps[:-64]
                self._on_char(ch, gap)
            else:
                del a[0]
            if self._phase != "detect":
                return

    def _on_char(self, ch: int, gap: int | None = None) -> None:
        if self.calling:
            # Do not mistake pre-response noise/text for an ADP, and do not
            # resurrect detection after T400 expired. The answerer needs four
            # alternating ODP characters before it can legally respond.
            if self.bits_sent < ADP_ARM_BITS or self.bits_sent > self._t400_at:
                return
            # V.42 §7.2.1.2 requires characters from at least two adjacent
            # ADPs before recognizing the answerer pattern. One E/C pair can
            # be a transient or corrupted detection and is not enough.
            tail = self._chars[-4:]
            # V.42's ADP spacing is 8..16 mark bits. Keep the documented
            # no-extra-mark peer form (gap 0) for real-world compatibility;
            # reject other malformed short or overlong gaps.
            valid_gaps = (len(self._char_gaps[-3:]) == 3 and
                          all(n is not None and
                              (n == 0 or DETECTION_MIN_MARK_BITS <= n <= DETECTION_MAX_MARK_BITS)
                              for n in self._char_gaps[-3:]))
            if valid_gaps and tail == [CHAR_E, CHAR_NULL, CHAR_E, CHAR_NULL]:
                self._event("ADP: no error control desired")
                self.state = "unsupported"
            elif valid_gaps and tail == [CHAR_E, CHAR_C, CHAR_E, CHAR_C]:
                self._event("ADP received")
                self.adp_seen = True
                self._phase = "protocol"
                self._begin_protocol()
        else:
            if ch in (DC1_EVEN, DC1_ODD):
                if (self._dc1 and (gap is None or
                                   not DETECTION_MIN_MARK_BITS <= gap <= DETECTION_MAX_MARK_BITS)):
                    self._dc1.clear()
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
        if self.state in ("down", "unsupported", "release_pending", "released"):
            return
        # A caller may begin its flags before the answerer has completed the
        # mandatory ten ADPs. Protocol-phase detection must not truncate that
        # minimum; finish the remaining patterns before queuing XID/SABME.
        if (not self.calling and self._answer_adp
                and self._adp_count < ADP_MIN_REPETITIONS):
            self._tx.extend(ADP_BLOCK)
            self._adp_count += 1
            return
        if not self.calling and self._deferred_xid_response is not None:
            info, self._deferred_xid_response = self._deferred_xid_response, None
            self._u(XID, False, 0, info)
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
            if self.state == "lapm" and self._t401_at is None and not self._poll_out:
                self._start_t403()
        if (self.state == "lapm" and self._t401_at is None and not self._poll_out
                and self._t403_at is not None and now >= self._t403_at):
            self._tries = 0
            self._poll_out = True
            self._t403_at = None
            self.stats["t403_expiries"] += 1
            self._event("T403 expired; connection-verification poll sent")
            self._send_status_poll()
            self._start_t401()
        if (self.state == "lapm" and self._break_tx_pending is not None
                and self._break_t401_at is not None and now >= self._break_t401_at):
            if self._break_tx_tries >= N400:
                self._finish_break_tx(False, "BRK retries exhausted")
            else:
                self._break_tx_tries += 1
                self._send_break_frame()
                self._event(f"BRK T401 retry {self._break_tx_tries}/{N400}")
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
        code = ctrl & 0xEF
        counter = {DISC: "disc_tx", UA: "ua_tx", DM: "dm_tx"}.get(code)
        if counter is not None:
            self.stats[counter] += 1
        addr = self.cmd_addr if command else self.rsp_addr
        self._send(bytes((addr, ctrl | (0x10 if pf else 0))) + info)

    def _s(self, ctrl: int, command: bool, pf: int) -> None:
        addr = self.cmd_addr if command else self.rsp_addr
        self._send(bytes((addr, ctrl, (self.vr << 1) | (1 if pf else 0))))

    def _status_control(self) -> int:
        """Choose RR/RNR/REJ for a peer poll or timer-recovery poll (§8.4.6)."""
        if self._local_busy:
            return RNR
        if self._need_rej and not self._rej_sent:
            self._need_rej = False
            self._rej_sent = True
            self._ack_pending = False
            self.stats["rej_tx"] += 1
            return REJ
        # RR is correct both in the ordinary ready state and when an
        # already-sent REJ is still awaiting its requested I frame.
        return RR

    def _send_status_poll(self) -> None:
        """Send the §8.4.6/8.4.8 supervisory command matching receive state."""
        self._s(self._status_control(), True, 1)

    def _send_status_final(self) -> None:
        """Answer a peer's P=1 poll using the current receiver state."""
        self._s(self._status_control(), False, 1)
        self._ack_pending = False

    def _start_t401(self) -> None:
        self._t403_at = None
        self._t401_at = self.bits_sent + self._t401_bits

    def _start_t403(self) -> None:
        if (self._t403_bits is not None and self.state == "lapm"
                and self._t401_at is None and not self._poll_out):
            self._t403_at = self.bits_sent + self._t403_bits
        else:
            self._t403_at = None

    def _send_xid_command(self) -> None:
        self._xid_sent += 1
        self.stats["xid_tx"] += 1
        self._u(XID, True, 0, build_xid(
            self.tx_n401, self.tx_k, self._offer,
            rx_n401=self.rx_n401, rx_k=self.rx_k,
            include_test=True,
        ))
        self._start_t401()

    def _xid_fallback_uses_only_defaults(self) -> bool:
        """Whether unanswered XID can safely fall back to the standard profile.

        V.42 Appendix III.3 warns that a peer may process XID even if every
        response is lost. Continuing with defaults is only compatible when
        our request itself requires those defaults and does not offer V.42bis;
        otherwise the peer may have committed different frame/compression
        state before we send SABME.
        """
        return (
            self._offer is None
            and self.tx_n401 == DEFAULT_N401
            and self.rx_n401 == DEFAULT_N401
            and self.tx_k == DEFAULT_K
            and self.rx_k == DEFAULT_K
        )

    def _t401_expired(self) -> None:
        self._tries += 1
        self.stats["t401_expiries"] += 1
        if self.state == "releasing":
            if self._tries > N400:
                self._event("DISC release retries exhausted; entering disconnected state")
                self.state = "released"
                self._t401_at = None
            else:
                self._u(DISC, True, 1)
                self._start_t401()
            return
        if self.state == "lapm" and self._renegotiation is not None:
            if self._tries > N400:
                self._event("connected XID renegotiation failed after N400 retries")
                self._renegotiation = None
                self._xid_poll_out = False
                self._xid_pause_tx = False
                self._local_busy = False
                self._t401_at = None
                self._tries = 0
                self._s(RR, True, 0)
            else:
                self._event(f"connected XID renegotiation retry {self._tries}/{N400}")
                self._send_renegotiation_xid()
            return
        if self._tries > N400:
            self._event("N400 retransmissions exhausted")
            self.state = "down"
            self._xid_poll_out = False
            return
        if self.state == "idle":
            if self.calling:
                if self._sabme_sent:
                    self._u(SABME, True, 1)
                elif (self._xid_sent >= XID_ATTEMPTS and self.xid_fallback
                      and self._xid_fallback_uses_only_defaults()):
                    # Parameter negotiation is optional (V.42 7.2.2); a peer that never answers the XID may not implement it
                    # and wait for the SABME, so ask for the connection with the default parameters instead of repeating
                    # the XID for N400 * T401. Disabling xid_fallback restores
                    # the XID-only behavior.
                    self._event(f"XID unanswered after {self._xid_sent} commands: sending SABME with default parameters")
                    # The local offer is not an agreement. If the peer ignores
                    # XID, both sides use V.42 defaults after SABME; retaining
                    # a locally configured smaller N401 here makes us reject
                    # a conforming peer's default-size I-frame.
                    self.tx_n401 = self.rx_n401 = DEFAULT_N401
                    self.tx_k = self.rx_k = DEFAULT_K
                    self._t401_bits = int(max(
                        1.5, 2.2 * (self.tx_n401 + 8) * 8 / self.rate + 1.0
                    ) * self.rate)
                    self._event(
                        f"unanswered XID: restored default N401={self.tx_n401}/{self.rx_n401}, "
                        f"K={self.tx_k}/{self.rx_k}"
                    )
                    self._agreed = None
                    self._tries = 0
                    self._sabme_sent = True
                    self._xid_fallback = True
                    self._u(SABME, True, 1)
                else:
                    if (self._xid_sent >= XID_ATTEMPTS and self.xid_fallback
                            and not self._xid_fallback_uses_only_defaults()
                            and not self._xid_fallback_suppressed):
                        self._event(
                            "XID default fallback suppressed: non-default LAPM parameters "
                            "or V.42bis offer could leave peer state unsynchronized"
                        )
                        self._xid_fallback_suppressed = True
                    self._send_xid_command()
                self._start_t401()
            else:
                self._t401_at = None
            return
        # connected: poll the peer (timer recovery) and resend what is outstanding when it answers
        self._send_status_poll()
        self._poll_out = True
        self._start_t401()

    # -- data transfer (§8.4) ---------------------------------------------------------------------------------
    def _window_open(self) -> bool:
        return (len(self._unacked) < self.k and not self._peer_busy
                and not self._poll_out and not self._xid_pause_tx)

    def _schedule_data(self) -> None:
        if self._need_rej and not self._rej_sent and not self._local_busy:
            self._s(REJ, False, 0)
            self.stats["rej_tx"] += 1
            self._rej_sent = True
            self._need_rej = False
            self._ack_pending = False
            return
        # Advertise receiver-busy before retransmitting or sending I frames.
        # If a sequence error is pending while busy, RNR goes first and the
        # deferred REJ remains pending until busy clears.
        if self._ack_pending and self._local_busy:
            self._s(RNR, False, 0)
            self._ack_pending = False
            return
        if self._retx and not self._peer_busy and not self._xid_pause_tx:
            ns, payload = self._retx.popleft()
            self.stats["i_resent"] += 1
            # V(S) is the next I-frame sequence number to transmit. REJ and
            # timer-recovery responses rewind it to N(R); each retransmitted
            # frame then advances it just like the corresponding sequential
            # transmission (V.42 §§8.2.3.2.2, 8.4.4 and 8.4.8).
            self.vs = (self.vs + 1) & 0x7F
            self._send_i(ns, payload)
            return
        if self._ack_pending:
            self._s(RR, False, 0)
            self._ack_pending = False
            return
        if self._maybe_send_ordered_break():
            return
        # A BRK is confirmed through the L-SIGNAL service. Keep subsequent
        # DTE bytes behind it until confirmation so they cannot overtake it.
        if self._break_tx_pending is not None:
            self._tx.extend(FLAG_BITS)
            return
        window_open = self._window_open()
        if window_open and self._enc is not None and not self._cq and self._txq:
            # compress a slice small enough that the result normally fits one frame; the byte stream may be cut anywhere
            raw = bytes(self._txq[:max(1, self.n401 * 3 // 4)])
            del self._txq[:len(raw)]
            packed = self._enc.compress(raw) + self._enc.flush()
            self.compressed_out += len(packed)
            for i in range(0, len(packed), self.n401):
                self._cq.append(packed[i:i + self.n401])
        if window_open:
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
        if (self._ua_pending_until is not None
                and self.bits_sent < self._ua_pending_until):
            # V.42 §8.2.4.10: a received mode-setting command is not processed
            # until our UA response has been transmitted. I/S traffic also
            # cannot be accepted into the newly selected mode before UA is on
            # the line. A UA/DM response to our own outstanding command still
            # has to be processed, as in simultaneous SABME establishment.
            normalized = ctrl & 0xEF
            if ((ctrl & 1) == 0 or (ctrl & 3) == 1
                    or (addr == self.rsp_addr and normalized in (SABME, DISC))):
                self._event("frame ignored until pending UA response is transmitted")
                return
        elif self._ua_pending_until is not None:
            self._ua_pending_until = None
        is_command = addr == self.rsp_addr                           # the peer's commands carry my response address
        if ctrl & 0x01 == 0 and not is_command:
            # V.42 Table 8 defines I frames as commands only (their control
            # field carries P, not a response F). A response-address I frame
            # is therefore an undefined command/response control field under
            # §8.2.4.1 and a frame-rejection condition under §8.5.5. While
            # disconnected, discard it without allowing it to confirm SABME.
            if self.state == "lapm":
                self._terminate_for_frame_reject("I frame received in response role")
            else:
                self._event("I frame received in response role while disconnected; discarded")
            return
        if (self.state == "idle" and self.calling and self._sabme_sent
                and (ctrl & 0x01 == 0 or ctrl & 0x03 == 0x01)):
            # V.42 §8.3.2 permits the originator to infer that the peer
            # accepted SABME when its UA was lost, if an I or supervisory
            # frame arrives first. Enter connected state, then process that
            # frame instead of silently discarding the first data/ack.
            self._event("I/S frame received after SABME; assuming UA was lost")
            self._connected()
        if ctrl & 0x01 == 0:
            self._on_i(f)
        elif ctrl & 0x03 == 0x01:
            self._on_s(f, is_command)
        else:
            self._on_u(f, is_command)

    def _on_u(self, f: bytes, is_command: bool) -> None:
        ctrl, pf = f[1] & 0xEF, (f[1] >> 4) & 1
        # V.42 Table 8 and §8.2.4.13 define XID with P/F=0. Treat the
        # otherwise-recognized XID control with P/F=1 as an invalid control
        # field; do this before XID pause bookkeeping or parameter parsing can
        # mutate negotiation state.
        if ctrl == XID and pf:
            if self.state == "lapm":
                self._terminate_for_frame_reject("XID P/F bit set")
            else:
                self._event("XID P/F bit set while disconnected; discarded")
            return
        peer_xid_pause = False
        if self.state == "lapm":
            if ctrl == XID and is_command:
                # Once an L-SETPARM indication has started, retransmitted XID
                # commands are valid until the peer clears the busy condition
                # with RR. This is required to recover a lost XID response.
                peer_xid_pause = self._peer_xid_pause or self._peer_xid_pending
            self._peer_xid_pause = False
        counter = {DISC: "disc_rx", UA: "ua_rx", DM: "dm_rx"}.get(ctrl)
        if counter is not None:
            self.stats[counter] += 1
        # Table 8/V.42 defines which U controls are commands, responses, or
        # either. A recognized control in the wrong C/R role is still an
        # undefined command/response control field for §8.5.5 purposes.
        command_only = (SABME, DISC, TEST)
        response_only = (UA, DM, FRMR)
        if ((ctrl in command_only and not is_command)
                or (ctrl in response_only and is_command)):
            if self.state == "lapm":
                self._terminate_for_frame_reject(
                    f"U control 0x{ctrl:02x} received in invalid {'response' if not is_command else 'command'} role"
                )
            return
        if ctrl in (SABME, DISC, UA, DM) and len(f) != 2:
            if self.state == "lapm":
                self._terminate_for_frame_reject(f"incorrect U-frame length for 0x{ctrl:02x}")
            return
        if ctrl == FRMR:
            if len(f) != 7:
                if self.state == "lapm":
                    self._terminate_for_frame_reject("incorrect FRMR length")
            elif not is_command and self.state == "lapm":
                self._terminate_for_frame_reject("FRMR received")
            elif is_command and self.state == "lapm":
                self._terminate_for_frame_reject("FRMR received as command")
            return
        # N401 is the active limit for user/control information, but an XID
        # can negotiate a smaller value than the current default. Until that
        # exchange completes, validate XID against the V.42 default rather
        # than the local proposed value; otherwise a legal XID response (for
        # example, a 43-octet V.42bis parameter set with a 32-octet offer) is
        # silently discarded and SABME can establish mismatched frame sizes.
        info_limit = (DEFAULT_N401
                      if ctrl == XID and self.state != "lapm"
                      else self.rx_n401)
        if ctrl in (XID, UI, TEST) and len(f) - 2 > info_limit:
            if self.state == "lapm":
                self._terminate_for_frame_reject(
                    f"U-frame information length {len(f) - 2} exceeds N401={info_limit}"
                )
            return
        if ctrl == UI and self.state == "lapm" and self._handle_break_frame(f, is_command, pf):
            return
        if ctrl == XID:
            xid_info = f[2:]
            if xid_info and xid_info[0] != 0x82:
                # V.42 §12.2.2 requires FI=0x82 for its general-purpose
                # negotiation and says unrecognized fields are ignored. Do
                # not misclassify a different XID format as malformed LAPM.
                self._event(f"unrecognized XID format 0x{xid_info[0]:02x} ignored")
                return
            xid = parse_xid(xid_info)
            if xid is None:
                if self.state == "lapm":
                    self._terminate_for_frame_reject("malformed XID")
                return
            xid_error = xid.get(_XID_V42BIS_PROCEDURAL_ERROR)
            if xid_error:
                # V.42bis §5.1 explicitly requires disconnection for an
                # out-of-range P1/P2. Do this before local capability clamping
                # can turn the peer's invalid proposal into a valid selection.
                self._event(f"{xid_error}; disconnecting")
                self._fatal_protocol_error = True
                self.state = "down"
                self._t401_at = None
                self._poll_out = False
                self._xid_poll_out = False
                return
            if is_command:
                self._event(f"XID command received ({len(f) - 2} info octets): {xid}")
                if self.state == "lapm":
                    # V.42 §8.10.1 requires the negotiation initiator to send
                    # RNR(P=1) before XID. This N401/k API leaves active
                    # V.42bis dictionary changes to a separate procedure.
                    if (not peer_xid_pause or not self._peer_busy
                            or self.compression is not None or xid.get("p0", 0) != 0):
                        self._event("connected XID ignored: missing RNR(P=1) pause or unsupported compression renegotiation")
                        return
                    self._peer_xid_pending = True
                self._answer_xid(xid or {})
            elif (xid is not None and self.state == "idle" and self.calling
                  and (not self._default_parameters or self._xid_sent)):
                self._event(f"XID response received ({len(f) - 2} info octets): {xid}")
                self._xid_reply(xid)
            elif (xid is not None and self.state == "lapm"
                  and not is_command and self._renegotiation is not None):
                self._event(f"connected XID response received ({len(f) - 2} info octets): {xid}")
                self._xid_renegotiation_reply(xid)
        elif ctrl == SABME and is_command:
            if self.state == "releasing":
                # §8.9.2: our DISC and the peer's SABME are different
                # set-mode commands. Reject the peer command with DM, but
                # keep our release request active until DM(F=1) arrives.
                self._u(DM, False, pf)
                self._event("SABME/DISC collision; DM response sent")
                return
            if self.state == "lapm":
                # V.42 §8.4.9.1 permits an answering entity to accept a
                # retransmitted initial SABME if its first UA was lost. Outside
                # that initial T401 window, SABME terminates an established
                # connection.
                # Our T401 starts when the frame is queued, a few HDLC bits
                # before its final flag reaches the peer. Include one octet
                # of framing/clock granularity at the boundary.
                retry_window = (not self.calling and self._connected_at is not None
                                and self.bits_sent - self._connected_at <= self._t401_bits + 8)
                if retry_window:
                    self._u(UA, False, pf)
                    # The repeated SABME is still a mode-setting command.
                    # Do not process I/S traffic (including a REJ
                    # retransmission request) until its UA is on the line,
                    # as required by §8.2.4.10.
                    self._ua_pending_until = self.bits_sent + len(self._tx)
                    self._event("repeated establishment SABME acknowledged")
                else:
                    self._event("SABME received on established LAPM link: terminating")
                    self.state = "down"
                    self._t401_at = None
                    self._poll_out = self._xid_poll_out = False
                return
            if self.state == "idle":
                simultaneous = self.calling and self._sabme_sent
                if not simultaneous:
                    self._connected()
                self._u(UA, False, pf)
                self._ua_pending_until = self.bits_sent + len(self._tx)
                if simultaneous:
                    # §8.9.1: both entities sent SABME. We have answered the
                    # peer's command, but enter connected state only after its
                    # UA answers our outstanding command.
                    self._event("simultaneous SABME; awaiting UA response")
        elif ctrl == UA and not is_command:
            if self.state == "releasing" and pf:
                self._t401_at = None
                self.state = "release_pending" if self._tx else "released"
                self._event("DISC acknowledged by UA; LAPM release completing")
            elif self.state == "idle" and self.calling and self._sabme_sent and pf:
                self._connected()
        elif ctrl == DM and not is_command:
            if self.state == "releasing":
                if pf:
                    self._t401_at = None
                    self.state = "release_pending" if self._tx else "released"
                    self._event("DISC acknowledged by DM; LAPM release completing")
                return
            if self.state == "lapm" and pf:
                if self._poll_out:
                    # V.42 §8.4.9: receipt of DM(F=1) during timer recovery
                    # terminates the error-corrected connection.
                    self._event("DM(F=1) during timer recovery: terminating LAPM")
                    self.state = "down"
                    self._t401_at = None
                    self._poll_out = False
                    self._xid_poll_out = False
                else:
                    # V.42 §8.5.7/Table 9: an unsolicited DM(F=1) outside
                    # timer recovery is ignored, with a protocol violation
                    # reported to the control function.
                    self._event("unsolicited DM(F=1) ignored on connected LAPM link")
                return
            if self.state == "idle" and self.calling and self._sabme_sent and pf:
                # §8.9.2: a different set-mode command collided with our
                # outstanding SABME; the peer's final DM completes release.
                self._t401_at = None
                self.state = "release_pending" if self._tx else "released"
                self._event("SABME/DISC collision resolved by DM; disconnected")
                return
            if self.state == "idle":
                # V.42 §8.8/Table 9: DM(F=0) in the disconnected state may
                # trigger establishment only if the control function elects
                # to do so; unsolicited DM(F=1) is ignored. This link has no
                # unsolicited-establishment policy. A solicited DM(F=1) for
                # our outstanding SABME was handled above.
                self._event(
                    "unsolicited DM(F=1) ignored while disconnected" if pf else
                    "unsolicited DM(F=0) ignored while disconnected"
                )
                return
            self._event("DM received")
            self.state = "down"
            self._poll_out = self._xid_poll_out = False
        elif ctrl == DISC and is_command:
            if self.state in ("lapm", "releasing"):
                collision = self.state == "releasing"
                self._release_tx_snapshot = self._snapshot_release_tx()
                # Complete the orderly-release response on the line before
                # the V.42 layer reports disconnection to its control
                # function. Abort any queued partial frame, then resync.
                self._tx.clear()
                self._tx.extend([1] * 8)
                self._u(UA, False, pf)
                self._tx.extend(FLAG_BITS)
                self._txq.clear()
                self._cq.clear()
                self._unacked.clear()
                self._retx.clear()
                self.vs = self.va = self.vr = 0
                if not collision:
                    self._t401_at = None
                self._poll_out = False
                self.state = "releasing" if collision else "release_pending"
                self._event("simultaneous DISC; UA sent, awaiting UA response" if collision
                            else "DISC received; orderly LAPM release")
            elif self.state == "idle":
                # A disconnected LAPM entity answers DISC with DM (§8.8).
                self._u(DM, False, pf)
                if self.calling and self._sabme_sent:
                    # §8.9.2: the pending SABME and incoming DISC differ.
                    # Retain T401 and wait for the peer's DM(F=1).
                    self._event("SABME/DISC collision; DM response sent")
                else:
                    self._event("DISC received while disconnected")
        elif ctrl == UA and self.state == "lapm":
            pass
        elif ctrl == TEST and self.state == "lapm":
            if pf != 0 or not self._test_negotiated:
                self._terminate_for_frame_reject("unnegotiated or invalid TEST received")
            else:
                self._test_indications.append(bytes(f[2:]))
                self._event(f"TEST indication received ({len(f) - 2} information octets)")
        elif ctrl not in (UI, TEST, SABME, UA, DM, DISC) and self.state == "lapm":
            self._terminate_for_frame_reject(f"undefined U control 0x{ctrl:02x}")

    def _answer_xid(self, x: dict) -> None:
        """Responder chooses values within the initiator-request/default interval (§9.2)."""
        # The initiator's value is a lower bound, not a maximum. The
        # responder selects a value from [initiator request, V.42 default].
        # Taking min() here made a local 32-octet setting override a peer's
        # request for 128, contrary to §9.2.3, and then the peer sent 128.
        # The incoming command describes the initiator's direction: its TX
        # request constrains our RX, and its RX request constrains our TX.
        if self.state == "lapm":
            # A connected XID is accepted only after the peer's RNR(P=1).
            # Select its requested receive limits while preserving room for
            # already-outstanding local I frames that the RNR did not ack.
            rx_request = min(DEFAULT_N401, max(1, x.get("tx_n401", DEFAULT_N401)))
            tx_request = min(DEFAULT_N401, max(1, x.get("rx_n401", DEFAULT_N401)))
            self.rx_n401 = rx_request
            outstanding_size = max((len(payload) for _, payload in self._unacked), default=0)
            self.tx_n401 = max(tx_request, outstanding_size)
            self.rx_k = min(DEFAULT_K, max(1, x.get("tx_k", DEFAULT_K)))
            outstanding_window = len(self._unacked)
            self.tx_k = max(min(DEFAULT_K, max(1, x.get("rx_k", DEFAULT_K))),
                            outstanding_window)
        else:
            self.rx_n401 = min(DEFAULT_N401, max(
                self.rx_n401, min(DEFAULT_N401, max(1, x.get("tx_n401", DEFAULT_N401)))
            ))
            self.tx_n401 = min(DEFAULT_N401, max(
                self.tx_n401, min(DEFAULT_N401, max(1, x.get("rx_n401", DEFAULT_N401)))
            ))
            self.rx_k = min(DEFAULT_K, max(
                self.rx_k, min(DEFAULT_K, max(1, x.get("tx_k", DEFAULT_K)))
            ))
            self.tx_k = min(DEFAULT_K, max(
                self.tx_k, min(DEFAULT_K, max(1, x.get("rx_k", DEFAULT_K)))
            ))
        self._t401_bits = int(max(1.5, 2.2 * (self.tx_n401 + 8) * 8 / self.rate + 1.0) * self.rate)
        comp = None
        if "p0" in x and self._offer:
            p0 = x["p0"] & self._offer[0]
            p1 = min(x.get("p1", 512), self._offer[1])
            p2 = min(x.get("p2", 6), self._offer[2])
            comp = (p0, p1, p2)
        self._agreed = comp
        self._event(
            f"XID parameters selected: N401 tx/rx={self.tx_n401}/{self.rx_n401}, "
            f"k tx/rx={self.tx_k}/{self.rx_k}"
        )
        connected = self.state == "lapm"
        if not connected:
            self._test_negotiated = bool(x.get("functions", 0) & 0x00002000)
        response = build_xid(
            self.tx_n401, self.tx_k, comp,
            rx_n401=self.rx_n401, rx_k=self.rx_k,
            include_functions=not connected,
            include_test=self._test_negotiated and not connected,
            include_compression=not connected,
        )
        if self._answer_adp and self._adp_count < ADP_MIN_REPETITIONS:
            self._deferred_xid_response = response
            self._event(
                f"XID response deferred until {ADP_MIN_REPETITIONS} ADPs are transmitted"
            )
        else:
            self._u(XID, False, 0, response)

    def _xid_renegotiation_reply(self, x: dict) -> None:
        """Apply a connected N401/k response and clear the RNR pause."""
        pending = self._renegotiation
        if pending is None:
            return

        def selected(field_name: str, requested: int) -> int:
            value = x.get(field_name, DEFAULT_N401 if "n401" in field_name else DEFAULT_K)
            maximum = DEFAULT_N401 if "n401" in field_name else DEFAULT_K
            if not 1 <= value <= maximum:
                self._event(f"invalid connected XID selection {field_name}={value}; using default {maximum}")
                return maximum
            if value < requested:
                self._event(
                    f"non-conforming connected XID selection {field_name}={value} below request {requested}; "
                    "honoring peer selection to keep frame limits aligned"
                )
            return value

        self.tx_n401 = selected("rx_n401", pending["tx_n401"])
        self.rx_n401 = selected("tx_n401", pending["rx_n401"])
        self.tx_k = selected("rx_k", pending["tx_k"])
        self.rx_k = selected("tx_k", pending["rx_k"])
        self._t401_bits = int(max(
            1.5, 2.2 * (self.tx_n401 + 8) * 8 / self.rate + 1.0
        ) * self.rate)
        self._t401_at = None
        self._tries = 0
        self._renegotiation = None
        self._xid_poll_out = False
        self._xid_pause_tx = False
        self._local_busy = False
        self._s(RR, True, 0)
        self._event(
            f"connected XID parameters accepted: N401 tx/rx={self.tx_n401}/{self.rx_n401}, "
            f"k tx/rx={self.tx_k}/{self.rx_k}; RR sent to clear busy"
        )

    def _xid_reply(self, x: dict) -> None:
        """Originator: the XID response carries the agreed values; then request the connection (SABME)."""
        # The responder reports the selected value for each data direction;
        # that is the value both endpoints must use (V.42 §9.2.3/.4). A
        # conforming responder chooses between our request and the default.
        # If a peer violates the lower bound, honor its usable selection so
        # our framing still matches the peer, but make the violation visible.
        # Values outside this implementation's supported range cannot safely
        # be used, so fall back to the corresponding standard default.
        # Response directions are from the responder's perspective. Its RX
        # selection (PI 6/8) applies to our TX direction, and vice versa.
        def selected(key: str, requested: int, default: int) -> int:
            value = x.get(key, default)
            if not 1 <= value <= default:
                self._event(
                    f"invalid XID selection {key}={value}; using supported default {default}"
                )
                return default
            if value < requested:
                self._event(
                    f"non-conforming XID selection {key}={value} below request {requested}; "
                    "honoring peer selection to keep frame limits aligned"
                )
            return value

        self.tx_n401 = selected("rx_n401", self.tx_n401, DEFAULT_N401)
        self.rx_n401 = selected("tx_n401", self.rx_n401, DEFAULT_N401)
        self.tx_k = selected("rx_k", self.tx_k, DEFAULT_K)
        self.rx_k = selected("tx_k", self.rx_k, DEFAULT_K)
        self._t401_bits = int(max(1.5, 2.2 * (self.tx_n401 + 8) * 8 / self.rate + 1.0) * self.rate)
        self._event(
            f"XID parameters accepted: N401 tx/rx={self.tx_n401}/{self.rx_n401}, "
            f"k tx/rx={self.tx_k}/{self.rx_k}"
        )
        self._test_negotiated = bool(x.get("functions", 0) & 0x00002000)
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
        test_negotiated = self._test_negotiated
        pending_ua_until = self._ua_pending_until
        self._reset_link()
        # In simultaneous SABME establishment, the peer's UA may reach us
        # before our queued UA response has left. Preserve that transmit
        # boundary while completing our own establishment.
        self._ua_pending_until = pending_ua_until
        self._test_negotiated = test_negotiated
        self._t401_at = None
        self.state = "lapm"
        self._connected_at = self.bits_sent
        self._event("connected")
        self._start_t403()
        a = self._agreed
        if a and a[0] in (1, 2, 3) and a[1] >= 512 and 6 <= a[2] <= 250:
            self.compression = a
            # P0: bit 0 = initiator -> responder, bit 1 = responder -> initiator
            tx_dir = 1 if self.calling else 2
            rx_dir = 2 if self.calling else 1
            self._enc = Encoder(a[1], a[2]) if a[0] & tx_dir else None
            self._dec = Decoder(a[1], a[2]) if a[0] & rx_dir else None

    def _ack(self, nr: int, *, received_i: bool = False) -> None:
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
                self.va = nr
                # During timer recovery, an advancing N(R) does not answer
                # the outstanding poll. Preserve its deadline/retry count
                # until a supervisory response with F=1 arrives (§8.4.8).
                # Otherwise a delayed I/RR acknowledgement strands the link
                # with _poll_out=True and no timer to retry the lost final.
                if self._poll_out:
                    return
                if received_i and self._peer_busy:
                    # V.42 §8.4.3.2 Note 2: an I-frame acknowledgement
                    # cannot stop T401 while the peer-busy condition remains
                    # active. Keep the enquiry running even if this ACK
                    # removes the final outstanding I frame.
                    if self._t401_at is None:
                        self._start_t401()
                    return
                self._tries = 0
                self._t401_at = None
                if self._unacked:
                    self._start_t401()

    def _valid_nr(self, nr: int) -> bool:
        """Return whether N(R) lies in the inclusive V(A)..V(S) range (§8.5.2)."""
        outstanding = (self.vs - self.va) & 0x7F
        acknowledged = (nr - self.va) & 0x7F
        return acknowledged <= outstanding

    def _reject_invalid_nr(self, nr: int) -> bool:
        """Terminate a connected LAPM link on an invalid peer acknowledgement (§8.5.2/8.4.9)."""
        if self._valid_nr(nr):
            return False
        self.stats["nr_errors"] += 1
        self._event(
            f"invalid N(R) {nr} outside V(A)={self.va}..V(S)={self.vs}: terminating LAPM"
        )
        self.state = "down"
        self._t401_at = None
        self._poll_out = False
        self._xid_poll_out = False
        return True

    def _terminate_for_frame_reject(self, reason: str) -> None:
        """Leave the connected state on a V.42 frame-rejection condition (§8.5.5)."""
        self.stats["frame_reject_errors"] += 1
        self._event(f"{reason}: terminating LAPM")
        self.state = "down"
        self._t401_at = None
        self._poll_out = False
        self._xid_poll_out = False

    def _on_s(self, f: bytes, is_command: bool) -> None:
        if self.state != "lapm" or len(f) < 3:
            return
        ctrl, nr, pf = f[1], f[2] >> 1, f[2] & 1
        if ctrl == SREJ:
            self._terminate_for_frame_reject("unnegotiated SREJ received")
            return
        if ctrl not in (RR, RNR, REJ):
            self._terminate_for_frame_reject(f"undefined S control 0x{ctrl:02x}")
            return
        if len(f) != 3:
            self._terminate_for_frame_reject(f"incorrect S-frame length for 0x{ctrl:02x}")
            return
        # An F=1 supervisory response is solicited only when it answers our
        # outstanding P=1 poll (§8.5.7/Table 9). Validate the control field
        # and frame length first: malformed or unsupported S frames are
        # frame-rejection conditions even when their F bit is unpaired.
        # A well-formed unpaired final must not acknowledge data or change
        # peer-busy state. F=0 status responses remain meaningful, as does
        # REJ(F=0), which §8.5.1 uses for sequential retransmission.
        if not is_command and pf and not (self._poll_out or self._xid_poll_out):
            self._event("unsolicited supervisory F=1 response ignored; protocol violation")
            return
        if ctrl in (RR, RNR, REJ) and self._reject_invalid_nr(nr):
            return
        if is_command:
            self._peer_xid_pause = ctrl == RNR and pf == 1
            if ctrl == RR or (ctrl == RNR and pf == 1):
                self._peer_xid_pending = False
        if ctrl == RR or ctrl == RNR:
            self._peer_busy = ctrl == RNR
            self._ack(nr)
            if not is_command and pf and self._poll_out:
                self._poll_out = False
                self._tries = 0
                self._t401_at = None
                # A solicited RR/RNR final closes timer recovery and rewinds
                # the send state to the peer's next expected I-frame number.
                # Remaining unacknowledged frames are retransmitted from
                # there, advancing V(S) as they are emitted.
                self.vs = nr
                self._retx = deque((u[0], u[1]) for u in self._unacked)      # the poll answered: go back to V(A)
                # V.42 §8.4.8 requires T401 to restart after an RNR(F=1)
                # response so timer recovery continues to enquire until the
                # peer clears its busy condition. The response may acknowledge
                # the final outstanding I frame, leaving no _unacked entry to
                # otherwise keep the timer alive.
                if self._unacked or self._peer_busy:
                    self._start_t401()
            elif not is_command and pf and self._xid_poll_out:
                # The connected XID procedure begins with RNR(P=1), which
                # creates its own poll/final exchange (§8.10.1). Its final
                # reports peer busy state, but must not close/restart the
                # separate T401 exchange waiting for the XID response.
                self._xid_poll_out = False
            if ctrl == RNR and self._t401_at is None:
                # V.42 §8.4.6 keeps enquiry active while the peer is busy.
                # This covers an RNR(F/P=0) that acknowledges the last
                # outstanding frame; _ack() may otherwise stop T401 with no
                # remaining I frame to restart it.
                self._start_t401()
        elif ctrl == REJ:
            self.stats["rej_rx"] += 1
            self._peer_busy = False
            self._ack(nr)
            # A REJ received during timer recovery updates V(A), but only
            # a response with F=1 answers our outstanding poll (§8.4.4).
            # A peer command with P=1 must still be answered below without
            # cancelling recovery in this direction.
            if self._poll_out:
                if not is_command and pf:
                    # A solicited REJ final closes timer recovery. Outside
                    # that response, a REJ command is handled by §8.4.4's
                    # separate branch and does not rewind V(S) here.
                    self.vs = nr
                    self._poll_out = False
                    self._tries = 0
                    self._retx = deque((u[0], u[1]) for u in self._unacked)
                    self._t401_at = None
                    if self._unacked:
                        self._start_t401()
            elif not is_command and pf and self._xid_poll_out:
                # A solicited REJ(F=1) completes the XID pause poll. The XID
                # transaction retains ownership of T401 and sequence state.
                self._xid_poll_out = False
            else:
                # Outside timer recovery, REJ resets both V(A) and V(S).
                self.vs = nr
                self._tries = 0
                self._retx = deque((u[0], u[1]) for u in self._unacked)
                self._t401_at = None
                if self._unacked:
                    self._start_t401()
        if is_command and pf:
            self._send_status_final()

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
        # The same P/F bit is P on a command and F on a response. Only a
        # command I frame's P=1 solicits a supervisory reply.
        is_command = f[0] == self.rsp_addr
        self._peer_xid_pause = False
        ns, nr, pf = f[1] >> 1, f[2] >> 1, f[2] & 1
        if len(f) - 3 > self.rx_n401:
            self._terminate_for_frame_reject(
                f"I-frame information length {len(f) - 3} exceeds N401={self.rx_n401}"
            )
            return
        if self._reject_invalid_nr(nr):
            return
        self._ack(nr, received_i=True)
        if self._local_busy:
            # V.42 §8.4.7: while our receiver is busy, discard every I-frame
            # information field after processing N(R). A polled I command
            # still requires an immediate RNR response with F=1.
            if is_command and pf:
                self._s(RNR, False, 1)
                self._ack_pending = False
            return
        if ns == self.vr:
            self.vr = (self.vr + 1) & 0x7F
            self._rej_sent = False
            self._need_rej = False
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
                        self._compression_error = str(exc)
                        self._compression_error_octets = len(f[3:])
                        # Stop accepting peer payload immediately. ``rx_bits``
                        # may finish several complete LAPM frames before the
                        # control-function callback runs; setting busy here
                        # prevents a later frame in that same audio batch from
                        # being decoded with an already-invalid dictionary.
                        self.set_local_busy(True)
                        if pf:
                            self._s(RNR, False, 1)
                            self._ack_pending = False
                        return
                    self._dec = None                                    # assumed compression was wrong: deliver raw from here
            self.stats["bytes_in"] += len(payload)
            self._out += payload
            if is_command and pf:
                self._s(RNR if self._local_busy else RR, False, 1)
                self._ack_pending = False
        else:
            if not self._rej_sent:
                if is_command and pf:
                    # V.42 §§8.4.2/8.5.1: a sequence-gap I command with
                    # P=1 still solicits an immediate supervisory response.
                    # Since this is the first report of the sequence error,
                    # that response is REJ(F=1), not a deferred REJ(F=0).
                    self._s(REJ, False, 1)
                    self.stats["rej_tx"] += 1
                    self._rej_sent = True
                    self._need_rej = False
                    self._ack_pending = False
                else:
                    self._need_rej = True
            elif is_command and pf:
                self._s(RNR if self._local_busy else RR, False, 1)
