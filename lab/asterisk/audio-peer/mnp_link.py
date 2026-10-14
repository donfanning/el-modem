"""ITU-T V.42 Annex A "alternative procedure": the MNP-class (2, 3 and 4) reliable link, over a raw bit stream.

Written from the Annex A text of V.42 (1993/1996 editions; the 2002 edition dropped it), nothing else. The
three levels are the three things Annex A negotiates in its link request (LR):

* framing mode 2: start-stop, octet-oriented frames (SYN DLE STX ... DLE ETX CRC16)          (MNP 2)
* framing mode 3: bit-oriented HDLC-style frames (flags, bit stuffing, CRC-CCITT)               (MNP 3)
* "data phase optimization": 256-octet information fields and fixed-field LT/LA headers          (MNP 4)

The class has the same surface as spandsp_v42.V42Link so v42_layer.V42Layer can run either: ``tx_bit()`` /
``rx_bit(bit)`` once per line bit, ``write()`` / ``read()`` for user data, ``state`` ("idle" while the link is
being established, "lapm" in the data phase, "down" if it failed or the peer disconnected). There is no
detection phase here: the originator sends its LR as soon as the line is up, the answerer just listens.
Lost frames are recovered by go-back-N retransmission driven by LA frames and timer T401; flow control uses the
credit in LA frames. Not implemented: sending break (LN frames are acknowledged and ignored) and timer T403.
"""

from __future__ import annotations

from collections import deque
import math

SYN, DLE, STX, ETX = 0x16, 0x10, 0x02, 0x03
LR, LD, LT, LA, LN, LNA = 1, 2, 4, 5, 6, 7
LR_CONSTANT2 = bytes((1, 6, 1, 0, 0, 0, 0, 255))        # variable parameter 1 of every LR (A.6.4.1.4)
N400 = 12                                              # attempts per frame (A.7.5.5)
OPT_N401_256, OPT_FIXED_FIELDS = 1, 2                  # data phase optimization bits (A.6.4.1.8)
FLAG_BITS = (0, 1, 1, 1, 1, 1, 1, 0)                   # 0x7E, least significant bit first


# AT-level names for what to offer: MNP 2 = octet framing, 3 = bit framing, 4 = bit framing + optimization
MNP_LEVELS = {
    2: dict(framing=2, optimize=0),
    3: dict(framing=3, optimize=0),
    4: dict(framing=3, optimize=OPT_N401_256 | OPT_FIXED_FIELDS),
}


def crc16_v42a(data: bytes) -> int:
    """x^16 + x^15 + x^2 + 1 (A.3.6), reflected, initial value 0."""
    crc = 0
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = (crc >> 1) ^ 0xA001 if crc & 1 else crc >> 1
    return crc


def crc16_hdlc(data: bytes) -> int:
    """CRC-CCITT of 8.1.1.6.1 (X.25 FCS): reflected 0x1021, initial 0xFFFF, complemented."""
    crc = 0xFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = (crc >> 1) ^ 0x8408 if crc & 1 else crc >> 1
    return crc ^ 0xFFFF


def _octet_bits(value: int) -> list[int]:
    return [0] + [(value >> i) & 1 for i in range(8)] + [1]        # start, 8 data LSB first, stop


class MnpLink:
    def __init__(self, calling_party: bool, bit_rate: int = 9600, framing: int = 3, k: int = 8, n401: int = 64,
                 optimize: int = OPT_N401_256 | OPT_FIXED_FIELDS, round_trip: float = 1.0):
        self.calling = calling_party
        self.rate = max(bit_rate, 300)
        self.state = "idle"
        self.adp_seen = False                  # set once the peer's LR has been seen
        self.events: list[str] = []
        self.compression = None                # no compression with this link
        self.compressed_out = 0
        self._want = (framing, k, n401, optimize)
        self._round_trip = round_trip
        self.framing = 2                       # mode in force: 2 until the data phase starts
        self.k, self.n401, self.optimize = k, n401, 0
        self._tx: deque[int] = deque()
        self._bits = 0
        self._rx_out = bytearray()
        self._txbuf = bytearray()
        # establishment
        self._phase = "establish"
        self._lr_due = 0                       # bit count at which the originator (re)sends its LR
        self._lr_attempts = 0
        self._reply_lr: bytes | None = None    # negotiated LR the responder still has to send
        self._last_reply: bytes | None = None  # ... and the one it last sent, for resending
        self._reply_tries = 0
        self._reply_seen = False               # originator: the responder's LR has arrived
        self._confirm_la = False               # originator: send the confirming LA
        self._est_t401 = int(2.0 * self.rate)
        self._resp_t401: int | None = None
        # data phase
        self.vs = self.vr = self.vsa = self.vra = 1
        self._out: list[list] = []             # unacknowledged LT frames: [ns, data, attempts]
        self._retx: deque = deque()
        self._s_k = k
        self._unacked_rx = 0
        self._force_ack = False
        self._dup_seen = False
        self._pending_lna: int | None = None
        self._t401: int | None = None
        self._t402: int | None = None
        self._t404 = 0
        self._t401_bits = 0
        # receive side
        self._ob = _OctetDeframer()
        self._hb = _BitDeframer()
        self._async_buf: list[int] = []

    # -- control ------------------------------------------------------------------------------------
    def set_bit_rate(self, bit_rate: int) -> None:
        """Retain frames/credit and timer seconds when the physical rate changes."""
        rate = max(int(bit_rate), 300)
        if rate == self.rate:
            return
        ratio = rate / self.rate
        now = self._bits
        for name in ('_lr_due', '_resp_t401', '_t401', '_t402', '_t404'):
            deadline = getattr(self, name)
            if deadline is not None:
                setattr(self, name, now + math.ceil(max(0, deadline-now) * ratio))
        self.rate = rate
        self._est_t401 = int(2.0 * rate)
        self._compute_timers()

    def close(self) -> None:
        self._tx.clear()

    def write(self, data: bytes) -> None:
        self._txbuf += data

    def read(self) -> bytes:
        out, self._rx_out = bytes(self._rx_out), bytearray()
        return out

    def pending(self) -> int:
        return len(self._txbuf) + sum(len(f[1]) for f in self._out)

    # -- line side ----------------------------------------------------------------------------------
    def tx_bit(self) -> int:
        self._bits += 1
        if not self._tx:
            self._service()
        if self._tx:
            return self._tx.popleft()
        if self._phase == "data" and self.framing == 3:
            self._tx.extend(FLAG_BITS)              # flag fill between frames
            return self._tx.popleft()
        return 1

    def rx_bit(self, bit: int) -> None:
        if self.state == "down":
            return
        if self._phase == "data" and self.framing == 3:
            for frame, ok in self._hb.push(bit):
                self._on_frame(frame, ok)
            return
        buf = self._async_buf
        buf.append(bit)
        while len(buf) >= 10:
            if buf[0] == 0 and buf[9] == 1:
                octet = sum(buf[1 + j] << j for j in range(8))
                del buf[:10]
                result = self._ob.push(octet)
                if result is not None:
                    self._on_frame(*result)
                    if self._phase == "data" and self.framing == 3:
                        buf.clear()
                        break
            else:
                del buf[0]

    # -- frame building -----------------------------------------------------------------------------
    def _emit(self, body: bytes) -> None:
        if self._phase == "data" and self.framing == 3:
            frame = body + crc16_hdlc(body).to_bytes(2, "little")
            bits: list[int] = list(FLAG_BITS)
            ones = 0
            for byte in frame:
                for i in range(8):
                    b = (byte >> i) & 1
                    bits.append(b)
                    ones = ones + 1 if b else 0
                    if ones == 5:
                        bits.append(0)
                        ones = 0
            bits.extend(FLAG_BITS)
            self._tx.extend(bits)
            return
        stuffed = bytearray()
        for byte in body:
            stuffed.append(byte)
            if byte == DLE:
                stuffed.append(DLE)
        crc = crc16_v42a(body + bytes((ETX,)))
        octets = bytes((SYN, DLE, STX)) + bytes(stuffed) + bytes((DLE, ETX, crc & 0xFF, crc >> 8))
        for octet in octets:
            self._tx.extend(_octet_bits(octet))

    def _lr_body(self, framing: int, k: int, n401: int, optimize: int) -> bytes:
        rest = bytes((LR, 2)) + LR_CONSTANT2 + bytes((2, 1, framing, 3, 1, k, 4, 2, n401 & 0xFF, n401 >> 8))
        if optimize:
            rest += bytes((8, 1, optimize))
        return bytes((len(rest),)) + rest

    def _la_body(self, nr: int, credit: int) -> bytes:
        if self.optimize & OPT_FIXED_FIELDS and self._phase == "data":
            return bytes((3, LA, nr, credit))
        return bytes((7, LA, 1, 1, nr, 2, 1, credit))

    def _lt_body(self, ns: int, data: bytes) -> bytes:
        if self.optimize & OPT_FIXED_FIELDS:
            return bytes((2, LT, ns)) + data
        return bytes((4, LT, 1, 1, ns)) + data

    # -- timers -------------------------------------------------------------------------------------
    def _compute_timers(self) -> None:
        lb = 10 if self.framing == 2 else 8
        lf = 7 if self.framing == 2 else 4
        llt, lla = (3, 4) if self.optimize & OPT_FIXED_FIELDS else (5, 8)
        bits = self.k * lb * (lf + llt + self.n401) + lb * (lf + lla)
        self._t401_bits = int((bits / self.rate + self._round_trip) * self.rate)
        self._t404_bits = int((3.0 if self.rate >= 2400 else 7.0) * self.rate)

    # -- the transmit scheduler ---------------------------------------------------------------------
    def _service(self) -> None:
        if self.state == "down":
            return
        if self._phase == "establish":
            self._service_establish()
        else:
            self._service_data()

    def _service_establish(self) -> None:
        if self._reply_lr is not None:
            self._emit(self._reply_lr)
            self._last_reply, self._reply_lr = self._reply_lr, None
            self._resp_t401 = self._bits + self._est_t401
            return
        if not self.calling and self._resp_t401 is not None and self._bits >= self._resp_t401:
            self._reply_tries += 1                  # no confirming LA yet: send the reply LR again (A.7.1.3)
            if self._reply_tries > 2:
                self._fail("no confirming LA")
                return
            self._reply_lr = self._last_reply
            return
        if self._confirm_la:
            self._confirm_la = False
            self._emit(self._la_body(0, self.k) if not self.optimize & OPT_FIXED_FIELDS
                       else bytes((3, LA, 0, self.k)))
            self._enter_data()
            return
        if self.calling and self._bits >= self._lr_due and not self._reply_seen:
            self._lr_attempts += 1
            self._lr_due = self._bits + self._est_t401
            self._emit(self._lr_body(*self._want))
            self.events.append("LR sent")

    def _enter_data(self) -> None:
        self._phase = "data"
        self.state = "lapm"
        self._compute_timers()
        self._t404 = self._bits + self._t404_bits
        self._s_k = self.k
        self.events.append(f"data phase: framing {self.framing}, k {self.k}, N401 {self.n401}, opt {self.optimize}")

    def _service_data(self) -> None:
        now = self._bits
        if self._pending_lna is not None:
            nra, self._pending_lna = self._pending_lna, None
            self._emit(bytes((4, LNA, 1, 1, nra)))
            return
        if self._ack_due(now):
            self._send_la(now)
            return
        if self._t401 is not None and now >= self._t401:
            self._start_retransmission(now)
        if self._retx:
            frame = self._retx.popleft()
            if frame[2] >= N400:
                self._fail("retransmissions exceeded")
                return
            frame[2] += 1
            self._emit(self._lt_body(frame[0], frame[1]))
            if self._t401 is None or not self._retx:
                self._t401 = now + self._t401_bits
            return
        if self._txbuf and len(self._out) < self.k and self._s_k > 0:
            chunk = bytes(self._txbuf[:self.n401])
            del self._txbuf[:self.n401]
            frame = [self.vs, chunk, 1]
            self._out.append(frame)
            self.vs = (self.vs + 1) & 0xFF
            self._s_k -= 1
            self._emit(self._lt_body(frame[0], chunk))
            if self._t401 is None:
                self._t401 = now + self._t401_bits

    def _ack_due(self, now: int) -> bool:
        if self._force_ack or now >= self._t404:
            return True
        if self._unacked_rx == 0:
            return False
        if self.k == 1:
            return True
        if not self._txbuf and not self._retx:
            return True
        if self._unacked_rx * 2 >= self.k:
            return True
        return self._t402 is not None and now >= self._t402

    def _send_la(self, now: int) -> None:
        self._emit(self._la_body((self.vr - 1) & 0xFF, self.k))
        self._unacked_rx = 0
        self._force_ack = False
        self._t402 = None
        self._t404 = now + self._t404_bits

    def _start_retransmission(self, now: int) -> None:
        self._t401 = now + self._t401_bits
        if self._out:
            self._retx = deque(self._out)

    def _fail(self, why: str) -> None:
        self.events.append(f"failed: {why}")
        self.state = "down"

    # -- frame reception ----------------------------------------------------------------------------
    def _on_frame(self, body: bytes, ok: bool) -> None:
        if not ok:
            if self._phase == "data":
                self._force_ack = True              # an invalid frame is answered with an LA (A.7.3.3)
            return
        if len(body) < 2 or body[0] + 1 > len(body):
            return
        hdr_len, ftype = body[0], body[1]
        header, info = body[:hdr_len + 1], body[hdr_len + 1:]
        if ftype == LR:
            self._on_lr(header)
        elif self._phase == "establish":
            if ftype == LA and not self.calling and self._resp_t401 is not None:
                self._enter_data()
            elif ftype == LD:
                self._fail("peer rejected the link")
        elif ftype == LT:
            self._on_lt(header, info)
        elif ftype == LA:
            self._on_la(header)
        elif ftype == LN:
            self._on_ln(header)
        elif ftype == LD:
            self.events.append("LD received")
            self.state = "down"

    def _params(self, header: bytes, start: int) -> dict[int, bytes]:
        out, i = {}, start
        while i + 2 <= len(header):
            ptype, plen = header[i], header[i + 1]
            out[ptype] = header[i + 2:i + 2 + plen]
            i += 2 + plen
        return out

    def _on_lr(self, header: bytes) -> None:
        self.adp_seen = True
        if len(header) < 3 or header[2] != 2:
            self.events.append("LR with unexpected constant parameter 1")
            self._fail("bad LR")
            return
        params = self._params(header, 3)
        framing = params.get(2, b"\x02")[0]
        k = params.get(3, b"\x01")[0]
        n401 = int.from_bytes(params.get(4, b"\x40\x00")[:2].ljust(2, b"\0"), "little")
        opt = params.get(8, b"\x00")[0]
        w_framing, w_k, w_n401, w_opt = self._want
        framing, k, n401, opt = min(framing, w_framing), min(k, w_k), min(n401, w_n401), opt & w_opt
        if k < 1 or n401 < 1 or framing not in (2, 3):
            self._fail("unsupported LR parameters")
            return
        if self._phase != "establish":
            return
        if self.calling:                                   # the responder's reply: adopt and confirm
            self._reply_seen = True
            self._adopt(framing, k, n401, opt)
            self._confirm_la = True
        elif self._reply_lr is None:                       # also answers a repeated LR (A.7.1.3 c)
            self._adopt(framing, k, n401, opt)
            self._reply_lr = self._lr_body(framing, k, n401, opt)

    def _adopt(self, framing: int, k: int, n401: int, opt: int) -> None:
        """Record the negotiated parameters; they only take effect once the data phase starts."""
        self.k = k
        self.optimize = opt
        self.n401 = 256 if opt & OPT_N401_256 else n401
        self.framing = framing
        self._s_k = k

    def _on_la(self, header: bytes) -> None:
        if len(header) == 4:                                # fixed fields: length 3
            nr, credit = header[2], header[3]
        elif len(header) >= 8:
            nr, credit = header[4], header[7]
        else:
            return
        now = self._bits
        if self._out:
            idx = (nr - self._out[0][0]) & 0xFF
            if idx < len(self._out):
                del self._out[:idx + 1]
                self._retx = deque(f for f in self._retx if f in self._out)
                self._t401 = now + self._t401_bits if self._out else None
            else:
                if not self._retx:                          # a repeated N(R) asks for retransmission
                    self._retx = deque(self._out)
                    self._t401 = now + self._t401_bits
        self._s_k = max(0, credit - len(self._out))

    def _on_lt(self, header: bytes, info: bytes) -> None:
        if len(header) == 3:
            ns = header[2]
        elif len(header) >= 5:
            ns = header[4]
        else:
            return
        if not info:
            return
        if ns == self.vr and self._unacked_rx < self.k:
            self._rx_out += info
            self.vr = (self.vr + 1) & 0xFF
            self._unacked_rx += 1
            self._dup_seen = False
            if self._t402 is None:
                self._t402 = self._bits + self._t401_bits // 2
        elif ns == (self.vr - 1) & 0xFF and not self._dup_seen:
            self._dup_seen = True                           # the first duplicate is ignored (A.7.3.2.2)
        else:
            self._force_ack = True                          # out of sequence, or no receive credit

    def _on_ln(self, header: bytes) -> None:
        if len(header) >= 5:
            nsa = header[4]
            if nsa == self.vra:
                self.vra = (self.vra + 1) & 0xFF
            self._pending_lna = nsa


class _OctetDeframer:
    """Mode 2: SYN DLE STX body (DLE doubled) DLE ETX crc-lo crc-hi, fed one octet at a time."""

    def __init__(self) -> None:
        self.state = "hunt"
        self.hist = [0, 0, 0]
        self.body = bytearray()
        self.dle = False
        self.crc: list[int] = []

    def push(self, octet: int):
        if self.state == "hunt":
            self.hist = self.hist[1:] + [octet]
            if self.hist == [SYN, DLE, STX]:
                self.state, self.body, self.dle = "body", bytearray(), False
            return None
        if self.state == "body":
            if self.dle:
                self.dle = False
                if octet == DLE:
                    self.body.append(DLE)
                elif octet == ETX:
                    self.state, self.crc = "crc", []
                else:                                       # a DLE sequence that is not valid inside a frame
                    self.state, self.hist = "hunt", [0, 0, 0]
            elif octet == DLE:
                self.dle = True
            else:
                self.body.append(octet)
            if len(self.body) > 600:
                self.state, self.hist = "hunt", [0, 0, 0]
            return None
        self.crc.append(octet)
        if len(self.crc) < 2:
            return None
        self.state, self.hist = "hunt", [0, 0, 0]
        body = bytes(self.body)
        ok = (self.crc[0] | self.crc[1] << 8) == crc16_v42a(body + bytes((ETX,)))
        return body, ok


class _BitDeframer:
    """Mode 3: HDLC-style flags, zero deletion, CRC-CCITT; returns (frame without FCS, ok) at each closing flag."""

    def __init__(self) -> None:
        self.ones = 0
        self.bits: list[int] = []
        self.active = False

    def push(self, bit: int):
        out = []
        if bit:
            self.ones += 1
            if self.ones <= 6 and self.active:
                self.bits.append(1)
            if self.ones >= 7:                             # abort / idle: hunt for the next flag
                self.active, self.bits = False, []
            return out
        ones, self.ones = self.ones, 0
        if ones == 5:
            return out                                      # stuffed zero
        if ones == 6:                                       # flag (0 111111 0): its first zero and six ones were stored
            if self.active and len(self.bits) >= 7 + 24:
                data = self.bits[:-7]                       # minus the closing flag's leading zero and six ones
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
