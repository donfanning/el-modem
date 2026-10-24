"""ITU-T V.42bis data compression (1990 text), implemented from the Recommendation.

Sources: ITU-T V.42bis (01/90) sections 5-10 and Annex A. Behaviour that the text leaves open (when to switch between
compressed and transparent mode, §7.8) is a simple window heuristic of our own. Interoperability with another
implementation (SpanDSP's v42bis, used only as a test reference) is checked in test_v42bis.py; no code was taken from it.

    comp = V42bis(p0=3, n2=1024, n7=8)     # P0: 1 = initiator->responder, 2 = responder->initiator, 3 = both
    wire = comp.compress(data) + comp.flush()      # octets for the error-controlled link
    data = comp.decompress(wire)                    # (the peer's direction is a separate V42bis object)

``Encoder`` and ``Decoder`` are the two halves; V42bis wires one of each for the directions negotiated in P0.
"""

from __future__ import annotations

import math

ETM, FLUSH, STEPUP = 0, 1, 2                 # control codewords (§9.1, Table 2)
ECM, EID, RESET = 0, 1, 2                    # transparent mode command codes
N3 = 8                                       # character size
N4 = 1 << N3
N6 = 3                                       # control codewords
N5 = N4 + N6                                 # first dictionary entry used for a string
ESC_STEP = 51                                # escape character cycling (§9.2)


def _validate_parameters(n2: int, n7: int) -> None:
    # Annex A encodes P1 (N2) in two octets, so 65535 is the largest value a
    # peer can negotiate even though the algorithm itself specifies no upper
    # N2 bound. P2 (N7) is explicitly limited to 6..250 by §5.1.
    if not isinstance(n2, int) or not 512 <= n2 <= 65535:
        raise ValueError("P1 (N2) must be 512..65535")
    if not isinstance(n7, int) or not 6 <= n7 <= 250:
        raise ValueError("P2 (N7) must be 6..250")


class V42bisError(Exception):
    """A C-ERROR condition (§5.8): the decoder lost synchronisation or saw a procedural error."""


class Dictionary:
    """The tree dictionary of §6: root nodes for every character, then N5..N2-1 for strings; counter C1 (§6.5)."""

    def __init__(self, n2: int, n7: int):
        self.n2, self.n7 = n2, n7
        self.parent = [-1] * n2
        self.char = [0] * n2
        self.length = [0] * n2
        self.children: list[dict[int, int] | None] = [None] * n2
        self.in_use = [False] * n2
        self.reset()

    def reset(self) -> None:
        """§6.2: only the root nodes exist, C1 = N5."""
        for i in range(self.n2):
            self.parent[i] = -1
            self.children[i] = None
            self.in_use[i] = False
        for c in range(N4):
            cw = N6 + c
            self.char[cw], self.length[cw], self.in_use[cw] = c, 1, True
            self.children[cw] = {}
        self.c1 = N5

    @staticmethod
    def root(char: int) -> int:
        return N6 + char

    def child(self, node: int, char: int) -> int | None:
        return self.children[node].get(char)

    def is_leaf(self, node: int) -> bool:
        return not self.children[node]

    def string(self, node: int) -> bytes:
        out = bytearray()
        while node >= 0:
            out.append(self.char[node])
            node = self.parent[node]
        out.reverse()
        return bytes(out)

    def add(self, parent: int, char: int) -> int | None:
        """§6.4: append ``char`` to ``parent`` as a new entry unless that exceeds N7 or already exists; then §6.5."""
        if self.length[parent] + 1 > self.n7 or char in self.children[parent]:
            return None
        cw = self.c1
        self.parent[cw], self.char[cw], self.length[cw] = parent, char, self.length[parent] + 1
        self.children[cw] = {}
        self.children[parent][char] = cw
        self.in_use[cw] = True
        # §6.5: advance C1 to the next entry that is free, recovering a leaf if need be
        while True:
            self.c1 += 1
            if self.c1 > self.n2 - 1:
                self.c1 = N5
            c1 = self.c1
            if self.in_use[c1] and not self.is_leaf(c1):
                continue
            if self.in_use[c1]:                                    # a leaf: detach it from its parent
                del self.children[self.parent[c1]][self.char[c1]]
                self.in_use[c1] = False
            break
        return cw


class _BitWriter:
    def __init__(self) -> None:
        self.out = bytearray()
        self.acc = 0
        self.nbits = 0

    def put(self, value: int, width: int) -> None:
        """Least significant bit of a codeword first (§7.5, Figure 3)."""
        self.acc |= value << self.nbits
        self.nbits += width
        while self.nbits >= 8:
            self.out.append(self.acc & 0xFF)
            self.acc >>= 8
            self.nbits -= 8

    def align(self) -> None:
        if self.nbits:
            self.out.append(self.acc & 0xFF)
            self.acc = 0
            self.nbits = 0

    @property
    def pending(self) -> bool:
        return self.nbits > 0

    def take(self) -> bytes:
        data, self.out = bytes(self.out), bytearray()
        return data


class Encoder:
    def __init__(self, n2: int = 1024, n7: int = 8, auto_switch: bool = True):
        _validate_parameters(n2, n7)
        self.n2, self.n7 = n2, n7
        self.n1 = max(1, math.ceil(math.log2(n2)))
        self.d = Dictionary(n2, n7)
        self.auto = auto_switch
        self.bits = _BitWriter()
        self.init()

    def init(self) -> None:
        """C-INIT (§7.2): dictionary reset, C2 = N3 + 1, C3 = N4 x 2, transparent mode, escape character 0."""
        self.d.reset()
        self.bits = _BitWriter()
        self.c2, self.c3 = N3 + 1, N4 * 2
        self.compressed = False
        self.escape = 0
        self.cur: int | None = None             # node of the string being matched
        self.last_created: int | None = None    # entry created by the previous invocation of the matching procedure
        self.deferred: int | None = None        # string whose dictionary extension waits for the next character
        self.win_chars = 0                      # compressibility bookkeeping (§7.8)
        self.win_bits = 0
        self.out = bytearray()

    def request_reset(self) -> bytes:
        """Signal §7.8.3 RESET to the peer, then reinitialize this encoder."""
        if self.compressed:
            if self.cur is not None:
                self._codeword(self.cur)
            self.bits.put(ETM, self.c2)
            self.bits.align()
            self._drain_bits()
            self.compressed = False
        self._escape_command(RESET)
        wire = bytes(self.out)
        self.init()
        return wire

    # -- output helpers ------------------------------------------------------------------------------------
    def _codeword(self, cw: int) -> None:
        while cw >= self.c3:                    # §7.4: STEPUP until the codeword fits
            self.bits.put(STEPUP, self.c2)
            self.c2 += 1
            self.c3 *= 2
        self.bits.put(cw, self.c2)

    def _cost(self, cw: int) -> int:
        c2, c3, bits = self.c2, self.c3, 0
        while cw >= c3:
            bits += c2
            c2 += 1
            c3 *= 2
        return bits + c2

    def _transparent_char(self, ch: int) -> None:
        self.out.append(ch)
        if ch == self.escape:                   # §9.2: escape in data, then cycle the escape character
            self.out.append(EID)
            self.escape = (self.escape + ESC_STEP) & 0xFF

    def _note_escape(self, ch: int) -> None:
        if ch == self.escape:
            self.escape = (self.escape + ESC_STEP) & 0xFF

    def _drain_bits(self) -> None:
        self.out += self.bits.take()

    # -- the matching procedure (§6.3) ---------------------------------------------------------------------
    def _mismatch(self, ch: int) -> int | None:
        created = self.d.add(self.cur, ch)
        self.last_created = created
        return created

    def feed(self, data: bytes) -> None:
        for ch in data:
            self._char(ch)

    def _char(self, ch: int) -> None:
        d = self.d
        if self.cur is None:
            if self.deferred is not None:                       # §7.9 b): the extension waits for this character
                self.last_created = d.add(self.deferred, ch)
                self.deferred = None
            self.cur = d.root(ch)
            if self.compressed:
                self._note_escape(ch)
            else:
                self._transparent_char(ch)
            return
        nxt = d.child(self.cur, ch)
        if nxt is not None and nxt != self.last_created:        # step b): the string grows
            self.cur = nxt
            if self.compressed:
                self._note_escape(ch)
            else:
                self._transparent_char(ch)
            return
        # step c): self.cur is the longest match, ch the unmatched character
        self.win_bits += self._cost(self.cur)                   # what this codeword costs (also when we send chars)
        self.win_chars += d.length[self.cur]
        if self.compressed:
            self._codeword(self.cur)
        self._mismatch(ch)
        want = self._decide()
        if self.compressed:
            if want == "transparent":                           # §7.8.2
                self.bits.put(ETM, self.c2)
                self.bits.align()
                self._drain_bits()
                self.compressed = False
                self._transparent_char(ch)
            else:
                self._note_escape(ch)
        else:
            if want == "compressed":                            # §7.8.1
                self._escape_command(ECM)
                self.compressed = True
                self._note_escape(ch)
            else:
                self._transparent_char(ch)
        self.cur = d.root(ch)
        if self.compressed:
            self._drain_bits()

    def _escape_command(self, code: int) -> None:
        self.out.append(self.escape)
        self.out.append(code)

    def _decide(self) -> str | None:
        """§7.8 compressibility test (our own heuristic): compare the bits a window would need either way."""
        if not self.auto or self.win_chars < 128:
            return None
        raw, used = 8 * self.win_chars, self.win_bits
        self.win_chars = self.win_bits = 0
        if self.compressed:
            return "transparent" if used > raw * 1.05 else None
        return "compressed" if used < raw * 0.9 else None

    def flush(self) -> bytes:
        """C-FLUSH (§7.9): push every partially encoded string out; returns the octets produced since the last call."""
        if self.compressed:
            if self.cur is not None:
                self._codeword(self.cur)
                self.deferred = self.cur
                self.cur = None
            if self.bits.pending:
                self.bits.put(FLUSH, self.c2)
                self.bits.align()
            self._drain_bits()
        out, self.out = bytes(self.out), bytearray()
        return out

    def compress(self, data: bytes) -> bytes:
        """Encode ``data``; octets come out as they complete (call flush() when the stream goes idle)."""
        self.feed(data)
        if self.compressed:
            self._drain_bits()
        out, self.out = bytes(self.out), bytearray()
        return out


class Decoder:
    def __init__(self, n2: int = 1024, n7: int = 8):
        _validate_parameters(n2, n7)
        self.n2, self.n7 = n2, n7
        self.n1 = max(1, math.ceil(math.log2(n2)))
        self.d = Dictionary(n2, n7)
        self.init()

    def init(self) -> None:
        self.d.reset()
        self.c2, self.c3 = N3 + 1, N4 * 2
        self.compressed = False
        self.escape = 0
        self.esc_seen = False                    # transparent mode: previous octet was the escape character
        self.cur_t: int | None = None            # transparent mode string matching (§8: keeps the dictionary in step)
        self.last_created: int | None = None
        self.pending: int | None = None          # compressed mode: node whose extension waits for the next first char
        self.acc = 0
        self.nbits = 0

    def _escape_step(self, ch: int) -> None:
        if ch == self.escape:
            self.escape = (self.escape + ESC_STEP) & 0xFF

    def _transparent_data(self, ch: int, out: bytearray) -> None:
        d = self.d
        out.append(ch)
        if self.cur_t is None:
            if self.pending is not None:                         # first character after ETM / FLUSH-less transition
                self.last_created = d.add(self.pending, ch)
                self.pending = None
            self.cur_t = d.root(ch)
            return
        nxt = d.child(self.cur_t, ch)
        if nxt is not None and nxt != self.last_created:
            self.cur_t = nxt
        else:
            self.last_created = d.add(self.cur_t, ch)
            self.cur_t = d.root(ch)

    def decompress(self, data: bytes) -> bytes:
        out = bytearray()
        for byte in data:
            if self.compressed:
                self.acc |= byte << self.nbits
                self.nbits += 8
                self._codewords(out)
            else:
                self._transparent_octet(byte, out)
        return bytes(out)

    def _transparent_octet(self, b: int, out: bytearray) -> None:
        if self.esc_seen:
            self.esc_seen = False
            if b == EID:                                         # the escape character itself was data
                self._transparent_data(self.escape, out)
                self._escape_step(self.escape)
            elif b == ECM:
                self.compressed = True
                self.pending = self.cur_t                        # §7.8.1 a): extension completes with the next first char
                self.cur_t = None
                self.acc, self.nbits = 0, 0
            elif b == RESET:
                self.d.reset()
                self.c2, self.c3 = N3 + 1, N4 * 2
                self.cur_t = self.last_created = self.pending = None
                self.escape = 0
                self.acc, self.nbits = 0, 0
            else:
                raise V42bisError(f"reserved command code {b}")
        elif b == self.escape:
            self.esc_seen = True
        else:
            self._transparent_data(b, out)

    def _codewords(self, out: bytearray) -> None:
        d = self.d
        while self.compressed and self.nbits >= self.c2:
            w = self.acc & ((1 << self.c2) - 1)
            self.acc >>= self.c2
            self.nbits -= self.c2
            if w == ETM:
                drop = self.nbits % 8                            # §7.5: zero bits up to the octet boundary
                self.acc >>= drop
                self.nbits -= drop
                self.compressed = False
                self.cur_t = None                                # pending extension completes with the next octet
                while self.nbits >= 8:                           # whole octets already loaded are transparent data
                    b = self.acc & 0xFF
                    self.acc >>= 8
                    self.nbits -= 8
                    self._transparent_octet(b, out)
                    if self.compressed:                          # (an ECM can follow immediately)
                        break
                return
            if w == FLUSH:
                drop = self.nbits % 8                            # §7.5: zero bits up to the octet boundary
                self.acc >>= drop
                self.nbits -= drop
                continue
            if w == STEPUP:
                if self.c2 + 1 > self.n1:
                    raise V42bisError("STEPUP beyond the maximum codeword size")
                self.c2 += 1
                self.c3 *= 2
                continue
            if w == d.c1 or not d.in_use[w]:
                raise V42bisError(f"codeword {w} is empty or equals C1")
            s = d.string(w)
            if self.pending is not None:
                d.add(self.pending, s[0])
            self.pending = w
            for ch in s:
                self._escape_step(ch)
            out += s


class V42bis:
    """Both directions as negotiated: ``p0`` bit 0 = we are the initiator and compress (or responder, for the peer's
    direction), as in the XID parameter; ``local`` selects which role we play (see the constructor arguments)."""

    def __init__(self, compress_dir: bool, decompress_dir: bool, n2: int = 1024, n7: int = 8):
        self.enc = Encoder(n2, n7) if compress_dir else None
        self.dec = Decoder(n2, n7) if decompress_dir else None

    def compress(self, data: bytes) -> bytes:
        return self.enc.compress(data) if self.enc else data

    def flush(self) -> bytes:
        return self.enc.flush() if self.enc else b""

    def decompress(self, data: bytes) -> bytes:
        return self.dec.decompress(data) if self.dec else data
