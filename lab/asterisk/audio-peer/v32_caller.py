"""V.32 caller transmitter (written from scratch; pure Python, no numpy).

Structure follows ITU-T V.32 as implemented in the MIT-licensed ``fisher-modem``
(read for the algorithm) and was checked against what a V.32 answerer actually
sends (see lab/slmodem-peer/README.md). Nothing here comes from the Smart Link blob.

Signal: 2400 baud QAM on an 1800 Hz carrier, root-raised-cosine shaping (the
answerer's transmit spectrum is flat 600-3000 Hz with ~300 Hz roll-offs, i.e. RRC
with roll-off ~0.25), generated directly at the 8 kHz output instants: symbol k is
centred at t = k/2400, output sample n at t = n/8000, so the pulse is needed at
offsets (3n - 10k)/24000 -- a fixed table.

Symbol stream (caller): silence; then
    S      256 x ABAB...
    S-bar   16 x CDCD...
    TRN   8044 symbols from the caller's scrambler GPC (tap 18), reset to 0 and fed
          with ones: the first 256 use only A/C, the rest all four points
    rate signal: the 16-bit rate word repeated, MSB first, scrambled with GPC and
          sent as 4800 bps differentially-coded dibits.
"""

from __future__ import annotations

import cmath
import math
from collections import deque

FS = 8000
BAUD = 2400
CARRIER = 1800.0
SPS24 = 10                 # pulse-table points per symbol on the 24 kHz grid (8 kHz x 3)
SPAN = 6                   # pulse half-width in symbols

#            A         B         D         C            (fisher's ztab2 order)
POINTS = [(-3 - 1j), (1 - 3j), (-1 + 3j), (3 + 1j)]
A, B, D, C = POINTS[0], POINTS[1], POINTS[2], POINTS[3]

NR_DIFF = [1, 0, 3, 2, 3, 1, 2, 0, 0, 2, 1, 3, 2, 3, 0, 1]

# V.32bis rate bits and the base of the rate word
RATE_BITS = {4800: 0x0400, 7200: 0x0040, 9600: 0x0200, 12000: 0x0020, 14400: 0x0008}   # fisher modem.h (verified: 4800 and 9600 against the blob)
RATE_WORD_BASE = 0x0991

TRN_LENGTH = 8044          # measured from the Smart Link answerer; the rate word follows at once
S_LENGTH, SBAR_LENGTH, TRN_AC_LENGTH = 256, 16, 256

GPC = 18   # caller's scrambler tap (1 + x^-18 + x^-23)
GPA = 5    # answerer's scrambler tap (1 + x^-5 + x^-23)


def rate_word(*bps: int) -> int:
    word = RATE_WORD_BASE
    for rate in bps:
        word |= RATE_BITS[rate]
    return word


class Scrambler:
    """Self-synchronising scrambler 1 + x^-tap + x^-23 (fisher's scrambler)."""

    def __init__(self, tap: int):
        self.tap = tap
        self.wd = 0

    def reset(self) -> None:
        self.wd = 0

    def _taps(self) -> int:
        return ((self.wd >> (23 - self.tap)) & 1) ^ (self.wd & 1)

    def fwd(self, bit: int) -> int:
        out = bit ^ self._taps()
        self.wd = (self.wd >> 1) | (out << 22)
        return out

    def rev(self, bit: int) -> int:
        out = bit ^ self._taps()
        self.wd = (self.wd >> 1) | (bit << 22)
        return out


# 16-point constellation and 8-state trellis of the 7200 bps mode (fisher's coder.C)
ZTAB4 = [(3 - 3j), (-1 + 1j), (-3 + 3j), (1 - 1j), (3 + 1j), (-1 - 3j), (-3 - 1j), (1 + 3j),
         (-1 + 3j), (3 - 1j), (1 - 3j), (-3 + 1j), (-3 - 3j), (1 + 1j), (3 + 3j), (-1 - 1j)]
TFSM = [[0, 3, 2, 1], [4, 5, 7, 6], [1, 2, 3, 0], [7, 6, 4, 5],
        [2, 1, 0, 3], [6, 7, 5, 4], [3, 0, 1, 2], [5, 4, 6, 7]]
TE_DIFF = [0, 1, 2, 3, 1, 0, 3, 2, 2, 3, 1, 0, 3, 2, 0, 1]


class Trellis7200Encoder:
    """3 bits/symbol: bits 1-2 differentially coded, convolutional encoder on those, bit 3 uncoded."""

    def __init__(self, q: int = 0) -> None:
        self.q = q
        self.t = 0

    def encode(self, b: int) -> complex:
        self.q = ((self.q << 2) | (b >> 1)) & 0xF
        k = TE_DIFF[self.q]
        self.q = (self.q & 0xC) | k
        e = ((self.t & 1) << 3) | (k << 1) | (b & 1)
        self.t = TFSM[self.t][k]
        return ZTAB4[e]


# V.32 9600 bps trellis-coded signal-state mapping, keyed (Y0, Y1, Y2, Q3, Q4) -> Re+Im*j.
# Source: ITU-T V.32 Table 2 (trellis column) as reproduced in T. Lindebaum, "Design of a Simulation for
# Trellis Coded Modulation" (Univ. of Cape Town), Table 3.2 and Fig. 3.4; checked against the Smart Link
# answerer, whose transmitted symbols lie on exactly these 32 points.
V32_TCM_9600 = {
    (0,0,0,0,0):(-4,1),
    (0,0,0,0,1):(0,-3),
    (0,0,0,1,0):(0,1),
    (0,0,0,1,1):(4,1),
    (0,0,1,0,0):(4,-1),
    (0,0,1,0,1):(0,3),
    (0,0,1,1,0):(0,-1),
    (0,0,1,1,1):(-4,-1),
(0,1,0,0,0):(-2,3),
    (0,1,0,0,1):(-2,-1),
    (0,1,0,1,0):(2,3),
    (0,1,0,1,1):(2,-1),
    (0,1,1,0,0):(2,-3),
    (0,1,1,0,1):(2,1),
    (0,1,1,1,0):(-2,-3),
    (0,1,1,1,1):(-2,1),
(1,0,0,0,0):(-3,-2),
    (1,0,0,0,1):(1,-2),
    (1,0,0,1,0):(-3,2),
    (1,0,0,1,1):(1,2),
    (1,0,1,0,0):(3,2),
    (1,0,1,0,1):(-1,2),
    (1,0,1,1,0):(3,-2),
    (1,0,1,1,1):(-1,-2),
(1,1,0,0,0):(1,4),
    (1,1,0,0,1):(-3,0),
    (1,1,0,1,0):(1,0),
    (1,1,0,1,1):(1,-4),
    (1,1,1,0,0):(-1,-4),
    (1,1,1,0,1):(3,0),
    (1,1,1,1,0):(-1,0),
    (1,1,1,1,1):(-1,4)
}
V32_TCM_9600 = {k: complex(*v) for k, v in V32_TCM_9600.items()}


def _scaled(table, power=10.0):
    """Label-string table -> {bit tuple: complex} scaled to the given average power."""
    pts = {tuple(int(b) for b in k): complex(*v) for k, v in table.items()}
    p = sum(abs(z) ** 2 for z in pts.values()) / len(pts)
    f = (power / p) ** 0.5
    return {k: z * f for k, z in pts.items()}


# the V.32bis 12000/14400 figures are in their own units; scaled to the training power like 9600
from v32bis_tables import V32BIS_12000, V32BIS_14400  # noqa: E402

V32_TCM = {9600: V32_TCM_9600, 12000: _scaled(V32BIS_12000), 14400: _scaled(V32BIS_14400)}
TCM_BITS = {9600: 4, 12000: 5, 14400: 6}              # data bits per symbol


class TrellisEncoder:
    """V.32/V.32bis TCM at 9600, 12000 or 14400 bps: Q1,Q2 -> differential (Y2Y1 = previous + Q2Q1
    mod 4) -> 8-state encoder (state bit 0 is Y0, next state = TFSM[state][(Y1<<1)|Y2], the same code as
    7200); Q3.. uncoded. The first scrambler bit is Q1. 9600 verified bit-for-bit against the blob and
    on a real call; 12000/14400 use the ITU V.32bis figures (see v32bis_tables.py)."""

    def __init__(self, rate: int = 9600) -> None:
        self.table = V32_TCM[rate]
        self.nbits = TCM_BITS[rate]
        self.t = 0
        self.y = 0

    def encode(self, b: int) -> complex:
        n = self.nbits
        q1, q2 = (b >> (n - 1)) & 1, (b >> (n - 2)) & 1
        rest = tuple((b >> (n - 3 - i)) & 1 for i in range(n - 2))
        self.y = (self.y + ((q2 << 1) | q1)) & 3
        y2, y1 = self.y >> 1, self.y & 1
        z = self.table[(self.t & 1, y1, y2) + rest]
        self.t = TFSM[self.t][(y1 << 1) | y2]
        return z


class Trellis9600Encoder(TrellisEncoder):
    def __init__(self) -> None:
        super().__init__(9600)


class Differential4800Encoder:
    """Non-redundant 4800 bps coding: two bits per symbol, differentially coded."""

    def __init__(self) -> None:
        self.q = 0

    def encode(self, dibit: int) -> complex:
        self.q = ((self.q << 2) | dibit) & 0xF
        k = NR_DIFF[self.q & 0xF]
        self.q = (self.q & 0xC) | k
        return POINTS[k]


def rrc_pulse(rolloff: float, span: int = SPAN, per_symbol: int = SPS24) -> list[float]:
    """Root-raised-cosine pulse on a grid of ``per_symbol`` points per symbol (10 =
    the 24 kHz grid used for 8 kHz output), returned over -span*per_symbol ..
    +span*per_symbol (centre at ``span * per_symbol``)."""
    b = rolloff
    out = []
    for j in range(-span * per_symbol, span * per_symbol + 1):
        t = j / per_symbol
        if j == 0:
            v = 1.0 - b + 4.0 * b / math.pi
        elif abs(abs(4.0 * b * t) - 1.0) < 1e-9:
            v = (b / math.sqrt(2.0)) * (
                (1 + 2 / math.pi) * math.sin(math.pi / (4 * b))
                + (1 - 2 / math.pi) * math.cos(math.pi / (4 * b))
            )
        else:
            v = (math.sin(math.pi * t * (1 - b)) + 4 * b * t * math.cos(math.pi * t * (1 + b))) / (
                math.pi * t * (1 - (4 * b * t) ** 2)
            )
        out.append(v)
    return out


class V32CallerTx:
    """Generates the caller's audio. Call ``start_training()`` when it is time (after
    the answerer's R1), then ``tx_samples(n)`` for every 8 kHz block."""

    def __init__(self, my_rates=(9600,), amplitude: float = 900.0, rolloff: float = 0.25,
                 scrambler_tap: int = GPC, sample_rate: int = FS, span: int = SPAN,
                 role: str = "caller", s_length: int = S_LENGTH, trn_length: int = TRN_LENGTH,
                 carrier: float = CARRIER):
        self.role = role
        self.carrier = carrier
        self.s_length = s_length
        self.trn_length = trn_length
        if role == "answerer" and scrambler_tap == GPC:
            scrambler_tap = GPA
        if sample_rate not in (8000, 9600):
            raise ValueError("sample_rate must be 8000 or 9600")
        self.fs = sample_rate
        # grid: 24 kHz for 8 kHz output (3 grid steps per sample, 10 per symbol);
        # the sample rate itself for 9.6 kHz output (1 step per sample, 4 per symbol)
        self._g_sample = 3 if sample_rate == 8000 else 1
        self._g_symbol = 10 if sample_rate == 8000 else 4
        self.rate_word = rate_word(*my_rates)
        self.amplitude = amplitude
        self._span = span
        self._pulse = rrc_pulse(rolloff, span, self._g_symbol)
        self._centre = span * self._g_symbol
        # Integer-grid polyphase support, preserving the original summation
        # order. Only the symbol origin changes between samples of a phase.
        gy = self._g_symbol
        self._pulse_phases = [
            [(offset, self._pulse[self._centre + phase - gy * offset])
             for offset in range(-span + bool(phase), span + 1)]
            for phase in range(gy)
        ]
        self._scr = Scrambler(scrambler_tap)
        self._enc = Differential4800Encoder()
        self._symbol_source = self._silence()
        self._sym: dict[int, complex] = {}     # absolute symbol index -> value
        self._k_next = 0                       # next symbol index to be generated
        self._n = 0                            # absolute output sample index
        self._ac_reversed = False
        self._send_e = False
        self._e_word = None
        self.data_rate = max(my_rates)
        self.bit_source = None
        self._tx_bytes: deque[int] = deque()
        self._byte_bits: deque[int] = deque()
        self._current_byte = None
        self._current_bits_left = 0
        self.state = "silent"                  # silent | ac | training | rate

    # -- symbol generators ----------------------------------------------------
    @staticmethod
    def _silence():
        while True:
            yield 0j

    def _training(self, gap_symbols: int = 0):
        scr = self._scr
        scr.reset()
        for _ in range(gap_symbols):                # silence between AC and S (answerer)
            yield 0j
        last = 0
        s_len = self.s_length
        for bc in range(s_len + SBAR_LENGTH + self.trn_length):
            if bc < s_len:                          # S: ABAB...
                yield POINTS[1] if bc & 1 else POINTS[0]
            elif bc < s_len + SBAR_LENGTH:          # S-bar: CDCD...
                yield POINTS[2] if bc & 1 else POINTS[3]
            else:                                   # TRN
                b1, b2 = scr.fwd(1), scr.fwd(1)
                if bc < s_len + SBAR_LENGTH + TRN_AC_LENGTH:
                    b2 = b1
                last = (b1 << 1) | b2
                yield POINTS[last]
        self.state = "rate"
        yield from self._rate_signal(last)

    def _rate_signal(self, last_point: int = 0):
        enc = self._enc
        enc.q = last_point                          # differential coding continues from the TRN
        scr = self._scr                             # ... and so does the scrambler
        word = self.rate_word
        bits = []                                   # the word starts right after the TRN
        sent_e = False
        while True:
            if self._send_e and not sent_e and not bits:
                # E: the agreed rate (one bit) with the top nibble set (fisher: rateword | 0xf000)
                word_now, sent_e, self.state = (self._e_word if self._e_word is not None else (word | 0xF000)), True, "e"
            elif sent_e:
                word_now = None                     # after E: scrambled ones until data starts
            else:
                word_now = word
            if word_now is None and self.data_rate in (7200, 9600, 12000, 14400):
                break                               # E is out; switch to trellis data
            if word_now is None:
                bits += [self._next_data_bit() for _ in range(16)]
            else:
                bits += [(word_now >> (15 - i)) & 1 for i in range(16)]
            while len(bits) >= 2:
                b1 = scr.fwd(bits.pop(0))
                b2 = scr.fwd(bits.pop(0))
                yield enc.encode((b1 << 1) | b2)
        # trellis modes: scrambled bits per symbol through the trellis encoder (E went out at 4800)
        if self.data_rate in TCM_BITS:
            tenc, nbits = TrellisEncoder(self.data_rate), TCM_BITS[self.data_rate]
        else:
            tenc, nbits = Trellis7200Encoder(0), 3
        while True:
            b = 0
            for _ in range(nbits):
                b = (b << 1) | scr.fwd(self._next_data_bit())
            yield tenc.encode(b)

    def feed_bytes(self, data: bytes) -> None:
        """Queue application bytes; sent async-framed (start 0, 8 data LSB first, stop 1) once E
        has been sent. Idle is ones (marks), as in fisher's putasync."""
        self._tx_bytes.extend(data)

    def _frame_bit(self) -> int:
        """Next bit of the async-framed byte queue (idle marks)."""
        if not self._byte_bits and self._tx_bytes:
            b = self._tx_bytes.popleft()
            self._current_byte = b
            self._current_bits_left = 10
            self._byte_bits.extend([0] + [(b >> i) & 1 for i in range(8)] + [1])
        bit = self._byte_bits.popleft() if self._byte_bits else 1
        if self._current_bits_left:
            self._current_bits_left -= 1
            if not self._current_bits_left:
                self._current_byte = None
        return bit

    def feed_marks(self, n: int) -> None:
        """Queue ``n`` idle mark bits (lets a peer's async receiver resynchronise)."""
        self._byte_bits.extend([1] * n)

    def _next_data_bit(self) -> int:
        if self.bit_source is not None:               # e.g. a V.42 layer owns the bit stream
            return self.bit_source()
        return self._frame_bit()

    def request_e(self, rate: int | None = None) -> None:
        """Send E (once) at the next rate-word boundary: the answerer's R3 has been seen.
        ``rate`` picks the agreed rate (default: the highest we offered); E carries just that bit."""
        if rate is not None:
            self.data_rate = rate
            self._e_word = 0xF000 | RATE_WORD_BASE | RATE_BITS[rate]
        self._send_e = True

    def start_rate_change(self, exchange):
        """Install a section-8 symbol stream without resetting PCM clock/history.

        Internal transmitter hook only. A live owner must validate receive
        preamble/E and account for the pulse shaper's generated-ahead symbols
        before declaring rate completion. Existing startup stays unchanged.
        """
        from v32_rate_change import RateChangeTx
        if not isinstance(exchange, RateChangeTx):
            raise TypeError('RateChangeTx required')
        expected_role = 'call' if self.role == 'caller' else 'answer'
        if exchange.role != expected_role or self.state not in ('e', 'data'):
            raise ValueError('matching connected transmitter without active exchange required')
        self._rate_change = exchange
        self.rate_change_start_symbol = self._k_next
        self.rate_change_data_symbol = None
        self.rate_change_terminal_symbol = None
        self.state = 'rrp_pending'
        self._symbol_source = self._rate_change_symbols(exchange)
        return self.rate_change_start_symbol

    def _rate_change_symbols(self, exchange):
        for symbol in exchange.symbols(self._next_data_bit):
            self.state = 'data' if exchange.state == 'data' else 'rrp_' + exchange.state
            if exchange.state in ('ones', 'data'):
                self.data_rate = exchange.selected_rate
            if exchange.state == 'data' and self.rate_change_data_symbol is None:
                self.rate_change_data_symbol = self._k_next
            yield symbol
        self.state = 'rrp_cleared'
        self.rate_change_terminal_symbol = self._k_next
        yield from self._silence()

    # -- answerer-side signals (used by the test rig to play the far end) ---------
    def _alternating(self):
        """AC: A and C alternating (600 + 3000 Hz); the order flips on ``reverse_ac``."""
        while True:
            first, second = (A, C) if not self._ac_reversed else (C, A)
            yield first
            yield second

    def start_ac(self) -> None:
        self._ac_reversed = False
        self.state = "ac"
        self._symbol_source = self._alternating()
        self._sym = {k: 0j for k in self._sym}

    def reverse_ac(self) -> None:
        """Phase-reverse the alternating signal (AC <-> CA)."""
        self._ac_reversed = not self._ac_reversed

    def stop(self) -> None:
        """Cease transmitting (the symbols already generated ahead still ramp out)."""
        self.state = "silent"
        self._symbol_source = self._silence()

    def start_training_now(self, gap_symbols: int = 0) -> None:
        """Begin S/S-bar/TRN from any state (the answerer does this after the AC/CA/AC dance;
        the Smart Link answerer leaves ~29 symbols of silence between its AC and its S)."""
        self.state = "training"
        self._symbol_source = self._training(gap_symbols)
        self._sym = {k: 0j for k in self._sym}

    # -- control ------------------------------------------------------------
    def start_training(self) -> None:
        """Begin S/S-bar/TRN at the next symbol boundary."""
        if self.state != "silent":
            return
        self.state = "training"
        self._symbol_source = self._training()
        # symbols already generated (look-ahead) stay silent; drop them so the
        # training begins as soon as the pulse window allows
        self._sym = {k: 0j for k in self._sym}

    # -- audio -----------------------------------------------------------------
    def _symbol(self, k: int) -> complex:
        while self._k_next <= k:
            self._sym[self._k_next] = next(self._symbol_source) * self.amplitude
            self._k_next += 1
        return self._sym.get(k, 0j)

    def tx_samples(self, count: int) -> list[int]:
        out = []
        phases = self._pulse_phases
        gs, gy = self._g_sample, self._g_symbol
        symbols = self._sym
        get_symbol = symbols.get
        for _ in range(count):
            n = self._n
            origin, phase = divmod(gs * n, gy)
            k_hi = origin + self._span
            if self._k_next <= k_hi:
                self._symbol(k_hi)
            acc = 0j
            for offset, coefficient in phases[phase]:
                v = get_symbol(origin + offset, 0j)
                if v:
                    acc += v * coefficient
            theta = 2.0 * math.pi * self.carrier * n / self.fs
            out.append(round(acc.real * math.cos(theta) - acc.imag * math.sin(theta)))
            self._n += 1
        # forget symbols that can no longer influence the output
        horizon = (gs * self._n) // gy - self._span - 2
        for k in [k for k in self._sym if k < horizon]:
            del self._sym[k]
        return out
