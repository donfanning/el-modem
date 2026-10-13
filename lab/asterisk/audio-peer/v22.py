"""ITU-T V.22 / V.22bis modem (1200 bit/s 4-phase DPSK and 2400 bit/s 16-QAM, 600 baud), written from the 1988
Recommendations (V.22bis text, which also contains the V.22 compatible 1200 bit/s mode and handshake).

``V22Modem`` has the surface of the SpanDSP-backed class it replaces (spandsp_v22.V22bisModem): ``tx_samples`` /
``feed_pcm`` at 8 kHz, ``feed_bytes``, ``pending_bits``, ``connected``, ``is_settled``, ``bit_rate``, ``events`` as
(name, code) pairs and the V.42 / MNP hooks of ``V42Layer``. Calling modems transmit in the low channel (1200 Hz) and receive
in the high one (2400 Hz); answering modems the other way round.

Implemented: scrambler 1 + x^-14 + x^-17 with the 64-ones guard (§5), quadrant-change encoding and the 16-point constellation
of Figure 2 (§2.5), square-root raised cosine 75% shaping (§2.4), the GSTN handshake of §6.3.1 for 1200 and 2400 bit/s with
S1 / unscrambled binary 1 / scrambled binary 1 and the 155, 456, 100, 600, 450, 200, 270 and 765 ms timings. Receiver: matched
filter, symbol timing from the 600 Hz line of the squared envelope, differential 4-phase detection for the handshake and
decision-directed carrier tracking for 16-QAM. Not implemented: the retrain and rate-change sequences of §6.4-6.6 (a lost
carrier drops the call), loop tests, guard tones, Bell 212A answer tone (``bell212a`` is accepted and ignored: the waveform
and handshake are the V.22 ones), an adaptive equalizer (a linear channel at 600 baud does not need one in the lab; real lines
may).

Credit: ITU-T V.22bis (11/88) is the specification. SpanDSP's v22bis.c (LGPL, Steve Underwood) was used as an independent
reference peer in tests and to check the handshake timing margins real peers tolerate; no code was taken from it.
"""

from __future__ import annotations

import math
import json
import os
from collections import deque

import numpy as np

from v34_tx import rrc
from v42_layer import V42Layer, _Framer

FS = 8000
BAUD = 600
T = FS / BAUD                                            # 13.33 samples per symbol
LOW, HIGH = 1200.0, 2400.0
# Signed 16-bit telephony PCM: a full-scale sine is +3.14 dBm0, using
# the conventional G.711/SpanDSP linear reference (G.711 section 5).
# This is an RMS reference; constellation energy is accounted for below.
_ZERO_DBM0_POWER = (32768.0 ** 2 / 2.0) * 10 ** (-3.14 / 10.0)
_ZERO_DBM0_RMS = math.sqrt(_ZERO_DBM0_POWER)

# Figure 2: (re, im) of the element labelled with the last two bits of a quadbit, per quadrant 1..4
_ELEMENTS = {
    1: {0b00: (1, 1), 0b01: (3, 1), 0b10: (1, 3), 0b11: (3, 3)},
    2: {0b00: (-1, 1), 0b10: (-3, 1), 0b01: (-1, 3), 0b11: (-3, 3)},
    3: {0b00: (-1, -1), 0b01: (-3, -1), 0b10: (-1, -3), 0b11: (-3, -3)},
    4: {0b00: (1, -1), 0b10: (3, -1), 0b01: (1, -3), 0b11: (3, -3)},
}
# Table 1: first two bits -> quadrant change in 90 degree steps (counter-clockwise)
_CHANGE = {0b00: 1, 0b01: 0, 0b11: 3, 0b10: 2}
_CHANGE_INV = {v: k for k, v in _CHANGE.items()}
POINTS = [(q, bits, complex(*xy)) for q, d in _ELEMENTS.items() for bits, xy in d.items()]
_PTS = np.array([p[2] for p in POINTS])
# 1200 bit/s: the element labelled 01 of whatever quadrant is entered
_DPSK_POINT = {q: complex(*_ELEMENTS[q][0b01]) for q in (1, 2, 3, 4)}
_DPSK_ANGLE = math.atan2(1, 3)                           # 18.43 degrees: angle of the 01 element in quadrant 1


class Scrambler:
    """1 + x^-14 + x^-17, self-synchronising; with the 64 consecutive ones guard of §5.1."""

    def __init__(self) -> None:
        self.reg = [0] * 17
        self.ones = 0

    def step(self, di: int) -> int:
        if self.ones >= 64:                               # §5.1: invert the next input to break scrambler lock-up
            di ^= 1
            self.ones = 0
        ds = di ^ self.reg[13] ^ self.reg[16]
        self.reg = [ds] + self.reg[:-1]
        self.ones = self.ones + 1 if ds else 0
        return ds


class Descrambler:
    def __init__(self) -> None:
        self.reg = [0] * 17

    def step(self, ds: int) -> int:
        do = ds ^ self.reg[13] ^ self.reg[16]
        self.reg = [ds] + self.reg[:-1]
        return do


class _Shaper:
    """Root raised cosine (75%) pulse shaping at 600 baud on the 8 kHz grid, symbol by symbol (exact timing)."""

    def __init__(self, beta: float = 0.75, span: int = 6):
        self.span = span
        self.tables = []
        for p in range(3):                                # symbol centres repeat every 3 symbols = 40 samples
            c = p * T
            lo, hi = math.ceil(c - span * T), math.floor(c + span * T)
            n = np.arange(lo, hi + 1)
            self.tables.append((lo, rrc((n - c) / FS, 1.0 / BAUD, beta)))
        self.buf = np.zeros(0, dtype=complex)            # baseband not yet emitted; index 0 = sample ``self.base``
        self.base = 0
        self.k = 0                                       # next symbol index

    def push(self, a: complex) -> None:
        lo, w = self.tables[self.k % 3]
        start = (self.k // 3) * 40 + lo - self.base
        need = start + len(w)
        if need > len(self.buf):
            self.buf = np.concatenate([self.buf, np.zeros(need - len(self.buf) + 64, dtype=complex)])
        s = max(start, 0)
        self.buf[s:start + len(w)] += a * w[s - start:]
        self.k += 1

    def final_through(self) -> int:
        """First sample index that later symbols can still change."""
        return int((self.k * T) - self.span * T) - 1

    def take(self, n: int) -> np.ndarray:
        if len(self.buf) < n:
            self.buf = np.concatenate([self.buf, np.zeros(n - len(self.buf), dtype=complex)])
        out, self.buf = self.buf[:n], self.buf[n:]
        self.base += n
        return out


class V22Modem(V42Layer):
    is_v32 = False
    is_v22 = True
    capture_detect_bits = False

    def __init__(self, calling_party: bool, bit_rate: int = 1200, bell212a: bool = True, guard_tone: int = 0,
                 tx_power_dbm0: float = -13.5, rms: float | None = None, ans_tone_ms: float = 0.0):
        self.calling = calling_party
        self.target_rate = 2400 if bit_rate >= 2400 else 1200
        self.bell212a = bell212a
        self.events: list[tuple[str, int]] = []
        self.connected = False
        self._rate = 1200                                    # line rate once the handshake has finished
        self.fc_tx, self.fc_rx = (LOW, HIGH) if calling_party else (HIGH, LOW)
        self.rms = rms if rms is not None else _ZERO_DBM0_RMS * 10 ** (tx_power_dbm0 / 20.0)
        self._shaper = _Shaper()
        self._tx_n = 0                                       # samples emitted
        self._tx_sym = 0                                     # symbols generated
        self._tx_mode = "silence"
        self._tx_rate_now = 1200
        self._tx_quadrant = 1
        self._scr = Scrambler()
        self._s1_phase = 0
        self._actions: list[tuple[int, object]] = []         # (tx symbol index, callable)
        self._app_bits: deque[int] = deque()
        self._deframe = _Framer()
        self._rx_bytes = bytearray()
        self._discard_batches = 0
        self._rx_samples = 0
        self._connected_at = 0
        self._v42_setup()
        self._rb = _Rx(self.fc_rx)
        self._desc = Descrambler()
        self._carrier = False
        self._ph = 0.0
        self._freq = 0.0
        self._agc = 1.0
        self._rx_quadrant = 1
        self._qam_from: int | None = None                    # rx symbol index from which 16-QAM decisions are made
        self._n11 = self._alt = self._ones = self._zeros = 0
        self._prev_dibit: int | None = None
        self._s1_active = False
        self._qam_ones = 0
        self._state = "wait"
        self._rx_ready = False
        try:
            self._bis_settle = max(0.0, float(os.environ.get("V22_BIS_SETTLE_S", "0")))
        except ValueError:
            self._bis_settle = 0.0
        self._hold_failed = False
        self._mute_tx = os.environ.get("V22_TEST_MUTE_TX") == "1"            # test hook: never transmit (forces a failed attempt)
        self._rx_buf: list[int] = []
        if calling_party:
            self._set_tx("silence")
        else:
            if ans_tone_ms:
                self._set_tx("ans")
                self._at(ans_tone_ms, lambda: self._set_tx("unscr1"))
            else:
                self._set_tx("unscr1")

    # -- surface shared with the SpanDSP engine -------------------------------------------------------------------
    @property
    def bit_rate(self) -> int:
        return self._rate

    @property
    def rx_signal_power_dbm0(self) -> float:
        p = self._rb.power
        return 10 * math.log10(p / _ZERO_DBM0_POWER) if p > 0 else -99.0

    @property
    def connected_for_seconds(self) -> float:
        return 0.0 if not self.connected else (self._rx_samples - self._connected_at) / FS

    @property
    def state(self) -> str:
        """The DCE reads ``"failed"`` as "the line never trained" and ends a call that has not yet been connected."""
        return "failed" if self._hold_failed else ""

    def _bis_held(self) -> bool:
        """True while the 2400 link is still inside the opt-in ``V22_BIS_SETTLE_S`` hold, i.e. CONNECT has not been announced."""
        return (self._bis_settle > 0.1 and self._rate >= 2400 and self.connected_for_seconds < self._bis_settle)

    def is_settled(self) -> bool:
        # A short hold after training so the host's CONNECT is out before the peer's first data bits arrive
        # (they start 765 ms after training at 1200 bit/s, 200 ms after at 2400).
        hold = 0.3 if self._rate < 2400 else 0.1
        if self._rate >= 2400 and self.calling and self._bis_settle > 0:
            hold = max(hold, self._bis_settle)           # opt-in: see ``_bis_held``
        return self.connected and not self._hold_failed and self._rate >= self.target_rate and self.connected_for_seconds >= hold

    def feed_bytes(self, data: bytes) -> None:
        self.v42_feed_bytes(data)

    def pending_bits(self) -> int:
        return len(self._app_bits) + self.v42_pending_bits()

    def feed_bits(self, bits) -> None:
        self._app_bits.extend(1 if b else 0 for b in bits)

    def fill_missing(self, count: int) -> None:
        for z in self._rb.feed(np.zeros(count)):
            self._on_symbol(z)

    def close(self) -> None:
        self._actions.clear()

    # V42Layer hooks
    def _plain_tx_bit(self) -> int:
        return self._app_bits.popleft() if self._app_bits else 1

    def _plain_feed_bytes(self, data: bytes) -> None:
        for v in data:
            self._app_bits.extend([0] + [(v >> i) & 1 for i in range(8)] + [1])

    def _plain_feed_marks(self, n: int) -> None:
        self._app_bits.extend([1] * n)

    def _plain_rx_bits(self, bits) -> bytes:
        return self._deframe.push(list(bits))

    def _event(self, name: str, code: int = 0) -> None:
        self.events.append((name, code))

    # -- transmit ------------------------------------------------------------------------------------------------
    def _at(self, ms: float, fn) -> None:
        """Run ``fn`` when the transmitter reaches the symbol that is ``ms`` from the one being generated now."""
        self._actions.append((self._tx_sym + int(round(ms * BAUD / 1000.0)), fn))
        self._actions.sort(key=lambda a: a[0])

    def _set_tx(self, mode: str, rate: int = 1200) -> None:
        self._tx_mode, self._tx_rate_now = mode, rate

    def _next_symbol(self) -> complex:
        while self._actions and self._actions[0][0] <= self._tx_sym:
            self._actions.pop(0)[1]()
        self._tx_sym += 1
        mode, rate = self._tx_mode, self._tx_rate_now
        if mode == "silence":
            return 0j
        if mode == "ans":                                            # V.25 answer tone, a 2100 Hz tone in this channel
            return 0j
        if mode == "unscr1":
            bits = [1, 1]
        elif mode == "s1":
            bits = [0, 0] if self._s1_phase == 0 else [1, 1]
            self._s1_phase ^= 1
        elif mode == "scr1":
            bits = [self._scr.step(1) for _ in range(4 if rate >= 2400 else 2)]
        else:                                                         # user data
            bits = [self._scr.step(self.v42_tx_bit()) for _ in range(4 if rate >= 2400 else 2)]
        change = _CHANGE[(bits[0] << 1) | bits[1]]
        self._tx_quadrant = (self._tx_quadrant - 1 + change) % 4 + 1
        if rate >= 2400 and mode in ("scr1", "data"):
            return complex(*_ELEMENTS[self._tx_quadrant][(bits[2] << 1) | bits[3]])
        return _DPSK_POINT[self._tx_quadrant]

    def tx_samples(self, count: int) -> list[int]:
        sh = self._shaper
        while sh.final_through() < self._tx_n + count:
            sh.push(self._next_symbol())
        base = sh.take(count)
        n = self._tx_n + np.arange(count)
        self._tx_n += count
        # every constellation here has an average symbol power of 10: passband rms = sqrt(P / 2) * scale
        out = (base * np.exp(2j * math.pi * self.fc_tx * n / FS)).real * (self.rms / math.sqrt(5.0))
        if self._tx_mode == "ans":
            out = out + self.rms * math.sqrt(2) * np.sin(2 * math.pi * 2100.0 * n / FS)
        out = np.clip(np.round(out), -32768, 32767).astype(int).tolist()
        return [0] * count if self._mute_tx else out

    # -- receive -------------------------------------------------------------------------------------------------
    def feed_pcm(self, pcm: bytes) -> bytes:
        self._rx_n = getattr(self, "_rx_n", 0) + len(pcm) // 2
        x = np.frombuffer(pcm, dtype="<i2").astype(float)
        self._rx_samples += len(x)
        for z in self._rb.feed(x):
            self._on_symbol(z)
        if not self._rb.carrier:                       # no symbols arrive once the carrier is gone
            self._carrier_gone()
        data = bytes(self._rx_bytes)
        self._rx_bytes.clear()
        if self.v42_mode in ("detect", "lapm"):
            data += self.v42_read()
        return data

    def _carrier_gone(self) -> None:
        if self._carrier:
            self._carrier = False
            self._event("CARRIER_DOWN", -1)
            self._n11 = self._alt = self._ones = self._zeros = 0
            self._prev_dibit, self._s1_active = None, False
        if self.connected and self._rb.carrier_lost_for > int(0.5 * FS):
            self.connected = False

    def _on_symbol(self, z: complex) -> None:
        rb = self._rb
        if not rb.carrier:
            self._carrier_gone()
            return
        if not self._carrier:
            self._carrier = True
            self._event("CARRIER_UP", -2)
        first2, bits = self._decide(z)
        for b in bits:
            self._on_bit(b)
        self._after_symbol(first2, len(bits))

    def _qam_active(self) -> bool:
        return self._qam_from is not None and self._rb.symbols >= self._qam_from

    def _decide(self, z: complex):
        """Differential 4-phase decision (1200) or 16-QAM decision (2400); returns (first dibit, scrambled bits)."""
        y = z * complex(math.cos(-self._ph), math.sin(-self._ph)) / self._agc
        if not self._qam_active():
            ang = math.atan2(y.imag, y.real)
            q = int(round((ang - _DPSK_ANGLE) / (math.pi / 2))) % 4 + 1
            err = (ang - (_DPSK_ANGLE + (q - 1) * math.pi / 2) + math.pi) % (2 * math.pi) - math.pi
            self._agc += 0.05 * ((abs(z) / math.sqrt(10.0)) - self._agc)
            self._track(err)
            first2 = _CHANGE_INV[(q - self._rx_quadrant) % 4]
            self._rx_quadrant = q
            return first2, [first2 >> 1, first2 & 1]
        idx = int(np.argmin(np.abs(_PTS - y)))
        q, last2, point = POINTS[idx]
        e = y * point.conjugate()
        self._track(math.atan2(e.imag, e.real))
        self._agc += 0.002 * ((abs(z) / abs(point)) - self._agc)
        first2 = _CHANGE_INV[(q - self._rx_quadrant) % 4]
        self._rx_quadrant = q
        return first2, [first2 >> 1, first2 & 1, last2 >> 1, last2 & 1]

    def _track(self, err: float) -> None:
        self._freq += 0.002 * err
        self._ph += 0.15 * err + self._freq

    def _on_bit(self, scrambled: int) -> None:
        do = self._desc.step(scrambled)
        if do:
            self._ones += 1
            self._zeros = 0
        else:
            self._zeros += 1
            self._ones = 0
        if self._qam_active():
            self._qam_ones = self._qam_ones + 1 if do else 0
        else:
            self._qam_ones = 0
        if self._rx_ready:
            if self._discard_batches:
                self._discard_batches -= 1
                if self._discard_batches == 0:
                    self._rx_buf = []
                return
            self._rx_buf.append(do)
            if len(self._rx_buf) >= 16:
                bits, self._rx_buf = self._rx_buf, []
                if self.v42_mode in ("detect", "lapm"):
                    self.v42_rx_bits(bits)
                else:
                    self._rx_bytes += self._plain_rx_bits(bits)

    # -- handshake (§6.3.1) ------------------------------------------------------------------------------------------
    def _after_symbol(self, first2: int, nbits: int) -> None:
        was_s1 = self._s1_active
        self._n11 = self._n11 + 1 if (first2 == 0b11 and nbits == 2) else 0
        prev = self._prev_dibit
        if prev is not None and first2 != prev and {first2, prev} == {0b00, 0b11}:
            self._alt += 1
        else:
            self._alt = 0
        self._prev_dibit = first2
        self._s1_active = self._alt >= 12
        s1_ended = was_s1 and not self._s1_active
        sym_ms = lambda ms: int(round(ms * BAUD / 1000.0))
        st = self._state
        if self.calling:
            if st == "wait" and self._n11 >= sym_ms(155):
                self._state = "hs1"
                self._at(456, lambda: self._set_tx("s1" if self.target_rate >= 2400 else "scr1"))
                if self.target_rate >= 2400:
                    self._at(456 + 100, lambda: self._set_tx("scr1"))
            elif st == "hs1":
                if s1_ended and self.target_rate >= 2400:
                    self._enter_bis()
                elif self._ones >= sym_ms(270) * 2 and self._n11 < 20 and not self._s1_active:
                    self._enter_v22()
        else:
            if st == "wait":
                if s1_ended and self.target_rate >= 2400:
                    self._set_tx("s1")
                    self._at(100, lambda: self._set_tx("scr1"))
                    self._enter_bis()
                elif (self._ones >= sym_ms(270) * 2 or self._zeros >= sym_ms(270) * 2) and self._n11 < 20 \
                        and not self._s1_active:
                    self._set_tx("scr1")
                    self._enter_v22()
        mode = os.environ.get("V22_S1_FALLBACK", "restart")
        if (self._state == "bis" and self.calling and s1_ended and self._rx_ready and self._rate >= 2400
                and mode != "0" and not self._hold_failed):
            if self._bis_held():
                # CONNECT is still held back: end the attempt (the DCE reports NO CARRIER and the caller can redial at 1200)
                self._hold_failed = True
                self._event("BIS_REJECTED", -7)
            elif mode == "inplace":
                self._fall_back_to_1200()
            else:
                self._cold_restart_1200()
        if self._state == "bis" and not self._rx_ready and self._qam_active() and self._qam_ones >= 32:
            self._rx_ready = True                                              # 32 consecutive scrambled ones at 2400
            self._discard_batches = 60                                         # discard ~0.4 s of the stream while it syncs
            self._rate = 2400
            self._ready()

    def _mark(self, name: str, sample: int) -> None:
        """Opt-in diagnostic: append a sample-clock timing mark (EL_V22_TIMING_FILE). Changes nothing else."""
        path = os.environ.get("EL_V22_TIMING_FILE")
        if path:
            with open(path, "a") as fh:
                fh.write(json.dumps({"mark": name, "sample": int(sample), "calling": bool(self.calling)}) + "\n")

    def _enter_bis(self) -> None:
        self._state = "bis"
        self._mark("S1_END_RX", getattr(self, "_rx_n", 0))
        self._qam_from = self._rb.symbols + int(round(0.450 * BAUD))              # 450 ms after the end of S1
        self._at(600, lambda: (self._mark("SCR1_2400_TX", self._tx_n), self._set_tx("scr1", 2400)))  # 600 ms after the end of S1
        self._at(600 + 200, lambda: self._set_tx("data", 2400))                   # data after 200 ms of scrambled ones

    def _cold_restart_1200(self) -> None:
        """Same trigger as ``_fall_back_to_1200``, but go quiet and start over as a plain 1200 caller: no S1, transmit only after
        the answerer's unscrambled ones are heard again (the sequence NIST ACTS completed in nist-4). The in-place switch left
        ACTS in its V.22bis state (nist-5). ``V22_S1_FALLBACK=inplace`` selects the in-place switch, ``0`` disables both.
        ``connected`` stays True so the host does not see a carrier loss; ``TRAINING_SUCCEEDED`` is raised again on success."""
        self._event("RATE_RESTART_1200", -6)
        self.target_rate = 1200
        self._actions.clear()
        self._qam_from, self._qam_ones = None, 0
        self._state, self._rate = "wait", 1200
        self._rx_ready = False
        self._rx_buf = []
        self._n11 = self._alt = self._ones = self._zeros = 0
        self._s1_active, self._prev_dibit = False, None
        self._set_tx("silence")

    def _fall_back_to_1200(self) -> None:
        """The answerer sent S1 again after we had locked at 2400: it restarted the handshake because it did not accept our
        2400 signal (NIST ACTS does this every ~2.1 s, lab/V34_NIST_ACTS_CALLS_2026_10_07.md). The retrain and rate-change
        sequences of §6.4-6.6 are not implemented, so answer as a V.22 caller: DPSK receive, scrambled ones at 1200, data
        765 ms later (§6.3.1.2). ``V22_S1_FALLBACK=0`` disables this."""
        self._event("RATE_FALLBACK_1200", -5)
        self.target_rate = 1200
        self._actions.clear()
        self._qam_from, self._qam_ones = None, 0
        self._state, self._rate = "v22", 1200
        self._rx_buf = []
        self._discard_batches = 60
        self._set_tx("scr1", 1200)
        self._at(765, lambda: self._set_tx("data", 1200))

    def _enter_v22(self) -> None:
        self._state = "v22"
        self._rate = 1200
        self._rx_ready = True
        self._discard_batches = 60                                              # drop engine warm-up bits at rate 1200
        self._at(765, lambda: self._set_tx("data", 1200))                         # 765 ms after the handshake (§6.3.1.2)
        self._ready()

    def _ready(self) -> None:
        self.connected = True
        self._connected_at = self._rx_samples
        self._event("TRAINING_SUCCEEDED", -4)


class _Rx:
    """Matched filter, symbol timing from the squared envelope's 600 Hz line, interpolation to symbol instants."""

    def __init__(self, fc: float, beta: float = 0.75, span: int = 6):
        self.fc = fc
        taps = int(2 * span * T) | 1
        t = (np.arange(taps) - taps // 2) / FS
        h = rrc(t, 1.0 / BAUD, beta)
        self.h = h / np.sqrt(np.sum(h ** 2))
        self.n = 0                                        # input samples consumed
        self.tail = np.zeros(len(self.h) - 1, dtype=complex)
        self.bb = np.zeros(0, dtype=complex)              # matched filter output, index 0 = absolute sample bb_base
        self.bb_base = 0
        self.symbols = 0
        self.t_next: float | None = None
        self.power = 0.0
        self.carrier = False
        self.up_at = 0                                    # absolute sample index where the carrier appeared
        self._quiet = 0
        self.carrier_lost_for = 0
        self.THRESH_ON, self.THRESH_OFF = 100.0, 30.0

    def feed(self, x: np.ndarray):
        n = self.n + np.arange(len(x))
        z = x * np.exp(-2j * math.pi * self.fc * n / FS)
        self.n += len(x)
        joined = np.concatenate([self.tail, z])
        full = np.convolve(joined, self.h, mode="valid")
        self.tail = joined[-(len(self.h) - 1):]
        start_abs = self.bb_base + len(self.bb)
        self.bb = np.concatenate([self.bb, full])
        self.power = 0.9 * self.power + 0.1 * (float(np.mean(x ** 2)) if len(x) else 0.0)
        e = np.convolve(np.abs(full) ** 2, np.ones(20) / 20, mode="same")
        if not self.carrier:
            idx = np.nonzero(e > self.THRESH_ON)[0]
            if len(idx):
                self.carrier = True
                self.up_at = start_abs + int(idx[0])
                self.t_next = None
                self._quiet = 0
        elif float(np.mean(np.abs(full) ** 2)) < self.THRESH_OFF:
            self._quiet += len(x)
            if self._quiet > FS // 25:
                self.carrier = False
        else:
            self._quiet = 0
        self.carrier_lost_for = 0 if self.carrier else self.carrier_lost_for + len(x)
        return self._symbols()

    def _timing(self) -> float | None:
        first = max(self.up_at + len(self.h) // 2, self.bb_base)
        last = self.bb_base + len(self.bb)
        first = max(first, last - int(0.25 * FS))
        if last - first < 240:                                           # need ~30 ms of signal for a first estimate
            return None
        e = np.abs(self.bb[first - self.bb_base:]) ** 2
        n = np.arange(first, last)
        E = np.sum(e * np.exp(-2j * math.pi * BAUD * n / FS))
        return (-np.angle(E) / (2 * math.pi * BAUD)) * FS % T

    def _symbols(self):
        out = []
        if not self.carrier:
            self.t_next = None
            return out
        tau = self._timing()
        if tau is None:
            return out
        end = self.bb_base + len(self.bb)
        if self.t_next is None:
            k = math.ceil((self.up_at + len(self.h) // 2 - tau) / T)
            self.t_next = tau + k * T
        else:
            ph = (self.t_next - tau + T / 2) % T - T / 2                 # smooth timing correction
            self.t_next -= 0.15 * ph
        while self.t_next + 8 < end:
            i = int(math.floor(self.t_next)) - self.bb_base
            frac = self.t_next - math.floor(self.t_next)
            if i - 5 < 0:
                self.t_next += T
                continue
            m = np.arange(-5, 7)
            d = m - frac
            w = np.sinc(d) * (0.5 + 0.5 * np.cos(math.pi * d / 6))
            out.append(complex(np.dot(self.bb[i + m], w)))
            self.symbols += 1
            self.t_next += T
        keep = int(0.6 * FS)
        if len(self.bb) > keep:
            cut = len(self.bb) - keep
            self.bb = self.bb[cut:]
            self.bb_base += cut
        return out
