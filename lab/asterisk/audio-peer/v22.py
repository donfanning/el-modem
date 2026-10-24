"""ITU-T V.22 / V.22bis and Bell 212A compatibility profile, implemented with 600/1200 bit/s DPSK and 2400 bit/s 16-QAM.
The Bell 212A profile has scoped Smart Link interoperability evidence; Bell System conformance is not established.

The V.22/V.22bis engine implements the 600-baud waveforms from the 1988
Recommendations (V.22bis text, which also contains the V.22 compatible 1200 bit/s mode and handshake).

``V22Modem`` has the surface of the SpanDSP-backed class it replaces (spandsp_v22.V22bisModem): ``tx_samples`` /
``feed_pcm`` at 8 kHz, ``feed_bytes``, ``pending_bits``, ``connected``, ``is_settled``, ``bit_rate``, ``events`` as
(name, code) pairs and the V.42 / MNP hooks of ``V42Layer``. Calling modems transmit in the low channel (1200 Hz) and receive
in the high one (2400 Hz); answering modems the other way round.

Implemented for V.22/V.22bis: scrambler 1 + x^-14 + x^-17 with the 64-ones guard (§5), one-bit phase-change encoding at 600 bit/s,
quadrant-change encoding at 1200 bit/s and the 16-point constellation of Figure 2 (§2.5), square-root raised cosine 75% shaping
(§2.4), the GSTN handshake of §6.3.1 for 600/1200 and V.22bis 2400 bit/s, including S1 where applicable, and the 155, 456, 100,
600, 450, 200, 270 and 765 ms timings. Receiver: matched
filter, symbol timing from the 600 Hz line of the squared envelope, differential 4-phase detection for the handshake and
decision-directed carrier tracking for 16-QAM. The high-channel answer transmitter supports the 1800-Hz guard tone at
the specified -6 dB level and an optional 550-Hz tone with a declared -6 dB equipment policy; Hayes ``&G`` selects these
options. The receive path implements the §6.5 data clamp across carrier dropout and its 100-ms post-return retrain
observation window. An S1-triggered §6.4 response now emits the 1200-bit/s pattern, returns to 2400 bit/s and waits for
32 descrambled ones. The optional §6.6 rate-change sequence is implemented for
manual V.22bis rate changes and routed through online `ATO4`/`ATO5`; paired
DSP/DCE-control and production RTP tests cover 2400→1200→2400 with exact raw
payload afterward. Paired-DSP coverage also verifies LAPM continuity and
payload after a 2400→1200 change, and unanswered-request retry is tested.
Independent-peer rate-change qualification, malformed-response handling,
circuit-111 initiation and loss-of-equalization initiation remain open.
Loop tests and an adaptive
equalizer (a linear channel at 600 baud does not need one in the lab; real lines may) are also not implemented. The separate
``bell212a`` profile emits/detects the 2225 Hz answer mark and follows a
Bell-specific 1200 bit/s startup sequence around the shared DPSK datapath.
Scoped Smart Link tests cover both calling and answering interoperability;
broader peer, channel-impairment, and Bell System conformance remain open. The
Bell path supports selectable 9/10-bit no-parity characters and 11/12-bit parity
characters, with seven-bit masking in 9-bit mode and E/O/mark/space parity on
11/12-bit formats. Its raw-bit Mode-B stop-bit insertion/deletion model is
driven by an explicit DTE baud parameter and includes BREAK normalization and
recovery. These are tested over internal modem pairs and the production DCE/RTP
path.
The live POSIX PTY path supplies its configured nominal baud, including Linux
custom termios rates; this is metadata, not a measurement of the DTE's actual
clock. TCP and Windows pipe paths do not supply baud metadata. LAPM's
byte-oriented path bypasses raw async framing. Physical BREAK timing and
independent peer testing of DTE-rate mismatch, long-idle
robustness, primary Bell timing limits, and the 300 bit/s Bell 103 fallback
remain unqualified/open.

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


class _BellAsyncDeframer:
    """Recover Bell Mode-B characters when their stop bit is stretched or deleted.

    The transmitter may add mark bits between characters or omit a stop bit
    under the Mode-B rate adaptation rules. Start/data positions remain
    fixed; a mark is a stop bit and a zero after the data field is the next
    character's start bit (the previous stop was deleted).
    """

    def __init__(self, data_bits: int, *, parity: str | None = None, character_bits: int | None = None):
        if data_bits not in (7, 8):
            raise ValueError("Bell async framing supports seven or eight data bits")
        if parity not in (None, "E", "O", "M", "S"):
            raise ValueError("Bell async parity must be E, O, M, S or disabled")
        self.data_bits = data_bits
        self.parity = parity
        self.character_bits = character_bits or (data_bits + 2 + (parity is not None))
        self.break_threshold = 2 * self.character_bits + 3
        self._state = "hunt"
        self._bits: list[int] = []
        self._zero_run = 0
        self._pending_bytes = bytearray()
        self._break_events: deque[int] = deque()
        self._in_break = False
        self._frame_valid = True

    def _parity_bit(self) -> int:
        if self.parity == "M":
            return 1
        if self.parity == "S":
            return 0
        odd = sum(self._bits) & 1
        return odd if self.parity == "E" else odd ^ 1

    def push(self, bits) -> bytes:
        out = bytearray()
        for bit in bits:
            bit = 1 if bit else 0
            if self._in_break:
                if bit:
                    self._break_events.append(self._zero_run)
                    self._in_break = False
                    self._state = "hunt"
                    self._zero_run = 0
                else:
                    self._zero_run += 1
                continue
            if bit:
                out.extend(self._pending_bytes)
                self._pending_bytes.clear()
                self._zero_run = 0
            else:
                self._zero_run += 1
                if self._zero_run >= self.break_threshold:
                    # Bytes tentatively framed inside a continuous start-polarity
                    # run are a BREAK, not NUL characters.
                    self._pending_bytes.clear()
                    self._in_break = True
                    self._state = "hunt"
                    self._bits.clear()
                    continue
            if self._state == "hunt":
                if bit == 0:
                    self._bits = []
                    self._frame_valid = True
                    self._state = "data"
            elif self._state == "data":
                self._bits.append(bit)
                if len(self._bits) == self.data_bits:
                    self._state = "parity" if self.parity is not None else "stop"
            elif self._state == "parity":
                self._frame_valid = bit == self._parity_bit()
                self._state = "stop"
            elif self._state == "stop":
                if self._frame_valid:
                    self._pending_bytes.append(sum(value << index for index, value in enumerate(self._bits)))
                if bit:
                    out.extend(self._pending_bytes)
                    self._pending_bytes.clear()
                    self._state = "mark"
                else:
                    # A deleted stop is followed immediately by the next
                    # character's start. Reuse this zero as that start bit.
                    self._bits = []
                    self._state = "data"
            elif bit == 0:
                self._bits = []
                self._state = "data"
        return bytes(out)

    def take_break_events(self) -> list[int]:
        events = list(self._break_events)
        self._break_events.clear()
        return events

    def reset_partial(self) -> None:
        """Discard an incomplete async character at a physical carrier gap."""
        self._state = "hunt"
        self._bits.clear()
        self._zero_run = 0
        self._pending_bytes.clear()
        self._in_break = False
        self._frame_valid = True


class Scrambler:
    """1 + x^-14 + x^-17, self-synchronising; with the 64 consecutive ones guard of §5.1."""

    def __init__(self, guard: bool = True) -> None:
        self.reg = [0] * 17
        self.ones = 0
        self.guard = guard

    def step(self, di: int) -> int:
        if self.guard and self.ones >= 64:                 # §5.1: invert the next input to break scrambler lock-up
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

    def __init__(self, calling_party: bool, bit_rate: int = 1200, profile: str = "v22", guard_tone: int = 0,
                 tx_power_dbm0: float = -13.5, rms: float | None = None, ans_tone_ms: float = 0.0,
                 bell212_char_bits: int = 10, bell212_parity: str = "E",
                 bell212_dte_baud: int = 1200, min_rate: int | None = None,
                 bell212_leased_line: bool = False):
        self.calling = calling_party
        if profile not in ("v22", "v22bis", "bell212a"):
            raise ValueError(f"unknown V.22-family profile: {profile!r}")
        if profile == "bell212a" and bit_rate != 1200:
            raise ValueError("Bell 212A profile supports 1200 bit/s only")
        if bell212_char_bits not in (9, 10, 11, 12):
            raise ValueError("Bell 212A async character length must be 9, 10, 11 or 12 bits")
        if bell212_parity not in ("E", "O", "M", "S"):
            raise ValueError("Bell 212A parity must be E, O, M or S")
        if (profile == "bell212a" and
                (isinstance(bell212_dte_baud, bool) or not isinstance(bell212_dte_baud, int)
                 or not 1170 <= bell212_dte_baud <= 1212)):
            raise ValueError("Bell 212A Mode-B DTE rate must be 1170..1212 bit/s")
        self.profile = profile
        if bell212_leased_line and profile != "bell212a":
            raise ValueError("Bell 212A leased-line startup requires the Bell 212A profile")
        self.bell212_leased_line = bool(bell212_leased_line)
        if min_rate is None:
            min_rate = 300 if profile == "bell212a" else (bit_rate if profile == "v22" else 1200)
        if isinstance(min_rate, bool) or not isinstance(min_rate, int) or min_rate < 0:
            raise ValueError("V.22 minimum rate must be a non-negative integer")
        if min_rate > bit_rate:
            raise ValueError("V.22 minimum rate exceeds the selected modem rate")
        self.min_rate = min_rate
        if guard_tone not in (0, 1, 2):
            raise ValueError("V.22 guard tone must be 0 (off), 1 (550 Hz), or 2 (1800 Hz)")
        self.guard_tone = int(guard_tone) if profile in ("v22", "v22bis") else 0
        self.bell212_char_bits = bell212_char_bits if profile == "bell212a" else 10
        self.bell212_parity = bell212_parity if profile == "bell212a" else "E"
        self._dte_data_bits = (7 if self.bell212_char_bits == 9 else 8) if profile == "bell212a" else 8
        self._dte_parity = self.bell212_parity if profile == "bell212a" and self.bell212_char_bits >= 11 else None
        self.bell212_dte_baud = bell212_dte_baud if profile == "bell212a" else 1200
        self._bell_tx_target_numerator = 0
        self._bell_tx_emitted_bits = 0
        self.target_rate = bit_rate
        self.events: list[tuple[str, int]] = []
        self.connected = False
        self._rate = 1200                                    # line rate once the handshake has finished
        self._rx_600_switch_sample: int | None = None
        self._rx_600_active = bit_rate == 600
        self.fc_tx, self.fc_rx = (LOW, HIGH) if calling_party else (HIGH, LOW)
        self.rms = rms if rms is not None else _ZERO_DBM0_RMS * 10 ** (tx_power_dbm0 / 20.0)
        self._shaper = _Shaper()
        self._tx_n = 0                                       # samples emitted
        self._tx_sym = 0                                     # symbols generated
        self._tx_mode = "silence"
        self._tx_rate_now = 600 if bit_rate == 600 else 1200
        self._tx_quadrant = 1
        self._scr = Scrambler(guard=profile != "bell212a")
        self._s1_phase = 0
        self._actions: list[tuple[int, object]] = []         # (tx symbol index, callable)
        self._app_bits: deque[int] = deque()
        self._deframe = (_BellAsyncDeframer(self._dte_data_bits, parity=self._dte_parity,
                                             character_bits=self.bell212_char_bits) if profile == "bell212a"
                         else _Framer(data_bits=self._dte_data_bits))
        self._rx_bytes = bytearray()
        # Raw V.22bis DTE output is held for the S1 detector's qualification
        # span, allowing a later-qualified retrain request to discard the
        # preceding symbols rather than leak them as terminal bytes.
        self._s1_rx_holdback: deque[tuple[int, bytes]] = deque()
        self._discard_batches = 0
        self._rx_samples = 0
        self._connected_at = 0
        self._v42_setup()
        self._rb = _Rx(self.fc_rx)
        self._desc = Descrambler()
        self._carrier = False
        # V.22bis §6.5: after a received-line-signal dropout, clamp received
        # DTE data while the carrier returns and for the 100-ms retrain
        # detection window.  None means the post-dropout window is not active.
        self._rx_data_clamped = False
        self._rx_clamp_until_symbol: int | None = None
        self._rx_retrain_pending = False
        self._ph = 0.0
        self._freq = 0.0
        self._agc = 1.0
        self._rx_quadrant = 1
        self._qam_from: int | None = None                    # rx symbol index from which 16-QAM decisions are made
        self._n11 = self._alt = self._ones = self._zeros = 0
        self._raw_ones = 0
        self._prev_dibit: int | None = None
        self._s1_active = False
        self._qam_ones = 0
        self._retrain_attempts = 0
        # Optional V.22bis §6.6 rate signalling. The state is deliberately
        # separate from the §6.4 retrain state: an S1 followed by 32 equal
        # descrambled rate dibits is a rate request; S1 followed by scrambled
        # ones at the already-active rate is a retrain.
        self._rate_control = None
        self._rate_dibit_bits: list[int] = []
        self._rate_dibit_value: int | None = None
        self._rate_dibit_run = 0
        self._rate_s1_start_symbol: int | None = None
        self._qam_until: int | None = None
        self._rate_change_attempts = 0
        self._state = "wait"
        self._rx_ready = False
        try:
            self._bis_settle = max(0.0, float(os.environ.get("V22_BIS_SETTLE_S", "0")))
        except ValueError:
            self._bis_settle = 0.0
        self._hold_failed = False
        self._mute_tx = os.environ.get("V22_TEST_MUTE_TX") == "1"            # test hook: never transmit (forces a failed attempt)
        self._rx_buf: list[int] = []
        self._answer_tone_samples: deque[float] = deque(maxlen=320)
        self._answer_tone_pending: list[float] = []
        self._answer_tone_run = 0
        self._bell_answer_tone_present = False
        self._bell_answer_tone_start = 0
        self._bell_answer_tone_end: int | None = None
        self._bell_answer_tone_gate_until: int | None = None
        self._bell103_fallback_engine = None
        self._bell103_fallback = False
        self._bell_leased_recovering = False
        self._bell103_mark_samples = 0
        if profile == "bell212a" and not calling_party:
            # Bell 212A is dual-speed on answer: keep an independent Bell 103
            # receiver warm while the high-speed answer mark is sent. It is
            # selected only by the qualified 1270-Hz low-speed calling mark.
            from bell103_engine import Bell103Engine

            self._bell103_fallback_engine = Bell103Engine(
                False, amplitude=self.rms * math.sqrt(2.0))
        if calling_party:
            if profile == "bell212a":
                if self.bell212_leased_line:
                    # A dedicated circuit has no dial/answer-tone event. The
                    # configured originate side establishes the low-channel
                    # carrier; the answer side qualifies it and returns its
                    # high-channel mark.
                    self._state = "bell_leased_call_mark"
                    self._set_tx("scr1", 1200)
                else:
                    self._state = "bell_call_wait_tone"
                    self._set_tx("silence")
            else:
                self._set_tx("silence")
        else:
            if profile == "bell212a":
                if self.bell212_leased_line:
                    # On a leased circuit the answer side waits for the
                    # configured originate side instead of emitting switched
                    # call's 2225-Hz answer tone.
                    self._state = "bell_leased_answer_wait_mark"
                    self._set_tx("silence")
                else:
                    # Bell 212A's billing silence precedes the 2225 Hz answer mark.
                    self._state = "bell_answer_silence"
                    self._set_tx("silence")
                    self._bell_answer_tone_start = 2 * FS
                    self._at(2000, lambda: (self._set_tx("bell_tone"), setattr(self, "_state", "bell_answer_tone")))
            elif ans_tone_ms:
                self._set_tx("ans")
                self._at(ans_tone_ms, lambda: self._set_tx("unscr1"))
            else:
                self._set_tx("unscr1")

    # -- surface shared with the SpanDSP engine -------------------------------------------------------------------
    @property
    def bit_rate(self) -> int:
        return 300 if self._bell103_fallback else self._rate

    @property
    def bell103_fallback(self) -> bool:
        """True after Bell 212A answer mode has selected 300-bit/s Bell 103."""
        return self._bell103_fallback

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
        if self._bell103_fallback:
            # The UDS low-speed sequence asserts answer-side ready after the
            # 155-ms carrier qualification; imposing the 1200-mode post-train
            # delay would lose the caller's first data characters.
            return self.connected and not self._hold_failed
        # A short hold after training so the host's CONNECT is out before the peer's first data bits arrive
        # (they start 765 ms after training at 1200 bit/s, 200 ms after at 2400).
        hold = 0.3 if self._rate < 2400 else 0.1
        if self._rate >= 2400 and self.calling and self._bis_settle > 0:
            hold = max(hold, self._bis_settle)           # opt-in: see ``_bis_held``
        if self.target_rate == 600 and not (self._rx_600_active and self._tx_mode == "data"
                                            and self._tx_rate_now == 600):
            return False
        if self.profile == "v22bis" and self._state == "bis":
            # A completed §6.6 downshift remains a connected, settled call even
            # though the original negotiated ceiling was 2400 bit/s.
            return (self.connected and self._rx_ready and not self._hold_failed
                    and self.connected_for_seconds >= hold)
        return self.connected and not self._hold_failed and self._rate >= self.target_rate and self.connected_for_seconds >= hold

    @property
    def dte_transmit_ready(self) -> bool:
        """Whether circuit 106 may release queued DTE data in this mode."""
        if self.profile not in ("v22", "v22bis"):
            return True
        return (self._tx_mode == "data" and self._rx_ready and not self._rx_data_clamped
                and not self._hold_failed)

    @property
    def v42_data_ready(self) -> bool:
        """Whether this modem has entered the physical synchronous data phase.

        Bell 212A can assert ``connected`` while it is still sending the
        post-training mark interval. V.42 detection is meaningful only once
        the line bit stream is available; callers may pre-create the control
        function, but a peer test must not start its T400 clock at CONNECT.
        """
        return (self.connected and self._tx_mode == "data" and self._rx_ready
                and not self._rx_data_clamped and not self._hold_failed)

    def feed_bytes(self, data: bytes) -> None:
        self.v42_feed_bytes(data)

    def pending_bits(self) -> int:
        if self._bell103_fallback:
            return self._bell103_fallback_engine.pending_bits()
        return len(self._app_bits) + self.v42_pending_bits()

    def feed_bits(self, bits) -> None:
        self._app_bits.extend(1 if b else 0 for b in bits)

    def bell212_feed_break(self) -> bool:
        """Queue the Mode-B BREAK minimum as continuous start-polarity bits.

        UDS 212A-compatible spec §3.3 describes a break of M through 2M+3
        start-polarity bits as being normalized to 2M+3 bits, with longer
        breaks passed through. The serial API exposes BREAK as an event rather
        than its original duration, so emit the normative minimum.
        """
        if (self.profile != "bell212a" or self._bell103_fallback
                or self.v42_mode in ("detect", "lapm", "recovering")):
            return False
        count = 2 * self.bell212_char_bits + 3
        self._app_bits.extend([0] * count)
        self._app_bits.extend([1] * (2 * self.bell212_char_bits))
        return True

    def bell212_read_break_events(self) -> list[dict]:
        if self.profile != "bell212a":
            return []
        return [dict(type="break_indication", start_bits=count,
                     character_bits=self.bell212_char_bits)
                for count in self._deframe.take_break_events()]

    def fill_missing(self, count: int) -> None:
        for z in self._rb.feed(np.zeros(count)):
            self._on_symbol(z)

    def close(self) -> None:
        self._actions.clear()

    # V42Layer hooks
    def _plain_tx_bit(self) -> int:
        return self._app_bits.popleft() if self._app_bits else 1

    def _plain_feed_bytes(self, data: bytes) -> None:
        if self._bell103_fallback:
            self._bell103_fallback_engine.feed_bytes(data)
            return
        if self.profile == "bell212a":
            char_bits = self.bell212_char_bits
            fixed_bits = [0]
            for value in data:
                fixed_bits.extend((value >> bit) & 1 for bit in range(self._dte_data_bits))
                if self._dte_parity is not None:
                    ones = value.bit_count() & 1
                    parity_bit = {"E": ones, "O": ones ^ 1, "M": 1, "S": 0}[self._dte_parity]
                    fixed_bits.append(parity_bit)
                # V.22 uses 600 symbols/s with two coded bits per symbol,
                # hence the Bell channel's fixed 1200 bit/s line rate.
                self._bell_tx_target_numerator += char_bits * (2 * BAUD)
                target_bits = (
                    self._bell_tx_target_numerator + self.bell212_dte_baud // 2
                ) // self.bell212_dte_baud
                total_bits = target_bits - self._bell_tx_emitted_bits
                self._bell_tx_emitted_bits += total_bits
                stop_bits = total_bits - len(fixed_bits)
                if stop_bits < 0:
                    raise RuntimeError("Bell Mode-B rate adaptation underflowed the fixed character bits")
                self._app_bits.extend(fixed_bits)
                self._app_bits.extend([1] * stop_bits)
                fixed_bits = [0]
            return
        for v in data:
            self._app_bits.extend([0] + [(v >> i) & 1 for i in range(self._dte_data_bits)] + [1])

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
        if mode in ("ans", "bell_tone"):                              # pure V.25 / Bell answer tone over the audio output
            return 0j
        if mode == "unscr1":
            bits = [1, 1]
        elif mode == "s1":
            bits = [0, 0] if self._s1_phase == 0 else [1, 1]
            self._s1_phase ^= 1
        elif mode == "rate_dibit":
            dibit = self._rate_dibit(self._rate_control_target)
            bits = [self._scr.step((dibit >> 1) & 1), self._scr.step(dibit & 1)]
        elif rate == 600 and mode in ("unscr1", "scr1", "data"):
            if mode == "unscr1":
                bit = 1
            elif mode == "scr1":
                bit = self._scr.step(1)
            else:
                bit = self._scr.step(self.v42_tx_bit())
            change = 3 if bit else 1  # V.22 Table 1: 0 -> +90°, 1 -> +270°.
            self._tx_quadrant = (self._tx_quadrant - 1 + change) % 4 + 1
            return _DPSK_POINT[self._tx_quadrant]
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
        if self._bell103_fallback_engine is not None:
            # Advance its answer-mark phase even before selecting low speed so
            # the 2225-Hz FSK mark is already running when the selector flips.
            low_speed = self._bell103_fallback_engine.tx_samples(count)
            if self._bell103_fallback:
                return low_speed
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
        elif (self.guard_tone and not self.calling and self.profile in ("v22", "v22bis")
              and self._tx_mode not in ("silence", "bell_tone")):
            # V.22 §2.1 permits a national guard tone only on the high-channel
            # transmitter (answer mode). For 1800 Hz, §2.2 specifies -6 dB
            # relative to high-channel data power and constant total line
            # power across channels. The optional 550-Hz level is for further
            # study; this implementation uses the same explicit -6 dB policy.
            relative_power = 10.0 ** (-6.0 / 10.0)
            data_scale = 1.0 / math.sqrt(1.0 + relative_power)
            guard_hz = 550.0 if self.guard_tone == 1 else 1800.0
            guard_peak = self.rms * data_scale * math.sqrt(relative_power * 2.0)
            out = out * data_scale + guard_peak * np.sin(2 * math.pi * guard_hz * n / FS)
        if self.profile == "bell212a" and not self.calling and not self.bell212_leased_line:
            tone_end = self._bell_answer_tone_end if self._bell_answer_tone_end is not None else self._tx_n
            tone_mask = (n >= self._bell_answer_tone_start) & (n < tone_end)
            out = out + tone_mask * self.rms * math.sqrt(2) * np.sin(2 * math.pi * 2225.0 * n / FS)
        out = np.clip(np.round(out), -32768, 32767).astype(int).tolist()
        return [0] * count if self._mute_tx else out

    # -- receive -------------------------------------------------------------------------------------------------
    def feed_pcm(self, pcm: bytes) -> bytes:
        self._rx_n = getattr(self, "_rx_n", 0) + len(pcm) // 2
        x = np.frombuffer(pcm, dtype="<i2").astype(float)
        self._rx_samples += len(x)
        if self._rx_600_switch_sample is not None and self._rx_samples >= self._rx_600_switch_sample:
            self._rx_600_active = True
            self._rate = 600
            self._deframe.buf.clear()
            self._rx_600_switch_sample = None
            self._discard_batches = 0
            self._rx_buf.clear()
            self._rx_ready = True
            self._ready()
        fallback_data = b""
        if self._bell103_fallback_engine is not None:
            fallback_data = self._bell103_fallback_engine.feed_pcm(pcm)
            if self._bell103_fallback:
                self.connected = self._bell103_fallback_engine.connected
                data = fallback_data
                if self.v42_mode in ("detect", "lapm"):
                    data += self.v42_read()
                return data
            if self._state == "bell_answer_tone" and self._bell_fsk_mark_present(x):
                self._bell103_mark_samples += len(x)
            else:
                self._bell103_mark_samples = 0
            if self._bell103_mark_samples >= int(0.155 * FS):
                self._select_bell103_fallback()
                # Bytes from the pre-selection waveform are not data-mode
                # payload. The fallback receiver is restarted at selection,
                # so expose only symbols decoded from the selected mode.
                return b""
        answer_tone = False
        detect_2225 = self.calling and (
            self.profile == "bell212a" and self._state == "bell_call_wait_tone"
            or self.profile in ("v22", "v22bis") and self._state == "wait"
        )
        if detect_2225:
            # This is a startup detector, not a permanent notch filter. Once
            # a 2225 Hz answer signal has qualified, stop running it once the
            # handshake leaves its initial wait state. In V.22 this is the
            # documented national substitute for unscrambled binary 1; in
            # Bell 212A it is the answer mark. Never notch the data phase.
            gate_until = self._bell_answer_tone_gate_until
            if gate_until is None or self._rx_samples <= gate_until:
                answer_tone = self._qualify_2225_answer_tone(x)
            else:
                self._bell_answer_tone_present = False
                self._answer_tone_pending.clear()
                self._answer_tone_samples.clear()
        # The 2225 Hz answer mark is not a DPSK carrier. Suppress it until it
        # ends so the high-channel V.22 demodulator cannot acquire the tone.
        rx = np.zeros_like(x) if answer_tone else x
        for z in self._rb.feed(rx):
            self._on_symbol(z)
        if not self._rb.carrier:                       # no symbols arrive once the carrier is gone
            self._carrier_gone()
        data = bytes(self._rx_bytes)
        self._rx_bytes.clear()
        if self.v42_mode in ("detect", "lapm"):
            data += self.v42_read()
        if self.profile == "v22bis" and self.v42_mode not in ("detect", "lapm"):
            if self._rx_data_clamped:
                self._s1_rx_holdback.clear()
                return b""
            if data:
                self._s1_rx_holdback.append((self._rb.symbols, data))
            released = bytearray()
            while (self._s1_rx_holdback
                   and self._rb.symbols - self._s1_rx_holdback[0][0] >= 12):
                _, ready = self._s1_rx_holdback.popleft()
                released.extend(ready)
            data = bytes(released)
        return data

    @staticmethod
    def _bell_fsk_mark_present(samples: np.ndarray) -> bool:
        """Detect the Bell 103-originating 1270-Hz mark in a full-duplex mix.

        The DCE also hears its own 2225-Hz answer tone on networks with
        electrical/packet-loop echo, so qualify tone purity relative to total
        window energy rather than looking only for absolute 1270-Hz power.
        A 0.35 purity floor still leaves margin for an equal-power answer-tone
        echo while rejecting the phase-modulated 1200-Hz Bell 212 signal.
        """
        n = len(samples)
        if n < 80:
            return False
        energy = float(samples @ samples)
        if energy < n * 80.0 ** 2:
            return False
        centered = samples - float(np.mean(samples))
        best = 0.0
        indices = np.arange(n)
        for hz in (1265, 1270, 1275):
            ph = np.exp(-2j * math.pi * hz * indices / FS)
            power = float(abs(np.dot(centered, ph)) ** 2)
            best = max(best, 4.0 * power / (n * energy))
        return best >= 0.35

    def _select_bell103_fallback(self) -> None:
        if self.profile == "bell212a" and self.min_rate > 300:
            self._reject_rate_below_minimum(300)
            return
        self._bell103_fallback = True
        # The fallback engine has already observed the caller's 1270-Hz mark
        # for the qualification interval. Preserve that mark state so a
        # caller that starts async data immediately after CONNECT does not
        # lose its first start edge; discard only stale pre-selection frames.
        self._bell103_fallback_engine.select_after_qualified_mark()
        self._state = "bell_answer_fsk"
        self._bell_answer_tone_end = self._tx_n
        self._rate = self.target_rate = 300
        self._rx_ready = True
        self._rx_buf.clear()
        self.connected = self._bell103_fallback_engine.connected
        self._connected_at = self._rx_samples
        self._event("BELL212A_FALLBACK_103", 300)
        self._event("TRAINING_SUCCEEDED", -4)

    def _qualify_2225_answer_tone(self, samples: np.ndarray) -> bool:
        """Qualify the 2225-Hz Bell mark or V.22 national answer-tone variant."""
        present = self._bell_answer_tone_present
        self._answer_tone_pending.extend(float(v) for v in samples)
        while len(self._answer_tone_pending) >= 80:
            block = self._answer_tone_pending[:80]
            del self._answer_tone_pending[:80]
            self._answer_tone_samples.extend(block)
            valid = False
            if len(self._answer_tone_samples) == self._answer_tone_samples.maxlen:
                x = np.asarray(self._answer_tone_samples, dtype=float)
                x -= float(np.mean(x))
                energy = float(x @ x)
                if energy > len(x) * 40.0 ** 2:
                    n = np.arange(len(x))
                    powers = []
                    for hz in range(2215, 2236, 5):
                        ph = np.exp(-2j * math.pi * hz * n / FS)
                        powers.append((abs(np.dot(x, ph)) ** 2, hz))
                    power, peak = max(powers)
                    # The steady scrambled-mark data waveform can collapse
                    # to a 2250 Hz line. Keep its 25 Hz offset outside this
                    # answer-tone window so it cannot be mistaken for 2225.
                    valid = 4.0 * power / (len(x) * energy) > 0.65 and abs(peak - 2225) <= 10
            if valid:
                self._answer_tone_run += 80
            else:
                self._answer_tone_run = 0
            if (self._state == "bell_call_wait_tone"
                    and self._answer_tone_run >= int(0.155 * FS)):
                self._state = "bell_call_wait_mark"
                self._bell_answer_tone_gate_until = self._rx_samples + int(0.900 * FS)
                self._event("BELL212A_ANSWER_TONE", 2225)
                self._at(456, lambda: (self._set_tx("scr1", 1200), setattr(self, "_state", "bell_call_mark")))
            elif (self.calling and self.profile in ("v22", "v22bis")
                  and self._state == "wait" and self._answer_tone_run >= int(0.155 * FS)):
                # V.22 (11/88) §6.3.1.1 note: some national modems send a
                # 2225-Hz answering tone instead of unscrambled binary 1.
                # Follow the same 456-ms silence and call-mode training path
                # as the ordinary decoded-mark trigger in _after_symbol().
                self._state = "hs1"
                self._event("V22_NATIONAL_ANSWER_TONE", 2225)
                self._at(456, lambda: self._set_tx(
                    "s1" if self.target_rate >= 2400 else "scr1", 1200))
                if self.target_rate >= 2400:
                    self._at(456 + 100, lambda: self._set_tx("scr1", 1200))
            present = valid
        self._bell_answer_tone_present = present
        return present

    def _carrier_gone(self) -> None:
        if self._carrier:
            self._carrier = False
            self._event("CARRIER_DOWN", -1)
            self._n11 = self._alt = self._ones = self._zeros = 0
            self._prev_dibit, self._s1_active = None, False
            if self.profile == "bell212a" and self.bell212_leased_line and self.connected:
                # Do not join partial DTE characters or error-control bits
                # from opposite sides of a receive-carrier gap.
                self._rx_data_clamped = True
                self._rx_clamp_until_symbol = None
                self._rx_buf.clear()
                self._rx_bytes.clear()
                self._s1_rx_holdback.clear()
                self._deframe.reset_partial()
            if self.profile in ("v22", "v22bis") and self.connected:
                # Circuit 104 is clamped to binary 1 while carrier is absent.
                # Discard a partial byte so samples from opposite sides of a
                # dropout can never be joined into a fabricated character.
                self._rx_data_clamped = True
                self._rx_clamp_until_symbol = None
                self._rx_retrain_pending = False
                self._rx_buf.clear()
                self._s1_rx_holdback.clear()
                partial_frame = getattr(self._deframe, "buf", None)
                if partial_frame is not None:
                    partial_frame.clear()
        if self.connected and self._rb.carrier_lost_for > int(0.5 * FS):
            self.connected = False
            if self.profile == "bell212a" and self.bell212_leased_line:
                self._restart_bell_leased_startup()

    def _restart_bell_leased_startup(self) -> None:
        """Restart role-led Bell marks after the carrier-loss timer expires.

        The half-second loss declaration is the existing modem policy, not a
        Bell timing requirement. Queued transmit data is retained; receive
        bytes and partial framing are discarded until carrier is retrained.
        """
        self._bell_leased_recovering = True
        self._event("BELL212A_LEASE_RECOVERY_STARTED", 1200)
        self._actions.clear()
        self._rx_ready = False
        self._rx_data_clamped = True
        self._rx_clamp_until_symbol = None
        self._rx_buf.clear()
        self._rx_bytes.clear()
        self._s1_rx_holdback.clear()
        self._deframe.reset_partial()
        self._ones = self._zeros = 0
        self._rate = 1200
        if self.calling:
            self._state = "bell_leased_call_mark"
            self._set_tx("scr1", 1200)
        else:
            self._state = "bell_leased_answer_wait_mark"
            self._set_tx("silence", 1200)

    def _on_symbol(self, z: complex) -> None:
        rb = self._rb
        if not rb.carrier:
            self._carrier_gone()
            return
        if not self._carrier:
            self._carrier = True
            self._event("CARRIER_UP", -2)
            if (self.profile == "bell212a" and self.bell212_leased_line
                    and self.connected and not self._bell_leased_recovering):
                # A sub-threshold interruption did not trigger a new training
                # exchange. Resume the existing bearer after discarding the
                # partial frame at carrier-down.
                self._rx_data_clamped = False
                self._rx_clamp_until_symbol = None
            elif self._rx_data_clamped:
                # V.22bis §6.5 keeps circuit 104 clamped for 100 ms while
                # checking whether the returning signal is a retrain.
                self._rx_clamp_until_symbol = self._rb.symbols + int(round(0.100 * BAUD))
        first2, bits = self._decide(z)
        if self.target_rate == 600:
            self._raw_ones = self._raw_ones + 1 if bits[0] else 0
        for b in bits:
            self._on_bit(b)
        self._after_symbol(first2, len(bits))

    def _qam_active(self) -> bool:
        return (self._qam_from is not None and self._rb.symbols >= self._qam_from
                and (self._qam_until is None or self._rb.symbols < self._qam_until))

    def request_rate_change(self, rate: int) -> None:
        """Initiate optional V.22bis §6.6 rate signalling while connected.

        Only 1200 and 2400 bit/s are valid. This method models a circuit-111
        transition or the equivalent local control request; the DCE currently
        has no user-facing online rate command.
        """
        if self.profile != "v22bis" or not self.connected or self._state != "bis":
            raise RuntimeError("V.22bis rate change requires a connected V.22bis session")
        if rate not in (1200, 2400) or rate == self._rate:
            raise ValueError("V.22bis rate change target must be the other rate (1200 or 2400)")
        self._begin_rate_change(rate)

    @staticmethod
    def _rate_dibit(rate: int) -> int:
        # V.22bis Table 4: 2400 is 11; 1200 may use 01 or 10.
        return 0b11 if rate == 2400 else 0b01

    @staticmethod
    def _rate_from_dibit(dibit: int) -> int | None:
        return 2400 if dibit == 0b11 else 1200 if dibit in (0b01, 0b10) else None

    def _begin_rate_change(self, rate: int) -> None:
        self._event("RATE_CHANGE_STARTED", rate)
        self._rate_change_attempts = 0
        self._rate_control = "initiator_wait_s1"
        self._rate_control_bits_reset()
        self._rate_s1_start_symbol = None
        self._rx_data_clamped = True
        self._rx_clamp_until_symbol = None
        self._rx_retrain_pending = True
        self._rx_ready = False
        self._rx_buf.clear()
        self._rx_bytes.clear()
        self._s1_rx_holdback.clear()
        self._rate_control_target = rate
        self._s1_phase = 0
        self._actions.clear()
        self._transmit_rate_change_request()

    def _transmit_rate_change_request(self) -> None:
        self._rate_change_attempts += 1
        self._s1_phase = 0
        self._set_tx("s1", 1200)
        self._at(100, lambda: self._set_tx("rate_dibit", 1200))
        # §6.6.1 f) recommends a 1.2-second maximum expected round-trip
        # propagation interval before repeating when no peer S1 was received.
        self._at(1200, self._retry_rate_change_request)

    def _retry_rate_change_request(self) -> None:
        if self._rate_control != "initiator_wait_s1":
            return
        self._event("RATE_CHANGE_RETRY", self._rate_change_attempts)
        self._rate_control_bits_reset()
        self._rate_s1_start_symbol = None
        self._actions.clear()
        self._transmit_rate_change_request()

    def _rate_control_bits_reset(self) -> None:
        self._rate_dibit_bits.clear()
        self._rate_dibit_value = None
        self._rate_dibit_run = 0

    def _rate_control_bit(self, bit: int) -> None:
        if self._rate_control not in ("initiator_rate", "peer_rate"):
            return
        self._rate_dibit_bits.append(bit)
        if len(self._rate_dibit_bits) < 2:
            return
        dibit = (self._rate_dibit_bits[0] << 1) | self._rate_dibit_bits[1]
        self._rate_dibit_bits.clear()
        if self._rate_from_dibit(dibit) is None:
            self._rate_dibit_value = None
            self._rate_dibit_run = 0
            return
        if dibit == self._rate_dibit_value:
            self._rate_dibit_run += 1
        else:
            self._rate_dibit_value = dibit
            self._rate_dibit_run = 1
        if self._rate_dibit_run < 32:
            return
        rate = self._rate_from_dibit(dibit)
        if self._rate_control == "peer_rate":
            if rate == self._rate:
                # S1 plus scrambled ones at the existing 2400 rate is the
                # established §6.4 retrain exchange, not a rate change.
                if rate == 2400:
                    self._rate_control = None
                    self._rate_control_bits_reset()
                    self._start_retrain()
                return
            self._respond_rate_change(rate)
        else:
            self._complete_rate_change(rate)

    def _schedule_rate_switch(self, rate: int, *, rx_switch_symbol: int,
                              tx_switch_after_ms: float) -> None:
        self._rate_control_target = rate
        if rate == 2400:
            self._qam_from = rx_switch_symbol
            self._qam_until = None
        else:
            self._qam_until = rx_switch_symbol
        self._at(tx_switch_after_ms, lambda: self._set_tx("scr1", rate))
        self._at(tx_switch_after_ms + 200, lambda: self._finish_rate_change(rate))

    def _complete_rate_change(self, rate: int) -> None:
        # Initiator timing is referenced to the start of the peer's response
        # S1. It examined R2 after 32 dibits; preserve the 450/600-ms epochs.
        start = self._rate_s1_start_symbol
        if start is None:
            return
        self._rate_control = "initiator_transition"
        elapsed_ms = max(0.0, (self._rb.symbols - start) * 1000.0 / BAUD)
        self._qam_from = start + int(round(0.450 * BAUD)) if rate == 2400 else self._qam_from
        self._qam_until = start + int(round(0.450 * BAUD)) if rate == 1200 else None
        self._at(max(0.0, 600.0 - elapsed_ms), lambda: self._set_tx("scr1", rate))
        self._at(max(0.0, 800.0 - elapsed_ms), lambda: self._finish_rate_change(rate))

    def _respond_rate_change(self, rate: int) -> None:
        self._rate_control = None
        self._rate_control_bits_reset()
        self._rx_data_clamped = True
        self._rx_ready = False
        # Responder's receiver changes 450 ms after the 32-dibit R1 decision;
        # its transmitter sends scrambled ones at 600 ms and data at 800 ms.
        self._schedule_rate_switch(rate, rx_switch_symbol=self._rb.symbols + int(round(.450 * BAUD)),
                                  tx_switch_after_ms=600)
        self._s1_phase = 0
        self._set_tx("s1", 1200)
        self._at(100, lambda: self._set_tx("rate_dibit", 1200))
        self._event("RATE_CHANGE_ACCEPTED", rate)

    def _finish_rate_change(self, rate: int) -> None:
        self._rate = rate
        self.set_v42_rate(rate)
        self._state = "bis"
        self._set_tx("data", rate)
        self._rate_control = None
        self._rate_control_bits_reset()
        self._rx_ready = True
        self._rx_data_clamped = False
        self._rx_clamp_until_symbol = None
        self._rx_retrain_pending = False
        self._rx_buf.clear()
        self._event("RATE_CHANGE_SUCCEEDED", rate)

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
            if self._rx_600_active:
                previous = getattr(self, "_rx_previous_quadrant", q)
                delta = (q - previous) % 4
                self._rx_previous_quadrant = q
                # V.22 Table 1: +90° represents 0; +270° represents 1.
                # For an undecodable 0/180 transition, choose mark and let
                # the self-synchronizing descrambler recover following bits.
                return first2, [int(delta == 3)]
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
        self._rate_control_bit(do)
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
        if self._rx_ready and not self._rx_data_clamped:
            if self._discard_batches:
                self._discard_batches -= 1
                if self._discard_batches == 0:
                    self._rx_buf = []
                return
            self._rx_buf.append(do)
            if len(self._rx_buf) >= 16:
                bits, self._rx_buf = self._rx_buf, []
                if self.v42_mode in ("detect", "lapm", "recovering"):
                    self.v42_rx_bits(bits)
                else:
                    self._rx_bytes += self._plain_rx_bits(bits)

    # -- handshake (§6.3.1) ------------------------------------------------------------------------------------------
    def _after_symbol(self, first2: int, nbits: int) -> None:
        if self.profile == "bell212a":
            self._bell_after_symbol()
            return
        was_s1 = self._s1_active
        self._n11 = self._n11 + 1 if (first2 == 0b11 and nbits == 2) else 0
        prev = self._prev_dibit
        if prev is not None and first2 != prev and {first2, prev} == {0b00, 0b11}:
            self._alt += 1
        else:
            self._alt = 0
        self._prev_dibit = first2
        self._s1_active = self._alt >= 12
        s1_started = not was_s1 and self._s1_active
        s1_ended = was_s1 and not self._s1_active
        if (self._state == "bis" and self.connected and self.profile == "v22bis"
                and self._s1_active and not self._rx_data_clamped):
            # S1 is the in-band retrain request. Clamp circuit 104 as soon as
            # it is qualified so its training bits cannot escape as DTE data.
            self._rx_data_clamped = True
            self._rx_clamp_until_symbol = None
            self._rx_retrain_pending = True
            self._rx_buf.clear()
            self._s1_rx_holdback.clear()
            # Output accumulated earlier in this same audio block is not
            # returned to the DTE until feed_audio() completes. Drop it with
            # the rest of the transition window rather than leak pre-detection
            # S1 symbols as terminal garbage.
            self._rx_bytes.clear()
        if (self.connected and self.profile == "v22bis" and self._state == "bis"
                and s1_started):
            # S1 and the following rate dibits use the 1200-bit/s DPSK
            # waveform even when the current data mode is 2400-bit/s QAM.
            # Once the alternating S1 has qualified, stop making 16-way
            # decisions so the rate-dibit sequence can be descrambled.
            if self._rate >= 2400:
                self._qam_until = self._rb.symbols
            if self._rate_control == "initiator_wait_s1":
                self._rate_s1_start_symbol = self._rb.symbols
            elif self._rate_control is None and self._state == "bis":
                self._rate_control = "peer_wait_end"
                self._rate_s1_start_symbol = self._rb.symbols
                self._rx_data_clamped = True
                self._rx_clamp_until_symbol = None
                self._rx_retrain_pending = True
                self._rx_buf.clear()
                self._s1_rx_holdback.clear()
                self._rx_bytes.clear()
        if s1_ended and self._rate_control == "initiator_wait_s1":
            self._rate_control = "initiator_rate"
            self._rate_control_bits_reset()
        elif s1_ended and self._rate_control == "peer_wait_end":
            self._rate_control = "peer_rate"
            self._rate_control_bits_reset()
            partial_frame = getattr(self._deframe, "buf", None)
            if partial_frame is not None:
                partial_frame.clear()
        if self._rx_data_clamped and (self._s1_active or s1_ended):
            self._rx_retrain_pending = True
            self._rx_clamp_until_symbol = None
        self._release_data_clamp_after_recovery_window()
        sym_ms = lambda ms: int(round(ms * BAUD / 1000.0))
        st = self._state
        if self.target_rate == 600:
            mark_duration = sym_ms(155)
            scrambled_duration = sym_ms(270)
            if self.calling:
                if st == "wait" and self._raw_ones >= mark_duration:
                    self._state = "hs1"
                    self._at(456, lambda: self._set_tx("scr1", 600))
                elif st == "hs1" and self._ones >= scrambled_duration:
                    self._enter_v22()
            elif st == "wait" and (self._ones >= scrambled_duration or self._zeros >= scrambled_duration):
                self._set_tx("scr1", 600)
                self._enter_v22()
            return
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
        if (self._state == "bis" and s1_ended and self.connected
                and self._rate >= 2400 and self._rate_control is None
                and not self._bis_held() and not self._hold_failed):
                self._start_retrain()
        elif (self._state == "bis" and self.calling and s1_ended and self._rx_ready and self._rate >= 2400
                and mode != "0" and not self._hold_failed):
            if self._bis_held():
                # CONNECT is still held back: end the attempt (the DCE reports NO CARRIER and the caller can redial at 1200)
                self._hold_failed = True
                self._event("BIS_REJECTED", -7)
            elif mode == "inplace":
                self._fall_back_to_1200()
            else:
                self._cold_restart_1200()
        if (self._state in ("bis", "retrain") and self._rate_control is None and not self._rx_ready
                and self._qam_active() and self._qam_ones >= 32):
            self._rx_ready = True                                              # 32 consecutive scrambled ones at 2400
            self._rate = 2400
            if self._state == "retrain":
                self._state = "bis"
                self._rx_data_clamped = False
                self._rx_clamp_until_symbol = None
                self._rx_retrain_pending = False
                self._rx_buf.clear()
                self._event("RETRAIN_SUCCEEDED", 2400)
            else:
                self._discard_batches = 60                                     # discard ~0.4 s of the initial stream while it syncs
                self._ready()

    def _bell_after_symbol(self) -> None:
        """Bell 212A 1200-bit/s handshake around the common differential DPSK datapath."""
        mark_bits = int(round(0.270 * BAUD * 2))
        if (not self.calling
                and self._state in ("bell_answer_tone", "bell_leased_answer_wait_mark")
                and self._ones >= mark_bits):
            # Qualify the caller's low-channel mark, then answer on the high channel.
            self._state = "bell_answer_mark"
            self._bell_answer_tone_end = self._tx_n
            self._set_tx("scr1", 1200)
            self._rx_ready = True
            self._rate = 1200
            self._ready()
            self._at(765, self._bell_answer_data_ready)
        elif (self.calling
              and self._state in ("bell_call_mark", "bell_leased_call_mark")
              and self._ones >= mark_bits and not self._rx_ready):
            # The caller becomes receive-ready on the returned scrambled mark;
            # it keeps sending mark for the specified CTS delay.
            self._rx_ready = True
            self._rate = 1200
            self._ready()
            self._at(765, lambda: self._set_tx("data", 1200))

    def _bell_answer_data_ready(self) -> None:
        self._set_tx("data", 1200)

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

    def _start_retrain(self) -> None:
        """Answer a V.22bis §6.4 S1 request and reacquire the 2400-bit/s stream."""
        if self.profile != "v22bis" or self._rate < 2400:
            return
        if self._state != "retrain":
            self._event("RETRAIN_STARTED", 2400)
            self._retrain_attempts = 0
        self._state = "retrain"
        self._rx_ready = False
        self._rx_data_clamped = True
        self._rx_clamp_until_symbol = None
        self._rx_retrain_pending = True
        self._rx_buf.clear()
        partial_frame = getattr(self._deframe, "buf", None)
        if partial_frame is not None:
            partial_frame.clear()
        self._qam_from = self._rb.symbols + int(round(0.450 * BAUD))
        self._qam_until = None
        self._qam_ones = 0
        self._retrain_attempts += 1
        # S1 lasts 100 ms; the remote S1 end is the timing reference for the
        # 600-ms transition to 2400 scrambled ones and the 200-ms data delay.
        self._actions.clear()
        self._s1_phase = 0
        self._set_tx("s1", 1200)
        self._at(100, lambda: self._set_tx("scr1", 1200))
        self._at(600, lambda: self._set_tx("scr1", 2400))
        self._at(800, lambda: self._set_tx("data", 2400))
        self._at(2000, self._retry_retrain_if_unqualified)

    def _retry_retrain_if_unqualified(self) -> None:
        if self._state == "retrain" and not self._rx_ready:
            self._event("RETRAIN_RETRY", self._retrain_attempts)
            self._start_retrain()

    def _cold_restart_1200(self) -> None:
        """Same trigger as ``_fall_back_to_1200``, but go quiet and start over as a plain 1200 caller: no S1, transmit only after
        the answerer's unscrambled ones are heard again (the sequence NIST ACTS completed in nist-4). The in-place switch left
        ACTS in its V.22bis state (nist-5). ``V22_S1_FALLBACK=inplace`` selects the in-place switch, ``0`` disables both.
        ``connected`` stays True so the host does not see a carrier loss; ``TRAINING_SUCCEEDED`` is raised again on success."""
        if 1200 < self.min_rate:
            self._reject_rate_below_minimum(1200)
            return
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
        if 1200 < self.min_rate:
            self._reject_rate_below_minimum(1200)
            return
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
        if 1200 < self.min_rate:
            self._reject_rate_below_minimum(1200)
            return
        self._state = "v22"
        v22_rate = 600 if self.target_rate == 600 else 1200
        self._rate = v22_rate
        self._rx_ready = self.target_rate != 600
        self._discard_batches = 60                                              # drop engine warm-up bits at rate 1200
        if self.target_rate == 600:
            self._rx_previous_quadrant = self._rx_quadrant
            self._rx_600_switch_sample = self._rx_samples + int(round(0.765 * FS))
        self._at(765, lambda: self._set_tx("data", v22_rate))                  # §6.3.1.2
        if self.target_rate != 600:
            self._ready()

    def _reject_rate_below_minimum(self, rate: int) -> None:
        self._hold_failed = True
        self._state = "failed"
        self._set_tx("silence")
        self._event("RATE_BELOW_MINIMUM", int(rate))

    def _ready(self) -> None:
        self.connected = True
        self._bell_leased_recovering = False
        self._rx_data_clamped = False
        self._rx_clamp_until_symbol = None
        self._rx_retrain_pending = False
        self._connected_at = self._rx_samples
        self._event("TRAINING_SUCCEEDED", -4)

    def _release_data_clamp_after_recovery_window(self) -> None:
        """Release §6.5's binary-1 clamp if returned carrier was not a retrain."""
        if (self._rx_data_clamped and not self._rx_retrain_pending
                and self._rx_clamp_until_symbol is not None
                and self._rb.symbols >= self._rx_clamp_until_symbol):
            self._rx_data_clamped = False
            self._rx_clamp_until_symbol = None


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
