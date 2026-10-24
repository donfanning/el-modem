"""Project-only 1200 bps differential-QPSK fixture for the modem lab.

This fixture borrows a 600-baud differential-QPSK shape and Bell/V.22-like
carrier plan for exercising the lab's mode-switch plumbing. It is not an
implementation of Bell 212A or ITU-T V.22, and it is not intended to
interoperate with either. In particular, it does not implement their startup
procedures or real asynchronous-to-synchronous conversion. Each symbol here
encodes a dibit (2 bits) as a phase *change* relative to the previous symbol,
using this fixture's mapping:

    dibit  phase change
    00     0 deg
    01     +90 deg
    10     +180 deg
    11     +270 deg

Carriers: originate transmits 1200 Hz, answer transmits 2400 Hz (same
convention as Bell103's mark/space frequency pair -- a receiver is built
with the *peer's* role to listen on the frequency the other side sends).

Simplifications relative to the real Bell 212A standard (documented here,
not hidden, same spirit as lapm_lite.py's docstring):
 - **Async framing is carried inside the bit stream, not via real
   asynchronous-to-synchronous conversion.** A genuine Bell 212A modem is a
   synchronous line device; to carry asynchronous DTE data it performs
   stop-bit insertion/deletion (SBD) so a synchronous bit stream can still
   represent variable-length idle gaps between characters. This module
   instead keeps the simple Bell103-style convention of one dibit stream
   per character (start bit + 8 data bits + stop bit = 10 bits = exactly 5
   dibits/symbols per character, conveniently no remainder), with no SBD.
   This is sufficient for this project's own two softmodems to interoperate
   at 1200 bps, but will NOT bit-for-bit match a real Bell 212A/V.22
   hardware modem's line framing.
 - **Symbol-clock recovery is "detect carrier onset once, then count fixed
   training+data symbol durations from that edge"**, not a continuously
   reacquiring Costas/PLL loop. This mirrors the level of sophistication
   already accepted for bell103.py's receiver (which also has no continuous
   re-sync once it locks onto a character's start-bit edge) but is more
   exposed here because Bell 212A's carrier never drops to silence between
   characters the way Bell 103's FSK tone does, so there is no repeated
   per-character timing reference -- only the initial carrier-onset edge.
   Sustained clock drift over a long call is a known, untested risk (same
   caveat already called out for Bell 103 in plan.md/README.md).
"""

import math
import struct
from collections import deque

SAMPLE_RATE = 8000
BAUD = 600
BITS_PER_SYMBOL = 2
BITS_PER_CHAR = 10  # 1 start + 8 data + 1 stop (8N1)
SYMBOLS_PER_CHAR = BITS_PER_CHAR // BITS_PER_SYMBOL  # exactly 5, no remainder
TRAINING_SYMBOLS = 40
WINDOW = round(SAMPLE_RATE / BAUD) + 2  # a little more than one symbol period

FREQUENCIES = {
    "originate": 1200.0,
    "answer": 2400.0,
}

# Fixture mapping only; this table is not offered as a complete Bell 212A or
# ITU-T V.22 mapping/conformance claim.
_DIBIT_TO_PHASE = {
    0b00: 0.0,
    0b01: math.pi / 2.0,
    0b10: math.pi,
    0b11: 3.0 * math.pi / 2.0,
}
_PHASE_TO_DIBIT = {value: key for key, value in _DIBIT_TO_PHASE.items()}
_PHASE_STEPS = (0.0, math.pi / 2.0, math.pi, 3.0 * math.pi / 2.0)


def _nearest_phase_step(delta: float) -> float:
    delta %= 2.0 * math.pi
    return min(_PHASE_STEPS, key=lambda step: min(abs(delta - step), 2.0 * math.pi - abs(delta - step)))


class Bell212StreamTransmitter:
    """Continuously modulates bytes fed in over time as differential QPSK,
    holding a steady (no phase change) carrier when no data is queued --
    the Bell212 analogue of Bell103StreamTransmitter."""

    def __init__(self, role: str, baud: float = BAUD):
        if role not in FREQUENCIES:
            raise ValueError(f"unknown Bell 212 role: {role}")
        if baud <= 0:
            raise ValueError("baud rate must be positive")
        self.baud = baud
        self.carrier_hz = FREQUENCIES[role]
        self._phase = 0.0
        self._carrier_phase = 0.0
        self._bit_queue: deque[bool] = deque()
        self._samples_per_symbol = SAMPLE_RATE / baud
        self._samples_into_symbol = 0.0
        # A real training period (held-phase carrier, dibit 00) sent once
        # at the very start of the call, before any real data -- the
        # receiver relies on counting exactly this many symbols after
        # carrier onset before it starts treating dibits as real data.
        self._training_remaining = TRAINING_SYMBOLS

    def feed_bytes(self, data: bytes) -> None:
        for value in data:
            self._bit_queue.append(False)  # start bit
            self._bit_queue.extend(bool(value & (1 << bit)) for bit in range(8))
            self._bit_queue.append(True)  # stop bit

    def pending(self) -> int:
        return len(self._bit_queue)

    def _next_dibit(self) -> int:
        if self._training_remaining > 0:
            self._training_remaining -= 1
            return 0b00
        if len(self._bit_queue) >= 2:
            high = self._bit_queue.popleft()
            low = self._bit_queue.popleft()
        elif len(self._bit_queue) == 1:
            high = self._bit_queue.popleft()
            low = True
        else:
            high = low = True  # idle: mark bits, no phase change
        # Bits map to the *inverted* dibit so that mark (1) bits -- the
        # async idle level -- ride on a steady, no-phase-change carrier.
        return 0b11 ^ ((int(high) << 1) | int(low))

    def samples(self, count: int) -> list[int]:
        output: list[int] = []
        for _ in range(count):
            output.append(round(12000 * math.sin(self._carrier_phase + self._phase)))
            self._carrier_phase = (
                self._carrier_phase + 2.0 * math.pi * self.carrier_hz / SAMPLE_RATE
            ) % (2.0 * math.pi)
            self._samples_into_symbol += 1.0
            if self._samples_into_symbol >= self._samples_per_symbol:
                self._samples_into_symbol -= self._samples_per_symbol
                dibit = self._next_dibit()
                self._phase = (self._phase + _DIBIT_TO_PHASE[dibit]) % (2.0 * math.pi)
        return output


class Bell212Receiver:
    """Demodulates a Bell 212A DPSK stream fed in as PCM, as decoded off the
    audio channel. Call ``feed_pcm`` with 16-bit linear PCM chunks (any
    size) and it returns any newly-completed decoded bytes.

    Correlation is accumulated sample-by-sample against a continuously
    running local-oscillator reference (never resetting phase), and is
    finalized exactly at each symbol boundary -- not over a fixed-width
    sliding window. A fixed-width window wider than one symbol period
    (unavoidable here since samples_per_symbol is a non-integer ~13.33)
    would bleed samples from the *next* symbol into the correlation,
    corrupting the phase measurement whenever that neighbour's phase
    differs -- this showed up as intermittent decode errors concentrated
    around symbols with large phase steps during development.
    """

    def __init__(
        self,
        role: str,
        baud: float = BAUD,
        *,
        require_silence_before_carrier: bool = False,
    ):
        if role not in FREQUENCIES:
            raise ValueError(f"unknown Bell 212 role: {role}")
        if baud <= 0:
            raise ValueError("baud rate must be positive")
        self.baud = baud
        self.carrier_hz = FREQUENCIES[role]
        # Short fixed-size window used only for coarse carrier-onset
        # (energy) detection -- not for symbol phase decoding.
        self._energy_samples: deque[int] = deque(maxlen=WINDOW)
        # A receiver created during a modulation handoff may initially be fed
        # the tail of the previous carrier. Require the inter-mode quiet guard
        # before treating the next energetic window as a fresh symbol clock.
        self._require_silence_before_carrier = require_silence_before_carrier
        self._silence_samples = 0
        self._silence_armed = not require_silence_before_carrier
        self._ref_phase = 0.0
        self._sample_index = -1
        self._carrier_present = False
        self._locked = False
        self._next_symbol_sample = 0.0
        self._samples_per_symbol = SAMPLE_RATE / baud
        self._previous_phase: float | None = None
        self._bits: list[bool] = []
        # Running correlation accumulators for the *current* symbol
        # period, reset every time a symbol boundary is finalized.
        self._real_accum = 0.0
        self._imag_accum = 0.0
        # Training symbols carry a known, fixed (no phase change) pattern
        # and must be skipped rather than decoded as data -- the first
        # one seeds the differential reference, the rest are discarded.
        self._training_remaining = TRAINING_SYMBOLS - 1

    def _step_reference(self) -> tuple[float, float]:
        cos_ref = math.cos(self._ref_phase)
        sin_ref = math.sin(self._ref_phase)
        self._ref_phase = (
            self._ref_phase + 2.0 * math.pi * self.carrier_hz / SAMPLE_RATE
        ) % (2.0 * math.pi)
        return cos_ref, sin_ref

    def feed_pcm(self, pcm: bytes) -> bytes:
        if len(pcm) % 2:
            raise ValueError("PCM must contain whole 16-bit samples")
        decoded = bytearray()
        for (sample,) in struct.iter_unpack("<h", pcm):
            self._sample_index += 1
            cos_ref, sin_ref = self._step_reference()

            if not self._carrier_present:
                self._energy_samples.append(sample)
                if len(self._energy_samples) == WINDOW:
                    energy = sum(s * s for s in self._energy_samples) / WINDOW
                    if not self._silence_armed:
                        if energy < 200_000:
                            self._silence_samples += 1
                        else:
                            self._silence_samples = 0
                        if self._silence_samples >= round(SAMPLE_RATE * 0.01):
                            self._silence_armed = True
                            self._energy_samples.clear()
                        continue
                    if energy >= 2_000_000:
                        # Carrier onset: the single timing reference this
                        # simplified receiver relies on. Symbol boundaries
                        # (training, then data) are counted out in fixed
                        # durations from this edge, with no further
                        # re-sync; see the module docstring.
                        self._carrier_present = True
                        self._next_symbol_sample = (
                            self._sample_index + self._samples_per_symbol
                        )
                        self._real_accum = 0.0
                        self._imag_accum = 0.0
                continue

            self._real_accum += sample * sin_ref
            self._imag_accum += sample * cos_ref

            if self._sample_index + 1 < self._next_symbol_sample:
                continue

            # The true symbol boundary usually falls *inside* this sample's
            # duration (samples_per_symbol is a non-integer ~13.33), not
            # exactly on a sample edge. Rounding the boundary to the
            # nearest whole sample -- rather than splitting this sample's
            # contribution between the current and next symbol -- leaves a
            # small but real leakage bias in the correlation each symbol.
            # That bias is usually too small to matter, but it occasionally
            # pushes a borderline symbol across a quadrant decision
            # boundary and flips a dibit. Split the boundary sample
            # proportionally by how much of it belongs to each symbol.
            frac_in_current = max(0.0, min(1.0, self._next_symbol_sample - self._sample_index))
            frac_in_next = 1.0 - frac_in_current
            carry_real = sample * sin_ref * frac_in_next
            carry_imag = sample * cos_ref * frac_in_next
            self._real_accum -= carry_real
            self._imag_accum -= carry_imag

            real, imag = self._real_accum, self._imag_accum
            self._real_accum = carry_real
            self._imag_accum = carry_imag
            self._next_symbol_sample += self._samples_per_symbol
            phase = math.atan2(imag, real)

            if not self._locked:
                # The very first symbol boundary after onset only seeds
                # the differential-phase reference; it carries no dibit
                # of its own (there is no "previous" symbol to diff
                # against yet).
                self._previous_phase = phase
                self._locked = True
                continue

            assert self._previous_phase is not None
            delta = _nearest_phase_step(phase - self._previous_phase)
            dibit = _PHASE_TO_DIBIT[delta]
            self._previous_phase = phase

            if self._training_remaining > 0:
                self._training_remaining -= 1
                continue

            dibit ^= 0b11  # undo the transmitter's inversion
            self._bits.append(bool(dibit & 0b10))
            self._bits.append(bool(dibit & 0b01))
            while len(self._bits) >= BITS_PER_CHAR:
                char_bits = self._bits[:BITS_PER_CHAR]
                if char_bits[0] is False and char_bits[9] is True:
                    value = sum(
                        int(bit) << index for index, bit in enumerate(char_bits[1:9])
                    )
                    decoded.append(value)
                    del self._bits[:BITS_PER_CHAR]
                else:
                    # Framing mismatch (idle mark bits, or misaligned
                    # start): slide forward one bit and rescan so the
                    # next start bit re-establishes character alignment.
                    del self._bits[0]
        return bytes(decoded)
