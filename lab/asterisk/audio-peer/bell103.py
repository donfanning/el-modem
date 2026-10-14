"""Minimal Bell 103 asynchronous FSK endpoints for the modem lab."""

import math
import struct
from collections import deque

SAMPLE_RATE = 8000
BAUD = 300
WINDOW = 27
LEAD_BITS = 90

FREQUENCIES = {
    "originate": (1270.0, 1070.0),
    "answer": (2225.0, 2025.0),
}

# ITU-T V.21 (300 bps, same async FSK scheme, different tones): channel 1 is the caller's, channel 2 the
# answerer's; mark (binary 1) is the lower tone, space (0) the higher one.
V21_FREQUENCIES = {
    "originate": (980.0, 1180.0),
    "answer": (1650.0, 1850.0),
}
STANDARDS = {"bell103": FREQUENCIES, "v21": V21_FREQUENCIES}


class Bell103Transmitter:
    def __init__(
        self,
        role: str,
        payload: bytes,
        lead_bits: int = LEAD_BITS,
        baud: float = BAUD,
        standard: str = "bell103",
    ):
        FREQUENCIES = STANDARDS[standard]
        if role not in FREQUENCIES:
            raise ValueError(f"unknown Bell 103 role: {role}")
        if baud <= 0:
            raise ValueError("baud rate must be positive")
        self.baud = baud
        self.mark_hz, self.space_hz = FREQUENCIES[role]
        self.phase = 0.0
        self._symbols: list[tuple[bool, float]] = [(True, float(lead_bits))]
        for value in payload:
            self._symbols.append((False, 1.0))
            self._symbols.extend((bool(value & (1 << bit)), 1.0) for bit in range(8))
            self._symbols.append((True, 1.0))
        self._symbols.append((True, math.inf))
        self._symbol_index = 0
        self._elapsed_bits = float(lead_bits)
        self._symbol_end_sample = round(self._elapsed_bits * SAMPLE_RATE / self.baud)
        self._generated_samples = 0

    def samples(self, count: int) -> list[int]:
        output: list[int] = []
        for _ in range(count):
            mark, duration = self._symbols[self._symbol_index]
            frequency = self.mark_hz if mark else self.space_hz
            output.append(round(12000 * math.sin(self.phase)))
            self.phase = (self.phase + 2.0 * math.pi * frequency / SAMPLE_RATE) % (
                2.0 * math.pi
            )
            self._generated_samples += 1

            if self._generated_samples >= self._symbol_end_sample:
                if not math.isinf(duration):
                    self._symbol_index += 1
                    next_duration = self._symbols[self._symbol_index][1]
                    if math.isinf(next_duration):
                        self._symbol_end_sample = math.inf
                    else:
                        self._elapsed_bits += next_duration
                        self._symbol_end_sample = round(
                            self._elapsed_bits * SAMPLE_RATE / self.baud
                        )
        return output


class Bell103StreamTransmitter:
    """Continuously modulates bytes fed in over time, idling on mark (1)
    when no data is queued -- unlike Bell103Transmitter, which encodes one
    fixed payload chosen up front. Used for live online-data-mode calls
    where terminal bytes arrive asynchronously."""

    def __init__(self, role: str, baud: float = BAUD, standard: str = "bell103"):
        FREQUENCIES = STANDARDS[standard]
        if role not in FREQUENCIES:
            raise ValueError(f"unknown Bell 103 role: {role}")
        if baud <= 0:
            raise ValueError("baud rate must be positive")
        self.baud = baud
        self.mark_hz, self.space_hz = FREQUENCIES[role]
        self.phase = 0.0
        self._bit_queue: deque[bool] = deque()
        self._current_bit = True  # idle = continuous mark
        self._samples_per_bit = SAMPLE_RATE / baud
        self._samples_into_bit = 0.0

    def feed_bytes(self, data: bytes) -> None:
        for value in data:
            self._bit_queue.append(False)  # start bit (space)
            self._bit_queue.extend(bool(value & (1 << bit)) for bit in range(8))
            self._bit_queue.append(True)  # stop bit (mark)

    def pending(self) -> int:
        return len(self._bit_queue)

    def samples(self, count: int) -> list[int]:
        output: list[int] = []
        for _ in range(count):
            frequency = self.mark_hz if self._current_bit else self.space_hz
            output.append(round(12000 * math.sin(self.phase)))
            self.phase = (self.phase + 2.0 * math.pi * frequency / SAMPLE_RATE) % (
                2.0 * math.pi
            )
            self._samples_into_bit += 1.0
            if self._samples_into_bit >= self._samples_per_bit:
                self._samples_into_bit -= self._samples_per_bit
                self._current_bit = (
                    self._bit_queue.popleft() if self._bit_queue else True
                )
        return output


class Bell103Receiver:
    def __init__(self, role: str, baud: float = BAUD, standard: str = "bell103"):
        FREQUENCIES = STANDARDS[standard]
        if role not in FREQUENCIES:
            raise ValueError(f"unknown Bell 103 role: {role}")
        if baud <= 0:
            raise ValueError("baud rate must be positive")
        self.baud = baud
        self.mark_hz, self.space_hz = FREQUENCIES[role]
        self._samples: deque[int] = deque(maxlen=WINDOW)
        self._sample_index = -1
        self._last_tone: bool | None = None
        self._tone_start_index = 0
        self._frame_start: int | None = None
        self._frame_bits: list[bool] = []
        self._targets: list[int] = []
        self._cos = {
            frequency: tuple(
                math.cos(2.0 * math.pi * frequency * i / SAMPLE_RATE)
                for i in range(WINDOW)
            )
            for frequency in (self.mark_hz, self.space_hz)
        }
        self._sin = {
            frequency: tuple(
                math.sin(2.0 * math.pi * frequency * i / SAMPLE_RATE)
                for i in range(WINDOW)
            )
            for frequency in (self.mark_hz, self.space_hz)
        }

    def _classify(self) -> bool | None:
        window = tuple(self._samples)
        energies: list[float] = []
        for frequency in (self.mark_hz, self.space_hz):
            real = sum(sample * c for sample, c in zip(window, self._cos[frequency]))
            imag = sum(sample * s for sample, s in zip(window, self._sin[frequency]))
            energies.append(real * real + imag * imag)
        if sum(energies) < 1_000_000:
            return None
        return energies[0] >= energies[1]

    def feed_pcm(self, pcm: bytes) -> bytes:
        if len(pcm) % 2:
            raise ValueError("PCM must contain whole 16-bit samples")
        decoded = bytearray()
        for (sample,) in struct.iter_unpack("<h", pcm):
            self._sample_index += 1
            self._samples.append(sample)
            if len(self._samples) != WINDOW:
                continue

            center = self._sample_index - WINDOW // 2
            if (
                self._frame_start is not None
                and self._targets
                and center > self._targets[-1]
            ):
                # Overshot the last expected bit-sample position without
                # completing a character -- e.g. a target landed on a
                # silent/low-energy window (such as a gap-filled span
                # covering a lost RTP packet). Abandon the stuck frame
                # instead of wedging character detection forever.
                self._frame_start = None
                self._frame_bits = []
                self._targets = []
            tone = self._classify()
            if tone is None:
                continue
            if self._frame_start is not None:
                for bit_index, target in enumerate(self._targets):
                    if target == center:
                        self._frame_bits.append(tone)
                        break
                if len(self._frame_bits) == 9:
                    value = sum(int(bit) << bit_index for bit_index, bit in enumerate(self._frame_bits[:8]))
                    if self._frame_bits[8]:
                        decoded.append(value)
                    self._frame_start = None
                    self._frame_bits = []
                    self._targets = []
            elif (
                self._last_tone is True
                and tone is False
                and center - self._tone_start_index
                >= round(0.5 * SAMPLE_RATE / self.baud)
            ):
                self._frame_start = center
                self._frame_bits = []
                self._targets = [
                    center + round((1.5 + bit_index) * SAMPLE_RATE / self.baud)
                    for bit_index in range(8)
                ]
                self._targets.append(center + round(9.5 * SAMPLE_RATE / self.baud))

            if tone != self._last_tone:
                self._last_tone = tone
                self._tone_start_index = center
        return bytes(decoded)
