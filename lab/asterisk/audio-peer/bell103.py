"""Minimal Bell 103 asynchronous FSK endpoints for the modem lab."""

import math
import struct
import threading
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
        amplitude: float = 12000.0,
    ):
        FREQUENCIES = STANDARDS[standard]
        if role not in FREQUENCIES:
            raise ValueError(f"unknown Bell 103 role: {role}")
        if baud <= 0:
            raise ValueError("baud rate must be positive")
        self.baud = baud
        self.amplitude = float(amplitude)
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
            output.append(round(self.amplitude * math.sin(self.phase)))
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

    def __init__(self, role: str, baud: float = BAUD, standard: str = "bell103",
                 guard_hz: float = 0.0, guard_amp: float = 0.0,
                 v25_answer: bool = False, calling_tone: int = 0,
                 frequencies: tuple[float, float] | None = None,
                 amplitude: float = 12000.0,
                 v25_call_sequence: bool = False):
        """`guard_hz`/`guard_amp` add a continuous guard tone (ITU-T V.21
        1800 Hz answer-station tone, which disables echo suppressors on
        international circuits). Bell 103 has no guard tone; pass none.
        `v25_answer` prefixes the V.25 answer prologue (answer role only):
        2.0 s billing silence, 3.3 s of 2100 Hz with 180-degree reversals
        every 450 ms (echo-canceller disabling), 75 ms silence, then the
        normal modulated carrier.
        `calling_tone` (originate role only, opt-in like the commercial `^`
        dial modifier) prefixes that many V.25 calling-tone bursts — 1300 Hz
        for 0.6 s, silence for 1.75 s — then the normal mark idle. Like the
        commercial toggle it is off unless asked. `v25_call_sequence`
        (originate role) repeats one such burst/interruption cycle until
        `stop_calling_tone()` is called; after the caller's ANS gate completes,
        `release_call_sequence()` starts the FSK data channel."""
        FREQUENCIES = STANDARDS[standard]
        if role not in FREQUENCIES:
            raise ValueError(f"unknown Bell 103 role: {role}")
        if baud <= 0:
            raise ValueError("baud rate must be positive")
        if v25_answer and role != "answer":
            raise ValueError("the V.25 answer prologue is answer-role only")
        if calling_tone and role != "originate":
            raise ValueError("the V.25 calling tone is originate-role only")
        if v25_call_sequence and role != "originate":
            raise ValueError("the automatic V.25 call sequence is originate-role only")
        if v25_answer and calling_tone:
            raise ValueError("answer prologue and calling tone are exclusive")
        self.baud = baud
        self.amplitude = float(amplitude)
        self.mark_hz, self.space_hz = frequencies or FREQUENCIES[role]
        if min(self.mark_hz, self.space_hz) <= 0:
            raise ValueError("FSK mark/space frequencies must be positive")
        self.phase = 0.0
        self.guard_hz, self.guard_amp = guard_hz, guard_amp
        self._guard_phase = 0.0
        self._prologue: list[int] = []
        self._prologue_index = 0
        self._call_sequence_pattern: list[int] | None = None
        self._call_sequence_index = 0
        if v25_answer:
            self._prologue.extend([0] * int(2.0 * SAMPLE_RATE))
            phase = 0.0
            tone_len = int(3.3 * SAMPLE_RATE)
            flip_every = int(0.45 * SAMPLE_RATE)
            for i in range(tone_len):
                if i and i % flip_every == 0:
                    phase += math.pi
                self._prologue.append(round(self.amplitude * math.sin(phase)))
                phase = (phase + 2.0 * math.pi * 2100.0 / SAMPLE_RATE) % (2.0 * math.pi)
            self._prologue.extend([0] * int(0.075 * SAMPLE_RATE))
            self._prologue_index = 0
        if calling_tone and v25_call_sequence:
            raise ValueError("choose either the explicit calling-tone count or V.25 call sequence")
        self._call_sequence_silence = False
        if v25_call_sequence:
            phase = 0.0
            pattern = []
            for _ in range(int(0.6 * SAMPLE_RATE)):
                pattern.append(round(self.amplitude * math.sin(phase)))
                phase = (phase + 2.0 * math.pi * 1300.0 / SAMPLE_RATE) % (2.0 * math.pi)
            pattern.extend([0] * int(1.75 * SAMPLE_RATE))
            self._call_sequence_pattern = pattern
        if calling_tone:
            phase = 0.0
            for _ in range(calling_tone):
                for _ in range(int(0.6 * SAMPLE_RATE)):
                    self._prologue.append(round(self.amplitude * math.sin(phase)))
                    phase = (phase + 2.0 * math.pi * 1300.0 / SAMPLE_RATE) % (2.0 * math.pi)
                self._prologue.extend([0] * int(1.75 * SAMPLE_RATE))
            self._prologue_index = 0
        self._call_sequence_lock = threading.Lock()
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

    @property
    def prologue_remaining(self) -> int:
        with self._call_sequence_lock:
            return max(0, len(self._prologue) - self._prologue_index)

    def stop_calling_tone(self) -> None:
        """Stop the automatic caller tone and hold silence through ANS end."""
        with self._call_sequence_lock:
            self._prologue_index = len(self._prologue)
            self._call_sequence_pattern = None
            self._call_sequence_silence = True

    def release_call_sequence(self) -> None:
        """Resume FSK after the V.25 answer tone and trailing silence."""
        with self._call_sequence_lock:
            self._call_sequence_silence = False

    def samples(self, count: int) -> list[int]:
        with self._call_sequence_lock:
            if self._call_sequence_silence:
                return [0] * count
            output: list[int] = []
            prologue = self._prologue
            if prologue:
                take = min(count, len(prologue) - self._prologue_index)
                output.extend(prologue[self._prologue_index:self._prologue_index + take])
                self._prologue_index += take
            remaining = count - len(output)
            if self._call_sequence_pattern is not None:
                pattern = self._call_sequence_pattern
                for _ in range(remaining):
                    output.append(pattern[self._call_sequence_index])
                    self._call_sequence_index = (self._call_sequence_index + 1) % len(pattern)
                return output
            for _ in range(remaining):
                frequency = self.mark_hz if self._current_bit else self.space_hz
                value = self.amplitude * math.sin(self.phase)
                if self.guard_amp:
                    value += self.guard_amp * math.sin(self._guard_phase)
                    self._guard_phase = (self._guard_phase + 2.0 * math.pi * self.guard_hz / SAMPLE_RATE) % (
                        2.0 * math.pi
                    )
                output.append(round(value))
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
    def __init__(self, role: str, baud: float = BAUD, standard: str = "bell103",
                 frequencies: tuple[float, float] | None = None,
                 track_clock: bool = False):
        FREQUENCIES = STANDARDS[standard]
        if role not in FREQUENCIES:
            raise ValueError(f"unknown Bell 103 role: {role}")
        if baud <= 0:
            raise ValueError("baud rate must be positive")
        self.baud = baud
        self._nominal_baud = float(baud)
        self._track_clock = bool(track_clock)
        self._samples_per_bit = SAMPLE_RATE / self.baud
        self._frame_intervals: deque[tuple[int, int]] = deque(maxlen=48)
        self._last_valid_frame_start: int | None = None
        self.mark_hz, self.space_hz = frequencies or FREQUENCIES[role]
        if min(self.mark_hz, self.space_hz) <= 0:
            raise ValueError("FSK mark/space frequencies must be positive")
        # A symbol-centered discriminator must scale with the symbol period:
        # the historical 27-sample window is one Bell-103 bit at 300 baud,
        # but spans multiple symbols at V.23's 600/1200 baud. At 75 baud the
        # narrow 390/450-Hz return-channel spacing needs the full longer window.
        self.window = max(7, round(SAMPLE_RATE / baud))
        self._samples: deque[int] = deque(maxlen=self.window)
        self._sample_index = -1
        self._last_tone: bool | None = None
        self._tone_start_index = 0
        # More than one 8N1 start candidate may be live at once. A tentative
        # early edge can fail its stop-bit check just as a later, valid frame
        # reaches its first data-bit sample (observed on Bell 103 startup over
        # the production PCMU path). Keep overlapping candidates until their
        # framing is resolved instead of letting the tentative one block RX.
        self._candidates: list[dict] = []
        self._cos = {
            frequency: tuple(
                math.cos(2.0 * math.pi * frequency * i / SAMPLE_RATE)
                for i in range(self.window)
            )
            for frequency in (self.mark_hz, self.space_hz)
        }
        self._sin = {
            frequency: tuple(
                math.sin(2.0 * math.pi * frequency * i / SAMPLE_RATE)
                for i in range(self.window)
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

    def _start_candidate(self, center: int) -> None:
        targets = [
            center + round(0.5 * SAMPLE_RATE / self.baud),
            *(
                center + round((1.5 + bit_index) * SAMPLE_RATE / self.baud)
                for bit_index in range(8)
            ),
        ]
        targets.append(center + round(9.5 * SAMPLE_RATE / self.baud))
        self._candidates.append({
            "start": center,
            "targets": targets,
            "bits": [],
            "votes": [],
            "next": 0,
            "expired": False,
        })

    def prime_after_mark(self) -> None:
        """Discard pre-mode frame candidates but retain a qualified idle mark.

        A mode selector may have been observing the Bell 103 mark before it
        commits to this receiver.  If the far end starts data immediately at
        CONNECT, replacing the receiver would erase the mark-to-space edge
        needed to recognize the first async start bit.
        """
        self._candidates.clear()
        self._last_tone = True
        self._tone_start_index = self._sample_index - self.window // 2

    def reset_frame_sync(self) -> None:
        """Discard partial async characters after a confirmed carrier loss.

        Keep the absolute sample counter monotonic; a new mark interval must
        establish framing after a lost carrier.
        """
        self._samples.clear()
        self._last_tone = None
        self._tone_start_index = self._sample_index
        self._candidates.clear()
        self._reset_clock_tracking()

    def _reset_clock_tracking(self) -> None:
        self._samples_per_bit = SAMPLE_RATE / self._nominal_baud
        self.baud = self._nominal_baud
        self._frame_intervals.clear()
        self._last_valid_frame_start = None

    def _observe_valid_frame(self, start: int) -> None:
        """Track the relative clock from validated 8N1 frame start edges.

        FSK tone transitions inside arbitrary payload data are not clock edges.
        Only a candidate that passes both UART start/stop validation enters this
        estimator. Valid start edges in an uninterrupted asynchronous stream
        are ten or more symbol periods apart, allowing idle mark between bytes.
        """
        previous = self._last_valid_frame_start
        self._last_valid_frame_start = start
        if previous is None:
            return
        delta = start - previous
        bits = round(delta / (10 * self._samples_per_bit)) * 10
        if not 10 <= bits <= 100:
            return
        if abs(delta - bits * self._samples_per_bit) > max(2.0, bits * 0.035):
            return
        self._frame_intervals.append((bits, delta))
        if len(self._frame_intervals) < 8 or len(self._frame_intervals) % 4:
            return

        def fit(pairs: list[tuple[int, int]]) -> float:
            denominator = sum(bits * bits for bits, _ in pairs)
            return sum(bits * delta for bits, delta in pairs) / denominator

        estimate = fit(list(self._frame_intervals))
        clean = [
            (bits, delta) for bits, delta in self._frame_intervals
            if abs(delta - bits * estimate) <= max(2.0, bits * 0.02)
        ]
        if len(clean) < 8:
            return
        estimate = fit(clean)
        nominal = SAMPLE_RATE / self._nominal_baud
        estimate = min(nominal / 0.95, max(nominal / 1.05, estimate))
        # Apply a conservative PLL-like correction; validated frames arrive
        # about once per 8N1 character, and edge quantization is several percent
        # of a symbol at 1200 baud on an 8-kHz sample clock.
        self._samples_per_bit = 0.8 * self._samples_per_bit + 0.2 * estimate
        self.baud = SAMPLE_RATE / self._samples_per_bit

    def _advance_candidates(self, center: int, tone: bool | None) -> bytes:
        """Sample pending frames and emit the earliest non-overlapping valid one."""
        for candidate in self._candidates:
            targets = candidate["targets"]
            while candidate["next"] < 10:
                target = targets[candidate["next"]]
                # At V.23's 1200-baud forward rate there are only 6–7 PCM
                # samples per symbol. A single noisy discriminator result at
                # the sample nearest the bit center can flip a character. Vote
                # across three adjacent, overlapping discriminator windows
                # instead; this path is enabled only when validated-frame
                # clock tracking is requested (the production V.23 forward
                # receiver at either supported rate). Other FSK profiles
                # retain their established one-sample decisions.
                radius = 1 if self._track_clock else 0
                if center < target - radius:
                    break
                if center > target + radius:
                    candidate["expired"] = True
                    candidate["next"] += 1
                    candidate["votes"] = []
                    continue
                if tone is not None:
                    candidate["votes"].append((center, tone))
                if center < target + radius:
                    break
                votes = candidate["votes"]
                if len(votes) < (2 if radius else 1):
                    candidate["expired"] = True
                    candidate["next"] += 1
                    candidate["votes"] = []
                    continue
                marks = sum(bit for _, bit in votes)
                spaces = len(votes) - marks
                if marks == spaces:
                    bit = next((value for sample, value in votes if sample == target),
                               votes[-1][1])
                else:
                    bit = marks > spaces
                candidate["bits"].append(bit)
                candidate["next"] += 1
                candidate["votes"] = []

        decoded = bytearray()
        while self._candidates:
            candidate = self._candidates[0]
            if candidate["expired"]:
                self._candidates.pop(0)
                continue
            if candidate["next"] < 10:
                break
            # 8N1 framing requires a low start bit and a high stop bit. The
            # start edge is only a hypothesis from the sliding tone detector;
            # validate its center before accepting a byte so random
            # mark/space chatter cannot become DTE data just by ending high.
            if candidate["bits"][0] or not candidate["bits"][9]:
                self._candidates.pop(0)
                continue

            value = sum(
                int(bit) << bit_index
                for bit_index, bit in enumerate(candidate["bits"][1:9])
            )
            decoded.append(value)
            if self._track_clock:
                self._observe_valid_frame(candidate["start"])
            frame_end = candidate["start"] + round(10 * SAMPLE_RATE / self.baud)
            self._candidates = [
                later for later in self._candidates[1:]
                if later["start"] >= frame_end
            ]
        return bytes(decoded)

    def feed_pcm(self, pcm: bytes) -> bytes:
        if len(pcm) % 2:
            raise ValueError("PCM must contain whole 16-bit samples")
        decoded = bytearray()
        for (sample,) in struct.iter_unpack("<h", pcm):
            self._sample_index += 1
            self._samples.append(sample)
            if len(self._samples) != self.window:
                continue

            center = self._sample_index - self.window // 2
            tone = self._classify()
            decoded.extend(self._advance_candidates(center, tone))
            if tone is None:
                # Silence breaks tone continuity: retime from here so a later
                # mark→space edge is measured from the true mark onset, not
                # from a stale pre-silence timestamp. Without this, classifier
                # chatter on a silence→mark edge false-triggers a frame on a
                # huge stale hold, wedging the framer through the first real
                # character (seen after the V.25 post-tone silence).
                if tone != self._last_tone:
                    self._last_tone = tone
                    self._tone_start_index = center
                continue
            if (
                self._last_tone is True
                and tone is False
                and center - self._tone_start_index
                >= round(0.5 * SAMPLE_RATE / self.baud)
            ):
                self._start_candidate(center)

            if tone != self._last_tone:
                self._last_tone = tone
                self._tone_start_index = center
        return bytes(decoded)
