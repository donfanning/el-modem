"""Sample-clock controller for the V.90 analogue-caller Phase 3 procedure.

This owns protocol order and deadlines. A media/DSP adapter is responsible for
rendering S, S-bar, PP, TRN and Ja, and for producing qualified receive events.
The action timeline uses cumulative symbol boundaries so fractional samples per
symbol do not accumulate rounding drift across the long training sequence.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from v90_dil import expand_dil
from v90_info_reference import parse_ja_descriptor


FS = 8000


@dataclass(frozen=True)
class AnaloguePhase3Action:
    at: int
    signal: str
    duration_samples: int = 0
    symbols: int = 0
    value: object = None


class V90AnalogueCallerPhase3:
    """Drive the caller half of V.90 §9.3.2 through the Phase 4 boundary.

    ``ja_bits`` is the already-built caller Ja descriptor, transmitted by the
    media adapter after the 512T TRN minimum. ``descriptor`` is its parsed
    form and is used to determine the requested DIL length. Receiver events
    must be qualified by the analogue DSP; this class does not infer them
    from arbitrary PCM.
    """

    TX_PREAMBLE = "tx-preamble"
    WAIT_SD = "wait-sd-transition"
    RX_TRN1D = "receive-trn1d"
    WAIT_JD = "wait-jd"
    WAIT_JD_PRIME = "wait-jd-prime"
    RX_DIL = "receive-dil"
    PHASE4_READY = "phase4-ready"
    RETRAIN = "retrain-required"
    FAILED = "failed"
    JD_PRIME_TIMEOUT_MS = 5000

    def __init__(self, *, baud: float, ja_bits, descriptor: dict,
                 md_symbols: int = 0, sd_timeout_ms: int = 1500):
        if not math.isfinite(float(baud)) or baud <= 0:
            raise ValueError("V.90 baud must be positive and finite")
        bits = [int(bit) for bit in ja_bits]
        if not bits or any(bit not in (0, 1) for bit in bits):
            raise ValueError("Ja must contain a nonempty binary descriptor")
        if len(bits) % 2:
            raise ValueError("Ja descriptor must align to two bits per symbol")
        if (isinstance(md_symbols, bool) or not isinstance(md_symbols, int)
                or md_symbols < 0):
            raise ValueError("MD duration must be a nonnegative symbol count")
        if (isinstance(sd_timeout_ms, bool)
                or not isinstance(sd_timeout_ms, int)
                or not 500 <= sd_timeout_ms <= 10000):
            raise ValueError("Sd timeout must be an integer in 500..10000 ms")
        parsed = dict(descriptor)
        if parsed.get("valid") is not True or not parsed.get("crc_valid"):
            raise ValueError("caller Ja descriptor must be parsed and CRC-valid")
        if parsed.get("N") is None:
            raise ValueError("caller Ja descriptor must include N")
        encoded = parse_ja_descriptor(bits)
        if (encoded is None or not encoded.get("valid")
                or encoded.get("N") != parsed.get("N")
                or encoded.get("descriptor_bits") != parsed.get("descriptor_bits")):
            raise ValueError("Ja bits do not encode the supplied parsed descriptor")
        # Validate the DIL structure at setup, rather than discovering an
        # unusable descriptor after we have transmitted its Ja frame.
        self.dil_symbols = len(expand_dil(parsed))
        self.baud = float(baud)
        self.ja_bits = tuple(bits)
        self.descriptor = parsed
        self.md_symbols = md_symbols
        self.sd_timeout_ms = sd_timeout_ms
        self.state = self.TX_PREAMBLE
        self.actions: list[AnaloguePhase3Action] = []
        self._started = False
        self._deadline: int | None = None
        self._ja_start: int | None = None
        self._ja_end: int | None = None
        self._dil_samples = 0
        self._cursor = 0
        self._symbol_cursor = 0

    @property
    def phase4_ready(self) -> bool:
        return self.state == self.PHASE4_READY

    def start(self, at: int) -> list[AnaloguePhase3Action]:
        """Begin the 70 ms silent interval after caller INFO1a."""
        self._check_sample(at)
        if self._started:
            raise RuntimeError("analogue Phase 3 already started")
        self._started = True
        self._cursor = at
        # §9.3.2.1 requires 70 ± 5 ms after INFO1a.
        self._append("silence", duration_samples=70 * FS // 1000)
        self._align_symbols(self._cursor)
        self._append("S", symbols=128)
        self._append("Sbar", symbols=16)
        if self.md_symbols:
            self._append("MD", symbols=self.md_symbols)
            self._append("S", symbols=128)
            self._append("Sbar", symbols=16)
        self._append("PP", symbols=288)
        self._append("TRN", symbols=512)
        self._ja_start = self._cursor
        # Ja repeats until the answerer's Sd -> S-bar-d transition. A
        # one-shot finite Ja followed by silence would not implement §9.3.2.4.
        self.actions.append(AnaloguePhase3Action(
            self._cursor, "Ja-repeat", symbols=len(self.ja_bits) // 2,
            value=self.ja_bits))
        self.actions.append(AnaloguePhase3Action(
            self._cursor, "listen-sd-transition"))
        self._deadline = self._ja_start + self.sd_timeout_ms * FS // 1000
        self.state = self.WAIT_SD
        return self.drain_actions()

    def feed(self, event: str, at: int) -> list[AnaloguePhase3Action]:
        self._check_sample(at)
        if not self._started:
            raise RuntimeError("start analogue Phase 3 before receive events")
        if self.state in (self.PHASE4_READY, self.RETRAIN, self.FAILED):
            return []

        if event == "sd_transition" and self.state == self.WAIT_SD:
            if self._ja_start is not None and at < self._ja_start:
                raise ValueError("Sd transition arrived before Ja began")
            self._deadline = None
            self._ja_end = at
            self.actions.append(AnaloguePhase3Action(at, "stop-Ja"))
            self._align_symbols(at)
            self._append("silence")
            self._align_symbols(at)
            self._append("receive-trn1d", symbols=2040,
                         value="train-equalizer")
            self._deadline = self._ja_end + 4500 * FS // 1000
            self.state = self.RX_TRN1D
        elif event == "trn1d_2040_complete" and self.state == self.RX_TRN1D:
            self._align_symbols(at)
            self.state = self.WAIT_JD
            self.actions.append(AnaloguePhase3Action(at, "listen-jd"))
        elif event == "jd" and self.state == self.WAIT_JD:
            self._deadline = None
            self._align_symbols(at)
            # The Recommendation permits the caller to start immediately
            # after Jd; its 5 s value is an outer bound, not a required delay.
            self._append("S", symbols=128)
            self.actions.append(AnaloguePhase3Action(
                self._cursor, "listen-jd-prime"))
            # Keep the receive wait bounded if the digital modem never
            # terminates Jd. This watchdog is separate from the stream parser:
            # valid repeated Jd frames may continue while S propagates across
            # the round-trip path.
            self._deadline = (self._cursor
                              + self.JD_PRIME_TIMEOUT_MS * FS // 1000)
            self.state = self.WAIT_JD_PRIME
        elif event == "jd_prime" and self.state == self.WAIT_JD_PRIME:
            self._deadline = None
            self._align_symbols(at)
            self._append("Sbar", symbols=16)
            if self.dil_symbols == 0:
                self.state = self.PHASE4_READY
                self.actions.append(AnaloguePhase3Action(
                    self._cursor, "phase4-ready"))
            else:
                self._dil_samples = 0
                self.state = self.RX_DIL
                self.actions.append(AnaloguePhase3Action(
                    self._cursor, "receive-dil",
                    value=self.dil_symbols))
                self._deadline = self._cursor + 5000 * FS // 1000
        else:
            raise ValueError(f"event {event!r} is invalid in state {self.state!r}")
        return self.drain_actions()

    def feed_dil_codewords(self, codewords: bytes | bytearray, *, at: int
                           ) -> list[AnaloguePhase3Action]:
        """Count received DIL octets; send the final S/S-bar once requested DIL is complete."""
        self._check_sample(at)
        if self.state != self.RX_DIL:
            raise RuntimeError("DIL codewords are only valid during DIL reception")
        count = len(codewords)
        if count <= 0:
            return []
        self._dil_samples += count
        if self._dil_samples > self.dil_symbols:
            raise ValueError("received more DIL codewords than requested")
        if self._dil_samples == self.dil_symbols:
            self._deadline = None
            self._align_symbols(at)
            # §9.3.2.10: signal the end of requested DIL no later than 5 s
            # after the preceding S-bar. The controller chooses the earliest
            # valid time after the requested codewords have arrived.
            self._append("S", symbols=128)
            self._append("Sbar", symbols=16)
            self.state = self.PHASE4_READY
            self.actions.append(AnaloguePhase3Action(
                self._cursor, "phase4-ready"))
        return self.drain_actions()

    def tick(self, now: int) -> list[AnaloguePhase3Action]:
        """Run bounded Phase 3 recovery timers on the 8 kHz media clock."""
        self._check_sample(now)
        if not self._started:
            raise RuntimeError("start analogue Phase 3 before timer events")
        if self._deadline is None or now < self._deadline:
            return []
        if self.state == self.WAIT_SD:
            self.state = self.RETRAIN
            self.actions.append(AnaloguePhase3Action(now, "handoff-retrain"))
        elif self.state in (self.RX_TRN1D, self.WAIT_JD,
                            self.WAIT_JD_PRIME):
            self.state = self.RETRAIN
            self.actions.append(AnaloguePhase3Action(now, "handoff-retrain"))
        elif self.state == self.RX_DIL:
            self.state = self.RETRAIN
            self.actions.append(AnaloguePhase3Action(now, "handoff-retrain"))
        self._deadline = None
        return self.drain_actions()

    def _append(self, signal: str, *, symbols: int = 0,
                duration_samples: int = 0, value=None) -> None:
        if symbols:
            start = self._cursor
            end = self._symbol_cursor + symbols
            duration_samples = (int(round(end * FS / self.baud))
                                - int(round(self._symbol_cursor * FS / self.baud)))
            self.actions.append(AnaloguePhase3Action(
                start, signal, duration_samples, symbols, value))
            self._cursor += duration_samples
            self._symbol_cursor = end
        elif duration_samples:
            self.actions.append(AnaloguePhase3Action(
                self._cursor, signal, duration_samples, 0, value))
            self._cursor += duration_samples
        else:
            self.actions.append(AnaloguePhase3Action(self._cursor, signal,
                                                     value=value))

    def _align_symbols(self, at: int) -> None:
        """Start a fresh exact symbol/sample phase at an observed event."""
        self._cursor = at
        self._symbol_cursor = 0

    def drain_actions(self) -> list[AnaloguePhase3Action]:
        actions, self.actions = self.actions, []
        return actions

    @staticmethod
    def _check_sample(sample: int) -> None:
        if isinstance(sample, bool) or not isinstance(sample, int) or sample < 0:
            raise ValueError("sample positions must be nonnegative integers")
