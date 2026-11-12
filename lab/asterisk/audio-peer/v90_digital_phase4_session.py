"""Clocked V.90 digital-answerer Phase 4 session boundary.

The media adapter supplies already-sliced upstream symbols. This module joins
the CRC-valid CP/CPt decoder to the exact-law Phase 4 transmitter and exposes
fixed-size, clocked G.711 output. It does not perform timing/carrier recovery,
symbol slicing, or E detection.
"""
from __future__ import annotations

from collections import deque
from collections.abc import Callable, Sequence

from v90_cp import V90CpSymbolDecoder
from v90_digital_phase4_tx import V90DigitalPhase4Transmitter


class V90DigitalAnswererPhase4Session:
    """Drive Phase 4 between an upstream symbol slicer and RTP media clock."""

    def __init__(self, *, uinfo: int, law: str, constellation: int,
                 mp: dict | None = None):
        self.transmitter = V90DigitalPhase4Transmitter(
            uinfo=uinfo, law=law, mp=mp)
        self.receiver = V90CpSymbolDecoder(constellation)
        self.state = self.transmitter.IDLE
        self.events: list[tuple[str, dict | None]] = []
        self._output: deque[int] = deque()
        self._initial_ri_remaining = 0
        self._cpt_profile: dict | None = None
        self._cp_profile: dict | None = None
        self._cp_prime_profile: dict | None = None
        self._end_signal_seen = False

    @property
    def law(self) -> str:
        return self.transmitter.law

    def start(self) -> None:
        """Queue the required initial 192T Ri for clocked transmission."""
        if self.state != self.transmitter.IDLE:
            raise RuntimeError("V.90 Phase 4 session already started")
        self._output.extend(self.transmitter.start())
        self._initial_ri_remaining = 192
        self.state = self.transmitter.WAIT_CPT
        self.events.append(("ri-start", None))

    def feed_symbols(self, symbols) -> list[dict]:
        """Decode and apply newly received CPt/CP/CP′ messages."""
        parsed = self.receiver.feed(symbols)
        accepted = []
        for profile in parsed:
            if self.receive_profile(profile):
                accepted.append(profile)
        return accepted

    def receive_profile(self, profile: dict) -> bool:
        """Apply one CRC-valid parser result; return whether it advances state."""
        if not isinstance(profile, dict):
            raise ValueError("CP profile must be a parsed V.90 profile")
        kind = profile.get("kind")
        ack = profile.get("ack")
        if kind == "cpt":
            if self.state == self.transmitter.WAIT_CPT:
                # The wait state may have queued a partial 24T repeat to fill
                # the last fixed-size media read. CPt terminates that wait
                # pattern: preserve only any untransmitted part of the
                # required initial 192T, then start the specified fresh 24T
                # response. Appending behind a stale repeat tail moves TRN2d
                # off its protocol boundary (and breaks the caller's mapper).
                if self._initial_ri_remaining:
                    pending = list(self._output)
                    keep = min(self._initial_ri_remaining, len(pending))
                    self._output = deque(pending[:keep])
                else:
                    self._output.clear()
                self._output.extend(self.transmitter.receive_cpt(profile))
                self._cpt_profile = dict(profile)
                self.state = self.transmitter.WAIT_CP
                self.events.append(("cpt-accepted", dict(profile)))
                return True
            if (self._cpt_profile is not None
                    and self._profile_key(profile) == self._profile_key(self._cpt_profile)):
                return False  # CPt is sent repeatedly; don't restart TRN2d.
            raise RuntimeError("unexpected or changed CPt profile")

        if kind != "cp" or ack not in (0, 1):
            raise ValueError("expected parsed CPt, CP, or CP′ profile")
        if ack == 0:
            if self.state == self.transmitter.WAIT_CP:
                self.transmitter.receive_cp(profile)
                self._cp_profile = dict(profile)
                self.state = self.transmitter.WAIT_CP_PRIME
                self.events.append(("cp-selected", dict(profile)))
                return True
            if (self.state == self.transmitter.WAIT_CP_PRIME
                    and self._cp_profile is not None
                    and self._profile_key(profile) == self._profile_key(self._cp_profile)):
                return False  # Repeated CP while the sender changes to MP′.
            raise RuntimeError("unexpected or changed CP data profile")

        if self.state == self.transmitter.WAIT_CP_PRIME:
            if (self._cp_prime_profile is not None
                    and self._profile_key(profile) == self._profile_key(self._cp_prime_profile)):
                return False  # The analogue modem repeats CP′ until it sees MP′/Ed.
            if self._end_signal_seen:
                self._cp_prime_profile = dict(profile)
                return False  # E already qualified the same completion boundary.
            tail = self.transmitter.receive_cp_prime(profile)
            self._output.extend(tail)
            self._cp_prime_profile = dict(profile)
            self._end_signal_seen = True
            self.state = self.transmitter.state
            self.events.append(("cp-prime-accepted", dict(profile)))
            return True
        if (self.state == self.transmitter.DATA_READY
                and self._cp_prime_profile is not None
                and self._profile_key(profile) == self._profile_key(self._cp_prime_profile)):
            return False
        raise RuntimeError("unexpected or changed CP′ profile")

    def receive_e(self) -> None:
        """Apply a 20-bit E event already qualified by the upstream detector."""
        if self._end_signal_seen:
            return
        tail = self.transmitter.receive_e()
        self._output.extend(tail)
        self._end_signal_seen = True
        self.state = self.transmitter.state
        self.events.append(("e-accepted", None))

    def write_data_bits(self, bits) -> int:
        """Queue user bits after B1d; return the resulting G.711 octet count.

        LAPM and async framing must be supplied by the caller. The V.90 mapper
        retains incomplete S+K groups and continues the encoder state from B1d.
        """
        encoded = self.transmitter.encode_data_bits(bits)
        self._output.extend(encoded)
        return len(encoded)

    def read_g711(self, count: int, law: str | None = None, *,
                  data_bit_source: Callable[[int], Sequence[int]] | None = None) -> bytes:
        """Read up to ``count`` exact-law octets on the media transmit clock."""
        if type(count) is not int or count <= 0:
            raise ValueError("G.711 read count must be a positive integer")
        if law is not None and law.lower() != self.law:
            raise ValueError("G.711 law changed after V.90 Phase 4 start")
        if self.state == self.transmitter.IDLE:
            raise RuntimeError("V.90 Phase 4 session has not started")

        out = bytearray()
        while len(out) < count:
            if self._output:
                take = min(count - len(out), len(self._output))
                out.extend(self._output.popleft() for _ in range(take))
                self._initial_ri_remaining = max(
                    0, self._initial_ri_remaining - take)
                continue
            if self.state == self.transmitter.WAIT_CPT:
                self._output.extend(self.transmitter.repeat_ri())
            elif self.state in (self.transmitter.WAIT_CP,
                                self.transmitter.WAIT_CP_PRIME):
                self._output.extend(self.transmitter.next_mp())
                self.state = self.transmitter.state
            elif self.state == self.transmitter.DATA_READY and data_bit_source is not None:
                encoder = self.transmitter.data_encoder
                if encoder is None:
                    raise RuntimeError("V.90 data encoder is unavailable after B1d")
                frames = (count - len(out) + 5) // 6
                bits_needed = frames * (encoder.s + encoder.k)
                bits = data_bit_source(bits_needed)
                if len(bits) != bits_needed:
                    raise ValueError("data bit source returned an incomplete V.90 frame run")
                self._output.extend(self.transmitter.encode_data_bits(bits))
            else:
                break  # B1d completed; data framing is owned by the caller.
        return bytes(out)

    @staticmethod
    def _profile_key(profile: dict) -> dict:
        # start_bit is scanner metadata; acknowledge distinguishes CP from CP′.
        return {key: value for key, value in profile.items()
                if key not in ("ack", "start_bit")}
