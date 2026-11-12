"""Exact-codeword V.90 digital Phase 4 receive mapper.

This decodes the discrete PCMU/PCMA octets emitted by the digital-side
Phase 4 encoder. It is a mapper inverse, not an RTP clock, packet recovery,
or analog-channel demodulator. Spectral-sign shaping is rejected until its
stateful inverse is implemented.
"""
from __future__ import annotations

from collections.abc import Sequence

import v34_g711 as g711
from v90_dil import g711_ucode_codewords


class _GpcDescrambler:
    """Inverse of the encoder's 23-bit self-synchronizing GPC recurrence."""

    def __init__(self) -> None:
        self.history = [0] * 23

    def step(self, bit: int) -> int:
        decoded = int(bit) ^ self.history[17] ^ self.history[22]
        self.history = [int(bit), *self.history[:22]]
        return decoded


class V90DigitalPhase4Receiver:
    """Decode exact G.711 codewords into V.90 Phase 4 input bits (S_r=0).

    The profile is the selected CPt or CP mapper profile. Decode TRN2d and MP
    with the CPt profile; switch to a fresh CP profile receiver at the B1d
    boundary, then feed B1d and subsequent user-data codewords through that
    instance so mapper and GPC state continue.
    Arbitrary octet chunking is supported; incomplete six-codeword frames are
    buffered. Invalid codewords or out-of-range mixed-radix labels fail closed.
    """

    def __init__(self, profile: dict, law: str):
        self.law = law.lower()
        if self.law not in ("pcmu", "pcma"):
            raise ValueError("law must be 'pcmu' or 'pcma'")
        if str(profile.get("kind", "cp")).lower() not in ("cp", "cpt"):
            raise ValueError("V.90 receiver requires a parsed CPt or CP profile")
        if str(profile.get("codec", self.law)).lower() != self.law:
            raise ValueError("CP profile G.711 law differs from receiver law")
        self.sr = int(profile["spectral_sign_bits"])
        if self.sr != 0:
            raise NotImplementedError(
                "V.90 Phase 4 receive currently supports S_r=0 only")
        rate = float(profile["rate"])
        total_bits = round(rate * 6 / 8000)
        if abs(rate - total_bits * 8000 / 6) > 1e-6:
            raise ValueError("CP rate is not on the V.90 frame-rate grid")
        self.s = 6
        self.k = total_bits - self.s
        if not 1 <= self.k <= 39:
            raise ValueError("CP rate gives an unsupported modulus width")

        selectors = tuple(int(v) for v in profile["interval_constellations"])
        maps = (profile.get("codec_constellations")
                or profile.get("constellations"))
        if maps is None or len(selectors) != 6:
            raise ValueError("CP profile must define six selected constellations")
        self.code_sets = []
        for selector in selectors:
            if not 0 <= selector < len(maps):
                raise ValueError("CP selects a missing constellation")
            points = tuple(sorted({int(v) for v in maps[selector]}, reverse=True))
            if not points or any(not 0 <= point <= 127 for point in points):
                raise ValueError("CP contains an empty or invalid Ucode set")
            self.code_sets.append(points)
        capacity = 1
        for points in self.code_sets:
            capacity *= len(points)
        if (1 << self.k) > capacity:
            raise ValueError("CP constellation product cannot represent its rate")

        table = g711_ucode_codewords(self.law)
        if len(set(map(int, table))) != len(table):
            raise ValueError("G.711 Ucode table is not one-to-one")
        self._codeword_to_point: dict[int, tuple[int, int]] = {}
        for ucode, codeword in enumerate(table):
            codeword = int(codeword)
            self._codeword_to_point[codeword] = (ucode, 1)
            self._codeword_to_point[codeword ^ 0x80] = (ucode, 0)
        if len(self._codeword_to_point) != 256:
            raise ValueError("G.711 signed Ucode codewords are not unique")
        self.reset()

    def reset(self) -> None:
        """Reset at a standard-defined B1d or data-block boundary."""
        self._gpc = _GpcDescrambler()
        self._previous_sign5 = 0
        self._pending: list[int] = []

    def decode_codewords(self, codewords: bytes | bytearray | Sequence[int]) -> list[int]:
        """Decode complete six-octet mapper frames; retain any partial tail."""
        values = [int(value) for value in codewords]
        if any(not 0 <= value <= 255 for value in values):
            raise ValueError("G.711 codewords must be octets")
        self._pending.extend(values)
        output: list[int] = []
        while len(self._pending) >= 6:
            points = []
            signs = []
            for codeword in self._pending[:6]:
                try:
                    ucode, sign = self._codeword_to_point[codeword]
                except KeyError as exc:
                    raise ValueError(
                        f"invalid V.90 G.711 Ucode codeword 0x{codeword:02x}") from exc
                points.append(ucode)
                signs.append(sign)
            output.extend(self._decode_frame(points, signs))
            del self._pending[:6]
        return output

    def _decode_frame(self, points: Sequence[int], signs: Sequence[int]) -> list[int]:
        sign_inputs = [int(signs[0]) ^ self._previous_sign5]
        sign_inputs.extend(int(signs[index]) ^ int(signs[index - 1])
                           for index in range(1, 6))

        value = 0
        multiplier = 1
        for point, legal_points in zip(points, self.code_sets):
            try:
                index = legal_points.index(int(point))
            except ValueError as exc:
                raise ValueError(
                    f"Ucode {point} is not legal in its CP interval") from exc
            value += index * multiplier
            multiplier *= len(legal_points)
        if value >= (1 << self.k):
            raise ValueError("received mixed-radix label exceeds negotiated rate")
        modulus_bits = [(value >> index) & 1 for index in range(self.k)]
        scrambled = sign_inputs + modulus_bits
        decoded = [self._gpc.step(bit) for bit in scrambled]
        self._previous_sign5 = int(signs[-1])
        return decoded
