"""V.42 detection phase (ITU-T V.42 Annex A): ODP/ADP patterns and detector.

The originating (calling) modem sends the Originator Detection Pattern, ODP; an
answering modem that supports V.42 replies with the Answerer Detection
Pattern, ADP. Both are async-framed characters separated by 8-16 mark bits,
so they must be sent as exact bit sequences -- a byte stream through the DTE
cannot control the spacing.

Patterns, from SpanDSP's v42.c (each character is start bit, 8 data bits LSB
first, stop bit):

    ODP    DC1 (0x11), 8-16 ones, 0x91, 8-16 ones, repeated
           "0100010001 11111111 0100010011 11111111"
    ADP    'E' (0x45), 8-16 ones, 'C' (0x43), 8-16 ones     (V.42 supported)
           'E' (0x45), 8-16 ones, NUL,        8-16 ones     (V.42 disabled)

Seen as bytes through an async framer the ADP is simply ``E C E C ...`` (or
``E NUL ...``), which is what ``AdpDetector`` looks for.
"""

from __future__ import annotations

ODP_CHARS = (0x11, 0x91)
ADP_V42 = (0x45, 0x43)
ADP_NO_V42 = (0x45, 0x00)
IDLE_ONES = 8  # 8-16 ones between characters; the stop bit is not counted here


def char_bits(value: int) -> list[int]:
    """One async 8N1 character as bits: start, 8 data (LSB first), stop."""
    return [0] + [(value >> i) & 1 for i in range(8)] + [1]


def pattern_bits(chars: tuple[int, int], repeats: int, ones: int = IDLE_ONES) -> list[int]:
    bits: list[int] = []
    for _ in range(repeats):
        for value in chars:
            bits += char_bits(value) + [1] * ones
    return bits


def odp_bits(repeats: int = 10) -> list[int]:
    """The Originator Detection Pattern, ``repeats`` times through."""
    return pattern_bits(ODP_CHARS, repeats)


def adp_bits(v42: bool = True, repeats: int = 10) -> list[int]:
    return pattern_bits(ADP_V42 if v42 else ADP_NO_V42, repeats)


class _PatternDetector:
    """Looks for a two-character pattern repeated in a decoded byte stream."""

    def __init__(self, pair: tuple[int, int], needed: int = 2):
        self._pair = bytes(pair)
        self._needed = needed
        self._buf = bytearray()

    def feed(self, data: bytes) -> bool:
        self._buf += data
        del self._buf[:-64]  # only ever needs the recent past
        return self._buf.count(self._pair * 1) >= self._needed and (
            self._pair * self._needed in self._buf
        )


class AdpDetector:
    """Classifies an answerer from the bytes it sends after our ODP."""

    def __init__(self) -> None:
        self._v42 = _PatternDetector(ADP_V42)
        self._no_v42 = _PatternDetector(ADP_NO_V42)
        self.result: str | None = None  # "v42" | "no_v42" | None

    def feed(self, data: bytes) -> str | None:
        if self.result is None:
            if self._v42.feed(data):
                self.result = "v42"
            elif self._no_v42.feed(data):
                self.result = "no_v42"
        return self.result


class OdpDetector:
    """Spots an originator's ODP in decoded bytes (for the answering side)."""

    def __init__(self) -> None:
        self._det = _PatternDetector(ODP_CHARS)
        self.seen = False

    def feed(self, data: bytes) -> bool:
        if not self.seen:
            self.seen = self._det.feed(data)
        return self.seen
