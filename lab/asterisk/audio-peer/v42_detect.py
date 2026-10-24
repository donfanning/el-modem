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
# The answerer must recognize four alternating DC1 characters before it may
# send an ADP. With the minimum eight mark bits, the caller has emitted two
# complete 36-bit ODP blocks by then.
ADP_ARM_BITS = 72


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
        self._buf = bytearray()
        self._gaps: list[int | None] = []
        self.result: str | None = None  # "v42" | "no_v42" | None

    def feed(self, data: bytes, mark_gaps: list[int | None] | None = None) -> str | None:
        """Feed decoded bytes and optional mark-bit gaps before each byte.

        The standard ADP uses 8–16 mark bits between characters. Preserve the
        known back-to-back peer form (zero extra marks) as a compatibility
        exception, but do not let arbitrary long gaps join unrelated text into
        a false ADP. Callers without bit timing retain the historical lenient
        behavior.
        """
        if self.result is None:
            gaps = [None] * len(data) if mark_gaps is None else mark_gaps
            if len(gaps) != len(data):
                raise ValueError("one mark-gap value is required for each decoded byte")
            self._buf.extend(data)
            self._gaps.extend(gaps)
            if len(self._buf) > 64:
                del self._buf[:-64]
                del self._gaps[:-64]
            if len(self._buf) >= 4:
                pattern = bytes(self._buf[-4:])
                tail_gaps = self._gaps[-3:]
                valid_gaps = all(
                    gap is None or gap == 0 or 8 <= gap <= 16
                    for gap in tail_gaps
                )
                if valid_gaps and pattern == bytes(ADP_V42 * 2):
                    self.result = "v42"
                elif valid_gaps and pattern == bytes(ADP_NO_V42 * 2):
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
