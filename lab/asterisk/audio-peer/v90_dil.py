"""V.90 digital-side DIL generation in exact G.711 codewords.

This implements the Table 1 Ucode mapping and §8.4.1 expansion needed by a
digital answerer's downstream transmitter. It returns codec octets directly:
round-tripping through signed-linear PCM would erase PCMU negative zero.
"""

from __future__ import annotations

import numpy as np

import v34_g711 as g711


def g711_ucode_codewords(law: str) -> np.ndarray:
    """Return the 128 positive G.711 codewords in V.90 Ucode order."""
    law = law.lower()
    if law not in ("pcmu", "pcma"):
        raise ValueError("law must be 'pcmu' or 'pcma'")
    levels = g711.decode(np.arange(256, dtype=np.uint8), law)
    codes = [code for code in range(256)
             if (code & 0x80) and levels[code] >= 0
             and not (law == "pcmu" and code == 0x7F)]
    codes.sort(key=lambda code: int(levels[code]))
    if len(codes) < 128 or len({int(levels[code]) for code in codes[:128]}) != 128:
        raise ValueError(f"could not derive 128 distinct positive Ucodes for {law}")
    return np.asarray(codes[:128], dtype=np.uint8)


def expand_dil(descriptor: dict) -> list[dict]:
    """Expand one valid Ja descriptor into its ordered DIL symbols.

    The descriptor must be the parsed form from ``v90_info_reference``. Each
    DIL symbol is one 8 kHz G.711 codeword. SP and TP restart at every segment;
    their bits repeat independently when a segment is longer than a pattern.
    """
    n = int(descriptor["N"])
    signs, training = descriptor["SP"], descriptor["TP"]
    h, references = descriptor["H"], descriptor["REF"]
    ucodes = descriptor["Ucodes"]
    if not 0 <= n <= 255 or len(ucodes) != n:
        raise ValueError("invalid DIL segment count or Ucode list")
    if not signs or not training or len(h) != 8 or len(references) != 8:
        if n:
            raise ValueError("DIL requires SP/TP patterns and eight H/REF values")
    if any(bit not in (0, 1) for bit in signs + training):
        raise ValueError("SP and TP entries must be bits")
    if any(not 0 <= int(value) <= 127 for value in h + references + ucodes):
        raise ValueError("DIL H/REF/Ucode values must be 7-bit integers")

    out = []
    for segment, ucode_value in enumerate(ucodes):
        ucode = int(ucode_value)
        chord = ucode // 16
        count = (int(h[chord]) + 1) * 6
        reference = int(references[chord])
        for symbol in range(count):
            train = bool(training[symbol % len(training)])
            out.append({
                "segment": segment,
                "symbol_in_segment": symbol,
                "ucode": ucode if train else reference,
                "sign": 1 if signs[symbol % len(signs)] else -1,
                "training": train,
            })
    return out


def dil_g711_codewords(descriptor: dict, law: str) -> bytes:
    """Return one complete DIL pass as exact PCMU or PCMA octets."""
    return b"".join(dil_g711_segments(descriptor, law))


def dil_codewords_match(expected: bytes, received: bytes, law: str) -> bool:
    """Compare DIL codewords with the negotiated law's observable zero.

    PCMU 0x7f (negative zero) and 0xff (positive zero) decode to the same
    analog level. A PCM path can canonicalize one to the other, so that sign
    bit is not observable at the analogue receiver. Keep this exception
    limited to those two codewords; all nonzero DIL codewords remain exact.
    PCMA's two zero-near codes have distinct reconstructed levels and are not
    interchangeable.
    """
    law = law.lower()
    if law not in ("pcmu", "pcma"):
        raise ValueError("law must be 'pcmu' or 'pcma'")
    left, right = bytes(expected), bytes(received)
    if len(left) != len(right):
        return False
    return all(want == got or
               (law == "pcmu" and want in (0x7f, 0xff)
                and got in (0x7f, 0xff))
               for want, got in zip(left, right))


def dil_g711_segments(descriptor: dict, law: str) -> list[bytes]:
    """Return each requested DIL segment separately as exact G.711 octets."""
    table = g711_ucode_codewords(law)
    symbols = expand_dil(descriptor)
    if not symbols:
        return []
    segments = []
    current_segment = symbols[0]["segment"]
    codes = []
    for item in symbols:
        if item["segment"] != current_segment:
            segments.append(bytes(codes))
            codes = []
            current_segment = item["segment"]
        code = int(table[item["ucode"]])
        if item["sign"] < 0:
            code ^= 0x80
        codes.append(code)
    segments.append(bytes(codes))
    return segments


class V90DILCodewordStream:
    """Repeat descriptor DIL segments and stop only at a segment boundary.

    ``read(max_samples)`` returns no more than the remainder of the current
    segment so a phase controller can append the next signal at the exact
    boundary. A stop request completes the segment already in progress, then
    returns empty on subsequent reads. Without a stop, all N segments repeat
    as one sequence, as required by V.90 §8.4.1.
    """

    def __init__(self, descriptor: dict, law: str):
        self.law = law.lower()
        self.segments = dil_g711_segments(descriptor, self.law)
        if not self.segments:
            raise ValueError("DIL stream requires at least one segment")
        if any(not segment for segment in self.segments):
            raise ValueError("DIL segments must not be empty")
        self.segment_index = 0
        self.segment_offset = 0
        self.completed_passes = 0
        self._stop_requested = False
        self._stopped = False
        self._at_boundary = True

    def request_stop(self) -> None:
        """Finish the current DIL segment, then stop at its boundary."""
        self._stop_requested = True
        if self._at_boundary:
            self._stopped = True

    @property
    def stopped(self) -> bool:
        """Whether a requested stop has reached a complete segment boundary."""
        return self._stopped

    def read(self, max_samples: int) -> bytes:
        if not isinstance(max_samples, int) or max_samples <= 0:
            raise ValueError("max_samples must be a positive integer")
        if self._stopped:
            return b""
        segment = self.segments[self.segment_index]
        self._at_boundary = False
        end = min(len(segment), self.segment_offset + max_samples)
        result = segment[self.segment_offset:end]
        self.segment_offset = end
        if self.segment_offset == len(segment):
            self.segment_offset = 0
            self._at_boundary = True
            if self._stop_requested:
                self._stopped = True
            else:
                self.segment_index += 1
                if self.segment_index == len(self.segments):
                    self.segment_index = 0
                    self.completed_passes += 1
        return result
