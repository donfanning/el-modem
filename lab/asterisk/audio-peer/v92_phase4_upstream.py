"""V.92 Phase 4, analogue modem to digital modem: TRN2u, then SUVu / CPu frames (clause 8.7).

An offline coder pair at one PCM sample per symbol: ``build_burst`` makes the samples the analogue
modem sends, ``recover_frames`` decides them back into the frames the digital modem reads.

Per clauses 8.7.3, 8.7.5, 8.7.6: TRN2u is scrambled binary ones mapped to a 4- or 8-point
constellation (Tables 28/29); SUVu and CPu are "transmitted using the corresponding TRN2u
modulation": the same scrambler (reset at the start of TRN2u, not again), the same mapping, and the
sign bit differentially encoded against the previous sign. TRN2u and its frames are therefore one
continuous symbol stream.

ASSUMPTIONS, none checked against a printed figure or a capture, so a pass here shows the coder and
decoder agree with each other and with the tables as I read them, not that a V.92 modem would
accept the output:

1. The first scrambled bit of a symbol is the least-significant bit of the MSB:LSB label (the V.34
   order this repo uses elsewhere), so the last is the sign bit. Same as ``Trn2uModulator``.
2. Frame bits are scrambled one by one and packed 2 (4-point) or 3 (8-point) to a symbol in
   that order; the differential encoder carries on from TRN2u.
3. After the last frame the stream is padded with zero bits to a whole number of 12-symbol groups
   (clause 8.7.6 says TRN2u is a multiple of 12 symbols; it does not say how frames end).
4. The receiver knows LU. A real digital modem would measure it (SUVu carries the prefilter output
   level, which suggests it is measured and reported back).
"""

from __future__ import annotations

import math

import numpy as np

import v34_tx
import v92_cp
import v92_phase3

_TABLE = {4: v92_phase3._TRN2U_4, 8: v92_phase3._TRN2U_8}
_NORM = {4: math.sqrt(5.0), 8: math.sqrt(21.0)}


class SymbolCoder:
    def __init__(self, points: int, *, initial_sign: int = 0):
        if points not in (4, 8):
            raise ValueError("4- or 8-point constellation")
        self.points = points
        self.bits = 2 if points == 4 else 3
        self._scrambler = v34_tx.Scrambler("answer")
        self._sign = int(initial_sign) & 1
        self._rx_sign = self._sign
        self._levels = {label: value / _NORM[points] for label, value in _TABLE[points].items()}

    def encode(self, bits) -> list[float]:
        bits = [int(b) & 1 for b in bits]
        if len(bits) % self.bits:
            raise ValueError(f"{self.points}-point symbols take {self.bits} bits each")
        msb = self.bits - 1
        out = []
        for i in range(0, len(bits), self.bits):
            label = 0
            for position in range(self.bits):
                label |= self._scrambler.step(bits[i + position]) << position
            self._sign ^= (label >> msb) & 1
            label = (label & ~(1 << msb)) | (self._sign << msb)
            out.append(self._levels[label])
        return out

    def decode(self, symbols) -> list[int]:
        """Inverse of ``encode`` on unit-LU symbols: nearest level, undo the sign differential
        and the scrambler. State carries across calls (the stream is continuous)."""
        labels = sorted(self._levels)
        table = np.array([self._levels[label] for label in labels])
        tap = 5
        out = []
        reg = getattr(self, "_rx_reg", [0] * 23)
        msb = self.bits - 1
        previous = getattr(self, "_rx_sign", 0)
        for value in np.asarray(symbols, dtype=float):
            label = labels[int(np.argmin(np.abs(table - value)))]
            sign = (label >> msb) & 1
            diff = sign ^ previous
            previous = sign
            label = (label & ~(1 << msb)) | (diff << msb)
            for position in range(self.bits):
                y = (label >> position) & 1
                out.append(y ^ reg[tap - 1] ^ reg[22])
                reg = [y] + reg[:-1]
        self._rx_reg, self._rx_sign = reg, previous
        return out


def build_burst(frames, *, points: int = 4, lu: float = 2000.0, trn_symbols: int = 240,
                initial_sign: int = 0) -> np.ndarray:
    """TRN2u (scrambled ones), then each frame plus fill to a 12-symbol boundary, at level ``lu``."""
    if trn_symbols % 12 or trn_symbols <= 0:
        raise ValueError("TRN2u is a positive multiple of 12 symbols")
    coder = SymbolCoder(points, initial_sign=initial_sign)
    bits = [1] * (trn_symbols * coder.bits)
    group = 12 * coder.bits
    for frame in frames:
        bits.extend(int(b) & 1 for b in frame)
        bits.extend([0] * (-len(frame) % group))
    return np.array(coder.encode(bits)) * lu


def recover_frames(samples, *, points: int = 4, lu: float = 2000.0,
                   initial_sign: int = 0) -> list[dict]:
    """Decide the symbols, drop the TRN2u ones, and parse SUVu / CPu frames in order."""
    coder = SymbolCoder(points)
    coder._rx_sign = int(initial_sign) & 1
    bits = coder.decode(np.asarray(samples, dtype=float) / lu)
    try:
        first_zero = bits.index(0)
    except ValueError:
        return []
    pos = max(0, first_zero - 17)                    # a frame opens with 17 ones and a zero
    frames = []
    while pos < len(bits):
        chunk = bits[pos:]
        parsed = v92_cp.parse_suvu(chunk) or v92_cp.parse_cpu(chunk)
        if parsed is None:
            break
        frames.append(parsed)
        end = pos + int(parsed["frame_bits"])
        pos = end + (-end % (12 * coder.bits))
    return frames
