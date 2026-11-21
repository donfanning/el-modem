"""V.92 Phase 3 sequences (ITU-T V.92 clauses 8.5, 8.6): Jd, Jp, Jp', Ru, Su, TRN1u.

Bit/symbol-level builders only. They return exact bit patterns or symbol levels; timing,
Ucode-to-codeword mapping and RTP scheduling belong to the call controller.

V.92 Jd is the V.90 Jd with the "16-point" flags (bits 47:48) replaced by the Jd/Jp
identifier and a reserved bit, so ``v90_digital_tx.build_jd_bits`` with both flags clear
is a V.92 Jd. Jp (Table 22) is new.
"""

from __future__ import annotations

import math

import v34_info
import v34_tx
import v90_digital_tx

RATE_STEP = 8000 / 6
JD_BITS = 72
JD_RATE_BITS = 22                   # 28000 .. 56000 in 8000/6 steps (bits 18:33 and 35:40)
JP_FRACTION_BITS = 16
_TRAIN_MIN_SYMBOLS = 12
MAX_ADVERTISED_LOOKAHEAD = 1


def jd_rate_mask(rates) -> int:
    """Rate-capability mask (bit 0 = 28,000 bit/s) for Jd bits 18:33 and 35:40."""
    mask = 0
    for rate in rates:
        k = round((rate - 28000) / RATE_STEP)
        if abs(28000 + k * RATE_STEP - rate) > 1e-6 or not 0 <= k < JD_RATE_BITS:
            raise ValueError(f"{rate} is not a V.92 downstream rate (28,000..56,000 by 8000/6)")
        mask |= 1 << k
    return mask


def build_jd(rate_mask: int, *, max_lookahead: int = MAX_ADVERTISED_LOOKAHEAD) -> list[int]:
    """Table 21 Jd (72 bits; identifier bit 47 = 0)."""
    if rate_mask >> JD_RATE_BITS:
        raise ValueError("Jd rate mask is 22 bits (28,000..56,000)")
    bits = v90_digital_tx.build_jd_bits(rate_mask, max_lookahead=max_lookahead)
    assert bits[47] == 0 and bits[48] == 0
    return bits


def build_jp(phase_fraction: int, *, training_8point: bool = False,
             renegotiation_8point: bool = False) -> list[int]:
    """Table 22 Jp (72 bits). ``phase_fraction`` is the 16-bit fraction of a symbol by which
    the Su that follows Jp -> Jp' is to be extended."""
    if not 0 <= phase_fraction < (1 << JP_FRACTION_BITS):
        raise ValueError("phase fraction is a 16-bit unsigned integer")
    bits = [1] * 17 + [0]
    bits += [(phase_fraction >> i) & 1 for i in range(16)]       # 18:33
    bits.append(0)                                               # 34 start
    bits += [0] * 12                                             # 35:46 reserved
    bits += [1, int(training_8point), int(renegotiation_8point), 0]   # 47:50
    bits.append(0)                                               # 51 start
    crc = v34_info.crc16(bits[18:34] + bits[35:51])
    bits += [(crc >> i) & 1 for i in range(16)] + [0] * 4
    assert len(bits) == JD_BITS
    return bits


def jp_prime() -> list[int]:
    """Jp' terminates Jp: twelve zero bits."""
    return [0] * 12


def parse_jd_jp(bits) -> dict | None:
    """Parse one Jd or Jp frame (CRC, start bits, reserved bits, fill)."""
    data = [int(b) for b in bits[:JD_BITS]]
    if len(data) < JD_BITS or data[:18] != [1] * 17 + [0] or data[34] or data[51]:
        return None
    if data[68:72] != [0] * 4:
        return None
    crc = sum(b << i for i, b in enumerate(data[52:68]))
    if crc != v34_info.crc16(data[18:34] + data[35:51]):
        return None
    if any(data[41:47]) or data[48] and data[47] == 0 or data[50] and data[47]:
        return None
    if data[47] == 0:
        mask = sum(b << i for i, b in enumerate(data[18:34] + data[35:41]))
        look = data[49] + 2 * data[50]
        if not 1 <= look <= 3:
            return None
        return {"kind": "jd", "rate_mask": mask,
                "rates": [28000 + k * RATE_STEP for k in range(JD_RATE_BITS) if mask >> k & 1],
                "max_lookahead": look}
    if any(data[35:41]):
        return None
    fraction = sum(b << i for i, b in enumerate(data[18:34]))
    return {"kind": "jp", "phase_fraction": fraction, "phase_symbols": fraction / 65536,
            "training_8point": bool(data[48]), "renegotiation_8point": bool(data[49])}


class SignModulator:
    """Digital-modem J-sequence coding: scramble (GPC), differentially encode, and return the
    sign of the UINFO codeword (1 = positive, 0 = negative). Jd starts from the last
    TRN1d symbol's sign, Jp from the last Jd sign, Jp' from the last Jp sign."""

    def __init__(self, initial_sign: int = 0):
        self._scrambler = v34_tx.Scrambler("call")
        self._previous = int(initial_sign) & 1

    def encode(self, bits) -> list[int]:
        out = []
        for bit in bits:
            self._previous ^= self._scrambler.step(int(bit))
            out.append(self._previous)
        return out

    @property
    def last_sign(self) -> int:
        return self._previous


def ru(repeats: int, *, bar: bool = False) -> list[float]:
    """Ru (or its inverse): the 6-symbol pattern {+1,+1,+1,-1,-1,-1}, in units of LU."""
    unit = [1.0, 1.0, 1.0, -1.0, -1.0, -1.0]
    return [(-v if bar else v) for _ in range(repeats) for v in unit]


def su(repeats: int, *, bar: bool = False) -> list[float]:
    """Su (or its inverse), in units of LU; whole 12-symbol multiples when repeats is even."""
    a = math.sqrt(1.5)
    unit = [a, 0.0, a, -a, 0.0, -a]
    return [(-v if bar else v) for _ in range(repeats) for v in unit]


def trn1u(symbols: int) -> list[float]:
    """TRN1u: +/-LU from the analogue GPA scrambler fed ones, scrambler zeroed first
    (0 -> +LU, 1 -> -LU). A segment is a whole number of 12 symbols."""
    if symbols % _TRAIN_MIN_SYMBOLS:
        raise ValueError("TRN1u segments are multiples of 12 symbols")
    scrambler = v34_tx.Scrambler("answer")
    return [-1.0 if scrambler.step(1) else 1.0 for _ in range(symbols)]


# -- TRN2u (clause 8.7.6, Tables 28 and 29) ------------------------------------------------------------------
_TRN2U_4 = {0b00: 1.0, 0b01: 3.0, 0b10: -1.0, 0b11: -3.0}
_TRN2U_8 = {0b000: 1.0, 0b001: 3.0, 0b010: 5.0, 0b011: 7.0,
            0b100: -1.0, 0b101: -3.0, 0b110: -5.0, 0b111: -7.0}


class Trn2uModulator:
    """TRN2u: scrambled binary ones (analogue GPA, reset at the start) mapped to a 4- or
    8-point constellation, with the sign bit differentially encoded against the previous
    sign (initialised from the last sign of E1u). Levels are in units of LU.

    Assumption, not stated by the text: the first scrambler bit of each symbol is the
    least-significant bit of the MSB:LSB label (the V.34 bit order the V.90 control path in
    this repo uses), so the last bit is the sign."""

    def __init__(self, points: int = 4, initial_sign: int = 0):
        if points not in (4, 8):
            raise ValueError("TRN2u uses a 4- or 8-point constellation")
        self.points = points
        self._bits = 2 if points == 4 else 3
        self._table = _TRN2U_4 if points == 4 else _TRN2U_8
        self._norm = math.sqrt(5.0) if points == 4 else math.sqrt(21.0)
        self._scrambler = v34_tx.Scrambler("answer")
        self._sign = int(initial_sign) & 1

    def symbols(self, count: int) -> list[float]:
        if count % 12:
            raise ValueError("TRN2u is a whole number of 12-symbol groups")
        out = []
        msb = self._bits - 1
        for _ in range(count):
            label = 0
            for position in range(self._bits):
                label |= self._scrambler.step(1) << position
            self._sign ^= (label >> msb) & 1
            label = (label & ~(1 << msb)) | (self._sign << msb)
            out.append(self._table[label] / self._norm)
        return out

    @property
    def last_sign(self) -> int:
        return self._sign


# -- Rf (clause 8.8.4) --------------------------------------------------------------------------------------
_RF_SIGNS = (1, 1, -1, -1) * 3


def rf_codewords(masks, law: str, *, inverted: bool = False, repeats: int = 1) -> bytes:
    """Rf, or with ``inverted`` its complement: a 12-symbol pattern whose symbol i is the
    highest-power codeword (largest Ucode) of data-mode constellation ``i mod 6`` from the CPu,
    with signs + + - - + + - - + + - -  (inverted: - - + + ...). ``masks`` is the six
    per-interval Ucode sets."""
    from v90_dil import g711_ucode_codewords
    if len(masks) != 6 or any(not m for m in masks):
        raise ValueError("six non-empty per-interval constellations")
    table = g711_ucode_codewords(law)
    signs = tuple(-s for s in _RF_SIGNS) if inverted else _RF_SIGNS
    pattern = bytes(int(table[max(masks[i % 6])]) ^ (0 if signs[i] > 0 else 0x80)
                    for i in range(12))
    return pattern * repeats
