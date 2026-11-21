"""ANSpcm (ITU-T V.92 clause 8.3.1, Tables 6-10): the digital modem's answer tone in short Phase 1.

A repetitive sequence of 301 PCM codewords that makes a tone of about 2100 Hz (79/301 of 8000 Hz, 2099.7 Hz),
with a phase reversal every 3612 symbols (12 periods). The codewords come from

    x = floor(scl * sqrt(2) * cos(2 pi k 79 / 301 + theta) + 0.5),  k = 0..300,  theta = 0.25 pi / 301

quantised to G.711 (``scl`` by transmit level and law, Table 6). Checked against the printed Tables 7-10 (all four levels, both laws,
301 entries each) by ``tools/verify_anspcm_tables.py``, which reads the spec text. The check found, and this
implements, two conventions the text does not spell out:

* the linear value is at the G.711 input scale: 14-bit for mu-law, 13-bit for A-law (the A-law ``scl`` is half
  the mu-law one for that reason); and
* the quantiser works on the magnitude and sets the sign afterwards (sign-magnitude). For mu-law the standard
  algorithm agrees; for A-law the standard "-x - 1" rule for negative values is not what the tables use.
"""

from __future__ import annotations

import math

PERIOD = 301
REVERSAL = 3612                           # symbols between phase reversals: 12 periods
THETA = 0.25 * math.pi / 301
# transmit level (dBm0) -> (scl for mu-law, scl for A-law), Table 6
SCL = {-9.5: (1334, 667), -12.0: (1000, 500), -15.0: (708, 354), -18.0: (500, 250)}
_BIAS = 0x84
_CLIP = 32635
_A_SEGMENT_ENDS = (0x1F, 0x3F, 0x7F, 0xFF, 0x1FF, 0x3FF, 0x7FF, 0xFFF)


def _ulaw(linear16: int) -> int:
    sign = 0x80 if linear16 < 0 else 0
    magnitude = min(abs(linear16), _CLIP) + _BIAS
    exponent, mask = 7, 0x4000
    while exponent > 0 and not magnitude & mask:
        exponent -= 1
        mask >>= 1
    mantissa = (magnitude >> (exponent + 3)) & 0x0F
    return ~(sign | (exponent << 4) | mantissa) & 0xFF


def _alaw(linear13: int) -> int:
    magnitude = abs(linear13)
    mask = 0xD5 if linear13 >= 0 else 0x55            # sign-magnitude: no "-x - 1" for negatives
    segment = next((i for i, end in enumerate(_A_SEGMENT_ENDS) if magnitude <= end), 8)
    if segment >= 8:
        return 0x7F ^ mask
    value = segment << 4
    value |= ((magnitude >> 1) if segment < 2 else (magnitude >> segment)) & 0x0F
    return (value ^ mask) & 0xFF


def codewords(level_dbm0: float, law: str) -> bytes:
    """The 301 codewords of one period at ``level_dbm0`` (-9.5, -12, -15 or -18)."""
    if law not in ("pcmu", "pcma"):
        raise ValueError("law must be pcmu or pcma")
    if level_dbm0 not in SCL:
        raise ValueError(f"transmit level must be one of {sorted(SCL)} dBm0")
    scl = SCL[level_dbm0][0 if law == "pcmu" else 1]
    out = []
    for k in range(PERIOD):
        x = math.floor(scl * math.sqrt(2) * math.cos(2 * math.pi * k * 79 / PERIOD + THETA) + 0.5)
        out.append(_ulaw(x * 4) if law == "pcmu" else _alaw(x))
    return bytes(out)


def stream(symbols: int, level_dbm0: float, law: str, *, start_symbol: int = 0) -> bytes:
    """``symbols`` codewords of ANSpcm from ``start_symbol`` on: the period repeated, with the sign of every
    codeword inverted in alternate blocks of 3612 symbols (a 180 degree phase reversal)."""
    period = codewords(level_dbm0, law)
    out = bytearray()
    for n in range(start_symbol, start_symbol + symbols):
        code = period[n % PERIOD]
        if (n // REVERSAL) & 1:
            code ^= 0x80                               # the sign bit is the MSB in both laws
        out.append(code)
    return bytes(out)
