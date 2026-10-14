"""V.34 Modulation Parameter (MP) sequences, Type 0 and Type 1 (ITU-T V.34 10.1.3.9, Tables 20 and 21).

Bit 0 is sent first. Frame: 17 sync ones, start bit 0, fields, a 0 start bit between fields, 16-bit CRC (same CRC as the
INFO sequences, over the information bits only - not the sync, start or fill bits), 3 fill zeros (Type 0) or 1 (Type 1).
"""

from __future__ import annotations

from v34_info import crc16


def _f(v: int, w: int) -> list[int]:
    return [(v >> i) & 1 for i in range(w)]


def _val(bits, lo: int, hi: int) -> int:
    return sum(bits[lo + i] << i for i in range(hi - lo + 1))


def build_mp(mp_type: int = 0, max_c2a: int = 2, max_a2c: int = 2, aux: int = 0, trellis: int = 0, nonlinear: int = 0,
             shaping: int = 0, ack: int = 0, mask: int = 0b11, asym: int = 0, precoder=None) -> list[int]:
    """``mask`` bit i = data rate (i + 1) * 2400 supported (bit 0 = 2400 ... bit 13 = 33600).
    ``precoder`` for Type 1: six signed 16-bit integers (h1 re, h1 im, h2 re, h2 im, h3 re, h3 im)."""
    bits = [1] * 17 + [0]                                   # sync + start bit (bit 17)
    body1 = [mp_type, 0] + _f(max_c2a, 4) + _f(max_a2c, 4) + _f(aux, 1) + _f(trellis, 2) + _f(nonlinear, 1) \
        + _f(shaping, 1) + _f(ack, 1)
    body2 = _f(mask, 15) + _f(asym, 1)                      # bits 35..50
    crc_in = body1 + body2
    if mp_type == 0:
        body3 = [0] * 16                                    # bits 52..67 reserved
        crc_in = crc_in + body3
        crc = crc16(crc_in)
        return bits + body1 + [0] + body2 + [0] + body3 + [0] + _f(crc, 16) + [0, 0, 0]
    h = list(precoder or [0] * 6)
    fields = [_f(v & 0xFFFF, 16) for v in h]
    out = bits + body1 + [0] + body2
    crc_in = list(body1) + list(body2)
    for fld in fields:
        out += [0] + fld
        crc_in += fld
    reserved = [0] * 16
    out += [0] + reserved
    crc_in += reserved
    out += [0] + _f(crc16(crc_in), 16) + [0]
    return out


def parse_mp(bits) -> dict | None:
    """Parse one MP frame starting at bit 0 (the first sync one). Returns None if framing or CRC is wrong."""
    if len(bits) < 88 or any(b != 1 for b in bits[:17]) or bits[17] != 0:
        return None
    mp_type = bits[18]
    if mp_type == 0:
        n, crc_at = 88, 69
        spans = [(18, 33), (35, 50), (52, 67)]
    else:
        n, crc_at = 188, 171
        spans = [(18, 33), (35, 50), (52, 67), (69, 84), (86, 101), (103, 118), (120, 135), (137, 152), (154, 169)]
    if len(bits) < n:
        return None
    starts = [34, 51, 68] if mp_type == 0 else [34, 51, 68, 85, 102, 119, 136, 153, 170]
    if any(bits[i] != 0 for i in starts):
        return None
    info = []
    for lo, hi in spans:
        info += list(bits[lo:hi + 1])
    if _val(bits, crc_at, crc_at + 15) != crc16(info):
        return None
    out = {"type": mp_type, "max_c2a": _val(bits, 20, 23), "max_a2c": _val(bits, 24, 27), "aux": bits[28],
           "trellis": _val(bits, 29, 30), "nonlinear": bits[31], "shaping": bits[32], "ack": bits[33],
           "mask": _val(bits, 35, 49), "asym": bits[50]}
    if mp_type == 1:
        def s16(lo):
            v = _val(bits, lo, lo + 15)
            return v - 65536 if v >= 32768 else v
        out["precoder"] = [s16(lo) for lo in (52, 69, 86, 103, 120, 137)]
    return out
