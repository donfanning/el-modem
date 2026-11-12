"""V.90 Phase 4 MP/MP-prime framing from ITU-T V.90 Table 16.

This is deliberately separate from V.34 MP: the field offsets and meanings
differ. Bit 0 is transmitted first. Modulation of these bits into G.711 PCM is
handled by ``V90DigitalPhase4Encoder`` and is not part of this module.
"""
from __future__ import annotations

from collections.abc import Sequence

from v34_info import crc16


SYNC = (1,) * 17


def _bits(value: int, width: int) -> list[int]:
    return [(value >> i) & 1 for i in range(width)]


def _value(bits: Sequence[int], lo: int, width: int) -> int:
    return sum(int(bits[lo + i]) << i for i in range(width))


def _signed16(value: int) -> int:
    return value - 0x10000 if value & 0x8000 else value


def _frame_length(mp_type: int) -> int:
    # 86/188 information-through-fill bits padded to six-bit symbols.
    return 90 if mp_type == 0 else 192


def build_mp(*, mp_type: int = 0, max_a2d_drn: int = 14,
             trellis: int = 0, nonlinear: int = 0,
             shaping: int = 0, ack: int = 0,
             rate_mask: int = 0x1FFF,
             precoder: Sequence[int] | None = None) -> list[int]:
    """Build a V.90 MP or MP-prime frame, including symbol-boundary fill.

    ``rate_mask`` bit 0 represents 4800 bit/s and bit 12 represents 33600
    bit/s. Type 1 ``precoder`` is six signed 16-bit words ordered h1 real,
    h1 imaginary, h2 real, h2 imaginary, h3 real, h3 imaginary.
    """
    if type(mp_type) is not int or mp_type not in (0, 1):
        raise ValueError("MP type must be 0 or 1")
    if type(max_a2d_drn) is not int or max_a2d_drn not in (0, *range(2, 15)):
        raise ValueError("maximum analogue-to-digital DRN must be 0 or 2..14")
    if type(trellis) is not int or trellis not in (0, 1, 2):
        raise ValueError("trellis selector must be 0 (16-state), 1 (32-state), or 2 (64-state)")
    for name, value in (("nonlinear", nonlinear), ("shaping", shaping), ("ack", ack)):
        if type(value) is not int or value not in (0, 1):
            raise ValueError(f"{name} must be 0 or 1")
    if type(rate_mask) is not int or not 0 <= rate_mask < (1 << 13):
        raise ValueError("rate_mask must contain only the 13 V.90 analogue-transmit rate bits")

    body = ([mp_type] + [0] * 5 + _bits(max_a2d_drn, 4) + [0]
            + _bits(trellis, 2) + [nonlinear, shaping, ack]
            + [0, 0] + _bits(rate_mask, 13) + [0, 0, 0])
    if len(body) != 34:
        raise AssertionError("V.90 MP base fields must occupy bits 18..51")

    bits = list(SYNC) + [0] + body
    # The CRC covers bits 18..33 and 35..50, skipping only the start bits at
    # 34 and 51, as in V.34. Reserved bits 35 and 50 are inside it.
    information = body[:16] + body[17:33]
    if mp_type == 0:
        bits += [0] * 16  # Reserved bits 52..67
        bits += [0]        # Start bit 68
        bits += _bits(crc16(information + [0] * 16), 16)
        bits += [0]        # Fill bit 85
    else:
        words = list([0] * 6 if precoder is None else precoder)
        if len(words) != 6 or any(type(v) is not int or not -32768 <= v <= 32767 for v in words):
            raise ValueError("Type 1 precoder requires six signed 16-bit words")
        coeff_bits = []
        for word in words:
            coeff_bits.extend(_bits(word & 0xFFFF, 16))
            bits.extend(_bits(word & 0xFFFF, 16))
            bits.append(0)  # Separator after each 16-bit coefficient
        bits += [0] * 16  # Reserved bits 154..169
        bits += [0]        # Start bit 170
        crc_information = information + coeff_bits + [0] * 16
        bits += _bits(crc16(crc_information), 16)
        bits += [0]        # Fill bit 187

    target = _frame_length(mp_type)
    bits += [0] * (target - len(bits))
    if len(bits) != target:
        raise AssertionError("V.90 MP frame length calculation failed")
    return bits


def parse_mp(bits: Sequence[int]) -> dict | None:
    """Parse one V.90 MP/MP-prime frame at bit zero; return None on bad CRC/framing."""
    if len(bits) < 19 or tuple(bits[:17]) != SYNC or bits[17] != 0:
        return None
    mp_type = int(bits[18])
    if mp_type not in (0, 1):
        return None
    length = _frame_length(mp_type)
    if len(bits) < length or any(bit not in (0, 1) for bit in bits[:length]):
        return None
    if any(bits[index] for index in (19, 20, 21, 22, 23, 28, 34, 35, 49, 50, 51)):
        return None
    max_a2d_drn = _value(bits, 24, 4)
    if max_a2d_drn not in (0, *range(2, 15)):
        return None
    trellis = _value(bits, 29, 2)
    if trellis == 3:
        return None

    information = list(bits[18:34]) + list(bits[35:51])
    if mp_type == 0:
        if bits[68] or any(bits[52:68]) or bits[85] or any(bits[86:90]):
            return None
        information += list(bits[52:68])
        crc_at = 69
    else:
        separator_positions = (68, 85, 102, 119, 136, 153, 170, 187)
        if any(bits[index] for index in separator_positions):
            return None
        if any(bits[154:170]) or any(bits[188:192]):
            return None
        coeff_starts = (52, 69, 86, 103, 120, 137)
        coeffs = [_value(bits, lo, 16) for lo in coeff_starts]
        information += [int(bit) for lo, hi in
                        ((52, 68), (69, 85), (86, 102),
                         (103, 119), (120, 136), (137, 153))
                        for bit in bits[lo:hi]]
        information += list(bits[154:170])
        crc_at = 171
    if _value(bits, crc_at, 16) != crc16(information):
        return None
    result = {
        "type": mp_type,
        "max_a2d_drn": max_a2d_drn,
        "max_a2d_rate": max_a2d_drn * 2400,
        "trellis": trellis,
        "nonlinear": int(bits[31]),
        "shaping": int(bits[32]),
        "ack": int(bits[33]),
        "rate_mask": _value(bits, 36, 13),
        "frame_bits": length,
    }
    if mp_type == 1:
        result["precoder"] = [_signed16(value) for value in coeffs]
    return result


def scan_mp(bits: Sequence[int]) -> list[dict]:
    """Return all CRC-valid MP/MP-prime frames found in a continuous bitstream."""
    found = []
    run = 0
    for index, bit in enumerate(bits):
        if bit:
            run += 1
            continue
        if run >= 17 and index + 1 < len(bits):
            start = index - 17
            mp_type = int(bits[index + 1])
            if mp_type in (0, 1):
                length = _frame_length(mp_type)
                parsed = parse_mp(bits[start:start + length])
                if parsed is not None:
                    found.append({"start_bit": start, **parsed})
        run = 0
    return found


class V90MpScanner:
    """Incrementally find CRC-valid MP/MP′ frames with bounded retention."""

    def __init__(self, *, max_buffer_bits: int = 4096):
        if max_buffer_bits < 192:
            raise ValueError("MP scanner buffer must fit the longest MP-prime frame")
        self.max_buffer_bits = int(max_buffer_bits)
        self._bits: list[int] = []
        self._cursor = 0
        self._base_offset = 0

    @property
    def bits_seen(self) -> int:
        return self._base_offset + len(self._bits)

    def feed(self, bits: Sequence[int]) -> list[dict]:
        chunk = [int(bit) for bit in bits]
        if any(bit not in (0, 1) for bit in chunk):
            raise ValueError("MP scanner input must contain bits")
        self._bits.extend(chunk)
        found = []
        while self._cursor + 18 <= len(self._bits):
            start = self._cursor
            if (tuple(self._bits[start:start + 17]) != SYNC
                    or self._bits[start + 17] != 0):
                self._cursor += 1
                continue
            if start + 19 > len(self._bits):
                break
            mp_type = int(self._bits[start + 18])
            if mp_type not in (0, 1):
                self._cursor += 1
                continue
            length = _frame_length(mp_type)
            if start + length > len(self._bits):
                break
            profile = parse_mp(self._bits[start:start + length])
            if profile is None:
                self._cursor += 1
                continue
            found.append({"start_bit": self._base_offset + start, **profile})
            self._cursor = start + length

        self._cursor = min(self._cursor, max(0, len(self._bits) - 18 + 1))
        if self._cursor:
            del self._bits[:self._cursor]
            self._base_offset += self._cursor
            self._cursor = 0
        if len(self._bits) > self.max_buffer_bits:
            raise BufferError("MP scanner retained more than its configured bound")
        return found
