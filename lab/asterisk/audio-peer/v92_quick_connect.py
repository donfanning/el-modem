"""V.92 short Phase 1 tones, the V.8-type forms (ITU-T V.92 clause 8.2 and 8.3, Tables 2, 4, 11 and 13).

QC1a (analogue modem, V.21(L)), QCA1a (analogue, V.21(H)), QC1d (digital modem, V.21(L)) and QCA1d (digital,
V.21(H)) are 60 or 70 bit sequences of 10-bit V.8-type frames: ten ones, the synchronisation sequence 0101010101,
a frame (start bit 0, eight information bits, stop bit 1), ten ones, the synchronisation sequence again and the
frame again; the QCA forms end with ten more ones. Values given as bit patterns are transmitted leftmost bit
first (clause 8).

The analogue frame carries the 4-bit UQTS code W X Y Z (the Ucode of the PCM codeword the digital modem uses for
quiet-tone signalling: Table 2, 1111 meaning cleardown from the on-hold state) in the pattern W0XYZ1; the digital
frame carries the 2-bit ANSpcm level L M (Table 11: -9.5, -12, -15, -18 dBm0) in the pattern 000LM1. P is 1 for
LAPM. The V.8 bis forms (QC2a, QCA2a, QC2d, QCA2d) carry the same fields in a 16-bit identification field
(Tables 3, 5, 12, 14); only those fields are built here, not the V.8 bis signal structure around them.
"""

from __future__ import annotations

ONES = [1] * 10
SYNC = [0, 1] * 5
UQTS_UCODES = {0b0000: 61, 0b0001: 62, 0b0010: 63, 0b0011: 66, 0b0100: 67, 0b0101: 70, 0b0110: 71,
               0b0111: 74, 0b1000: 75, 0b1001: 78, 0b1010: 79, 0b1011: 82, 0b1100: 83, 0b1101: 86,
               0b1110: 87}                    # 0b1111: cleardown from on-hold state
UQTS_CLEARDOWN = 0b1111
LEVELS_DBM0 = {0b00: -9.5, 0b01: -12.0, 0b10: -15.0, 0b11: -18.0}


def _nibble(code: int) -> list[int]:
    return [(code >> 3) & 1, (code >> 2) & 1, (code >> 1) & 1, code & 1]       # W X Y Z, leftmost first


def _frame(kind: str, lapm: bool, code: int) -> list[int]:
    """The 10-bit frame: start 0, [analogue/digital, QC/QCA indication, P, 5 fields], stop 1."""
    p = int(bool(lapm))
    if kind in ("QC1a", "QCA1a"):
        w, x, y, z = _nibble(code)
        body = [0, 1 if kind == "QCA1a" else 0, p, w, 0, x, y, z]
    else:
        l, m = (code >> 1) & 1, code & 1
        body = [1, 1 if kind == "QCA1d" else 0, p, 0, 0, 0, l, m]
    return [0] + body + [1]


def build(kind: str, *, lapm: bool = False, code: int = 0) -> list[int]:
    """QC1a / QCA1a: ``code`` is the 4-bit UQTS code; QC1d / QCA1d: the 2-bit ANSpcm level code."""
    if kind not in ("QC1a", "QCA1a", "QC1d", "QCA1d"):
        raise ValueError("kind must be QC1a, QCA1a, QC1d or QCA1d")
    if not 0 <= code <= (15 if kind.endswith("a") else 3):
        raise ValueError("code out of range")
    frame = _frame(kind, lapm, code)
    bits = ONES + SYNC + frame + ONES + SYNC + frame
    if kind.startswith("QCA"):
        bits += ONES
    return bits


def parse(bits) -> dict | None:
    """Parse a 60-bit (QC) or 70-bit (QCA) sequence; None unless both copies of the frame agree."""
    b = [int(x) for x in bits]
    if len(b) not in (60, 70) or b[:10] != ONES or b[10:20] != SYNC or b[30:40] != ONES or b[40:50] != SYNC:
        return None
    frame = b[20:30]
    if frame != b[50:60] or frame[0] != 0 or frame[9] != 1:
        return None
    if len(b) == 70 and b[60:70] != ONES:
        return None
    digital, qca, p = frame[1], frame[2], frame[3]
    if digital == 0:
        if frame[5] != 0:
            return None
        code = frame[4] << 3 | frame[6] << 2 | frame[7] << 1 | frame[8]
        out = {"kind": "QCA1a" if qca else "QC1a", "uqts_code": code,
               "ucode": UQTS_UCODES.get(code), "cleardown": code == UQTS_CLEARDOWN}
    else:
        if frame[4:7] != [0, 0, 0]:
            return None
        code = frame[7] << 1 | frame[8]
        out = {"kind": "QCA1d" if qca else "QC1d", "level_code": code, "level_dbm0": LEVELS_DBM0[code]}
    if (len(b) == 70) != bool(qca):
        return None
    out["lapm"] = bool(p)
    return out


def identification_field(kind: str, *, lapm: bool = False, code: int = 0, revision: int = 0b0100) -> list[int]:
    """The 16-bit V.8 bis identification field of QC2a, QCA2a, QC2d or QCA2d (Tables 3, 5, 12, 14), bit 0 first.
    ``code`` is the UQTS code (analogue) or the level code (digital)."""
    if kind not in ("QC2a", "QCA2a", "QC2d", "QCA2d"):
        raise ValueError("kind must be QC2a, QCA2a, QC2d or QCA2d")
    msg_type = [1, 0, 1, 1]
    rev = [(revision >> 3) & 1, (revision >> 2) & 1, (revision >> 1) & 1, revision & 1]
    p = int(bool(lapm))
    qca = 1 if kind.startswith("QCA") else 0
    if kind.endswith("a"):
        if not 0 <= code <= 15:
            raise ValueError("UQTS code out of range")
        return msg_type + rev + _nibble(code) + [0, p, qca, 0]
    if not 0 <= code <= 3:
        raise ValueError("level code out of range")
    return msg_type + rev + [(code >> 1) & 1, code & 1] + [0, 0, 0, p, qca, 1]
