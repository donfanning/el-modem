"""V.92 Phase 4 control sequences: CPu/CPt (Table 23), CPus (Table 24), CPd (Table 30).

A CP sequence is a 17-bit sync word of ones followed by 17-bit words: a 0 start bit and
16 data bits, least-significant bit first. The CRC (V.34 10.1.2.3.2, ``v34_info.crc16``)
covers the data bits of every word after the sync, start bits excluded, and sits in the
last word; one 0 fill bit follows it. These layouts differ from the V.90 CP in
``v90_cp`` (rate field at bits 21:25 rather than 20:24, a two-bit type field, no rate
mask or silent-period bit), so that parser cannot read them.

Parsers return ``None`` for any framing, start-bit, reserved-bit, range or CRC fault.
"""

from __future__ import annotations

from v34_info import crc16

SYNC = [1] * 17
MAX_CONSTELLATION_INDEX = 5
MAX_SET_POINTS = 64                  # N = 2*LC and a set holds at most 128 points

# Table 23: seven fixed words, up to six transmitter constellations and six
# distinct codec-output constellations, one CRC word, 17-bit sync, fill bit.
MAX_CPU_FRAME_BITS = 17 + (7 + 8 * 6 * 2 + 1) * 17 + 1
SINGLE_WORD_FRAME_BITS = 17 + 2 * 17 + 1

CPT, CPU, CPUS = 0, 1, 2
RATE_STEP = 8000 / 6


def _field(value: int, width: int) -> list[int]:
    if not 0 <= value < (1 << width):
        raise ValueError(f"value {value} does not fit {width} bits")
    return [(value >> i) & 1 for i in range(width)]


def _signed_field(value: int, width: int) -> list[int]:
    if not -(1 << (width - 1)) <= value < (1 << (width - 1)):
        raise ValueError(f"value {value} does not fit signed {width} bits")
    return _field(value & ((1 << width) - 1), width)


def _frame(words: list[list[int]]) -> list[int]:
    """Assemble sync, start-bit-prefixed words, CRC word and the fill bit."""
    for word in words:
        if len(word) != 16:
            raise ValueError("a CP word has 16 data bits")
    information = [bit for word in words for bit in word]
    bits = list(SYNC)
    for word in words + [_field(crc16(information), 16)]:
        bits.append(0)
        bits.extend(word)
    bits.append(0)
    return bits


def _candidates(bits) -> list[tuple[list[list[int]], int]]:
    """Every way to read ``bits`` as sync, start-bit words, a matching CRC word and a 0 fill
    bit. Each entry is (data words without the CRC word, consumed bits including the fill).

    The sequences carry no length field, so a CRC match alone could end a sequence early
    (about 2**-16 per word); each parser keeps the candidate that satisfies its own
    length rule."""
    data = [int(b) for b in bits]
    if len(data) < 17 + 2 * 17 + 1 or data[:17] != SYNC or any(b not in (0, 1) for b in data):
        return []
    words: list[list[int]] = []
    found = []
    reg = 0xFFFF
    index = 17
    while index + 17 <= len(data) and data[index] == 0:
        words.append(data[index + 1:index + 17])
        index += 17
        if len(words) >= 2:
            # Running CRC over every word but the last (it was quadratic when recomputed per word).
            for b in words[-2]:
                feedback = (reg & 1) ^ b
                reg >>= 1
                if feedback:
                    reg ^= 0x8408
            if (sum(b << i for i, b in enumerate(words[-1])) == reg
                    and index < len(data) and data[index] == 0):
                found.append((list(words[:-1]), index + 1))
    return found


def _value(bits, lo: int, width: int) -> int:
    return sum(int(bits[lo + i]) << i for i in range(width))


def _signed(value: int, width: int) -> int:
    return value - (1 << width) if value & (1 << (width - 1)) else value


def _mask_words(ucodes) -> list[list[int]]:
    mask = set(ucodes)
    if any(not 0 <= u < 128 for u in mask):
        raise ValueError("Ucodes are 0..127")
    return [[1 if 16 * group + bit in mask else 0 for bit in range(16)] for group in range(8)]


def _mask_ucodes(words) -> tuple[int, ...]:
    return tuple(16 * g + b for g in range(8) for b in range(16) if words[g][b])


# -- CPu / CPt (Table 23) -------------------------------------------------------------------------------------
def build_cpu(*, kind: int = CPU, drn: int, sr: int = 0, ack: bool = False, alaw: bool = False,
              lookahead: int = 0, trn1d_rms_ratio: int = 0x2000, shaping=(0, 0, 0, 0),
              interval_sets=(0,) * 6, constellations, codec_constellations=None) -> list[int]:
    """CPu (kind=1) or CPt (kind=0). ``shaping`` is (a1, a2, b1, b2) in signed Q1.6 integers;
    ``trn1d_rms_ratio`` is unsigned Q3.13; constellations are iterables of Ucodes."""
    if kind not in (CPT, CPU):
        raise ValueError("kind must be CPT or CPU")
    if not 0 <= drn <= 22 or not 0 <= sr <= 3 or not 0 <= lookahead <= 3:
        raise ValueError("drn 0..22, sr 0..3, lookahead 0..3")
    if len(interval_sets) != 6 or any(not 0 <= i <= MAX_CONSTELLATION_INDEX for i in interval_sets):
        raise ValueError("six interval selectors, each 0..5")
    count = max(interval_sets) + 1
    if len(constellations) != count:
        raise ValueError("one constellation per index up to the highest selected")
    if codec_constellations is not None and len(codec_constellations) != count:
        raise ValueError("one codec constellation per transmit constellation")
    words = [
        [0] + _field(kind, 2) + _field(drn, 5) + _field(0, 5) + _field(sr, 2) + [int(ack)],
        [int(alaw)] + _field(0, 13) + _field(lookahead, 2),
        _field(trn1d_rms_ratio, 16),
        _signed_field(shaping[0], 8) + _signed_field(shaping[1], 8),
        _signed_field(shaping[2], 8) + _signed_field(shaping[3], 8),
        sum((_field(i, 4) for i in interval_sets[:4]), []),
        _field(interval_sets[4], 4) + _field(interval_sets[5], 4)
        + [int(codec_constellations is not None)] + _field(0, 7),
    ]
    for masks in list(constellations) + list(codec_constellations or ()):
        words.extend(_mask_words(masks))
    return _frame(words)


def parse_cpu(bits) -> dict | None:
    """Parse a CPu or CPt (Table 23)."""
    # The wire format has a strict maximum size.  The control scanner may pass
    # a much larger lookahead window while hunting through TRN2u; parsing the
    # whole window for every false sync candidate needlessly holds the shared
    # media lock and can starve TX.
    for words, consumed in _candidates(bits[:MAX_CPU_FRAME_BITS]):
        parsed = _parse_cpu_words(words, consumed)
        if parsed is not None:
            return parsed
    return None


def _parse_cpu_words(words, consumed) -> dict | None:
    if len(words) < 7 + 8:
        return None
    w1, w2, w3, w4, w5, w6, w7 = words[:7]
    kind = _value(w1, 1, 2)
    drn = _value(w1, 3, 5)
    if w1[0] or kind not in (CPT, CPU) or drn > 22 or any(w1[8:13]):
        return None
    if any(w2[1:14]):
        return None
    sets = tuple(_value(w6, 4 * i, 4) for i in range(4)) + (_value(w7, 0, 4), _value(w7, 4, 4))
    if max(sets) > MAX_CONSTELLATION_INDEX or any(w7[9:16]):
        return None
    count = max(sets) + 1
    differ = bool(w7[8])
    expected = 7 + 8 * count * (2 if differ else 1)
    if len(words) != expected:
        return None
    blocks = [tuple(_mask_ucodes(words[7 + 8 * n:15 + 8 * n]))
              for n in range(count * (2 if differ else 1))]
    return {
        "kind": "cpt" if kind == CPT else "cpu",
        "drn": drn,
        "rate": (drn + (8 if kind == CPT else 20)) * RATE_STEP if drn else None,
        "cleardown": drn == 0,
        "sr": _value(w1, 13, 2),
        "ack": w1[15],
        "codec": "pcma" if w2[0] else "pcmu",
        "lookahead": _value(w2, 14, 2),
        "trn1d_rms_ratio": _value(w3, 0, 16) / 8192.0,
        "shaping": tuple(_signed(_value(w, lo, 8), 8) / 64.0
                         for w, lo in ((w4, 0), (w4, 8), (w5, 0), (w5, 8))),
        "interval_sets": sets,
        "constellations": tuple(blocks[:count]),
        "codec_constellations": tuple(blocks[count:]) if differ else None,
        "frame_bits": consumed,
    }


# -- CPus (Table 24) -------------------------------------------------------------------------------------------
def build_cpus(*, drn: int, ack: bool = False) -> list[int]:
    if not 0 <= drn <= 22:
        raise ValueError("drn 0..22")
    return _frame([[0] + _field(CPUS, 2) + _field(drn, 5) + _field(0, 7) + [int(ack)]])


def parse_cpus(bits) -> dict | None:
    for words, consumed in _candidates(bits[:SINGLE_WORD_FRAME_BITS]):
        if len(words) == 1:
            return _parse_cpus_word(words[0], consumed)
    return None


def _parse_cpus_word(w, consumed) -> dict | None:
    drn = _value(w, 3, 5)
    if w[0] or _value(w, 1, 2) != CPUS or drn > 22 or any(w[8:15]):
        return None
    return {"kind": "cpus", "drn": drn, "rate": (drn + 20) * RATE_STEP if drn else None,
            "cleardown": drn == 0, "ack": w[15], "frame_bits": consumed}


# -- CPd (Table 30) --------------------------------------------------------------------------------------------
def build_cpd(*, drn: int, trellis: int = 0, extend_e2u: bool = False, ack: bool = False,
              gain4: int, moduli=None, precoder=None, interval_sets=None,
              constellations=None) -> list[int]:
    """CPd. ``gain4`` is 4*G in unsigned Q0.16. ``moduli``: twelve M values (8 bits each).
    ``precoder``: dict with integer lists z1 (Q0.15), p1 (Q1.14), z2 (Q0.15), p2 (Q1.14).
    ``interval_sets`` is the six selectors reused for intervals i and i+6; ``constellations``
    are lists of positive linear values (ascending), at most 64 per set, non-empty sets first.
    Any part left as None is omitted and its presence bit is clear."""
    if not 0 <= drn <= 19 or trellis not in (0, 1, 2) or not 0 < gain4 < (1 << 16):
        raise ValueError("drn 0..19, trellis 0..2, 0 < 4G < 1")
    words = [[0, int(moduli is not None), int(precoder is not None),
              int(constellations is not None)] + _field(drn, 5) + _field(trellis, 2)
             + [int(extend_e2u)] + _field(0, 3) + [int(ack)],
             _field(gain4, 16)]
    if moduli is not None:
        if len(moduli) != 12:
            raise ValueError("twelve moduli")
        for i in range(0, 12, 2):
            words.append(_field(moduli[i], 8) + _field(moduli[i + 1], 8))
    if precoder is not None:
        z1, p1, z2, p2 = (list(precoder.get(k, ())) for k in ("z1", "p1", "z2", "p2"))
        for taps in (z1, p1, z2, p2):
            words.append(_field(len(taps), 9) + _field(0, 7))
        for taps in (z1, p1, z2, p2):
            words.extend(_signed_field(int(t), 16) for t in taps)
    if constellations is not None:
        if interval_sets is None or len(interval_sets) != 6:
            raise ValueError("six interval selectors")
        sets = [list(s) for s in constellations]
        while len(sets) < 6:
            sets.append([])
        if (any(len(s) > MAX_SET_POINTS for s in sets)
                or any(not 0 <= i <= MAX_CONSTELLATION_INDEX for i in interval_sets)
                or any(not sets[i] for i in interval_sets)
                or any(not s and any(sets[j] for j in range(k + 1, 6)) for k, s in enumerate(sets))):
            raise ValueError("invalid constellation sets")
        words.append(sum((_field(i, 4) for i in interval_sets[:4]), []))
        words.append(_field(interval_sets[4], 4) + _field(interval_sets[5], 4) + _field(0, 8))
        for a, b in ((0, 1), (2, 3), (4, 5)):
            words.append(_field(len(sets[a]), 8) + _field(len(sets[b]), 8))
        for s in sets:
            for level in s:
                words.append(_field(int(level), 16))
    return _frame(words)


def with_cpd_gain(bits, gain4: int) -> list[int]:
    """The same CPd with its gain field (``gain4`` = 4*G in unsigned Q0.16) replaced."""
    if not 0 < gain4 < (1 << 16):
        raise ValueError("0 < 4G < 1")
    for words, consumed in _candidates(bits):
        if _parse_cpd_words(words, consumed) is not None:
            words = [list(w) for w in words]
            words[1] = _field(gain4, 16)
            return _frame(words) + [int(b) for b in list(bits)[consumed:]]
    raise ValueError("not a CPd")


def parse_cpd(bits) -> dict | None:
    for words, consumed in _candidates(bits):
        parsed = _parse_cpd_words(words, consumed)
        if parsed is not None:
            return parsed
    return None


def _parse_cpd_words(words, consumed) -> dict | None:
    if len(words) < 2:
        return None
    w1, w2 = words[0], words[1]
    if w1[0] or any(w1[12:15]):
        return None
    has_mod, has_pre, has_const = bool(w1[1]), bool(w1[2]), bool(w1[3])
    drn, trellis = _value(w1, 4, 5), _value(w1, 9, 2)
    if drn > 19 or trellis > 2 or not _value(w2, 0, 16):
        return None
    out = {"kind": "cpd", "drn": drn, "rate": (drn + 17) * RATE_STEP if drn else None,
           "cleardown": drn == 0, "trellis": trellis, "extend_e2u": w1[11], "ack": w1[15],
           "gain": _value(w2, 0, 16) / 65536.0 / 4.0, "frame_bits": consumed}
    index = 2
    if has_mod:
        if len(words) < index + 6:
            return None
        out["moduli"] = tuple(_value(words[index + n // 2], 8 * (n % 2), 8) for n in range(12))
        index += 6
    if has_pre:
        if len(words) < index + 4:
            return None
        counts = []
        for w in words[index:index + 4]:
            if any(w[9:16]):
                return None
            counts.append(_value(w, 0, 9))
        index += 4
        if len(words) < index + sum(counts):
            return None
        taps = []
        for count, scale in zip(counts, (32768.0, 16384.0, 32768.0, 16384.0)):
            taps.append(tuple(_signed(_value(w, 0, 16), 16) / scale
                              for w in words[index:index + count]))
            index += count
        out["precoder"] = dict(zip(("z1", "p1", "z2", "p2"), taps))
    if has_const:
        if len(words) < index + 5:
            return None
        a, b = words[index], words[index + 1]
        if any(b[8:16]):
            return None
        selectors = tuple(_value(a, 4 * i, 4) for i in range(4)) + (_value(b, 0, 4), _value(b, 4, 4))
        lengths = []
        for w in words[index + 2:index + 5]:
            lengths += [_value(w, 0, 8), _value(w, 8, 8)]
        index += 5
        if (max(selectors) > MAX_CONSTELLATION_INDEX or max(lengths) > MAX_SET_POINTS
                or len(words) != index + sum(lengths)):
            return None
        sets = []
        for length in lengths:
            sets.append(tuple(_value(w, 0, 16) for w in words[index:index + length]))
            index += length
        if (any(not sets[i] for i in selectors)
                or any(not s and any(sets[j] for j in range(k + 1, 6)) for k, s in enumerate(sets))
                or any(set_[x] >= set_[x + 1] for set_ in sets for x in range(len(set_) - 1))):
            return None
        out["interval_sets"] = selectors
        out["constellations"] = tuple(sets)
    elif len(words) != index:
        return None
    return out


# -- SUVu (Table 27) -------------------------------------------------------------------------------------------
SUV_NO_LEVEL = 16                   # Q2.2 -4.00: "no measurement has been taken"


def _q22(db: float | None) -> int:
    if db is None:
        return SUV_NO_LEVEL
    code = int(round(db * 4))
    if code == -16:
        code = -15                  # -4.00 is reserved for "no measurement"
    if not -15 <= code <= 15:
        raise ValueError("level must be within -3.75..+3.75 dB")
    return code & 0x1F


def build_suvu(*, level_db: float | None = None, wait_for_cpu: bool = False,
               silent_period: bool = False, ack: bool = False) -> list[int]:
    """SUVu (Table 27): reserved bits 19:25 zero; bit 26 asks the digital modem to wait for a
    CPu; bits 27:31 are 20*log10(L) in signed Q2.2 (16 = unmeasured); bit 32 asks for a silent
    period; bit 33 acknowledges a received CPd."""
    word = [1] + _field(0, 7) + [int(wait_for_cpu)] + _field(_q22(level_db), 5) \
        + [int(silent_period), int(ack)]
    return _frame([word])


def parse_suvu(bits) -> dict | None:
    for words, consumed in _candidates(bits[:SINGLE_WORD_FRAME_BITS]):
        if len(words) != 1:
            continue
        w = words[0]
        if not w[0] or any(w[1:8]):
            return None
        code = _value(w, 9, 5)
        return {"kind": "suvu", "wait_for_cpu": bool(w[8]),
                "level_db": None if code == SUV_NO_LEVEL else _signed(code, 5) / 4.0,
                "silent_period": bool(w[14]), "ack": bool(w[15]), "frame_bits": consumed}
    return None


def build_suvd(*, silent_period: bool = False, ack: bool = False) -> list[int]:
    """SUVd (Table 31): bit 18 = 1, bits 19:31 reserved 0, bit 32 silent-period request,
    bit 33 acknowledges a received CPu."""
    return _frame([[1] + _field(0, 13) + [int(silent_period), int(ack)]])


def parse_suvd(bits) -> dict | None:
    for words, consumed in _candidates(bits[:SINGLE_WORD_FRAME_BITS]):
        if len(words) != 1:
            continue
        w = words[0]
        if not w[0] or any(w[1:14]):
            return None
        return {"kind": "suvd", "silent_period": bool(w[14]), "ack": bool(w[15]),
                "frame_bits": consumed}
    return None
