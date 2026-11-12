"""V.90 Phase 2 INFO frame parsing and reference-audio decoding.

This module is an offline analysis aid. It reuses the common 600-baud DPSK
sampling and CRC convention from ``v34_info`` but parses V.90-specific fields.
It does not implement a live modem receiver.
"""

from __future__ import annotations

import numpy as np

import v34_info as vi
import v34_g711 as g711
import v90_dil as dil


INFO0D_BITS = 62
INFO0D_CRC_AT = 42
INFO0D_FIELDS = (
    ("rate2743", 12, 12), ("rate2800", 13, 13), ("rate3429", 14, 14),
    ("lo3000", 15, 15), ("hi3000", 16, 16), ("lo3200", 17, 17),
    ("hi3200", 18, 18), ("disallow3429", 19, 19),
    ("power_reduction", 20, 20), ("max_asym", 21, 23), ("cme", 24, 24),
    ("pts1664", 25, 25), ("reserved", 26, 27), ("ack_info0a", 28, 28),
    ("nominal_power", 29, 32), ("max_power", 33, 37),
    ("power_at_codec", 38, 38), ("alaw", 39, 39),
    ("v90_3429_upstream", 40, 40), ("reserved_41", 41, 41),
)

# Table 8/V.90: the analogue modem's INFO0a is a distinct 49-bit frame. Its
# capability fields parallel INFO0d through bit 28, but the CRC begins there;
# the digital modem's power/codec fields in INFO0d are not present in INFO0a.
INFO0A_BITS = 49
INFO0A_CRC_AT = 29
INFO0A_FIELDS = (
    ("rate2743", 12, 12), ("rate2800", 13, 13), ("rate3429", 14, 14),
    ("lo3000", 15, 15), ("hi3000", 16, 16), ("lo3200", 17, 17),
    ("hi3200", 18, 18), ("disallow3429", 19, 19),
    ("power_reduction", 20, 20), ("max_asym", 21, 23), ("cme", 24, 24),
    ("pts1664", 25, 25), ("reserved", 26, 27), ("ack_info0d", 28, 28),
)


def _value(bits, lo: int, hi: int) -> int:
    return sum(int(bits[lo + i]) << i for i in range(hi - lo + 1))


def _signed(value: int, width: int) -> int:
    return value - (1 << width) if value & (1 << (width - 1)) else value


def parse_info0d(bits) -> dict | None:
    """Parse digital INFO0d (V.90 Table 7), including framing and CRC."""
    if len(bits) < INFO0D_BITS or not vi._check(bits, INFO0D_CRC_AT):
        return None
    if tuple(bits[58:62]) != vi.FILL:
        return None
    return {name: _value(bits, lo, hi) for name, lo, hi in INFO0D_FIELDS}


def parse_info0a(bits) -> dict | None:
    """Parse analogue INFO0a (V.90 Table 8), including CRC and fill checks."""
    if len(bits) < INFO0A_BITS or not vi._check(bits, INFO0A_CRC_AT):
        return None
    if tuple(bits[45:49]) != vi.FILL or _value(bits, 26, 27) != 0:
        return None
    return {name: _value(bits, lo, hi) for name, lo, hi in INFO0A_FIELDS}


def parse_info1d(bits) -> dict | None:
    """Parse INFO1d, whose field layout and CRC match V.34 INFO1c/Table 9."""
    return vi.parse_info1c(bits)


def build_info0a(**values) -> list[int]:
    """Build an INFO0a test vector from V.90 Table 8 field names."""
    unknown = set(values) - {name for name, _, _ in INFO0A_FIELDS}
    if unknown:
        raise ValueError(f"unknown INFO0a fields: {sorted(unknown)}")
    if not 0 <= int(values.get("max_asym", 0)) <= 5:
        raise ValueError("INFO0a max_asym must be in 0..5")
    payload = [0] * 17
    for name, lo, hi in INFO0A_FIELDS:
        value = int(values.get(name, 0))
        maximum = (1 << (hi - lo + 1)) - 1
        if not 0 <= value <= maximum:
            raise ValueError(f"INFO0a field {name} outside 0..{maximum}")
        for bit in range(hi - lo + 1):
            payload[lo - 12 + bit] = (value >> bit) & 1
    if int(values.get("reserved", 0)) != 0:
        raise ValueError("INFO0a reserved bits must be zero")
    return vi._frame(payload)


def parse_info1a_v90(bits) -> dict | None:
    """Parse the V.90-selecting INFO1a variant (V.90 Table 10)."""
    if len(bits) < 70 or not vi._check(bits, 50):
        return None
    fields = {
        "reserved_12_17": _value(bits, 12, 17),
        "md_len": _value(bits, 18, 24),
        "uinfo": _value(bits, 25, 31),
        "reserved_32_33": _value(bits, 32, 33),
        "analog_symbol_rate": _value(bits, 34, 36),
        "digital_symbol_rate": _value(bits, 37, 39),
        "freq_offset": _signed(_value(bits, 40, 49), 10),
    }
    if fields["digital_symbol_rate"] != 6:
        return None
    return fields


def build_info0d(**values) -> list[int]:
    """Build an INFO0d test vector. Keyword names are the V.90 Table 7 fields."""
    payload = [0] * 30
    for name, lo, hi in INFO0D_FIELDS:
        value = int(values.get(name, 0))
        for bit in range(hi - lo + 1):
            payload[lo - 12 + bit] = (value >> bit) & 1
    return vi._frame(payload)


def build_info1a_v90(**values) -> list[int]:
    """Build a V.90-selecting INFO1a test vector (digital symbol rate is 6)."""
    payload = [0] * 38
    fields = (
        ("reserved_12_17", 12, 17), ("md_len", 18, 24), ("uinfo", 25, 31),
        ("reserved_32_33", 32, 33), ("analog_symbol_rate", 34, 36),
        ("digital_symbol_rate", 37, 39), ("freq_offset", 40, 49),
    )
    for name, lo, hi in fields:
        value = int(values.get(name, 0))
        if name == "freq_offset":
            value &= 0x3FF
        for bit in range(hi - lo + 1):
            payload[lo - 12 + bit] = (value >> bit) & 1
    payload[37 - 12:40 - 12] = [0, 1, 1]  # six, least-significant bit first
    return vi._frame(payload)


def parse_ja_descriptor(bits) -> dict | None:
    """Parse one CRC-protected V.90 Ja DIL descriptor (V.90 §8.3.1/Table 12).

    ``bits`` must start at the 17-one descriptor synchronization pattern.  The
    CRC covers descriptor information bits, excluding sync, every group-start
    marker, the terminal start marker, CRC itself, and fill.  Reserved bits are
    returned for diagnostics instead of rejected: captures can establish
    framing/CRC even when a vendor uses a permitted reserved-field value.
    """
    if len(bits) < 204 or tuple(int(x) for x in bits[:17]) != (1,) * 17:
        return None
    b = [int(x) for x in bits]
    if any(x not in (0, 1) for x in b):
        return None
    # Bit 50 is reserved (Table 12), not a start bit.  The V.34 CRC rule
    # excludes start bits, sync, and fill, but includes reserved information
    # bits.  This distinction matters for implementations that transmit a
    # zero in bit 50: it is still part of the CRC input.
    starts = {17, 34}
    if any(b[i] != 0 for i in (17, 34)):
        return None

    n = _value(b, 18, 25)
    lsp = _value(b, 35, 41) + 1
    ltp = _value(b, 43, 49) + 1
    alpha = ((lsp + 15) // 16) * 17
    beta = alpha + ((ltp + 15) // 16) * 17
    sp_start = 51
    tp_start = sp_start + alpha
    h_start = tp_start + ((ltp + 15) // 16) * 17
    ref_start = h_start + 4 * 17
    u_start = ref_start + 4 * 17
    terminal = u_start + ((n + 1) // 2) * 17
    crc_start = terminal + 1
    end = crc_start + 16
    if len(b) < end or b[terminal] != 0:
        return None
    starts.add(terminal)

    def unpack_pattern(start: int, count: int) -> list[int]:
        out = []
        for group in range((count + 15) // 16):
            base = start + group * 17
            starts.add(base)
            out.extend(b[base + 1:base + 17])
        return out[:count]

    sp = unpack_pattern(sp_start, lsp)
    tp = unpack_pattern(tp_start, ltp)

    def unpack_pairs(start: int, count: int) -> list[int]:
        out = []
        for group in range(4):
            base = start + group * 17
            starts.add(base)
            out.extend(( _value(b, base + 1, base + 7),
                         _value(b, base + 9, base + 15) ))
        return out[:count]

    h = unpack_pairs(h_start, 8)
    ref = unpack_pairs(ref_start, 8)
    ucodes = []
    for group in range((n + 1) // 2):
        base = u_start + group * 17
        starts.add(base)
        ucodes.append(_value(b, base + 1, base + 7))
        if len(ucodes) < n:
            ucodes.append(_value(b, base + 9, base + 15))

    information = [b[i] for i in range(18, terminal) if i not in starts]
    crc_got = _value(b, crc_start, crc_start + 15)
    crc_expected = vi.crc16(information)
    fill = b[end:end + 2]
    all_starts_zero = all(b[i] == 0 for i in starts)
    reserved = set(range(26, 34)) | {42, 50}
    for group in range((lsp + 15) // 16):
        used = min(16, lsp - group * 16)
        reserved.update(range(sp_start + group * 17 + 1 + used, sp_start + group * 17 + 17))
    for group in range((ltp + 15) // 16):
        used = min(16, ltp - group * 16)
        reserved.update(range(tp_start + group * 17 + 1 + used, tp_start + group * 17 + 17))
    for base in [h_start + 17 * i for i in range(4)] + [ref_start + 17 * i for i in range(4)]:
        reserved.update((base + 8, base + 16))
    for group in range((n + 1) // 2):
        base = u_start + group * 17
        reserved.add(base + 8)
        reserved.update(range(base + 16, base + 17))
        if 2 * group + 1 >= n:
            reserved.update(range(base + 9, base + 17))
    reserved_bits_zero = all(b[i] == 0 for i in reserved)
    return {
        "valid": (crc_got == crc_expected and all_starts_zero and reserved_bits_zero and len(fill) >= 1
                  and all(x == 0 for x in fill)),
        "crc_valid": crc_got == crc_expected,
        "start_bits_valid": all_starts_zero,
        "reserved_bits_zero": reserved_bits_zero,
        "crc": crc_got,
        "crc_expected": crc_expected,
        "N": n,
        "LSP": lsp,
        "LTP": ltp,
        "SP": sp,
        "TP": tp,
        "H": h,
        "REF": ref,
        "Ucodes": ucodes,
        "reserved_header": b[26:34],
        "reserved_lengths": [b[42], b[50]],
        "crc_start": crc_start,
        "descriptor_bits": end + len(fill),
        "fill": fill,
    }


def build_ja_descriptor(*, N: int, LSP: int = 1, LTP: int = 1, SP=None, TP=None,
                        H=None, REF=None, Ucodes=None) -> list[int]:
    """Build a V.90 Ja descriptor test vector from its decoded parameters.

    This is a small spec-derived fixture builder for parser tests and local
    transmitter experiments, not a production Ja generator.
    """
    if not 0 <= N <= 255 or not 1 <= LSP <= 128 or not 1 <= LTP <= 128:
        raise ValueError("Ja N/LSP/LTP outside V.90 field limits")
    SP = list(SP if SP is not None else [0] * LSP)
    TP = list(TP if TP is not None else [0] * LTP)
    H = list(H if H is not None else [0] * 8)
    REF = list(REF if REF is not None else [0] * 8)
    Ucodes = list(Ucodes if Ucodes is not None else [0] * N)
    if len(SP) != LSP or len(TP) != LTP or len(H) != 8 or len(REF) != 8 or len(Ucodes) != N:
        raise ValueError("Ja patterns/tables must match LSP, LTP, and N")
    if any(x not in (0, 1) for x in SP + TP):
        raise ValueError("SP/TP values must be bits")
    if any(not 0 <= x <= 127 for x in H + REF + Ucodes):
        raise ValueError("H, REF, and Ucode values must be 7-bit integers")

    alpha = ((LSP + 15) // 16) * 17
    beta = alpha + ((LTP + 15) // 16) * 17
    u_start = 187 + beta
    terminal = u_start + ((N + 1) // 2) * 17
    crc_start = terminal + 1
    out = [0] * (crc_start + 18)
    out[:17] = [1] * 17
    out[18:26] = vi._field(N, 8)
    out[35:42] = vi._field(LSP - 1, 7)
    out[43:50] = vi._field(LTP - 1, 7)
    starts = {17, 34, terminal}

    def put_pattern(start, values):
        for group in range((len(values) + 15) // 16):
            base = start + group * 17
            starts.add(base)
            count = values[group * 16:(group + 1) * 16]
            out[base + 1:base + 1 + len(count)] = count

    put_pattern(51, SP)
    put_pattern(51 + alpha, TP)

    h_start = 51 + beta
    for group in range(4):
        base = h_start + group * 17
        starts.add(base)
        out[base + 1:base + 8] = vi._field(H[2 * group], 7)
        out[base + 9:base + 16] = vi._field(H[2 * group + 1], 7)
    ref_start = h_start + 68
    for group in range(4):
        base = ref_start + group * 17
        starts.add(base)
        out[base + 1:base + 8] = vi._field(REF[2 * group], 7)
        out[base + 9:base + 16] = vi._field(REF[2 * group + 1], 7)
    for group in range((N + 1) // 2):
        base = u_start + group * 17
        starts.add(base)
        out[base + 1:base + 8] = vi._field(Ucodes[2 * group], 7)
        if 2 * group + 1 < N:
            out[base + 9:base + 16] = vi._field(Ucodes[2 * group + 1], 7)
    information = [out[i] for i in range(18, terminal) if i not in starts]
    out[crc_start:crc_start + 16] = vi._field(vi.crc16(information), 16)
    if len(out) & 1:
        out.append(0)
    return out


expand_dil = dil.expand_dil
g711_ucode_codewords = dil.g711_ucode_codewords
dil_g711_codewords = dil.dil_g711_codewords
dil_g711_segments = dil.dil_g711_segments
dil_codewords_match = dil.dil_codewords_match
V90DILCodewordStream = dil.V90DILCodewordStream


def dil_linear_samples(descriptor: dict, law: str) -> np.ndarray:
    """Map one expanded DIL pass to signed G.711 linear reconstruction values."""
    codewords = np.frombuffer(dil_g711_codewords(descriptor, law), dtype=np.uint8)
    return g711.decode(codewords, law).astype(np.int32)


def find_ja_descriptors(bits, *, include_bad_crc: bool = False) -> list[dict]:
    """Find repeated or isolated Ja descriptors in a descrambled bit stream."""
    stream = [int(x) for x in bits]
    found = []
    sync = [1] * 17 + [0]
    i = 0
    while i <= len(stream) - 204:
        if stream[i:i + 18] != sync:
            i += 1
            continue
        parsed = parse_ja_descriptor(stream[i:])
        if parsed is not None and (include_bad_crc or parsed["valid"]):
            parsed["start_bit"] = i
            found.append(parsed)
            i += max(1, parsed["descriptor_bits"])
        else:
            i += 1
    return found


def decode_info0d(signal, sender_role: str, phases: int = 16) -> list[dict]:
    """Find CRC-valid INFO0d frames in an 8 kHz waveform."""
    return _decode_frames(signal, sender_role, INFO0D_BITS, INFO0D_CRC_AT, parse_info0d, phases)


def decode_info0a(signal, sender_role: str = "answer", phases: int = 16) -> list[dict]:
    """Find CRC-valid analogue INFO0a frames in an 8 kHz waveform."""
    return _decode_frames(signal, sender_role, INFO0A_BITS, INFO0A_CRC_AT,
                          parse_info0a, phases)


def decode_info1d(signal, sender_role: str = "call", phases: int = 16) -> list[dict]:
    """Find CRC-valid digital INFO1d frames using the V.34 INFO1c layout."""
    return _decode_frames(signal, sender_role, 109, 89, parse_info1d, phases)


def decode_v90_info1a(signal, sender_role: str = "answer", phases: int = 16) -> list[dict]:
    """Find CRC-valid INFO1a frames that select V.90."""
    return _decode_frames(signal, sender_role, 70, 50, parse_info1a_v90, phases)


def _decode_frames(signal, sender_role, frame_bits, crc_at, parser, phases):
    demod = vi.InfoDemodulator(sender_role, phases=phases)
    z = demod._baseband(np.asarray(signal, dtype=float))
    spb = vi.FS / vi.BAUD
    n_sym = int((len(z) - 1) / spb) - 1
    pattern = list(vi.FILL) + list(vi.SYNC)
    found = []
    for phase in range(phases):
        positions = phase * spb / phases + np.arange(n_sym) * spb
        indices = np.clip(positions.astype(int), 0, len(z) - 2)
        symbols = z[indices] * (1 - (positions - indices)) + z[indices + 1] * (positions - indices)
        differential = symbols[1:] * np.conj(symbols[:-1])
        bits = (differential.real < 0).astype(np.uint8).tolist()
        for start in range(len(bits) - frame_bits + 1):
            if bits[start:start + 12] != pattern:
                continue
            frame = bits[start:start + frame_bits]
            parsed = parser(frame)
            if parsed is None:
                continue
            parsed["_start_sample"] = (start + 1 + phase / phases) * spb
            parsed["_phase"] = phase
            found.append(parsed)
    return _deduplicate(found)


def _deduplicate(frames, tolerance_samples: float = 50.0):
    """Collapse timing-phase copies of one frame while retaining distinct repeats."""
    frames.sort(key=lambda frame: frame["_start_sample"])
    unique = []
    for frame in frames:
        if unique and frame["_start_sample"] - unique[-1]["_start_sample"] <= tolerance_samples:
            continue
        unique.append(frame)
    return unique
