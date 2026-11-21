"""V.92 INFO signalling bits (ITU-T V.92 clause 8.4.1, Tables 15-19).

Offline parsing/building only; no modem state. V.92 reuses the V.90 INFO frames and
turns on bits that V.90 reserves:

* INFO0d bit 26 requests short Phase 2, bit 27 is V.92 capability (Table 15).
* INFO0a bit 26 is V.92 capability, bit 27 requests short Phase 2 (Table 16); these
  are the two bits V.34 calls ``clock``.
* INFO1d bit 70 (the slot V.34 uses as the 3429 ``hi_carrier`` flag) says whether the
  channel supports PCM upstream (Table 17).
* INFO1a Table 18 selects PCM upstream; Table 19 is the V.34-upstream form and has
  the V.90 INFO1a layout, so ``v90_info_reference`` parses it.

V.92 is used only if both modems set their capability bit; otherwise the V.90 bits
apply (clause 9.3).
"""

from __future__ import annotations

import v34_info as vi
import v90_info_reference as v90

INFO1A_PCM_BITS = 70
INFO1A_PCM_CRC_AT = 50
MD_UNIT_SYMBOLS = 276          # Table 18: MD length unit (34.5 ms), not V.34's 35 ms

# Table 18 bits 12:13 -> which precoder/prefilter sections the analogue modem supports.
FILTER_SECTIONS = {0: ("p1", "z2"), 1: ("z1", "p1", "z2"),
                   2: ("p1", "p2", "z2"), 3: ("z1", "p1", "p2", "z2")}


def info0d_flags(fields: dict) -> dict:
    """V.92 bits of a parsed INFO0d (``v90.parse_info0d``): reserved bit 0 = bit 26."""
    reserved = int(fields.get("reserved", 0))
    return {"short_phase2": bool(reserved & 1), "v92": bool(reserved & 2)}


def info0a_flags(fields: dict) -> dict:
    """V.92 bits of a parsed INFO0a (``vi.parse_info0``): ``clock`` bit 0 = bit 26."""
    clock = int(fields.get("clock", 0))
    return {"v92": bool(clock & 1), "short_phase2": bool(clock & 2)}


def negotiate(info0d: dict, info0a: dict) -> dict:
    """Clause 9.3/9.4: V.92 needs both capability bits; short Phase 2 needs both
    requests on top of that."""
    d, a = info0d_flags(info0d), info0a_flags(info0a)
    v92 = d["v92"] and a["v92"]
    return {"v92": v92, "short_phase2": v92 and d["short_phase2"] and a["short_phase2"]}


def parse_info0a(bits) -> dict | None:
    """INFO0a with the V.92 bits kept (V.90's parser rejects them as reserved)."""
    fields = vi.parse_info0(bits)
    if fields is None or tuple(bits[45:49]) != vi.FILL:
        return None
    return fields


# V.90 Table 8 names for the INFO0a bits whose V.34 Table 12 names differ.
_INFO0A_V90_NAMES = {"disallow3429": "allow3429", "ack_info0d": "ack"}


def build_info0a(*, v92: bool = False, short_phase2: bool = False, **fields) -> list[int]:
    """INFO0a with V.92 flags (``clock`` is bits 26:27). Accepts the V.90 Table 8
    field names the caller profile uses; the V.34 builder would silently drop
    them, losing the INFO0d acknowledgement (bit 28) and bit 19."""
    named = {_INFO0A_V90_NAMES.get(name, name): value for name, value in fields.items()}
    unknown = set(named) - {name for name, _, _ in vi.INFO0_FIELDS} | ({"clock"} & set(named))
    if unknown:
        raise ValueError(f"unknown INFO0a fields: {sorted(unknown)}")
    return vi.build_info0(**named, clock=int(bool(v92)) | (int(bool(short_phase2)) << 1))


def build_info0d(*, v92: bool = False, short_phase2: bool = False, **fields) -> list[int]:
    """INFO0d test vector with V.92 flags (``reserved`` is bits 26:27)."""
    return v90.build_info0d(**fields,
                            reserved=int(bool(short_phase2)) | (int(bool(v92)) << 1))


def pcm_upstream_supported(info1d: dict) -> bool:
    """Table 17 bit 70: 1 = the channel supports PCM upstream."""
    return bool(info1d["blocks"][5]["hi_carrier"])


_PCM_FIELDS = (
    ("sections", 12, 13), ("max_coeffs_code", 14, 15), ("max_section_code", 16, 17),
    ("md_len", 18, 24), ("uinfo", 25, 31), ("reserved_32_33", 32, 33),
    ("analog_symbol_rate", 34, 36), ("digital_symbol_rate", 37, 39),
    ("reserved_ones", 40, 49),
)


def parse_info1a_pcm(bits) -> dict | None:
    """Table 18: INFO1a selecting PCM upstream. Rejects a bad CRC/fill, symbol-rate
    codes other than 6 (8000), and UINFO <= 66."""
    if len(bits) < INFO1A_PCM_BITS or not vi._check(bits, INFO1A_PCM_CRC_AT):
        return None
    if tuple(bits[66:70]) != vi.FILL:
        return None
    fields = {name: v90._value(bits, lo, hi) for name, lo, hi in _PCM_FIELDS}
    if fields["analog_symbol_rate"] != 6 or fields["digital_symbol_rate"] != 6:
        return None
    if fields["uinfo"] <= 66:
        return None
    fields["filter_sections"] = FILTER_SECTIONS[fields["sections"]]
    fields["max_total_coeffs"] = 192 + 64 * fields["max_coeffs_code"]
    fields["max_section_coeffs"] = 128 + 64 * fields["max_section_code"]
    fields["md_symbols"] = fields["md_len"] * MD_UNIT_SYMBOLS
    return fields


def build_info1a_pcm(*, sections: int = 3, max_coeffs_code: int = 0,
                     max_section_code: int = 0, md_len: int = 0, uinfo: int = 67) -> list[int]:
    """Table 18 test vector. Symbol-rate fields are 6 and bits 40:49 are ones."""
    limits = {"sections": (sections, 3), "max_coeffs_code": (max_coeffs_code, 3),
              "max_section_code": (max_section_code, 3), "md_len": (md_len, 127),
              "uinfo": (uinfo, 127)}
    for name, (value, top) in limits.items():
        if not 0 <= int(value) <= top:
            raise ValueError(f"INFO1a PCM field {name} outside 0..{top}")
    if int(uinfo) <= 66:
        raise ValueError("INFO1a PCM UINFO must be greater than 66")
    payload = [0] * 38
    values = {"sections": sections, "max_coeffs_code": max_coeffs_code,
              "max_section_code": max_section_code, "md_len": md_len, "uinfo": uinfo,
              "reserved_32_33": 0, "analog_symbol_rate": 6, "digital_symbol_rate": 6,
              "reserved_ones": (1 << 10) - 1}
    for name, lo, hi in _PCM_FIELDS:
        for bit in range(hi - lo + 1):
            payload[lo - 12 + bit] = (int(values[name]) >> bit) & 1
    return vi._frame(payload)
