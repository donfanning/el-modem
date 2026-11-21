"""V.92 modem-on-hold MH sequences (ITU-T V.92 clause 8.9.2, Tables 32 and 33).

An MH sequence is a 40-bit INFO-style frame (same modulation as Phase 2 INFO): 4 fill ones,
the 01110010 sync, four signal-indication bits, four information bits, the V.34 CRC-16 and four
fill ones. Undefined indication codes must be ignored by a receiver (Note 1), so ``parse_mh``
returns None for them rather than guessing.
"""

from __future__ import annotations

import v34_info as vi

SIGNALS = {"MHreq": 0b0011, "MHack": 0b0101, "MHnack": 0b0111,
           "MHclrd": 0b1001, "MHcda": 0b1011, "MHfrr": 0b1101}
_NAMES = {code: name for name, code in SIGNALS.items()}
# Table 33; codes 0000, 1110 and 1111 are reserved for the ITU.
TIMEOUT_SECONDS = {1: 10, 2: 20, 3: 30, 4: 40, 5: 60, 6: 120, 7: 180, 8: 240, 9: 360,
                   10: 480, 11: 720, 12: 960, 13: None}      # None = no limit
CLEARDOWN_REASONS = {"incoming call": 0b0101, "outgoing call": 0b0110, "other": 0b1010}
_REASON_NAMES = {code: name for name, code in CLEARDOWN_REASONS.items()}


def _bits(value: int, width: int = 4) -> list[int]:
    return [(value >> i) & 1 for i in range(width)]


def build_mh(signal: str, *, timeout_code: int | None = None,
             cleardown_reason: str | None = None) -> list[int]:
    """Build an MH frame. MHack carries the on-hold timeout code (Table 33), MHclrd the
    cleardown reason; every other signal repeats its own indication bits."""
    if signal not in SIGNALS:
        raise ValueError(f"unknown MH signal {signal!r}")
    code = SIGNALS[signal]
    if signal == "MHack":
        if timeout_code not in TIMEOUT_SECONDS:
            raise ValueError("MHack needs a timeout code 1..13 (Table 33)")
        info = timeout_code
    elif signal == "MHclrd":
        if cleardown_reason not in CLEARDOWN_REASONS:
            raise ValueError("MHclrd needs a cleardown reason")
        info = CLEARDOWN_REASONS[cleardown_reason]
    else:
        if timeout_code is not None or cleardown_reason is not None:
            raise ValueError(f"{signal} carries no extra information")
        info = code
    return vi._frame(_bits(code) + _bits(info))


def parse_mh(bits) -> dict | None:
    if len(bits) < 40 or not vi._check(bits, 20) or tuple(bits[36:40]) != vi.FILL:
        return None
    code = sum(int(bits[12 + i]) << i for i in range(4))
    info = sum(int(bits[16 + i]) << i for i in range(4))
    name = _NAMES.get(code)
    if name is None:
        return None                                    # undefined: to be ignored
    out = {"signal": name}
    if name == "MHack":
        if info not in TIMEOUT_SECONDS:
            return None                                # reserved timeout code
        out["timeout_code"] = info
        out["timeout_seconds"] = TIMEOUT_SECONDS[info]
    elif name == "MHclrd":
        out["cleardown_reason"] = _REASON_NAMES.get(info, "reserved")
    elif info != code:
        return None                                    # must repeat the indication bits
    return out
