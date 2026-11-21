"""V.92 short Phase 2, ranging only (ITU-T V.92 clause 9.4, Figure 9), as an event timeline.

Used when both modems set their V.92 capability bits and both ask for it (INFO0d bit 26 and INFO0a bit 27;
``v92_info.negotiate``). Both send INFO0 right after the 75 ms of silence that ends Phase 1 (bit 28 clear), then
Tone B (digital) and Tone A (analogue). The digital modem, once it has INFO0a, has detected Tone A and has sent
Tone B for at least 50 ms, reverses Tone B and keeps it 10 ms more; the analogue modem answers the reversal with a
Tone A reversal 40 ms after the Tone B reversal reached it, keeps Tone A 10 ms and sends INFO1a. The digital
modem's round-trip estimate RTDEd is the time from its Tone B reversal to the Tone A reversal reaching it, minus
40 ms. There are no L1 / L2 probes and no INFO1d: the line is not characterised here, which is why the analogue
modem may ask for short Phase 2 only if it will connect in PCM upstream or V.90 data mode.

The INFO0 frames are the real bit sequences (``v92_info``) and are parsed on arrival; their durations are those of the
INFO modulation (600 baud). Tones and reversals are events here, not signals. ``turnaround_error_ms`` models the +-1 ms
tolerance of the analogue modem's 40 ms.
"""

from __future__ import annotations

import v90_info_reference as v90
import v92_info

INFO_BAUD = 600
INFO0D_BITS = 62
INFO0A_BITS = 49
TONE_B_MIN_MS = 50
REVERSAL_HOLD_MS = 10
TURNAROUND_MS = 40


def info_ms(bits: int) -> int:
    return round(bits * 1000 / INFO_BAUD)


def run(down_ms: int, up_ms: int, *, short_digital: bool = True, short_analogue: bool = True,
        v92_digital: bool = True, v92_analogue: bool = True, turnaround_error_ms: int = 0,
        start_ms: int = 0) -> dict:
    """One short (or, if not negotiated, a refusal to be short) Phase 2 between the two modems.
    ``down_ms`` is the one-way delay digital to analogue, ``up_ms`` analogue to digital."""
    d_bits = v92_info.build_info0d(v92=v92_digital, short_phase2=short_digital, rate3429=1)
    a_bits = v92_info.build_info0a(v92=v92_analogue, short_phase2=short_analogue, rate3429=1)
    d_fields = v90.parse_info0d(d_bits)
    a_fields = v92_info.parse_info0a(a_bits)
    if d_fields is None or a_fields is None:
        raise ValueError("INFO0 frame does not parse")
    negotiated = v92_info.negotiate(d_fields, a_fields)
    out = {"negotiated": negotiated, "events": []}
    if not negotiated["short_phase2"]:
        out["reason"] = "short Phase 2 needs both modems to be V.92 and both to ask for it"
        return out
    ev = out["events"]
    t = start_ms
    d_info_end = t + info_ms(len(d_bits))
    a_info_end = t + info_ms(len(a_bits))
    ev += [(t, "digital", "INFO0d"), (t, "analogue", "INFO0a"),
           (d_info_end, "digital", "tone B"), (a_info_end, "analogue", "tone A")]
    # The digital modem has INFO0a and Tone A when both have reached it, and needs Tone B on the line 50 ms.
    a_known_at_digital = a_info_end + up_ms
    b_reversal = max(d_info_end + TONE_B_MIN_MS, a_known_at_digital)
    ev.append((b_reversal, "digital", "tone B reversal"))
    ev.append((b_reversal + REVERSAL_HOLD_MS, "digital", "silence"))
    b_arrives = b_reversal + down_ms
    a_reversal = b_arrives + TURNAROUND_MS + turnaround_error_ms
    ev.append((b_arrives, "analogue", "tone B reversal detected"))
    ev.append((a_reversal, "analogue", "tone A reversal"))
    a_arrives = a_reversal + up_ms
    ev.append((a_arrives, "digital", "tone A reversal detected"))
    info1a_start = a_reversal + REVERSAL_HOLD_MS
    ev.append((info1a_start, "analogue", "INFO1a"))
    out["rtd_ms"] = (a_arrives - b_reversal) - TURNAROUND_MS
    out["info1a_at_digital_ms"] = info1a_start + up_ms
    out["events"].sort(key=lambda e: e[0])
    return out
