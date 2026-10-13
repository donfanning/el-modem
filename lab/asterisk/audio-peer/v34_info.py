"""V.34 phase 2 INFO sequences (ITU-T V.34 10.1.2.3): framing, CRC, 600 bit/s DPSK modulation and demodulation.

Written from the V.34 text only. INFO0, INFO1c and INFO1a are bit strings, bit 0 first: 4 fill ones, the frame sync
01110010, the payload, a 16-bit CRC (x^16+x^12+x^5+1, register preset to ones, bit 0 = LSB sent first) and 4 fill
ones. They are sent as binary DPSK at 600 baud: a 1 turns the carrier phase by 180 degrees, a 0 by 0 degrees. The
call modem uses a 1200 Hz carrier, the answer modem 2400 Hz (1 dB down, plus a 1800 Hz guard tone 7 dB down).
"""

from __future__ import annotations

import math

import numpy as np

FS = 8000
BAUD = 600
SYNC = (0, 1, 1, 1, 0, 0, 1, 0)
FILL = (1, 1, 1, 1)
CARRIER = {"call": 1200.0, "answer": 2400.0}


def crc16(bits) -> int:
    reg = 0xFFFF
    for b in bits:
        fb = (reg & 1) ^ b
        reg >>= 1
        if fb:
            reg ^= 0x8408
    return reg


def _field(value: int, width: int) -> list[int]:
    return [(value >> i) & 1 for i in range(width)]


def _value(bits, lo: int, hi: int) -> int:
    return sum(bits[lo + i] << i for i in range(hi - lo + 1))


def _frame(payload: list[int]) -> list[int]:
    """payload = the information bits after the frame sync and before the CRC."""
    crc = crc16(payload)
    return list(FILL) + list(SYNC) + payload + _field(crc, 16) + list(FILL)


# -- INFO0 (Table 14): payload is bits 12..28 ---------------------------------------------------------------
INFO0_FIELDS = (                       # name, lowest bit, highest bit (absolute positions in the frame)
    ("rate2743", 12, 12), ("rate2800", 13, 13), ("rate3429", 14, 14),
    ("lo3000", 15, 15), ("hi3000", 16, 16), ("lo3200", 17, 17), ("hi3200", 18, 18),
    ("allow3429", 19, 19), ("power_reduction", 20, 20), ("max_asym", 21, 23), ("cme", 24, 24),
    ("pts1664", 25, 25), ("clock", 26, 27), ("ack", 28, 28),
)


def build_info0(**kw) -> list[int]:
    payload = [0] * 17
    for name, lo, hi in INFO0_FIELDS:
        v = int(kw.get(name, 0))
        for i in range(hi - lo + 1):
            payload[lo - 12 + i] = (v >> i) & 1
    return _frame(payload)


def parse_info0(bits) -> dict | None:
    if len(bits) < 49 or not _check(bits, 29):
        return None
    return {name: _value(bits, lo, hi) for name, lo, hi in INFO0_FIELDS}


# -- INFO1c (Table 15) and INFO1a (Table 16) -----------------------------------------------------------------
def _rate_block(d: dict) -> list[int]:
    """9 bits: high carrier, pre-emphasis index (4), projected maximum rate (4)."""
    return _field(d.get("hi_carrier", 0), 1) + _field(d.get("preemph", 0), 4) + _field(d.get("max_rate", 0), 4)


def build_info1c(power_min=0, power_extra=0, md_len=0, blocks=None, freq_offset=0) -> list[int]:
    """blocks: five dicts (hi_carrier, preemph, max_rate) for symbol rates 2400, 2743, 2800, 3000, 3200, 3429 -> six."""
    blocks = blocks or [{}] * 6
    payload = _field(power_min, 3) + _field(power_extra, 3) + _field(md_len, 7)
    for blk in blocks:
        payload += _rate_block(blk)
    payload += _field(freq_offset & 0x3FF, 10)
    return _frame(payload)


def parse_info1c(bits) -> dict | None:
    if len(bits) < 109 or not _check(bits, 89):
        return None
    out = {"power_min": _value(bits, 12, 14), "power_extra": _value(bits, 15, 17), "md_len": _value(bits, 18, 24),
           "blocks": []}
    for i in range(6):
        lo = 25 + 9 * i
        out["blocks"].append({"hi_carrier": bits[lo], "preemph": _value(bits, lo + 1, lo + 4),
                              "max_rate": _value(bits, lo + 5, lo + 8)})
    out["freq_offset"] = _signed(_value(bits, 79, 88), 10)
    return out


def build_info1a(power_min=0, power_extra=0, md_len=0, hi_carrier=0, preemph=0, max_rate=0, sym_ans=0, sym_call=0,
                 freq_offset=0) -> list[int]:
    payload = (_field(power_min, 3) + _field(power_extra, 3) + _field(md_len, 7) + _field(hi_carrier, 1)
               + _field(preemph, 4) + _field(max_rate, 4) + _field(sym_ans, 3) + _field(sym_call, 3)
               + _field(freq_offset & 0x3FF, 10))
    return _frame(payload)


def info1a_fields(bits) -> dict:
    """INFO1a payload fields from a frame's bits, without any CRC check (see ``recover_info1a``)."""
    return {"power_min": _value(bits, 12, 14), "power_extra": _value(bits, 15, 17), "md_len": _value(bits, 18, 24),
            "hi_carrier": bits[25], "preemph": _value(bits, 26, 29), "max_rate": _value(bits, 30, 33),
            "sym_ans": _value(bits, 34, 36), "sym_call": _value(bits, 37, 39),
            "freq_offset": _signed(_value(bits, 40, 49), 10)}


def parse_info1a(bits) -> dict | None:
    if len(bits) < 70 or not _check(bits, 50):
        return None
    return info1a_fields(bits)


def _signed(v: int, width: int) -> int:
    return v - (1 << width) if v >= 1 << (width - 1) else v


def _check(bits, crc_at: int) -> bool:
    if tuple(bits[:4]) != FILL or tuple(bits[4:12]) != SYNC:
        return False
    return _value(bits, crc_at, crc_at + 15) == crc16(bits[12:crc_at])


FRAME_LENGTH = {"info0": 49, "info1c": 109, "info1a": 70}
PARSERS = {"info0": parse_info0, "info1c": parse_info1c, "info1a": parse_info1a}


# -- modulation -----------------------------------------------------------------------------------------------
def _pulse(rolloff: float = 0.5, span: int = 6, sps: int = 40) -> np.ndarray:
    """Raised-cosine pulse for the 600 baud symbols, sampled at ``sps`` points per symbol."""
    t = np.arange(-span * sps, span * sps + 1) / sps
    with np.errstate(divide="ignore", invalid="ignore"):
        h = np.sinc(t) * np.cos(math.pi * rolloff * t) / (1 - (2 * rolloff * t) ** 2)
    h[np.isclose(np.abs(2 * rolloff * t), 1)] = math.pi / 4 * np.sinc(1 / (2 * rolloff))
    return h


class InfoModulator:
    """Generate the line signal for a list of INFO bits (preceded by a reference point), at 8 kHz."""

    def __init__(self, role: str, amplitude: float = 3000.0, guard: bool = True):
        self.role = role
        self.amp = amplitude * (10 ** (-1 / 20) if role == "answer" else 1.0)
        self.guard = guard and role == "answer"

    def modulate(self, bits, tail_bits: int = 8, start_phase: float = 0.0) -> np.ndarray:
        symbols = [1.0]                                    # reference point
        sign = 1.0
        for b in list(bits) + [0] * tail_bits:
            if b:
                sign = -sign
            symbols.append(sign)
        sps = 40
        up = np.zeros(len(symbols) * sps + 1)
        up[::sps][:len(symbols)] = symbols
        base = np.convolve(up, _pulse(sps=sps))
        # resample the 24 kHz-equivalent (40 samples/symbol) pulse train to 8 kHz: 600 baud = 13.33 samples/symbol
        n_out = int(len(symbols) * FS / BAUD)
        pos = np.arange(n_out) * (BAUD * sps / FS) + 6 * sps
        idx = np.clip(pos.astype(int), 0, len(base) - 2)
        frac = pos - idx
        env = base[idx] * (1 - frac) + base[idx + 1] * frac
        t = np.arange(n_out) / FS
        sig = self.amp * env * np.cos(2 * math.pi * CARRIER[self.role] * t + start_phase)
        if self.guard:
            sig = sig + self.amp * 10 ** (-6 / 20) * np.cos(2 * math.pi * 1800.0 * t)
        return sig


class InfoDemodulator:
    """Find INFO frames in a received signal by trying symbol-timing phases and checking sync + CRC."""

    def __init__(self, sender_role: str, carrier_offset_hz: float = 0.0, phases: int = 24):
        self.fc = CARRIER[sender_role] + carrier_offset_hz
        self.phases = phases

    def _baseband(self, x: np.ndarray) -> np.ndarray:
        n = np.arange(len(x))
        z = x * np.exp(-2j * math.pi * self.fc * n / FS)
        taps = 63                                          # low-pass, ~ +-420 Hz
        k = np.arange(taps) - taps // 2
        h = np.sinc(2 * 420.0 / FS * k) * np.hamming(taps)
        h /= h.sum()
        return np.convolve(z, h, mode="same")

    def decode(self, x: np.ndarray, kind: str) -> list[dict]:
        z = self._baseband(np.asarray(x, dtype=float))
        spb = FS / BAUD
        found = []
        n_sym = int((len(z) - 1) / spb) - 1
        for ph in range(self.phases):
            t0 = ph * spb / self.phases
            pos = t0 + np.arange(n_sym) * spb
            i = np.clip(pos.astype(int), 0, len(z) - 2)
            s = z[i] * (1 - (pos - i)) + z[i + 1] * (pos - i)
            d = s[1:] * np.conj(s[:-1])
            bits = (d.real < 0).astype(int).tolist()        # 180 degree turn = 1
            frames = self._scan(bits, kind, ph)
            for frame in frames:
                start = (frame["_bit_index"] + 1 + ph / self.phases) * spb
                frame["_start_sample"] = start
                frame["_end_sample"] = start + FRAME_LENGTH[kind] * spb
            found += frames
        return found

    @staticmethod
    def _scan(bits: list[int], kind: str, phase: int) -> list[dict]:
        n = FRAME_LENGTH[kind]
        pat = list(FILL) + list(SYNC)
        out = []
        for i in range(len(bits) - n + 1):
            if bits[i:i + 12] == pat:
                parsed = PARSERS[kind](bits[i:i + n])
                if parsed is not None:
                    parsed["_bit_index"], parsed["_phase"] = i, phase
                    out.append(parsed)
        return out


def _info1a_sync_groups(x, phases: int = 8):
    """Groups of sync-pattern hits for INFO1a frames, one group per physical frame.

    One frame shows up at several symbol-timing phases, sometimes on adjacent bit indices (the phase offset wraps the
    index), so hits are grouped by physical position: chains of hits within half a bit. Each hit is
    (bit position, phase, start sample, the 70 frame bits)."""
    x = np.asarray(x, dtype=float)
    demod = InfoDemodulator("answer", phases=phases)
    z = demod._baseband(x)
    spb = FS / BAUD
    n_sym = int((len(z) - 1) / spb) - 1
    n = FRAME_LENGTH["info1a"]
    pat = list(FILL) + list(SYNC)
    hits = []
    for ph in range(phases):
        pos = ph * spb / phases + np.arange(n_sym) * spb
        i = np.clip(pos.astype(int), 0, len(z) - 2)
        s = z[i] * (1 - (pos - i)) + z[i + 1] * (pos - i)
        d = s[1:] * np.conj(s[:-1])
        bits = (d.real < 0).astype(int).tolist()
        for k in range(len(bits) - n + 1):
            if bits[k:k + 12] == pat:
                hits.append((k + ph / phases, ph, (k + 1 + ph / phases) * spb, bits[k:k + n]))
    hits.sort(key=lambda h: h[0])
    groups: list[list] = []
    for h in hits:
        if groups and h[0] - groups[-1][-1][0] <= 0.5:
            groups[-1].append(h)
        else:
            groups.append([h])
    return groups


def _plausible_info1a(f: dict) -> bool:
    return f["sym_ans"] <= 5 and f["sym_call"] <= 5 and 1 <= f["max_rate"] <= 14 and f["power_min"] <= 7


def recover_info1a(x, phases: int = 8, min_phases: int = 6, min_dropout_s: float = 0.008,
                   quiet_fraction: float = 0.05) -> dict | None:
    """Opt-in recovery of an INFO1a whose CRC was destroyed by a network media dropout.

    A real PSTN/RTP path skips 20 ms or more of audio now and then. INFO1a is a single 70-bit (117 ms) frame whose
    payload ends at bit 49; a dropout over the CRC (bits 50-65) or the trailing fill bits leaves the payload
    intact but fails the CRC, so ``InfoDemodulator.decode`` never returns it. This accepts such a frame only when
    all of the following hold, and marks it ``_crc_unverified``:

    * the sync pattern is found at ``min_phases`` or more symbol-timing phases with the *same* payload bits;
    * the payload fields are plausible (symbol-rate indices 0-5, a maximum rate of 1-14, power reduction 0-7);
    * a measured dropout (a run of near-silence of at least ``min_dropout_s``) starts at or after the end of the
      payload (bit 50) and overlaps the CRC/fill bits;
    * the payload itself contains no near-silent stretch of 2 ms or more.

    It is not a CRC substitute: a frame whose payload was damaged by noise, or with no dropout over its tail, is
    rejected. Returns the parsed fields (plus ``_start_sample`` / ``_end_sample`` relative to ``x``) or None.
    """
    x = np.asarray(x, dtype=float)
    spb = FS / BAUD
    n = FRAME_LENGTH["info1a"]
    groups = _info1a_sync_groups(x, phases)
    best = None
    for group in groups:
        members = [(ph, start, bits) for _, ph, start, bits in group]
        if any(parse_info1a(b) is not None for _, _, b in members):
            continue                                          # a CRC-valid frame exists: the normal path takes it
        counts: dict[tuple, int] = {}
        for _, _, b in members:
            counts[tuple(b[12:50])] = counts.get(tuple(b[12:50]), 0) + 1
        payload, support = max(counts.items(), key=lambda kv: kv[1])
        if support < min_phases:
            continue
        ph, start, bits = next(m for m in members if tuple(m[2][12:50]) == payload)
        f = info1a_fields(bits)
        if not _plausible_info1a(f):
            continue
        # Dropout evidence: runs of near-silence, measured against the frame's own carrier level.
        a, b50, end = int(start + 12 * spb), int(start + 50 * spb), int(start + n * spb)
        ref = float(np.sqrt(np.mean(x[a:b50] ** 2))) if b50 > a else 0.0
        if ref < 50.0 or end + 4 > len(x):
            continue
        peak = np.maximum.reduce([np.abs(x[o:len(x) - 4 + o]) for o in range(5)])      # 5-sample moving maximum
        quiet = peak < quiet_fraction * ref
        run_start, run = None, []
        for j in range(max(0, a - 8), min(len(quiet), end + int(0.04 * FS))):
            if quiet[j] and run_start is None:
                run_start = j
            elif not quiet[j] and run_start is not None:
                run.append((run_start, j)); run_start = None
        if run_start is not None:
            run.append((run_start, min(len(quiet), end + int(0.04 * FS))))
        if any(r1 - r0 >= int(0.002 * FS) for r0, r1 in run if r0 < b50 and r1 > a):
            continue                                          # near-silence inside the payload: do not trust it
        tail = [(r0, r1) for r0, r1 in run if r0 >= b50 - 1 and r0 < end and r1 - r0 >= int(min_dropout_s * FS)]
        if not tail:
            continue
        r0, r1 = max(tail, key=lambda r: r[1] - r[0])
        f.update(_crc_unverified=True, _dropout_ms=1000.0 * (r1 - r0) / FS, _support_phases=support, _phase=ph,
                 _start_sample=start, _end_sample=start + n * spb)
        best = f
        break
    return best


def recover_info1a_concealed(x, spans, phases: int = 8, min_phases: int = 6) -> dict | None:
    """Opt-in recovery of an INFO1a whose tail was replaced by the receiver's loss concealment.

    The DCE conceals a short RTP gap by repeating the previous frame (``modem_dce._gap_fill``). On a V.34 INFO frame
    that filler looks like signal (the envelope is intact) but its bits are wrong, so the CRC fails and a silence
    detector (``recover_info1a``) sees nothing. The RTP layer knows exactly which samples are concealment: ``spans``
    is a list of ``(start_sample, length)`` relative to ``x``. The frame is accepted, marked ``_crc_unverified``,
    only when:

    * the sync pattern is found at ``min_phases`` or more symbol-timing phases with the same payload bits, and the
      payload fields are plausible;
    * a concealed span starts at or after the payload end (bit 49, with one bit of timing tolerance) and inside the
      frame, i.e. it overlaps the CRC/fill bits;
    * no concealed span touches the payload bits (12-49).
    A frame with its payload inside the concealment, or with no concealment over its tail, is rejected.
    """
    x = np.asarray(x, dtype=float)
    spb = FS / BAUD
    n = FRAME_LENGTH["info1a"]
    spans = [(int(a), int(b)) for a, b in spans if b > 0]
    if not spans:
        return None
    for group in _info1a_sync_groups(x, phases):
        members = [(ph, start, bits) for _, ph, start, bits in group]
        if any(parse_info1a(b) is not None for _, _, b in members):
            continue                                          # a CRC-valid frame exists: the normal path takes it
        counts: dict[tuple, int] = {}
        for _, _, b in members:
            counts[tuple(b[12:50])] = counts.get(tuple(b[12:50]), 0) + 1
        payload, support = max(counts.items(), key=lambda kv: kv[1])
        if support < min_phases:
            continue
        ph, start, bits = next(m for m in members if tuple(m[2][12:50]) == payload)
        f = info1a_fields(bits)
        if not _plausible_info1a(f):
            continue
        payload_lo, payload_hi = start + 12 * spb, start + 49 * spb
        frame_end = start + n * spb
        over_payload = [sp for sp in spans if sp[0] < payload_hi and sp[0] + sp[1] > payload_lo]
        over_tail = [sp for sp in spans if sp[0] >= payload_hi and sp[0] < frame_end]
        if over_payload or not over_tail:
            continue
        a, length = max(over_tail, key=lambda sp: sp[1])
        f.update(_crc_unverified=True, _concealed_samples=int(length), _support_phases=support, _phase=ph,
                 _start_sample=start, _end_sample=frame_end)
        return f
    return None
