"""Digital-modem choice of the upstream constellation, moduli and data rate (clause 8.8.3).

From the codeword-domain levels available and the noise the digital modem expects at its
slicer (the TRN1u fit residual plus the precoder's own residual), pick one constellation set
with a minimum gap, then the largest even K = 2*(drn+17) in 36..72 that the moduli can carry
(clause 6.1: 24,000..48,000 bit/s in steps of 8000/6). The moduli respect the constraints of
clause 6.4.2: M_i <= N for trellis positions 0-2 and 2*M_i <= N for position 3 (N = 2*LC
points). One set is used for all intervals; the standard allows six.
"""

from __future__ import annotations

from math import ceil, log2

MIN_K, MAX_K = 36, 72
MAX_POSITIVE_POINTS = 64
DRN_OFFSET = 17


def _local_steps(levels) -> dict[float, float]:
    """For each level, the larger distance to its neighbours in the full level list (the
    mirror image bounds the lowest level, since negative levels mirror positive ones)."""
    ordered = sorted({float(v) for v in levels if v > 0})
    steps = {}
    for i, level in enumerate(ordered):
        below = level - ordered[i - 1] if i else 2.0 * level
        above = ordered[i + 1] - level if i + 1 < len(ordered) else below
        steps[level] = max(below, above)
    return steps


def select_constellation(levels, noise_rms: float, *, min_gap_sigmas: float = 9.0,
                         max_points: int = MAX_POSITIVE_POINTS) -> tuple[float, ...]:
    """Ascending positive levels. Adjacent chosen points must be at least
    ``min_gap_sigmas * noise_rms`` plus the mean of their local G.711 steps apart: the
    digital modem's A/D first snaps the received sample to the nearest level, which can
    move it by up to half a step before the constellation decision, so that step comes off
    the noise margin. The first level must clear zero by half that distance (negative
    points mirror positive ones). Truncated to ``max_points`` from the bottom."""
    if noise_rms < 0 or min_gap_sigmas <= 0:
        raise ValueError("noise and gap factor must be positive")
    steps = _local_steps(levels)
    chosen: list[float] = []
    for level in sorted(steps):
        step = steps[level]
        if chosen:
            required = min_gap_sigmas * noise_rms + (steps[chosen[-1]] + step) / 2.0
            if level - chosen[-1] < required:
                continue
        elif 2.0 * level < min_gap_sigmas * noise_rms + step:
            continue
        if len(chosen) == max_points:
            break
        chosen.append(level)
    return tuple(chosen)


def _moduli_for(points: int, bits: int):
    """Twelve moduli with product >= 2**bits within the per-position limits, or None."""
    largest_plain, largest_parity = points, points // 2
    if largest_parity < 1:
        return None
    target = 1 << bits
    for m3 in range(min(largest_parity, ceil(target ** (1 / 12))), largest_parity + 1):
        base = ceil((target / m3 ** 3) ** (1 / 9))
        for m1 in range(base, base + 2):
            if m1 > largest_plain:
                break
            if m1 ** 9 * m3 ** 3 >= target:
                return (m1, m1, m1, m3) * 3
    return None


def design_with_power(levels, measurement_noise: float, isi_ratio: float, max_rms_u: float,
                      **kw) -> dict | None:
    """:func:`design_with_isi`, with the constellation's rms held at or below ``max_rms_u``
    (the transmit-power budget divided by the u -> v gain of the precoder and prefilter).

    Levels above a cap are dropped and the cap lowered until the chosen constellation fits;
    fewer, smaller points mean a lower rate, or None when even 24,000 bit/s does not fit."""
    if max_rms_u <= 0:
        return None
    pool = sorted(float(v) for v in levels if v > 0)
    while pool:
        plan = design_with_isi(pool, measurement_noise, isi_ratio, **kw)
        if plan is None:
            return None
        points = plan["constellation"]
        rms = (sum(v * v for v in points) / len(points)) ** 0.5
        if rms <= max_rms_u:
            plan["rms_u"] = rms
            return plan
        pool = [v for v in pool if v < points[-1]]
    return None


def design_with_isi(levels, measurement_noise: float, isi_ratio: float, **kw) -> dict | None:
    """:func:`design` when the precoder leaves residual ISI proportional to the signal.

    The slicer noise is ``sqrt(measurement_noise**2 + (isi_ratio * rms(u))**2)`` where
    ``rms(u)`` is the rms of the chosen constellation, so the choice is iterated to a fixed
    point (larger constellations carry more ISI, which forces smaller ones)."""
    noise = measurement_noise
    plan = None
    for _ in range(8):
        plan = design(levels, noise, **kw)
        if plan is None:
            return None
        rms = (sum(v * v for v in plan["constellation"]) / len(plan["constellation"])) ** 0.5
        updated = (measurement_noise ** 2 + (isi_ratio * rms) ** 2) ** 0.5
        if abs(updated - noise) <= 1e-3 * max(noise, 1.0):
            break
        noise = max(noise, updated) if updated > noise else updated
    plan["noise_rms"] = noise
    return plan


def design(levels, noise_rms: float, *, max_drn: int = 19, trellis: int = 0,
           min_gap_sigmas: float = 9.0) -> dict | None:
    """Return ``{"constellation", "drn", "bits", "rate", "moduli", "trellis"}`` for the
    highest rate the levels and noise support, or None when even 24,000 bit/s is out of
    reach (the caller then falls back to V.34 upstream)."""
    constellation = select_constellation(levels, noise_rms, min_gap_sigmas=min_gap_sigmas)
    if not constellation:
        return None
    points = 2 * len(constellation)
    for drn in range(min(max_drn, MAX_K // 2 - DRN_OFFSET), MIN_K // 2 - DRN_OFFSET - 1, -1):
        bits = 2 * (drn + DRN_OFFSET)
        moduli = _moduli_for(points, bits)
        if moduli is not None:
            return {"constellation": constellation, "drn": drn, "bits": bits,
                    "rate": bits * 8000 / 12, "moduli": moduli, "trellis": trellis}
    return None


def _peer_moduli_for(lc: int, bits: int):
    """Twelve moduli with M_3 = LC / 2 and the smallest M_0..2 <= LC giving a product >= 2**bits, or None."""
    m3 = lc // 2
    if lc % 2 or m3 < 1:
        return None
    for m1 in range(2, lc + 1):
        if m1 ** 9 * m3 ** 3 >= 1 << bits:
            return (m1, m1, m1, m3) * 3
    return None


def design_for_peer(levels, noise_rms: float, *, max_rms: float, min_gap_sigmas: float = 4.0,
                    fallback_gap_sigmas=(3.0, 2.5, 2.0), max_drn: int = 19, trellis: int = 1) -> dict | None:
    """:func:`design` for analogue modems that index the candidate table only up to LC (SmartLink): moduli are
    bounded by the positive point count LC rather than N = 2 * LC (M_i <= LC on trellis positions 0-2), and the
    trellis position-3 modulus must be exactly LC / 2, so LC is even. Measured on the local SmartLink rig: with
    2 * M_3 < LC the firmware sends a zero for one position-3 index (LC 15 / M_3 7 lost eta +1, LC 11 / M_3 5 lost
    eta -2) and B1u fails; LC 10 / M_3 5 and plain-position M < LC (14 of 15) were clean.

    The constellation's rms stays at or below ``max_rms`` (levels above a cap are dropped until it fits). When no
    24,000 bit/s plan clears ``min_gap_sigmas``, the gap factors in ``fallback_gap_sigmas`` are tried in order;
    the result then carries ``"fallback_gap_sigmas"``. None when not even the last one gives a plan."""
    if max_rms <= 0 or noise_rms < 0:
        raise ValueError("max_rms must be positive and noise_rms not negative")
    for gap in (min_gap_sigmas, *fallback_gap_sigmas):
        pool = sorted(float(v) for v in levels if v > 0)
        constellation = ()
        while pool:
            constellation = select_constellation(pool, noise_rms, min_gap_sigmas=gap)
            if not constellation:
                break
            rms = (sum(v * v for v in constellation) / len(constellation)) ** 0.5
            if rms <= max_rms:
                break
            pool = [v for v in pool if v < constellation[-1]]
        if len(constellation) % 2:
            constellation = constellation[:-1]           # LC must be even; dropping the top level also lowers power
        if len(constellation) < 2:
            continue
        lc = len(constellation)
        for drn in range(min(max_drn, MAX_K // 2 - DRN_OFFSET), MIN_K // 2 - DRN_OFFSET - 1, -1):
            bits = 2 * (drn + DRN_OFFSET)
            moduli = _peer_moduli_for(lc, bits)
            if moduli is not None:
                plan = {"constellation": constellation, "drn": drn, "bits": bits, "rate": bits * 8000 / 12,
                        "moduli": moduli, "trellis": trellis, "gap_sigmas": gap, "noise_rms": noise_rms,
                        "rms": (sum(v * v for v in constellation) / lc) ** 0.5}
                if gap != min_gap_sigmas:
                    plan["fallback_gap_sigmas"] = gap
                return plan
    return None
