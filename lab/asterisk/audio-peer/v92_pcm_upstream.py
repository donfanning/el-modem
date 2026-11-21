"""V.92 PCM-upstream mapper (ITU-T V.92 clause 6.4): modulus encoder, precoder and
prefilter, inverse map and convolutional encoder, plus the matching digital-modem
demapper.

Offline and deterministic: it turns K scrambled bits per 12-interval data frame into
the 12 linear sample values the analogue modem sends, and back. It performs no
scrambling, framing, timing or RTP; callers supply scrambled bits and G.711-referenced
constellation levels.

Structure (Figure 1/2): data frame i=0..11; constellation frame j=i mod 6 selects one
of up to six constellation sets; trellis frame k=i mod 4. Per trellis frame the two
pairs (y0,y1),(y2,y3) of constellation-point indices go through the V.34 symbol-to-bit
map (odd-integer coordinates 2*y+1) and the V.34 convolutional encoder, whose 2T
delays become one step per trellis frame here; its output Y0 fixes the parity of the
fourth point's index.

The equivalence-class and precoder equations are transcribed from clause 6.4.2. The
V.34 trellis tables are shared with ``v34_data``.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod

import numpy as np

from v34_data import TABLE13, convolutional_encoder, subset_label

INTERVALS = 12
TRELLIS_FRAME = 4


def cpd_receive_scale(gain: float) -> float:
    """Convert equalized line samples LE=G*U back to negotiated CPd point U."""
    value = float(gain)
    if not np.isfinite(value) or value <= 0:
        raise ValueError("CPd gain must be finite and positive")
    return 1.0 / value


@dataclass(frozen=True)
class PcmUpstreamProfile:
    """Modulation parameters that CPd (Table 30) carries to the analogue modem.

    ``constellations`` holds up to six ascending tuples of positive linear levels (the
    zero point is excluded); a set of L positive levels has N = 2L points a(eta),
    -L <= eta < L, with negative eta mirroring the positive levels. ``interval_set``
    gives the set used in each of the 12 data frame intervals.
    """

    constellations: tuple[tuple[float, ...], ...]
    interval_set: tuple[int, ...]
    moduli: tuple[int, ...]
    trellis: int = 0                       # 0 = 16 state, 1 = 32, 2 = 64
    bits: int = 0                          # K, bits per data frame; 0 = the largest with 2**K <= M
    gain: float = 1.0                      # G at the prefilter output
    precoder_z1: tuple[float, ...] = ()    # z1(1..LZ1)
    precoder_p1: tuple[float, ...] = ()    # p1(1..LP1)
    prefilter_z2: tuple[float, ...] = (1.0,)   # z2(0..LZ2-1)
    prefilter_p2: tuple[float, ...] = ()   # p2(1..LP2)

    def __post_init__(self):
        if len(self.interval_set) != INTERVALS or len(self.moduli) != INTERVALS:
            raise ValueError("a data frame has 12 intervals")
        if not 1 <= len(self.constellations) <= 6:
            raise ValueError("one to six constellation sets")
        for levels in self.constellations:
            if not levels or any(b <= a for a, b in zip(levels, levels[1:])) or levels[0] <= 0:
                raise ValueError("constellation levels must be positive and ascending")
        if any(not 0 <= s < len(self.constellations) for s in self.interval_set):
            raise ValueError("interval refers to a missing constellation set")
        if self.trellis not in (0, 1, 2):
            raise ValueError("trellis selection must be 0, 1 or 2")
        for i, (m, s) in enumerate(zip(self.moduli, self.interval_set)):
            points = 2 * len(self.constellations[s])
            if m < 1 or (m > points if i % TRELLIS_FRAME < 3 else 2 * m > points):
                raise ValueError(f"modulus M{i}={m} does not fit its constellation")
        if prod(self.moduli) < 2:
            raise ValueError("M must be at least 2")
        if self.bits < 0 or (self.bits and 1 << self.bits > prod(self.moduli)):
            raise ValueError("K must satisfy 2**K <= M")
        if self.gain <= 0:
            raise ValueError("gain must be positive")

    @property
    def total_modulus(self) -> int:
        return prod(self.moduli)

    @property
    def bits_per_frame(self) -> int:
        """K. Explicit when the data rate fixes it (clause 6.1: K = 2*(drn+17) in CPd),
        otherwise the largest value with 2**K <= M."""
        return self.bits or self.total_modulus.bit_length() - 1

    def level(self, interval: int, eta: int) -> float:
        levels = self.constellations[self.interval_set[interval]]
        n = len(levels)
        if not -n <= eta < n:
            raise ValueError("constellation index out of range")
        return levels[eta] if eta >= 0 else -levels[-eta - 1]


def modulus_encode(bits, moduli, total, sign_state: int):
    """Clause 6.4.1: K bits -> (K0..K11, new differential sign state)."""
    r = sum(int(b) << i for i, b in enumerate(bits))
    if r >= total:
        raise ValueError("bit pattern exceeds the modulus")
    sign = 1 if 2 * r > total - 1 else 0
    r0 = r if sign_state == 0 else total - 1 - r
    out = []
    for m in moduli:
        out.append(r0 % m)
        r0 //= m
    return out, sign ^ sign_state


def modulus_decode(ks, moduli, total, sign_state: int, count: int):
    """Inverse of :func:`modulus_encode`: (K0..K11) -> (bits, new sign state)."""
    r0, scale = 0, 1
    for k, m in zip(ks, moduli):
        r0 += k * scale
        scale *= m
    if r0 >= total:
        raise ValueError("received label exceeds the modulus")
    r = r0 if sign_state == 0 else total - 1 - r0
    sign = 1 if 2 * r > total - 1 else 0
    if r >> count:
        raise ValueError("received value exceeds the negotiated rate")
    return [(r >> i) & 1 for i in range(count)], sign ^ sign_state


def _trellis_step(conv, etas) -> None:
    s0 = subset_label(complex(2 * etas[0] + 1, 2 * etas[1] + 1))
    s1 = subset_label(complex(2 * etas[2] + 1, 2 * etas[3] + 1))
    conv.step(TABLE13[s0][s1])


class PcmUpstreamTransmitter:
    """Analogue modem side. One call to :meth:`encode_frame` per data frame."""

    def __init__(self, profile: PcmUpstreamProfile, *, record: bool = False):
        self.profile = profile
        self.conv = convolutional_encoder(profile.trellis)
        self._sign = 0
        self.levels: list[float] | None = [] if record else None   # u(n), for diagnostics
        self._u: list[float] = []
        self._x: list[float] = []
        self._v: list[float] = []

    def _candidates(self, interval: int, k_value: int, parity_source: list[int]):
        profile = self.profile
        n = len(profile.constellations[profile.interval_set[interval]])
        m = profile.moduli[interval]
        if interval % TRELLIS_FRAME < 3:
            base, stride = k_value, m
        else:
            parity = (sum(parity_source) + self.conv.y0) & 1
            base, stride = 2 * k_value + parity, 2 * m
        first = base - ((base + n) // stride) * stride        # smallest eta >= -n in the class
        etas = []
        eta = first
        while eta < n:
            if eta >= -n:
                etas.append(eta)
            eta += stride
        return etas

    def _precode(self, u: float) -> float:
        p = self.profile
        # zip stops at the shorter of coefficients and history, which is the "only once k samples exist" rule.
        x = u + sum(h * c for h, c in zip(reversed(self._u), p.precoder_z1))
        return x + sum(h * c for h, c in zip(reversed(self._x), p.precoder_p1))

    def _prefilter(self, x: float) -> float:
        p = self.profile
        v = x * p.prefilter_z2[0] if p.prefilter_z2 else 0.0
        v += sum(h * c for h, c in zip(reversed(self._x), p.prefilter_z2[1:]))
        return v + sum(h * c for h, c in zip(reversed(self._v), p.prefilter_p2))

    def encode_frame(self, bits) -> list[float]:
        profile = self.profile
        if len(bits) != profile.bits_per_frame:
            raise ValueError(f"a data frame carries {profile.bits_per_frame} bits")
        ks, sign = modulus_encode(bits, profile.moduli, profile.total_modulus, self._sign)
        out = self.encode_intervals(ks)
        self._sign = sign
        return out

    def encode_intervals(self, ks) -> list[float]:
        """Map twelve modulus-encoder outputs K0..K11 to samples, skipping the modulus encoder.

        RM and RM' (Tables 25 and 26) are defined directly by their K values; they are trellis
        encoded and use the data-mode precoder and prefilter, so they go through the same
        selection, precoder, prefilter and convolutional encoder as data. The differential
        sign state of the modulus encoder is left alone."""
        profile = self.profile
        if len(ks) != INTERVALS or any(not 0 <= k < m for k, m in zip(ks, profile.moduli)):
            raise ValueError("twelve K values, each below its modulus")
        out: list[float] = []
        etas: list[int] = []
        for i in range(INTERVALS):
            options = self._candidates(i, ks[i], etas[:3] if i % TRELLIS_FRAME == 3 else [])
            if not options:
                raise ValueError(f"interval {i}: no constellation point in the class")
            scored = []
            for eta in options:
                u = profile.level(i, eta)
                scored.append((abs(self._precode(u)), abs(u), eta, u))
            _, _, eta, u = min(scored)
            x = self._precode(u)
            v = self._prefilter(x)
            if self.levels is not None:
                self.levels.append(u)
            self._u.append(u)
            self._x.append(x)
            self._v.append(v)
            out.append(profile.gain * v)
            etas.append(eta)
            if i % TRELLIS_FRAME == 3:
                _trellis_step(self.conv, etas)
                etas = []
        keep = max(len(profile.precoder_z1), len(profile.precoder_p1),
                   len(profile.prefilter_z2), len(profile.prefilter_p2), 1)
        self._u, self._x, self._v = self._u[-keep:], self._x[-keep:], self._v[-keep:]
        return out


class PcmUpstreamReceiver:
    """Digital modem side. Input is the 12 received levels of one data frame in the
    constellation domain: the precoder is designed so the channel output equals u(n), and
    the channel gain that G compensates is the caller's to remove (the transmitter's
    output is ``gain * v``)."""

    def __init__(self, profile: PcmUpstreamProfile):
        self.profile = profile
        self.conv = convolutional_encoder(profile.trellis)
        self._sign = 0
        self.trellis_violations = 0
        # Signed levels of each set in ascending order of eta = -L .. L-1, for a binary search.
        self._signed = []
        for levels in profile.constellations:
            n = len(levels)
            self._signed.append(np.array([-levels[-eta - 1] if eta < 0 else levels[eta]
                                          for eta in range(-n, n)], dtype=float))

    def _nearest(self, interval: int, value: float) -> int:
        signed = self._signed[self.profile.interval_set[interval]]
        n = len(signed) // 2
        i = int(np.searchsorted(signed, value))
        if i <= 0:
            return -n
        if i >= len(signed):
            return n - 1
        # the nearer of the two neighbours; on a tie the lower eta (as min() over ascending eta did)
        return (i - 1 if value - signed[i - 1] <= signed[i] - value else i) - n

    def decode_frame(self, samples) -> list[int]:
        ks = self.decode_intervals(samples)
        profile = self.profile
        bits, self._sign = modulus_decode(ks, profile.moduli, profile.total_modulus,
                                          self._sign, profile.bits_per_frame)
        return bits

    def decode_intervals(self, samples) -> list[int]:
        """Recover K0..K11 from twelve samples without the modulus decoder (for RM / RM')."""
        profile = self.profile
        if len(samples) != INTERVALS:
            raise ValueError("a data frame has 12 intervals")
        ks: list[int] = []
        etas: list[int] = []
        for i, value in enumerate(samples):
            eta = self._nearest(i, value)
            m = profile.moduli[i]
            if i % TRELLIS_FRAME < 3:
                ks.append(eta % m)
            else:
                parity = eta & 1
                if parity != (sum(etas) + self.conv.y0) & 1:
                    self.trellis_violations += 1
                ks.append(((eta - parity) // 2) % m)
            etas.append(eta)
            if i % TRELLIS_FRAME == 3:
                _trellis_step(self.conv, etas)
                etas = []
        return ks


def rm_intervals(moduli, *, prime: bool = False) -> list[int]:
    """K0..K11 of RM (Table 25) or, with ``prime``, RM' (Table 26)."""
    top = [m - 1 for m in moduli]
    if not prime:
        pattern = [top[0], top[1], 0, 0, top[4], top[5], 0, 0, top[8], top[9], 0, 0]
    else:
        pattern = [0, 0, top[2], top[3], 0, 0, top[6], top[7], 0, 0, top[10], top[11]]
    return pattern


def profile_from_cpd(parsed: dict) -> PcmUpstreamProfile:
    """Turn a parsed CPd (``v92_cp.parse_cpd``) that carries all three parts into the
    mapper profile: the six selectors apply to intervals i and i+6."""
    for part in ("moduli", "constellations", "interval_sets"):
        if part not in parsed:
            raise ValueError(f"CPd has no {part}")
    precoder = parsed.get("precoder") or {}
    sets = tuple(tuple(float(v) for v in s) for s in parsed["constellations"] if s)
    return PcmUpstreamProfile(
        constellations=sets,
        interval_set=tuple(parsed["interval_sets"]) * 2,
        moduli=tuple(parsed["moduli"]),
        trellis=parsed["trellis"],
        bits=2 * (parsed["drn"] + 17) if parsed["drn"] else 0,
        gain=parsed["gain"],
        precoder_z1=tuple(precoder.get("z1", ())),
        precoder_p1=tuple(precoder.get("p1", ())),
        prefilter_z2=tuple(precoder.get("z2", (1.0,))) or (1.0,),
        prefilter_p2=tuple(precoder.get("p2", ())),
    )
