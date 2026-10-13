"""V.34 data mode transmitter (ITU-T V.34 clauses 7-9), written from the text.

Pipeline per mapping frame (8 two-dimensional symbols): scrambled bits -> parser (b or b-1 bits, K of them to the shell
mapper, then I1/I2/I3 and Q bits for each 4D symbol) -> shell mapper -> differential encoder (I2, I3) -> mapper (ring
index + Q bits label a point of the quarter superconstellation, rotated by Z / Z + 2*I1 + U0) -> trellis encoder
(16-state convolutional encoder over subset labels, modulo encoder, superframe bit inversions).

Transmit precoding supports the three complex coefficients requested in MP Type 1. The nonlinear encoder supports
Theta = 0 and 0.3125 (9.7), with average transmit power normalized after the nonlinear projection.
"""

from __future__ import annotations

import math
from bisect import bisect_right
from functools import lru_cache

import numpy as np

# Table 7 / Table 9 style parameters: symbol rate -> (J, P)
FRAMING = {2400: (7, 12), 2743: (8, 12), 2800: (7, 14), 3000: (7, 15), 3200: (7, 16), 3429: (8, 15)}
INVERSION = {7: [0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0], 8: [0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0]}
INVERSION[8] = [int(c) for c in "0111011111111010"]      # Table 12 (J = 8): 01 11 01 11 11 11 10 10
INVERSION[7] = [int(c) for c in "01110111111110"]         # Table 12 (J = 7): 01 11 01 11 11 11 10


def quarter_points(n: int = 416) -> list[complex]:
    """Quarter superconstellation of Figure 5, labelled 0..n-1: points (4i+1, 4j+1) ordered by magnitude, ties by
    the greatest imaginary component first."""
    pts = [(x, y) for x in range(-47, 48) for y in range(-47, 48) if x % 4 == 1 and y % 4 == 1]
    pts.sort(key=lambda p: (p[0] ** 2 + p[1] ** 2, -p[1]))
    return [complex(x, y) for x, y in pts[:n]]


QUARTER = quarter_points()


def rotate_cw(p: complex, quarter_turns: int) -> complex:
    return p * (-1j) ** (quarter_turns % 4)


# -- framing ----------------------------------------------------------------------------------------------------
def frame_params(rate: int, baud: int) -> dict:
    """b (bits in a high mapping frame), r (number of high frames), J, P for a primary data rate and symbol rate."""
    J, P = FRAMING[round(baud)]
    N = round(rate * 0.28 / J)                              # bits per data frame (8-1)
    b = -(-N // P)                                          # smallest integer not less than N / P
    r = N - (b - 1) * P                                     # high frames per data frame (8-2)
    return {"J": J, "P": P, "N": N, "b": b, "r": r}


def swp_pattern(P: int, r: int) -> list[int]:
    """1 = high frame (b bits), 0 = low frame; counter algorithm of 8.2."""
    out, counter = [], 0
    for _ in range(P):
        counter += r
        if counter < P:
            out.append(0)
        else:
            out.append(1)
            counter -= P
    return out


def mapping_params(b: int, expanded: bool) -> dict:
    """K, M, q, L (9.2)."""
    q = 0
    while True:
        K = 0 if b <= 12 else b - 12 - 8 * q
        if K < 32:
            break
        q += 1
    k_eff = K
    m_min = math.ceil(2 ** (k_eff / 8) - 1e-12)
    m_exp = max(m_min, int(math.floor(1.25 * 2 ** (k_eff / 8) + 0.5)))
    M = m_exp if expanded else m_min
    return {"K": K, "M": M, "q": q, "L": 4 * M * 2 ** q}


# -- shell mapper (9.4) -----------------------------------------------------------------------------------------
class ShellMapper:
    def __init__(self, M: int):
        self.M = M
        self.g2 = [M - abs(p - M + 1) if 0 <= p <= 2 * (M - 1) else 0 for p in range(2 * M)]
        n4 = 4 * (M - 1) + 1
        self.g4 = [sum(self._g2(i) * self._g2(p - i) for i in range(p + 1)) for p in range(n4)]
        n8 = 8 * (M - 1) + 1
        self.g8 = [sum(self._g4(i) * self._g4(p - i) for i in range(p + 1)) for p in range(n8)]
        self.z8 = [0]
        for p in range(n8):
            self.z8.append(self.z8[-1] + self.g8[p])        # z8[p] = g8(0) + ... + g8(p-1)
        # Counts are exact Python integers. Each row includes a final sentinel
        # so a rightmost search preserves the unranking loops' zero-count ties.
        # These bounded, immutable tables also serve the inverse shell mapper.
        self.g2_prefix = self._prefix_products(self._g2, n4)
        self.g4_prefix = self._prefix_products(self._g4, n8)

    @staticmethod
    def _prefix_products(count, totals):
        rows = []
        for total in range(totals):
            row = [0]
            for left in range(total + 1):
                row.append(row[-1] + count(left) * count(total - left))
            rows.append(tuple(row))
        return tuple(rows)

    def _g2(self, p: int) -> int:
        return self.g2[p] if 0 <= p < len(self.g2) else 0

    def _g4(self, p: int) -> int:
        return self.g4[p] if 0 <= p < len(self.g4) else 0

    def indices(self, R0: int) -> list[int]:
        """Eight ring indices m[0,0], m[0,1], m[1,0], ... for the integer R0 built from the K shell bits."""
        M = self.M
        A = bisect_right(self.z8, R0, hi=len(self.z8) - 1) - 1
        if A < 0:
            raise ValueError('shell rank must be nonnegative')
        R1 = R0 - self.z8[A]
        row = self.g4_prefix[A]
        B = bisect_right(row, R1) - 1
        R1 -= row[B]
        g4B = self._g4(B)
        R2 = R1 % g4B
        R3 = (R1 - R2) // g4B
        row = self.g2_prefix[B]
        C = bisect_right(row, R2) - 1
        R4 = R2 - row[C]
        row = self.g2_prefix[A - B]
        D = bisect_right(row, R3) - 1
        R5 = R3 - row[D]
        E = R4 % self._g2(C)
        F = (R4 - E) // self._g2(C)
        G = R5 % self._g2(D)
        H = (R5 - G) // self._g2(D)

        def pair(total, low):
            if total < M:
                return low, total - low
            return total - (M - 1 - low), M - 1 - low

        m00, m01 = pair(C, E)
        m10, m11 = pair(B - C, F)
        m20, m21 = pair(D, G)
        m30, m31 = pair(A - B - D, H)
        return [m00, m01, m10, m11, m20, m21, m30, m31]


# -- trellis encoder (9.6.3) ------------------------------------------------------------------------------------
TABLE13 = {  # [s(2m)][s(2m+1)] -> (Y4, Y3, Y2, Y1) as a 4-bit integer
    0: [0b0000, 0b0000, 0b0001, 0b0001, 0b1000, 0b1000, 0b1001, 0b1001],
    1: [0b0011, 0b0010, 0b0010, 0b0011, 0b1011, 0b1010, 0b1010, 0b1011],
    2: [0b0101, 0b0101, 0b0100, 0b0100, 0b1101, 0b1101, 0b1100, 0b1100],
    3: [0b0110, 0b0111, 0b0111, 0b0110, 0b1110, 0b1111, 0b1111, 0b1110],
    4: [0b1000, 0b1000, 0b1001, 0b1001, 0b0000, 0b0000, 0b0001, 0b0001],
    5: [0b1011, 0b1010, 0b1010, 0b1011, 0b0011, 0b0010, 0b0010, 0b0011],
    6: [0b1101, 0b1101, 0b1100, 0b1100, 0b0101, 0b0101, 0b0100, 0b0100],
    7: [0b1110, 0b1111, 0b1111, 0b1110, 0b0110, 0b0111, 0b0111, 0b0110],
}


def _f(t: int) -> int:
    return ((t + 1) >> 1) & 1


def subset_label(y: complex) -> int:
    """3-bit subset label of Figure 9 for a point on the odd-integer grid."""
    a, b = (int(round(y.real)) - 1) // 2, (int(round(y.imag)) - 1) // 2
    return ((_f(a) ^ _f(b)) << 2) | ((a & 1) << 1) | ((a + b) & 1)


class ConvEncoder16:
    """16-state systematic feedback encoder of Figure 10 (inputs Y2 and Y1; Y3, Y4 unused)."""

    def __init__(self):
        self.d = [0, 0, 0, 0]                              # outputs of the four 2T delays, d[3] = Y0

    @property
    def y0(self) -> int:
        return self.d[3]

    def step(self, y4y3y2y1: int) -> None:
        y2, y1 = (y4y3y2y1 >> 1) & 1, y4y3y2y1 & 1
        d1, d2, d3, d4 = self.d
        self.d = [d4, d1 ^ y2 ^ d4, d2 ^ y2, d3 ^ y1]


class ConvEncoder32:
    """Figure 11: five 2T delays, feedback from the last delay."""
    def __init__(self):
        self.d = [0] * 5

    @property
    def y0(self):
        return self.d[-1]

    def step(self, y):
        a, b, c, d, e = self.d
        y1, y2, y4 = y & 1, (y >> 1) & 1, (y >> 3) & 1
        self.d = [e, a ^ y2, b ^ y1, c ^ y4, d ^ y2]


class ConvEncoder64:
    """Figure 12: six 2T delays and the two AND gates (02/98 p.25)."""
    def __init__(self):
        self.d = [0] * 6

    @property
    def y0(self):
        return self.d[-1]

    def step(self, y):
        a, b, c, d, e, f = self.d
        y1, y2, y3, y4 = [(y >> k) & 1 for k in range(4)]
        ab, by1 = a ^ b, b ^ y1
        self.d = [y4 ^ ab ^ (by1 & c),
                  ab ^ y3 ^ (y2 & c) ^ d,
                  by1 ^ c, c, f, e ^ y2 ^ c]


def convolutional_encoder(selection):
    if selection not in (0, 1, 2):
        raise ValueError("unsupported V.34 trellis selection")
    try:
        return (ConvEncoder16, ConvEncoder32, ConvEncoder64)[selection]()
    except (IndexError, TypeError):
        raise ValueError('unsupported V.34 trellis selection') from None


def modulo_bit(c1: complex, c2: complex) -> int:
    """C0(m): parity test on c(2m)/2 and c(2m+1)/2 (9.6.3.3)."""
    s1 = int(round(c1.real / 2)) + int(round(c1.imag / 2))
    s2 = int(round(c2.real / 2)) + int(round(c2.imag / 2))
    return 0 if (s1 % 2) == (s2 % 2) else 1


def precoder_from_mp(values) -> list[complex]:
    """MP Type 1 precoding coefficients (six signed 16-bit integers, 14 fractional bits) -> h(1..3)."""
    v = list(values)
    return [complex(v[2 * k], v[2 * k + 1]) / 16384.0 for k in range(3)]


def _round_half_to_smaller(v: float, step: float) -> float:
    """Nearest integer multiple of ``step``; exact ties go to the smaller magnitude (9.6.2 items 2 and 3)."""
    q = v / step
    n = math.ceil(abs(q) - 0.5)
    return math.copysign(n, q) * step if n else 0.0


def _round_complex(z: complex, step: float) -> complex:
    return complex(_round_half_to_smaller(z.real, step), _round_half_to_smaller(z.imag, step))


class DataTx:
    """Turn a stream of already scrambled primary-channel bits into 2D data-mode symbols."""

    def __init__(self, rate: int, baud: int, expanded: bool = False, precoder=None, nonlinear: bool = False, trellis: int = 0):
        """``precoder``: the three complex coefficients h(1..3) the receiver asked for (None = no precoding)."""
        self.baud = baud
        self.h = list(precoder) if precoder is not None else [0j, 0j, 0j]
        self.x_hist = [0j, 0j, 0j]                         # x(n-1), x(n-2), x(n-3)
        fp = frame_params(rate, baud)
        self.J, self.P, self.b = fp["J"], fp["P"], fp["b"]
        self.swp = swp_pattern(self.P, fp["r"])
        mp = mapping_params(self.b, expanded)
        self.K, self.M, self.q = mp["K"], mp["M"], mp["q"]
        self.shell = ShellMapper(self.M) if self.K else None
        self.bits_per_data_frame = fp["N"]
        self.w = 1 if self.b < 56 else 2                   # precoder quantization scale 2^w (9-29)
        self.z = 0                                         # differential encoder Z(m-1)
        self.conv = convolutional_encoder(trellis)
        self.m4 = 0                                        # 4D symbol counter since B1
        self._inv = INVERSION[self.J]
        self.inv_offset = 2 * self.J - 2                   # B1 is treated as the last data frame of a superframe
        self.nonlinear = nonlinear
        self._linear_power = self._estimate_power(rate, baud, expanded, self.h, trellis=trellis)
        self.avg_power = (self._estimate_power(rate, baud, expanded, self.h, nonlinear=True,
                                             linear_power=self._linear_power, trellis=trellis)
                          if nonlinear else self._linear_power)

    @staticmethod
    def _estimate_power(rate: int, baud: int, expanded: bool, h=None, nonlinear=False, linear_power=1.0, trellis=0) -> float:
        # Fixed-seed simulation uses a private mapper and depends only on these
        # immutable parameters. Cache the scalar, never live encoder history.
        coefficients=tuple(h) if h is not None else (0j, 0j, 0j)
        return DataTx._estimate_power_cached(rate, baud, expanded,
            coefficients, nonlinear, linear_power, trellis, tuple(type(v) for v in coefficients))

    @staticmethod
    @lru_cache(maxsize=128, typed=True)
    def _estimate_power_cached(rate, baud, expanded, h, nonlinear, linear_power, trellis, coefficient_types=None):
        """Average symbol energy for random data (used to keep the transmit power equal to the training signals')."""
        import random
        t = DataTx.__new__(DataTx)
        t.baud = baud
        fp = frame_params(rate, baud)
        t.J, t.P, t.b = fp["J"], fp["P"], fp["b"]
        t.swp = swp_pattern(t.P, fp["r"])
        mp = mapping_params(t.b, expanded)
        t.K, t.M, t.q = mp["K"], mp["M"], mp["q"]
        t.shell = ShellMapper(t.M) if t.K else None
        t.bits_per_data_frame = fp["N"]
        t.w = 1 if t.b < 56 else 2
        t.z, t.conv, t.m4, t._inv, t.inv_offset = 0, convolutional_encoder(trellis), 0, INVERSION[t.J], 0
        t.h, t.x_hist = list(h) if h is not None else [0j, 0j, 0j], [0j, 0j, 0j]
        t.nonlinear, t._linear_power = nonlinear, linear_power
        rnd = random.Random(1)
        acc, n = 0.0, 0
        for _ in range(40):
            sy = t.data_frame([rnd.randrange(2) for _ in range(fp["N"])])
            acc += sum(abs(v) ** 2 for v in sy)
            n += len(sy)
        return acc / n

    def _pc(self) -> tuple[complex, complex]:
        """p(n) and c(n) for the next symbol from the three most recent precoded symbols (9.6.2 items 1-3)."""
        if not any(self.h):
            return 0j, 0j
        q = sum(self.x_hist[k] * self.h[k] for k in range(3))
        p = _round_complex(q, 2.0 ** -7)
        c = _round_complex(p, 2.0 ** self.w)
        return p, c

    def _v0(self, m4: int) -> int:
        """Superframe-sync bit inversion V0(m): the pattern bit at the start of each half data frame (m = k * 2P)."""
        P2 = 2 * self.P
        if m4 % P2:
            return 0
        return self._inv[(m4 // P2 + self.inv_offset) % len(self._inv)]

    def mapping_frame(self, bits: list[int], high: bool) -> list[complex]:
        """8 symbols from b (high) or b-1 (low) bits."""
        K, q = self.K, self.q
        it = iter(bits)

        def take(n):
            return [next(it) for _ in range(n)]

        if K:
            shell_bits = take(K) if high else take(K - 1) + [0]
            R0 = sum(bit << i for i, bit in enumerate(shell_bits))
            rings = self.shell.indices(R0)
        else:
            rings = [0] * 8
        out: list[complex] = []
        if K:
            groups = []
            for j in range(4):
                i1, i2, i3 = take(3)
                qa = take(q)
                qb = take(q)
                groups.append((i1, i2, i3, qa, qb))
        else:
            n = self.b if high else self.b - 1
            groups = []
            for j in range(4):
                vals = take(3 if j < n - 8 else 2)
                vals = vals + [0] * (3 - len(vals))
                groups.append((vals[0], vals[1], vals[2], [], []))
        for j, (i1, i2, i3, qa, qb) in enumerate(groups):
            m4 = self.m4
            self.m4 += 1
            self.z = (self.z + (i2 + 2 * i3)) % 4
            ring0, ring1 = rings[2 * j], rings[2 * j + 1]
            qi0 = sum(bit << t for t, bit in enumerate(qa)) + (ring0 << q)
            qi1 = sum(bit << t for t, bit in enumerate(qb)) + (ring1 << q)
            v0, v1 = QUARTER[qi0], QUARTER[qi1]
            v_inv = self._v0(m4)
            u0 = rotate_cw(v0, self.z)
            p0, c0v = self._pc()                           # p(2m), c(2m)
            y0p = u0 + c0v
            x0 = y0p - p0
            self.x_hist = [x0] + self.x_hist[:2]
            p1, c1v = self._pc()                           # p(2m+1), c(2m+1) (needs x(2m))
            u0_bit = self.conv.y0 ^ modulo_bit(c0v, c1v) ^ v_inv
            u1 = rotate_cw(v1, self.z + 2 * i1 + u0_bit)
            y1p = u1 + c1v
            x1 = y1p - p1
            self.x_hist = [x1] + self.x_hist[:2]
            out += [x0, x1]
            s0, s1 = subset_label(y0p), subset_label(y1p)
            self.conv.step(TABLE13[s0][s1])
        return out

    def data_frame(self, bits: list[int]) -> list[complex]:
        """One data frame: exactly ``bits_per_data_frame`` scrambled bits -> P mapping frames of symbols."""
        out, pos = [], 0
        for high in self.swp:
            n = self.b if high else self.b - 1
            out += self.mapping_frame(bits[pos:pos + n], bool(high))
            pos += n
        if self.nonlinear:
            # V.34 9.7, equations 9-33..35. Precoding history remains x(n);
            # only the transmitted symbols become x'(n). Power normalization
            # uses the average energy after this radial expansion.
            expanded = []
            for x in out:
                zeta = 0.3125 * abs(x) ** 2 / self._linear_power
                expanded.append(x * (1 + zeta / 6 + zeta * zeta / 120))
            return expanded
        return out
