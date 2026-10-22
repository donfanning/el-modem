"""V.34 data mode receiver pieces: inverse shell mapper, mapping-frame decoder (hard decisions with trellis-state
tracking) and an incremental symbol sampler / equalizer for the stream after B1.

The mapping decoder accepts decided mapper points. Precoder state is reconstructed with
the normative rounding rules; trellis detection is implemented in v34_viterbi.
"""

from __future__ import annotations

import math
import os
from collections import deque

import numpy as np

import v34_data as vd
import v34_rx

FS = 8000

# Bounded opt-in trace for diagnosing post-B1 transport gaps. It stores decoder
# coordinates/states and error deltas, never payload bits or audio samples.
DATA_STATE_TRACE_ENABLED = os.environ.get('V34_DATA_STATE_TRACE') == '1'


class _DataStateTrace:
    PRE_FRAMES = 32
    POST_FRAMES = 256
    MAX_GAPS = 8

    def __init__(self):
        self.recent = deque(maxlen=self.PRE_FRAMES)
        self.gaps = []
        self.active = None

    def mark_gap(self, start, count, core):
        event = dict(receive_sample=int(start), concealed_samples=int(count),
                     rx_sample_at_mark=int(getattr(core, 'rx_t', 0)),
                     pre_frames=list(self.recent), post_frames=[])
        if len(self.gaps) >= self.MAX_GAPS:
            self.gaps.pop(0)
        self.gaps.append(event)
        self.active = event

    def record(self, row):
        self.recent.append(row)
        if self.active is not None:
            self.active['post_frames'].append(row)
            if len(self.active['post_frames']) >= self.POST_FRAMES:
                self.active = None

    def report(self):
        return dict(enabled=True, coordinate='V.34 receive symbol / core PCM sample',
                    payload_or_audio_captured=False, pre_frames=self.PRE_FRAMES,
                    post_frames=self.POST_FRAMES, gaps=self.gaps)


def mark_data_state_gap(core, start, count):
    if DATA_STATE_TRACE_ENABLED:
        trace = getattr(core, '_data_state_trace', None)
        if trace is None:
            trace = core._data_state_trace = _DataStateTrace()
        trace.mark_gap(start, count, core)


def begin_data_state_frame(core, eq, drx, states, trellis_quality=None):
    if not DATA_STATE_TRACE_ENABLED:
        return None
    if getattr(core, '_data_state_trace', None) is None:
        core._data_state_trace = _DataStateTrace()
    eq_state = eq.timing_snapshot() if hasattr(eq, 'timing_snapshot') else None
    return dict(rx_sample=int(getattr(core, 'rx_t', 0)), symbol=int(eq.k),
                equalizer_state=eq_state,
                mapping_frame=int(drx.frame_pos),
                mapper_m4=int(drx.m4), mapper_rotation=int(drx.z),
                precoder_history=[[float(complex(x).real), float(complex(x).imag)] for x in drx.x_hist],
                viterbi_emission_quality=trellis_quality,
                mapper_state_before=list(drx.conv.d), mapper_errors_before=int(drx.errors),
                survivor_states=None if states is None else [int(s) for s in states])


def end_data_state_frame(core, token, drx):
    if token is None:
        return
    token.update(mapper_state_after=list(drx.conv.d), mapper_errors_after=int(drx.errors),
                 mapper_m4_after=int(drx.m4), mapper_rotation_after=int(drx.z),
                 precoder_history_after=[[float(complex(x).real), float(complex(x).imag)] for x in drx.x_hist])
    core._data_state_trace.record(token)


def inverse_shell(sm: vd.ShellMapper, rings: list[int]) -> int:
    """R0 from the eight ring indices (inverse of ShellMapper.indices)."""
    M = sm.M
    (m00, m01, m10, m11, m20, m21, m30, m31) = rings
    C = m00 + m01
    Bc = m10 + m11                                            # = B - C
    D = m20 + m21
    Ad = m30 + m31                                            # = A - B - D
    B = C + Bc
    A = B + D + Ad
    E = m00 if C < M else M - 1 - m01
    F = m10 if Bc < M else M - 1 - m11
    G = m20 if D < M else M - 1 - m21
    H = m30 if Ad < M else M - 1 - m31
    g2, g4 = sm._g2, sm._g4
    R4 = E + g2(C) * F
    R5 = G + g2(D) * H
    R2 = R4 + sm.g2_prefix[B][C]
    R3 = R5 + sm.g2_prefix[A - B][D]
    R1 = R2 + g4(B) * R3
    return sm.z8[A] + sm.g4_prefix[A][B] + R1


class DataRx:
    """Decode mapping frames of 2D symbols back to the scrambled primary bits."""

    def __init__(self, rate: int, baud: int, expanded: bool = False, precoder=None):
        tx = vd.DataTx(rate, baud, expanded)
        self.b, self.P, self.K, self.M, self.q = tx.b, tx.P, tx.K, tx.M, tx.q
        self.J = tx.J
        self.swp = tx.swp
        self.shell = tx.shell
        self.N = tx.bits_per_data_frame
        self.avg_power = tx.avg_power
        L = 4 * self.M * 2 ** self.q
        self.points = []                                      # (point, Q index, rotation)
        for qi in range(L // 4):
            for r in range(4):
                self.points.append((vd.rotate_cw(vd.QUARTER[qi], r), qi, r))
        self._pts = np.array([p[0] for p in self.points])
        self._exact_points = {complex(point): (qi, r, 0.) for point,qi,r in self.points}
        self.z = 0
        self.conv = vd.ConvEncoder16()
        self.h = list(precoder) if precoder is not None else [0j] * 3
        self.x_hist = [0j] * 3
        self.w = tx.w
        self.m4 = 0
        self.frame_pos = 0                                    # index into the SWP pattern
        self._inv = vd.INVERSION[self.J]
        self.inv_offset = 2 * self.J - 2
        self.errors = 0

    def nearest(self, y: complex) -> tuple[int, int, float]:
        exact=self._exact_points.get(y)
        if exact is not None:return exact
        d = np.abs(self._pts - y)
        i = int(np.argmin(d))
        return self.points[i][1], self.points[i][2], float(d[i])

    def nearest_point(self, y: complex) -> complex:
        """Exact nearest mapper point, with a constant-time interior-grid path.

        Every point has odd integer coordinates. If the nearest point of the
        infinite grid belongs to this finite constellation, it is also the
        nearest permitted point. Preserve the reference first-wins search on
        cell boundaries (including floating-point near ties) and outside the
        constellation, where independent coordinate rounding is insufficient.
        """
        if math.isfinite(y.real) and math.isfinite(y.imag):
            point = complex(2 * math.floor(y.real / 2) + 1,
                            2 * math.floor(y.imag / 2) + 1)
            if (abs(y.real - point.real) < 1 - 1e-10
                    and abs(y.imag - point.imag) < 1 - 1e-10
                    and point in self._exact_points):
                return point
        return complex(self._pts[int(np.argmin(np.abs(self._pts - y)))])

    def decode_frame(self, symbols, *, trellis_states=None) -> tuple[list[int], float]:
        """Eight symbols (on the odd-integer grid) -> the b or b-1 bits of one mapping frame, and the worst distance."""
        if trellis_states is not None:
            if len(trellis_states) != 4 or any(not isinstance(s, (int, np.integer)) or not 0 <= s < 16 for s in trellis_states):
                raise ValueError('four 16-state survivor states required')
        high = bool(self.swp[self.frame_pos % self.P])
        self.frame_pos += 1
        q = self.q
        groups, rings, worst = [], [], 0.0
        for j in range(4):
            if trellis_states is not None:
                # Bounded traceback decisions may switch survivor histories.
                # Interpret this interval using the state of its chosen path.
                s = int(trellis_states[j])
                self.conv.d = [(s >> shift) & 1 for shift in (3, 2, 1, 0)]
            qa, ra, da = self.nearest(symbols[2 * j])
            qb, rb, db = self.nearest(symbols[2 * j + 1])
            worst = max(worst, da, db)
            z_new = ra
            i_val = (z_new - self.z) % 4
            self.z = z_new
            m4 = self.m4
            self.m4 += 1
            v_inv = 0
            if m4 % (2 * self.P) == 0:
                v_inv = self._inv[(m4 // (2 * self.P) + self.inv_offset) % len(self._inv)]
            # Reconstruct c from decided mapper points, never warped line symbols.
            u0p = vd.rotate_cw(vd.QUARTER[qa], ra)
            u1p = vd.rotate_cw(vd.QUARTER[qb], rb)
            if not any(self.h):
                # Zero MP coefficients imply exactly zero p and c; retain the
                # same reconstructed history without running their quantizers.
                cs=[0j,0j]
                self.x_hist=[u1p,u0p,self.x_hist[0]]
            else:
                cs = []
                for u in (u0p, u1p):
                    p = vd._round_complex(sum(x*h for x,h in zip(self.x_hist, self.h)), 2**-7)
                    c = vd._round_complex(p, 2**self.w)
                    self.x_hist = [u+c-p] + self.x_hist[:2]
                    cs.append(c)
            u0_bit = self.conv.y0 ^ v_inv ^ vd.modulo_bit(*cs)
            diff = (rb - z_new - u0_bit) % 4
            if diff % 2:
                self.errors += 1
            i1 = diff // 2 % 2
            u0p = vd.rotate_cw(vd.QUARTER[qa], ra)
            u1p = vd.rotate_cw(vd.QUARTER[qb], rb)
            self.conv.step(vd.TABLE13[vd.subset_label(u0p + cs[0])][vd.subset_label(u1p + cs[1])])
            mask = (1 << q) - 1
            qbits_a = [(qa >> t) & 1 for t in range(q)]
            qbits_b = [(qb >> t) & 1 for t in range(q)]
            rings += [qa >> q, qb >> q]
            groups.append((i1, i_val & 1, i_val >> 1, qbits_a, qbits_b))
        bits: list[int] = []
        if self.K:
            r0 = inverse_shell(self.shell, rings)
            sb = [(r0 >> t) & 1 for t in range(self.K)]
            bits += sb if high else sb[:-1]
            for g in groups:
                bits += [g[0], g[1], g[2]] + g[3] + g[4]
        else:
            n = self.b if high else self.b - 1
            for j, g in enumerate(groups):
                bits += [g[0], g[1]] + ([g[2]] if j < n - 8 else [])
        return bits, worst


def descramble_stream(bits: list[int], hist: list[int], role: str = "answer") -> list[int]:
    """Self-synchronizing descrambler; ``hist`` carries the previous received (scrambled) bits."""
    tap = 5 if role == "answer" else 18
    allb = hist + bits
    n0 = len(hist)
    out = []
    for i in range(n0, len(allb)):
        a = allb[i - tap] if i >= tap else 0
        b = allb[i - 23] if i >= 23 else 0
        out.append(allb[i] ^ a ^ b)
    return out
