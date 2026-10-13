"""16-state Viterbi detector for V.34 data mode (with optional precoder and nonlinear projection), working on pairs of equalized 2D symbols.

Per 4D interval m the two symbols (u0, u1) are constrained by the trellis: u1's rotation r1 must have the parity
r0 + U0 where U0 = Y0(state) xor V0(m), and the next state follows from the subset labels of u0 and u1 (Table 13 ->
Y2, Y1). For every received pair the detector keeps, for each state, the best path; the best point per
(subset label, rotation parity) class is found once per symbol, so the branch search costs 16 x 8 x 8 and not L^2.
"""

from __future__ import annotations

import numpy as np
import os

import v34_data as vd


class TrellisDetector:
    def __init__(self, L: int, J: int, P: int, depth: int = 24, start_state=(0, 0, 0, 0), m4_start: int = 0, precoder=None, w=1, nonlinear=False, energy=1.0, x_hist=None, trace=False, backend=None):
        self.backend=backend or ('numba' if os.environ.get('V34_NATIVE_TRELLIS')=='1' else 'python')
        if self.backend not in ('python','numba'):raise ValueError('unknown trellis backend')
        self.pts, self.lab, self.par = [], [], []
        for qi in range(L // 4):
            for r in range(4):
                p = vd.rotate_cw(vd.QUARTER[qi], r)
                self.pts.append(p)
                self.lab.append(vd.subset_label(p))
                self.par.append(r & 1)
        self.pts = np.array(self.pts)
        self.lab = np.array(self.lab)
        self.par = np.array(self.par)
        self.cls = [[np.nonzero((self.lab == s) & (self.par == p))[0] for p in (0, 1)] for s in range(8)]
        # classes laid out contiguously so one reduceat gives the best distance of every (label, parity) class
        order, starts, self.cls_ids = [], [], []
        for s in range(8):
            for p in (0, 1):
                c = self.cls[s][p]
                if len(c):
                    starts.append(len(order))
                    order += list(c)
                    self.cls_ids.append((s, p))
        self._order = np.array(order)
        self._starts = np.array(starts)
        self._cls_s = np.array([c[0] for c in self.cls_ids])
        self._cls_p = np.array([c[1] for c in self.cls_ids])
        self.J, self.P = J, P
        self._inv = vd.INVERSION[J]
        self.inv_offset = 2 * J - 2
        self.m4 = m4_start
        self.depth = depth
        # state = (d1, d2, d3, d4) as a 4-bit integer d1<<3 | d2<<2 | d3<<1 | d4
        self.next = np.zeros((16, 8, 8), dtype=int)
        for st in range(16):
            d = [(st >> 3) & 1, (st >> 2) & 1, (st >> 1) & 1, st & 1]
            for s0 in range(8):
                for s1 in range(8):
                    nib = vd.TABLE13[s0][s1]
                    y2, y1 = (nib >> 1) & 1, nib & 1
                    d1, d2, d3, d4 = d
                    n = [d4, d1 ^ y2 ^ d4, d2 ^ y2, d3 ^ y1]
                    self.next[st, s0, s1] = (n[0] << 3) | (n[1] << 2) | (n[2] << 1) | n[3]
        flat_next = self.next.reshape(-1)
        self._cand = np.array([np.nonzero(flat_next == n)[0] for n in range(16)])   # (16, 64) transitions into each state
        self._cand_prev = self._cand // 64
        self._cand_s0 = (self._cand % 64) // 8
        self._cand_s1 = self._cand % 8
        self.y0 = np.array([st & 1 for st in range(16)])
        s0 = (start_state[0] << 3) | (start_state[1] << 2) | (start_state[2] << 1) | start_state[3]
        self.metric = np.full(16, 1e18)
        self.metric[s0] = 0.0
        self.hist: list[dict] = []
        self.decided: list[tuple[complex, complex]] = []
        self.emit_states = False
        self.decided_states: list[int] = []
        self.snapshots: list[dict] = []     # per decided interval: surviving state + metric, for offline reachability checks
        self.record_snapshots=True
        self.trace = [] if trace else None  # opt-in; retained even after flush
        self.metric_offset = 0.0
        self.h = list(precoder) if precoder is not None else [0j]*3
        self.x_hist = list(x_hist) if x_hist is not None else [0j]*3
        self.w, self.nonlinear, self.energy = w, nonlinear, energy
        self._layouts = {}
        if self.backend=='numba':
            from v34_trellis_native import layouts, njit
            if njit is None:raise RuntimeError('native trellis requires Numba')
            # The njit kernel reimplements the production detector (unprecode and its
            # x_hist update, label layout, distances, best-point selection) and never
            # calls these methods. A subclass that overrides any of them would silently
            # be decoded under production semantics instead, so refuse rather than
            # decode under the wrong detector. Attributes a subclass reassigns after
            # __init__ (pts, lab, par) cannot be detected here; such a subclass must
            # pin backend='python' as ChannelOutputDetector does.
            overridden=sorted(n for n,v in vars(TrellisDetector).items()
                if callable(v) and n!='__init__' and getattr(type(self),n,None) is not v)
            if overridden:
                raise ValueError('native trellis cannot honour overridden detector methods: '
                                 +', '.join(overridden)+'; use backend=\'python\'')
            self._native_labels=layouts(self.pts)
            self._native_previous=np.empty((self.depth+1,16),np.int64)

    def _layout(self, c):
        # Figure 9 repeats every eight coordinate units.
        key = (int(c.real) % 8, int(c.imag) % 8)
        if key not in self._layouts:
            lab = np.array([vd.subset_label(p + c) for p in self.pts])
            cls = [[np.nonzero((lab == s) & (self.par == p))[0] for p in (0,1)] for s in range(8)]
            order, starts, ids = [], [], []
            for s in range(8):
                for p in (0,1):
                    ix = cls[s][p]
                    if len(ix):
                        starts.append(len(order)); order.extend(ix); ids.append((s,p))
            self._layouts[key] = cls, np.array(order), np.array(starts), np.array(ids)
        return self._layouts[key]

    def _distances(self, y, delta):
        x = self.pts+delta
        if self.nonlinear:
            z=.3125*np.abs(x)**2/self.energy
            x=x*(1+z/6+z*z/120)
        return np.abs(x-y)**2

    def _best(self, y, c=0j, wave=None, delta=0j):
        cls, order, starts, ids = self._layout(c)
        distance = np.abs(self.pts-y)**2 if wave is None else self._distances(wave,delta)
        mins = np.minimum.reduceat(distance[order], starts)
        best = np.full((8,2), 1e18)
        best[ids[:,0], ids[:,1]] = mins
        return best

    def _argbest(self, y, s, p, c=0j, wave=None, delta=0j):
        ix = self._layout(c)[0][s][p]
        distance = np.abs(self.pts-y)**2 if wave is None else self._distances(wave,delta)
        return complex(self.pts[ix[int(np.argmin(distance[ix]))]])

    def _unprecode(self, y):
        from v34_receive import inverse_projection
        x = complex(inverse_projection(y, self.energy)) if self.nonlinear else y
        p = vd._round_complex(sum(x*h for x,h in zip(self.x_hist,self.h)), 2**-7)
        c = vd._round_complex(p, 2**self.w)
        # Decision feedback avoids changing c at a quantizer tie solely because
        # of line noise. Histories contain reconstructed x, not noisy samples.
        u = complex(self.pts[int(np.argmin(np.abs(self.pts-(x+p-c))))])
        self.x_hist = [u+c-p] + self.x_hist[:2]
        return x+p-c, c, c-p

    def feed(self, y0: complex, y1: complex) -> None:
        interval = self.m4
        metric_before = self.metric.copy()
        history_before = list(self.x_hist)
        v0 = 0
        P2 = 2 * self.P
        if self.m4 % P2 == 0:
            v0 = self._inv[(self.m4 // P2 + self.inv_offset) % len(self._inv)]
        self.m4 += 1
        if self.backend=='numba':
            return self._native_feed(y0,y1,interval,v0,metric_before,history_before)
        wave0,wave1=y0,y1
        y0, c0, delta0 = self._unprecode(y0)
        y1, c1, delta1 = self._unprecode(y1)
        b0 = self._best(y0, c0, wave0, delta0)
        b1 = self._best(y1, c1, wave1, delta1)
        cost = np.zeros((2, 8, 8))
        par0 = np.zeros((2, 8, 8), dtype=int)
        for U in (0, 1):
            a = b0[:, 0][:, None] + b1[:, 0 ^ U][None, :]
            b = b0[:, 1][:, None] + b1[:, 1 ^ U][None, :]
            par0[U] = (b < a)
            cost[U] = np.minimum(a, b)
        Us = self.y0 ^ v0 ^ vd.modulo_bit(c0, c1)                                   # U0 for each state
        flat = (self.metric[:, None, None] + cost[Us]).reshape(-1)
        cand_cost = flat[self._cand]                        # (16, 64)
        k = np.argmin(cand_cost, axis=1)
        rows = np.arange(16)
        new = cand_cost[rows, k]
        prev = self._cand_prev[rows, k]
        bs0 = self._cand_s0[rows, k]
        bs1 = self._cand_s1[rows, k]
        bp0 = par0[Us[prev], bs0, bs1]
        normalization = float(new.min())
        new = new - normalization
        self.metric_offset += normalization
        self.metric = new
        h = {"prev": prev, "s0": bs0, "s1": bs1, "p0": bp0, "y0": y0, "y1": y1, "U": Us, "c0": c0, "c1": c1, "wave0":wave0,"wave1":wave1,"delta0":delta0,"delta1":delta1,
             "interval": interval, "metric_before": metric_before, "metric_after": new.copy()}
        self.hist.append(h)
        if self.trace is not None:
            self.trace.append(dict(h, branch_cost=cost.copy(), parity_choice=par0.copy(),
                best0=b0.copy(), best1=b1.copy(), inversion_bit=v0,
                x_hist_before=history_before, metric_normalization=normalization,
                cumulative_metric_offset=self.metric_offset))
        if len(self.hist) > self.depth:
            self._decide()

    def _native_feed(self,wave0,wave1,interval,v0,metric_before,history_before):
        from v34_trellis_native import advance
        ys,cs,delta,history,best,indices,cost,parity,Us,new,prev,s0,s1,p0,normalization=advance(
            self,wave0,wave1,v0,self._native_labels)
        self.x_hist=list(history);self.metric=new;self.metric_offset+=normalization
        h=dict(prev=prev,s0=s0,s1=s1,p0=p0,y0=ys[0],y1=ys[1],U=Us,
            c0=cs[0],c1=cs[1],wave0=wave0,wave1=wave1,delta0=delta[0],delta1=delta[1],
             interval=interval,metric_before=metric_before,metric_after=new.copy(),native_indices=indices)
        self.hist.append(h)
        self._native_previous[interval%len(self._native_previous)]=prev
        if self.trace is not None:
            self.trace.append(dict(h,branch_cost=cost.copy(),parity_choice=parity.copy(),
                best0=best[0].copy(),best1=best[1].copy(),inversion_bit=v0,
                x_hist_before=history_before,metric_normalization=normalization,
                cumulative_metric_offset=self.metric_offset))
        if len(self.hist)>self.depth:self._decide()

    def _chosen_pair(self,h,s0,s1,p0,U0):
        if 'native_indices' in h:
            return (complex(self.pts[h['native_indices'][0,s0,p0]]),
                    complex(self.pts[h['native_indices'][1,s1,p0^U0]]))
        return (self._argbest(h['y0'],s0,p0,h['c0'],h['wave0'],h['delta0']),
                self._argbest(h['y1'],s1,p0^U0,h['c1'],h['wave1'],h['delta1']))

    def _path(self, steps: int, with_states=False):
        """Trace back from the best state; returns decisions for the oldest of the last ``steps`` intervals."""
        st = int(np.argmin(self.metric))
        if self.backend=='numba':
            from v34_trellis_native import traceback
            before,after=traceback(self._native_previous,self.hist[-1]['interval'],steps,st)
            h=self.hist[len(self.hist)-steps]
            last=(h,int(h['s0'][after]),int(h['s1'][after]),int(h['p0'][after]),int(h['U'][before]))
            return last+(int(before),int(after)) if with_states else last
        for t in range(len(self.hist) - 1, len(self.hist) - 1 - steps, -1):
            h = self.hist[t]
            s0, s1, p0, U = h["s0"][st], h["s1"][st], h["p0"][st], None
            pst = h["prev"][st]
            U0 = int(h["U"][pst])
            last = (h, int(s0), int(s1), int(p0), U0)
            states = (int(pst), st)
            st = int(pst)
        return last + states if with_states else last

    def _decide(self) -> None:
        h, s0, s1, p0, U0, before, after = self._path(len(self.hist), with_states=True)
        self.decided.append(self._chosen_pair(h,s0,s1,p0,U0))
        if self.emit_states:
            self.decided_states.append(before)
        if self.record_snapshots:
            self.snapshots.append(self._snapshot(h, s0, s1, p0, U0, before, after))
        self.hist.pop(0)

    def flush(self) -> None:
        while self.hist:
            n = len(self.hist)
            h, s0, s1, p0, U0, before, after = self._path(n, with_states=True)
            self.decided.append(self._chosen_pair(h,s0,s1,p0,U0))
            if self.emit_states:
                self.decided_states.append(before)
            if self.record_snapshots:
                self.snapshots.append(self._snapshot(h, s0, s1, p0, U0, before, after))
            self.hist.pop(0)

    def _snapshot(self, h, s0, s1, p0, U0, before, after):
        """Per-decided-interval record of what the trellis survived to, for offline reachability checks."""
        return dict(
            s0=int(s0), s1=int(s1), p0=int(p0), U0=int(U0),
            c0=[h["c0"].real, h["c0"].imag], c1=[h["c1"].real, h["c1"].imag],
            y0=[h["y0"].real, h["y0"].imag], y1=[h["y1"].real, h["y1"].imag],
            wave0=[h["wave0"].real, h["wave0"].imag], wave1=[h["wave1"].real, h["wave1"].imag],
            delta0=[h["delta0"].real, h["delta0"].imag], delta1=[h["delta1"].real, h["delta1"].imag],
            interval=h["interval"], state_before=before, state_after=after,
            metric_before=h["metric_before"].tolist(), metric=h["metric_after"].tolist(),
            decision_interval=self.m4-1, decision_metric=self.metric.tolist())
