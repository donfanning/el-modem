"""Opt-in bounded streaming detector using a TRN-fitted residual predictor.

The ordinary V.34 detector remains the default. This class deliberately keeps
only the configured Viterbi traceback depth and per-survivor predictor state.
"""
import os
import numpy as np

from v34_viterbi import TrellisDetector
from v34_noise_kernel import _stream_step, _tables, _spatial_tables, available


class PredictiveTrellisDetector(TrellisDetector):
    def __init__(self, receiver, coefficients, residual_history, *, final_state=None, depth=24):
        start = tuple(receiver.drx.conv.d)
        drx=receiver.drx
        from v34_transport import trellis_hold_enabled
        state=(start[0]<<3)|(start[1]<<2)|(start[2]<<1)|start[3]
        if final_state is not None and state!=int(final_state):
            raise ValueError('mapper and predictive trellis terminal states disagree')
        super().__init__(4*drx.M*2**drx.q, drx.J, drx.P, depth=depth, start_state=start,
                         m4_start=receiver.drx.m4, precoder=receiver.h,
                         w=receiver.tx.w, nonlinear=receiver.profile.nonlinear,
                         energy=receiver.energy, x_hist=receiver.drx.x_hist,
                         erasure_hold=trellis_hold_enabled(receiver.profile.precoder),
                         backend='python')
        self.coefficients=np.ascontiguousarray(coefficients,dtype=np.complex128)
        self._classes=_tables(len(self.pts))[1]
        self._counts=_tables(len(self.pts))[2]
        self._transitions=_tables(len(self.pts))[3]
        (self._grid,self._point_labels,self._point_parities,
         self._min_re,self._min_im)=_spatial_tables(len(self.pts))
        self.metric[:]=np.inf
        initial=(start[0]<<3)|(start[1]<<2)|(start[2]<<1)|start[3]
        self.metric[initial]=0.
        self._survivor_x=np.zeros((16,3),np.complex128)
        self._survivor_e=np.zeros((16,3),np.complex128)
        self.search_radius=4
        self._survivor_x[initial]=np.asarray(self.x_hist,np.complex128)
        self._survivor_e[initial]=np.asarray(residual_history,np.complex128)
        self.backend='predictive-numba' if available() else 'predictive-python'
        self.record_snapshots=False

    def feed(self,y0,y1,*,erasure=False):
        if self.erasure_hold:self._update_erasure_hold(bool(erasure))
        if erasure:self.erased_intervals+=1
        interval=self.m4
        inversion=0
        period=2*self.P
        if interval%period==0:
            inversion=self._inv[(interval//period+self.inv_offset)%len(self._inv)]
        metrics,xhist,ehist,previous,choice0,choice1,normalization=_stream_step(
            complex(y0),complex(y1),self.pts,np.asarray(self.h,np.complex128),
            self.coefficients,float(self.energy),bool(self.nonlinear),float(2**self.w),
            self._classes,self._counts,self._transitions,int(inversion),self.metric,
            self._survivor_x,self._survivor_e,self._grid,self._point_labels,
            self._point_parities,self._min_re,self._min_im,
            self.search_radius,bool(erasure))
        self.m4+=1
        self.metric=metrics
        self.metric_offset+=float(normalization)
        self._survivor_x=xhist;self._survivor_e=ehist
        self.x_hist=list(xhist[int(np.argmin(metrics))])
        self.hist.append(dict(prev=previous,s0=choice0,s1=choice1,interval=interval,
                              metric_after=metrics.copy()))
        if len(self.hist)>self.depth:self._decide_oldest()

    def _trace_oldest(self):
        state=int(np.argmin(self.metric));last=None
        for item in reversed(self.hist):
            predecessor=int(item['prev'][state])
            if predecessor<0:raise ValueError('predictive traceback reached an invalid state')
            last=(item,state,predecessor)
            state=predecessor
        item,after,before=last
        return item,after,before

    def _emit_oldest(self):
        item,after,before=self._trace_oldest()
        self.decided.append((complex(self.pts[item['s0'][after]]),
                             complex(self.pts[item['s1'][after]])))
        if self.emit_states:self.decided_states.append(before)
        if self.emit_metrics:
            ordered=np.partition(self.metric,1)
            self.decided_metrics.append(dict(decision_interval=int(item['interval']),
                lookahead_interval=int(self.m4-1),survivor_gap=float(ordered[1]-ordered[0]),
                metric_spread=float(ordered[-1]-ordered[0])))
        self.hist.pop(0)

    def _decide_oldest(self):
        self._emit_oldest()

    def flush(self):
        while self.hist:self._emit_oldest()
