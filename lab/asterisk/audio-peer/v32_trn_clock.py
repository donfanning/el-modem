"""Optional V.32 sampling-rate fit using only known training symbols.

Measure delay against consecutive known TRN blocks before decision-directed
tracking. Delay is in T/2 samples, so clock fraction = -delay slope / 2.
No channel parameters, payload bytes or decoder errors enter this estimator.
"""
import numpy as np
from v32_rx import expected_trn


def training_delay(engine, done, reference):
    count, half = 512, 8
    start = done-count
    first = 2*(engine._s0+start)+engine._ph-engine.ubase
    base = first+2*np.arange(count)
    shifts = np.arange(-4.,4.0001,.0625)
    integer = np.floor(shifts).astype(int)
    taps = np.arange(-half+1,half+1)
    indices = base[None,:,None]+integer[:,None,None]+taps[None,None,:]
    if indices.min() < 0 or indices.max() >= len(engine.u):
        return None
    distance = taps[None,:]-(shifts-integer)[:,None]
    weights = np.sinc(distance)*(.5+.5*np.cos(np.pi*distance/half))
    weights /= weights.sum(axis=1)[:,None]
    samples = np.einsum('gnt,gt->gn',np.asarray(engine.u)[indices],weights)
    desired = reference[start:done]
    denominator = np.sum(np.abs(samples)**2,axis=1)*np.vdot(desired,desired).real
    if not np.all(np.isfinite(denominator)) or np.any(denominator <= 0):
        return None
    correlations = np.abs(samples@desired.conj())**2/denominator
    best = int(np.argmax(correlations))
    if best in (0,len(shifts)-1):
        return None
    left,center,right = correlations[best-1:best+2]
    curvature = left-2*center+right
    offset = .5*(left-right)/curvature if curvature else 0.
    return dict(symbol=start+(count-1)/2,delay=float(shifts[best]+.0625*offset),
                correlation=float(np.sqrt(center)),units='T/2 samples')


class TrainingClock:
    def __init__(self):
        self.key = None
        self.rows = []
        self.reference = None
        self.finished = False
        self.fits = []
        self.peak_rows = 0

    def feed(self, engine, done):
        key = (engine._rx_mode,engine._s0)
        if key != self.key:
            self.key, self.rows, self.reference, self.finished = key, [], None, False
        if self.finished:
            return None
        # Six overlapping 512-symbol correlations cover 1792 TRN symbols;
        # their center-to-center regression baseline is 1280 symbols. Fit while
        # still in the 2976-symbol TRN emitted by our answering engine; waiting
        # for symbol 5000 silently excluded that entire receive direction.
        if 1024 <= done <= 2304 and done % 256 == 0:
            if self.reference is None:
                self.reference = np.asarray(expected_trn(7000,engine._rx_tap))
            row = training_delay(engine,done,self.reference)
            if row is not None:
                self.rows.append(row)
                self.peak_rows = max(self.peak_rows,len(self.rows))
        if done != 2304:
            return None
        fit = dict(stage=key,method='known-TRN correlation',rows=len(self.rows),
                   applied=False,symbol=done,old_step=engine.bb.step)
        if len(self.rows) == 6:
            a = np.array([r['symbol'] for r in self.rows])
            b = np.array([r['delay'] for r in self.rows])
            slope = float(np.dot(a-a.mean(),b-b.mean())/np.dot(a-a.mean(),a-a.mean()))
            ppm = -slope/2*1e6
            fit.update(residual_clock_ppm=ppm,
                       minimum_correlation=min(r['correlation'] for r in self.rows))
            if 20 <= abs(ppm) <= 500 and fit['minimum_correlation'] >= .90:
                engine.bb.step /= 1+ppm*1e-6
                fit.update(applied=True,new_step=engine.bb.step)
        self.fits.append(fit)
        self.release()
        return fit

    def release(self):
        self.rows, self.reference, self.finished = [], None, True

    def report(self):
        return dict(enabled=True,fits=self.fits,peak_rows=self.peak_rows,
                    retained_rows=len(self.rows),retained_reference_symbols=0 if self.reference is None else len(self.reference),
                    units='delay: T/2 samples; residual_clock_ppm: parts per million',
                    selection='six known-TRN correlations >= 0.90; abs(ppm) 20..500; apply at symbol 2304')
