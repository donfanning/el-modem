"""Optional compiled kernel for the lab-only reduced-state B1 search.

Strict floating point, normative half-toward-zero quantization, and stable
candidate ordering. This changes execution, not the search/pruning rule.
"""
import math
from functools import lru_cache
import numpy as np

try:
    from numba import njit
except ImportError:
    njit = None


def available():
    return njit is not None


def _quantize(z, step):
    re = abs(z.real) / step
    im = abs(z.imag) / step
    r = math.floor(re)
    i = math.floor(im)
    return complex(math.copysign((r + (re-r > .5))*step, z.real),
                   math.copysign((i + (im-i > .5))*step, z.imag))


def _project(x, nonlinear, energy):
    if not nonlinear:
        return x
    z = .3125 * (x.real*x.real+x.imag*x.imag) / energy
    return x * (1 + z/6 + z*z/120)


def _search(samples, points, h, coefficients, energy, nonlinear, step,
            classes, counts, transitions, inversions, period, inv_offset, initial):
    n = len(samples)//2
    metrics = np.full(16, np.inf)
    metrics[initial] = 0.
    histories = np.zeros((16, 3), np.complex128)
    residuals = np.zeros((16, 3), np.complex128)
    previous = np.full((n, 16), -1, np.int64)
    chosen = np.zeros((n, 16, 2), np.int64)
    # Scratch belongs to this invocation; no shared mutable state or Python
    # objects in the symbol loop. At most two candidates in each of 16 classes.
    x0 = np.empty(len(points), np.complex128)
    e0 = np.empty(len(points), np.complex128)
    cost0 = np.empty(len(points), np.float64)
    first = np.empty(32, np.int64)
    labels = np.empty(32, np.int64)
    parities = np.empty(32, np.int64)
    for interval in range(n):
        y0, y1 = samples[2*interval], samples[2*interval+1]
        inversion = inversions[(interval//period+inv_offset)%len(inversions)] if interval%period == 0 else 0
        new = np.full(16, np.inf)
        nh = np.zeros((16, 3), np.complex128)
        ne = np.zeros((16, 3), np.complex128)
        for state in range(16):
            if not math.isfinite(metrics[state]):
                continue
            history = histories[state]
            eh = residuals[state]
            p0 = _quantize(history[0]*h[0]+history[1]*h[1]+history[2]*h[2], 2**-7)
            c0 = _quantize(p0, step)
            prediction = coefficients[0]*eh[0]+coefficients[1]*eh[1]+coefficients[2]*eh[2]
            for i in range(len(points)):
                x0[i] = points[i]+c0-p0
                e0[i] = y0-_project(x0[i], nonlinear, energy)
                innovation = e0[i]-prediction
                cost0[i] = innovation.real*innovation.real+innovation.imag*innovation.imag
            cr, ci = int(c0.real)%8, int(c0.imag)%8
            rows = 0
            for label in range(8):
                for parity in range(2):
                    cls = 2*label+parity
                    best, runner = -1, -1
                    bc, rc = np.inf, np.inf
                    for j in range(counts[cr, ci, cls]):
                        index = classes[cr, ci, cls, j]
                        cost = cost0[index]
                        if cost < bc:
                            runner, rc = best, bc
                            best, bc = index, cost
                        elif cost < rc:
                            runner, rc = index, cost
                    if best >= 0:
                        first[rows], labels[rows], parities[rows] = best, label, parity
                        rows += 1
                    if runner >= 0:
                        first[rows], labels[rows], parities[rows] = runner, label, parity
                        rows += 1
            for row in range(rows):
                i0 = first[row]
                p1 = _quantize(x0[i0]*h[0]+history[0]*h[1]+history[1]*h[2], 2**-7)
                c1 = _quantize(p1, step)
                modulo = (int(round(c0.real/2))+int(round(c0.imag/2))
                          +int(round(c1.real/2))+int(round(c1.imag/2)))%2
                parity1 = parities[row] ^ (state & 1) ^ inversion ^ modulo
                cr1, ci1 = int(c1.real)%8, int(c1.imag)%8
                predicted1 = coefficients[0]*e0[i0]+coefficients[1]*eh[0]+coefficients[2]*eh[1]
                for label1 in range(8):
                    best, best_cost = -1, np.inf
                    best_x, best_e = 0j, 0j
                    cls = 2*label1+parity1
                    for j in range(counts[cr1, ci1, cls]):
                        index = classes[cr1, ci1, cls, j]
                        x = points[index]+(c1-p1)
                        e = y1-_project(x, nonlinear, energy)
                        innovation = e-predicted1
                        cost = innovation.real*innovation.real+innovation.imag*innovation.imag
                        if cost < best_cost:
                            best, best_cost, best_x, best_e = index, cost, x, e
                    if best < 0:
                        continue
                    target = transitions[state, labels[row], label1]
                    cost = (metrics[state]+cost0[i0])+best_cost
                    if cost < new[target]:
                        new[target] = cost
                        previous[interval, target] = state
                        chosen[interval, target, 0] = i0
                        chosen[interval, target, 1] = best
                        nh[target, 0], nh[target, 1], nh[target, 2] = best_x, x0[i0], history[0]
                        ne[target, 0], ne[target, 1], ne[target, 2] = best_e, e0[i0], eh[0]
        if not np.any(np.isfinite(new)):
            raise ValueError('no surviving candidate path')
        metrics, histories, residuals = new, nh, ne
    state = int(np.argmin(metrics))
    total = metrics[state]
    final_state = state
    final_history = histories[state].copy()
    final_residual = residuals[state].copy()
    output = np.empty(2*n, np.complex128)
    for interval in range(n-1, -1, -1):
        output[2*interval] = points[chosen[interval, state, 0]]
        output[2*interval+1] = points[chosen[interval, state, 1]]
        state = previous[interval, state]
    if state != initial:
        raise ValueError('traceback failed to reach initial state')
    return output, total, final_history, final_residual, final_state


def _stream_step_exhaustive(y0, y1, points, h, coefficients, energy, nonlinear, step,
                 classes, counts, transitions, inversion, metrics, histories,
                 residuals, erasure=False):
    """Advance one 4D interval while retaining per-survivor channel state."""
    new = np.full(16, np.inf)
    nh = np.zeros((16, 3), np.complex128)
    ne = np.zeros((16, 3), np.complex128)
    previous = np.full(16, -1, np.int64)
    chosen0 = np.zeros(16, np.int64)
    chosen1 = np.zeros(16, np.int64)
    x0 = np.empty(len(points), np.complex128)
    e0 = np.empty(len(points), np.complex128)
    cost0 = np.empty(len(points), np.float64)
    first = np.empty(32, np.int64)
    labels = np.empty(32, np.int64)
    parities = np.empty(32, np.int64)
    for state in range(16):
        if not math.isfinite(metrics[state]):
            continue
        history = histories[state]
        eh = residuals[state]
        p0 = _quantize(history[0]*h[0]+history[1]*h[1]+history[2]*h[2], 2**-7)
        c0 = _quantize(p0, step)
        prediction = coefficients[0]*eh[0]+coefficients[1]*eh[1]+coefficients[2]*eh[2]
        for i in range(len(points)):
            x0[i] = points[i]+c0-p0
            e0[i] = 0j if erasure else y0-_project(x0[i], nonlinear, energy)
            innovation = e0[i]-prediction
            cost0[i] = 0. if erasure else innovation.real*innovation.real+innovation.imag*innovation.imag
        cr, ci = int(c0.real)%8, int(c0.imag)%8
        rows = 0
        for label in range(8):
            for parity in range(2):
                cls = 2*label+parity
                best, runner = -1, -1
                bc, rc = np.inf, np.inf
                for j in range(counts[cr, ci, cls]):
                    index = classes[cr, ci, cls, j]
                    value = cost0[index]
                    if value < bc:
                        runner, rc = best, bc
                        best, bc = index, value
                    elif value < rc:
                        runner, rc = index, value
                if best >= 0:
                    first[rows], labels[rows], parities[rows] = best, label, parity
                    rows += 1
                if runner >= 0:
                    first[rows], labels[rows], parities[rows] = runner, label, parity
                    rows += 1
        for row in range(rows):
            i0 = first[row]
            p1 = _quantize(x0[i0]*h[0]+history[0]*h[1]+history[1]*h[2], 2**-7)
            c1 = _quantize(p1, step)
            modulo = (int(round(c0.real/2))+int(round(c0.imag/2))
                      +int(round(c1.real/2))+int(round(c1.imag/2)))%2
            parity1 = parities[row] ^ (state & 1) ^ inversion ^ modulo
            cr1, ci1 = int(c1.real)%8, int(c1.imag)%8
            predicted1 = coefficients[0]*e0[i0]+coefficients[1]*eh[0]+coefficients[2]*eh[1]
            for label1 in range(8):
                best, best_cost = -1, np.inf
                best_x, best_e = 0j, 0j
                cls = 2*label1+parity1
                for j in range(counts[cr1, ci1, cls]):
                    index = classes[cr1, ci1, cls, j]
                    x = points[index]+(c1-p1)
                    e = 0j if erasure else y1-_project(x, nonlinear, energy)
                    innovation = e-predicted1
                    value = 0. if erasure else innovation.real*innovation.real+innovation.imag*innovation.imag
                    if value < best_cost:
                        best, best_cost, best_x, best_e = index, value, x, e
                if best < 0:
                    continue
                target = transitions[state, labels[row], label1]
                value = (metrics[state]+cost0[i0])+best_cost
                if value < new[target]:
                    new[target] = value
                    previous[target] = state
                    chosen0[target], chosen1[target] = i0, best
                    nh[target, 0], nh[target, 1], nh[target, 2] = best_x, x0[i0], history[0]
                    ne[target, 0], ne[target, 1], ne[target, 2] = best_e, e0[i0], eh[0]
    if not np.any(np.isfinite(new)):
        raise ValueError('no surviving predictive candidate path')
    normalization = float(np.min(new))
    return (new-normalization, nh, ne, previous, chosen0, chosen1, normalization)


def _inverse_project(y, energy):
    """Invert the monotone radial nonlinear projection used by V.34."""
    radius=math.sqrt(y.real*y.real+y.imag*y.imag)
    if radius==0.0:return 0j
    a=.3125/(6*energy);b=.3125**2/(120*energy**2)
    x=min(radius,(radius/a)**(1/3),(radius/b)**.2)
    for _ in range(5):
        x-=(x+a*x**3+b*x**5-radius)/(1+3*a*x*x+5*b*x**4)
    return y*(x/radius)


def _stream_step(y0, y1, points, h, coefficients, energy, nonlinear, step,
                 classes, counts, transitions, inversion, metrics, histories,
                 residuals, grid, labels, parities, min_re, min_im, radius=4,
                 erasure=False):
    """Exact bounded-neighborhood trellis update with exhaustive certification.

    The radial nonlinear map expands distances from its inverse target: both
    eigenvalues of its Jacobian are >= 1.  Thus a candidate outside the local
    grid box cannot beat a class candidate whose metric is below the box's
    minimum outside-distance.  Any class that cannot be certified falls back
    to the original exhaustive scan.
    """
    new=np.full(16,np.inf);nh=np.zeros((16,3),np.complex128)
    ne=np.zeros((16,3),np.complex128);previous=np.full(16,-1,np.int64)
    chosen0=np.zeros(16,np.int64);chosen1=np.zeros(16,np.int64)
    first=np.empty(32,np.int64);first_label=np.empty(32,np.int64)
    first_parity=np.empty(32,np.int64)
    for state in range(16):
        if not math.isfinite(metrics[state]):continue
        history=histories[state];eh=residuals[state]
        p0=_quantize(history[0]*h[0]+history[1]*h[1]+history[2]*h[2],2**-7)
        c0=_quantize(p0,step);delta0=c0-p0
        prediction=coefficients[0]*eh[0]+coefficients[1]*eh[1]+coefficients[2]*eh[2]
        target0=complex(y0)-prediction
        center0=(_inverse_project(target0,energy) if nonlinear else target0)-delta0
        center_error0=abs(_project(center0+delta0,nonlinear,energy)-target0) if nonlinear else 0.0
        cr,ci=int(c0.real)%8,int(c0.imag)%8
        best=np.full(16,np.inf);runner=np.full(16,np.inf)
        best_ix=np.full(16,-1,np.int64);runner_ix=np.full(16,-1,np.int64)
        best_e=np.zeros(len(points),np.complex128);best_x=np.zeros(len(points),np.complex128)
        point_cost=np.full(len(points),np.inf)
        qre=(center0.real-min_re)/2.;qim=(center0.imag-min_im)/2.
        ire=int(math.floor(qre));iim=int(math.floor(qim))
        r0=max(0.,2.*radius-center_error0)
        # Erasures intentionally assign every reachable branch zero cost, so
        # geometry cannot certify the reference's lowest-index tie break.
        local_r0=(r0>0.) and not erasure
        if local_r0:
            for gi in range(max(0,ire-radius),min(grid.shape[0],ire+radius+1)):
                for gj in range(max(0,iim-radius),min(grid.shape[1],iim+radius+1)):
                    index=grid[gi,gj]
                    if index<0:continue
                    x=points[index]+delta0
                    e=0j if erasure else complex(y0)-_project(x,nonlinear,energy)
                    innovation=e-prediction
                    value=0. if erasure else innovation.real*innovation.real+innovation.imag*innovation.imag
                    point_cost[index]=value
                    cls=int(labels[cr,ci,index])*2+int(parities[index])
                    if value<best[cls] or (value==best[cls] and index<best_ix[cls]):
                        runner[cls],runner_ix[cls]=best[cls],best_ix[cls]
                        best[cls],best_ix[cls]=value,index
                    elif value<runner[cls] or (value==runner[cls] and index<runner_ix[cls]):
                        runner[cls],runner_ix[cls]=value,index
                    best_e[index],best_x[index]=e,x
            for cls in range(16):
                if counts[cr,ci,cls] and (runner_ix[cls]<0 or runner[cls]>=r0*r0):
                    local_r0=False;break
        if not local_r0:
            best[:]=np.inf;runner[:]=np.inf;best_ix[:]=-1;runner_ix[:]=-1
            for index in range(len(points)):
                x=points[index]+delta0
                e=0j if erasure else complex(y0)-_project(x,nonlinear,energy)
                innovation=e-prediction
                value=0. if erasure else innovation.real*innovation.real+innovation.imag*innovation.imag
                point_cost[index]=value
                cls=int(labels[cr,ci,index])*2+int(parities[index])
                if value<best[cls] or (value==best[cls] and index<best_ix[cls]):
                    runner[cls],runner_ix[cls]=best[cls],best_ix[cls]
                    best[cls],best_ix[cls]=value,index
                elif value<runner[cls] or (value==runner[cls] and index<runner_ix[cls]):
                    runner[cls],runner_ix[cls]=value,index
                best_e[index],best_x[index]=e,x
        rows=0
        for cls in range(16):
            if best_ix[cls]>=0:
                first[rows]=best_ix[cls];first_label[rows]=cls//2
                first_parity[rows]=cls%2;rows+=1
            if runner_ix[cls]>=0:
                first[rows]=runner_ix[cls];first_label[rows]=cls//2
                first_parity[rows]=cls%2;rows+=1
        for row in range(rows):
            i0=first[row];x0=best_x[i0];e0=best_e[i0]
            p1=_quantize(x0*h[0]+history[0]*h[1]+history[1]*h[2],2**-7)
            c1=_quantize(p1,step);delta1=c1-p1
            modulo=(int(round(c0.real/2))+int(round(c0.imag/2))+
                    int(round(c1.real/2))+int(round(c1.imag/2)))%2
            parity1=int(first_parity[row])^(state&1)^inversion^modulo
            predicted1=coefficients[0]*e0+coefficients[1]*eh[0]+coefficients[2]*eh[1]
            target1=complex(y1)-predicted1
            center1=(_inverse_project(target1,energy) if nonlinear else target1)-delta1
            center_error1=abs(_project(center1+delta1,nonlinear,energy)-target1) if nonlinear else 0.0
            cr1,ci1=int(c1.real)%8,int(c1.imag)%8
            best1=np.full(8,np.inf);best_ix1=np.full(8,-1,np.int64)
            best_e1=np.zeros(8,np.complex128);best_x1=np.zeros(8,np.complex128)
            qre1=(center1.real-min_re)/2.;qim1=(center1.imag-min_im)/2.
            ire1=int(math.floor(qre1));iim1=int(math.floor(qim1))
            r1=max(0.,2.*radius-center_error1);local_r1=(r1>0.) and not erasure
            if local_r1:
                for gi in range(max(0,ire1-radius),min(grid.shape[0],ire1+radius+1)):
                    for gj in range(max(0,iim1-radius),min(grid.shape[1],iim1+radius+1)):
                        index=grid[gi,gj]
                        if index<0 or int(parities[index])!=parity1:continue
                        label=int(labels[cr1,ci1,index])
                        x=points[index]+delta1
                        e=0j if erasure else complex(y1)-_project(x,nonlinear,energy)
                        innovation=e-predicted1
                        value=0. if erasure else innovation.real*innovation.real+innovation.imag*innovation.imag
                        if value<best1[label] or (value==best1[label] and index<best_ix1[label]):
                            best1[label],best_ix1[label]=value,index
                            best_e1[label],best_x1[label]=e,x
                for label in range(8):
                    if counts[cr1,ci1,label*2+parity1] and (best_ix1[label]<0 or best1[label]>=r1*r1):
                        local_r1=False;break
            if not local_r1:
                best1[:]=np.inf;best_ix1[:]=-1;best_e1[:]=0j;best_x1[:]=0j
                for label in range(8):
                    cls=label*2+parity1
                    for j in range(counts[cr1,ci1,cls]):
                        index=classes[cr1,ci1,cls,j]
                        x=points[index]+delta1
                        e=0j if erasure else complex(y1)-_project(x,nonlinear,energy)
                        innovation=e-predicted1
                        value=0. if erasure else innovation.real*innovation.real+innovation.imag*innovation.imag
                        if value<best1[label] or (value==best1[label] and index<best_ix1[label]):
                            best1[label],best_ix1[label]=value,index
                            best_e1[label],best_x1[label]=e,x
            for label1 in range(8):
                index=best_ix1[label1]
                if index<0:continue
                value0=point_cost[i0]
                value=metrics[state]+value0+best1[label1]
                target=transitions[state,int(first_label[row]),label1]
                if value<new[target]:
                    new[target]=value;previous[target]=state
                    chosen0[target],chosen1[target]=i0,index
                    nh[target,0],nh[target,1],nh[target,2]=best_x1[label1],x0,history[0]
                    ne[target,0],ne[target,1],ne[target,2]=best_e1[label1],e0,eh[0]
    if not np.any(np.isfinite(new)):
        raise ValueError('no surviving predictive candidate path')
    normalization=float(np.min(new))
    return new-normalization,nh,ne,previous,chosen0,chosen1,normalization


if njit is not None:
    _quantize = njit(cache=True, fastmath=False, inline='always')(_quantize)
    _project = njit(cache=True, fastmath=False, inline='always')(_project)
    _search = njit(cache=True, nogil=True, fastmath=False)(_search)
    _stream_step_exhaustive = njit(cache=True, nogil=True, fastmath=False)(_stream_step_exhaustive)
    _inverse_project = njit(cache=True, fastmath=False, inline='always')(_inverse_project)
    _stream_step = njit(cache=True, nogil=True, fastmath=False)(_stream_step)


@lru_cache(maxsize=8)
def _tables(length):
    from v34_viterbi import TrellisDetector
    detector = TrellisDetector(length, 8, 15)
    # Layout only depends on constellation length and c modulo eight.
    counts = np.zeros((8, 8, 16), np.int64)
    layouts = []
    re_points = detector.pts.real.astype(np.int64)
    im_points = detector.pts.imag.astype(np.int64)
    width = 0
    for re in range(8):
        for im in range(8):
            # Vectorize the exact integer Figure 9 calculation, avoiding
            # thousands of Python scalar label calls at the first B1 burst.
            a = (re_points+re-1)//2
            b = (im_points+im-1)//2
            labels = ((((a+1)//2 ^ (b+1)//2)&1)<<2) | ((a&1)<<1) | ((a+b)&1)
            cls = [[np.flatnonzero((labels==label)&(detector.par==parity))
                    for parity in range(2)] for label in range(8)]
            layouts.append(cls)
            width = max(width, max(len(ix) for group in cls for ix in group))
    classes = np.full((8, 8, 16, width), -1, np.int64)
    for re in range(8):
        for im in range(8):
            for label in range(8):
                for parity in range(2):
                    ix = layouts[re*8+im][label][parity]
                    counts[re, im, label*2+parity] = len(ix)
                    classes[re, im, label*2+parity, :len(ix)] = ix
    return detector.pts, classes, counts, detector.next


@lru_cache(maxsize=8)
def _spatial_tables(length):
    """Return exact lattice lookup and class labels for local certified search."""
    from v34_viterbi import TrellisDetector
    detector=TrellisDetector(length,8,15)
    points=detector.pts
    min_re=float(np.min(points.real));min_im=float(np.min(points.imag))
    width=int((np.max(points.real)-min_re)/2)+1
    height=int((np.max(points.imag)-min_im)/2)+1
    grid=np.full((width,height),-1,np.int64)
    grid[((points.real-min_re)/2).astype(np.int64),
         ((points.imag-min_im)/2).astype(np.int64)]=np.arange(len(points))
    labels=np.empty((8,8,len(points)),np.int64)
    _,classes,counts,_=_tables(length)
    for re in range(8):
        for im in range(8):
            for cls in range(16):
                for j in range(counts[re,im,cls]):
                    labels[re,im,classes[re,im,cls,j]]=cls//2
    return grid,labels,detector.par.astype(np.int64),min_re,min_im


def warmup():
    """Compile batch B1 and bounded streaming kernels before live media starts."""
    if not available():
        return False
    points, classes, counts, transitions = _tables(16)
    grid,labels,parities,min_re,min_im=_spatial_tables(16)
    _search(np.zeros(2, np.complex128), points, np.zeros(3, np.complex128),
            np.zeros(3, np.complex128), 1., False, 2., classes, counts,
            transitions, np.zeros(1, np.int64), 30, 14, 0)
    metrics=np.full(16,np.inf);metrics[0]=0.
    _stream_step(0j,0j,points,np.zeros(3,np.complex128),np.zeros(3,np.complex128),
                 1.,False,2.,classes,counts,transitions,0,metrics,
                 np.zeros((16,3),np.complex128),np.zeros((16,3),np.complex128),
                 grid,labels,parities,min_re,min_im,4,False)
    return True


def detect_native(samples, receiver, coefficients, *, return_state=False):
    if not available():
        raise RuntimeError('Numba predictive backend is unavailable')
    samples = np.ascontiguousarray(samples, dtype=np.complex128).reshape(-1)
    coefficients = np.ascontiguousarray(coefficients, dtype=np.complex128)
    if coefficients.shape != (3,):
        raise ValueError('fixed three-coefficient predictor required')
    if len(samples)%2:
        raise ValueError('complete 4D symbol pairs required')
    if not np.all(np.isfinite(samples)) or not np.all(np.isfinite(coefficients)):
        raise ValueError('finite samples and predictor coefficients required')
    if receiver.profile.nonlinear and (not math.isfinite(receiver.energy) or receiver.energy <= 0):
        raise ValueError('nonlinear energy must be finite and positive')
    detector = receiver.detector()
    points, classes, counts, transitions = _tables(len(detector.pts))
    initial = sum(v << (3-i) for i, v in enumerate(receiver.initial_state))
    result=_search(samples, points, np.asarray(receiver.h, np.complex128), coefficients,
                   float(receiver.energy), bool(receiver.profile.nonlinear), float(2**receiver.tx.w),
                   classes, counts, transitions, np.asarray(detector._inv, np.int64),
                   int(2*detector.P), int(detector.inv_offset), initial)
    return result if return_state else result[:2]
