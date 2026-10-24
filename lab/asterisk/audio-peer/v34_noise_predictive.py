"""Received-sample B1 prediction with per-survivor histories; no known branch targets."""
import numpy as np
import v34_data as data
import v34_receive as receive
import v34_b1_reference as independent

def detect(samples, receiver, coefficients, *, backend='auto', return_state=False):
    """Use the optional native lab kernel, retaining an explicit reference path."""
    if backend not in ('auto', 'python', 'numba'):
        raise ValueError('unknown predictive backend')
    if backend != 'python':
        from v34_noise_kernel import available, detect_native
        if available():
            return detect_native(samples, receiver, coefficients, return_state=return_state)
        if backend == 'numba':
            raise RuntimeError('Numba predictive backend is unavailable')
    return detect_reference(samples, receiver, coefficients, return_state=return_state)


def detect_reference(samples, receiver, coefficients, *, return_state=False):
    coefficients=np.asarray(coefficients,complex)
    if coefficients.shape != (3,):
        raise ValueError('fixed three-coefficient predictor required')
    detector=receiver.detector()
    points=detector.pts
    initial=sum(v << (3-i) for i,v in enumerate(receiver.initial_state))
    metrics=np.full(16,np.inf);metrics[initial]=0.
    histories=np.zeros((16,3),complex)
    residuals=np.zeros((16,3),complex)
    records=[]
    def project(x):
        return receive.projection(x,receiver.energy) if receiver.profile.nonlinear else x
    for interval,(y0,y1) in enumerate(np.asarray(samples).reshape(-1,2)):
        inversion=detector._inv[(interval//(2*detector.P)+detector.inv_offset)%len(detector._inv)] if interval%(2*detector.P)==0 else 0
        new=np.full(16,np.inf); nh=np.zeros_like(histories);ne=np.zeros_like(residuals)
        previous=np.full(16,-1,int);chosen=np.zeros((16,2),complex)
        for state in np.flatnonzero(np.isfinite(metrics)):
            history=histories[state];eh=residuals[state]
            p0=independent.quantize(sum(history*receiver.h),2**-7)
            c0=independent.quantize(p0,2**receiver.tx.w)
            x0=points+c0-p0
            e0=y0-project(x0)
            cost0=abs(e0-coefficients@eh)**2
            classes=detector._layout(c0)[0]
            first=[];labels=[];parities=[]
            for label in range(8):
                for parity in range(2):
                    indices=classes[label][parity]
                    order=np.argsort(cost0[indices],kind='stable')[:2]
                    first.extend(indices[order]);labels.extend([label]*len(order));parities.extend([parity]*len(order))
            first=np.asarray(first);labels=np.asarray(labels);parities=np.asarray(parities)
            p1=np.array([independent.quantize(x0[i]*receiver.h[0]+history[0]*receiver.h[1]+history[1]*receiver.h[2],2**-7) for i in first])
            c1=np.array([independent.quantize(p,2**receiver.tx.w) for p in p1])
            linear1=points[None,:]+(c1-p1)[:,None]
            errors1=y1-project(linear1)
            predicted1=coefficients[0]*e0[first]+coefficients[1]*eh[0]+coefficients[2]*eh[1]
            costs1=abs(errors1-predicted1[:,None])**2
            # Batch identical subset layouts/parities. Preserve the original
            # candidate order and tie rule, but avoid a Python loop per branch.
            groups={}
            for row in range(len(first)):
                parity1=int(parities[row]) ^ (int(state)&1) ^ inversion ^ data.modulo_bit(c0,c1[row])
                key=(int(c1[row].real)%8,int(c1[row].imag)%8,parity1)
                groups.setdefault(key,[]).append(row)
            second=np.full((len(first),8),-1,int)
            for key,rows in groups.items():
                rows=np.asarray(rows)
                classes=detector._layout(c1[rows[0]])[0]
                for label1 in range(8):
                    indices=classes[label1][key[2]]
                    if len(indices):
                        local=costs1[np.ix_(rows,indices)]
                        second[rows,label1]=indices[np.argmin(local,axis=1)]
            costs=metrics[state]+cost0[first,None]+costs1[np.arange(len(first))[:,None],second]
            costs[second<0]=np.inf
            destinations=detector.next[state,labels[:,None],np.arange(8)[None,:]]
            for target in np.unique(destinations):
                candidates=np.flatnonzero(destinations.ravel()==target)
                selected=candidates[np.argmin(costs.ravel()[candidates])]
                row,label1=divmod(int(selected),8)
                cost=costs[row,label1]
                if cost < new[target]:
                    i0=first[row];i1=second[row,label1]
                    new[target]=cost;previous[target]=state;chosen[target]=(points[i0],points[i1])
                    nh[target]=(linear1[row,i1],x0[i0],history[0])
                    ne[target]=(errors1[row,i1],e0[i0],eh[0])
        if not np.any(np.isfinite(new)):raise ValueError('no surviving candidate path')
        records.append((previous,chosen));metrics,histories,residuals=new,nh,ne
    state=int(np.argmin(metrics));cost=float(metrics[state])
    final_state=state;final_history=histories[state].copy();final_residual=residuals[state].copy()
    pairs=[]
    for previous,chosen in reversed(records):
        pairs.append(chosen[state].copy());state=int(previous[state])
    if state != initial:raise ValueError('traceback failed to reach initial state')
    result=(np.asarray(list(reversed(pairs))).reshape(-1),cost)
    if return_state:return result+(final_history,final_residual,final_state)
    return result
