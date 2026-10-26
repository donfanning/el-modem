"""Optional native execution of the ordinary shared-feedback trellis detector.

This preserves the existing algorithm; it is not noise-predictive data mode.
"""
import math
import numpy as np
try:
    from numba import njit
except ImportError:
    njit=None


def _round(z,step):
    re=math.ceil(abs(z.real/step)-.5)
    im=math.ceil(abs(z.imag/step)-.5)
    return complex(math.copysign(re,z.real)*step if re else 0.,
                   math.copysign(im,z.imag)*step if im else 0.)


def _inverse(y,energy):
    r=abs(y)
    if not r:return 0j
    a=.3125/(6*energy);b=.3125**2/(120*energy**2)
    x=min(r,(r/a)**(1/3),(r/b)**.2)
    for _ in range(10):
        x-=(x+a*x**3+b*x**5-r)/(1+3*a*x*x+5*b*x**4)
    return y*(x/r)


def _traceback(previous,last_interval,steps,state):
    before=after=state
    for j in range(steps):
        after=state
        before=previous[(last_interval-j)%len(previous),after]
        state=before
    return before,after


def _block(waves,pts,h,hist,nonlinear,energy,step,labels,par,metric,cprev,cs0,cs1,
           inversions,grid,min_re,min_im,available):
    """Advance consecutive clean intervals within one nopython call."""
    count=len(waves)
    ys=np.empty((count,2),np.complex128);cs=np.empty((count,2),np.complex128)
    deltas=np.empty((count,2),np.complex128);indices=np.empty((count,2,8,2),np.int64)
    prevs=np.empty((count,16),np.int64);s0s=np.empty((count,16),np.int64)
    s1s=np.empty((count,16),np.int64);p0s=np.empty((count,16),np.int64)
    uss=np.empty((count,16),np.int64);normalizations=np.empty(count,np.float64)
    metrics=np.empty((count,16),np.float64)
    current_hist=hist.copy();current_metric=metric.copy()
    for j in range(count):
        result=_pair(waves[j],pts,h,current_hist,nonlinear,energy,step,labels,par,
            current_metric,cprev,cs0,cs1,int(inversions[j]),grid,min_re,min_im,
            available,False)
        y,c,delta,current_hist,best,ix,cost,parity,us,new,prev,s0,s1,p0,norm=result
        ys[j]=y;cs[j]=c;deltas[j]=delta;indices[j]=ix
        prevs[j]=prev;s0s[j]=s0;s1s[j]=s1;p0s[j]=p0;uss[j]=us
        normalizations[j]=norm;current_metric=new;metrics[j]=new
    return ys,cs,deltas,current_hist,current_metric,indices,prevs,s0s,s1s,p0s,uss,normalizations,metrics


def _pair(waves,pts,h,hist,nonlinear,energy,step,labels,par,
          metric,cprev,cs0,cs1,inversion,grid,min_re,min_im,available,erasure):
    ys=np.empty(2,np.complex128);c=np.empty(2,np.complex128)
    delta=np.empty(2,np.complex128);history=hist.copy()
    best=np.full((2,8,2),1e18);indices=np.full((2,8,2),-1,np.int64)
    for t in range(2):
        x=_inverse(waves[t],energy) if nonlinear else waves[t]
        p=_round(history[0]*h[0]+history[1]*h[1]+history[2]*h[2],2**-7)
        c[t]=_round(p,step);delta[t]=c[t]-p;ys[t]=x+p-c[t]
        nearest=0;nearest_cost=np.inf
        cr,ci=int(c[t].real)%8,int(c[t].imag)%8
        # With zero displacement and a linear channel, both existing distance
        # expressions have exactly the same operands. Reuse the calculation;
        # keep the waveform-domain cost for every other profile.
        same_distance = not nonlinear and delta[t] == 0j and ys[t] == waves[t]
        certified=False
        if (same_distance and len(pts)>=256
                and min_re-8 <= ys[t].real <= min_re+2*(grid.shape[0]-1)+8
                and min_im-8 <= ys[t].imag <= min_im+2*(grid.shape[1]-1)+8):
            re0=int(math.floor((ys[t].real-min_re)/2))
            im0=int(math.floor((ys[t].imag-min_im)/2))
            for re in range(max(0,re0-4),min(grid.shape[0],re0+6)):
                for im in range(max(0,im0-4),min(grid.shape[1],im0+6)):
                    i=grid[re,im]
                    if i<0:continue
                    e=pts[i]-ys[t];distance=e.real*e.real+e.imag*e.imag
                    if distance>=64:continue
                    if distance<nearest_cost or distance==nearest_cost and i<nearest:
                        nearest,nearest_cost=i,distance
                    label,pa=labels[cr,ci,i],par[i]
                    if (distance<best[t,label,pa] or
                            distance==best[t,label,pa] and i<indices[t,label,pa]):
                        best[t,label,pa]=distance;indices[t,label,pa]=i
            # Every existing class has a point strictly inside radius 8.
            # Empty subset/parity slots retain their original sentinel cost.
            # All unvisited points are outside and cannot win.
            certified=True
            for label in range(8):
                for pa in range(2):
                    if available[cr,ci,label,pa] and best[t,label,pa]>=64:
                        certified=False
            if not certified:
                best[t,:,:]=1e18;indices[t,:,:]=-1
                nearest=0;nearest_cost=np.inf
        for i in range(0 if certified else len(pts)):
            e=pts[i]-ys[t];distance=e.real*e.real+e.imag*e.imag
            if distance<nearest_cost:nearest,nearest_cost=i,distance
            if same_distance:
                cost=distance
            else:
                raw=pts[i]+delta[t]
                if nonlinear:
                    z=.3125*(raw.real*raw.real+raw.imag*raw.imag)/energy
                    raw=raw*(1+z/6+z*z/120)
                e=raw-waves[t];cost=e.real*e.real+e.imag*e.imag
            label,parity=labels[cr,ci,i],par[i]
            if cost<best[t,label,parity]:
                best[t,label,parity]=cost;indices[t,label,parity]=i
        history[2]=history[1];history[1]=history[0];history[0]=pts[nearest]+delta[t]
    modulo=(int(round(c[0].real/2))+int(round(c[0].imag/2))+
            int(round(c[1].real/2))+int(round(c[1].imag/2)))%2
    us=np.empty(16,np.int64)
    cost=np.empty((2,8,8));parity=np.empty((2,8,8),np.int64)
    for u in range(2):
        for s0 in range(8):
            for s1 in range(8):
                a=best[0,s0,0]+best[1,s1,u]
                b=best[0,s0,1]+best[1,s1,1^u]
                parity[u,s0,s1]=int(b<a)
                cost[u,s0,s1]=0.0 if erasure and (a<1e18 or b<1e18) else min(a,b)
    for state in range(16):us[state]=(state&1)^inversion^modulo
    new=np.empty(16);prev=np.empty(16,np.int64)
    s0=np.empty(16,np.int64);s1=np.empty(16,np.int64);p0=np.empty(16,np.int64)
    for target in range(16):
        best_cost=np.inf;kbest=0
        for k in range(cprev.shape[1]):
            pst,a,b=cprev[target,k],cs0[target,k],cs1[target,k]
            candidate=metric[pst]+cost[us[pst],a,b]
            if candidate<best_cost:best_cost,kbest=candidate,k
        prev[target]=cprev[target,kbest];s0[target]=cs0[target,kbest];s1[target]=cs1[target,kbest]
        p0[target]=parity[us[prev[target]],s0[target],s1[target]];new[target]=best_cost
    normalization=np.min(new);new-=normalization
    return ys,c,delta,history,best,indices,cost,parity,us,new,prev,s0,s1,p0,normalization


if njit is not None:
    _round=njit(cache=True,inline='always',fastmath=False)(_round)
    _inverse=njit(cache=True,inline='always',fastmath=False)(_inverse)
    _pair=njit(cache=True,nogil=True,fastmath=False)(_pair)
    _block=njit(cache=True,nogil=True,fastmath=False)(_block)
    _traceback=njit(cache=True,nogil=True,fastmath=False)(_traceback)


def layouts(points):
    labels=np.empty((8,8,len(points)),np.int64)
    for re in range(8):
        for im in range(8):
            a=(points.real.astype(np.int64)+re-1)//2
            b=(points.imag.astype(np.int64)+im-1)//2
            labels[re,im]=((((a+1)//2^(b+1)//2)&1)<<2)|((a&1)<<1)|((a+b)&1)
    return labels


def warmup():
    if njit is None:raise RuntimeError('native trellis requires the optional Numba lab image')
    from v34_viterbi import TrellisDetector
    d=TrellisDetector(16,8,15,backend='python')
    labels=layouts(d.pts)
    advance(d,0j,0j,0,labels)
    advance_block(d,np.zeros((2,2),np.complex128),np.zeros(2,np.int64),labels)
    _traceback(np.zeros((25,16),np.int64),24,25,0)


def traceback(previous,last_interval,steps,state):
    if njit is None:raise RuntimeError('native traceback requires Numba')
    return _traceback(previous,last_interval,steps,state)


def advance(detector,y0,y1,inversion,labels,*,erasure=False):
    if njit is None:raise RuntimeError('native trellis requires Numba')
    if detector.nonlinear and (not math.isfinite(detector.energy) or detector.energy<=0):
        raise ValueError('nonlinear energy must be finite and positive')
    if not np.isfinite(y0) or not np.isfinite(y1):raise ValueError('finite received symbols required')
    if not hasattr(detector,'_native_point_grid'):
        re,im=detector.pts.real,detector.pts.imag
        min_re,min_im=float(re.min()),float(im.min())
        grid=np.full((int((re.max()-min_re)/2)+1,int((im.max()-min_im)/2)+1),-1,np.int64)
        grid[((re-min_re)/2).astype(np.int64),((im-min_im)/2).astype(np.int64)]=np.arange(len(re))
        available=np.zeros((8,8,8,2),np.bool_)
        for re_offset in range(8):
            for im_offset in range(8):
                available[re_offset,im_offset,labels[re_offset,im_offset],detector.par]=True
        detector._native_point_grid=(grid,min_re,min_im,available)
    grid,min_re,min_im,available=detector._native_point_grid
    return _pair(np.asarray([y0,y1],np.complex128),detector.pts,
        np.asarray(detector.h,np.complex128),np.asarray(detector.x_hist,np.complex128),
        bool(detector.nonlinear),float(detector.energy),float(2**detector.w),labels,
        detector.par,detector.metric,detector._cand_prev,detector._cand_s0,
        detector._cand_s1,int(inversion),grid,min_re,min_im,available,bool(erasure))


def advance_block(detector,waves,inversions,labels):
    """Advance matching clean symbol pairs while preserving trellis order."""
    if njit is None:raise RuntimeError('native trellis requires Numba')
    if detector.nonlinear and (not math.isfinite(detector.energy) or detector.energy<=0):
        raise ValueError('nonlinear energy must be finite and positive')
    waves=np.asarray(waves,dtype=np.complex128)
    inversions=np.asarray(inversions,dtype=np.int64)
    if waves.ndim!=2 or waves.shape[1]!=2 or len(waves)!=len(inversions) or not len(waves):
        raise ValueError('trellis block must contain matching non-empty symbol pairs')
    if not np.isfinite(waves).all():raise ValueError('finite received symbols required')
    if not hasattr(detector,'_native_point_grid'):
        re,im=detector.pts.real,detector.pts.imag
        min_re,min_im=float(re.min()),float(im.min())
        grid=np.full((int((re.max()-min_re)/2)+1,int((im.max()-min_im)/2)+1),-1,np.int64)
        grid[((re-min_re)/2).astype(np.int64),((im-min_im)/2).astype(np.int64)]=np.arange(len(re))
        available=np.zeros((8,8,8,2),np.bool_)
        for ro in range(8):
            for io in range(8):available[ro,io,labels[ro,io],detector.par]=True
        detector._native_point_grid=(grid,min_re,min_im,available)
    grid,min_re,min_im,available=detector._native_point_grid
    return _block(waves,detector.pts,np.asarray(detector.h,np.complex128),
        np.asarray(detector.x_hist,np.complex128),bool(detector.nonlinear),float(detector.energy),
        float(2**detector.w),labels,detector.par,detector.metric,detector._cand_prev,
        detector._cand_s0,detector._cand_s1,inversions,grid,min_re,min_im,available)
