"""Opt-in fused V.34 data slicer and equalizer feedback kernel."""
import math

try:
    from numba import njit
except ImportError:
    njit=None


def _round_component(value,step):
    q=value/step
    n=math.ceil(abs(q)-.5)
    return math.copysign(n,q)*step if n else 0.


def _inverse_projection(y,energy):
    r=abs(y)
    if r==0.:return 0j
    a=.3125/(6.*energy);b=(.3125*.3125)/(120.*energy*energy)
    x=min(r,(r/a)**(1./3.),(r/b)**.2)
    for _ in range(10):
        x-=(x+a*x*x*x+b*x*x*x*x*x-r)/(1.+3.*a*x*x+5.*b*x*x*x*x)
    return y*(x/r)


def _projection(x,energy):
    z=.3125*(x.real*x.real+x.imag*x.imag)/energy
    return x*(1.+z/6.+z*z/120.)


if njit:
    _round_component=njit(inline='always')(_round_component)
    _inverse_projection=njit(inline='always')(_inverse_projection)
    _projection=njit(inline='always')(_projection)


def _step(eq_h,w,yr,phase,freq,hist,precoder,points,point_grid,grid_x0,grid_y0,
          nonlinear,energy,word_step,mu,kp,ki,known_energy=-1.):
    x=_inverse_projection(yr,energy) if nonlinear else yr
    if precoder[0]!=0j or precoder[1]!=0j or precoder[2]!=0j:
        p=hist[0]*precoder[0]+hist[1]*precoder[1]+hist[2]*precoder[2]
        p=complex(_round_component(p.real,2.**-7),_round_component(p.imag,2.**-7))
        c=complex(_round_component(p.real,word_step),_round_component(p.imag,word_step))
    else:p=c=0j
    v=x+p-c
    # Preserve DataRx.nearest_point's interior odd-grid fast path exactly.
    point_found=False
    point=0j
    if math.isfinite(v.real) and math.isfinite(v.imag):
        pr=2.*math.floor(v.real/2.)+1.
        pi=2.*math.floor(v.imag/2.)+1.
        if abs(v.real-pr)<1.-1e-10 and abs(v.imag-pi)<1.-1e-10:
            ix=int((pr-grid_x0)/2.);iy=int((pi-grid_y0)/2.)
            if 0<=ix<point_grid.shape[0] and 0<=iy<point_grid.shape[1] and point_grid[ix,iy]:
                point=complex(pr,pi);point_found=True
    if not point_found:
        best=math.inf;point=points[0]
        for i in range(len(points)):
            dx=points[i].real-v.real;dy=points[i].imag-v.imag
            distance=math.hypot(dx,dy)
            if distance<best:best=distance;point=points[i]
    target=point+c-p
    hist[2]=hist[1];hist[1]=hist[0];hist[0]=target
    d=_projection(target,energy) if nonlinear else target
    if known_energy<0.:
        power=0.
        for j in range(len(w)):power+=w[j].real*w[j].real+w[j].imag*w[j].imag
    else:power=known_energy
    error=d-yr;rotation=complex(math.cos(phase),math.sin(phase));correction=error*rotation
    for j in range(len(eq_h)):
        eq_h[j]+=(mu*w[j].conjugate())*correction/(power+1e-9)
    if abs(d)>0.:
        cross=yr*d.conjugate();perr=math.atan2(cross.imag,cross.real)
        freq+=ki*perr;phase+=kp*perr+freq
    return d,phase,freq


step=njit(cache=True,nogil=True)(_step) if njit else _step


def warmup(points):
    if njit is None:raise RuntimeError('native decision feedback requires Numba')
    import numpy as np
    points=np.asarray(points,dtype=np.complex128)
    xs=points.real.astype(np.int64);ys=points.imag.astype(np.int64);x0=int(xs.min());y0=int(ys.min())
    grid=np.zeros(((int(xs.max())-x0)//2+1,(int(ys.max())-y0)//2+1),dtype=np.bool_)
    grid[((xs-x0)//2),((ys-y0)//2)]=True
    hist=np.zeros(3,np.complex128);h=np.zeros(48,np.complex128)
    # Equalizer trajectory-window slices may be strided views. Compile both
    # layouts before media starts so the first data symbol never JIT-compiles.
    for w in (np.ones(48,np.complex128),np.ones(96,np.complex128)[::2]):
        step(h,w,1+0j,0.,0.,hist,np.zeros(3,np.complex128),points,grid,x0,y0,
             False,1.,1.,.2,.01,5e-6,-1.)
