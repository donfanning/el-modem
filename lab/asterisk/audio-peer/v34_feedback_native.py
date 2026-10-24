"""Opt-in fused equalizer feedback; no fastmath or loop/decision changes."""
import math
import numpy as np
try:
    from numba import njit
except ImportError:
    njit=None


def _feedback(h,w,yr,d,mu,kp,ki,phase,freq,known_energy=-1.0):
    energy=window_energy(w) if known_energy<0. else known_energy
    error=d-yr
    rotation=complex(math.cos(phase),math.sin(phase))
    correction=error*rotation
    for j in range(len(h)):
        h[j]+=(mu*w[j].conjugate())*correction/(energy+1e-9)
    if abs(d)>0:
        cross=yr*d.conjugate()
        perr=math.atan2(cross.imag,cross.real)
        freq+=ki*perr
        phase+=kp*perr+freq
    return phase,freq


def _window_energy(w):
    energy=0.
    for j in range(len(w)):
        energy+=w[j].real*w[j].real+w[j].imag*w[j].imag
    return energy


def _centroid(h,span):
    energy=0.;moment=0.
    for j in range(len(h)):
        value=h[j].real*h[j].real+h[j].imag*h[j].imag
        energy+=value;moment+=j*value
    return moment/(energy+1e-12)-span


feedback=njit(cache=True)(_feedback) if njit else _feedback
window_energy=njit(cache=True)(_window_energy) if njit else _window_energy
centroid=njit(cache=True)(_centroid) if njit else _centroid


def warmup():
    h=np.zeros(24,np.complex128);w=np.ones(24,np.complex128)
    window_energy(w)
    feedback(h,w,1+1j,1+1j,.02,.03,.0001,0.,0.)
    centroid(h,12)
    return njit is not None
