"""Optional compiled version of the existing windowed-sinc interpolation.

No fast-math or approximation. Startup-gradient calls opt in; NumPy remains
available as the independent reference. Warm before processing live media.
"""
import math
import numpy as np
try:
    from numba import njit
except ImportError:
    njit=None

# Only integer tap phases are cached. Fractional sample coordinates remain
# unquantized. The original formula handles other supports and integer limits.
_COS = np.zeros((9,17))
_SIN = np.zeros((9,17))
for _half in (2,6,8):
    for _i,_m in enumerate(range(-_half,_half+1)):
        _COS[_half,_i] = math.cos(math.pi*_m/_half)
        _SIN[_half,_i] = math.sin(math.pi*_m/_half)


def _weight(m,fraction,half,direct,sine,cosine_window,sine_window):
    distance=m-fraction
    if direct:
        sinc=1. if distance==0. else math.sin(math.pi*distance)/(math.pi*distance)
        return sinc*(.5+.5*math.cos(math.pi*distance/half))
    # Center the angle at the nearest integer. Using sin(pi*f) near f=1
    # otherwise loses relative accuracy for the main interpolation tap.
    n=m-int(fraction>.5)
    sinc=(sine if n%2 else -sine)/(math.pi*distance)
    i=n+half
    window=_COS[half,i]*cosine_window+_SIN[half,i]*sine_window
    return sinc*(.5+.5*window)


if njit:
    _weight=njit(cache=True,inline='always',fastmath=False)(_weight)


def _interp(z, positions, half):
    result=np.empty(len(positions),np.complex128)
    for j in range(len(positions)):
        whole=int(math.floor(positions[j]))
        fraction=positions[j]-whole
        direct=half not in (2,6,8) or fraction<1e-7 or 1-fraction<1e-7
        sine=cosine_window=sine_window=0.
        if not direct:
            phase=fraction-int(fraction>.5)
            sine=math.sin(math.pi*phase)
            cosine_window=math.cos(math.pi*phase/half)
            sine_window=math.sin(math.pi*phase/half)
        value=0j
        for m in range(-half+1,half+1):
            index=max(0,min(len(z)-1,whole+m))
            weight=_weight(m,fraction,half,direct,sine,cosine_window,sine_window)
            value+=z[index]*weight
        result[j]=value
    return result


kernel=njit(cache=True)(_interp) if njit else _interp


def _windows(source, positions, rotation, shift):
    result=np.empty(positions.shape,np.complex128)
    for row in range(len(positions)):
        for column in range(positions.shape[1]):
            position=positions[row,column]+shift[row]
            whole=int(math.floor(position))
            fraction=position-whole
            direct=fraction<1e-7 or 1-fraction<1e-7
            sine=cosine_window=sine_window=0.
            if not direct:
                phase=fraction-int(fraction>.5)
                sine=math.sin(math.pi*phase)
                cosine_window=math.cos(math.pi*phase/6)
                sine_window=math.sin(math.pi*phase/6)
            value=0j
            for m in range(-5,7):
                index=whole+m
                if index<0 or index>=source.shape[1]:
                    raise ValueError('timing intervention exceeds retained interpolation support')
                weight=_weight(m,fraction,6,direct,sine,cosine_window,sine_window)
                value+=source[row,index]*weight
            result[row,column]=value*rotation[row]
    return result


# A guarded clock fit evaluates hundreds of rows synchronously. Let the
# independent media sender run while this arithmetic executes.
window_kernel=njit(cache=True, nogil=True)(_windows) if njit else _windows


def windows(source,positions,rotation,shift):
    return window_kernel(source,positions,rotation,shift)


def interpolate(z,positions,half=6):
    return kernel(z,positions,half)


def warmup():
    if njit is not None:
        kernel(np.zeros(32,np.complex128),np.array([12.25,13.]),6)
        window_kernel(np.zeros((1,32),np.complex128),np.array([[12.25]]),
                      np.ones(1,np.complex128),np.zeros(1))
    return njit is not None
