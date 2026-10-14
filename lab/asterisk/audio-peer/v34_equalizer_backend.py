"""Explicit experimental arithmetic backend. Missing Rust never falls back."""
import os


def selected():
    name=os.environ.get('V34_EQUALIZER_BACKEND','legacy')
    if name not in ('legacy','rust'):
        raise ValueError('V34_EQUALIZER_BACKEND must be legacy or rust')
    if name=='legacy':return None
    import el_modem_dsp
    if not getattr(el_modem_dsp,'EXPERIMENTAL',False):
        raise ImportError('unrecognized el_modem_dsp arithmetic extension')
    return el_modem_dsp


def warmup():
    backend=selected()
    if backend is None:return 'legacy'
    import numpy as np
    taps=np.zeros(48,np.complex128);taps[24]=1.
    window=backend.interpolate(np.ones(128,np.complex128),np.arange(48,dtype=float)+20.25)
    backend.feedback(taps,window,1+0j,1+0j,.07,.01,5e-6,0.,0.)
    backend.centroid(taps,24)
    return 'rust'
