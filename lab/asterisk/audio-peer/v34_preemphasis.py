"""Stateful passband pre-emphasis; own coefficients from ITU V.34 5.4.

Index 0 is exactly flat. Nonzero FIRs have 64 samples (8 ms) group delay.
A filter instance lasts for a continuous QAM burst, including training/data.
"""
import numpy as np
from v34_preemphasis_coefficients import COEFFICIENTS
from v34_tx import CARRIERS


class PreEmphasis:
    def __init__(self, baud, carrier, index):
        nominal = round(baud)
        if index not in range(11) or nominal not in CARRIERS:
            raise ValueError('unsupported V.34 pre-emphasis configuration')
        carriers = CARRIERS[nominal]
        high = min(range(2), key=lambda k: abs(carriers[k] - carrier))
        if abs(carriers[high] - carrier) > .01:
            raise ValueError('unsupported V.34 pre-emphasis carrier')
        self.taps = np.array(COEFFICIENTS[(nominal, high, index)])
        self.history = np.zeros(len(self.taps)-1)
        self.delay_samples = (len(self.taps)-1)//2

    def process(self, x):
        x = np.asarray(x, dtype=float)
        if not len(x):
            return x.copy()
        if len(self.taps) == 1:
            return x.copy()
        v = np.concatenate((self.history, x))
        out = np.convolve(v, self.taps, mode='valid')
        self.history = v[-len(self.history):].copy()
        return out
