"""A linear equaliser for the digital modem's receive path, trained on TRN1u (ITU-T V.92 clause 9.5.1).

The digital modem knows the TRN1u it receives (scrambled ones, +/-LU) and fits a finite impulse response ``w`` so that
``y[n] = sum_j w[j] r[n - j]`` reproduces the sent symbol ``D`` positions earlier, ``y[n] ~ s[n - D]``, for the
delay ``D`` that fits best. With ``y`` in place of ``r`` the Phase 4 frames of the analogue modem (SUVu, CPu, E) can be decided on a
dispersive line; the gain is one (the target is the sent level), so a later signal at another level keeps its level.

Data mode does not use it: there the analogue modem's precoder and prefilter are what make the line output equal the
constellation level. ``StreamFilter`` applies ``w`` to a stream in chunks, keeping the filter state.
"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def train(received, sent, *, taps: int = 48, max_delay: int = 40, skip: int = 64) -> tuple[np.ndarray, int, float]:
    """Least-squares fit of ``w`` and the delay D. ``sent`` is the +/-LU TRN1u in the coordinates of ``received``.
    Returns ``(w, D, residual_rms)``."""
    r = np.asarray(received, dtype=float)
    s = np.asarray(sent, dtype=float)
    if r.shape != s.shape or r.ndim != 1:
        raise ValueError("received and sent must be equal-length vectors")
    start = max(skip, taps + max_delay)
    if r.size - start < 4 * taps:
        raise ValueError("too few samples to train the equaliser")
    rows = np.arange(start, r.size)
    matrix = np.stack([r[rows - j] for j in range(taps)], axis=1)
    best = None
    for delay in range(max_delay + 1):
        target = s[rows - delay]
        w, *_ = np.linalg.lstsq(matrix, target, rcond=None)
        residual = float(np.sqrt(np.mean((matrix @ w - target) ** 2)))
        # Prefer the smallest delay: a larger one can always fit as well by shifting the taps.
        if best is None or residual < best[2] * 0.99 - 1e-6:
            best = (w, delay, residual)
    return best


def train_affine(received, sent, *, taps: int = 48, max_delay: int = 40,
                 skip: int = 64) -> tuple[np.ndarray, int, float, float]:
    """Fit an FIR and additive output bias to a known training sequence.

    Returns ``(weights, delay, residual_rms, bias)`` such that
    ``sum_j weights[j] * received[n-j] + bias`` approximates
    ``sent[n-delay]``. The bias is in the same units as ``sent`` and is
    deliberately separate from the FIR so callers can decide where to apply it.
    """
    r = np.asarray(received, dtype=float)
    s = np.asarray(sent, dtype=float)
    if r.shape != s.shape or r.ndim != 1:
        raise ValueError("received and sent must be equal-length vectors")
    start = max(skip, taps + max_delay)
    if r.size - start < 4 * taps:
        raise ValueError("too few samples to train the equaliser")
    rows = np.arange(start, r.size)
    matrix = np.stack([r[rows - j] for j in range(taps)], axis=1)
    affine = np.column_stack((matrix, np.ones(len(rows))))
    # Every delay shares the matrix: one solve with a target column per delay.
    targets = np.stack([s[rows - delay] for delay in range(max_delay + 1)], axis=1)
    coefficients, *_ = np.linalg.lstsq(affine, targets, rcond=None)
    residuals = np.sqrt(np.mean((affine @ coefficients - targets) ** 2, axis=0))
    best = None
    for delay in range(max_delay + 1):
        residual = float(residuals[delay])
        if best is None or residual < best[2] * 0.99 - 1e-6:
            best = (coefficients[:-1, delay].copy(), delay, residual, float(coefficients[-1, delay]))
    return best


class StreamFilter:
    """FIR filter over a stream arriving in chunks: ``y[n] = sum_j w[j] x[n - j]`` with zero history at the start."""

    def __init__(self, w):
        self.w = np.asarray(w, dtype=float)
        self._history = np.zeros(len(self.w) - 1)

    def apply(self, x) -> np.ndarray:
        x = np.asarray(x, dtype=float).reshape(-1)
        joined = np.concatenate((self._history, x))
        y = np.convolve(joined, self.w, mode="valid")
        self._history = joined[len(x):] if len(joined) - len(x) == len(self._history) else joined[-len(self._history):]
        return y


class AdaptiveStreamFilter(StreamFilter):
    """``StreamFilter`` that can take a normalised LMS step on the samples of its latest ``apply`` call.

    The analogue modem slaves its transmit clock to its downstream timing recovery, so the upstream's sampling phase
    wanders during a call and a FIR fitted once on TRN2u loses margin over tens of seconds (arm 10c: data sigma 26 -> 65
    in 36 s). ``adapt`` makes one block step of ``mu`` times the mean error gradient over the outputs selected by
    ``mask``, normalised by the input power per tap, so the step does not depend on the chunk length or the tap count.
    The caller's decision-domain output is ``scale * (y + bias)`` and ``error`` is the decision minus that output. The
    outputs ``apply`` returned were computed with the weights before the step."""

    def __init__(self, w, mu: float):
        super().__init__(w)
        if not np.isfinite(mu) or mu <= 0:
            raise ValueError("mu must be finite and positive")
        self.mu = float(mu)
        self.initial = self.w.copy()
        self.updates = 0
        self._windows = np.zeros((0, len(self.w)))

    def apply(self, x) -> np.ndarray:
        x = np.asarray(x, dtype=float).reshape(-1)
        joined = np.concatenate((self._history, x))
        self._windows = sliding_window_view(joined, len(self.w))[:, ::-1]    # row n: x[n], x[n-1], ...
        y = self._windows @ self.w
        self._history = joined[len(x):]
        return y

    def adapt(self, first: int, error, mask, scale: float) -> bool:
        """Step on outputs ``first:`` of the latest ``apply`` where ``mask`` is true; false if too few were usable."""
        windows = self._windows[first:first + len(error)][np.asarray(mask, dtype=bool)]
        error = np.asarray(error, dtype=float)[np.asarray(mask, dtype=bool)]
        if len(error) < 8:
            return False
        power = float(np.mean(windows * windows))
        if not np.isfinite(power) or power <= 0:
            return False
        self.w = self.w + self.mu * (windows.T @ error) / (len(error) * float(scale) * power)
        self.updates += 1
        return True
