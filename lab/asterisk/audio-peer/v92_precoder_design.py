"""Digital-modem design of the V.92 upstream prefilter and precoder (clause 6.4.2 / 8.8.3).

The analogue modem sends x(n) = u(n) + sum p1(k) x(n-k) through a prefilter z2, and the
codec's A/D output must equal u(n). With the end-to-end symbol-spaced channel response
``h`` (analogue-modem DAC to codeword domain), the composite ``h * z2`` is made to follow a
*minimum-phase* monic target B(z) delayed by ``d`` symbols, B being the spectral factor of
the channel's power response (the usual MMSE decision-feedback structure moved into the
transmitter). The feedback section then cancels B exactly: ``p1(k) = -B[k]`` makes the
codeword-domain output equal u(n-d) in the noise-free case.

B must be minimum phase: the precoder divides by B, so a zero outside the unit circle makes
x(n) diverge. Only the p1/z2 sections are produced (INFO1a filter-section code 0 allows
exactly that). The digital modem must account for the pure delay ``d`` when aligning data
frames.
"""

from __future__ import annotations

import numpy as np

FFT_SIZE = 8192
NOISE_FLOOR = 1.0e-3             # default regularisation, relative to the peak of |H|^2 (-30 dB)
STABILITY_MARGIN = 0.98          # largest allowed |zero| of B


def _levinson(r, order: int) -> np.ndarray:
    """Monic prediction-error polynomial from autocorrelation ``r`` (minimum phase)."""
    a = np.zeros(order + 1)
    a[0] = 1.0
    error = r[0]
    for i in range(1, order + 1):
        k = -(r[i] + a[1:i] @ r[i - 1:0:-1]) / error
        a[1:i + 1] = a[1:i + 1] + k * np.concatenate([a[1:i][::-1], [1.0]])
        error *= 1.0 - k * k
    return a


def minimum_phase_target(h, postcursors: int, noise_floor: float = NOISE_FLOOR) -> np.ndarray:
    """Monic FIR B of ``postcursors`` taps after the main, minimum phase by construction.

    B is the prediction-error filter of the *inverse* power spectrum 1/|H|^2 (autocorrelation
    method), so |B|^2 approximates |H|^2 up to gain and every zero of B lies inside the unit
    circle. A truncated cepstral factor offers no such guarantee. ``noise_floor`` keeps the
    zeros away from the circle where the line has a spectral null (real telephone channels
    do, at DC and near 4 kHz); it trades residual ISI near the nulls for a stable precoder."""
    spectrum = np.abs(np.fft.fft(np.asarray(h, dtype=float), FFT_SIZE)) ** 2
    spectrum = spectrum + noise_floor * spectrum.max()
    autocorrelation = np.fft.ifft(1.0 / spectrum).real
    return _levinson(autocorrelation[:postcursors + 1], postcursors)


def is_stable(target) -> bool:
    """True when every zero of the monic target polynomial is inside the unit circle."""
    target = np.asarray(target, dtype=float)
    return target.size < 2 or bool(np.all(np.abs(np.roots(target)) < STABILITY_MARGIN))


def feedback_power_gain(p1, length: int = 4096) -> float:
    """Sum of squares of the impulse response of 1/(1 - P1): how much the feedback section
    amplifies white input power on the way to the transmitted x(n)."""
    p1 = np.asarray(p1, dtype=float)
    response = np.zeros(length)
    response[0] = 1.0
    for n in range(1, length):
        taps = min(n, p1.size)
        response[n] = float(p1[:taps] @ response[n - 1::-1][:taps]) if taps else 0.0
    return float(response @ response)


def transmit_gain(z2, p1, length: int = 8192) -> float:
    """rms gain from white u(n) to the prefilter output v(n): the impulse response of
    z2(z) / (1 - P1(z)). Multiply by rms(u) (and G) for the transmitted level."""
    p1 = np.asarray(p1, dtype=float)
    z2 = np.asarray(z2, dtype=float)
    feedback = np.zeros(length)
    feedback[0] = 1.0
    for n in range(1, length):
        taps = min(n, p1.size)
        feedback[n] = float(p1[:taps] @ feedback[n - 1::-1][:taps]) if taps else 0.0
    response = np.convolve(feedback, z2)[:length]
    return float(np.sqrt(response @ response))


def design(h, *, prefilter_taps: int, postcursors: int, delays=None,
           noise_floor: float = NOISE_FLOOR) -> dict:
    """Return ``{"z2", "p1", "delay", "residual", "stable"}`` for the best delay.

    ``residual`` is the energy of ``h*z2`` (main cursor 1) outside the main cursor and the
    ``postcursors`` taps the feedback section cancels: precursors and the tail. When the truncated B has a zero at or
    outside the unit circle ``stable`` is False; callers must not use such a design."""
    h = np.asarray(h, dtype=float)
    if h.ndim != 1 or not h.size or not np.any(h):
        raise ValueError("channel response must be a non-zero vector")
    if prefilter_taps < 1 or postcursors < 0:
        raise ValueError("need at least one prefilter tap and a non-negative postcursor count")
    target = minimum_phase_target(h, postcursors, noise_floor)
    length = h.size + prefilter_taps - 1
    conv = np.zeros((length, prefilter_taps))
    for k in range(prefilter_taps):
        conv[k:k + h.size, k] = h
    energy = float(target @ target)
    best = None
    candidates = range(max(1, length - target.size + 1)) if delays is None else delays
    solver = np.linalg.pinv(conv)                 # one factorisation serves every delay
    for d in candidates:
        wanted = np.zeros(length)
        wanted[d:d + target.size] = target[:max(0, length - d)]
        f = solver @ wanted
        composite = conv @ f
        main = composite[d]
        if abs(main) < 1e-12:
            continue
        f, composite = f / main, composite / main
        achieved = composite[d:d + postcursors + 1]
        achieved = np.concatenate([achieved, np.zeros(postcursors + 1 - achieved.size)])
        # Cancel the postcursors the filter actually produced (not just the target's), when
        # that polynomial is still minimum phase; the target's taps are the safe fallback.
        use_achieved = is_stable(achieved)
        feedback = achieved if use_achieved else target
        outside = np.ones(length, dtype=bool)
        outside[d:d + postcursors + 1] = False
        stray = composite[outside]
        if not use_achieved:                     # target postcursors cancel, the rest stays
            stray = np.concatenate([stray, (achieved - target)[1:]])
        residual = float(stray @ stray)
        if best is None or residual < best["residual"] * (1 - 1e-9):
            best = {"z2": tuple(float(v) for v in f),
                    "p1": tuple(float(-v) for v in feedback[1:]),
                    "delay": int(d), "residual": residual,
                    "stable": is_stable(feedback), "energy": float(feedback @ feedback)}
    if best is None:
        raise ValueError("no usable delay for this channel")
    best["isi_ratio"] = float(np.sqrt(best["residual"] * feedback_power_gain(best["p1"]))
                              ) if best["stable"] else float("inf")
    return best


def apply_channel(samples, h):
    """Convolve and truncate to the input length (causal channel)."""
    return np.convolve(np.asarray(samples, dtype=float), np.asarray(h, dtype=float))[:len(samples)]
