"""Digital-modem estimate of the upstream channel from TRN1u (clause 8.5.7 / 9.5.1).

TRN1u is +/-LU from the analogue GPA scrambler fed ones, sent through the channel with the
precoder and prefilter bypassed (clause 8.5.5). The digital modem knows the pattern, so a
least-squares fit of the codeword-domain samples against delayed copies of it gives the
symbol-spaced response that ``v92_precoder_design`` needs. G.711 quantisation of the
received samples appears as measurement noise; the fit residual is returned so callers can
size constellations against it.
"""

from __future__ import annotations

import numpy as np


def estimate(received, sent, taps: int, *, skip: int = 0) -> dict:
    """Fit ``received[n] = sum_k h[k] * sent[n-k]`` for ``k = 0..taps-1``.

    ``sent`` is the known +/-LU sequence (``v92_phase3.trn1u`` scaled by LU) aligned to
    ``received``; the first ``skip`` samples are ignored (start-up before the sequence
    reached the digital modem). Returns ``{"h", "residual_rms", "signal_rms"}``."""
    received = np.asarray(received, dtype=float)
    sent = np.asarray(sent, dtype=float)
    if received.shape != sent.shape or received.ndim != 1:
        raise ValueError("received and sent must be equal-length vectors")
    if taps < 1 or received.size - max(skip, taps) < 2 * taps:
        raise ValueError("too few samples for the requested number of taps")
    start = max(skip, taps - 1)
    rows = np.arange(start, received.size)
    matrix = np.stack([sent[rows - k] for k in range(taps)], axis=1)
    target = received[rows]
    h, *_ = np.linalg.lstsq(matrix, target, rcond=None)
    residual = target - matrix @ h
    return {"h": h, "residual_rms": float(np.sqrt(np.mean(residual ** 2))),
            "signal_rms": float(np.sqrt(np.mean(target ** 2)))}
