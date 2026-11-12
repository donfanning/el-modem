"""Shared RTP sample/media clock mapping for V.90 engines."""

from __future__ import annotations

import math
import time
from collections import deque

import numpy as np

FS = 8000


class RtpMediaClock:
    """Map ordered RX sample positions to the local TX sample timeline.

    RTP's remote timestamp origin is unrelated to our sender's origin. Each
    admitted packet supplies its playout deadline; bounded least squares over
    those anchors estimates the remote 8 kHz clock's offset and rate. The
    caller supplies ``origin`` and ``media_at`` from the exact V.8-to-engine
    selection boundary, so this does not assume the clocks share an origin.
    """

    def __init__(self, *, origin: float, media_at: int, max_anchors: int = 64,
                 max_rate_error_ppm: float = 2000.0,
                 low_delay_percentile: float = 15.0,
                 min_fit_span_samples: int = 6400,
                 offset_step_samples: int = 80):
        if (not math.isfinite(origin) or media_at < 0 or max_anchors < 2
                or not math.isfinite(float(max_rate_error_ppm))
                or float(max_rate_error_ppm) <= 0
                or isinstance(min_fit_span_samples, bool)
                or not isinstance(min_fit_span_samples, int)
                or min_fit_span_samples < 1
                or isinstance(offset_step_samples, bool)
                or not isinstance(offset_step_samples, int)
                or not 1 <= offset_step_samples <= FS):
            raise ValueError("invalid RTP media clock origin or anchor limit")
        if (not math.isfinite(float(low_delay_percentile))
                or not 5.0 <= float(low_delay_percentile) <= 50.0):
            raise ValueError("RTP low-delay percentile must be in 5..50")
        self.origin = float(origin)
        self.media_at = int(media_at)
        self.max_rate_error = float(max_rate_error_ppm) * 1e-6
        self.low_delay_percentile = float(low_delay_percentile)
        self.min_fit_span_samples = min_fit_span_samples
        self.offset_step_samples = offset_step_samples
        self.anchors: deque[tuple[float, float]] = deque(maxlen=max_anchors)
        self._offset_history: deque[float] = deque(maxlen=16)
        self.slope = 1.0
        self.intercept = float(media_at)
        self.last_fit: dict | None = None

    def observe(self, rx_at: int, context: dict) -> None:
        deadline = context.get("packet_playout_deadline_monotonic")
        if deadline is None or not math.isfinite(float(deadline)):
            raise ValueError("RTP packet lacks a finite playout deadline for V.90 timing")
        media_at = self.media_at + (float(deadline) - self.origin) * FS
        anchor = (float(rx_at), media_at)
        offset = media_at - anchor[0]
        previous_offsets = list(self._offset_history)
        self._offset_history.append(offset)
        if not self.anchors:
            self.anchors.append(anchor)
            self.intercept = media_at - rx_at
            return

        # Engine selection can replay two RTP pre-roll packets whose media
        # deadlines belong to the preceding V.8 chain epoch. The first packet
        # after the handoff may therefore step by >=10 ms even though the RTP
        # sample clock is unchanged. Detect a persistent offset step as soon
        # as two old plus three new offsets are available; waiting for the
        # longer steady-state window below lets stale anchors make the playout
        # delay change look like an impossible sample rate.
        if len(self._offset_history) >= 5:
            recent = list(self._offset_history)[-5:]
            before = float(np.median(recent[:2]))
            after = float(np.median(recent[-3:]))
            if abs(after - before) >= self.offset_step_samples:
                self.anchors.clear()
                self.anchors.append(anchor)
                self._offset_history.clear()
                self._offset_history.append(offset)
                self.intercept = media_at - self.slope * rx_at
                return

        # A persistent change in packet arrival/playout delay moves the
        # offset between RTP samples and local media time without changing
        # either clock rate. Do not fit across that discontinuity: three
        # consecutive samples must establish a persistent offset step, while
        # isolated late packets remain ordinary jitter outliers.
        if len(self._offset_history) >= 11:
            history = list(self._offset_history)
            before = float(np.median(history[-11:-3]))
            after = float(np.median(history[-3:]))
            if abs(after - before) >= self.offset_step_samples:
                # The three-point detector may include old-epoch samples;
                # seed the new epoch with only the current anchor.
                self.anchors.clear()
                self.anchors.append(anchor)
                self._offset_history.clear()
                self._offset_history.append(offset)
                self.intercept = media_at - self.slope * rx_at
                return

        # Keep isolated early/late arrivals out of the regression. If the
        # offset change persists, the three-point step detector above starts
        # a new epoch; until then it is packet jitter, not clock evidence.
        if previous_offsets and abs(offset - float(np.median(
                previous_offsets[-8:]))) >= 80.0:
            return
        self.anchors.append(anchor)

        x = np.fromiter((row[0] for row in self.anchors), dtype=float)
        y = np.fromiter((row[1] for row in self.anchors), dtype=float)
        offsets = y - x
        # Arrival/playout deadlines can move abruptly when RTP packets arrive
        # in a burst and the jitter buffer changes its delay. Fitting all
        # anchors treats that delay step as a sample-clock-rate change. Use
        # only the low-delay envelope for the rate estimate; those samples
        # retain the sender clock while late packets contribute as positive
        # delay outliers. Wait for a sufficiently long span and at least six
        # low-delay anchors before trusting a slope.
        low_delay_limit = float(np.percentile(offsets, self.low_delay_percentile))
        low_delay = offsets <= low_delay_limit
        if (x.max() - x.min() < self.min_fit_span_samples
                or int(np.count_nonzero(low_delay)) < 6):
            self.intercept = float(np.median(y - self.slope * x))
            return
        fit_x = x[low_delay]
        fit_y = y[low_delay]
        x_mean = float(fit_x.mean())
        y_mean = float(fit_y.mean())
        centered = fit_x - x_mean
        denom = float(centered @ centered)
        if denom <= 0:
            raise ValueError("RTP timing anchors do not advance")
        slope = float(centered @ (fit_y - y_mean)) / denom
        sample_span = float(x.max() - x.min())
        self.last_fit = {
            "slope": slope,
            "error_ppm": (slope - 1.0) * 1_000_000.0,
            "persistent_offset_step_samples": self.offset_step_samples,
            "configured_limit_ppm": self.max_rate_error * 1_000_000.0,
            "effective_limit_ppm": max(self.max_rate_error, 0.02) * 1_000_000.0,
            "anchor_count": int(len(x)),
            "low_delay_anchor_count": int(np.count_nonzero(low_delay)),
            "low_delay_percentile": self.low_delay_percentile,
            "sample_span": sample_span,
            "anchors": [
                {"rx_sample": float(rx), "media_sample": float(media),
                 "offset_samples": float(media - rx),
                 "low_delay": bool(is_low)}
                for rx, media, is_low in zip(x.tolist(), y.tolist(),
                    (offsets <= low_delay_limit).tolist())
            ],
        }
        if abs(slope - 1.0) > max(self.max_rate_error, 0.02):
            raise ValueError(
                "RTP receive clock exceeds configured V.90 timing envelope "
                f"(fit={slope:.6f}, error_ppm={(slope - 1.0) * 1_000_000.0:+.0f}, "
                f"limit_ppm={max(self.max_rate_error, 0.02) * 1_000_000.0:.0f}, "
                f"anchors={len(x)}, low_delay={int(np.count_nonzero(low_delay))}, "
                f"span_samples={sample_span:.0f})")
        self.slope = slope
        self.intercept = float(np.median(fit_y - slope * fit_x))

    def rx_to_media(self, rx_sample: int) -> int:
        if not self.anchors:
            raise RuntimeError("RTP media clock has no receive anchor")
        return int(round(self.slope * rx_sample + self.intercept))

    def media_to_rx(self, media_sample: int) -> int:
        """Map a media timestamp back to the nearest receive sample index."""
        if self.slope <= 0:
            raise ValueError("RTP media clock slope must be positive")
        return int(round((media_sample - self.intercept) / self.slope))

    def tx_now(self, now: float | None = None) -> int:
        when = time.monotonic() if now is None else float(now)
        return int(round(self.media_at + (when - self.origin) * FS))
