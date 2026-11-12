"""Streaming acquisition/training wrapper for the V.90 analogue upstream.

V.34 taught us to keep waveform acquisition, equalizer fitting, and symbol
tracking as explicit stages with independently inspectable outcomes. This
wrapper applies that split to the V.90 caller-to-digital-answerer path: it
waits for the caller's S/S-bar/PP waveform, fits the existing V.34 Phase 3
front end to the known PP/TRN, then hands the same retained baseband timeline
to the V.90 Ja tracker.

This is a DSP component only. The caller must provide the negotiated upstream
baud/carrier and decoded 8 kHz linear samples. It does not infer a profile,
drive the digital Phase 3 transmitter, or implement Phase 4/data.
"""

from __future__ import annotations

from collections import deque
import os
import time

import numpy as np

import v34_rx
from v90_upstream_rx import V90UpstreamPhase3Receiver


FS = 8000


class V90UpstreamPhase3Session:
    """Acquire a streaming caller waveform and extract CRC-valid Ja descriptors.

    ``baud`` and ``carrier`` are explicit negotiated values. Input chunks are
    signed/float linear PCM in sample units, not G.711 codewords. The session
    retains at most ``max_seconds`` of PCM through the Phase 4 control start;
    exceeding the bound fails closed instead of growing without limit.
    """

    ACQUIRE_IDLE_SAMPLES = 4000          # 0.5 s of silence ends the acquisition probing
    ACQUIRE_INTERVAL_SAMPLES = 1600       # probe at 200 ms, not every 20 ms media packet
    ACQUIRE_ROUGH_FLOOR = 0.15            # full-window score is only a candidate seed
    ACQUIRE_WINDOW_SAMPLES = 8000          # bound FFT cost to the most recent second
    MAX_SYMBOLS_PER_MEDIA_FEED = 128     # bound post-acquisition catch-up work
    ACQUIRING = "acquiring"
    TRAINING = "training"
    TRACKING = "tracking"
    COMPLETE = "complete"
    FAILED = "failed"

    def __init__(self, baud: float, carrier: float, *, trn_scrambler_role: str = "answer",
                 span: int = 12, train_symbols: int = 512,
                 max_training_mse: float = 0.25, max_seconds: float = 40.0,
                 max_data_seconds: float = 300.0,
                 min_acquisition_metric: float = 0.6,
                 probe_frequency_offset_units: int = 0,
                 equalizer_mu: float | None = None):
        if baud <= 0 or carrier <= 0:
            raise ValueError("V.90 upstream baud and carrier must be positive")
        if trn_scrambler_role not in ("answer", "call"):
            raise ValueError("TRN scrambler role must be 'answer' or 'call'")
        if span not in (12, 24):
            raise ValueError("equalizer span must be 12 or 24")
        if (isinstance(probe_frequency_offset_units, bool)
                or not isinstance(probe_frequency_offset_units, int)
                or not -512 <= probe_frequency_offset_units <= 511):
            raise ValueError("probe frequency offset must be -512 or in -511..511 units")
        if max_data_seconds < max_seconds:
            raise ValueError("data retention cannot be shorter than the startup bound")
        if train_symbols < 512 or max_training_mse <= 0 or max_seconds <= 0:
            raise ValueError("invalid V.90 upstream training or retention limit")
        self.baud = float(baud)
        self.carrier = float(carrier)
        self.probe_frequency_offset_units = int(probe_frequency_offset_units)
        self.equalizer_mu = equalizer_mu
        # V.90 INFO1d Table 9 reports the digital receiver's measured offset
        # of the received L2 probing tones in 0.02 Hz units. -512 means the
        # measurement is unavailable. A common oscillator error scales both
        # the received symbol clock and carrier by the measured ratio at
        # 1050 Hz; use that ratio to center the V.34-compatible upstream
        # receiver. Keep the negotiated nominal profile separately for rate
        # policy and reporting.
        offset_hz = (0.0 if probe_frequency_offset_units == -512
                     else probe_frequency_offset_units * 0.02)
        self.receive_clock_ratio = 1.0 + offset_hz / 1050.0
        self.receive_baud = self.baud * self.receive_clock_ratio
        self.receive_carrier = self.carrier * self.receive_clock_ratio
        self.role = trn_scrambler_role
        self.span = int(span)
        self.train_symbols = int(train_symbols)
        self.max_training_mse = float(max_training_mse)
        self.max_samples = int(max_seconds * FS)
        # ``max_seconds`` bounds startup (acquisition through the Phase 4 control
        # start). Once data tracking begins the call may legitimately run longer
        # (a 64 KiB transfer takes 15-50 s), so a separate, larger bound applies;
        # the PCM history is still kept in full, hence a bound at all.
        self.max_data_samples = int(max_data_seconds * FS)
        self._data_phase = False
        self.min_acquisition_metric = float(min_acquisition_metric)
        self.state = self.ACQUIRING
        self.failure_reason: str | None = None
        self.frontend: v34_rx.Phase3Rx | None = None
        self.receiver: V90UpstreamPhase3Receiver | None = None
        self.training_mse: float | None = None
        self.detected_trn_role: str | None = None
        self.acquisition: dict | None = None
        self.samples_seen = 0
        self._pcm_storage = np.empty(0, dtype=np.float64)
        self._pcm_length = 0
        self._pcm = self._pcm_storage[:0]
        self._next_acquire_at = 0
        self._last_active_at = 0            # samples_seen when the line last carried a signal
        self.descriptors: list[dict] = []
        self._startup_trace_enabled = os.environ.get("V90_UPSTREAM_TRACE") == "1"
        self._startup_trace: deque[dict] = deque(maxlen=512)
        self._startup_trace_truncated = False

    def startup_trace_report(self) -> dict | None:
        """Return the opt-in acquisition/training CPU trace."""
        if not self._startup_trace_enabled:
            return None
        rows = list(self._startup_trace)
        return {"enabled": True, "count": len(rows), "capacity": 512,
                "truncated": self._startup_trace_truncated,
                "max_cpu_ns": max((int(row["cpu_ns"]) for row in rows),
                                  default=0),
                "rows": rows}

    def _append_startup_trace(self, row: dict) -> None:
        """Keep earliest startup work; later data rows must not evict it."""
        if not self._startup_trace_enabled:
            return
        if len(self._startup_trace) >= self._startup_trace.maxlen:
            self._startup_trace_truncated = True
            return
        self._startup_trace.append(row)

    def _append_pcm(self, chunk: np.ndarray) -> None:
        """Append PCM without recopying the retained call on every RTP packet."""
        required = self._pcm_length + int(chunk.size)
        storage = self._pcm_storage
        if required > storage.size:
            capacity = min(
                self.max_data_samples,
                max(required, 1024, 2 * storage.size),
            )
            expanded = np.empty(capacity, dtype=np.float64)
            if self._pcm_length:
                expanded[:self._pcm_length] = storage[:self._pcm_length]
            self._pcm_storage = storage = expanded
        storage[self._pcm_length:required] = chunk
        self._pcm_length = required
        self._pcm = storage[:required]

    def feed(self, samples) -> list[dict]:
        """Consume one ordered linear-PCM chunk and return newly parsed Ja frames."""
        chunk = np.asarray(samples, dtype=np.float64).reshape(-1)
        if not np.all(np.isfinite(chunk)):
            return self._fail("non-finite PCM sample")
        if self.state == self.FAILED:
            return []
        if not self._data_phase and self.samples_seen + chunk.size > self.max_samples:
            return self._fail("startup exceeded bounded V.90 Phase 3 capture")
        if self._data_phase and self.samples_seen + chunk.size > self.max_data_samples:
            return self._fail("data phase exceeded bounded V.90 upstream capture")
        if chunk.size:
            self._append_pcm(chunk)
            self.samples_seen += int(chunk.size)
            # A-law idle decodes to about +-8, mu-law idle to 0; a real upstream signal is
            # hundreds to thousands, so an RMS of 32 separates "something is there" from idle.
            if float(np.dot(chunk, chunk)) >= 1024.0 * chunk.size:
                self._last_active_at = self.samples_seen

        try:
            if self.state == self.ACQUIRING:
                self._acquire()
            if self.state == self.TRAINING:
                self._train()
            if self.state in (self.TRACKING, self.COMPLETE) and chunk.size:
                update_started = (time.thread_time_ns()
                                  if self._startup_trace_enabled else 0)
                self.frontend._z = self.frontend.update_baseband(self._pcm)
                update_cpu = (time.thread_time_ns() - update_started
                              if self._startup_trace_enabled else 0)
                feed_started = (time.thread_time_ns()
                                if self._startup_trace_enabled else 0)
                start_symbol = self.receiver.eq.k
                # Acquisition retains the waveform prefix. Once timing and the
                # equalizer are established, do not drain a multi-second
                # symbol backlog in one RTP receive callback; carry the cursor
                # forward over later media ticks while preserving symbol order.
                found = self.receiver.feed(
                    max_symbols=self.MAX_SYMBOLS_PER_MEDIA_FEED)
                if self._startup_trace_enabled:
                    feed_cpu = time.thread_time_ns() - feed_started
                    self._append_startup_trace({
                        "kind": "receiver_feed",
                        "samples_seen": int(self.samples_seen),
                        "baseband_samples": int(len(self.frontend._z)),
                        "symbols_before": int(start_symbol),
                        "symbols_after": int(self.receiver.eq.k),
                        "symbols_advanced": int(self.receiver.eq.k - start_symbol),
                        "update_baseband_cpu_ns": int(update_cpu),
                        "feed_cpu_ns": int(feed_cpu),
                        "cpu_ns": int(update_cpu + feed_cpu),
                    })
                if found:
                    self.descriptors.extend(found)
                    self.state = self.COMPLETE
                    return found
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            return self._fail(f"{type(exc).__name__}: {exc}")
        return []

    def read_new_symbol_rotations(self) -> list[int]:
        """Drain new upstream symbol decisions for a continuous Phase 4 RX."""
        if self.receiver is None:
            return []
        return self.receiver.read_new_symbol_rotations()

    def begin_phase4(self, constellation: int) -> None:
        """Apply the negotiated Jd constellation to subsequent RX symbols."""
        if self.receiver is None:
            raise RuntimeError("cannot enter Phase 4 before upstream training")
        self.receiver.begin_phase4(constellation)

    def read_new_symbol_decisions(self) -> list[tuple[int, int]]:
        """Drain new (rotation, quarter) decisions for CP/CP′/E processing."""
        if self.receiver is None:
            return []
        return self.receiver.read_new_symbol_decisions()

    def read_new_symbol_observations(self):
        """Drain phase-four decisions with aligned equalizer input powers."""
        if self.receiver is None:
            return []
        return self.receiver.read_new_symbol_observations()

    def read_new_equalized_observations(self):
        """Drain aligned equalizer outputs, phase-four decisions and powers."""
        if self.receiver is None:
            return []
        return self.receiver.read_new_equalized_observations()

    def hold_b1_feedback(self) -> None:
        if self.receiver is None:
            raise RuntimeError("cannot hold B1 feedback before upstream tracking")
        self.receiver.eq.adapt = False
        self.receiver.eq.hold_carrier_phase = True

    def set_symbol_callback(self, callback) -> None:
        if self.receiver is None:
            raise RuntimeError("cannot register symbol callback before upstream tracking")
        self.receiver.set_symbol_callback(callback)

    def begin_data_tracking(self, receiver) -> None:
        if self.receiver is None:
            raise RuntimeError("cannot start data tracking before upstream tracking")
        eq = self.receiver.eq
        # B1's complex gain/tap fit is referenced to the held phase. Resume
        # carrier NCO integration only after that fit has qualified.
        eq.hold_carrier_phase = False
        eq.begin_data_tracking(precoded=bool(any(receiver.h)))
        self.receiver.begin_data_tracking(receiver.decide)
        eq.adapt = True
        self._data_phase = True

    def _acquire(self) -> None:
        frontend = self.frontend
        if frontend is None:
            frontend = self.frontend = v34_rx.Phase3Rx(
                self.receive_baud, self.receive_carrier, self.role)
        # Avoid rerunning the full correlation on every RTP packet while still
        # probing at the media packet cadence. The acquisition correlator can
        # locate a preamble anywhere in the retained startup window.
        if self.samples_seen < self._next_acquire_at:
            return
        minimum = int((432 + 2 * frontend.LEAD) * frontend.spl) + 100
        if self.samples_seen < minimum:
            return
        self._next_acquire_at = self.samples_seen + self.ACQUIRE_INTERVAL_SAMPLES
        if self.samples_seen - self._last_active_at > self.ACQUIRE_IDLE_SAMPLES:
            # A silent line has nothing to find, and the correlation covers the whole retained
            # window: left running through a long silence it costs more than real time and the
            # RTP media loop falls behind (seen while a retrain waited for a peer that was
            # silent). The last probe that saw the signal already searched all of it.
            return
        started = time.thread_time_ns() if self._startup_trace_enabled else 0
        # The S/S-bar and PP sections can have different effective amplitude
        # after the transmit power normalization required by V.34 §10.1.3.
        # A single full-window correlation then biases both timing and delay.
        # Use that inexpensive-to-interpret peak only as a local search seed;
        # the V.90-specific segmented check below makes the actual decision.
        scan_origin = max(0, int(self._pcm.size) - self.ACQUIRE_WINDOW_SAMPLES)
        scan_pcm = self._pcm[scan_origin:]
        result = frontend.acquire(scan_pcm, delays=(0.0,), min_metric=0.0)
        rough_metric = (None if result is None else float(result["metric"]))
        if result is not None and result["metric"] >= self.ACQUIRE_ROUGH_FLOOR:
            result = self._refine_acquisition(frontend, result)
            if result is not None:
                result["lag"] += scan_origin
                frontend.lag = int(result["lag"])
        else:
            result = None
        if self._startup_trace_enabled:
            best_candidate = getattr(frontend, "last_best", None)
            self._append_startup_trace({
                "kind": "acquire",
                "samples_seen": int(self.samples_seen),
                "capture_samples": int(self._pcm.size),
                "scan_start": scan_origin,
                "scan_samples": int(scan_pcm.size),
                "accepted": result is not None,
                "metric": (None if result is None else float(result["metric"])),
                "rough_metric": rough_metric,
                "segment_metrics": (None if result is None
                                    else result.get("segment_metrics")),
                "best_candidate": (None if result is not None
                                   or best_candidate is None
                                   else dict(best_candidate)),
                "cpu_ns": int(time.thread_time_ns() - started),
            })
        if result is None:
            return
        # `Phase3Rx.acquire()` searches every complete correlation window that
        # fits in the current buffer. Near the right edge, a short prefix of
        # the real S/S-bar/PP preamble can correlate with an earlier Phase 2
        # tail and win before the complete preamble is available. Do not
        # commit that tentative location until its full reference window and
        # trailing guard have arrived; the next scan can then compare the
        # complete preamble and reject the partial-window peak.
        acquisition_tail = int(
            (432 + 2 * frontend.LEAD) * frontend.spl) + 100
        if self.samples_seen < int(result["lag"]) + acquisition_tail:
            # The peak is stable but its trailing guard has not arrived yet.
            # Retry on the next media packet rather than waiting one full
            # coarse-search interval.
            self._next_acquire_at = self.samples_seen
            return
        self.acquisition = result
        self.state = self.TRAINING

    def _refine_acquisition(self, frontend: v34_rx.Phase3Rx,
                            rough: dict) -> dict | None:
        """Refine a whole-preamble peak using independent S/S-bar and PP gains.

        The two scores must agree on a single sample lag and fractional delay.
        Trimming eight symbols from each segment edge avoids giving the
        pulse-shaping transition between differently scaled sections undue
        weight. The bounded local scan is small; the FFT correlation remains
        the coarse search over the retained call audio.
        """
        z = frontend._z
        if z is None:
            return None
        seq = np.concatenate((v34_rx.tx.s_signal(128),
                              v34_rx.tx.s_signal(16, True),
                              v34_rx.tx.pp_signal()))
        center = int(rough["lag"])
        lags = np.arange(max(0, center - 48), center + 49, dtype=np.int64)
        best = None
        for delay in (0.0, 0.25, 0.5, 0.75):
            ref = v34_rx.tx.qam_modulate(
                seq, self.receive_baud, 0, 1, delay=delay, baseband=True)
            windows = []
            for first, last in ((8, 136), (152, 424)):
                lo = int((frontend.LEAD + first + delay) * frontend.spl)
                hi = int((frontend.LEAD + last + delay) * frontend.spl)
                template = ref[lo:hi]
                signal_start = int(lags[0]) + lo
                signal_stop = int(lags[-1]) + hi
                if signal_start < 0 or signal_stop > z.size:
                    windows.append(None)
                    continue
                block = z[signal_start:signal_stop]
                view = np.lib.stride_tricks.sliding_window_view(
                    block, template.size)
                corr = view @ np.conjugate(template)
                energy = np.einsum("ij,ij->i", view.conj(), view).real
                denominator = np.sqrt(
                    energy * float(np.vdot(template, template).real) + 1e-12)
                windows.append(np.abs(corr) / denominator)
            if any(row is None for row in windows):
                continue
            score = np.minimum(windows[0], windows[1])
            idx = int(np.argmax(score))
            candidate = (float(score[idx]), int(lags[idx]), float(delay),
                         float(windows[0][idx]), float(windows[1][idx]))
            if best is None or candidate[0] > best[0]:
                best = candidate
        if best is None or best[0] < self.min_acquisition_metric:
            return None
        metric, lag, delay, s_metric, pp_metric = best
        frontend.lag, frontend.delay = lag, delay
        frontend.metric = metric
        # The equalizer fit determines its own complex gain. This value is
        # retained for diagnostics only and is not used as a decision scale.
        frontend.gain = rough["gain"]
        return {"metric": metric, "lag": lag, "delay": delay,
                "gain": float(rough["gain"]),
                "segment_metrics": {"s_sbar": s_metric, "pp": pp_metric}}

    def _train(self) -> None:
        frontend = self.frontend
        started = time.thread_time_ns() if self._startup_trace_enabled else 0
        frontend._z = frontend.update_baseband(self._pcm)
        required = 144 + 288 + self.train_symbols + 24
        available = int((len(frontend._z) - frontend.lag) / frontend.spl) - frontend.LEAD
        if available < required:
            if self._startup_trace_enabled:
                self._append_startup_trace({
                    "kind": "train_wait",
                    "samples_seen": int(self.samples_seen),
                    "available_symbols": int(available),
                    "required_symbols": int(required),
                    "cpu_ns": int(time.thread_time_ns() - started),
                })
            return
        fit_started = time.thread_time_ns() if self._startup_trace_enabled else 0
        mse, role, trn_scale = self._train_power_normalized(frontend)
        if self._startup_trace_enabled:
            self._append_startup_trace({
                "kind": "trn_fit",
                "samples_seen": int(self.samples_seen),
                "available_symbols": int(available),
                "required_symbols": int(required),
                "mse": float(mse),
                "role": str(role),
                "trn_scale": {"real": float(trn_scale.real),
                              "imag": float(trn_scale.imag)},
                "cpu_ns": int(time.thread_time_ns() - fit_started),
            })
        self.training_mse = float(mse)
        self.detected_trn_role = role
        if mse > self.max_training_mse:
            self._fail(
                f"equalizer training residual {mse:.4g} exceeds "
                f"limit {self.max_training_mse:.4g}")
            return
        self.receiver = V90UpstreamPhase3Receiver(
            frontend, role, equalizer_mu=self.equalizer_mu)
        self.state = self.TRACKING

    def _train_power_normalized(self, frontend: v34_rx.Phase3Rx
                                ) -> tuple[float, str, complex]:
        """Fit PP and TRN with the received four-point power normalization.

        V.34 §10.1.3 requires the transmitted training signals to maintain the
        selected power level. The 4-point constellation's raw coordinates
        have mean energy two while PP has unit energy, so the receiver must
        estimate the complex TRN scale instead of assuming both raw symbol
        sets arrive at identical amplitudes. The final taps are normalized so
        the existing symbol slicer sees the conventional point-zero radius.
        """
        pp = v34_rx.tx.pp_signal()[:288]
        k_start = 128 + 16
        order = [self.role] + [role for role in ("answer", "call")
                               if role != self.role]
        best = None
        for role in order:
            # Establish timing/channel from PP, then measure the peer's TRN
            # scale against that independent fit. A wrong scrambler role has
            # near-zero coherent scale and a near-unit residual.
            frontend.train_known(pp, k_start, span=self.span)
            trn, _ = v34_rx.tx.trn_symbols(self.train_symbols, role)
            observed = frontend.equalize(k_start + len(pp), self.train_symbols)
            scale = np.vdot(trn, observed) / max(float(np.vdot(trn, trn).real), 1e-12)
            residual = float(np.mean(np.abs(observed - scale * trn) ** 2)
                             / max(float(np.mean(np.abs(observed) ** 2)), 1e-12))
            if abs(scale) < 0.2:
                residual = max(residual, 1.0)
            candidate = (residual, role, complex(scale))
            if best is None or candidate[0] < best[0]:
                best = candidate
            if residual <= self.max_training_mse:
                break
        if best is None or best[0] > self.max_training_mse:
            return (float("inf"), self.role, 1.0 + 0.0j)

        residual, role, scale = best
        trn, _ = v34_rx.tx.trn_symbols(self.train_symbols, role)
        known = np.concatenate((pp, scale * trn))
        joint_mse = frontend.train_known(known, k_start, span=self.span)
        fitted = frontend.equalize(k_start + len(pp), self.train_symbols)
        fitted_scale = np.vdot(trn, fitted) / max(
            float(np.vdot(trn, trn).real), 1e-12)
        if abs(fitted_scale) < 0.2:
            return (float("inf"), role, complex(fitted_scale))
        frontend.taps = frontend.taps / fitted_scale
        # Re-score in the scale-normalized constellation domain used by the
        # existing TRN tracker and slicer.
        normalized = frontend.equalize(k_start + len(pp), self.train_symbols)
        normalized_mse = float(np.mean(np.abs(normalized - trn) ** 2)
                               / max(float(np.mean(np.abs(trn) ** 2)), 1e-12))
        return normalized_mse, role, complex(fitted_scale)

    def _fail(self, reason: str) -> list[dict]:
        self.state = self.FAILED
        self.failure_reason = reason
        return []
