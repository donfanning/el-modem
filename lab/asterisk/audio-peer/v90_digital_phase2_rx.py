"""Bounded incremental receive-side scanner for V.90 Phase 2 INFO frames.

It uses the existing CRC-checked INFO decoder with the analogue modem's
transmit role. This emits frame events only; carrier/tone/probe event
detection remains a separate signal-processing stage.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

import v34_info
import v90_info_reference as v90

FS = 8000
_WINDOW_SAMPLES = int(1.5 * FS)
_SCAN_STRIDE = FS // 5
_DUPLICATE_SAMPLES = 32
_TONE_WIN = 20
_PROBE_WIN = FS // 50  # 20 ms; integer cycles for the 150 Hz probing grid
_PROBE_TONES = (150, 300, 450, 600, 750, 1050, 1350, 1500, 1650, 1950,
                2100, 2250, 2550, 2700, 2850, 3000, 3150, 3300, 3450,
                3600, 3750)
_PROBE_OMITTED = (900, 1200, 1800, 2400)


@lru_cache(maxsize=4)
def _probe_reference_bank(sample_count: int, frequencies: tuple[float, ...]) -> np.ndarray:
    """Cache oscillator references shared by the two L2 probe estimators.

    Both estimators use the same 500 ms L2 window. Building a complex
    exponential separately for each tone was several milliseconds of Python
    receive-callback time. The immutable matrix is keyed by the exact sample
    count and frequencies, so shifted-frequency measurements remain exact.
    """
    t = np.arange(sample_count, dtype=float) / FS
    refs = np.exp(-2j * np.pi * np.asarray(frequencies, dtype=float)[:, None]
                  * t[None, :])
    refs.setflags(write=False)
    return refs


def warm_probe_reference_bank() -> None:
    """Prepare the nominal 500 ms probe bank before the first call arrives."""
    frequencies = _PROBE_TONES + _PROBE_OMITTED
    _probe_reference_bank(FS // 2,
                          tuple(float(freq) for freq in frequencies))


def estimate_probe_frequency_offset_units(samples) -> int:
    """Estimate a common L2 probe-tone offset in 0.02 Hz units.

    Returns the INFO1d sentinel -512 when the available tones do not provide
    a coherent, adequately consistent estimate. The caller supplies the
    actual L2 interval, not the preceding L1 or local echo.
    """
    x = np.asarray(samples, dtype=float).reshape(-1)
    if x.size < FS // 2 or not np.all(np.isfinite(x)):
        return -512
    n = x.size // 2
    if n < 200:
        return -512
    x = x - float(np.mean(x))
    estimates = []
    weights = []
    frequencies = _PROBE_TONES + _PROBE_OMITTED
    refs = _probe_reference_bank(
        x.size, tuple(float(freq) for freq in frequencies))
    tone_refs = refs[:len(_PROBE_TONES)]
    first_values = np.sum(tone_refs[:, :n] * x[None, :n], axis=1)
    second_values = np.sum(tone_refs[:, n:2 * n] * x[None, n:2 * n], axis=1)
    for freq, first, second in zip(_PROBE_TONES, first_values, second_values):
        weight = min(abs(first), abs(second))
        if weight <= 1e-6:
            continue
        tone_offset = float(np.angle(second * np.conj(first)) / (2 * np.pi * n / FS))
        # The INFO field is defined at 1050 Hz. A common oscillator error
        # scales each probe tone in proportion to its nominal frequency, so
        # normalize every line's measured shift to that reference before the
        # robust combination below.
        offset = tone_offset * 1050.0 / freq
        if abs(offset) <= 10.0:
            estimates.append(offset)
            weights.append(float(weight))
    if len(estimates) < 4:
        return -512
    values = np.asarray(estimates)
    weights_np = np.asarray(weights)
    # A weighted median resists one omitted-tone/interference bin while
    # retaining the high-SNR probes as the dominant measurements.
    order = np.argsort(values)
    values, weights_np = values[order], weights_np[order]
    median_index = int(np.searchsorted(np.cumsum(weights_np), weights_np.sum() / 2))
    offset = float(values[min(median_index, len(values) - 1)])
    residual = np.abs(values - offset)
    inliers = residual <= 0.25
    if int(np.count_nonzero(inliers)) < 4:
        return -512
    offset = float(np.average(values[inliers], weights=weights_np[inliers]))
    if abs(offset) > 10.22:
        return -512
    return int(max(-511, min(511, round(offset / 0.02))))


def measure_probe_quality(samples, *, frequency_offset_units: int = -512) -> dict | None:
    """Measure L2 tone levels and a coarse in-band probe SNR.

    The returned SNR is a diagnostic for rate selection, not a calibrated
    line SNR: it compares power in the known L2 tone comb with power at the
    four deliberately omitted frequencies. A failed/short measurement returns
    None so callers do not turn silence into a high-quality channel.
    """
    x = np.asarray(samples, dtype=float).reshape(-1)
    if x.size < FS // 2 or not np.all(np.isfinite(x)):
        return None
    x = x - float(np.mean(x))
    n = len(x)
    offset = (int(frequency_offset_units) * 0.02
              if -511 <= int(frequency_offset_units) <= 511 else 0.0)
    frequencies = _PROBE_TONES + _PROBE_OMITTED
    shifted = tuple(float(freq * (1.0 + offset / 1050.0))
                    for freq in frequencies)
    refs = _probe_reference_bank(n, shifted)
    amplitudes = np.sqrt(2.0) * np.abs(refs @ x) / n
    levels = dict(zip(_PROBE_TONES, amplitudes[:len(_PROBE_TONES)].tolist()))
    noise_levels = dict(zip(_PROBE_OMITTED,
                            amplitudes[len(_PROBE_TONES):].tolist()))
    signal_power = float(np.mean(np.square(list(levels.values()))))
    omitted_power = float(np.mean(np.square(list(noise_levels.values()))))
    if signal_power <= 1e-12:
        return None
    snr_db = 10.0 * np.log10(signal_power / max(omitted_power, 1e-12))
    noise_freqs = np.asarray(_PROBE_OMITTED, dtype=float)
    noise_db = 20.0 * np.log10(np.maximum(
        np.asarray([noise_levels[freq] for freq in _PROBE_OMITTED]), 1e-12))
    tone_snr = {
        freq: float(np.clip(
            20.0 * np.log10(max(levels[freq], 1e-12))
            - np.interp(float(freq), noise_freqs, noise_db),
            -120.0, 120.0))
        for freq in _PROBE_TONES
    }
    return {
        "tone_rms_by_hz": levels,
        "omitted_rms_by_hz": noise_levels,
        "tone_snr_db_by_hz": tone_snr,
        "tone_rms": float(np.sqrt(signal_power)),
        "omitted_rms": float(np.sqrt(omitted_power)),
        "snr_db": float(np.clip(snr_db, -120.0, 120.0)),
        "frequency_offset_units": int(frequency_offset_units),
    }


@dataclass(frozen=True)
class Phase2RxEvent:
    kind: str
    at: int
    complete_at: int
    fields: dict


class V90DigitalPhase2FrameRx:
    """Decode analogue-origin INFO0a and INFO1a from arbitrary PCM chunks.

    ``feed`` accepts mono, 8 kHz linear PCM. It retains at most 1.5 seconds,
    scans eligible frame kinds at 200 ms intervals, and suppresses the same
    physical frame found at several demodulator timing phases. Each event
    includes absolute frame start and completion positions in the concatenated
    input stream; consumers must not act on INFO1a until completion.
    """

    def __init__(self, phases: int = 8):
        if phases <= 0:
            raise ValueError("phases must be positive")
        self.phases = int(phases)
        self._rx = np.zeros(0, dtype=float)
        self._base = 0
        self._total = 0
        self._until_scan = 0
        # A scan window can contain several genuine frames of the same kind.
        # Comparing only with the last returned start lets older frames be
        # emitted again on every later scan as the detector walks that window
        # in start-time order. Retain all recent starts bounded by the receive
        # window so each physical frame is reported once.
        self._seen_frame_starts: dict[str, list[int]] = {}
        # Keep the demodulator's baseband low-pass state warm across the 200 ms
        # INFO scans. Re-filtering the whole 1.5 s receive window at every scan
        # repeated almost all of the same convolution work.
        self._demod = v34_info.InfoDemodulator("answer", phases=self.phases)
        taps = 63
        k = np.arange(taps) - taps // 2
        self._baseband_taps = np.sinc(2 * 420.0 / FS * k) * np.hamming(taps)
        self._baseband_taps /= self._baseband_taps.sum()
        self._mixed_tail = np.zeros(31, dtype=np.complex128)
        self._baseband_history = np.zeros(0, dtype=np.complex128)
        self._baseband_base = 0
        self._baseband_end = 0
        self._mixed_total = 0

    def feed(self, samples, *, kinds: tuple[str, ...] = ("info0", "info1a")) -> list[Phase2RxEvent]:
        x = np.asarray(samples, dtype=float).reshape(-1)
        if not np.all(np.isfinite(x)):
            raise ValueError("receive audio must contain finite samples")
        if not set(kinds) <= {"info0", "info1a"}:
            raise ValueError("supported frame kinds are info0 and info1a")
        if not len(x):
            return []
        self._append_baseband(x)
        self._rx = np.concatenate((self._rx, x))
        self._total += len(x)
        self._until_scan += len(x)
        keep = self._rx[-_WINDOW_SAMPLES:]
        dropped = len(self._rx) - len(keep)
        if dropped:
            self._rx = keep
            self._base += dropped
        if self._until_scan < _SCAN_STRIDE:
            return []
        self._until_scan %= _SCAN_STRIDE
        if not kinds:
            return []

        # The shared INFO API's role labels denote carrier direction/frequency,
        # not the V.90 modem's calling/answering role: "answer" is 2400 Hz.
        # V.90 INFO0a/INFO1a are sent by the analogue call modem at 2400 Hz.
        window = self._baseband_window()
        decoded = self._demod.decode_baseband(window, kinds)
        events: list[Phase2RxEvent] = []
        names = {"info0": "info0a", "info1a": "info1a"}
        for frame_kind in kinds:
            frames = decoded[frame_kind]
            frames.sort(key=lambda frame: frame["_start_sample"])
            for frame in frames:
                at = self._base + int(round(frame["_start_sample"]))
                seen = self._seen_frame_starts.setdefault(frame_kind, [])
                oldest = self._base - _DUPLICATE_SAMPLES
                if seen and seen[0] < oldest:
                    self._seen_frame_starts[frame_kind] = seen = [
                        previous for previous in seen if previous >= oldest]
                if any(abs(at - previous) <= _DUPLICATE_SAMPLES
                       for previous in seen):
                    continue
                seen.append(at)
                complete_at = self._base + int(round(frame["_end_sample"]))
                fields = {key: value for key, value in frame.items()
                          if not key.startswith("_")}
                if frame_kind == "info1a":
                    fields["mode"] = int(fields["sym_call"])
                    if fields["mode"] == 6:
                        v90_fields = v90.parse_info1a_v90(frame["_frame_bits"])
                        if v90_fields is None:
                            continue
                        fields.update({f"v90_{key}": value
                                       for key, value in v90_fields.items()})
                events.append(Phase2RxEvent(names[frame_kind], at, complete_at, fields))
        return sorted(events, key=lambda event: event.at)

    def _append_baseband(self, samples: np.ndarray) -> None:
        """Incrementally low-pass the mixed signal, retaining filter overlap."""
        start = self._mixed_total
        n = np.arange(start, start + len(samples), dtype=float)
        mixed = np.asarray(samples, dtype=float) * np.exp(
            -2j * np.pi * self._demod.fc * n / FS)
        combined = np.concatenate((self._mixed_tail, mixed))
        if len(combined) >= len(self._baseband_taps):
            filtered = np.convolve(combined, self._baseband_taps, mode="valid")
            self._baseband_history = np.concatenate((self._baseband_history, filtered))
            self._baseband_end += len(filtered)
        self._mixed_tail = combined[-(len(self._baseband_taps) - 1):]
        self._mixed_total += len(samples)

        keep = _WINDOW_SAMPLES + 64
        if len(self._baseband_history) > keep:
            drop = len(self._baseband_history) - keep
            self._baseband_history = self._baseband_history[drop:]
            self._baseband_base += drop

    def _baseband_window(self) -> np.ndarray:
        """Return the current receive window, matching same-mode FIR edges."""
        count = len(self._rx)
        result = np.zeros(count, dtype=np.complex128)
        available = max(0, min(count, self._baseband_end - self._base))
        history_start = self._base - self._baseband_base
        if available and history_start >= 0:
            result[:available] = self._baseband_history[
                history_start:history_start + available]

        # The incremental filter only publishes samples with a full 31-sample
        # future context. Recreate the two short window edges with zero padding,
        # as the former full-window np.convolve(..., mode="same") did.
        edge = min(31, count)
        if edge:
            phase = np.exp(-2j * np.pi * self._demod.fc * self._base / FS)
            left = self._demod._baseband(self._rx[:min(count, 94)]) * phase
            result[:edge] = left[:edge]
            right_start = max(0, count - 94)
            right = self._demod._baseband(self._rx[right_start:]) * np.exp(
                -2j * np.pi * self._demod.fc * (self._base + right_start) / FS)
            result[-edge:] = right[-edge:]
        return result

    @property
    def samples_seen(self) -> int:
        return self._total


class V90DigitalPhase2SignalRx:
    """Detect analogue-side Tone A/reversals and the L1/L2 probe interval.

    This is deliberately an event detector, not a Phase 2 state machine. Tone
    A is a coherent 2400 Hz carrier with an optional 1800 Hz guard; the
    reversal fit models both carriers so the guard cannot create or hide a
    phase transition. The L1 detector uses orthogonal 20 ms projections of the
    150 Hz-spaced tone comb. After a confirmed L1 onset, it reports L1 complete
    at 160 ms and L2 complete at the selected 500 ms upper limit. The controller
    may then send Tone B; a product implementation can choose a shorter L2
    receive window based on its channel estimator.

    The caller should route these events only while the controller is in the
    corresponding receive state. This prevents unrelated 2400 Hz data or
    training from being treated as startup Tone A.
    """

    def __init__(self):
        self._pending = np.zeros(0, dtype=float)
        self._audio_base = 0
        self._sample_total = 0
        self._probe_pending = np.zeros(0, dtype=float)
        self._probe_base = 0
        self._tone_phasor: list[complex] = []
        self._tone_stable: list[bool] = []
        self._tone_audio_history = np.zeros(0, dtype=float)
        self._tone_a_active = False
        self._last_a_reversal: int | None = None
        self._comb_run_start: int | None = None
        self._comb_first_rms: float | None = None
        self._probe_start: int | None = None
        self._l1_reported = False
        self._l2_reported = False
        self._probe_complete = False
        self._probe_history = np.zeros(0, dtype=float)
        self._probe_history_base = 0
        self.probe_frequency_offset_units: int | None = None
        self.probe_quality: dict | None = None
        self._last_expect: str | None = None

    @property
    def samples_seen(self) -> int:
        return self._sample_total

    def reset_probe_detection(self) -> None:
        """Open a fresh L1/L2 receive interval without resetting the media clock."""
        self._comb_run_start = None
        self._comb_first_rms = None
        self._probe_start = None
        self._l1_reported = False
        self._l2_reported = False
        self._probe_complete = False
        self.probe_frequency_offset_units = None
        self.probe_quality = None

    def feed(self, samples, *, expect: str = "all") -> list[Phase2RxEvent]:
        if expect not in ("all", "tone_a", "probe", "none"):
            raise ValueError("expect must be all, tone_a, probe, or none")
        if self._last_expect == "probe" and expect != "probe":
            # A new receive phase is the only reliable boundary between the
            # first and second probing exchange. Do not reinterpret a still
            # present comb as a second L1 after the 500 ms observation limit.
            self._comb_run_start = None
            self._comb_first_rms = None
            self._probe_start = None
            self._l1_reported = self._l2_reported = False
            self._probe_complete = False
        self._last_expect = expect
        x = np.asarray(samples, dtype=float).reshape(-1)
        if not np.all(np.isfinite(x)):
            raise ValueError("receive audio must contain finite samples")
        if not len(x):
            return []
        self._sample_total += len(x)
        self._probe_history = np.concatenate((self._probe_history, x))
        if len(self._probe_history) > int(.9 * FS):
            drop = len(self._probe_history) - int(.9 * FS)
            self._probe_history = self._probe_history[drop:].copy()
            self._probe_history_base += drop
        self._pending = np.concatenate((self._pending, x))
        self._probe_pending = np.concatenate((self._probe_pending, x))
        events: list[Phase2RxEvent] = []

        while len(self._pending) >= _TONE_WIN:
            block = self._pending[:_TONE_WIN]
            self._pending = self._pending[_TONE_WIN:]
            at = self._audio_base
            self._audio_base += _TONE_WIN
            events.extend(self._tone_window(block, at,
                                            enabled=expect in ("all", "tone_a")))

        while len(self._probe_pending) >= _PROBE_WIN:
            block = self._probe_pending[:_PROBE_WIN]
            self._probe_pending = self._probe_pending[_PROBE_WIN:]
            at = self._probe_base
            self._probe_base += _PROBE_WIN
            self._probe_window(block, at, enabled=expect in ("all", "probe"))

        end = self._sample_total
        if self._probe_start is not None:
            l1_end = self._probe_start + int(.160 * FS)
            l2_end = l1_end + int(.500 * FS)
            if not self._l1_reported and end >= l1_end:
                events.append(Phase2RxEvent("l1_complete", l1_end, end, {}))
                self._l1_reported = True
            if not self._l2_reported and end >= l2_end:
                fields = {}
                start = self._probe_start + int(.160 * FS)
                lo = start - self._probe_history_base
                hi = l2_end - self._probe_history_base
                if lo >= 0 and hi <= len(self._probe_history):
                    l2_samples = self._probe_history[lo:hi]
                    if self.probe_frequency_offset_units is None:
                        measured = estimate_probe_frequency_offset_units(
                            l2_samples)
                        self.probe_frequency_offset_units = measured
                        fields["freq_offset"] = measured
                    self.probe_quality = measure_probe_quality(
                        l2_samples,
                        frequency_offset_units=(self.probe_frequency_offset_units
                                                if self.probe_frequency_offset_units is not None
                                                else -512),
                    )
                    fields["probe_quality"] = self.probe_quality
                events.append(Phase2RxEvent("l2_complete", l2_end, end, fields))
                self._l2_reported = True
                self._probe_start = None
                self._comb_run_start = None
                self._comb_first_rms = None
                self._probe_complete = True
        return sorted(events, key=lambda event: event.at)

    @staticmethod
    def _tone_features(block: np.ndarray, at: int, freq: float) -> tuple[complex, float, float]:
        t = at + np.arange(_TONE_WIN)
        z = block * np.exp(-2j * np.pi * freq * t / FS)
        return z.sum(), abs(z.sum()) / (np.abs(z).sum() + 1e-12), float(np.abs(block).mean())

    def _tone_window(self, block: np.ndarray, at: int, *, enabled: bool) -> list[Phase2RxEvent]:
        events: list[Phase2RxEvent] = []
        self._tone_audio_history = np.concatenate((self._tone_audio_history, block))[-(3 * _TONE_WIN):]
        phasor, coh, level = self._tone_features(block, at, 2400.0)
        if len(self._tone_stable) == 16:
            self._tone_phasor.pop(0)
            self._tone_stable.pop(0)
        self._tone_phasor.append(phasor)
        stable = coh > .45 and level > 60
        self._tone_stable.append(stable)
        j = len(self._tone_stable) - 1

        if not enabled:
            self._tone_a_active = False
            return events

        # Emit a tone-on event from the beginning of a six-window confirmation
        # run. Phase reversals do not end the carrier-present state.
        if (not self._tone_a_active and j >= 5
                and all(self._tone_stable[j - k] for k in range(6))):
            self._tone_a_active = True
            events.append(Phase2RxEvent("tone_a", at - 5 * _TONE_WIN, at, {}))
        elif self._tone_a_active and j >= 1 and not stable and not self._tone_stable[j - 1]:
            self._tone_a_active = False

        # Phase-transition fit: three 2.5 ms windows, jointly fitting the
        # reversed 2400 Hz signal and the non-reversing 1800 Hz guard.
        # The immediately preceding window can straddle the phase reversal;
        # qualify the carrier from the ten clean windows before that edge.
        # This matches ToneTracker's j-11 .. j-2 guard and avoids suppressing
        # real reversals when the edge window is naturally incoherent.
        if (j >= 12 and all(self._tone_stable[j - k] for k in range(2, 12))
                and self._tone_stable[j] and self._tone_stable[j - 2]
                and (self._tone_phasor[j] * np.conj(self._tone_phasor[j - 2])).real < 0):
            start = at - 2 * _TONE_WIN
            if self._last_a_reversal is None or start - self._last_a_reversal >= 400:
                fit = self._fit_a_reversal(start)
                if fit is not None:
                    reversal_at, nmse = fit
                    self._last_a_reversal = reversal_at
                    events.append(Phase2RxEvent(
                        "tone_a_reversal", reversal_at, at + _TONE_WIN,
                        {"normalized_fit_error": nmse},
                    ))

        return events

    def _fit_a_reversal(self, start: int) -> tuple[int, float] | None:
        if len(self._tone_audio_history) < 3 * _TONE_WIN:
            return None
        x = self._tone_audio_history
        n = len(x)
        t = start + np.arange(n)
        carrier = np.column_stack((
            np.cos(2 * np.pi * 2400.0 * t / FS),
            np.sin(2 * np.pi * 2400.0 * t / FS)))
        guard = np.column_stack((
            np.cos(2 * np.pi * 1800.0 * t / FS),
            np.sin(2 * np.pi * 1800.0 * t / FS)))
        splits = np.arange(_TONE_WIN - 4, 2 * _TONE_WIN + 5)

        # For any split, the carrier columns are both multiplied by the same
        # +/-1 sign, so their 2x2 Gram block is constant. The guard block is
        # constant too; only the carrier/guard cross block and carrier part of
        # A.T*x vary. Prefix sums build those for every candidate at once,
        # replacing one least-squares decomposition per split (29 LAPACK
        # calls) with a single batched solve of the tiny normal matrices.
        carrier_gram = carrier.T @ carrier
        guard_gram = guard.T @ guard
        cross_rows = np.einsum("ni,nj->nij", carrier, guard)
        cross_prefix = np.concatenate((
            np.zeros((1, 2, 2)), np.cumsum(cross_rows, axis=0)), axis=0)
        cross = 2.0 * cross_prefix[splits] - cross_rows.sum(axis=0)
        gram = np.empty((len(splits), 4, 4), dtype=float)
        gram[:, :2, :2] = carrier_gram
        gram[:, 2:, 2:] = guard_gram
        gram[:, :2, 2:] = cross
        gram[:, 2:, :2] = cross.transpose(0, 2, 1)

        carrier_x = carrier * x[:, None]
        carrier_prefix = np.concatenate((
            np.zeros((1, 2)), np.cumsum(carrier_x, axis=0)), axis=0)
        rhs = np.column_stack((
            2.0 * carrier_prefix[splits] - carrier_x.sum(axis=0),
            np.broadcast_to(guard.T @ x, (len(splits), 2))))
        # Make the batched RHS explicitly vector-shaped. NumPy 2 treats a
        # two-dimensional RHS as a matrix rather than a batch of vectors,
        # which otherwise mismatches (batch, 4, 4) with (batch, 4).
        coefficients = np.linalg.solve(gram, rhs[..., None])[..., 0]
        residual_energy = float(x @ x) - np.sum(rhs * coefficients, axis=1)
        best_index = int(np.argmin(residual_energy))
        best_at = int(splits[best_index])

        # Refit the winning split with the original least-squares routine so
        # the reported normalized error retains its previous numeric meaning.
        sign = np.where(np.arange(n) < best_at, 1.0, -1.0)
        matrix = np.column_stack((sign[:, None] * carrier, guard))
        coeff, *_ = np.linalg.lstsq(matrix, x, rcond=None)
        residual = x - matrix @ coeff
        error = float(residual @ residual)
        energy = float(np.sum((matrix[:, :2] @ coeff[:2]) ** 2))
        nmse = error / (energy + 1e-12)
        if nmse > .2:
            return None
        return start + best_at, nmse

    def _probe_window(self, block: np.ndarray, at: int, *, enabled: bool) -> None:
        if not enabled or self._probe_complete:
            self._comb_run_start = None
            return
        x = block - float(np.mean(block))
        energy = float(x @ x)
        rms = float(np.sqrt(energy / len(x)))
        if energy <= 0:
            comb_fraction = 0.0
        else:
            n = np.arange(len(x))
            line_energy = 0.0
            for freq in _PROBE_TONES:
                z = np.sum(x * np.exp(-2j * np.pi * freq * n / FS))
                line_energy += 2.0 * float(abs(z) ** 2) / len(x)
            comb_fraction = line_energy / energy
        # A partial first 20 ms window has spectral leakage, so use a loose
        # threshold to mark a candidate; require a full-strength comb window
        # before confirming the receive interval.
        comb_candidate = rms > 50 and comb_fraction > .20
        if comb_candidate:
            if self._comb_run_start is None:
                self._comb_run_start = at
                self._comb_first_rms = rms
            elif (self._probe_start is None
                  and at - self._comb_run_start >= _PROBE_WIN
                  and comb_fraction > .55):
                # A non-overlapping 20 ms detector window can straddle the
                # actual burst onset. For silence-to-L1 startup, estimate the
                # occupied fraction from its energy relative to the next full
                # comb window. This avoids starting the 160 ms L1 timer early
                # by as much as one detector block.
                first_rms = self._comb_first_rms or rms
                occupied = min(1.0, max(0.0, (first_rms / max(rms, 1e-12)) ** 2))
                onset_offset = min(
                    _PROBE_WIN,
                    int(round((1.0 - occupied) * _PROBE_WIN)) + FS // 200,
                )
                self._probe_start = self._comb_run_start + onset_offset
                self._l1_reported = self._l2_reported = False
        else:
            self._comb_run_start = None
            self._comb_first_rms = None
