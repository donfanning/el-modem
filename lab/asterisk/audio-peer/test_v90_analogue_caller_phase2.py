import time
import os
import socket
import struct
import sys
import unittest
from unittest import mock
from pathlib import Path

import numpy as np

from v90_analogue_caller_phase2 import V90AnalogueCallerPhase2
from v90_analogue_caller_engine import V90AnalogueCallerPhase2Engine
from v90_analogue_caller_phase3 import V90AnalogueCallerPhase3
from v90_analogue_downstream_rx import V90AnalogueDownstreamPhase3Receiver
from v90_analogue_phase2_rx import (V90AnaloguePhase2Adapter,
                                    V90AnaloguePhase2SignalRx,
                                    AnaloguePhase2RxEvent)
from v90_analogue_phase2_tx import (AnaloguePhase2Action,
                                    V90AnaloguePhase2Transmitter)
from v90_analogue_phase3_session import V90AnaloguePhase3Session
from v90_analogue_phase4 import V90AnaloguePhase4Controller
from v90_analogue_phase3_tx import V90AnaloguePhase3Transmitter
from v90_analogue_phase4 import V90AnaloguePhase4MediaTx
from v90_digital_phase2 import V90DigitalPhase2Signals
from v90_digital_phase2_adapter import V90DigitalPhase2Adapter
from v90_digital_phase2_control import Phase2Action, V90DigitalPhase2Controller
from v90_digital_phase2_tx import V90DigitalPhase2Transmitter
import v34_call
import v34_data
import v34_g711
import v34_tx
import v90_info_reference as v90
from g711 import alaw_to_linear, linear_to_alaw, linear_to_ulaw, ulaw_to_linear
from v90_cp import parse_cp
from v8 import V8_PCM_ANALOG, V8_PCM_DIGITAL, V8_PROTOCOL_LAPM_V42
from v90_digital_phase2_engine import V90DigitalAnswererPhase2Engine
from v90_digital_phase2_rx import V90DigitalPhase2SignalRx
from modem_dce import OnlineAudioSession
from v90_digital_phase4 import V90DigitalPhase4Encoder
from v90_digital_phase4 import ri_codewords
from v90_digital_phase4_rx import V90DigitalPhase4Receiver
from v90_lapm import V90LapmLink
from v90_mp_policy import answerer_mp_profile, select_analog_upstream_profile
from v90_upstream_data import V90UpstreamDataReceiver
from test_v90_digital_phase3_session import (
    _encode_continuous_4point_segments, _paired_lapm_caller_bits)
from test_v90_cp import _encode_symbols, _frame

import v90_test_ja_waveform as ja_control


def _delayed_echo(tx_history, start, count, delay_samples=320, gain=0.25):
    """Small, delayed local-hybrid return path for paired PCM tests."""
    source = np.arange(start, start + count) - delay_samples
    result = np.zeros(count, dtype=float)
    valid = (source >= 0) & (source < len(tx_history))
    result[valid] = gain * tx_history[source[valid]]
    return result


class _PairedNlmsEchoProbe:
    """Test-only NLMS with a TX/RX coherence gate and delay acquisition."""

    ACQUISITION_COHERENCE = 0.80
    ADAPTATION_COHERENCE = 0.80

    def __init__(self, taps=16, mu=0.25, max_delay_samples=1600):
        self.weights = np.zeros(int(taps), dtype=float)
        self.history = np.zeros(int(taps) - 1, dtype=float)
        self.mu = float(mu)
        self.max_delay_samples = int(max_delay_samples)
        self.delay_samples = None
        self.delay_fit_correlation = None
        self.delay_fit_gain = None
        self._training_pcm = np.zeros(0, dtype=float)
        self._training_at = np.zeros(0, dtype=np.int64)
        self._candidate_delay = None
        self._candidate_count = 0
        self._candidate_blocks = []
        self.best_acquisition_coherence = 0.0
        self.coherent_candidate_blocks = 0
        self.trained = False
        self.training_samples = 0
        self.adaptation_blocks = 0
        self.oracle_silent_blocks = 0
        self.oracle_active_blocks = 0
        self.false_adaptation_blocks = 0
        self.missed_silent_blocks = 0
        self.last_coherence = 0.0
        self.last_adapt = False

    def _acquire_delay(self, received, tx_history, rx_at):
        if float(np.sqrt(np.mean(received ** 2))) < 50.0:
            return
        at0 = int(rx_at)
        max_delay = min(self.max_delay_samples, at0)
        y = np.asarray(received, dtype=float)
        y_centered = y - float(np.mean(y))
        reference = tx_history[at0 - max_delay:at0 + len(y)]
        windows = np.lib.stride_tricks.sliding_window_view(
            reference, len(y))
        x_mean = np.mean(windows, axis=1)
        x_centered = windows - x_mean[:, None]
        cross = x_centered @ y_centered
        x_energy = np.einsum("ij,ij->i", x_centered, x_centered)
        y_energy = float(y_centered @ y_centered)
        denom = np.sqrt(x_energy * y_energy)
        correlations = np.divide(cross, denom, out=np.zeros_like(cross),
                                 where=denom > 1.0)
        best_index = int(np.argmax(np.abs(correlations)))
        best_delay = max_delay - best_index
        best_correlation = float(correlations[best_index])
        self.best_acquisition_coherence = max(
            self.best_acquisition_coherence, abs(best_correlation))
        if abs(best_correlation) < self.ACQUISITION_COHERENCE:
            self._candidate_delay = None
            self._candidate_count = 0
            self._candidate_blocks.clear()
            return
        self.coherent_candidate_blocks += 1
        if (self._candidate_delay is not None
                and abs(best_delay - self._candidate_delay) <= 1):
            self._candidate_count += 1
        else:
            self._candidate_delay = best_delay
            self._candidate_count = 1
            self._candidate_blocks.clear()
        self._candidate_blocks.append((y.copy(), at0))
        if self._candidate_count < 3:
            return
        rows = []
        ys = []
        for block, position in self._candidate_blocks[-3:]:
            sample_positions = np.arange(position, position + len(block))
            rows.append(np.column_stack([
                tx_history[sample_positions - best_delay - tap]
                for tap in range(len(self.weights))]))
            ys.append(block)
        design = np.concatenate(rows, axis=0)
        y_fit = np.concatenate(ys)
        coefficients, *_ = np.linalg.lstsq(design, y_fit, rcond=None)
        gain = float(coefficients[0])
        self._training_pcm = y_fit.copy()
        self._training_at = np.concatenate([
            np.arange(position, position + len(block))
            for block, position in self._candidate_blocks[-3:]])
        self.delay_samples = best_delay
        self.delay_fit_correlation = best_correlation
        self.delay_fit_gain = gain
        self.weights[:] = coefficients
        self.training_samples = len(y_fit)
        self.trained = True

    def _coherence(self, received, reference):
        y = np.asarray(received, dtype=float).reshape(-1)
        x = np.asarray(reference, dtype=float).reshape(-1)
        y = y - float(np.mean(y))
        x = x - float(np.mean(x))
        denom = float(np.linalg.norm(y) * np.linalg.norm(x))
        if denom <= 1.0:
            return 0.0
        return float(x @ y) / denom

    def record_oracle_label(self, caller_silent):
        """Score only; this label never participates in delay/gate decisions."""
        if caller_silent:
            self.oracle_silent_blocks += 1
            if not self.last_adapt:
                self.missed_silent_blocks += 1
        else:
            self.oracle_active_blocks += 1
            if self.last_adapt:
                self.false_adaptation_blocks += 1

    def process(self, received, tx_history, *, rx_at):
        y = np.asarray(received, dtype=float).reshape(-1)
        tx_history = np.asarray(tx_history, dtype=float).reshape(-1)
        if self.delay_samples is None:
            self._acquire_delay(y, tx_history, rx_at)
        if self.delay_samples is None:
            self.last_adapt = False
            return y.copy()
        positions = np.arange(rx_at, rx_at + len(y), dtype=np.int64)
        source = positions - self.delay_samples
        x = np.zeros(len(y), dtype=float)
        valid = (source >= 0) & (source < len(tx_history))
        x[valid] = tx_history[source[valid]]
        coherence = self._coherence(y, x)
        self.last_coherence = coherence
        adapt = abs(coherence) >= self.ADAPTATION_COHERENCE
        self.last_adapt = bool(adapt)
        if adapt:
            self.adaptation_blocks += 1
        padded = np.concatenate((self.history, x))
        out = np.empty_like(y)
        for n, sample in enumerate(y):
            vector = padded[n:n + len(self.weights)][::-1]
            estimate = float(self.weights @ vector) if self.trained else 0.0
            error = float(sample - estimate)
            out[n] = error
            if adapt:
                self.training_samples += 1
                self.weights += (self.mu * error / (float(vector @ vector) + 1.0)) * vector
                self.trained = float(self.weights @ self.weights) > 1e-8
        if len(self.history):
            self.history[:] = padded[-len(self.history):]
        return out


class _PairedTwoStageEchoExperiment:
    """Test-only delay-seed then guarded Phase-3 FIR fit/cancel experiment."""

    def __init__(self, *, phase2_samples=5600, phase3_samples=6400,
                 max_delay_samples=1600, taps=16, noise_std=0.0, seed=1):
        self.phase2_samples = int(phase2_samples)
        self.phase3_samples = int(phase3_samples)
        self.max_delay_samples = int(max_delay_samples)
        self.taps = int(taps)
        self.noise_std = float(noise_std)
        self.rng = np.random.default_rng(seed)
        self._p2_start = None
        self._p2_rx = []
        self.delay_samples = None
        self.phase2_gain = None
        self.delay_peak = 0.0
        self.delay_competitor = 0.0
        self.phase3_sessions = 0
        self._phase3_obj = None
        self.phase3_start = None
        self._p3_rx = []
        self.weights = None
        self.gain = None
        self.fit_correlation = 0.0
        self.fit_improvement = 0.0
        self.reference_periodicity = 0.0
        self.cancel_start = None
        self.noise_added = 0
        self.active = False
        self.cancelled_samples = 0

    def _fit_delay(self, tx_history):
        start = int(self._p2_start)
        if start < self.max_delay_samples:
            raise ValueError(
                "Phase-2 capture starts before the full delay search window")
        y = np.concatenate(self._p2_rx)
        positions = np.arange(start, start + len(y), dtype=np.int64)
        scores = np.zeros(self.max_delay_samples + 1, dtype=float)
        x_energy = np.zeros_like(scores)
        for delay in range(len(scores)):
            x = np.asarray(tx_history[positions - delay], dtype=float)
            scores[delay] = float(x @ y)
            x_energy[delay] = float(x @ x)
        denom = np.sqrt(x_energy * float(y @ y))
        corr = np.divide(scores, denom, out=np.zeros_like(scores),
                         where=denom > 1.0)
        order = np.argsort(np.abs(corr))[::-1]
        best = int(order[0])
        competitor = next((int(i) for i in order[1:]
                           if abs(int(i) - best) > 16), int(order[1]))
        self.delay_samples = best
        best_reference = np.asarray(tx_history[positions - best], dtype=float)
        self.phase2_gain = float(best_reference @ y) / max(
            float(best_reference @ best_reference), 1.0)
        self.delay_peak = float(abs(corr[best]))
        self.delay_competitor = float(abs(corr[competitor]))

    def _fit_phase3(self, tx_history):
        start = int(self.phase3_start)
        y = np.concatenate(self._p3_rx)
        positions = np.arange(start, start + len(y), dtype=np.int64)
        design = np.column_stack([
            tx_history[positions - self.delay_samples - tap]
            for tap in range(self.taps)])
        weights, *_ = np.linalg.lstsq(design, y, rcond=None)
        predicted = design @ weights
        residual = y - predicted
        denom = float(np.linalg.norm(predicted) * np.linalg.norm(y))
        self.fit_correlation = (float(predicted @ y) / denom
                                if denom > 1.0 else 0.0)
        self.fit_improvement = (
            1.0 - float(residual @ residual) / max(float(y @ y), 1.0))
        self.gain = float(weights[0])
        self.weights = weights
        self.cancel_start = start + len(y)
        self.reference_periodicity = self._reference_periodicity(
            design[:, 0])

    @staticmethod
    def _reference_periodicity(reference, min_lag=16, max_lag=1600):
        x = np.asarray(reference, dtype=float).reshape(-1)
        x = x - float(np.mean(x))
        energy = float(x @ x)
        if energy <= 1.0:
            return 1.0
        best = 0.0
        for lag in range(int(min_lag), min(int(max_lag), len(x) - 1) + 1):
            left, right = x[:-lag], x[lag:]
            den = float(np.linalg.norm(left) * np.linalg.norm(right))
            if den > 1.0:
                best = max(best, abs(float(left @ right)) / den)
        return best

    def __call__(self, start, caller_pcm, _answerer_pcm, tx_history,
                 received_pcm, answerer, caller):
        start = int(start)
        y = np.asarray(received_pcm, dtype=float).copy()
        phase2 = answerer._phase2
        caller_quiet = float(np.sqrt(np.mean(np.asarray(caller_pcm) ** 2))) < 1.0
        if (self.delay_samples is None and caller._phase3_controller is None
                and phase2.state == phase2.TX_PROBE and caller_quiet):
            if self._p2_start is None:
                self._p2_start = start
            expected = self._p2_start + len(self._p2_rx) * len(y)
            if start != expected:
                self._p2_start = start
                self._p2_rx.clear()
            remaining = self.phase2_samples - sum(map(len, self._p2_rx))
            if remaining > 0:
                self._p2_rx.append(y[:remaining].copy())
            if sum(map(len, self._p2_rx)) >= self.phase2_samples:
                self._fit_delay(tx_history)

        phase3_obj = answerer._phase3
        if phase3_obj is not None and phase3_obj is not self._phase3_obj:
            self._phase3_obj = phase3_obj
            self.phase3_sessions += 1
        phase3_state = str(answerer.phase3_state)
        if (self.phase3_sessions >= 2 and self.phase3_start is None
                and phase3_state == "initial-signals"):
            self.phase3_start = start

        if (self.phase3_start is not None and self.phase3_sessions >= 2
                and self.weights is None):
            expected = self.phase3_start + sum(map(len, self._p3_rx))
            if start == expected:
                if self.noise_std:
                    y += self.rng.normal(0.0, self.noise_std, size=len(y))
                    self.noise_added += len(y)
                remaining = self.phase3_samples - sum(map(len, self._p3_rx))
                if remaining > 0:
                    self._p3_rx.append(y[:remaining].copy())
                if sum(map(len, self._p3_rx)) >= self.phase3_samples:
                    self._fit_phase3(tx_history)
            elif start > expected:
                # Keep the fit window contiguous; a gap invalidates the fit.
                self.phase3_start = None
                self._p3_rx.clear()

        if self.phase3_start is not None and self.noise_std and self.weights is not None:
            if self.cancel_start is not None and start >= self.cancel_start:
                y += self.rng.normal(0.0, self.noise_std, size=len(y))
                self.noise_added += len(y)

        if (self.weights is not None and self.cancel_start is not None
                and start >= self.cancel_start):
            positions = np.arange(start, start + len(y), dtype=np.int64)
            design = np.column_stack([
                tx_history[positions - self.delay_samples - tap]
                for tap in range(self.taps)])
            # Reject weak/ambiguous fits. The thresholds are exploratory and
            # are deliberately local to this test probe, not production policy.
            active = (abs(self.gain) >= 0.01
                      and self.fit_correlation >= 0.10
                      and self.reference_periodicity < 0.98)
            if active:
                self.active = True
                self.cancelled_samples += len(y)
                y -= design @ self.weights
        return y


def _minimum_phase3_ja_pcm():
    """Build the independently encoded minimum V.90 preamble/TRN/Ja input."""
    descriptor = ja_control._ja_descriptor(7, 19, 31)
    sequence, _, _ = ja_control._preamble_and_trn()
    scrambled, state = ja_control._scramble([1] * (2 * 512))
    trn = []
    for offset in range(0, len(scrambled), 2):
        i1, i2 = scrambled[offset:offset + 2]
        trn.append((1 + 1j) * (1, -1j, -1, 1j)[(2 * i2 + i1) % 4])
    final_rotation = (2 * scrambled[-1] + scrambled[-2]) % 4
    ja, _ = ja_control._differential(descriptor * 3, final_rotation, state)
    symbols = np.concatenate((sequence[:432], np.asarray(trn), ja))
    return ja_control._modulate(symbols)


def _zero_dil_phase3_to_cpt_pcm(law):
    """Build a paired answerer input from minimum Ja through one CPt offer."""
    descriptor = ja_control._ja_descriptor(0, 19, 31)
    sequence, _, _ = ja_control._preamble_and_trn()
    scrambled, state = ja_control._scramble([1] * (2 * 512))
    trn = np.asarray([
        (1 + 1j) * (1, -1j, -1, 1j)[(2 * scrambled[i + 1] + scrambled[i]) % 4]
        for i in range(0, len(scrambled), 2)
    ])
    final_rotation = (2 * scrambled[-1] + scrambled[-2]) % 4
    ja, _ = ja_control._differential(descriptor * 3, final_rotation, state)

    # The answerer receives its first S transition after Jd. With N=0, J′d
    # is followed directly by Phase 4, so there is no DIL or second S event.
    # The synthetic answerer finishes its current Jd and emits J-prime 93T
    # after S onset. Match the caller's reactive response: the phase change
    # starts on J-prime, not on a fixed 128T timer.
    s_pair = np.concatenate((v34_tx.s_signal(96),
                             v34_tx.s_signal(16, True)))
    cpt_bits = _frame(
        kind="cpt", drn=7, selectors=(0, 1, 2, 0, 1, 2),
        tx_masks=(tuple(range(128)),) * 3,
        law=int(law == "pcma"), ack=0, spectral_sign_bits=0)
    cpt_decisions = _encode_symbols(cpt_bits * 3, constellation=4)
    cpt = np.asarray([
        v34_tx.P0 * v34_tx.ROT_CW[rotation]
        for rotation, _quarter in cpt_decisions
    ])
    symbols = np.concatenate((
        sequence[:432], trn, ja,
        np.zeros(int(round(1.0 * ja_control.BAUD))),
        s_pair,
        # Wait beyond the answerer's required 192T Ri before starting CPt.
        np.zeros(640), cpt,
    ))
    return ja_control._modulate(symbols)


def _zero_dil_answerer_payload_waveform(law, *, extra_s_delay_ms=0):
    """Build one paired Ja-to-LAPM analog-upstream waveform for the engine."""
    baud = ja_control.BAUD
    carrier = ja_control.CARRIER
    if (isinstance(extra_s_delay_ms, bool)
            or not isinstance(extra_s_delay_ms, int)
            or extra_s_delay_ms < 0):
        raise ValueError("extra S delay must be a nonnegative integer in ms")
    descriptor = ja_control._ja_descriptor(0, 19, 31)
    sequence, _, _ = ja_control._preamble_and_trn()
    scrambled, gpa_state = ja_control._scramble([1] * (2 * 512))
    trn = np.asarray([
        (1 + 1j) * (1, -1j, -1, 1j)[(2 * scrambled[i + 1] + scrambled[i]) % 4]
        for i in range(0, len(scrambled), 2)
    ])
    final_rotation = (2 * scrambled[-1] + scrambled[-2]) % 4
    ja, _ = ja_control._differential(descriptor * 3, final_rotation, gpa_state)

    common = dict(drn=7, selectors=(0, 1, 2, 0, 1, 2),
                  tx_masks=(tuple(range(128)),) * 3,
                  spectral_sign_bits=0)
    law_bit = int(law == "pcma")
    cpt_bits = _frame(kind="cpt", ack=0, law=law_bit, **common)
    cp_bits = _frame(kind="cp", ack=0, law=law_bit, **common)
    scr_symbols = int(round(0.75 * baud))
    (cpt_run, scr_run, cp_run, e_run) = (
        _encode_continuous_4point_segments(
            [cpt_bits * 3, [1] * (2 * scr_symbols), cp_bits * 12,
             [1] * 20]))

    cp_profile = parse_cp(cp_bits)
    upstream_profile = select_analog_upstream_profile(
        cp_profile, answerer_mp_profile(baud), baud=baud, carrier=carrier)
    upstream_rx = V90UpstreamDataReceiver(upstream_profile, remote_role="answer")
    upstream_tx = v34_data.DataTx(
        upstream_profile.rate, int(round(upstream_profile.baud)),
        expanded=upstream_profile.shaping, precoder=upstream_profile.precoder,
        nonlinear=upstream_profile.nonlinear, trellis=upstream_profile.trellis)
    (payload_bits, caller_payload, answerer_payload, up_bits_per_slice,
     down_bits_per_slice) = _paired_lapm_caller_bits(
        upstream_profile.rate, cp_profile, law, upstream_tx.bits_per_data_frame)
    scrambler = v34_tx.Scrambler("answer")
    b1_bits = [scrambler.step(1)
               for _ in range(upstream_tx.bits_per_data_frame)]
    if b1_bits != upstream_rx.receiver.expected:
        raise AssertionError("paired caller B1 does not match production receiver")
    b1_run = np.asarray(upstream_tx.data_frame(b1_bits), dtype=np.complex128)
    payload_run = []
    for offset in range(0, len(payload_bits), upstream_tx.bits_per_data_frame):
        frame = payload_bits[offset:offset + upstream_tx.bits_per_data_frame]
        payload_run.extend(upstream_tx.data_frame(
            [scrambler.step(bit) for bit in frame]))
    payload_run = np.asarray(payload_run, dtype=np.complex128)
    data_run = np.concatenate((b1_run, payload_run))
    data_run *= np.sqrt(2.0 / np.mean(np.abs(data_run) ** 2))
    b1_run, payload_run = data_run[:len(b1_run)], data_run[len(b1_run):]

    # The synthetic answerer finishes its current Jd and emits J-prime 93T
    # after S onset. Match the caller's reactive response: the phase change
    # starts on J-prime, not on a fixed 128T timer.
    s_pair = np.concatenate((v34_tx.s_signal(96),
                             v34_tx.s_signal(16, True)))
    # This knob models extra caller response latency after Ja, not transport
    # delay. Keep it isolated to a paired boundary test of the answerer's
    # Phase-3 S watchdog; it is not a production timer recommendation.
    pre_s_silence_symbols = int(round(baud)) + int(round(
        extra_s_delay_ms * baud / 1000.0))
    symbols = np.concatenate((
        sequence[:432], trn, ja,
        np.zeros(pre_s_silence_symbols), s_pair,
        np.zeros(640), cpt_run, scr_run, cp_run, e_run,
        b1_run, payload_run, np.zeros(32),
    ))
    waveform = v34_tx.qam_modulate(
        symbols, baud, carrier, .1, beta=.15, span=8, power=2.0)
    waveform = v34_g711.roundtrip(waveform * 32768.0, law) / 32768.0
    return (waveform, caller_payload, answerer_payload, len(payload_bits),
            up_bits_per_slice, down_bits_per_slice)


def _start_to_tone_b_reversal_wait(*, late_l2_echo_policy="fail",
                                   late_l2_echo_max_samples=0):
    c = V90AnalogueCallerPhase2(
        late_l2_echo_policy=late_l2_echo_policy,
        late_l2_echo_max_samples=late_l2_echo_max_samples)
    c.start(0)
    c.feed("info0a_complete", 392)
    c.feed("info0d", 500, payload={"ack_info0a": 0})
    c.feed("tone_b", 900)
    assert c.state == c.WAIT_TONE_B_REVERSAL
    return c


class AnaloguePhase2SignalRxTests(unittest.TestCase):
    def test_g711_tone_b_reversal_at_live_boundary_under_both_laws(self):
        # The local SIP failures timed out after the caller reversed Tone A at
        # sample 23,413. Verify the phase detector itself can see a Tone B
        # reversal at that exact sample after either G.711 decode path.
        sample_count = 36_000
        reversal_at = 23_413
        t = np.arange(sample_count)
        tone = 900.0 * np.cos(2 * np.pi * 1200.0 * t / 8000.0)
        tone *= np.where(t < reversal_at, 1.0, -1.0)

        for law, encoder, decoder in (
                ("pcmu", linear_to_ulaw, ulaw_to_linear),
                ("pcma", linear_to_alaw, alaw_to_linear)):
            with self.subTest(law=law):
                encoded = np.fromiter(
                    (encoder(int(round(value))) for value in tone),
                    dtype=np.uint8, count=sample_count)
                decoded = np.fromiter(
                    (decoder(int(value)) for value in encoded),
                    dtype=np.float64, count=sample_count)
                receiver = V90AnaloguePhase2SignalRx()
                events = []
                for start in range(0, sample_count, 160):
                    events.extend(receiver.feed(
                        decoded[start:start + 160], rx_at=start,
                        expect="tone_b"))
                reversals = [event.at for event in events
                             if event.kind == "tone_b_reversal"]
                self.assertEqual(reversals, [reversal_at])


class AnalogueCallerPhase2RecoveryTests(unittest.TestCase):
    def test_paired_v90_dil_accepts_only_pcmu_zero_sign_equivalence(self):
        """The real paired Phase 2/3 path survives observable PCMU zero."""
        for law in ("pcmu", "pcma"):
            with self.subTest(law=law):
                caller, answerer, rtp = self._run_production_paired_phase2(
                    law, echo_delay_samples=0, echo_gain=0.0,
                    phase2_late_l2_echo_policy="advance-after-window",
                    peer_info0d_fields={
                        "rate3429": 1, "v90_3429_upstream": 1,
                    },
                    peer_info1d_blocks=[{}, {}, {}, {}, {},
                                        {"hi_carrier": 0, "max_rate": 12}],
                    continue_v34_fallback=True,
                    v34_fallback_hold_samples=4 * 8000,
                    paired_duration_samples=25 * 8000)
                try:
                    self.assertTrue(any(name == "phase3-ready"
                                        for name, _ in caller.events),
                                    caller.events[-12:])
                    self.assertNotIn("failed", caller.phase3_state)
                    self.assertEqual(caller.phase3_state, "phase4-ready")
                    self.assertEqual(answerer.phase3_state, "phase4-ready")
                    expected = caller._phase3_expected_dil
                    received = bytes(caller._phase3_received_dil)
                    self.assertEqual(len(received), len(expected))
                    self.assertTrue(v90.dil_codewords_match(
                        expected, received, law))
                    if law == "pcmu":
                        self.assertNotEqual(received, expected)
                        self.assertTrue(all(
                            want == got or (want, got) in (
                                (0x7f, 0xff), (0xff, 0x7f))
                            for want, got in zip(expected, received)))
                    else:
                        self.assertEqual(received, expected)
                finally:
                    rtp.stop()
                    rtp._sock.close()
                    caller.close()
                    answerer.close()

    def test_advance_after_echo_window_reverses_at_v90_600ms_plus_rtde(self):
        caller = _start_to_tone_b_reversal_wait(
            late_l2_echo_policy="advance-after-window")
        caller.feed("tone_b_reversal", 2020)
        rtde = caller.rtde_samples
        l2_start = caller._local_l2_start_at
        self.assertEqual(rtde, 800)

        # Model Tone B returned after local L2 has begun, with no local L2
        # echo. Once the echo window ends, the recovery path sends Tone A for
        # 50 ms before reversing it. With L2 lasting 550 ms + RTDE, this puts
        # the reversal at L2-start + 600 ms + RTDE (V.90 §9.2.2.2.3).
        caller.feed("tone_b_after_probes", l2_start + 1000)
        echo_window_end = l2_start + 550 * 8 + rtde
        actions = caller.tick(echo_window_end)
        reversal = next(action.at for action in actions
                        if action.signal == "reverse-tone-a")
        self.assertEqual(reversal, l2_start + 600 * 8 + rtde)
        self.assertEqual(reversal - echo_window_end, 50 * 8)

    def test_info1d_timeout_opt_in_initiates_one_v90_retrain(self):
        info0d = {"rate3429": 1, "v90_3429_upstream": 0}
        caller = V90AnalogueCallerPhase2(info1d_timeout_retrain=True)
        caller.start(0)
        caller.feed("info0a_complete", 392)
        caller.feed("info0d", 500, payload=info0d)
        caller.feed("tone_b", 900)
        caller.feed("tone_b_reversal", 1220)
        caller.feed("tone_b_after_probes", 4000)
        caller.feed("l2_echo_complete", 4100)
        caller.feed("tone_b_reversal_after_probes", 5000)
        caller.feed("l1_complete", 6280,
                    payload={"receive_sample_at": 6280})
        caller.feed("l2_window_end", 10280,
                    payload={"receive_sample_at": 10280})
        self.assertEqual(caller.state, caller.WAIT_INFO1D)

        deadline = caller._info1d_deadline_at
        actions = caller.tick(deadline)
        tone_a_at = deadline + 70 * 8
        self.assertEqual(caller.state, caller.WAIT_TONE_B)
        self.assertEqual(caller._info1d_retrain_attempts, 1)
        self.assertEqual(caller.info0d, info0d)
        self.assertEqual(
            [(action.at, action.signal) for action in actions],
            [(deadline, "silence"), (tone_a_at, "tone_a"),
             (None, "listen-tone-b")])

        # A qualified Tone B resumes at §9.2.2.1.3; it doesn't send INFO0a.
        tone_b_at = tone_a_at + 50 * 8
        actions = caller.feed("tone_b", tone_b_at, now=tone_b_at)
        self.assertEqual(caller.state, caller.WAIT_TONE_B_REVERSAL)
        self.assertEqual([action.signal for action in actions],
                         ["reverse-tone-a"])
        self.assertEqual(actions[0].at, tone_b_at)
        peer_reversal = tone_b_at + 40 * 8 + 80 * 8
        actions = caller.feed("tone_b_reversal", peer_reversal)
        self.assertEqual(caller.state, caller.TX_LOCAL_PROBES)
        self.assertEqual(caller.rtde_samples, 80 * 8)
        self.assertEqual(caller.l2_samples, caller.l2_base_samples + 80 * 8)
        self.assertFalse(any(action.signal == "info0a" for action in actions))

    def test_info1d_timeout_retrain_second_expiry_hands_off_safely(self):
        caller = V90AnalogueCallerPhase2(info1d_timeout_retrain=True)
        caller._started = True
        caller.state = caller.WAIT_INFO1D
        caller._info1d_retrain_attempts = 1
        caller._info1d_deadline_at = 10000

        actions = caller.tick(10000)

        self.assertEqual(caller.state, caller.FAILED)
        self.assertEqual([action.signal for action in actions],
                         ["handoff-retrain-v90-phase2"])
        self.assertEqual(actions[0].at, 10000)

    def test_v90_retrain_continues_without_repeating_info0a(self):
        info0d = {"rate3429": 1, "v90_3429_upstream": 0}
        caller = V90AnalogueCallerPhase2()

        actions = caller.start_v90_retrain_response(1000, info0d=info0d)
        self.assertEqual(caller.state, caller.WAIT_TONE_B_REVERSAL)
        self.assertEqual(caller.info0d, info0d)
        self.assertEqual(
            [(action.at, action.signal) for action in actions],
            [(1000, "silence"), (1560, "tone_a"),
             (1960, "reverse-tone-a"),
             (None, "listen-tone-b-reversal")])
        self.assertFalse(any(action.signal == "info0a" for action in actions))

        # The peer reverses after the standard 40 ms turnaround and returns
        # after 80 ms RTDE. The resumed controller must use that exchange to
        # size its L2, then continue through the normal remote L1/L2 and INFO1d
        # states without starting a fresh INFO0 exchange.
        peer_reversal = 1960 + 40 * 8 + 80 * 8
        actions = caller.feed("tone_b_reversal", peer_reversal)
        self.assertEqual(caller.rtde_samples, 80 * 8)
        self.assertEqual(caller.l2_samples, caller.l2_base_samples + 80 * 8)
        self.assertEqual(caller.state, caller.TX_LOCAL_PROBES)
        self.assertEqual([action.signal for action in actions], [
            "reverse-tone-a", "l1", "l2", "listen-tone-b"])
        self.assertFalse(any(action.signal == "info0a" for action in actions))

        local_l2_end = caller._local_l2_start_at + caller.l2_samples
        caller.feed("l2_echo_complete", local_l2_end)
        actions = caller.feed("tone_b_after_probes", local_l2_end)
        self.assertEqual(caller.state,
                         caller.WAIT_TONE_B_REVERSAL_AFTER_PROBES)
        caller.feed("tone_b_reversal_after_probes", local_l2_end + 1000)
        l1_end = local_l2_end + 1000 + 160 * 8
        caller.feed("l1_complete", l1_end,
                    payload={"receive_sample_at": l1_end})
        l2_window_end = l1_end + 500 * 8
        actions = caller.feed("l2_window_end", l2_window_end,
                              payload={"receive_sample_at": l2_window_end})
        self.assertEqual(caller.state, caller.WAIT_INFO1D)
        self.assertTrue(any(action.signal == "listen-info1d"
                            for action in actions))
        caller.feed("info1d", l2_window_end + 500,
                    payload={"blocks": [{}] * 6})
        self.assertEqual(caller.state, caller.COMPLETE)

    def test_phase3_retrain_initiation_sends_tone_a_then_resumes_phase2(self):
        info0d = {"rate3429": 1, "v90_3429_upstream": 1}
        caller = V90AnalogueCallerPhase2()
        actions = caller.start_v90_retrain_initiation(1000, info0d=info0d)
        self.assertEqual(caller.state, caller.WAIT_TONE_B)
        self.assertEqual(caller.info0d, info0d)
        self.assertEqual(
            [(action.at, action.signal) for action in actions],
            [(1000, "silence"), (1560, "tone_a"),
             (None, "listen-tone-b")])

        # Tone B arrives after Tone A's required 50 ms. Reversal, returned
        # reversal, ranging, and fresh INFO1d follow the ordinary Phase-2 FSM.
        actions = caller.feed("tone_b", 1960, now=1960)
        self.assertEqual(caller.state, caller.WAIT_TONE_B_REVERSAL)
        self.assertEqual([(action.at, action.signal) for action in actions],
                         [(1960, "reverse-tone-a")])
        actions = caller.feed("tone_b_reversal", 2920)
        self.assertEqual(caller.state, caller.TX_LOCAL_PROBES)
        self.assertEqual(caller.rtde_samples, 80 * 8)
        self.assertEqual([action.signal for action in actions], [
            "reverse-tone-a", "l1", "l2", "listen-tone-b"])
        self.assertFalse(any(action.signal == "info0a" for action in actions))

    def test_phase3_retrain_tone_b_watchdog_hands_off_safely(self):
        caller = V90AnalogueCallerPhase2()
        caller.start_v90_retrain_initiation(
            1000, info0d={"rate3429": 1})
        deadline = caller._deadline_at
        actions = caller.tick(deadline)
        self.assertEqual(caller.state, caller.FAILED)
        self.assertEqual([action.signal for action in actions],
                         ["handoff-retrain-v90-phase2"])
        self.assertEqual(actions[0].at, deadline)

    def test_production_pair_selects_peer_supported_lower_v90_rate(self):
        caller, answerer, rtp = self._run_production_paired_phase2(
            "pcmu", echo_delay_samples=1240 * 8,
            phase2_late_l2_echo_policy="advance-after-window",
            caller_info0a_fields={
                "rate3429": 1, "disallow3429": 1, "lo3200": 1,
            },
            peer_info0d_fields={
                "rate3429": 1, "v90_3429_upstream": 0,
            },
            peer_info1d_blocks=[
                {}, {}, {}, {}, {"hi_carrier": 0, "max_rate": 8},
                {"hi_carrier": 0, "max_rate": 12},
            ])
        try:
            self.assertEqual(caller._controller.state,
                             caller._controller.COMPLETE)
            self.assertEqual(
                caller._profile_info1a_fields["analog_symbol_rate"], 4)
            selected = [value for name, value in caller.events
                        if name == "phase2-rate-selected"]
            self.assertEqual(len(selected), 1)
            self.assertEqual(selected[0]["analog_symbol_rate_index"], 4)
            self.assertEqual(selected[0]["baud"], 3200)
            self.assertIsNotNone(caller._phase3_start_at)
            self.assertEqual(answerer._phase2.state,
                             answerer._phase2.COMPLETE)
        finally:
            rtp.stop()
            rtp._sock.close()
            caller.close()
            answerer.close()

    def test_production_pair_selects_optional_3000_baud_v90_rate(self):
        """Exercise V.90's optional 3000-baud upstream selection end-to-end."""
        for law in ("pcmu", "pcma"):
            with self.subTest(law=law):
                caller, answerer, rtp = self._run_production_paired_phase2(
                    law, echo_delay_samples=1240 * 8,
                    phase2_late_l2_echo_policy="advance-after-window",
                    caller_info0a_fields={
                        "rate3429": 1, "disallow3429": 1, "lo3000": 1,
                    },
                    peer_info0d_fields={
                        "rate3429": 1, "v90_3429_upstream": 0,
                    },
                    peer_info1d_blocks=[
                        {}, {}, {},
                        {"hi_carrier": 0, "max_rate": 8}, {}, {},
                    ])
                try:
                    self.assertEqual(caller._controller.state,
                                     caller._controller.COMPLETE)
                    self.assertEqual(
                        caller._profile_info1a_fields["analog_symbol_rate"],
                        3)
                    selected = [value for name, value in caller.events
                                if name == "phase2-rate-selected"]
                    self.assertEqual(len(selected), 1)
                    self.assertEqual(
                        selected[0]["analog_symbol_rate_index"], 3)
                    self.assertEqual(selected[0]["baud"], 3000)
                    self.assertIsNotNone(caller._phase3_start_at)
                    self.assertEqual(answerer._phase2.state,
                                     answerer._phase2.COMPLETE)
                finally:
                    rtp.stop()
                    rtp._sock.close()
                    caller.close()
                    answerer.close()

    def test_production_pair_selects_v34_fallback_when_no_v90_rate_is_mutual(self):
        caller, answerer, rtp = self._run_production_paired_phase2(
            "pcmu", echo_delay_samples=1240 * 8,
            phase2_late_l2_echo_policy="advance-after-window",
            peer_info0d_fields={
                "rate3429": 1, "v90_3429_upstream": 0,
            },
            peer_info1d_blocks=[
                {}, {}, {}, {}, {}, {"hi_carrier": 0, "max_rate": 12},
            ], continue_v34_fallback=True,
            v34_fallback_hold_samples=4 * 8000,
            # The 1.24 s echo exercises Phase 2. Left on into V.34 Phase 3 it
            # lands the answerer's own PP on the caller's PP once the caller
            # answers promptly; no PSTN echo is that long.
            cancel_synthetic_echo_after_fallback=True)
        try:
            self.assertEqual(caller._controller.state,
                             caller._controller.COMPLETE)
            self.assertFalse(caller._failed)
            self.assertIsNotNone(caller._v34_fallback_engine)
            self.assertIn(caller._v34_fallback_engine.answerer.stage,
                          {"p4", "data"})
            selected = [value for name, value in caller.events
                        if name == "phase2-v34-fallback-selected"]
            self.assertEqual(len(selected), 1)
            self.assertEqual(selected[0]["symbol_rate_index"], 5)
            self.assertEqual(selected[0]["peer_info0d_v90_3429"], 0)
            self.assertEqual(answerer._phase2.state,
                             answerer._phase2.COMPLETE)
            self.assertIsNotNone(answerer._v34_fallback_engine)
            self.assertTrue(any(name == "phase3-v34-fallback-started"
                                for name, _ in answerer.events))
            prerolls = [details for name, details in answerer.events
                        if name == "phase3-v34-fallback-preroll-replayed"]
            self.assertEqual(len(prerolls), 1)
            self.assertGreater(prerolls[0]["samples"], 0)
            self.assertIn(caller._v34_fallback_engine.answerer.stage,
                          {"p4", "data"})
            self.assertIn(getattr(answerer._v34_fallback_engine.caller,
                                  "stage", None), {"p4_trn", "data"})
            self.assertFalse(answerer._failed)
        finally:
            rtp.stop()
            rtp._sock.close()
            caller.close()
            answerer.close()

    def test_production_v34_fallback_reaches_lapm_and_transfers_payload(self):
        caller_payload = bytes((i * 73 + 19) & 0xFF for i in range(1024))
        answer_payload = bytes((i * 151 + 7) & 0xFF for i in range(1024))
        # These are perturbations to RTP receive-arrival timestamps. The PCM
        # sample clock itself is not resampled, so this is not a line-clock
        # drift qualification.
        for arrival_ppm, jitter_ms in ((0, 1.5), (-100, 1.5),
                                       (100, 1.5), (500, 5.0)):
            with self.subTest(arrival_ppm=arrival_ppm,
                              jitter_ms=jitter_ms):
                caller, answerer, rtp = self._run_production_paired_phase2(
                    "pcmu", echo_delay_samples=1240 * 8,
                    phase2_late_l2_echo_policy="advance-after-window",
                    continue_v34_fallback=True,
                    v34_fallback_hold_samples=4 * 8000,
                    cancel_synthetic_echo_after_fallback=True,
                    fallback_payloads=(caller_payload, answer_payload),
                    arrival_clock_ppm=arrival_ppm,
                    arrival_jitter_ms=jitter_ms,
                    peer_info0d_fields={"rate3429": 1,
                                        "v90_3429_upstream": 0},
                    peer_info1d_blocks=[{}, {}, {}, {}, {},
                                        {"hi_carrier": 0, "max_rate": 12}])
                try:
                    calling = answerer._v34_fallback_engine
                    answering = caller._v34_fallback_engine
                    self.assertIsNotNone(calling)
                    self.assertIsNotNone(answering)
                    self.assertTrue(calling.connected, calling.events[-8:])
                    self.assertTrue(answering.connected, answering.events[-8:])
                    self.assertEqual((calling.bit_rate, answering.bit_rate),
                                     (28800, 28800))
                    self.assertEqual((calling.v42_mode, answering.v42_mode),
                                     ("lapm", "lapm"))
                    self.assertEqual(caller._test_fallback_payload_rx,
                                     answer_payload)
                    self.assertEqual(answerer._test_fallback_payload_rx,
                                     caller_payload)
                finally:
                    rtp.stop()
                    rtp._sock.close()
                    caller.close()
                    answerer.close()

    def test_late_hybrid_echo_phase2_fallback_reaches_exact_lapm_payload(self):
        """Carry a delayed-return Phase 2 case through V.34 and LAPM."""
        caller_payload = bytes((i * 73 + 19) & 0xFF for i in range(2048))
        answer_payload = bytes((i * 151 + 7) & 0xFF for i in range(2048))
        for law in ("pcmu", "pcma"):
            with self.subTest(law=law):
                caller, answerer, rtp = self._run_production_paired_phase2(
                    law, echo_delay_samples=1610 * 8,
                    phase2_late_l2_echo_policy="advance-after-window",
                    continue_v34_fallback=True,
                    v34_fallback_hold_samples=4 * 8000,
                    cancel_synthetic_echo_after_fallback=True,
                    fallback_payloads=(caller_payload, answer_payload),
                    fallback_wait_for_lapm_ack=True,
                    paired_duration_samples=30 * 8000,
                    peer_info0d_fields={"rate3429": 1,
                                        "v90_3429_upstream": 0},
                    peer_info1d_blocks=[{}, {}, {}, {}, {},
                                        {"hi_carrier": 0, "max_rate": 12}])
                try:
                    local = caller._v34_fallback_engine
                    remote = answerer._v34_fallback_engine
                    self.assertTrue(local.connected, local.events[-10:])
                    self.assertTrue(remote.connected, remote.events[-10:])
                    self.assertEqual((local.bit_rate, remote.bit_rate),
                                     (28800, 28800))
                    self.assertEqual((local.v42_mode, remote.v42_mode),
                                     ("lapm", "lapm"))
                    self.assertEqual(caller._test_fallback_payload_rx,
                                     answer_payload)
                    self.assertEqual(answerer._test_fallback_payload_rx,
                                     caller_payload)
                    self.assertTrue(
                        caller._controller.l2_echo_advanced_without_echo)
                    self.assertEqual(local._v42.stats["fcs_errors"], 0)
                    self.assertEqual(remote._v42.stats["fcs_errors"], 0)
                finally:
                    rtp.stop()
                    rtp._sock.close()
                    caller.close()
                    answerer.close()

    def test_captured_peer_call1_profile_with_late_return_reaches_exact_lapm(self):
        """Apply call 1's decoded INFO/MP profile to a paired delayed path."""
        caller_payload = bytes((i * 37 + 11) & 0xFF for i in range(2048))
        answer_payload = bytes((i * 91 + 3) & 0xFF for i in range(2048))
        caller, answerer, rtp = self._run_production_paired_phase2(
            "pcmu", echo_delay_samples=1610 * 8,
            phase2_late_l2_echo_policy="advance-after-window",
            continue_v34_fallback=True,
            v34_fallback_hold_samples=4 * 8000,
            cancel_synthetic_echo_after_fallback=True,
            fallback_payloads=(caller_payload, answer_payload),
            fallback_wait_for_lapm_ack=True,
            paired_duration_samples=35 * 8000,
            peer_v34_mp_request={"nonlinear": 1, "shaping": 1,
                                 "trellis": 0,
                                 "precoder": [1052, 1075, 276, 932,
                                              914, 520]},
            peer_info0d_fields={
                "rate2743": 1, "rate2800": 1, "rate3429": 1,
                "lo3000": 1, "hi3000": 1, "lo3200": 1, "hi3200": 1,
                "disallow3429": 1, "power_reduction": 1, "max_asym": 0,
                "cme": 0, "pts1664": 1, "reserved": 0,
                "ack_info0a": 0, "nominal_power": 4, "max_power": 23,
                "power_at_codec": 1, "alaw": 0,
                "v90_3429_upstream": 0, "reserved_41": 0,
            },
            peer_info1d_blocks=[
                {"hi_carrier": 0, "preemph": 0, "max_rate": 5},
                {"hi_carrier": 0, "preemph": 0, "max_rate": 0},
                {"hi_carrier": 0, "preemph": 0, "max_rate": 0},
                {"hi_carrier": 0, "preemph": 0, "max_rate": 0},
                {"hi_carrier": 0, "preemph": 0, "max_rate": 0},
                {"hi_carrier": 0, "preemph": 0, "max_rate": 9},
            ])
        try:
            local = caller._v34_fallback_engine
            remote = answerer._v34_fallback_engine
            self.assertTrue(local.connected, local.events[-10:])
            self.assertTrue(remote.connected, remote.events[-10:])
            self.assertEqual((local.bit_rate, remote.bit_rate),
                             (21600, 21600))
            self.assertEqual((local.v42_mode, remote.v42_mode),
                             ("lapm", "lapm"))
            self.assertEqual(caller._test_fallback_payload_rx, answer_payload)
            self.assertEqual(answerer._test_fallback_payload_rx, caller_payload)
            self.assertTrue(caller._controller.l2_echo_advanced_without_echo)
            self.assertEqual(local._v42.stats["fcs_errors"], 0)
            self.assertEqual(remote._v42.stats["fcs_errors"], 0)
        finally:
            rtp.stop()
            rtp._sock.close()
            caller.close()
            answerer.close()

    def test_production_21600_v34_fallback_reaches_lapm_with_exact_payload(self):
        """Exercise the 21.6 kb/s profile selected by captured peer call 1."""
        caller_payload = bytes((i * 73 + 19) & 0xFF for i in range(2048))
        answer_payload = bytes((i * 151 + 7) & 0xFF for i in range(2048))
        caller, answerer, rtp = self._run_production_paired_phase2(
            "pcmu", echo_delay_samples=1240 * 8,
            phase2_late_l2_echo_policy="advance-after-window",
            continue_v34_fallback=True,
            v34_fallback_hold_samples=4 * 8000,
            cancel_synthetic_echo_after_fallback=True,
            fallback_payloads=(caller_payload, answer_payload),
            # Call 1's decoded peer capability/profile values. V.90 Table 7
            # requires the digital modem to transmit reserved bits 26:27 as
            # zero; the capture decoded them as 2, so the paired peer emits
            # the standard value while preserving the negotiated capabilities.
            # The clear V.90 3429-upstream bit forces V.34 fallback despite
            # the 3429-baud INFO1d block.
            peer_info0d_fields={
                "rate2743": 1, "rate2800": 1, "rate3429": 1,
                "lo3000": 1, "hi3000": 1, "lo3200": 1, "hi3200": 1,
                "disallow3429": 1, "power_reduction": 1, "max_asym": 0,
                "cme": 0, "pts1664": 1, "reserved": 0,
                "ack_info0a": 0, "nominal_power": 4, "max_power": 23,
                "power_at_codec": 1, "alaw": 0,
                "v90_3429_upstream": 0, "reserved_41": 0,
            },
            peer_info1d_blocks=[
                {"hi_carrier": 0, "preemph": 0, "max_rate": 5},
                {"hi_carrier": 0, "preemph": 0, "max_rate": 0},
                {"hi_carrier": 0, "preemph": 0, "max_rate": 0},
                {"hi_carrier": 0, "preemph": 0, "max_rate": 0},
                {"hi_carrier": 0, "preemph": 0, "max_rate": 0},
                {"hi_carrier": 0, "preemph": 0, "max_rate": 9},
            ])
        try:
            calling = answerer._v34_fallback_engine
            answering = caller._v34_fallback_engine
            self.assertIsNotNone(calling)
            self.assertIsNotNone(answering)
            self.assertTrue(calling.connected, calling.events[-8:])
            self.assertTrue(answering.connected, answering.events[-8:])
            self.assertEqual((calling.bit_rate, answering.bit_rate),
                             (21600, 21600))
            self.assertEqual((calling.v42_mode, answering.v42_mode),
                             ("lapm", "lapm"))
            self.assertEqual(caller._test_fallback_payload_rx, answer_payload)
            self.assertEqual(answerer._test_fallback_payload_rx, caller_payload)
        finally:
            rtp.stop()
            rtp._sock.close()
            caller.close()
            answerer.close()





    def test_captured_peer_v34_mp_precoder_requests_preserve_exact_fallback_payload(self):
        """Exercise captured captured peer V.34 MP precoders through exact fallback payloads."""
        caller_payload = bytes((i * 73 + 19) & 0xFF for i in range(1024))
        answer_payload = bytes((i * 151 + 7) & 0xFF for i in range(1024))
        profiles = {
            "call-1": (3758, -2532, 2243, -582, 3010, 262),
            "call-3": (1080, 907, 405, -48, -511, 689),
            # Fresh call-budget captures from 2026-11-06. Both negotiated
            # 21.6 kbit/s; call 1 heard the peer E, while call 2 did not.
            # Keeping both MP vectors in the paired regression separates
            # profile support from the live E/B1 divergence.
            "budget-call-1-2026-11-06": (1052, 1075, 276, 932, 914, 520),
            "budget-call-2-2026-11-06": (5084, -1324, 2228, -1061, 2246, -372),
            # captured peer 2026-11-07 calls 8 and 10: both pass B1, but only
            # call 8 gets a decoded LAPM response. These requests describe
            # the far caller's receive profile and therefore configure this
            # endpoint's TX, not its RX.
            "captured_peer-call-8-2026-11-07": (1786, -109, -66, -236, 363, 139),
            "captured_peer-call-10-2026-11-07": (2348, 183, -532, -142, 612, 319),
        }
        for capture, coefficients in profiles.items():
            with self.subTest(capture=capture):
                max_rate = 5 if capture.startswith("captured_peer-call-") else 9
                caller, answerer, rtp = self._run_production_paired_phase2(
                    "pcmu", echo_delay_samples=1240 * 8,
                    phase2_late_l2_echo_policy="advance-after-window",
                    accept_early_peer_recovery=True,
                    continue_v34_fallback=True,
                    v34_fallback_hold_samples=4 * 8000,
                    cancel_synthetic_echo_after_fallback=True,
                    fallback_payloads=(caller_payload, answer_payload),
                    peer_v34_mp_request={
                        "nonlinear": 1, "shaping": 1, "trellis": 0,
                        "precoder": coefficients,
                    },
                    peer_info0d_fields={"rate3429": 1,
                                        "v90_3429_upstream": 0},
                    peer_info1d_blocks=[{}, {}, {}, {}, {},
                                        {"hi_carrier": 0,
                                         "max_rate": max_rate}])
                try:
                    remote_caller = answerer._v34_fallback_engine.caller
                    local_answerer = caller._v34_fallback_engine.answerer
                    self.assertEqual(remote_caller._local_mp["precoder"],
                                     list(coefficients))
                    self.assertEqual(local_answerer.negotiated.tx.precoder,
                                     coefficients)
                    self.assertTrue(local_answerer.negotiated.tx.nonlinear)
                    self.assertTrue(local_answerer.negotiated.tx.shaping)
                    self.assertEqual(local_answerer.negotiated.rx.precoder,
                                     (0, 0, 0, 0, 0, 0))
                    self.assertFalse(local_answerer.negotiated.rx.nonlinear)
                    self.assertFalse(local_answerer.negotiated.rx.shaping)
                    self.assertEqual(local_answerer.negotiated.tx.rate,
                                     max_rate * 2400)
                    self.assertTrue(caller._v34_fallback_engine.connected)
                    self.assertTrue(answerer._v34_fallback_engine.connected)
                    self.assertEqual(
                        (caller._v34_fallback_engine.v42_mode,
                         answerer._v34_fallback_engine.v42_mode),
                        ("lapm", "lapm"))
                    self.assertEqual(caller._test_fallback_payload_rx,
                                     answer_payload)
                    self.assertEqual(answerer._test_fallback_payload_rx,
                                     caller_payload)
                finally:
                    rtp.stop()
                    rtp._sock.close()
                    caller.close()
                    answerer.close()


    def test_phase4_remote_audio_silence_after_local_e_blocks_b1(self):
        """A source-side 68 ms mute after local E makes B1 fail.

        The clean captured-profile control is
        test_captured_peer_v34_mp_precoder_requests_preserve_exact_fallback_payload.
        Here the only channel change is 544 samples of remote-to-local PCMU
        silence at the local E edge; packets and the opposite direction remain
        present. The local receiver does not accept B1. This is a
        fault-sensitivity test, not a proposed default.
        """
        state = {"start": None, "muted_samples": 0}

        def mute_return_audio_after_e(local, peer, sample,
                                      caller_rx_pcm, answerer_rx_pcm):
            fallback = local._v34_fallback_engine
            if fallback is None:
                return
            if fallback.answerer.e_sent and state["start"] is None:
                state["start"] = int(sample)
            if state["start"] is None:
                return
            lo = max(0, state["start"] - sample)
            hi = min(160, state["start"] + 544 - sample)
            if hi > lo:
                caller_rx_pcm[lo:hi] = 0.0
                state["muted_samples"] += hi - lo

        caller_payload = bytes((i * 73 + 19) & 0xFF for i in range(1024))
        answer_payload = bytes((i * 151 + 7) & 0xFF for i in range(1024))
        with mock.patch.dict("os.environ", {"V34_INPUT_POWER_GUARD": "0"}):
            caller, answerer, rtp = self._run_production_paired_phase2(
                "pcmu", echo_delay_samples=1240 * 8,
                phase2_late_l2_echo_policy="advance-after-window",
                accept_early_peer_recovery=True,
                continue_v34_fallback=True,
                v34_fallback_hold_samples=4 * 8000,
                cancel_synthetic_echo_after_fallback=True,
                fallback_payloads=(caller_payload, answer_payload),
                fallback_wait_for_lapm_ack=True,
                paired_duration_samples=45 * 8000,
                peer_v34_mp_request={
                    "nonlinear": 1, "shaping": 1, "trellis": 0,
                    "precoder": [1052, 1075, 276, 932, 914, 520],
                },
                peer_info0d_fields={"rate3429": 1,
                                    "v90_3429_upstream": 0},
                peer_info1d_blocks=[{}, {}, {}, {}, {},
                                    {"hi_carrier": 0, "max_rate": 9}],
                fallback_rx_audio_hook=mute_return_audio_after_e)
        try:
            local = caller._v34_fallback_engine
            peer = answerer._v34_fallback_engine
            self.assertIsNotNone(state["start"])
            self.assertEqual(state["muted_samples"], 544)
            self.assertIsNotNone(local)
            self.assertIsNotNone(peer)
            self.assertFalse(local.connected)
            # The mute blocks B1 either way: the local receiver finds no B1
            # marker, or finds it and rejects it. Which one depends on how far
            # tracking has got when the mute starts.
            if local.answerer._rx_b1 is not None:
                self.assertEqual(local.answerer.failed_reason,
                                 "B1 receive qualification failed")
            self.assertTrue(any("E sent, B1 sent" in message
                                for _, message in peer.events))
            # Whether the peer then qualifies B1 depends on how much of the
            # local B1 went out before the local receiver failed and stopped
            # transmitting; that is timing, not what this test checks.
            self.assertEqual(local.v42_mode, "off")
            self.assertEqual(len(caller._test_fallback_payload_rx), 0)
            self.assertEqual(len(answerer._test_fallback_payload_rx), 0)
        finally:
            rtp.stop()
            rtp._sock.close()
            caller.close()
            answerer.close()



    def _run_paired_v90_gap_payload_case(self, gap_kind):
        """Run a clean control, then erase one peer-to-local media interval.

        The interval is anchored to the clean peer's transition into data so
        the two expectedFailure tests exercise repeatable MP-overlap and B1
        placements rather than depending on an absolute call-start offset.
        """
        caller_payload = bytes((i * 37 + 11) & 0xFF for i in range(512))
        answer_payload = bytes((i * 91 + 3) & 0xFF for i in range(512))
        gap_samples = 904
        coefficients = [1786, -109, -66, -236, 363, 139]

        def run_case(gap_start=None):
            state = {"peer_b1_start": None, "muted": 0,
                     "gap_phase": None, "gap_peer_stage": None,
                     "gap_e_sent": None}

            def tick_hook(local, peer, sample):
                fallback = peer._v34_fallback_engine
                if (fallback is not None and fallback.caller.stage == "data"
                        and state["peer_b1_start"] is None):
                    state["peer_b1_start"] = int(sample)

            def rx_hook(local, peer, sample, local_rx, peer_rx):
                if gap_start is None:
                    return
                gap_end = int(gap_start) + gap_samples
                lo = max(0, int(gap_start) - int(sample))
                hi = min(160, gap_end - int(sample))
                if hi <= lo:
                    return
                fallback = local._v34_fallback_engine
                answerer = fallback.answerer if fallback is not None else None
                remote = peer._v34_fallback_engine
                remote_caller = remote.caller if remote is not None else None
                if state["gap_phase"] is None:
                    state["gap_phase"] = getattr(answerer, "_p4_state", None)
                    state["gap_peer_stage"] = getattr(remote_caller, "stage", None)
                    state["gap_e_sent"] = getattr(answerer, "e_sent", None)
                local_rx[lo:hi] = 0.0
                state["muted"] += hi - lo

            pair = self._run_production_paired_phase2(
                "pcmu", echo_delay_samples=1240 * 8,
                phase2_late_l2_echo_policy="advance-after-window",
                accept_early_peer_recovery=True,
                continue_v34_fallback=True,
                v34_fallback_hold_samples=4 * 8000,
                cancel_synthetic_echo_after_fallback=True,
                fallback_payloads=(caller_payload, answer_payload),
                fallback_wait_for_lapm_ack=True,
                fallback_tick_hook=tick_hook,
                fallback_rx_audio_hook=rx_hook,
                paired_duration_samples=35 * 8000,
                peer_v34_mp_request={
                    "nonlinear": 1, "shaping": 1, "trellis": 0,
                    "precoder": coefficients,
                },
                peer_info0d_fields={"rate3429": 1,
                                    "v90_3429_upstream": 0},
                peer_info1d_blocks=[{}, {}, {}, {}, {},
                                    {"hi_carrier": 0, "max_rate": 5}])
            return pair, state

        control, control_state = run_case()
        try:
            self.assertIsNotNone(control_state["peer_b1_start"], repr(control_state))
            self.assertTrue(control[0]._v34_fallback_engine.connected)
            self.assertTrue(control[1]._v34_fallback_engine.connected)
            self.assertEqual(control[0]._test_fallback_payload_rx, answer_payload)
            self.assertEqual(control[1]._test_fallback_payload_rx, caller_payload)
            self.assertEqual((control[0]._v34_fallback_engine.v42_mode,
                              control[1]._v34_fallback_engine.v42_mode),
                             ("lapm", "lapm"))
            b1_start = control_state["peer_b1_start"]
        finally:
            control[2].stop()
            control[2]._sock.close()
            control[0].close()
            control[1].close()

        if gap_kind == "mp-overlap":
            gap_start = b1_start - 640
        elif gap_kind == "b1-erasure":
            gap_start = b1_start
        else:
            raise ValueError(f"unknown V.90 gap kind {gap_kind!r}")
        damaged, state = run_case(gap_start)
        return damaged, state, caller_payload, answer_payload, gap_samples

    @unittest.expectedFailure
    def test_v90_mp_overlap_gap_still_connects_with_exact_lapm_payload(self):
        """A 113 ms media erasure overlapping MP should recover the link."""
        pair, state, caller_payload, answer_payload, gap_samples = (
            self._run_paired_v90_gap_payload_case("mp-overlap"))
        try:
            self.assertEqual(state["muted"], gap_samples)
            self.assertEqual(state["gap_phase"], "mp", repr(state))
            self.assertFalse(state["gap_e_sent"], repr(state))
            self.assertTrue(pair[0]._v34_fallback_engine.connected, repr(state))
            self.assertTrue(pair[1]._v34_fallback_engine.connected, repr(state))
            self.assertEqual(pair[0]._test_fallback_payload_rx, answer_payload)
            self.assertEqual(pair[1]._test_fallback_payload_rx, caller_payload)
        finally:
            pair[2].stop()
            pair[2]._sock.close()
            pair[0].close()
            pair[1].close()


    @unittest.expectedFailure
    def test_v90_b1_erasure_still_connects_with_exact_lapm_payload(self):
        """A 113 ms peer-to-local B1 erasure should recover exact LAPM data."""
        pair, state, caller_payload, answer_payload, gap_samples = (
            self._run_paired_v90_gap_payload_case("b1-erasure"))
        try:
            self.assertEqual(state["muted"], gap_samples)
            self.assertTrue(state["gap_e_sent"], repr(state))
            self.assertTrue(pair[0]._v34_fallback_engine.connected, repr(state))
            self.assertTrue(pair[1]._v34_fallback_engine.connected, repr(state))
            self.assertEqual(pair[0]._test_fallback_payload_rx, answer_payload)
            self.assertEqual(pair[1]._test_fallback_payload_rx, caller_payload)
        finally:
            pair[2].stop()
            pair[2]._sock.close()
            pair[0].close()
            pair[1].close()

    def test_v90_fallback_complementary_lapm_role_assignments(self):
        """Use V.34-assigned LAPM roles on fallback and test the inverse too."""
        caller_payload = bytes((i * 37 + 11) & 0xFF for i in range(8192))
        answer_payload = bytes((i * 91 + 3) & 0xFF for i in range(8192))
        cases = (
            ("V90-fallback-V34-roles", None, (False, True)),
            ("complementary-inversion", ("originator", "answerer"),
             (True, False)),
        )
        for name, overrides, expected_roles in cases:
            with self.subTest(role_pair=name):
                caller, answerer, rtp = self._run_production_paired_phase2(
                    "pcmu", echo_delay_samples=1240 * 8,
                    phase2_late_l2_echo_policy="advance-after-window",
                    accept_early_peer_recovery=True,
                    continue_v34_fallback=True,
                    v34_fallback_hold_samples=4 * 8000,
                    cancel_synthetic_echo_after_fallback=True,
                    fallback_payloads=(caller_payload, answer_payload),
                    fallback_wait_for_lapm_ack=True,
                    paired_duration_samples=45 * 8000,
                    peer_info0d_fields={"rate3429": 1,
                                        "v90_3429_upstream": 0},
                    peer_info1d_blocks=[{}, {}, {}, {}, {},
                                        {"hi_carrier": 0, "max_rate": 12}],
                    fallback_v42_role_overrides=overrides)
                try:
                    local_link = caller._v34_fallback_engine._v42
                    remote_link = answerer._v34_fallback_engine._v42
                    self.assertIsNotNone(local_link)
                    self.assertIsNotNone(remote_link)
                    self.assertEqual(
                        (local_link.calling, remote_link.calling),
                        expected_roles)
                    self.assertEqual((local_link.state, remote_link.state),
                                     ("lapm", "lapm"))
                    self.assertEqual(caller._test_fallback_payload_rx,
                                     answer_payload)
                    self.assertEqual(answerer._test_fallback_payload_rx,
                                     caller_payload)
                    self.assertEqual(local_link.stats["wire_acked"],
                                     len(caller_payload))
                    self.assertEqual(remote_link.stats["wire_acked"],
                                     len(answer_payload))
                    self.assertEqual(local_link.stats["fcs_errors"], 0)
                    self.assertEqual(remote_link.stats["fcs_errors"], 0)
                finally:
                    rtp.stop()
                    rtp._sock.close()
                    caller.close()
                    answerer.close()

    def test_v90_fallback_forced_v42_detection_carries_exact_payload(self):
        """Run V.42 ODP/ADP after V.8 LAPM agreement on V.90 fallback."""
        caller_payload = bytes((i * 37 + 11) & 0xFF for i in range(1024))
        answer_payload = bytes((i * 91 + 3) & 0xFF for i in range(1024))
        caller, answerer, rtp = self._run_production_paired_phase2(
            "pcmu", echo_delay_samples=1240 * 8,
            phase2_late_l2_echo_policy="advance-after-window",
            accept_early_peer_recovery=True,
            continue_v34_fallback=True,
            v34_fallback_hold_samples=4 * 8000,
            cancel_synthetic_echo_after_fallback=True,
            fallback_payloads=(caller_payload, answer_payload),
            fallback_wait_for_lapm_ack=True,
            fallback_force_v42_detection_after_v8=True,
            paired_duration_samples=30 * 8000,
            peer_info0d_fields={"rate3429": 1,
                                "v90_3429_upstream": 0},
            peer_info1d_blocks=[{}, {}, {}, {}, {},
                                {"hi_carrier": 0, "max_rate": 12}])
        try:
            local_link = caller._v34_fallback_engine._v42
            remote_link = answerer._v34_fallback_engine._v42
            self.assertIsNotNone(local_link)
            self.assertIsNotNone(remote_link)
            self.assertTrue(caller._test_v42_force_detect_started)
            self.assertTrue(answerer._test_v42_force_detect_started)
            self.assertEqual((local_link.state, remote_link.state),
                             ("lapm", "lapm"))
            self.assertEqual(caller._test_fallback_payload_rx, answer_payload)
            self.assertEqual(answerer._test_fallback_payload_rx, caller_payload)
            self.assertEqual(local_link.stats["wire_acked"],
                             len(caller_payload))
            self.assertEqual(remote_link.stats["wire_acked"],
                             len(answer_payload))
            self.assertEqual(local_link.stats["fcs_errors"], 0)
            self.assertEqual(remote_link.stats["fcs_errors"], 0)
        finally:
            rtp.stop()
            rtp._sock.close()
            caller.close()
            answerer.close()

    def test_v90_fallback_direct_originator_auto_detect_answerer(self):
        """Verify role-asymmetric V.42 detection and direct startup."""
        caller_payload = bytes((i * 37 + 11) & 0xFF for i in range(256))
        answer_payload = bytes((i * 91 + 3) & 0xFF for i in range(256))
        caller, answerer, rtp = self._run_production_paired_phase2(
            "pcmu", echo_delay_samples=1240 * 8,
            phase2_late_l2_echo_policy="advance-after-window",
            accept_early_peer_recovery=True,
            continue_v34_fallback=True,
            v34_fallback_hold_samples=4 * 8000,
            cancel_synthetic_echo_after_fallback=True,
            fallback_payloads=(caller_payload, answer_payload),
            fallback_wait_for_lapm_ack=True,
            # Order is V.34-call endpoint, V.34-answer endpoint. Under the
            # negotiated V.34 fallback roles these are V.42 answerer, then
            # originator: the answer side detects while the originator starts
            # LAPM directly.
            fallback_force_v42_detection_after_v8=(True, False),
            fallback_lapm_trace=True,
            paired_duration_samples=30 * 8000,
            peer_info0d_fields={"rate3429": 1,
                                "v90_3429_upstream": 0},
            peer_info1d_blocks=[{}, {}, {}, {}, {},
                                {"hi_carrier": 0, "max_rate": 12}])
        try:
            local_link = caller._v34_fallback_engine._v42
            remote_link = answerer._v34_fallback_engine._v42
            self.assertIsNotNone(local_link)
            self.assertIsNotNone(remote_link)
            self.assertTrue(caller._test_v42_force_detect_started)
            self.assertFalse(getattr(answerer, "_test_v42_force_detect_started", False))
            # With the V.42 originator direct and the answerer detecting, this
            # paired exchange must still reach LAPM and deliver both payloads
            # exactly.
            self.assertEqual((local_link.calling, remote_link.calling),
                             (False, True))
            self.assertEqual((local_link.state, remote_link.state),
                             ("lapm", "lapm"))
            self.assertEqual(caller._test_fallback_payload_rx, answer_payload)
            self.assertEqual(answerer._test_fallback_payload_rx, caller_payload)
            self.assertEqual(local_link.stats["wire_acked"], len(caller_payload))
            self.assertEqual(remote_link.stats["wire_acked"], len(answer_payload))
            self.assertEqual(local_link.stats["fcs_errors"], 0)
            self.assertEqual(remote_link.stats["fcs_errors"], 0)
        finally:
            rtp.stop()
            rtp._sock.close()
            caller.close()
            answerer.close()

    def test_v90_fallback_reversed_detection_asymmetry_does_not_connect(self):
        """Negative control: direct LAPM answerer vs detecting originator."""
        caller_payload = bytes((i * 37 + 11) & 0xff for i in range(256))
        answer_payload = bytes((i * 91 + 3) & 0xff for i in range(256))
        caller, answerer, rtp = self._run_production_paired_phase2(
            "pcmu", echo_delay_samples=1240 * 8,
            phase2_late_l2_echo_policy="advance-after-window",
            accept_early_peer_recovery=True,
            continue_v34_fallback=True,
            v34_fallback_hold_samples=4 * 8000,
            cancel_synthetic_echo_after_fallback=True,
            fallback_payloads=(caller_payload, answer_payload),
            fallback_wait_for_lapm_ack=True,
            fallback_force_v42_detection_after_v8=(False, True),
            fallback_lapm_trace=True,
            paired_duration_samples=30 * 8000,
            peer_info0d_fields={"rate3429": 1,
                                "v90_3429_upstream": 0},
            peer_info1d_blocks=[{}, {}, {}, {}, {},
                                {"hi_carrier": 0, "max_rate": 12}])
        try:
            local_link = caller._v34_fallback_engine._v42
            remote_link = answerer._v34_fallback_engine._v42
            self.assertIsNotNone(local_link)
            self.assertIsNotNone(remote_link)
            self.assertFalse(getattr(caller, "_test_v42_force_detect_started", False))
            self.assertTrue(answerer._test_v42_force_detect_started)
            self.assertEqual((local_link.calling, remote_link.calling),
                             (False, True))
            self.assertEqual((local_link.state, remote_link.state),
                             ("idle", "unsupported"))
            self.assertEqual(local_link.stats["xid_tx"], 0)
            self.assertEqual(remote_link.stats["xid_tx"], 0)
            self.assertNotEqual(caller._test_fallback_payload_rx, answer_payload)
            self.assertNotEqual(answerer._test_fallback_payload_rx, caller_payload)
        finally:
            rtp.stop()
            rtp._sock.close()
            caller.close()
            answerer.close()

    def test_v90_fallback_recovers_when_processed_first_xid_response_is_lost(self):
        """A peer may process XID even when its first response is lost."""
        caller_payload = bytes((i * 37 + 11) & 0xFF for i in range(1024))
        answer_payload = bytes((i * 91 + 3) & 0xFF for i in range(1024))
        fault = {"installed": False, "dropped": 0}

        def drop_first_response(caller, answerer, _sample):
            if fault["installed"]:
                return
            for engine in (caller, answerer):
                fallback = getattr(engine, "_v34_fallback_engine", None)
                link = getattr(fallback, "_v42", None)
                if link is None or link.calling:
                    continue
                original = link._answer_xid

                def answer_with_one_lost_response(info, *, _link=link,
                                                  _answer=original):
                    if fault["dropped"]:
                        _answer(info)
                        return
                    tx_before = len(_link._tx)
                    _answer(info)  # Process and retain the peer's XID choices.
                    _link._deferred_xid_response = None
                    while len(_link._tx) > tx_before:
                        _link._tx.pop()
                    fault["dropped"] += 1
                    _link._event(
                        "test fault: first processed XID response discarded")

                link._answer_xid = answer_with_one_lost_response
                fault["installed"] = True
                return

        caller, answerer, rtp = self._run_production_paired_phase2(
            "pcmu", echo_delay_samples=1240 * 8,
            phase2_late_l2_echo_policy="advance-after-window",
            accept_early_peer_recovery=True,
            continue_v34_fallback=True,
            v34_fallback_hold_samples=4 * 8000,
            cancel_synthetic_echo_after_fallback=True,
            fallback_payloads=(caller_payload, answer_payload),
            fallback_wait_for_lapm_ack=True,
            fallback_v42_frame_size=32,
            fallback_dce_connect_gate=True,
            fallback_lapm_trace=True,
            fallback_tick_hook=drop_first_response,
            paired_duration_samples=30 * 8000,
            peer_info0d_fields={"rate3429": 1,
                                "v90_3429_upstream": 0},
            peer_info1d_blocks=[{}, {}, {}, {}, {},
                                {"hi_carrier": 0, "max_rate": 12}])
        try:
            local_link = caller._v34_fallback_engine._v42
            remote_link = answerer._v34_fallback_engine._v42
            self.assertTrue(fault["installed"])
            self.assertEqual(fault["dropped"], 1)
            self.assertEqual((local_link.state, remote_link.state),
                             ("lapm", "lapm"))
            self.assertGreaterEqual(
                remote_link.stats["xid_tx"], 2)
            self.assertTrue(any("XID response received" in event
                                for event in remote_link.events))
            self.assertTrue(any(
                "test fault: first processed XID response discarded" in event
                for event in local_link.events))
            self.assertEqual(caller._test_fallback_payload_rx, answer_payload)
            self.assertEqual(answerer._test_fallback_payload_rx, caller_payload)
            for link in (local_link, remote_link):
                self.assertEqual(link.stats["fcs_errors"], 0)
                self.assertEqual(link.stats["hdlc_bad"], 0)
                self.assertGreater(link.stats["hdlc_good"], 0)
        finally:
            rtp.stop()
            rtp._sock.close()
            caller.close()
            answerer.close()

    def test_v90_fallback_accepts_processed_xid_response_delayed_before_t401(self):
        """A processed response delayed on the wire still establishes LAPM."""
        from v42_lapm import XID

        caller_payload = bytes((i * 37 + 11) & 0xFF for i in range(1024))
        answer_payload = bytes((i * 91 + 3) & 0xFF for i in range(1024))
        delay_samples = 9600  # 1.2 s: near, but below, the minimum T401.
        fault = {"installed": False, "response_bits": None,
                 "generated_at": None, "released_at": None,
                 "current_sample": 0}

        def delay_first_response(caller, answerer, sample):
            fault["current_sample"] = int(sample)
            responder = next((link for engine in (caller, answerer)
                              if (link := getattr(getattr(engine,
                                  "_v34_fallback_engine", None), "_v42", None))
                              is not None and not link.calling), None)
            if responder is None:
                return
            if not fault["installed"]:
                original_u = responder._u

                def hold_first_xid_response(ctrl, command, pf=0, info=b""):
                    if ((ctrl & 0xEF) != XID or command
                            or fault["response_bits"] is not None):
                        return original_u(ctrl, command, pf, info)
                    tx_before = len(responder._tx)
                    original_u(ctrl, command, pf, info)
                    held = []
                    while len(responder._tx) > tx_before:
                        held.append(responder._tx.pop())
                    held.reverse()
                    if not held:
                        raise AssertionError("XID response produced no wire bits")
                    fault["response_bits"] = held
                    fault["generated_at"] = fault["current_sample"]
                    responder._event(
                        "test fault: processed first XID response held before transmit")

                responder._u = hold_first_xid_response
                fault["installed"] = True
            if (fault["response_bits"] is not None
                    and fault["released_at"] is None
                    and sample >= fault["generated_at"] + delay_samples):
                responder._tx.extend(fault["response_bits"])
                fault["released_at"] = int(sample)
                responder._event(
                    f"test fault: first XID response released after "
                    f"{(sample - fault['generated_at']) / 8000:.3f}s")

        caller, answerer, rtp = self._run_production_paired_phase2(
            "pcmu", echo_delay_samples=1240 * 8,
            phase2_late_l2_echo_policy="advance-after-window",
            accept_early_peer_recovery=True, continue_v34_fallback=True,
            v34_fallback_hold_samples=4 * 8000,
            cancel_synthetic_echo_after_fallback=True,
            fallback_payloads=(caller_payload, answer_payload),
            fallback_wait_for_lapm_ack=True, fallback_v42_frame_size=32,
            fallback_dce_connect_gate=True, fallback_lapm_trace=True,
            fallback_tick_hook=delay_first_response,
            paired_duration_samples=30 * 8000,
            peer_info0d_fields={"rate3429": 1, "v90_3429_upstream": 0},
            peer_info1d_blocks=[{}, {}, {}, {}, {},
                                {"hi_carrier": 0, "max_rate": 12}])
        try:
            local_link = caller._v34_fallback_engine._v42
            remote_link = answerer._v34_fallback_engine._v42
            self.assertTrue(fault["installed"])
            self.assertIsNotNone(fault["response_bits"])
            self.assertIsNotNone(fault["released_at"])
            self.assertGreaterEqual(
                fault["released_at"] - fault["generated_at"], delay_samples)
            self.assertLess(delay_samples / 8000,
                            remote_link._t401_bits / remote_link.rate)
            self.assertEqual((local_link.state, remote_link.state),
                             ("lapm", "lapm"))
            self.assertEqual(caller._test_fallback_payload_rx, answer_payload)
            self.assertEqual(answerer._test_fallback_payload_rx, caller_payload)
            self.assertTrue(any("XID response received" in event
                                for event in remote_link.events))
            self.assertFalse(any("XID unanswered after 2 commands" in event
                                 for event in remote_link.events))
            self.assertEqual(remote_link.stats["xid_tx"], 1)
            for link in (local_link, remote_link):
                self.assertEqual(link.stats["fcs_errors"], 0)
                self.assertEqual(link.stats["hdlc_bad"], 0)
                self.assertGreater(link.stats["hdlc_good"], 0)
        finally:
            rtp.stop()
            rtp._sock.close()
            caller.close()
            answerer.close()

    def test_v90_fallback_recovers_when_processed_xid_response_crosses_t401(self):
        """A delayed original XID response can complete negotiation after retry."""
        from v42_lapm import XID

        caller_payload = bytes((i * 37 + 11) & 0xFF for i in range(1024))
        answer_payload = bytes((i * 91 + 3) & 0xFF for i in range(1024))
        delay_samples = 20000  # 2.5 s: past first T401, before retry exhaustion.
        fault = {
            "installed": False,
            "response_bits": None,
            "generated_at": None,
            "released_at": None,
            "current_sample": 0,
            "responses_suppressed": 0,
        }

        def delay_first_response(caller, answerer, sample):
            fault["current_sample"] = int(sample)
            responder = None
            for engine in (caller, answerer):
                fallback = getattr(engine, "_v34_fallback_engine", None)
                link = getattr(fallback, "_v42", None)
                if link is not None and not link.calling:
                    responder = link
                    break
            if responder is None:
                return

            if not fault["installed"]:
                original_u = responder._u

                def hold_first_xid_response(ctrl, command, pf=0, info=b""):
                    if (ctrl & 0xEF) != XID or command:
                        return original_u(ctrl, command, pf, info)
                    tx_before = len(responder._tx)
                    original_u(ctrl, command, pf, info)
                    held = []
                    while len(responder._tx) > tx_before:
                        held.append(responder._tx.pop())
                    held.reverse()
                    if not held:
                        raise AssertionError("XID response produced no wire bits")
                    if fault["response_bits"] is None:
                        fault["response_bits"] = held
                        fault["generated_at"] = fault["current_sample"]
                        responder._event(
                            "test fault: processed first XID response held before transmit")
                    elif fault["released_at"] is None:
                        fault["responses_suppressed"] += 1
                        responder._event(
                            "test fault: retry XID response suppressed until original response release")
                    else:
                        responder._tx.extend(held)

                responder._u = hold_first_xid_response
                fault["installed"] = True

            held = fault["response_bits"]
            if (held is not None and fault["released_at"] is None
                    and sample >= fault["generated_at"] + delay_samples):
                responder._tx.extend(held)
                fault["released_at"] = int(sample)
                responder._event(
                    f"test fault: first XID response released after "
                    f"{(sample - fault['generated_at']) / 8000:.3f}s")

        caller, answerer, rtp = self._run_production_paired_phase2(
            "pcmu", echo_delay_samples=1240 * 8,
            phase2_late_l2_echo_policy="advance-after-window",
            accept_early_peer_recovery=True,
            continue_v34_fallback=True,
            v34_fallback_hold_samples=4 * 8000,
            cancel_synthetic_echo_after_fallback=True,
            fallback_payloads=(caller_payload, answer_payload),
            fallback_wait_for_lapm_ack=True,
            fallback_payloads_after_lapm_samples=40000,
            fallback_v42_frame_size=32,
            fallback_dce_connect_gate=True,
            fallback_lapm_trace=True,
            fallback_tick_hook=delay_first_response,
            paired_duration_samples=60 * 8000,
            peer_info0d_fields={"rate3429": 1,
                                "v90_3429_upstream": 0},
            peer_info1d_blocks=[{}, {}, {}, {}, {},
                                {"hi_carrier": 0, "max_rate": 12}])
        try:
            local_link = caller._v34_fallback_engine._v42
            remote_link = answerer._v34_fallback_engine._v42
            self.assertTrue(fault["installed"])
            self.assertIsNotNone(fault["response_bits"])
            self.assertIsNotNone(fault["released_at"])
            self.assertGreaterEqual(fault["responses_suppressed"], 1)
            self.assertGreaterEqual(
                fault["released_at"] - fault["generated_at"], delay_samples)
            self.assertGreater(
                delay_samples / 8000,
                remote_link._t401_bits / remote_link.rate)
            self.assertGreaterEqual(remote_link.stats["t401_expiries"], 1)
            self.assertEqual((local_link.state, remote_link.state),
                             ("lapm", "lapm"),
                             (local_link.events, remote_link.events,
                              local_link.stats, remote_link.stats))
            self.assertEqual(caller._test_fallback_payload_rx, answer_payload)
            self.assertEqual(answerer._test_fallback_payload_rx, caller_payload)
            self.assertTrue(any("XID response received" in event
                                for event in remote_link.events))
            self.assertFalse(any("XID unanswered after 2 commands" in event
                                 for event in remote_link.events))
            self.assertGreaterEqual(remote_link.stats["xid_tx"], 2)
            for link in (local_link, remote_link):
                self.assertEqual(link.stats["fcs_errors"], 0)
                self.assertEqual(link.stats["hdlc_bad"], 0)
                self.assertGreater(link.stats["hdlc_good"], 0)
        finally:
            rtp.stop()
            rtp._sock.close()
            caller.close()
            answerer.close()

    def test_v90_fallback_same_lapm_roles_reproduce_startup_deadlock(self):
        """A same-role pair cannot establish LAPM; roles must be complementary.

        The captured peer role probe produced the same all-flags/no-frame signature
        when both sides waited. Also check simultaneous initiators: blindly
        flipping both endpoints is not a recovery strategy either.
        """
        for name, roles, calling in (
                ("both-answerers", ("answerer", "answerer"), (False, False)),
                ("both-originators", ("originator", "originator"),
                 (True, True))):
            with self.subTest(role_pair=name):
                caller, answerer, rtp = self._run_production_paired_phase2(
                    "pcmu", echo_delay_samples=1240 * 8,
                    phase2_late_l2_echo_policy="advance-after-window",
                    accept_early_peer_recovery=True,
                    continue_v34_fallback=True,
                    v34_fallback_hold_samples=4 * 8000,
                    cancel_synthetic_echo_after_fallback=True,
                    fallback_payloads=(b"", b""),
                    peer_info0d_fields={"rate3429": 1,
                                        "v90_3429_upstream": 0},
                    peer_info1d_blocks=[{}, {}, {}, {}, {},
                                        {"hi_carrier": 0, "max_rate": 12}],
                    fallback_v42_role_overrides=roles,
                    paired_duration_samples=8 * 8000)
                try:
                    links = (caller._v34_fallback_engine._v42,
                             answerer._v34_fallback_engine._v42)
                    self.assertEqual(tuple(link.calling for link in links),
                                     calling)
                    self.assertEqual(tuple(link.state for link in links),
                                     ("idle", "idle"))
                    self.assertEqual(tuple(link.stats["hdlc_good"]
                                           for link in links), (0, 0))
                    if name == "both-answerers":
                        self.assertEqual(tuple(link.stats["xid_tx"]
                                               for link in links), (0, 0))
                    else:
                        self.assertEqual(tuple(link.stats["xid_tx"]
                                               for link in links), (1, 1))
                finally:
                    rtp.stop()
                    rtp._sock.close()
                    caller.close()
                    answerer.close()






    def test_v90_fallback_rtp_gap_feedback_preserves_lapm_payload(self):
        """Reproduce captured peer's late PCMU gap across V.34 fallback LAPM."""
        caller_payload = bytes((i * 37 + 11) & 0xFF for i in range(8192))
        answer_payload = bytes((i * 91 + 3) & 0xFF for i in range(8192))
        # captured peer call 5's RTP hole occurred about 22 s after CONNECT/LAPM.
        gap_after_lapm_samples = 22 * 8000

        def run_arm(*, law, gap, recovery):
            env = {
                "V34_GAP_RECOVERY": "1" if recovery else "0",
                "V34_GAP_FEEDBACK": "0",
                "V34_GAP_TRELLIS_ERASURE": "0",
                "V34_GAP_TRELLIS_HOLD": "0",
            }
            with mock.patch.dict("os.environ", env):
                caller, answerer, rtp = self._run_production_paired_phase2(
                    law, echo_delay_samples=1240 * 8,
                    phase2_late_l2_echo_policy="advance-after-window",
                    accept_early_peer_recovery=True,
                    continue_v34_fallback=True,
                    v34_fallback_hold_samples=12 * 8000,
                    cancel_synthetic_echo_after_fallback=True,
                    fallback_payloads=(caller_payload, answer_payload),
                    fallback_payloads_after_rtp_gap=gap,
                    fallback_v42_role_overrides=("originator", "answerer"),
                    fallback_payloads_after_lapm_samples=(
                        None if gap else gap_after_lapm_samples),
                    fallback_rtp_drop_after_connect_samples=(
                        gap_after_lapm_samples if gap else None),
                    fallback_lapm_trace=True,
                    fallback_wait_for_lapm_ack=True,
                    paired_duration_samples=(
                        (45 if gap and recovery else 35) * 8000),
                    peer_info0d_fields={"rate3429": 1,
                                        "v90_3429_upstream": 0},
                    peer_info1d_blocks=[{}, {}, {}, {}, {},
                                        {"hi_carrier": 0, "max_rate": 12}])
            try:
                links = (caller._v34_fallback_engine._v42,
                         answerer._v34_fallback_engine._v42)
                guards = []
                for engine in (caller, answerer):
                    fallback = engine._v34_fallback_engine
                    core = (getattr(fallback, "answerer", None)
                            or getattr(fallback, "caller", None))
                    eq = getattr(getattr(getattr(core, "_p4rcv", None),
                                         "eq", None), "_transport_guard", None)
                    guards.append(None if eq is None else eq.report())
                return {
                    "caller_rx": caller._test_fallback_payload_rx,
                    "answerer_rx": answerer._test_fallback_payload_rx,
                    "link_states": tuple(link.state for link in links),
                    "stats": tuple(dict(link.stats) for link in links),
                    "tx_state": tuple({
                        "va": int(link.va), "vs": int(link.vs),
                        "unacked": tuple((int(seq), len(payload))
                                         for seq, payload in link._unacked),
                        "ack_pending": bool(link._ack_pending),
                        "poll_out": bool(link._poll_out),
                        "retx_frames": len(link._retx),
                        "t401_at": link._t401_at,
                    } for link in links),
                    "lapm_connected_sample": caller._test_lapm_connected_sample,
                    "gap_sample": caller._test_fallback_gap_sample,
                    "trace": list(caller._test_v42_media_trace),
                    "bad_frame_trace": caller._test_v42_lapm_bad_frame_trace,
                    "concealed_samples": rtp._rtp_counts["concealed_samples"],
                    "guards": tuple(guards),
                }
            finally:
                rtp.stop()
                rtp._sock.close()
                caller.close()
                answerer.close()

        # captured peer call 5 negotiated PCMU; also cover the other common G.711 law
        # under the same complementary LAPM role pair.
        for law in ("pcmu", "pcma"):
            with self.subTest(law=law):
                clean = run_arm(law=law, gap=False, recovery=False)
                baseline = run_arm(law=law, gap=True, recovery=False)
                recovered = run_arm(law=law, gap=True, recovery=True)

                self.assertEqual((clean["caller_rx"], clean["answerer_rx"]),
                                 (answer_payload, caller_payload))
                self.assertEqual(clean["concealed_samples"], 0)
                for stats in clean["stats"]:
                    self.assertGreater(stats["hdlc_good"], 0)

                self.assertEqual(baseline["gap_sample"],
                                 baseline["lapm_connected_sample"]
                                 + gap_after_lapm_samples)
                baseline_fcs = baseline["stats"][0]["fcs_errors"]
                self.assertGreater(baseline_fcs, 0, baseline["trace"][-12:])
                self.assertNotEqual(baseline["caller_rx"], answer_payload)
                # At the captured peer-matched late offset, LAPM remains nominally
                # up even though frames are corrupt and payload is incomplete.
                self.assertEqual(baseline["link_states"][0], "lapm")
                first_fcs = next(row for row in baseline["trace"]
                                 if row["endpoint"] == "caller"
                                 and row["fcs_errors"] > 0)
                self.assertGreaterEqual(first_fcs["media_sample"],
                                        baseline["gap_sample"])

                self.assertEqual((recovered["caller_rx"],
                                  recovered["answerer_rx"]),
                                 (answer_payload, caller_payload))
                self.assertEqual(recovered["link_states"], ("lapm", "lapm"))
                self.assertEqual(tuple(s["fcs_errors"]
                                       for s in recovered["stats"]), (0, 0))
                self.assertEqual(tuple(s["wire_acked"]
                                       for s in recovered["stats"]),
                                 (len(caller_payload), len(answer_payload)))
                self.assertEqual(tuple(state["unacked"]
                                       for state in recovered["tx_state"]),
                                 ((), ()))
                self.assertEqual(recovered["guards"][0]["registered_samples"], 160)
                self.assertGreater(recovered["guards"][0]["held_symbols"], 0)
                self.assertGreater(
                    recovered["guards"][0]["trellis_erasure_intervals"], 0)
                self.assertTrue(
                    recovered["bad_frame_trace"]["caller"]["enabled"])


    def test_advance_after_standard_l2_echo_window_is_bounded(self):
        c = _start_to_tone_b_reversal_wait(
            late_l2_echo_policy="advance-after-window")
        c.feed("tone_b_reversal", 1220)
        c.feed("tone_b_after_probes", 4000)
        deadline = c._local_l2_start_at + 550 * 8 + c.rtde_samples

        actions = c.tick(deadline)

        self.assertTrue(c.l2_echo_advanced_without_echo)
        self.assertFalse(c._l2_echo_complete)
        self.assertEqual(c.state, c.WAIT_TONE_B_REVERSAL_AFTER_PROBES)
        self.assertEqual([a.signal for a in actions], [
            "tone_a", "reverse-tone-a", "silence", "listen-tone-b-reversal"])
        self.assertEqual(actions[0].at, c._local_l2_start_at + c.l2_samples)

    def test_late_l2_echo_continue_is_explicit_diagnostic_policy(self):
        c = _start_to_tone_b_reversal_wait(late_l2_echo_policy="continue")
        c.feed("tone_b_reversal", 1220)
        c.feed("tone_b_after_probes", 4000)
        deadline = c._local_l2_start_at + 550 * 8 + c.rtde_samples
        now = deadline + 100

        actions = c.feed("l2_echo_complete", now, now=now)

        self.assertEqual(c.late_l2_echo_samples, 100)
        self.assertEqual(c.state, c.WAIT_TONE_B_REVERSAL_AFTER_PROBES)
        self.assertEqual([a.signal for a in actions], [
            "tone_a", "reverse-tone-a", "silence", "listen-tone-b-reversal"])
        self.assertEqual(actions[0].at, now)

    def test_late_l2_echo_default_policy_still_fails(self):
        c = _start_to_tone_b_reversal_wait()
        c.feed("tone_b_reversal", 1220)
        c.feed("tone_b_after_probes", 4000)
        deadline = c._local_l2_start_at + 550 * 8 + c.rtde_samples

        actions = c.feed("l2_echo_complete", deadline + 1, now=deadline + 1)

        self.assertEqual(c.state, c.FAILED)
        self.assertEqual(c.late_l2_echo_samples, 1)
        self.assertEqual([a.signal for a in actions], ["recovery-required"])

    def test_local_l2_echo_at_standard_deadline_is_accepted(self):
        c = _start_to_tone_b_reversal_wait()
        c.feed("tone_b_reversal", 1220)
        c.feed("tone_b_after_probes", 4000)
        deadline = c._local_l2_start_at + 550 * 8 + c.rtde_samples

        actions = c.feed("l2_echo_complete", deadline, now=deadline)

        self.assertEqual(c.state, c.WAIT_TONE_B_REVERSAL_AFTER_PROBES)
        self.assertEqual(c.late_l2_echo_samples, None)
        self.assertTrue(c._l2_echo_complete)
        self.assertFalse(c.l2_echo_advanced_without_echo)
        self.assertEqual([action.signal for action in actions], [
            "tone_a", "reverse-tone-a", "silence", "listen-tone-b-reversal"])

    def test_bounded_late_l2_echo_wait_tracks_event_and_has_timeout(self):
        # The local late-echo allowance remains subordinate to V.90's
        # absolute INFO1d deadline from Tone B detection.
        maximum = 1200 * 8
        c = _start_to_tone_b_reversal_wait(
            late_l2_echo_policy="bounded",
            late_l2_echo_max_samples=maximum)
        c.feed("tone_b_reversal", 1220)
        c.feed("tone_b_after_probes", 4000)
        nominal = c._local_l2_start_at + 550 * 8 + c.rtde_samples

        actions = c.feed("l2_echo_complete", nominal + 3000,
                         now=nominal + 3000)

        self.assertEqual(c.late_l2_echo_samples, 3000)
        self.assertEqual(c.state, c.WAIT_TONE_B_REVERSAL_AFTER_PROBES)
        self.assertEqual([action.signal for action in actions], [
            "tone_a", "reverse-tone-a", "silence", "listen-tone-b-reversal"])

        c = _start_to_tone_b_reversal_wait(
            late_l2_echo_policy="bounded",
            late_l2_echo_max_samples=maximum)
        c.feed("tone_b_reversal", 1220)
        c.feed("tone_b_after_probes", 4000)
        actions = c.tick(c._l2_echo_deadline_at)
        self.assertEqual(c.state, c.FAILED)
        self.assertEqual([action.signal for action in actions],
                         ["recovery-required"])

    def test_remote_probe_window_uses_receive_clock_not_rounded_media_clock(self):
        c = _start_to_tone_b_reversal_wait()
        c.feed("tone_b_reversal", 1220)
        c.feed("tone_b_after_probes", 4000)
        c.feed("l2_echo_complete", 4100)
        c.feed("tone_b_reversal_after_probes", 5000)
        c.feed("l1_complete", 35072,
               payload={"receive_sample_at": 13640})

        # A 52 ppm fitted clock maps these exact 4,000 RX samples to 4,001
        # rounded media samples.  The standard's receive window is exactly
        # 500 ms in the RX sample domain, so this must pass.
        c.feed("l2_complete", 39073,
               payload={"receive_sample_at": 17640})

        self.assertTrue(c._remote_l2_complete)
        self.assertEqual(c._remote_probe_end_at, 17640)

    def test_remote_probe_window_still_rejects_rx_interval_over_500_ms(self):
        c = _start_to_tone_b_reversal_wait()
        c.feed("tone_b_reversal", 1220)
        c.feed("tone_b_after_probes", 4000)
        c.feed("l2_echo_complete", 4100)
        c.feed("tone_b_reversal_after_probes", 5000)
        c.feed("l1_complete", 35072,
               payload={"receive_sample_at": 13640})

        with self.assertRaisesRegex(ValueError, "remote L2 exceeded"):
            c.feed("l2_complete", 39074,
                   payload={"receive_sample_at": 17641})

    def test_initial_tone_a_reversal_is_not_backdated_before_tone_b(self):
        c = V90AnalogueCallerPhase2()
        c.start(0)
        c.feed("info0a_complete", 392)
        c.feed("info0d", 500, payload={"ack_info0a": 0})

        actions = c.feed("tone_b", 900, now=800)

        reversal = next(a for a in actions if a.signal == "reverse-tone-a")
        assert reversal.at == 900

    def test_tone_b_before_info0a_completion_waits_for_50ms_of_tone_a(self):
        c = V90AnalogueCallerPhase2()
        c.start(0)
        c.feed("info0d", 100, payload={"ack_info0a": 0})

        assert c.feed("tone_b", 300, now=320) == []
        assert c.state == c.WAIT_TONE_B

        actions = c.feed("info0a_complete", 400, now=420)
        reversal = next(a for a in actions if a.signal == "reverse-tone-a")
        assert reversal.at == 800
        assert c.state == c.WAIT_TONE_B_REVERSAL

    def test_missed_first_tone_b_reversal_reacquires_tone_b(self):
        c = _start_to_tone_b_reversal_wait()
        due = c._deadline_at
        assert due == 900 + 2000 * 8
        assert c.tick(due - 1) == []
        actions = c.tick(due)
        assert c.state == c.WAIT_RECOVERY_TONE_B
        assert [a.signal for a in actions] == ["listen-tone-b"]

        actions = c.feed("tone_b", due + 100)
        assert c.state == c.WAIT_TONE_B_REVERSAL
        reversal = [a for a in actions if a.signal == "reverse-tone-a"]
        assert len(reversal) == 1
        assert reversal[0].at == due + 100

    def test_missing_recovery_tone_b_hands_off_after_bounded_wait(self):
        c = _start_to_tone_b_reversal_wait()
        first_deadline = c._deadline_at
        c.tick(first_deadline)
        recovery_deadline = c._deadline_at
        assert recovery_deadline == first_deadline + 2000 * 8
        assert c.tick(recovery_deadline - 1) == []
        actions = c.tick(recovery_deadline)
        assert c.state == c.FAILED
        assert [a.signal for a in actions] == ["handoff-retrain-v90-phase2"]

    def test_missing_post_l2_recovery_tone_b_is_bounded(self):
        c = _start_to_tone_b_reversal_wait()
        c.feed("tone_b_reversal", 1220)
        first_deadline = c._deadline_at
        c.tick(first_deadline)
        recovery_deadline = c._deadline_at
        assert recovery_deadline == first_deadline + 2000 * 8
        actions = c.tick(recovery_deadline)
        assert c.state == c.FAILED
        assert [a.signal for a in actions] == ["handoff-retrain-v90-phase2"]


    def test_missed_tone_b_after_l2_recovers_with_tone_a_exchange(self):
        c = _start_to_tone_b_reversal_wait()
        c.feed("tone_b_reversal", 1220)  # RTDE = 0 samples.
        assert c.state == c.TX_LOCAL_PROBES
        deadline = c._deadline_at
        assert deadline == c._local_l2_start_at + 600 * 8

        polled_at = deadline + 120
        actions = c.tick(polled_at)
        assert c.state == c.WAIT_RECOVERY_TONE_B
        assert [a.signal for a in actions] == ["tone_a", "listen-tone-b"]
        assert all(a.at == polled_at for a in actions)

        actions = c.feed("tone_b", polled_at + 1)
        assert c.state == c.WAIT_TONE_B_REVERSAL
        assert next(a.at for a in actions if a.signal == "reverse-tone-a") == polled_at + 400

    def test_post_l2_tone_a_waits_for_inflight_local_l2_to_finish(self):
        c = _start_to_tone_b_reversal_wait()
        c.feed("tone_b_reversal", 1220)
        assert c._local_l2_start_at is not None
        assert c.l2_samples is not None
        local_l2_end = c._local_l2_start_at + c.l2_samples

        # The far-end Tone B and returned probe echo may be detected before
        # our finite L2 renderer reaches its queued end-silence.
        c.feed("l2_echo_complete", local_l2_end - 500)
        actions = c.feed("tone_b_after_probes", local_l2_end - 200,
                         now=local_l2_end - 200)

        tone_a = next(a for a in actions if a.signal == "tone_a")
        reversal = next(a for a in actions if a.signal == "reverse-tone-a")
        silence = next(a for a in actions if a.signal == "silence")
        assert tone_a.at == local_l2_end
        assert reversal.at - tone_a.at == 50 * 8
        assert silence.at - reversal.at == 10 * 8

    def test_recovered_tone_b_reversal_receives_remote_probes(self):
        c = _start_to_tone_b_reversal_wait()
        c.feed("tone_b_reversal", 1220)
        initial_rtde = c.rtde_samples
        deadline = c._deadline_at
        actions = c.tick(deadline)
        assert [a.signal for a in actions] == ["tone_a", "listen-tone-b"]
        assert c._recovery_after_l2

        c.feed("tone_b", deadline + 100)
        assert c.state == c.WAIT_TONE_B_REVERSAL
        actions = c.feed("tone_b_reversal", deadline + 800)
        assert c.state == c.RX_REMOTE_PROBES
        assert c.rtde_samples == initial_rtde, (
            "the post-L2 recovery reversal must not overwrite the initial RTDE")
        assert not any(a.signal in ("l1", "l2") for a in actions)
        assert not c._recovery_after_l2

    def test_recovery_tone_b_detection_does_not_backdate_reversal(self):
        c = _start_to_tone_b_reversal_wait()
        c.feed("tone_b_reversal", 1220)
        deadline = c._deadline_at
        c.tick(deadline)
        detected_at = deadline + 100
        tx_now = detected_at + 600

        actions = c.feed("tone_b", detected_at, now=tx_now)

        reverse = next(a for a in actions if a.signal == "reverse-tone-a")
        silence = next(a for a in actions if a.signal == "silence")
        assert reverse.at == tx_now
        assert silence.at == tx_now + 10 * 8


    def test_info1d_timeout_selects_bounded_retrain_handoff(self):
        c = _start_to_tone_b_reversal_wait()
        c.feed("tone_b_reversal", 1220)
        c.feed("tone_b_after_probes", 4000)
        c.feed("l2_echo_complete", 4100)
        assert c.state == c.WAIT_TONE_B_REVERSAL_AFTER_PROBES
        c.feed("tone_b_reversal_after_probes", 5000)
        deadline = 4000 + 2000 * 8 + 2 * (c.rtde_samples or 0)
        assert c.state == c.RX_REMOTE_PROBES
        assert c._info1d_deadline_at == deadline
        c.feed("l1_complete", 6280)
        c.feed("l2_window_end", 10280)
        assert c.state == c.WAIT_INFO1D
        assert c._info1d_deadline_at == deadline  # cannot extend it

        polled_at = deadline + 120
        actions = c.tick(polled_at)
        assert c.done
        assert [a.signal for a in actions] == ["handoff-retrain-v90-phase2"]
        assert actions[0].at == polled_at

    def test_missing_remote_l1_is_bounded_from_tone_b_detection(self):
        c = _start_to_tone_b_reversal_wait()
        c.feed("tone_b_reversal", 1220)
        c.feed("tone_b_after_probes", 4000)
        c.feed("l2_echo_complete", 4100)
        c.feed("tone_b_reversal_after_probes", 5000)

        # The remote-probe watchdog starts from Tone B detection, before
        # waiting for either L1 or the optional L2 receive window, and
        # remains independent of the Tone-B-reversal watchdog.
        deadline = 4000 + 2000 * 8 + 2 * (c.rtde_samples or 0)
        assert c.state == c.RX_REMOTE_PROBES
        assert c._info1d_deadline_at == deadline
        assert c._deadline_kind is None
        assert c.tick(deadline - 1) == []
        actions = c.tick(deadline)

        assert c.state == c.FAILED
        assert [action.signal for action in actions] == [
            "handoff-retrain-v90-phase2"]
        assert actions[0].at == deadline

    def test_info1d_deadline_runs_alongside_tone_reversal_deadline(self):
        c = _start_to_tone_b_reversal_wait()
        c.feed("tone_b_reversal", 1220)
        c.feed("tone_b_after_probes", 4000)
        c.feed("l2_echo_complete", 4100)

        protocol_deadline = 4000 + 2000 * 8 + 2 * (c.rtde_samples or 0)
        reversal_deadline = c._deadline_at
        assert c.state == c.WAIT_TONE_B_REVERSAL_AFTER_PROBES
        assert c._info1d_deadline_at == protocol_deadline
        assert reversal_deadline is not None
        assert protocol_deadline < reversal_deadline

        actions = c.tick(protocol_deadline)

        assert c.state == c.FAILED
        assert [action.signal for action in actions] == [
            "handoff-retrain-v90-phase2"]

    def test_info1d_frame_is_accepted_before_but_not_at_deadline(self):
        def wait_for_info1d():
            c = _start_to_tone_b_reversal_wait()
            c.feed("tone_b_reversal", 1220)
            c.feed("tone_b_after_probes", 4000)
            c.feed("l2_echo_complete", 4100)
            c.feed("tone_b_reversal_after_probes", 5000)
            c.feed("l1_complete", 6280,
                   payload={"receive_sample_at": 6280})
            c.feed("l2_window_end", 10280,
                   payload={"receive_sample_at": 10280})
            assert c.state == c.WAIT_INFO1D
            return c, c._info1d_deadline_at

        accepted, deadline = wait_for_info1d()
        assert deadline is not None
        accepted_actions = accepted.feed("info1d", deadline - 1,
                                          now=deadline - 1)
        self.assertEqual(accepted.state, accepted.COMPLETE)
        self.assertIsNotNone(accepted.info1d)
        self.assertTrue(any(action.signal == "info1a"
                            for action in accepted_actions))

        expired, deadline = wait_for_info1d()
        late_actions = expired.feed("info1d", deadline, now=deadline)
        self.assertEqual(expired.state, expired.FAILED)
        self.assertIsNone(expired.info1d)
        self.assertEqual([action.signal for action in late_actions], [
            "handoff-retrain-v90-phase2"])


    def test_premature_tone_b_repeats_info0a_then_acknowledges_info0d(self):
        c = V90AnalogueCallerPhase2()
        c.start(0)
        actions = c.feed("tone_b", 100)
        assert [a.signal for a in actions] == ["info0a", "tone_a",
                                               "listen-info0d-tone-b"]
        actions = c.feed("info0d", 500, payload={"ack_info0a": 0})
        assert any(a.signal == "info0a" and a.value == 1 for a in actions)
        assert c.state == c.WAIT_TONE_B


    def test_recovery_info0a_actions_schedule_after_inflight_frame(self):
        c = V90AnalogueCallerPhase2()
        tx = V90AnaloguePhase2Transmitter(
            info0a_fields={}, info1a_builder=lambda mode: [0] * 70,
            probe_builder=lambda duration, rms: np.zeros(1),
            nominal_peak=.5, probe_rms=.1)
        tx.submit(c.start(0))
        initial_end = tx._completion["info0a-complete"]

        actions = c.feed("info0a_complete", initial_end)
        assert actions == []
        recovery = c.feed("tone_b", initial_end + 100)
        tx.submit(recovery)
        repeated_end = tx._completion["info0a-complete"]
        assert repeated_end > initial_end

        repeated = c.feed("info0a_complete", repeated_end)
        assert any(a.signal == "info0a" and a.value == 0 for a in repeated)
        tx.submit(repeated)
        repeated_end = tx._completion["info0a-complete"]
        c.feed("info0a_complete", repeated_end)
        ack = c.feed("info0d", repeated_end + 100)
        assert any(a.signal == "info0a" and a.value == 1 for a in ack)
        tx.submit(ack)

    def test_peer_reversal_preempts_stale_info0a_retries_before_l1(self):
        """A valid peer reversal must retire recovery retries queued behind it."""
        c = _start_to_tone_b_reversal_wait()
        tx = V90AnaloguePhase2Transmitter(
            info0a_fields={"rate3429": 1, "disallow3429": 1},
            info1a_builder=lambda mode: v90.build_info1a_v90(
                md_len=0, uinfo=90, analog_symbol_rate=5,
                digital_symbol_rate=6, freq_offset=0),
            probe_builder=lambda duration, rms: np.full(
                int(round(duration * 8000)), rms),
            nominal_peak=1200, probe_rms=300, start_at=94982)

        # Model the last in-flight INFO0a recovery retry, then two more
        # repeated-INFO0d responses queued behind it. On call 8 these retries
        # accumulated at 773 samples each until the final finite end was 97301.
        tx.submit([AnaloguePhase2Action(
            94982, "info0a", 1, tag="info0a-retry")])
        for info_at, tone_at in ((91000, 92000), (93000, 94000)):
            retry = c.feed("info0d", info_at, now=info_at,
                           payload={"ack_info0a": 0})
            tx.submit(retry)
            reversal = c.feed("tone_b", tone_at, now=tone_at)
            tx.submit(reversal)

        # The active retry ends at 95755, while additional retries are still
        # queued. The peer's valid reversal requires L1 at 95634. Retire the
        # now-stale retry tail rather than crashing or delaying the response.
        tx.render(95200 - tx.cursor)
        actions = c.feed("tone_b_reversal", 95234, now=95200)
        tx.submit(actions)
        self.assertEqual(tx._completion["l1-complete"], 95634 + 1280)
        rendered = tx.render(96914 - tx.cursor)
        # The reversal must replace the in-flight INFO0a retry at 95554,
        # producing reversed Tone A until L1 starts 10 ms later.  Checking
        # only from L1 onward misses the protocol-critical reversal interval.
        reversal = rendered[95554 - 95200:95634 - 95200]
        t = np.arange(95554, 95634)
        expected_reversed_tone_a = (
            -1200 * 10 ** (-1 / 20) * np.cos(2 * np.pi * 2400 * t / 8000)
            + 1200 * 10 ** (-7 / 20) * np.cos(2 * np.pi * 1800 * t / 8000))
        np.testing.assert_allclose(reversal, expected_reversed_tone_a,
                                   rtol=0, atol=1e-8)
        # L1 follows immediately at its required boundary.
        self.assertTrue(np.all(rendered[95634 - 95200:] == 600.0))

    def test_cancel_does_not_move_completed_tagged_completion(self):
        tx = V90AnaloguePhase2Transmitter(
            info0a_fields={"rate3429": 1, "disallow3429": 1},
            info1a_builder=lambda mode: v90.build_info1a_v90(
                md_len=0, uinfo=90, analog_symbol_rate=5,
                digital_symbol_rate=6, freq_offset=0),
            probe_builder=lambda duration, rms: np.zeros(
                int(round(duration * 8000))),
            nominal_peak=1200, probe_rms=300, start_at=0)
        tx.submit([AnaloguePhase2Action(0, "info0a", 0,
                                        tag="info0a-retry")])
        completed = tx._completion["info0a-complete"]
        tx.submit([AnaloguePhase2Action(
            completed + 100, "reverse-tone-a",
            cancel_tag="info0a-retry")])
        self.assertEqual(tx._completion["info0a-complete"], completed)

    def test_l1_cancellation_does_not_preempt_an_ordinary_info0a(self):
        tx = V90AnaloguePhase2Transmitter(
            info0a_fields={"rate3429": 1, "disallow3429": 1},
            info1a_builder=lambda mode: v90.build_info1a_v90(
                md_len=0, uinfo=90, analog_symbol_rate=5,
                digital_symbol_rate=6, freq_offset=0),
            probe_builder=lambda duration, rms: np.zeros(
                int(round(duration * 8000))),
            nominal_peak=1200, probe_rms=300, start_at=94982)
        tx.submit([AnaloguePhase2Action(94982, "info0a", 0)])
        with self.assertRaisesRegex(ValueError, "may not overlap"):
            tx.submit([AnaloguePhase2Action(
                95634, "l1", cancel_tag="info0a-retry")])


    def test_phase2_adapter_pumps_protocol_deadline(self):
        c = V90AnalogueCallerPhase2()
        adapter = V90AnaloguePhase2Adapter(c, rx_to_media=lambda sample: sample)
        adapter.start(0)
        c.feed("info0a_complete", 392)
        c.feed("info0d", 500)
        c.feed("tone_b", 900)

        actions = adapter.tick(media_now=c._deadline_at)
        assert c.state == c.WAIT_RECOVERY_TONE_B
        assert [a.signal for a in actions] == ["listen-tone-b"]

    def test_valid_info0d_carrier_is_not_misclassified_as_premature_tone_b(self):
        digital = V90DigitalPhase2Signals(
            info0d={"rate3429": 1, "alaw": 0}, amplitude=1000)
        digital.set_phase_at_sample(0)
        info0d = digital.modulate_info0d(ack_info0a=0)
        t = np.arange(4000) / 8000
        tone_b = 1000 * np.cos(2 * np.pi * 1200 * t)
        received = np.concatenate((info0d, tone_b))

        c = V90AnalogueCallerPhase2()
        adapter = V90AnaloguePhase2Adapter(c, rx_to_media=lambda sample: sample)
        adapter.start(0)
        adapter.feed_tx_complete("info0a_complete", media_at=800, tx_now=800)
        actions = []
        for start in range(0, len(received), 80):
            chunk = received[start:start + 80]
            actions.extend(adapter.feed(chunk, rx_at=start,
                                        tx_now=start + len(chunk)))

        assert c.info0d is not None
        assert c.state == c.WAIT_TONE_B_REVERSAL
        assert not any(action.signal == "info0a" for action in actions)
        assert any(action.signal == "reverse-tone-a" for action in actions)
        tone_b = [event for event in adapter.event_trace
                  if event["event"] == "tone_b" and event["routed"] == "tone_b"]
        assert len(tone_b) == 1
        # Once INFO0d validates the immediately following 1200-Hz Tone B,
        # route it in the current receive batch. Do not wait for the
        # pre-INFO candidate hold window, which can outlast the peer's
        # reversal deadline by hundreds of milliseconds.
        assert tone_b[0]["tx_now"] - tone_b[0]["rx_at"] <= 160
        assert adapter._pending_early_tones == []

    def test_premature_tone_b_is_released_after_info0d_search_window(self):
        c = V90AnalogueCallerPhase2()
        adapter = V90AnaloguePhase2Adapter(c, rx_to_media=lambda sample: sample)
        adapter.start(0)
        t = np.arange(4000) / 8000
        tone_b = 1000 * np.cos(2 * np.pi * 1200 * t)
        actions = []
        for start in range(0, len(tone_b), 80):
            chunk = tone_b[start:start + 80]
            actions.extend(adapter.feed(chunk, rx_at=start,
                                        tx_now=start + len(chunk)))
        assert any(action.signal == "info0a" for action in actions)
        assert c.state == c.WAIT_INFO0D
        assert adapter._pending_early_tones == []

    def test_deferred_pre_info_tone_is_not_rerouted_after_phase_advanced(self):
        c = V90AnalogueCallerPhase2()
        adapter = V90AnaloguePhase2Adapter(c, rx_to_media=lambda sample: sample)
        adapter.start(0)
        c.state = c.TX_LOCAL_PROBES
        c._local_l2_start_at = 7000
        adapter._last_routed_tone_b_rx_at = 3200
        adapter._pending_early_tones = [AnaloguePhase2RxEvent(
            "tone_b", at=3020, complete_at=3140, fields={})]

        actions = adapter.feed(np.zeros(6400), rx_at=0, tx_now=6400)

        assert c.state == c.TX_LOCAL_PROBES
        assert not actions
        trace = adapter.event_trace[-1]
        assert trace["event"] == "tone_b"
        assert trace["routed"] is None
        assert trace["discarded_reason"] == "stale pre-INFO tone candidate"

    def test_deferred_pre_info_tone_still_routes_when_not_already_handled(self):
        c = V90AnalogueCallerPhase2()
        adapter = V90AnaloguePhase2Adapter(c, rx_to_media=lambda sample: sample)
        adapter.start(0)
        c.state = c.WAIT_TONE_B
        c._info0a_complete_at = 800
        adapter._pending_early_tones = [AnaloguePhase2RxEvent(
            "tone_b", at=3020, complete_at=3140, fields={})]

        actions = adapter.feed(np.zeros(6400), rx_at=0, tx_now=6400)

        assert c.state == c.WAIT_TONE_B_REVERSAL
        assert any(action.signal == "reverse-tone-a" for action in actions)
        trace = adapter.event_trace[-1]
        assert trace["routed"] == "tone_b"
        assert "discarded_reason" not in trace

    def test_paired_pcm_phase2_reaches_both_phase3_handoffs(self):
        for law in ("pcmu", "pcma"):
            with self.subTest(law=law):
                self._run_paired_phase2(law=law)

    def test_paired_pcm_phase2_recovers_after_lost_post_l2_tone_b(self):
        for law in ("pcmu", "pcma"):
            with self.subTest(law=law):
                # Drop the initial post-L2 Tone B, but restore media before
                # the answerer's §9.2.1.2.4 recovery reversal. A 700 ms blanket
                # outage also erases that recovery transmission and is not a
                # test of the caller's Tone-B recovery behavior.
                self._run_paired_phase2(law=law, hide_post_l2_tone_b_ms=100)

    def test_paired_pcm_phase2_advances_when_local_echo_exceeds_window(self):
        """Keep a responsive peer while the local L2 return is delayed.

        The controller's local-echo-completion event is deliberately delayed
        past its normative 550 ms + RTDE ceiling. This fixture does not delay
        the PCM waveform itself. The opt-in policy must advance on its timer
        and complete Phase 2 without relying on the late callback.
        """
        for law in ("pcmu", "pcma"):
            with self.subTest(law=law):
                caller, _ = self._run_paired_phase2(
                    law=law,
                    caller_late_l2_echo_policy="advance-after-window",
                    caller_echo_delay_ms=800)
                self.assertTrue(caller.l2_echo_advanced_without_echo)
                self.assertFalse(caller.early_peer_recovery_accepted)
                self.assertEqual(caller.mode, 6)

    def test_paired_pcm_phase2_measures_analogue_probe_clock_offset(self):
        for law in ("pcmu", "pcma"):
            with self.subTest(law=law):
                for offset_hz, expected_units, tolerance in (
                        (0.10, 5, 1), (0.12, 6, 0)):
                    with self.subTest(offset_hz=offset_hz):
                        _, answerer_rx = self._run_paired_phase2(
                            law=law,
                            probe_offset_hz_at_1050=offset_hz)
                        measured = (
                            answerer_rx.signals.probe_frequency_offset_units)
                        self.assertLessEqual(
                            abs(measured - expected_units), tolerance)

    def test_production_digital_engine_hands_off_to_phase3_under_both_laws(self):
        """Exercise caller Phase 2 through the RTP-facing answerer engine."""
        for law in ("pcmu", "pcma"):
            with self.subTest(law=law):
                engine, caller_tx, caller = self._run_engine_phase2(law)
                try:
                    self.assertEqual(caller.state, caller.COMPLETE)
                    self.assertEqual(caller.mode, 6)
                    self.assertEqual(engine.phase2_state, engine._phase2.COMPLETE,
                                     (caller.state, engine.events[-12:],
                                      engine._phase2.info1a))
                    self.assertIsNotNone(
                        engine._phase2._second_l2_echo_complete_at,
                        "answerer must detect its delayed L2 echo from received PCM")
                    self.assertTrue(engine._phase2._tone_a_during_second_l2)
                    self.assertTrue(caller._l2_echo_complete,
                                    "caller must detect its delayed L2 echo from received PCM")
                    self.assertEqual(engine.phase3_handoff, "handoff-v90-phase3")
                    self.assertIsNotNone(engine._phase3)
                    self.assertEqual(engine._phase3.law, law)
                    self.assertTrue(engine._phase3_enable_lapm)
                    self.assertTrue(any(name == "phase3-session-created"
                                        for name, _ in engine.events))
                    self.assertFalse(engine._failed)
                    self.assertFalse(caller_tx.late_actions)
                    self.assertFalse(engine._renderer.late_actions)
                    caller_phase3_tx, caller_phase3 = self._feed_phase3_ja(
                        engine, law)
                    self.assertEqual(caller_phase3.state,
                                     caller_phase3.RX_TRN1D)
                    self.assertEqual(caller_phase3_tx.late_stop_samples, 0)
                    np.testing.assert_array_equal(
                        caller_phase3_tx.read_pcm(160), np.zeros(160))
                    self.assertTrue(engine._phase3.transmit_ready)
                    self.assertTrue(any(name == "ja-accepted"
                                        for name, _ in engine._phase3.events))
                    phase3_output = engine.tx_g711(160, law)
                    self.assertEqual(len(phase3_output), 160)
                    silence = 0xFF if law == "pcmu" else 0xD5
                    self.assertNotEqual(phase3_output, bytes((silence,)) * 160)
                finally:
                    engine.close()

    def test_paired_phase2_bounded_vs_advance_synthetic_return(self):
        """Verify the standard echo deadline against synthetic stress lags.

        The earlier 1.24–1.61 s values were selected periodic-L1 correlation
        peaks, not captured delay measurements. Here they are only synthetic
        stress points. Bounded mode must not wait beyond V.90's 550 ms + RTDE
        echo window; advance mode completes the same paired PCM exchange by
        sending Tone A at that boundary.
        """
        for law in ("pcmu", "pcma"):
            for delay_ms in (1240, 1450, 1610):
                with self.subTest(law=law, echo_delay_ms=delay_ms):
                    bounded, _, bounded_caller = self._run_engine_phase2(
                        law, local_echo_delay_samples=delay_ms * 8,
                        local_echo_gain=2.0,
                        late_l2_echo_policy="bounded",
                        late_l2_echo_max_samples=2500 * 8)
                    try:
                        self.assertEqual(bounded_caller.state,
                                         bounded_caller.FAILED,
                                         (bounded_caller.state,
                                          bounded.events[-10:]))
                        self.assertTrue(
                            bounded._phase2.missing_tone_a_recovery_started)
                        self.assertIsNone(bounded_caller.info1d)
                    finally:
                        bounded.close()

                    advancing, advancing_tx, advancing_caller = (
                        self._run_engine_phase2(
                            law, local_echo_delay_samples=delay_ms * 8,
                            local_echo_gain=2.0,
                            late_l2_echo_policy="advance-after-window"))
                    try:
                        self.assertEqual(advancing_caller.state,
                                         advancing_caller.COMPLETE,
                                         (advancing_caller.state,
                                          advancing_tx.late_actions,
                                          advancing.events[-10:]))
                        self.assertEqual(advancing.phase2_state,
                                         advancing._phase2.COMPLETE)
                        self.assertTrue(
                            advancing_caller.l2_echo_advanced_without_echo)
                        self.assertEqual(advancing.phase3_handoff,
                                         "handoff-v90-phase3")
                        self.assertFalse(advancing_tx.late_actions)
                        self.assertFalse(advancing._renderer.late_actions)
                    finally:
                        advancing.close()

    def test_paired_phase2_silence_intervals_and_lost_reversal_recovery(self):
        # A 40 ms gap can be recovered when the remaining PCMA L1/L2 is strong
        # enough to decode INFO1d; the PCMU cases still take the bounded
        # retrain path. A separate targeted gap below removes the Tone B
        # reversal itself and must always fail into bounded retrain.
        for law, delay_ms in (("pcmu", 1240), ("pcma", 1610)):
            for direction in ("caller-rx", "answerer-rx"):
                with self.subTest(law=law, case="probe-gap", direction=direction):
                    gaps = {"caller_rx_gaps": ((12000, 320),)} \
                        if direction == "caller-rx" else {
                            "answerer_rx_gaps": ((12000, 320),)}
                    engine, caller_tx, caller = self._run_engine_phase2(
                        law, local_echo_delay_samples=delay_ms * 8,
                        local_echo_gain=2.0,
                        late_l2_echo_policy="bounded",
                        late_l2_echo_max_samples=2500 * 8,
                        accept_early_peer_recovery=True, **gaps)
                    try:
                        if law == "pcma":
                            self.assertEqual(
                                caller.state, caller.COMPLETE,
                                caller._test_adapter.event_trace[-8:])
                            self.assertIsNotNone(caller.info1d)
                            self.assertEqual(engine._phase2.state,
                                             engine._phase2.COMPLETE)
                            self.assertTrue(any(
                                handoff.signal == "handoff-v90-phase3"
                                for handoff in caller_tx.handoffs))
                        else:
                            self.assertEqual(
                                caller.state, caller.FAILED,
                                caller._test_adapter.event_trace[-8:])
                            self.assertTrue(any(
                                handoff.signal == "handoff-retrain-v90-phase2"
                                for handoff in caller_tx.handoffs))
                    finally:
                        engine.close()

        # If the one packet erased contains the peer's short Tone B reversal,
        # Phase 2 cannot pretend it observed the transition. It must leave the
        # state bounded and issue a retrain handoff instead of waiting forever.
        engine, tx, caller = self._run_engine_phase2(
            "pcmu", local_echo_delay_samples=1240 * 8,
            local_echo_gain=2.0, late_l2_echo_policy="bounded",
            late_l2_echo_max_samples=2500 * 8,
            accept_early_peer_recovery=True,
            caller_rx_gaps=((9920, 160),))
        try:
            self.assertEqual(caller.state, caller.FAILED)
            self.assertTrue(any(handoff.signal == "handoff-retrain-v90-phase2"
                                for handoff in tx.handoffs))
            self.assertLessEqual(caller._deadline_at or 0, 0)
        finally:
            engine.close()

    def test_production_pair_routes_timestamp_gaps_through_caller_engine(self):
        # Exercise the actual raw-G.711 caller engine's feed_gap path against
        # the production digital-answerer engine, rather than replacing the
        # missing interval with a zero-valued PCM block in the fixture.
        for law in ("pcmu", "pcma"):
            for label, gaps, expected in (
                    ("one-40ms", ((9600, 320),), [320]),
                    ("two-separated-20ms", ((8000, 160), (9600, 160)),
                     [160, 160])):
                with self.subTest(law=law, gap=label):
                    caller, answerer, rtp = self._run_production_paired_phase2(
                        law, echo_delay_samples=1610 * 8,
                        dropped_samples=gaps,
                        accept_early_peer_recovery=True)
                    try:
                        self.assertEqual(caller._controller.state,
                                         caller._controller.COMPLETE)
                        self.assertIsNotNone(caller._controller.info1d)
                        self.assertEqual(answerer._phase2.state,
                                         answerer._phase2.COMPLETE)
                        actual = [value["samples"]
                                  for name, value in caller.events
                                  if name == "phase2-rx-gap-filled"]
                        self.assertEqual(actual, expected)
                        # This short Phase 2 exchange ends before the
                        # configured minimum fit span. It proves ordered RTP
                        # anchors were consumed, not that a rate fit qualified.
                        self.assertGreaterEqual(len(caller._clock.anchors), 2)
                        self.assertIsNotNone(caller._phase3_start_at)
                    finally:
                        rtp.stop()
                        rtp._sock.close()
                        caller.close()
                        answerer.close()

        # Drop the packet at the paired trace's peer-recovery Tone-B reversal.
        # With the opt-in receiver path disabled, the call must take its
        # bounded retrain handoff rather than fabricate the missing event.
        caller, answerer, rtp = self._run_production_paired_phase2(
            "pcmu", echo_delay_samples=1610 * 8,
            dropped_samples=((10240, 160),), arrival_jitter_ms=0.0,
            paired_duration_samples=5 * 8000)
        try:
            retry = next((value for name, value in caller.events
                          if name == "phase2-retrain-started"), None)
            self.assertIsNotNone(retry, caller.events[-24:])
            self.assertEqual(retry["attempt"], 1)
            self.assertTrue(retry["retained_info0d"])
            self.assertTrue(retry["same_engine"])
            self.assertFalse(caller._failed)
            self.assertNotEqual(caller._controller.state,
                                caller._controller.FAILED)
            self.assertFalse(caller.connected)
            self.assertTrue(
                answerer._phase2.missing_tone_a_recovery_started)
            # Hold the peer silent through the single retry's watchdog. The
            # same production engine must then fail closed rather than report
            # CONNECT or loop into another Phase-2 restart.
            caller.tx_samples(6 * 8000)
            self.assertTrue(caller._failed)
            self.assertFalse(caller.connected)
            self.assertEqual(caller._phase2_retrain_attempts, 1)
            terminal = [value for name, value in caller.events
                        if name == "phase2-retrain-failed"]
            self.assertTrue(terminal, caller.events[-24:])
            self.assertEqual(terminal[-1]["reason"],
                             "V.90 Phase-2 retrain limit reached")
            # Model OnlineAudioSession's engine binding and verify that the
            # bounded second failure reaches its startup NO CARRIER gate.
            rtp._v22 = caller
            self.assertTrue(rtp.failed)
            self.assertTrue(any(name == "phase2-rx-gap-filled"
                                and value["samples"] == 160
                                for name, value in caller.events))
        finally:
            rtp.stop()
            rtp._sock.close()
            caller.close()
            answerer.close()

    def test_production_pair_caller_phase2_retrain_reaches_phase3(self):
        """A reactive digital peer lets one caller retrain resume Phase 2."""
        for law in ("pcmu", "pcma"):
            with self.subTest(law=law):
                caller, answerer, rtp = self._run_production_paired_phase2(
                    law, echo_delay_samples=1610 * 8,
                    dropped_samples=((10240, 160),),
                    arrival_jitter_ms=0.0,
                    paired_duration_samples=15 * 8000,
                    answerer_info1a_timeout_retrain=True,
                    caller_phase2_retrain_hybrid_echo=True)
                try:
                    retry = next((value for name, value in caller.events
                                  if name == "phase2-retrain-started"), None)
                    self.assertEqual(retry, {
                        "media_at": 24640, "attempt": 1,
                        "previous_state": "failed", "retained_info0d": True,
                        "same_engine": True, "initial_silence_ms": 70,
                        "tone_b_watchdog_ms": 5000,
                    })
                    self.assertFalse(caller._failed, caller.events[-16:])
                    self.assertFalse(answerer._failed, answerer.events[-16:])
                    self.assertEqual(caller._phase2_retrain_attempts, 1)
                    self.assertTrue(answerer._phase2.retrain_response_started)
                    self.assertEqual(caller._controller.state,
                                     caller._controller.COMPLETE)
                    self.assertEqual(answerer._phase2.state,
                                     answerer._phase2.COMPLETE)
                    self.assertIsNotNone(caller._phase3_start_at)
                    self.assertIsNotNone(caller._phase3_controller)
                    self.assertIsNotNone(answerer._phase3)
                    self.assertEqual(
                        [value["signal"] for name, value in caller.events
                         if name == "phase2-handoff"],
                        ["handoff-retrain-v90-phase2", "handoff-v90-phase3"])
                    self.assertFalse(caller.connected,
                                     "Phase-2 recovery must not fabricate CONNECT")
                finally:
                    rtp.stop()
                    rtp._sock.close()
                    caller.close()
                    answerer.close()

    def test_production_pair_bounded_gap_matrix_under_clock_drift(self):
        # Broaden the synthetic envelope without changing the experimental
        # profile. These loss windows lie in the recovery/probe receive period
        # before INFO1d; the missing-reversal case is tested separately.
        gap_cases = (
            ("adjacent-40ms", ((9600, 320),), [320]),
            ("three-separated-20ms",
             ((8000, 160), (9600, 160), (11200, 160)), [160, 160, 160]),
            ("mixed-40-plus-20ms", ((8000, 320), (9600, 160)),
             [320, 160]),
        )
        for law in ("pcmu", "pcma"):
            for drift_ppm in (-1500.0, 1500.0):
                for label, gaps, expected in gap_cases:
                    with self.subTest(law=law, drift_ppm=drift_ppm,
                                      gaps=label):
                        caller, answerer, rtp = self._run_production_paired_phase2(
                            law, echo_delay_samples=1610 * 8,
                            dropped_samples=gaps,
                            arrival_clock_ppm=drift_ppm,
                            arrival_jitter_ms=2.0,
                            accept_early_peer_recovery=True)
                        try:
                            self.assertEqual(
                                caller._controller.state,
                                caller._controller.COMPLETE,
                                caller.events[-12:])
                            self.assertIsNotNone(caller._controller.info1d)
                            self.assertEqual(answerer._phase2.state,
                                             answerer._phase2.COMPLETE)
                            actual = [value["samples"]
                                      for name, value in caller.events
                                      if name == "phase2-rx-gap-filled"]
                            self.assertEqual(actual, expected)
                            # At these short startup lengths the fit may not
                            # reach its minimum sample span. The test covers
                            # clocked RTP-gap progression, not clock-rate
                            # estimation qualification.
                            self.assertGreaterEqual(
                                len(caller._clock.anchors), 2)
                        finally:
                            rtp.stop()
                            rtp._sock.close()
                            caller.close()
                            answerer.close()

    def test_production_pair_observed_60ms_gap_boundary(self):
        for law in ("pcmu", "pcma"):
            caller, answerer, rtp = self._run_production_paired_phase2(
                law, echo_delay_samples=1610 * 8,
                dropped_samples=((9600, 480),),
                phase2_gap_max_samples=480, arrival_clock_ppm=1500,
                arrival_jitter_ms=2.0,
                accept_early_peer_recovery=True)
            try:
                self.assertEqual(caller._controller.state,
                                 caller._controller.COMPLETE,
                                 caller.events[-12:])
                self.assertIsNotNone(caller._controller.info1d)
                self.assertEqual(answerer._phase2.state,
                                 answerer._phase2.COMPLETE)
                self.assertEqual(
                    [value["samples"] for name, value in caller.events
                     if name == "phase2-rx-gap-filled"], [480])
                self.assertGreaterEqual(len(caller._clock.anchors), 2)
            finally:
                rtp.stop()
                rtp._sock.close()
                caller.close()
                answerer.close()

        # The experimental 60 ms ceiling remains a hard bound: larger loss
        # exits through the terminal recovery path instead of extending fill.
        caller, answerer, rtp = self._run_production_paired_phase2(
            "pcmu", echo_delay_samples=1610 * 8,
            dropped_samples=((9600, 640),), phase2_gap_max_samples=480,
            arrival_clock_ppm=1500, arrival_jitter_ms=2.0,
            accept_early_peer_recovery=True)
        try:
            self.assertEqual(caller._controller.state,
                             caller._controller.FAILED)
            self.assertTrue(caller._failed)
            self.assertTrue(any(
                name == "phase2-failed"
                and value.get("reason") ==
                "RTP gap during V.90 analogue-caller Phase2"
                and value.get("samples") == 640
                for name, value in caller.events))
        finally:
            rtp.stop()
            rtp._sock.close()
            caller.close()
            answerer.close()

    def test_production_pair_advances_after_late_hybrid_echo_window(self):
        # The captured peer captures return the caller's probe about a second after
        # the standard echo window. Verify that the opt-in fallback keeps
        # Tone A aligned to the end of our own L2 instead of waiting for that
        # late echo, while both G.711 laws still complete Phase 2 as a pair.
        for law in ("pcmu", "pcma"):
            with self.subTest(law=law):
                caller, answerer, rtp = self._run_production_paired_phase2(
                    law, echo_delay_samples=1610 * 8,
                    phase2_late_l2_echo_policy="advance-after-window")
                try:
                    self.assertEqual(caller._controller.state,
                                     caller._controller.COMPLETE,
                                     caller.events[-12:])
                    self.assertEqual(answerer._phase2.state,
                                     answerer._phase2.COMPLETE)
                    self.assertTrue(
                        caller._controller.l2_echo_advanced_without_echo)
                    self.assertIsNotNone(caller._controller.info1d)
                finally:
                    rtp.stop()
                    rtp._sock.close()
                    caller.close()
                    answerer.close()

    def test_production_pair_contrasts_bounded_and_advance_echo_policies(self):
        """Hold the peer and 1.61 s returned-L1 delay fixed across policies."""
        for law in ("pcmu", "pcma"):
            outcomes = {}
            for policy in ("bounded", "advance-after-window"):
                with self.subTest(law=law, policy=policy):
                    caller, answerer, rtp = self._run_production_paired_phase2(
                        law, echo_delay_samples=1610 * 8,
                        phase2_late_l2_echo_policy=policy,
                        accept_early_peer_recovery=False)
                    try:
                        controller = caller._controller
                        if policy == "bounded":
                            # The now standards-faithful digital peer times
                            # out waiting for the second caller reversal and
                            # sends its §9.2.1.2.4 recovery probes. A caller
                            # without the experimental receive-path option
                            # misses that early reversal/L1 and cannot finish.
                            self.assertEqual(controller.state,
                                             controller.FAILED,
                                             caller.events[-12:])
                            self.assertIsNone(controller.info1d)
                            self.assertTrue(
                                answerer._phase2.missing_tone_a_recovery_started)
                            self.assertFalse(
                                controller.early_peer_recovery_accepted)
                            outcomes[policy] = {
                                "caller_state": controller.state,
                                "peer_recovery": answerer._phase2.missing_tone_a_recovery_started,
                                "early_recovery_accepted": controller.early_peer_recovery_accepted,
                            }
                        else:
                            self.assertEqual(controller.state,
                                             controller.COMPLETE,
                                             caller.events[-12:])
                            self.assertEqual(answerer._phase2.state,
                                             answerer._phase2.COMPLETE)
                            self.assertIsNotNone(controller.info1d)
                            self.assertTrue(
                                controller.l2_echo_advanced_without_echo)
                            self.assertFalse(
                                answerer._phase2.missing_tone_a_recovery_started)
                            outcomes[policy] = {
                                "caller_state": controller.state,
                                "peer_recovery": answerer._phase2.missing_tone_a_recovery_started,
                                "advanced": controller.l2_echo_advanced_without_echo,
                            }
                    finally:
                        rtp.stop()
                        rtp._sock.close()
                        caller.close()
                        answerer.close()

            # Bounded waits long enough for the local echo but misses the
            # peer's standards-defined early recovery sequence; advancing at
            # the 550 ms + RTDE boundary sends Tone A in time and avoids that
            # recovery branch entirely.
            self.assertEqual(outcomes["bounded"]["caller_state"],
                             V90AnalogueCallerPhase2.FAILED, outcomes)
            self.assertTrue(outcomes["bounded"]["peer_recovery"], outcomes)
            self.assertFalse(outcomes["bounded"]["early_recovery_accepted"],
                             outcomes)
            self.assertTrue(outcomes["advance-after-window"]["advanced"],
                            outcomes)

    def test_saved_captured_peer_phase2_event_order_survives_advance_policy(self):
        """Lock in the detected B-reversal/L1/L2 order from saved RX audio."""
        outcomes = {}
        for policy in ("bounded", "advance-after-window"):
            caller = V90AnalogueCallerPhase2(
                late_l2_echo_policy=policy,
                late_l2_echo_max_samples=(20000 if policy == "bounded" else 0))
            adapter = V90AnaloguePhase2Adapter(
                caller, rx_to_media=lambda sample: sample)
            adapter.start(0)
            caller.feed("info0a_complete", 392)
            caller.feed("info0d", 500, payload={"ack_info0a": 0})
            caller.feed("tone_b", 900)
            caller.feed("tone_b_reversal", 1220)
            caller.feed("tone_b_after_probes", 4000)

            echo_deadline = (caller._local_l2_start_at
                             + 550 * 8 + caller.rtde_samples)
            remote_reversal = echo_deadline + 500
            remote_l1_complete = remote_reversal + 1378
            remote_l2_complete = remote_l1_complete + 4000

            def detect(kind, at):
                event = AnaloguePhase2RxEvent(kind, at, at, {})
                routed = adapter._route(event)
                if routed is None:
                    return None
                name, sample = routed
                payload = {}
                if name in ("l1_complete", "l2_complete"):
                    payload["receive_sample_at"] = sample
                caller.feed(name, sample, now=at, payload=payload)
                return name

            if policy == "advance-after-window":
                adapter.tick(media_now=echo_deadline)

            early_reversal_route = detect("tone_b_reversal", remote_reversal)
            early_l1_route = detect("l1_complete", remote_l1_complete)
            early_l2_route = detect("l2_complete", remote_l2_complete)

            if policy == "bounded":
                # This detector event is state-misclassified as the delayed
                # local echo. The actual remote probes have already passed.
                later_reversal_route = detect(
                    "tone_b_reversal", remote_l2_complete + 2800)
                info1d_route = detect("info1d", remote_l2_complete + 3800)
                outcomes[policy] = {
                    "early_routes": (early_reversal_route, early_l1_route,
                                     early_l2_route),
                    "later_reversal_route": later_reversal_route,
                    "remote_l1_seen": caller._remote_l1_complete,
                    "info1d_route": info1d_route,
                    "state": caller.state,
                }
            else:
                info1d_route = detect("info1d", remote_l2_complete + 1000)
                outcomes[policy] = {
                    "early_routes": (early_reversal_route, early_l1_route,
                                     early_l2_route),
                    "info1d_route": info1d_route,
                    "remote_l1_seen": caller._remote_l1_complete,
                    "remote_l2_seen": caller._remote_l2_complete,
                    "state": caller.state,
                }

        self.assertEqual(outcomes["bounded"]["early_routes"],
                         (None, None, "l2_echo_complete"), outcomes)
        self.assertEqual(outcomes["bounded"]["later_reversal_route"],
                         "tone_b_reversal_after_probes", outcomes)
        self.assertFalse(outcomes["bounded"]["remote_l1_seen"], outcomes)
        self.assertIsNone(outcomes["bounded"]["info1d_route"], outcomes)
        self.assertEqual(outcomes["advance-after-window"]["early_routes"],
                         ("tone_b_reversal_after_probes", "l1_complete",
                          "l2_complete"), outcomes)
        self.assertTrue(outcomes["advance-after-window"]["remote_l1_seen"],
                        outcomes)
        self.assertTrue(outcomes["advance-after-window"]["remote_l2_seen"],
                        outcomes)
        self.assertEqual(outcomes["advance-after-window"]["info1d_route"],
                         "info1d", outcomes)
        self.assertEqual(outcomes["advance-after-window"]["state"],
                         V90AnalogueCallerPhase2.COMPLETE, outcomes)

    def test_saved_captured_peer_recovery_sequence_routes_to_remote_probes(self):
        """Replay probe-5's event clocks through the peer-recovery path.

        The saved trace has a peer Tone-B reversal followed by L1 and L2 while
        the caller is still waiting for its returned local-L2 echo.  In the
        capture those events were discarded/misclassified.  With peer recovery
        enabled, the same receive-clock events must instead qualify the
        remote probe sequence and advance to INFO1d.
        """
        caller = V90AnalogueCallerPhase2(
            late_l2_echo_policy="bounded", late_l2_echo_max_samples=20000,
            accept_early_peer_recovery=True)
        rx_to_media_offset = 61315
        adapter = V90AnaloguePhase2Adapter(
            caller, rx_to_media=lambda sample: sample + rx_to_media_offset)
        adapter.start(62900)
        adapter.feed_tx_complete("info0a_complete", media_at=62973,
                                 tx_now=62973)

        # These are probe-5's recorded receive and media event positions.
        sequence = (
            ("info0d", 3110, 64425, 64480),
            ("tone_b", 3200, 64515, 64640),
            ("tone_b_reversal", 5443, 66758, 66880),
            ("tone_b", 10200, 71514, 71680),
            ("tone_b_reversal", 16163, 77478, 77600),
            ("l1_complete", 17540, 78855, 78880),
            ("l2_complete", 21540, 82855, 82880),
        )
        routed = []
        actions = []
        for kind, rx_at, media_at, tx_now in sequence:
            event = AnaloguePhase2RxEvent(kind, rx_at, rx_at, {})
            target = adapter._route(event)
            routed.append(None if target is None else target[0])
            if target is None:
                continue
            name, sample = target
            payload = {}
            if name in ("l1_complete", "l2_complete"):
                payload["receive_sample_at"] = sample + rx_to_media_offset
            actions.extend(caller.feed(name, media_at, now=tx_now,
                                       payload=payload))

        self.assertEqual(routed, [
            "info0d", "tone_b", "tone_b_reversal", "tone_b_after_probes",
            "early_peer_recovery_tone_b_reversal", "l1_complete",
            "l2_complete",
        ])
        self.assertTrue(caller.early_peer_recovery_accepted)
        self.assertTrue(caller._remote_l1_complete)
        self.assertTrue(caller._remote_l2_complete)
        self.assertEqual(caller.state, caller.WAIT_INFO1D)
        self.assertTrue(any(action.signal == "tone_a" for action in actions))

    def test_opt_in_peer_recovery_accepts_reversal_before_late_local_echo(self):
        caller = V90AnalogueCallerPhase2(
            late_l2_echo_policy="bounded", late_l2_echo_max_samples=20000,
            accept_early_peer_recovery=True)
        adapter = V90AnaloguePhase2Adapter(
            caller, rx_to_media=lambda sample: sample)
        adapter.start(0)
        caller.feed("info0a_complete", 392)
        caller.feed("info0d", 500, payload={"ack_info0a": 0})
        caller.feed("tone_b", 900)
        caller.feed("tone_b_reversal", 1220)
        caller.feed("tone_b_after_probes", 4000)

        echo_deadline = (caller._local_l2_start_at
                         + 550 * 8 + caller.rtde_samples)
        recovery_reversal = echo_deadline + 500

        last_actions = []

        def route(kind, at):
            nonlocal last_actions
            event = AnaloguePhase2RxEvent(kind, at, at, {})
            routed = adapter._route(event)
            if routed is None:
                return None
            name, sample = routed
            payload = ({"receive_sample_at": sample}
                       if name in ("l1_complete", "l2_complete") else {})
            last_actions = caller.feed(name, sample, now=at, payload=payload)
            return name

        self.assertEqual(route("tone_b_reversal", recovery_reversal),
                         "early_peer_recovery_tone_b_reversal")
        self.assertEqual(caller.state, caller.RX_REMOTE_PROBES)
        l1_end = recovery_reversal + 1378
        l2_end = l1_end + 4000
        self.assertEqual(route("l1_complete", l1_end), "l1_complete")
        self.assertEqual(route("l2_complete", l2_end), "l2_complete")
        self.assertEqual(caller.state, caller.WAIT_INFO1D)
        self.assertTrue(caller.early_peer_recovery_accepted)
        self.assertTrue(any(action.signal == "tone_a"
                            for action in last_actions))
        self.assertEqual(route("info1d", l2_end + 1000), "info1d")
        self.assertEqual(caller.state, caller.COMPLETE)

    def test_opt_in_tone_b_onset_completes_waiting_recovery_reversal(self):
        """A recovery carrier can arrive as a fresh tone after a silent gap."""
        caller = V90AnalogueCallerPhase2(
            late_l2_echo_policy="bounded", late_l2_echo_max_samples=20000,
            accept_early_peer_recovery=True)
        adapter = V90AnaloguePhase2Adapter(
            caller, rx_to_media=lambda sample: sample)
        adapter.start(0)
        caller.feed("info0a_complete", 392)
        caller.feed("info0d", 500, payload={"ack_info0a": 0})
        caller.feed("tone_b", 900)
        caller.feed("tone_b_reversal", 1220)
        caller.feed("tone_b_after_probes", 4000)
        local_l2_end = caller._local_l2_start_at + caller.l2_samples
        caller.feed("l2_echo_complete", local_l2_end, now=local_l2_end)
        self.assertEqual(caller.state,
                         caller.WAIT_TONE_B_REVERSAL_AFTER_PROBES)

        tone = 800.0 * np.cos(
            2 * np.pi * 1200.0 * np.arange(1600) / 8000.0)
        actions = []
        for offset in range(0, len(tone), 160):
            actions.extend(adapter.feed(
                tone[offset:offset + 160], rx_at=17000 + offset,
                tx_now=17160 + offset))
        self.assertEqual(caller.state, caller.RX_REMOTE_PROBES)

        l1_end = 18378
        l2_end = 22378
        for kind, at in (("l1_complete", l1_end),
                         ("l2_complete", l2_end)):
            routed = adapter._route(AnaloguePhase2RxEvent(kind, at, at, {}))
            self.assertEqual(routed, (kind, at))
            caller.feed(kind, at, now=at,
                        payload={"receive_sample_at": at})
        self.assertEqual(caller.state, caller.WAIT_INFO1D)

    def test_paired_normal_peer_does_not_trigger_early_recovery_opt_in(self):
        """The opt-in must leave the ordinary paired exchange on its normal path."""
        for law in ("pcmu", "pcma"):
            with self.subTest(law=law):
                caller, answerer, rtp = self._run_production_paired_phase2(
                    law, echo_delay_samples=120 * 8,
                    phase2_late_l2_echo_policy="bounded",
                    accept_early_peer_recovery=True,
                    echo_gain=0.25)
                try:
                    controller = caller._controller
                    self.assertEqual(controller.state, controller.COMPLETE,
                                     caller.events[-12:])
                    self.assertEqual(answerer._phase2.state,
                                     answerer._phase2.COMPLETE)
                    self.assertFalse(
                        controller.early_peer_recovery_accepted)
                    self.assertFalse(
                        answerer._phase2.missing_tone_a_recovery_started)
                finally:
                    rtp.stop()
                    rtp._sock.close()
                    caller.close()
                    answerer.close()

    def test_paired_digital_peer_timeout_recovery_reaches_v34_lapm_payload(self):
        """Exercise §9.2.1.2.4 against both production Phase 2 engines."""
        caller_payload = bytes((index * 73 + 19) & 0xFF
                               for index in range(1024))
        answer_payload = bytes((index * 151 + 7) & 0xFF
                               for index in range(1024))
        for law in ("pcmu", "pcma"):
            with self.subTest(law=law):
                uncorrected, uncorrected_peer, uncorrected_rtp = (
                    self._run_production_paired_phase2(
                        law, echo_delay_samples=1240 * 8,
                        phase2_late_l2_echo_policy="bounded",
                        phase2_late_l2_echo_max_samples=20000,
                        accept_early_peer_recovery=True))
                try:
                    self.assertEqual(uncorrected._controller.state,
                                     uncorrected._controller.FAILED,
                                     uncorrected.events[-20:])
                    self.assertFalse(any(
                        name == "phase2-local-echo-reference"
                        for name, _ in uncorrected.events))
                finally:
                    uncorrected_rtp.stop()
                    uncorrected_rtp._sock.close()
                    uncorrected.close()
                    uncorrected_peer.close()

                caller, answerer, rtp = self._run_production_paired_phase2(
                    law, echo_delay_samples=1610 * 8,
                    phase2_late_l2_echo_policy="bounded",
                    accept_early_peer_recovery=True,
                    continue_v34_fallback=True,
                    v34_fallback_hold_samples=4 * 8000,
                    cancel_synthetic_echo_after_fallback=True,
                    fallback_payloads=(caller_payload, answer_payload),
                    peer_info0d_fields={"rate3429": 1,
                                        "v90_3429_upstream": 0},
                    peer_info1d_blocks=[{}, {}, {}, {}, {},
                                        {"hi_carrier": 0, "max_rate": 12}])
                try:
                    self.assertTrue(caller._controller._remote_l1_complete,
                                    caller.events[-20:])
                    self.assertTrue(caller._controller._remote_l2_complete,
                                    caller.events[-20:])
                    self.assertTrue(
                        answerer._phase2.missing_tone_a_recovery_started,
                        answerer.events[-20:])
                    self.assertEqual(caller._controller.state,
                                     caller._controller.COMPLETE,
                                     caller.events[-20:])
                    self.assertEqual(answerer._phase2.state,
                                     answerer._phase2.COMPLETE,
                                     answerer.events[-20:])
                    calling = answerer._v34_fallback_engine
                    answering = caller._v34_fallback_engine
                    self.assertIsNotNone(calling)
                    self.assertIsNotNone(answering)
                    self.assertTrue(calling.connected, calling.events[-12:])
                    self.assertTrue(answering.connected, answering.events[-12:])
                    self.assertEqual((calling.v42_mode, answering.v42_mode),
                                     ("lapm", "lapm"))
                    self.assertEqual(caller._test_fallback_payload_rx,
                                     answer_payload)
                    self.assertEqual(answerer._test_fallback_payload_rx,
                                     caller_payload)
                finally:
                    rtp.stop()
                    rtp._sock.close()
                    caller.close()
                    answerer.close()

    def test_paired_info1d_loss_runs_caller_initiated_retrain_then_v34_payload(self):
        """A lost INFO1d must recover through the paired Phase 2 controllers."""
        caller_payload = bytes((index * 31 + 11) & 0xFF
                               for index in range(512))
        answer_payload = bytes((index * 97 + 5) & 0xFF
                               for index in range(512))
        for law in ("pcmu", "pcma"):
            with self.subTest(law=law):
                caller, answerer, rtp = self._run_production_paired_phase2(
                    law, echo_delay_samples=120 * 8,
                    echo_gain=0.25,
                    phase2_late_l2_echo_policy="bounded",
                    accept_early_peer_recovery=True,
                    info1d_timeout_retrain=True,
                    drop_answerer_info1d=True,
                    answerer_info1a_timeout_retrain=True,
                    continue_v34_fallback=True,
                    v34_fallback_hold_samples=2 * 8000,
                    cancel_synthetic_echo_after_fallback=True,
                    fallback_payloads=(caller_payload, answer_payload),
                    peer_info0d_fields={"rate3429": 1,
                                        "v90_3429_upstream": 0},
                    peer_info1d_blocks=[{}, {}, {}, {}, {},
                                        {"hi_carrier": 0, "max_rate": 12}])
                try:
                    self.assertEqual(caller._controller._info1d_retrain_attempts,
                                     1)
                    self.assertEqual(len(answerer._test_dropped_info1d_ranges),
                                     2)
                    self.assertTrue(answerer._phase2.retrain_response_started,
                                    {"answerer": answerer.events[-20:],
                                     "caller": caller.events[-20:],
                                     "caller_state": caller._controller.state,
                                     "answerer_state": answerer._phase2.state,
                                     "masked": answerer._test_dropped_info1d_ranges})
                    self.assertEqual(caller._controller.state,
                                     caller._controller.COMPLETE,
                                     {"caller": caller.events[-20:],
                                      "answerer": answerer.events[-20:],
                                      "answerer_state": answerer._phase2.state,
                                      "answerer_info1a": answerer._phase2.info1a,
                                      "answerer_response": answerer._phase2.retrain_response_started,
                                      "l2_started": answerer._phase2.l2_started_at,
                                      "echo_at": answerer._phase2._second_l2_echo_complete_at,
                                      "answerer_trace": [
                                          {key: row.get(key) for key in
                                           ("event", "media_at", "routed",
                                            "state", "state_after", "actions")}
                                          for row in answerer._adapter.event_trace[-12:]],
                                      "info1d_ranges": answerer._test_dropped_info1d_ranges})
                    self.assertEqual(answerer._phase2.state,
                                     answerer._phase2.COMPLETE,
                                     answerer.events[-20:])
                    calling = answerer._v34_fallback_engine
                    answering = caller._v34_fallback_engine
                    self.assertIsNotNone(calling)
                    self.assertIsNotNone(answering)
                    self.assertTrue(calling.connected, calling.events[-12:])
                    self.assertTrue(answering.connected, answering.events[-12:])
                    self.assertEqual((calling.v42_mode, answering.v42_mode),
                                     ("lapm", "lapm"))
                    self.assertEqual(caller._test_fallback_payload_rx,
                                     answer_payload)
                    self.assertEqual(answerer._test_fallback_payload_rx,
                                     caller_payload)
                finally:
                    rtp.stop()
                    rtp._sock.close()
                    caller.close()
                    answerer.close()

    def test_paired_missing_phase3_sd_retrain_reaches_phase4_boundary(self):
        """Paired Phase-3 recovery reaches Phase 4, not just retrain start."""
        for law in ("pcmu", "pcma"):
            with self.subTest(law=law):
                caller, answerer, rtp = self._run_production_paired_phase2(
                    law, echo_delay_samples=120 * 8, echo_gain=0.25,
                    phase2_late_l2_echo_policy="bounded",
                    accept_early_peer_recovery=True,
                    continue_v34_fallback=True,
                    drop_answerer_phase3_until_caller_retrain=True,
                    answerer_echo_gain_after_caller_phase3=0.05,
                    caller_echo_gain_after_phase3=0.0,
                    answerer_echo_gain_during_phase3_phase2_probe=0.18,
                    fallback_retrain_caller_echo_gain=0.0,
                    fallback_retrain_answerer_echo_gain=0.25)
                try:
                    caller_events = [name for name, _ in caller.events]
                    peer_events = [name for name, _ in answerer.events]
                    self.assertTrue(caller._test_dropped_phase3_ranges,
                                    {"caller": caller.events[-30:],
                                     "peer": answerer.events[-30:]})
                    self.assertEqual(caller_events.count(
                        "phase3-retrain-started"), 1, caller.events[-30:])
                    self.assertIn("retrain-start", peer_events,
                                  {"peer": answerer.events[-30:],
                                   "caller": caller.events[-30:],
                                   "dropped": caller._test_dropped_phase3_ranges[-5:]})
                    self.assertTrue(any(
                        name == "retrain-start" and fields.get("reason") == "peer tone"
                        for name, fields in answerer.events),
                        answerer.events[-30:])
                    peer_start = next(fields for name, fields in answerer.events
                                      if name == "retrain-start")
                    prebuilt = next(fields for name, fields in answerer.events
                                    if name == "retrain-phase2-prebuilt")
                    self.assertTrue(peer_start["preamble_clock_origin"]["listener"])
                    # The earlier implementation added the listener's local
                    # reversal offset to the detection-time offset a second
                    # time, putting this boundary ~2 s late. It now maps near
                    # the paired Tone-A event (inside this 160 ms packet loop).
                    self.assertGreaterEqual(prebuilt["reversal_media_at"],
                                            peer_start["at"])
                    self.assertLessEqual(prebuilt["reversal_media_at"]
                                         - peer_start["at"], 2000)
                    self.assertTrue(any(name == "retrain-phase2-restart"
                                        for name, _ in answerer.events))
                    self.assertEqual(caller._phase3_retrain_attempts, 1)
                    caller_phase3_events = [
                        fields["event"] for name, fields in caller.events
                        if name == "phase3-event"]
                    for required in ("sd_transition", "jd", "jd_prime"):
                        self.assertIn(required, caller_phase3_events,
                                      caller.events[-40:])
                    ready = next((fields for name, fields in caller.events
                                  if name == "phase3-ready"), None)
                    self.assertIsNotNone(ready, caller.events[-40:])
                    self.assertTrue(ready["matches_descriptor"], ready)
                    self.assertIsNotNone(caller._phase4_controller)
                    self.assertEqual(caller._phase4_controller.state,
                                     "wait-mp-prime")
                    self.assertIsNotNone(answerer._phase3)
                    self.assertIsNotNone(answerer._phase3.phase4)
                    self.assertEqual(
                        caller._phase4_controller.constellation,
                        answerer._phase3.phase4_constellation)
                    self.assertEqual(answerer._phase3.phase4.state, "wait-cp")
                finally:
                    rtp.stop()
                    rtp._sock.close()
                    caller.close()
                    answerer.close()








    def test_paired_v90_b1d_failures_walk_down_and_deliver_downstream(self):
        """Three B1d failures walk down the JD mask and deliver answerer data."""
        original_decode = V90AnaloguePhase4Controller._decode_b1d_frame
        forced = {"count": 0}

        def fail_first_b1d(controller, frame):
            bits = original_decode(controller, frame)
            if forced["count"] < 3:
                forced["count"] += 1
                bits = list(bits)
                bits[0] = 0
            return bits

        with mock.patch.object(V90AnaloguePhase4Controller,
                               "_decode_b1d_frame", fail_first_b1d):
            caller, answerer, rtp = self._run_production_paired_phase2(
                "pcmu", echo_delay_samples=120 * 8, echo_gain=0.25,
                phase2_late_l2_echo_policy="bounded",
                accept_early_peer_recovery=True,
                continue_v34_fallback=True,
                answerer_echo_gain_after_caller_phase3=0.0,
                answerer_echo_delay_samples=320,
                caller_echo_gain_after_phase3=0.0,
                answerer_echo_gain_during_phase3_phase2_probe=0.18,
                fallback_retrain_caller_echo_gain=0.0,
                fallback_retrain_answerer_echo_gain=0.25,
                )
        try:
            self.assertEqual(
                forced["count"], 3,
                {"phase4_events": [name for name, _
                                    in caller._phase4_controller.events[-24:]],
                 "selection": caller._phase4_downstream_selection})
            self.assertFalse(caller._failed, caller.events[-24:])
            self.assertFalse(answerer._failed, answerer.events[-24:])
            stepdowns = [fields for name, fields in caller.events
                         if name == "phase4-rate-stepdown"]
            self.assertEqual(len(stepdowns), 3, caller.events[-80:])
            self.assertEqual(stepdowns[0]["failed_drn"], 16)
            self.assertEqual(stepdowns[0]["retry_cap_drn"], 15)
            self.assertEqual(stepdowns[1]["failed_drn"], 15)
            self.assertEqual(stepdowns[1]["retry_cap_drn"], 14)
            self.assertEqual(stepdowns[2]["failed_drn"], 14)
            self.assertEqual(stepdowns[2]["retry_cap_drn"], 13)
            self.assertEqual(caller._phase3_retrain_attempts, 3)
            self.assertEqual(caller._phase4_rate_fallback_attempts, 3)
            starts = [fields["downstream_rate"]["drn"]
                      for name, fields in caller.events
                      if name == "phase4-start"]
            self.assertEqual(starts, [16, 15, 14, 13], caller.events[-100:])
            self.assertEqual(caller._phase4_controller.state,
                             caller._phase4_controller.DATA)
            self.assertEqual(answerer._phase3.phase4.state, "data-ready")

            stats = self._send_v90_phase4_payload_with_echo(
                caller, answerer, rtp, law="pcmu", echo_gain=0.0,
                echo_delay_samples=320, caller_payload=b"",
                answerer_payload=b"Lower-rate login banner\r\n",
                require_bidirectional_ack=False)
            self.assertEqual(stats["answerer_payload_received"],
                             b"Lower-rate login banner\r\n", stats)
            self.assertLessEqual(stats["caller_fcs_errors"], 1, stats)
        finally:
            rtp.stop()
            rtp._sock.close()
            caller.close()
            answerer.close()

    def test_paired_v90_downstream_banner_reaches_analogue_caller(self):
        """Paired direct-media path delivers answerer data without synthetic echo."""
        caller, answerer, rtp = self._run_production_paired_phase2(
            "pcmu", echo_delay_samples=120 * 8, echo_gain=0.25,
            phase2_late_l2_echo_policy="bounded",
            accept_early_peer_recovery=True,
            continue_v34_fallback=True,
            drop_answerer_phase3_until_caller_retrain=True,
            answerer_echo_gain_after_caller_phase3=0.0,
            answerer_echo_delay_samples=320,
            caller_echo_gain_after_phase3=0.0,
            answerer_echo_gain_during_phase3_phase2_probe=0.18,
            fallback_retrain_caller_echo_gain=0.0,
            fallback_retrain_answerer_echo_gain=0.25,
            )
        try:
            self.assertFalse(caller._failed, caller.events[-16:])
            self.assertFalse(answerer._failed, answerer.events[-16:])
            self.assertEqual(caller._phase4_controller.state, "data")
            self.assertEqual(answerer._phase3.phase4.state, "data-ready")
            banner = b"Test BBS\r\nLogin: "
            stats = self._send_v90_phase4_payload_with_echo(
                caller, answerer, rtp, law="pcmu", echo_gain=0.0,
                echo_delay_samples=320,
                caller_payload=b"", answerer_payload=banner,
                require_bidirectional_ack=False)
            self.assertEqual(stats["answerer_payload_received"], banner, stats)
            self.assertEqual(stats["caller_fcs_errors"], 0, stats)
        finally:
            rtp.stop()
            rtp._sock.close()
            caller.close()
            answerer.close()

    def _send_v90_phase4_payload_with_echo(self, caller, answerer, rtp, *,
                                          law, echo_gain,
                                          echo_delay_samples,
                                          echo_path=None,
                                          caller_payload=None,
                                          answerer_payload=None,
                                          require_bidirectional_ack=True):
        if caller_payload is None:
            caller_payload = bytes((index * 29 + 7) & 0xFF
                                   for index in range(512))
        if answerer_payload is None:
            answerer_payload = bytes((index * 53 + 11) & 0xFF
                                     for index in range(512))
        caller._phase4_controller.feed_bytes(caller_payload)
        answerer._phase3.feed_bytes(answerer_payload)
        caller_received = bytearray()
        answerer_received = bytearray()
        origin = caller._clock.origin
        media_at = caller._rx_at
        self.assertEqual((caller._rx_at, caller._tx_media,
                          answerer._rx_at, answerer._tx_media),
                         (media_at, media_at, media_at, media_at))
        tx_history = answerer._test_v90_tx_history
        for tick in range(100):
            at = media_at + tick * 160
            caller_pcm = np.asarray(caller.tx_samples(160), dtype=float)
            answerer_wire = answerer.tx_g711(160, law)
            answerer_pcm = v34_g711.decode(
                np.frombuffer(answerer_wire, dtype=np.uint8), law).astype(float)
            tx_history[at:at + 160] = answerer_pcm
            echo = np.zeros(160, dtype=float)
            if echo_path is None:
                echo += _delayed_echo(tx_history, at, 160,
                                      delay_samples=echo_delay_samples,
                                      gain=echo_gain)
            else:
                for relative_delay, path_gain in echo_path:
                    echo += _delayed_echo(
                        tx_history, at, 160,
                        delay_samples=echo_delay_samples + relative_delay,
                        gain=path_gain)
            answerer_rx_pcm = caller_pcm + echo
            context = {
                "packet_playout_deadline_monotonic": origin + at / 8000,
                "payload_samples": 160,
                "timestamp": at,
            }
            answerer_received.extend(answerer.feed_g711(
                bytes(v34_g711.encode(answerer_rx_pcm, law)), law, context))
            packet = struct.pack(
                "!BBHII", 0x80, rtp._rtp_payload_type,
                (700 + tick) & 0xFFFF, at & 0xFFFFFFFF,
                0x12345678) + answerer_wire
            caller_received.extend(rtp.handle_rtp_packet(
                packet, received_at=origin + at / 8000 + 0.020))
            caller_link = caller._phase4_controller.lapm
            answerer_link = answerer._phase3.lapm
            if (caller_received == answerer_payload
                    and answerer_received == caller_payload
                    and caller_link._v42.stats["wire_acked"]
                    >= len(caller_payload)
                    and (not require_bidirectional_ack
                         or answerer_link._v42.stats["wire_acked"]
                         >= len(answerer_payload))
                    and not caller_link._v42._unacked
                    and (not require_bidirectional_ack
                         or not answerer_link._v42._unacked)):
                break
            self.assertFalse(caller._failed, caller.events[-12:])
            self.assertFalse(answerer._failed, answerer.events[-12:])
        self.assertEqual(bytes(caller_received), answerer_payload)
        self.assertEqual(bytes(answerer_received), caller_payload,
                         (len(answerer_received), answerer.phase3_state,
                          answerer._phase3.upstream.state,
                          len(answerer._phase3._upstream_data_bits),
                          answerer._phase3.phase4_error,
                          answerer._phase3.events[-20:]))
        caller_lapm = caller._phase4_controller.lapm
        answerer_lapm = answerer._phase3.lapm
        self.assertGreaterEqual(caller_lapm._v42.stats["wire_acked"],
                                len(caller_payload))
        if require_bidirectional_ack:
            self.assertGreaterEqual(answerer_lapm._v42.stats["wire_acked"],
                                    len(answerer_payload))
        # A startup FCS can be recovered by LAPM. Payload equality and drained
        # unacked queues remain mandatory regardless of this diagnostic.
        self.assertLessEqual(caller_lapm._v42.stats["fcs_errors"], 1)
        self.assertLessEqual(answerer_lapm._v42.stats["fcs_errors"], 1)
        return {
            "caller_fcs_errors": caller_lapm._v42.stats["fcs_errors"],
            "answerer_fcs_errors": answerer_lapm._v42.stats["fcs_errors"],
            "caller_wire_acked": caller_lapm._v42.stats["wire_acked"],
            "answerer_wire_acked": answerer_lapm._v42.stats["wire_acked"],
            "answerer_payload_received": bytes(caller_received),
        }



    def test_info1d_wait_tone_b_restart_follows_9213_sequence(self):
        """V.90 §9.2.2.2.4 returns to the Tone-B exchange in §9.2.2.1.3."""
        caller = V90AnalogueCallerPhase2(late_l2_echo_policy="continue")
        adapter = V90AnaloguePhase2Adapter(
            caller, rx_to_media=lambda sample: sample)
        adapter.start(0)
        adapter.feed_tx_complete("info0a_complete", media_at=392, tx_now=392)

        def route(kind, at, *, payload=None):
            event = AnaloguePhase2RxEvent(kind, at, at, payload or {})
            target = adapter._route(event)
            self.assertIsNotNone(target, (kind, caller.state))
            name, sample = target
            produced = caller.feed(name, sample, now=at, payload=payload)
            return name, produced

        route("info0d", 500, payload={"ack_info0a": 0})
        route("tone_b", 900)
        route("tone_b_reversal", 1220)
        route("tone_b", 4000)  # returned local L2 echo / Tone B
        echo_end = caller._local_l2_start_at + caller.l2_samples
        caller.feed("l2_echo_complete", echo_end, now=echo_end)
        self.assertEqual(caller.state,
                         caller.WAIT_TONE_B_REVERSAL_AFTER_PROBES)
        route("tone_b_reversal", echo_end + 800)
        route("l1_complete", echo_end + 800 + 1280,
              payload={"receive_sample_at": echo_end + 800 + 1280})
        route("l2_complete", echo_end + 800 + 1280 + 4000,
              payload={"receive_sample_at": echo_end + 800 + 1280 + 4000})
        self.assertEqual(caller.state, caller.WAIT_INFO1D)

        restart_at = echo_end + 800 + 1280 + 4000 + 200
        actions = []
        peer_reversal_at = None
        for offset in range(0, 2 * 8000, 160):
            absolute = restart_at + offset + np.arange(160)
            phase = 2 * np.pi * 1200.0 * (absolute - restart_at) / 8000.0
            polarity = (np.ones(160) if peer_reversal_at is None else
                        np.where(absolute < peer_reversal_at, 1.0, -1.0))
            block_actions = adapter.feed(
                800.0 * np.cos(phase) * polarity,
                rx_at=restart_at + offset,
                tx_now=restart_at + offset + 160)
            actions.extend(block_actions)
            reversal = next((action.at for action in block_actions
                             if action.signal == "reverse-tone-a"), None)
            if reversal is not None:
                # Peer B reverses 40 ms after it receives the caller's A
                # reversal, matching the digital-side turnaround.
                peer_reversal_at = reversal + 40 * 8
            if caller.state == caller.TX_LOCAL_PROBES:
                break

        self.assertIsNotNone(peer_reversal_at)
        names = [action.signal for action in actions]
        self.assertEqual(names[:3], ["tone_a", "reverse-tone-a",
                                     "listen-tone-b-reversal"])
        self.assertEqual(names[3:], ["reverse-tone-a", "l1", "l2",
                                     "listen-tone-b"])
        reversal_at = [action.at for action in actions
                       if action.signal == "reverse-tone-a"][1]
        self.assertGreaterEqual(reversal_at - restart_at, 50 * 8)
        self.assertEqual(peer_reversal_at - reversal_at, 40 * 8)
        self.assertEqual(caller.state, caller.TX_LOCAL_PROBES)
        route("tone_b", reversal_at + 5000)
        self.assertEqual(caller.state, caller.WAIT_TONE_B_AFTER_PROBES)
        echo2_end = caller._local_l2_start_at + caller.l2_samples
        caller.feed("l2_echo_complete", echo2_end, now=echo2_end)
        route("tone_b_reversal", echo2_end + 800)
        l1_end = echo2_end + 800 + 1280
        route("l1_complete", l1_end,
              payload={"receive_sample_at": l1_end})
        l2_end = l1_end + 4000
        route("l2_complete", l2_end,
              payload={"receive_sample_at": l2_end})
        route("info1d", l2_end + 500, payload={"blocks": [{}] * 6})
        self.assertEqual(caller.state, caller.COMPLETE)

    def test_info1d_tone_b_restart_reacts_with_paired_digital_controller(self):
        """Close the restart's PCM loop through both production controllers."""
        caller = V90AnalogueCallerPhase2(late_l2_echo_policy="continue")
        caller_rx = V90AnaloguePhase2Adapter(
            caller, rx_to_media=lambda sample: sample)
        caller_rx.start(0)
        caller_rx.feed_tx_complete("info0a_complete", media_at=392,
                                   tx_now=392)

        # Establish the prior exchange at the exact state where the caller
        # receives a fresh Tone B instead of INFO1d. The peer response below is
        # driven by its own controller and PCM transmitter, not injected event
        # labels.
        for kind, at, payload in (
                ("info0d", 500, {"ack_info0a": 0}),
                ("tone_b", 900, {}),
                ("tone_b_reversal", 1220, {}),
                ("tone_b", 4000, {})):
            routed = caller_rx._route(
                AnaloguePhase2RxEvent(kind, at, at, payload))
            self.assertIsNotNone(routed, (kind, caller.state))
            caller.feed(routed[0], routed[1], now=at, payload=payload)
        echo_end = caller._local_l2_start_at + caller.l2_samples
        caller.feed("l2_echo_complete", echo_end, now=echo_end)
        at = echo_end + 800
        for kind, payload in (
                ("tone_b_reversal", {}),
                ("l1_complete", {"receive_sample_at": at + 1280}),
                ("l2_complete", {"receive_sample_at": at + 5280})):
            routed = caller_rx._route(
                AnaloguePhase2RxEvent(kind, at, at, payload))
            self.assertIsNotNone(routed, (kind, caller.state))
            caller.feed(routed[0], routed[1], now=at, payload=payload)
            at += 1280 if kind == "tone_b_reversal" else 4000
        self.assertEqual(caller.state, caller.WAIT_INFO1D)

        restart_at = at + 200
        law = "pcmu"
        caller_tx = V90AnaloguePhase2Transmitter(
            info0a_fields={"rate3429": 1, "disallow3429": 1},
            info1a_builder=lambda mode: v90.build_info1a_v90(
                md_len=0, uinfo=90, analog_symbol_rate=5,
                digital_symbol_rate=6, freq_offset=0),
            probe_builder=v34_call.probing_signal,
            nominal_peak=1200, probe_rms=300, start_at=restart_at)

        # INFO0a was already negotiated in the interrupted exchange. Seed that
        # protocol fact, start the peer's real controller and render the fresh
        # Tone B that the caller's §9.2.2.2.4 handler must recognize.
        digital = V90DigitalPhase2Controller()
        digital.info0a = {"ack_info0d": 0}
        digital_rx = V90DigitalPhase2Adapter(
            digital, rx_to_media=lambda sample: sample, trace_events=True)
        digital_rx.start(restart_at)
        digital_tx = V90DigitalPhase2Transmitter(
            signals=V90DigitalPhase2Signals(
                info0d={"rate3429": 1, "alaw": 0}, amplitude=1200),
            probe_builder=v34_call.probing_signal,
            tone_b_peak=800, probe_rms=300, start_at=restart_at)
        # The restarted exchange begins with the peer's Tone B; INFO0d belongs
        # to the already-completed negotiation and is not retransmitted here.
        digital_tx.submit([Phase2Action(restart_at, "tone_b")])

        caller_actions = []
        digital_actions = []
        restart_pair_entered = False
        caller_echo_completed = False
        digital_echo_completed = False
        for start in range(restart_at, restart_at + 10 * 8000, 160):
            caller_pcm = v34_g711.roundtrip(caller_tx.render(160), law)
            digital_pcm = v34_g711.roundtrip(digital_tx.render(160), law)
            # Model a small zero-delay hybrid return while the opposite
            # controller supplies real PCM. L2 completion is signaled below
            # as a timed fixture event; the echo itself is not used to detect it.
            caller_echo = 0.10 * caller_pcm
            digital_echo = 0.10 * digital_pcm
            now = start + 160
            for action in caller_rx.feed(
                    digital_pcm + caller_echo, rx_at=start, tx_now=now):
                caller_actions.append(action.signal)
                caller_tx.submit([action])
            for action in digital_rx.feed(
                    caller_pcm + digital_echo, rx_at=start, tx_now=now):
                digital_actions.append(action.signal)
                digital_tx.submit([action])
            if (caller.state == caller.TX_LOCAL_PROBES
                    and digital.state == digital.RX_PROBE):
                restart_pair_entered = True
            if (not caller_echo_completed
                    and caller._local_l2_start_at is not None
                    and now >= caller._local_l2_start_at + caller.l2_samples):
                echo_at = caller._local_l2_start_at + caller.l2_samples
                if caller.state in (caller.TX_LOCAL_PROBES,
                                    caller.WAIT_TONE_B_AFTER_PROBES):
                    for action in caller_rx.feed_l2_echo_complete(
                            rx_at=echo_at, tx_now=now):
                        caller_actions.append(action.signal)
                        caller_tx.submit([action])
                caller_echo_completed = True
            if (not digital_echo_completed
                    and digital.state == digital.TX_PROBE
                    and digital.l2_started_at is not None
                    and now >= digital.l2_started_at + digital.max_l2_samples):
                echo_at = digital.l2_started_at + digital.max_l2_samples
                for action in digital.feed(
                        "l2_echo_complete", echo_at, now=now):
                    digital_actions.append(action.signal)
                    digital_tx.submit([action])
                digital_echo_completed = True
            if caller.state == caller.COMPLETE and digital.state == digital.COMPLETE:
                break

        self.assertTrue(restart_pair_entered,
                        (caller.state, digital.state,
                         caller_rx.event_trace[-16:],
                         digital_rx.event_trace[-16:]))
        self.assertTrue(caller_echo_completed)
        self.assertTrue(digital_echo_completed)
        self.assertEqual(caller.state, caller.COMPLETE,
                         caller_rx.event_trace[-16:])
        self.assertEqual(digital.state, digital.COMPLETE,
                         digital_rx.event_trace[-16:])
        self.assertEqual(caller.mode, 6)
        self.assertIsNotNone(digital.info1a)
        self.assertIn("info1d", digital_actions)
        self.assertEqual(caller_actions[:3], [
            "tone_a", "reverse-tone-a", "listen-tone-b-reversal"])
        self.assertIn("reverse-tone-b", digital_actions)
        self.assertEqual(digital._a_rev_count, 2)

    def test_info1d_restart_accepts_waveform_probes_and_info1d(self):
        """Exercise the restart's receive half through production PCM detectors."""
        caller = V90AnalogueCallerPhase2(late_l2_echo_policy="continue")
        adapter = V90AnaloguePhase2Adapter(
            caller, rx_to_media=lambda sample: sample)
        adapter.start(0)
        adapter.feed_tx_complete("info0a_complete", media_at=392,
                                 tx_now=392)

        # Establish the preceding exchange up to WAIT_INFO1D. This isolates
        # §9.2.2.2.4; the restart response below is actual PCM, not injected
        # Tone-B/L1/L2/INFO1d event labels.
        for kind, at, payload in (
                ("info0d", 500, {"ack_info0a": 0}),
                ("tone_b", 900, {}),
                ("tone_b_reversal", 1220, {}),
                ("tone_b", 4000, {})):
            routed = adapter._route(AnaloguePhase2RxEvent(kind, at, at, payload))
            self.assertIsNotNone(routed, (kind, caller.state))
            caller.feed(routed[0], routed[1], now=at, payload=payload)
        echo_end = caller._local_l2_start_at + caller.l2_samples
        caller.feed("l2_echo_complete", echo_end, now=echo_end)
        self.assertEqual(caller.state,
                         caller.WAIT_TONE_B_REVERSAL_AFTER_PROBES)
        at = echo_end + 800
        for kind, payload in (
                ("tone_b_reversal", {}),
                ("l1_complete", {"receive_sample_at": at + 1280}),
                ("l2_complete", {"receive_sample_at": at + 5280})):
            routed = adapter._route(AnaloguePhase2RxEvent(kind, at, at, payload))
            self.assertIsNotNone(routed, (kind, caller.state))
            caller.feed(routed[0], routed[1], now=at, payload=payload)
            at += 1280 if kind == "tone_b_reversal" else 4000
        self.assertEqual(caller.state, caller.WAIT_INFO1D)

        rx_at = at + 200
        actions = []
        peer_reversal_at = None
        # The fresh 1200 Hz carrier restarts §9.2.2.1.3. The peer reverses
        # 40 ms after the caller's reversal, as in the paired digital peer.
        for _ in range(2 * 8000 // 160):
            absolute = rx_at + np.arange(160)
            polarity = (np.ones(160) if peer_reversal_at is None else
                        np.where(absolute < peer_reversal_at, 1.0, -1.0))
            samples = 800.0 * np.cos(
                2 * np.pi * 1200.0 * absolute / 8000.0) * polarity
            produced = adapter.feed(samples, rx_at=rx_at, tx_now=rx_at + 160)
            actions.extend(produced)
            reversal = next((action.at for action in produced
                             if action.signal == "reverse-tone-a"), None)
            if reversal is not None:
                peer_reversal_at = reversal + 40 * 8
            rx_at += 160
            if caller.state == caller.TX_LOCAL_PROBES:
                break
        self.assertIsNotNone(peer_reversal_at)
        self.assertEqual(caller.state, caller.TX_LOCAL_PROBES)

        # Model the local hybrid return as a timed completion event, then let
        # the digital peer's fresh Tone B onset be discovered from PCM.
        local_l2_end = caller._local_l2_start_at + caller.l2_samples
        echo_at = local_l2_end + 800
        while rx_at < echo_at:
            count = min(160, echo_at - rx_at)
            adapter.feed(np.zeros(count), rx_at=rx_at, tx_now=rx_at + count)
            rx_at += count
        adapter.feed_l2_echo_complete(rx_at=echo_at, tx_now=echo_at)
        rx_at = echo_at
        tone_b_start = rx_at + 160
        while rx_at < tone_b_start:
            adapter.feed(np.zeros(160), rx_at=rx_at, tx_now=rx_at + 160)
            rx_at += 160
        remote_reversal = tone_b_start + 1600
        for _ in range(40):
            absolute = rx_at + np.arange(160)
            polarity = np.where(absolute < remote_reversal, 1.0, -1.0)
            tone = (800.0 * np.cos(
                2 * np.pi * 1200.0 * absolute / 8000.0) * polarity)
            actions.extend(adapter.feed(tone, rx_at=rx_at,
                                        tx_now=rx_at + 160))
            rx_at += 160
            if caller.state == caller.RX_REMOTE_PROBES:
                break
        self.assertEqual(caller.state, caller.RX_REMOTE_PROBES,
                         adapter.event_trace[-12:])
        while rx_at < remote_reversal + 400:
            adapter.feed(np.zeros(160), rx_at=rx_at, tx_now=rx_at + 160)
            rx_at += 160
        l1 = v34_call.probing_signal(.160, 300)
        l2 = v34_call.probing_signal(.500, 300)
        probes = np.concatenate((l1, l2))
        for offset in range(0, len(probes), 160):
            block = probes[offset:offset + 160]
            actions.extend(adapter.feed(block, rx_at=rx_at,
                                        tx_now=rx_at + len(block)))
            rx_at += len(block)
        # The comb detector confirms onset using two 20 ms windows and emits
        # the protocol-duration event only once the full receive window has
        # elapsed; provide its confirmation tail before asserting L2.
        for _ in range(12):
            adapter.feed(np.zeros(160), rx_at=rx_at, tx_now=rx_at + 160)
            rx_at += 160
        self.assertTrue(caller._remote_l1_complete,
                        adapter.event_trace[-12:])
        self.assertTrue(caller._remote_l2_complete,
                        adapter.event_trace[-12:])
        self.assertEqual(caller.state, caller.WAIT_INFO1D)

        fields = {
            "power_min": 2, "power_extra": 3, "md_len": 0,
            "blocks": [{"hi_carrier": 0, "preemph": 1, "max_rate": 14}
                       for _ in range(6)],
            "freq_offset": 0,
        }
        signals = V90DigitalPhase2Signals(info1d=fields, amplitude=3000)
        signals.set_phase_at_sample(rx_at)
        info1d = signals.modulate_info1d()
        for offset in range(0, len(info1d), 160):
            block = info1d[offset:offset + 160]
            actions.extend(adapter.feed(block, rx_at=rx_at,
                                        tx_now=rx_at + len(block)))
            rx_at += len(block)
        # The rolling scanner may need its next 200 ms scan boundary after
        # the complete frame is present in the 1.5 s analysis window.
        for _ in range(12):
            adapter.feed(np.zeros(160), rx_at=rx_at, tx_now=rx_at + 160)
            rx_at += 160
        self.assertEqual(caller.state, caller.COMPLETE,
                         adapter.event_trace[-12:])
        self.assertEqual(caller.mode, 6)

    def test_reactive_phase2_to_phase4_b1_pair_on_one_media_clock(self):
        """Carry paired startup through concurrent B1/B1d and LAPM payload."""
        for law in ("pcmu", "pcma"):
            with self.subTest(law=law):
                # Route the analogue caller's Phase 2 receive stream through
                # the production RTP timestamp-gap dispatcher. The selected
                # 40 ms gap is outside the short Tone-B reversal, so the
                # handshake should recover and the same pair must still carry
                # exact LAPM payload after B1/B1d.
                engine, _, caller_phase2 = self._run_engine_phase2(
                    law, caller_rtp_gaps=((12000, 320),))
                try:
                    self.assertEqual(caller_phase2.state, caller_phase2.COMPLETE)
                    self.assertEqual(caller_phase2._test_rtp_gap_samples, [320])
                    self.assertEqual(engine.phase2_state, engine._phase2.COMPLETE)
                    # This test makes assertions about the receiver's
                    # diagnostic B1 trace. The production default keeps that
                    # allocation disabled, so opt it in explicitly here.
                    engine._phase3._capture_symbol_trace = True
                    media_at = engine._rx_at
                    ja = ja_control._ja_descriptor(0, 19, 31)
                    descriptor = v90.parse_ja_descriptor(ja)
                    caller_controller = V90AnalogueCallerPhase3(
                        baud=ja_control.BAUD, ja_bits=ja,
                        descriptor=descriptor)
                    caller_tx = V90AnaloguePhase3Transmitter(
                        caller_controller, carrier=ja_control.CARRIER,
                        rms=2000.0)
                    caller_rx = V90AnalogueDownstreamPhase3Receiver(
                        engine._phase3.transmitter.uinfo, law)
                    caller = V90AnaloguePhase3Session(
                        caller_controller, caller_rx, caller_tx)
                    caller.start(media_at)
                    common = dict(
                        drn=7, selectors=(0, 1, 2, 0, 1, 2),
                        tx_masks=(tuple(range(128)),) * 3,
                        spectral_sign_bits=0)
                    phase4 = V90AnaloguePhase4Controller(
                        cpt_fields={
                            "analog_rate_mask": 0x1081, "codec_masks": None,
                            "lookahead": 1, "gain_q13": 0x4000,
                            "shape_q6": (0x10, -16, 0x08, -24), **common},
                        cp_fields={
                            "analog_rate_mask": 0x1081, "codec_masks": None,
                            "lookahead": 1, "gain_q13": 0x4000,
                            "shape_q6": (0x10, -16, 0x08, -24), **common},
                        law=law, constellation=4, baud=ja_control.BAUD,
                        carrier=ja_control.CARRIER)
                    phase4_started = False
                    phase4_wire = bytearray()
                    phase4_initial_octets = 0
                    phase4_cpt_decisions = None

                    completed = False
                    phase3_events = []
                    # A SIP/PBX codec path may canonicalize μ-law's two zero
                    # codewords without changing the PCM waveform. Reproduce
                    # the captured wire variation only across the initial
                    # S_d/S'_d markers, leaving TRN, J_d, Phase 4, and PCMA
                    # byte-exact. The marker occupies 384 + 48 symbols.
                    marker_octets_left = 432
                    marker_octets_canonicalized = 0

                    def canonicalize_marker_zeroes(payload):
                        nonlocal marker_octets_left
                        nonlocal marker_octets_canonicalized
                        if (law != "pcmu" or marker_octets_left == 0
                                or engine._phase3 is None
                                or not engine._phase3.transmit_ready):
                            return payload
                        count = min(marker_octets_left, len(payload))
                        prefix = payload[:count].replace(b"\x7f", b"\xff")
                        marker_octets_canonicalized += sum(
                            1 for before, after in zip(payload[:count], prefix)
                            if before == 0x7f and after == 0xff)
                        marker_octets_left -= count
                        return prefix + payload[count:]

                    for _ in range(500):
                        downstream = engine.tx_g711(160, law)
                        downstream = canonicalize_marker_zeroes(downstream)
                        self.assertEqual(len(downstream), 160)
                        events = caller.feed_downstream_g711(
                            downstream, at=media_at)
                        phase3_events.extend(events)
                        jd_prime = next((event for event in events
                                         if event["event"] == "jd_prime"), None)
                        if jd_prime is not None and not phase4_started:
                            phase4_cpt_decisions = phase4.start()
                            phase4_started = True
                            # The timestamp is the first sample after J′d;
                            # start parsing the exact Phase 4 stream there.
                            offset = max(0, min(
                                len(downstream), jd_prime["at"] - media_at))
                            phase4_wire.extend(downstream[offset:])
                            phase4.feed_downstream(downstream[offset:])
                            phase4_initial_octets += len(downstream[offset:])
                        pcm = caller_tx.read_pcm(160)
                        engine.feed_g711(
                            bytes(v34_g711.encode(pcm, law)), law, {
                                "packet_playout_deadline_monotonic":
                                    engine._clock.origin + media_at / 8000,
                                "payload_samples": 160,
                                "timestamp": media_at,
                            })
                        media_at += 160
                        if (caller_controller.phase4_ready
                                and engine._phase3 is not None
                                and engine._phase3.phase4 is not None
                                and engine._phase3._phase4_receive_armed):
                            completed = True
                            break
                        self.assertFalse(
                            engine._failed,
                            (law, engine.phase2_state, engine.events[-8:]))

                    self.assertTrue(completed, (
                        law, caller_controller.state,
                        caller_rx.state,
                        [event["event"] for event in phase3_events],
                        None if engine._phase3 is None
                        else engine._phase3.transmitter.state,
                        [] if engine._phase3 is None
                        else engine._phase3.events[-8:],
                        caller_rx.codeword_errors,
                    ))
                    self.assertEqual(caller_controller.state,
                                     caller_controller.PHASE4_READY)
                    self.assertIn("sd_transition",
                                  [event["event"] for event in phase3_events])
                    if law == "pcmu":
                        self.assertEqual(marker_octets_left, 0)
                        self.assertGreater(marker_octets_canonicalized, 0)
                    else:
                        self.assertEqual(marker_octets_canonicalized, 0)
                    self.assertEqual(caller_rx.codeword_errors, 0)
                    self.assertEqual(caller_tx.late_stop_samples, 0)
                    self.assertFalse(engine._renderer.late_actions)

                    # Continue the same RTP-facing answerer into its actual
                    # R-i output and acquire it with the caller's production
                    # Phase 4 receive parser. MP follows only after CPt.
                    self.assertTrue(phase4_started)
                    ri_seen = phase4._ri_pattern is not None
                    for _ in range(100):
                        downstream = engine.tx_g711(160, law)
                        phase4_wire.extend(downstream)
                        phase4.feed_downstream(downstream)
                        phase4_initial_octets += len(downstream)
                        pcm = caller_tx.read_pcm(160)
                        engine.feed_g711(
                            bytes(v34_g711.encode(pcm, law)), law, {
                                "packet_playout_deadline_monotonic":
                                    engine._clock.origin + media_at / 8000,
                                "payload_samples": 160,
                                "timestamp": media_at,
                            })
                        media_at += 160
                        if phase4._ri_pattern is not None:
                            ri_seen = True
                            break
                        self.assertFalse(engine._failed,
                                         (law, engine.events[-8:]))
                    self.assertTrue(ri_seen, (
                        law, phase4._ri_pattern,
                        [name for name, _ in phase4.events],
                        phase4_wire[:48].hex(),
                        engine._phase3.events[-8:]))
                    self.assertEqual(phase4._ri_pattern,
                                     ri_codewords(
                                         engine._phase3.transmitter.uinfo, law,
                                         duration_symbols=24))

                    # Keep the caller quiet until the answerer's mandatory
                    # initial R-i interval has elapsed, then put the caller's
                    # actual shaped CPt waveform back through the RTP-facing
                    # receiver on this same sample clock.
                    while phase4_initial_octets < 320:
                        downstream = engine.tx_g711(160, law)
                        phase4_wire.extend(downstream)
                        phase4.feed_downstream(downstream)
                        phase4_initial_octets += len(downstream)
                        pcm = caller_tx.read_pcm(160)
                        engine.feed_g711(
                            bytes(v34_g711.encode(pcm, law)), law, {
                                "packet_playout_deadline_monotonic":
                                    engine._clock.origin + media_at / 8000,
                                "payload_samples": 160,
                                "timestamp": media_at,
                            })
                        media_at += 160

                    phase4_media = V90AnaloguePhase4MediaTx(
                        phase4, baud=ja_control.BAUD,
                        carrier=ja_control.CARRIER, rms=2000.0)
                    phase4_media.start(decisions=phase4_cpt_decisions)
                    cpt_received = False
                    cp_selected = False
                    answerer_cp_selected = False
                    for _ in range(500):
                        downstream = engine.tx_g711(160, law)
                        phase4_wire.extend(downstream)
                        phase4_media.feed_downstream(downstream)
                        pcm = phase4_media.read_pcm(160)
                        engine.feed_g711(
                            bytes(v34_g711.encode(pcm, law)), law, {
                                "packet_playout_deadline_monotonic":
                                    engine._clock.origin + media_at / 8000,
                                "payload_samples": 160,
                                "timestamp": media_at,
                            })
                        media_at += 160
                        cpt_received = bool(
                            engine._phase3._phase4_cpt_acquired)
                        cp_selected = phase4.state == phase4.WAIT_MP_PRIME
                        answerer_cp_selected = (
                            engine._phase3.phase4.transmitter.state
                            == engine._phase3.phase4.transmitter.WAIT_CP_PRIME)
                        if cp_selected and answerer_cp_selected:
                            break
                        self.assertFalse(engine._failed,
                                         (law, engine.events[-8:]))
                    self.assertTrue(cpt_received, (
                        law, engine.events[-16:],
                        engine._phase3.events[-20:],
                        phase4.state, phase4.events[-8:],
                        engine._phase3._phase4_cpt_active_run,
                        engine._phase3._phase4_cpt_quiet_count,
                        len(engine._phase3._phase4_cpt_candidates),
                        len(engine._phase3._phase4_cpt_recent),
                        engine._phase3._phase4_cpt_recent[-5:],
                        engine._phase3.upstream.receiver.eq._input_guard.reference,
                        len(engine._phase3.upstream.receiver.indices)))
                    self.assertTrue(any(
                        event[0] == "phase4-profile-accepted"
                        and event[1].get("kind") == "cpt"
                        for event in engine._phase3.events))
                    self.assertTrue(cp_selected, (
                        f"{law}: caller={phase4.state} fail={phase4.events[-1:]}, "
                        f"ri_phase={phase4._ri_alignment_phase} "
                        f"awaiting_trn2d={phase4._awaiting_trn2d} "
                        f"pending={bytes(phase4._downstream_pending[:24]).hex()}, "
                        f"answerer={engine._phase3.phase4.state} "
                        f"answerer_events={engine._phase3.phase4.events[-4:]}, "
                        f"profiles={[(e[0], e[1].get('kind'), e[1].get('ack')) for e in engine._phase3.events if e[0] == 'phase4-profile-accepted']}"))
                    self.assertTrue(answerer_cp_selected, (
                        law, engine._phase3.phase4.transmitter.state,
                        engine._phase3.phase4.events[-8:],
                        engine._phase3.events[-12:]))

                    # The caller now receives MP′, repeats CP′ until the
                    # answerer's Ed, then returns E+B1 and qualifies B1d.
                    b1d_qualified = False
                    for _ in range(500):
                        downstream = engine.tx_g711(160, law)
                        phase4_wire.extend(downstream)
                        phase4_media.feed_downstream(downstream)
                        pcm = phase4_media.read_pcm(160)
                        engine.feed_g711(
                            bytes(v34_g711.encode(pcm, law)), law, {
                                "packet_playout_deadline_monotonic":
                                    engine._clock.origin + media_at / 8000,
                                "payload_samples": 160,
                                "timestamp": media_at,
                            })
                        media_at += 160
                        b1d_qualified |= phase4.state == phase4.DATA
                        e_reached_answerer = any(
                            event[0] == "phase4-e-detected"
                            for event in engine._phase3.events)
                        upstream_qualified = (
                            engine._phase3.upstream_data is not None
                            and engine._phase3.upstream_data.state
                            == engine._phase3.upstream_data.DATA)
                        if (b1d_qualified and e_reached_answerer
                                and upstream_qualified):
                            break
                        self.assertFalse(engine._failed,
                                         (law, engine.events[-8:]))
                    self.assertTrue(b1d_qualified, (
                        law, phase4.state, phase4.events[-12:],
                        engine._phase3.phase4.state,
                        engine._phase3.phase4.events[-12:],
                        engine._phase3.events[-16:], engine.events[-8:]))
                    self.assertIn(("e-sent", 20), phase4.events)
                    self.assertTrue(any(
                        event[0] == "phase4-e-detected"
                        for event in engine._phase3.events),
                        engine._phase3.events[-20:])
                    self.assertTrue(any(
                        event[0] == "upstream-b1-start"
                        for event in engine._phase3.events),
                        engine._phase3.events[-20:])
                    upstream = engine._phase3.upstream_data
                    self.assertEqual(
                        phase4_media.controller.upstream_transmitter.b1_sent,
                        True)
                    self.assertIsNotNone(phase4_media.upstream_b1_symbols)
                    self.assertEqual(upstream.b1_start_symbol,
                                     next(value for name, value
                                          in upstream.events
                                          if name == "b1-start"))
                    b1_trace = engine._phase3._upstream_b1_trace
                    self.assertGreaterEqual(
                        sum(index >= upstream.b1_start_symbol
                            for index, _, _ in b1_trace),
                        upstream.b1_symbols_required)
                    self.assertTrue(all(
                        index < upstream.b1_start_symbol
                        for index, _, _ in b1_trace[:1]))
                    self.assertEqual(upstream.state, upstream.DATA, (
                        law, upstream.failure_reason, upstream.events[-4:],
                        len(upstream._b1), upstream.next_symbol,
                        engine._phase3._phase4_receive_armed,
                        len(engine._phase3.upstream.receiver.indices),
                        engine._phase3.upstream.state,
                        {key: upstream.receiver.diagnostics.get(key)
                         for key in ("qualified", "bit_errors", "bit_count",
                                     "held_out_nmse", "gain_real", "gain_imag")}))
                    self.assertTrue(any(
                        event[0] == "b1-qualified"
                        for event in upstream.events), upstream.events[-8:])
                    self.assertFalse(engine._failed)

                    caller_payload = bytes(
                        (index * 29 + 7) & 0xFF for index in range(96))
                    answerer_payload = bytes(
                        (index * 53 + 11) & 0xFF for index in range(96))
                    phase4_media.controller.feed_bytes(caller_payload)
                    engine.feed_bytes(answerer_payload)
                    caller_received = bytearray()
                    answerer_received = bytearray()
                    for _ in range(800):
                        downstream = engine.tx_g711(160, law)
                        phase4_media.feed_downstream(downstream)
                        pcm = phase4_media.read_pcm(160)
                        answerer_received.extend(engine.feed_g711(
                            bytes(v34_g711.encode(pcm, law)), law, {
                                "packet_playout_deadline_monotonic":
                                    engine._clock.origin + media_at / 8000,
                                "payload_samples": 160,
                                "timestamp": media_at,
                            }))
                        caller_received.extend(
                            phase4_media.controller.take_received())
                        media_at += 160
                        if (caller_received == answerer_payload
                                and answerer_received == caller_payload):
                            break
                        self.assertFalse(engine._failed,
                                         (law, engine.events[-8:]))
                    self.assertEqual(bytes(caller_received), answerer_payload,
                                     (law, len(caller_received),
                                      phase4_media.controller.lapm.v42_mode))
                    self.assertEqual(bytes(answerer_received), caller_payload,
                                     (law, len(answerer_received),
                                      engine._phase3.lapm.v42_mode,
                                      phase4_media.controller.lapm.v42_mode))
                    self.assertEqual(
                        (phase4_media.controller.lapm.v42_mode,
                         engine._phase3.lapm.v42_mode),
                        ("lapm", "lapm"))
                finally:
                    engine.close()

    def test_production_answerer_engine_reaches_phase4_cpt_on_one_media_clock(self):
        """Join paired Phase 2, zero-DIL Phase 3 and decoded Phase 4 CPt."""
        for law in ("pcmu", "pcma"):
            with self.subTest(law=law):
                engine, caller_tx, caller = self._run_engine_phase2(law)
                try:
                    self.assertEqual(caller.state, caller.COMPLETE)
                    self.assertFalse(engine._failed)
                    samples = _zero_dil_phase3_to_cpt_pcm(law) * 20000.0
                    for offset in range(0, len(samples), 160):
                        chunk = samples[offset:offset + 160]
                        if len(chunk) != 160:
                            chunk = np.pad(chunk, (0, 160 - len(chunk)))
                        # TX and RX advance on the same 20 ms media clock, as
                        # they do in the RTP-facing engine. The incoming PCM
                        # is one continuous paired caller waveform.
                        engine.tx_g711(160, law)
                        media_at = engine._rx_at
                        payload = bytes(v34_g711.encode(chunk, law))
                        engine.feed_g711(payload, law, {
                            "packet_playout_deadline_monotonic":
                                engine._clock.origin + media_at / 8000,
                            "payload_samples": len(payload),
                            "timestamp": media_at,
                        })
                        if engine._failed:
                            self.fail(
                                f"production V.90 phase 3 failed: {engine.events[-8:]}")

                    phase3 = engine._phase3
                    self.assertTrue(phase3.transmit_ready)
                    self.assertEqual(phase3.transmitter.state,
                                     phase3.transmitter.PHASE4_READY)
                    self.assertIsNotNone(phase3.phase4)
                    self.assertTrue(phase3._phase4_cpt_acquired,
                                    phase3.events[-12:])
                    self.assertTrue(any(
                        name == "phase4-profile-accepted"
                        and profile.get("kind") == "cpt"
                        for name, profile in phase3.events))
                    self.assertEqual(phase3.phase4.state,
                                     phase3.phase4.transmitter.WAIT_CP)
                    self.assertFalse(engine._failed)
                    self.assertFalse(caller_tx.late_actions)
                    self.assertFalse(engine._renderer.late_actions)
                finally:
                    engine.close()

    def test_production_answerer_engine_receives_lapm_payload_after_phase2(self):
        """Exercise Phase 2 through B1 and LAPM payload on the engine clock."""
        for law in ("pcmu", "pcma"):
            with self.subTest(law=law):
                # Use a long hybrid-return delay measured in the captured peer
                # captures. Different delays across the two G.711 laws make
                # this a timing-variability qualification, not only a clean
                # zero-delay paired path.
                delay_ms = 1240 if law == "pcmu" else 1610
                engine, caller_tx, caller = self._run_engine_phase2(
                    law, local_echo_delay_samples=delay_ms * 8,
                    local_echo_gain=2.0,
                    late_l2_echo_policy="advance-after-window")
                (waveform, caller_payload, answerer_payload, payload_bit_count,
                 up_bits_per_slice, down_bits_per_slice) = (
                    _zero_dil_answerer_payload_waveform(law))
                # The answerer's LAPM peer has data waiting when CP selects the
                # downstream profile. Queue it before that negotiation, as a
                # DTE may do while the modem is completing startup.
                engine.feed_bytes(answerer_payload)
                received = bytearray()
                downstream = bytearray()
                try:
                    self.assertEqual(caller.state, caller.COMPLETE)
                    samples = waveform * 32768.0
                    for offset in range(0, len(samples), 160):
                        chunk = samples[offset:offset + 160]
                        if len(chunk) != 160:
                            chunk = np.pad(chunk, (0, 160 - len(chunk)))
                        media_at = engine._rx_at
                        downstream.extend(engine.tx_g711(160, law))
                        payload = bytes(v34_g711.encode(chunk, law))
                        received.extend(engine.feed_g711(payload, law, {
                            "packet_playout_deadline_monotonic":
                                engine._clock.origin + media_at / 8000,
                            "payload_samples": len(payload),
                            "timestamp": media_at,
                        }))
                        if engine._failed:
                            self.fail(
                                f"production V.90 engine failed: {engine.events[-12:]}")

                    phase3 = engine._phase3
                    self.assertTrue(phase3.transmit_ready)
                    self.assertEqual(phase3.transmitter.state,
                                     phase3.transmitter.PHASE4_READY)
                    self.assertTrue(phase3._phase4_cpt_acquired,
                                    phase3.events[-12:])
                    self.assertTrue(any(
                        name == "phase4-profile-accepted"
                        and profile.get("kind") == "cp"
                        and profile.get("ack") == 0
                        for name, profile in phase3.events))
                    self.assertTrue(any(
                        name == "phase4-e-detected" for name, _ in phase3.events),
                        ([name for name, _ in phase3.events[-30:]],
                         phase3.phase4.state,
                         phase3.phase4.transmitter.mp_prime_frames_sent,
                         phase3.phase4.transmitter._end_signal,
                         phase3.phase4_error))
                    self.assertEqual(phase3.analog_upstream_rate, 33600)
                    self.assertIsNotNone(phase3.upstream_data)
                    b1 = phase3.upstream_data.receiver.diagnostics
                    self.assertTrue(b1["qualified"], b1)
                    self.assertEqual(b1["bit_errors"], 0, b1)
                    self.assertEqual(bytes(received), caller_payload)
                    self.assertEqual(engine.v42_mode, "lapm")
                    self.assertTrue(engine.v42_resolved)
                    self.assertTrue(engine.connected)

                    # Recover the answerer's actual production Phase 4 G.711
                    # output, including its B1d and LAPM frames, and deliver
                    # the decoded bits to the paired caller LAPM state.
                    phase4 = phase3.phase4
                    cp_profile = phase4.transmitter.cp_profile
                    self.assertIsNotNone(cp_profile)
                    downstream_encoder = V90DigitalPhase4Encoder(
                        cp_profile, law)
                    b1d_bits = [1] * (48 * (downstream_encoder.s
                                              + downstream_encoder.k))
                    expected_b1d = downstream_encoder.encode_bits(b1d_bits)
                    b1d_at = downstream.find(expected_b1d)
                    self.assertGreaterEqual(b1d_at, 0,
                                            "production downstream B1d absent")
                    downstream_rx = V90DigitalPhase4Receiver(cp_profile, law)
                    decoded_b1d = downstream_rx.decode_codewords(expected_b1d)
                    self.assertEqual(decoded_b1d, b1d_bits)
                    down_tail = downstream[b1d_at + len(expected_b1d):]
                    down_tail = down_tail[:len(down_tail) // 6 * 6]
                    down_bits = downstream_rx.decode_codewords(down_tail)

                    caller_lapm = V90LapmLink(
                        rate=phase3.analog_upstream_rate, calling_party=True)
                    caller_lapm.feed_bytes(caller_payload)
                    caller_lapm.feed_line_bits(b1d_bits)
                    received_answerer = bytearray()
                    down_offset = 0
                    for _offset in range(0, payload_bit_count,
                                         up_bits_per_slice):
                        if bytes(received_answerer) == answerer_payload:
                            break
                        # Advance caller LAPM's transmit sequence. The paired
                        # PCM stimulus is a prior valid trace, so it need not
                        # equal this replay caller's response to the current
                        # answerer acknowledgements.
                        caller_lapm.read_tx_bits(up_bits_per_slice)
                        down_chunk = down_bits[
                            down_offset:down_offset + down_bits_per_slice]
                        if len(down_chunk) != down_bits_per_slice:
                            break
                        caller_lapm.feed_line_bits(down_chunk)
                        down_offset += len(down_chunk)
                        received_answerer.extend(caller_lapm.take_received())
                    self.assertEqual(bytes(received_answerer), answerer_payload)
                    self.assertEqual(caller_lapm.v42_mode, "lapm")
                finally:
                    engine.close()

    def test_production_answerer_accepts_late_s_and_delivers_lapm_payload(self):
        """Late S carries LAPM across RTDs, delay splits, and jitter changes."""
        extra_s_delay_ms = 3700
        # Vary both total RTD and its one-way distribution. This distinguishes
        # use of measured round trip from a fixed timeout or a symmetric-path
        # assumption.
        media_delay_cases = (
            ("symmetric-200ms-rtd", 800, 800),
            ("symmetric", 1600, 1600),
            ("caller-to-answerer-heavy", 2400, 800),
            ("answerer-to-caller-heavy", 800, 2400),
            ("symmetric-600ms-rtd", 2400, 2400),
        )
        for law in ("pcmu", "pcma"):
            for jitter_mode in ("isolated-15ms", "persistent-40ms"):
                for (delay_name, caller_to_answerer,
                     answerer_to_caller) in media_delay_cases:
                    with self.subTest(law=law, delay=delay_name,
                                      jitter=jitter_mode):
                        engine, _, caller = self._run_engine_phase2(
                            law, local_echo_delay_samples=320,
                            local_echo_gain=0.25,
                            caller_to_answerer_delay_samples=caller_to_answerer,
                            answerer_to_caller_delay_samples=answerer_to_caller,
                            late_l2_echo_policy="advance-after-window")
                        phase3_start_sample = engine._rx_at
                        measured_rtd_samples = int(engine._phase2.rtd_samples or 0)
                        self.assertLessEqual(
                            abs(measured_rtd_samples
                                - caller_to_answerer - answerer_to_caller), 160)
                        (waveform, caller_payload, answerer_payload,
                         payload_bit_count, up_bits_per_slice,
                         down_bits_per_slice) = (
                            _zero_dil_answerer_payload_waveform(
                                law, extra_s_delay_ms=extra_s_delay_ms))
                        # The full two-way delay measured during Phase 2 elapses
                        # before the caller's Phase-3 response can reach the
                        # answerer. Apply it ahead of the waveform to keep the S
                        # deadline, downstream sampling origin, and B1 windows on
                        # one media clock.
                        waveform = np.concatenate((
                            np.zeros(measured_rtd_samples, dtype=float), waveform))
                        engine.feed_bytes(answerer_payload)
                        received = bytearray()
                        downstream = bytearray()
                        jitter_outlier_injected = False
                        jitter_step_started = False
                        try:
                            self.assertEqual(caller.state, caller.COMPLETE)
                            samples = waveform * 32768.0
                            for offset in range(0, len(samples), 160):
                                chunk = samples[offset:offset + 160]
                                if len(chunk) != 160:
                                    chunk = np.pad(chunk, (0, 160 - len(chunk)))
                                media_at = engine._rx_at
                                downstream.extend(engine.tx_g711(160, law))
                                payload = bytes(v34_g711.encode(chunk, law))
                                deadline = engine._clock.origin + media_at / 8000
                                phase3 = engine._phase3
                                if (not jitter_outlier_injected
                                        and not jitter_step_started
                                        and phase3 is not None
                                        and phase3.transmitter._trn1d_start_sample
                                        is not None):
                                    elapsed = (media_at - phase3_start_sample
                                               - phase3.transmitter._trn1d_start_sample)
                                    probe_at = (
                                        5100 * 8
                                        + phase3.transmitter.round_trip_delay_samples
                                        - 200 * 8)
                                    if elapsed <= probe_at < elapsed + len(chunk):
                                        if jitter_mode == "isolated-15ms":
                                            # One packet's playout deadline arrives
                                            # 15 ms late near S. Preserve PCM and
                                            # RTP sample position while probing the
                                            # isolated-jitter admission path.
                                            deadline += 0.015
                                            jitter_outlier_injected = True
                                        else:
                                            # Jitter-buffer offset changes by 40 ms
                                            # and remains at the new offset.
                                            jitter_step_started = True
                                if (jitter_mode == "persistent-40ms"
                                        and jitter_step_started):
                                    deadline += 0.040
                                received.extend(engine.feed_g711(payload, law, {
                                    "packet_playout_deadline_monotonic": deadline,
                                    "payload_samples": len(payload),
                                    "timestamp": media_at,
                                }))
                                if engine._failed:
                                    self.fail(
                                        f"late-S V.90 engine failed: {engine.events[-12:]}")

                            phase3 = engine._phase3
                            if jitter_mode == "isolated-15ms":
                                self.assertTrue(
                                    jitter_outlier_injected,
                                    "late-S packet jitter probe was not applied")
                            else:
                                self.assertTrue(
                                    jitter_step_started,
                                    "late-S playout offset step was not applied")
                            mapped_offset = (
                                engine._clock.rx_to_media(engine._rx_at)
                                - engine._clock.media_at - engine._rx_at)
                            expected_offset_step = (
                                320 if jitter_mode == "persistent-40ms" else 0)
                            self.assertAlmostEqual(
                                mapped_offset, expected_offset_step, delta=80,
                                msg=f"{jitter_mode} did not map the RTP offset")
                            self.assertTrue(any(name == "s-detected"
                                                for name, _ in phase3.events),
                                            phase3.events[-12:])
                            self.assertFalse(any(name == "phase3-s-timeout"
                                                 for name, _ in phase3.events),
                                             phase3.events[-12:])
                            s_detected_at = next(
                                at for name, at in phase3.events
                                if name == "s-detected")
                            # Keep a regression check that the digital
                            # answerer's V.90 media remains active while it
                            # waits for a delayed S transition.
                            wait_end = s_detected_at - phase3_start_sample
                            wait_audio = np.frombuffer(
                                downstream[wait_end - 800:wait_end],
                                dtype=np.uint8)
                            self.assertEqual(len(wait_audio), 800)
                            wait_pcm = v34_g711.decode(
                                wait_audio, law).astype(float)
                            wait_rms = float(np.sqrt(np.mean(wait_pcm ** 2)))
                            self.assertGreater(
                                wait_rms, 1000.0,
                                "digital answerer went silent while awaiting S")
                            s_wait_samples = (
                                s_detected_at - phase3_start_sample
                                - phase3.transmitter._trn1d_start_sample)
                            deadline_samples = (
                                5100 * 8
                                + phase3.transmitter.round_trip_delay_samples)
                            self.assertGreater(
                                s_wait_samples, deadline_samples - 100 * 8)
                            self.assertLess(s_wait_samples, deadline_samples)
                            self.assertEqual(phase3.transmitter.state,
                                             phase3.transmitter.PHASE4_READY)
                            b1 = phase3.upstream_data.receiver.diagnostics
                            self.assertTrue(b1["qualified"], b1)
                            self.assertEqual(b1["bit_errors"], 0, b1)
                            self.assertEqual(bytes(received), caller_payload)
                            self.assertEqual(engine.v42_mode, "lapm")
                            self.assertTrue(engine.v42_resolved)

                            phase4 = phase3.phase4
                            cp_profile = phase4.transmitter.cp_profile
                            self.assertIsNotNone(cp_profile)
                            downstream_encoder = V90DigitalPhase4Encoder(
                                cp_profile, law)
                            b1d_bits = [1] * (48 * (downstream_encoder.s
                                                      + downstream_encoder.k))
                            expected_b1d = downstream_encoder.encode_bits(b1d_bits)
                            b1d_at = downstream.find(expected_b1d)
                            self.assertGreaterEqual(b1d_at, 0,
                                                    "late-S downstream B1d absent")
                            downstream_rx = V90DigitalPhase4Receiver(cp_profile, law)
                            self.assertEqual(
                                downstream_rx.decode_codewords(expected_b1d), b1d_bits)
                            down_tail = downstream[
                                b1d_at + len(expected_b1d):]
                            down_tail = down_tail[:len(down_tail) // 6 * 6]
                            down_bits = downstream_rx.decode_codewords(down_tail)
                            caller_lapm = V90LapmLink(
                                rate=phase3.analog_upstream_rate, calling_party=True)
                            caller_lapm.feed_bytes(caller_payload)
                            caller_lapm.feed_line_bits(b1d_bits)
                            received_answerer = bytearray()
                            down_offset = 0
                            for _ in range(0, payload_bit_count, up_bits_per_slice):
                                if bytes(received_answerer) == answerer_payload:
                                    break
                                caller_lapm.read_tx_bits(up_bits_per_slice)
                                down_chunk = down_bits[
                                    down_offset:down_offset + down_bits_per_slice]
                                if len(down_chunk) != down_bits_per_slice:
                                    break
                                caller_lapm.feed_line_bits(down_chunk)
                                down_offset += len(down_chunk)
                                received_answerer.extend(caller_lapm.take_received())
                            self.assertEqual(bytes(received_answerer), answerer_payload)
                            self.assertEqual(caller_lapm.v42_mode, "lapm")
                        finally:
                            engine.close()

    @staticmethod
    def _feed_phase3_ja(engine, law):
        ja_bits = ja_control._ja_descriptor(7, 19, 31)
        descriptor = v90.parse_ja_descriptor(ja_bits)
        caller_phase3 = V90AnalogueCallerPhase3(
            baud=ja_control.BAUD, ja_bits=ja_bits, descriptor=descriptor)
        caller_tx = V90AnaloguePhase3Transmitter(
            caller_phase3, carrier=ja_control.CARRIER, rms=2000.0)
        media_start = engine._rx_at
        actions = caller_tx.start(media_start)
        # Send the complete initial burst and three Ja copies, matching the
        # bounded independent reference transcript while obtaining samples
        # from the production caller Phase 3 renderer.
        ja_symbols = len(ja_bits) // 2
        signal_samples = sum(action.duration_samples for action in actions[1:5])
        signal_samples += int(round(3 * ja_symbols * 8000 / ja_control.BAUD))
        total_samples = actions[0].duration_samples + signal_samples
        total_samples = ((total_samples + 159) // 160) * 160
        final_packet_start = media_start + total_samples - 160
        while engine._rx_at < final_packet_start:
            chunk = caller_tx.read_pcm(160)
            media_at = engine._rx_at
            engine.tx_g711(160, law)
            payload = bytes(v34_g711.encode(chunk, law))
            engine.feed_g711(payload, law, {
                "packet_playout_deadline_monotonic":
                    engine._clock.origin + media_at / 8000,
                "payload_samples": len(payload),
                "timestamp": media_at,
            })
            if engine._failed:
                raise AssertionError(f"production V.90 phase 3 failed: {engine.events[-8:]}")
        stop_at = engine._rx_at + 80
        caller_tx.notify_sd_transition(stop_at)
        chunk = caller_tx.read_pcm(160)
        np.testing.assert_array_equal(chunk[80:], np.zeros(80))
        media_at = engine._rx_at
        engine.tx_g711(160, law)
        engine.feed_g711(bytes(v34_g711.encode(chunk, law)), law, {
            "packet_playout_deadline_monotonic":
                engine._clock.origin + media_at / 8000,
            "payload_samples": 160,
            "timestamp": media_at,
        })
        if engine._failed:
            raise AssertionError(f"production V.90 phase 3 failed: {engine.events[-8:]}")
        return caller_tx, caller_phase3

    def _run_engine_phase2(self, law, *, local_echo_delay_samples=320,
                           local_echo_gain=0.25,
                           caller_to_answerer_delay_samples=0,
                           answerer_to_caller_delay_samples=0,
                           late_l2_echo_policy="fail",
                           late_l2_echo_max_samples=0,
                           accept_early_peer_recovery=False,
                           caller_rx_gaps=(), answerer_rx_gaps=(),
                           caller_rtp_gaps=()):
        caller = V90AnalogueCallerPhase2(
            late_l2_echo_policy=late_l2_echo_policy,
            late_l2_echo_max_samples=late_l2_echo_max_samples,
            accept_early_peer_recovery=accept_early_peer_recovery)
        caller_rx = V90AnaloguePhase2Adapter(
            caller, rx_to_media=lambda sample: sample)
        caller._test_adapter = caller_rx
        caller_rtp = None
        if caller_rtp_gaps:
            class _Phase2RtpReceiver:
                def __init__(self):
                    self.rx_at = 0
                    self.tx_now = 0
                    self.gaps = []

                def _feed(self, pcm):
                    for action in caller_rx.feed(
                            pcm, rx_at=self.rx_at, tx_now=self.tx_now):
                        caller_tx.submit([action])
                    self.rx_at += len(pcm)

                def feed_g711(self, payload, codec, context):
                    if codec != law:
                        raise AssertionError((codec, law))
                    timestamp = int(context["timestamp"])
                    if timestamp != self.rx_at:
                        raise AssertionError((timestamp, self.rx_at))
                    pcm = v34_g711.decode(
                        np.frombuffer(payload, dtype=np.uint8), law).astype(float)
                    self._feed(pcm)
                    return b""

                def feed_gap(self, samples, codec, context):
                    if codec != law:
                        raise AssertionError((codec, law))
                    self.gaps.append(int(samples))
                    self._feed(np.zeros(int(samples), dtype=float))
                    return b""

            caller_rtp_receiver = _Phase2RtpReceiver()
            caller._test_rtp_gap_samples = caller_rtp_receiver.gaps
            caller_rtp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            caller_rtp = OnlineAudioSession(
                caller_rtp_socket, ("127.0.0.1", 9), "originate", "answer",
                line_mode="bell103", g711_codec=law)
            caller_rtp._rtp_jitter_samples = 0
            caller_rtp.set_raw_g711_receiver(caller_rtp_receiver)
            # The test drives ingress synchronously. Stop the normal media
            # sender so the fixture has no background RTP traffic.
            caller_rtp._stop.set()
        caller_tx = V90AnaloguePhase2Transmitter(
            info0a_fields={"rate3429": 1, "disallow3429": 1},
            info1a_builder=lambda mode: v90.build_info1a_v90(
                md_len=0, uinfo=90, analog_symbol_rate=5, digital_symbol_rate=6,
                freq_offset=0),
            probe_builder=v34_call.probing_signal,
            nominal_peak=1200, probe_rms=300)
        for name, samples in (
                ("caller_to_answerer_delay_samples",
                 caller_to_answerer_delay_samples),
                ("answerer_to_caller_delay_samples",
                 answerer_to_caller_delay_samples)):
            if (isinstance(samples, bool) or not isinstance(samples, int)
                    or samples < 0):
                raise ValueError(f"{name} must be a nonnegative integer")
        media_origin = time.monotonic()
        engine = V90DigitalAnswererPhase2Engine(
            role="digital-answerer",
            negotiation={"pcm_negotiated_bits": V8_PCM_DIGITAL,
                         "protocols": V8_PROTOCOL_LAPM_V42},
            media_at=0, media_origin=media_origin, law=law,
            info0d_fields={"rate3429": 1, "v90_3429_upstream": 1},
            info1d_fields={
                "blocks": ([{"hi_carrier": 0} for _ in range(5)]
                           + [{"hi_carrier": 0, "max_rate": 12}]),
                "freq_offset": 0,
            },
            tone_b_peak=800, probe_rms=300,
            probe_builder=v34_call.probing_signal,
            phase3_rate_mask=0xFFFF, phase3_enable_lapm=True)

        # These paired state-machine tests exercise INFO negotiation and the
        # Phase 3 handoff. The synthetic echo/channel fixture is not a
        # calibrated V.90 line, so its measured probe SNR is not a meaningful
        # rate-selection input. Keep the production estimator covered by the
        # dedicated Phase 2 signal tests, and give this protocol fixture an
        # explicit clean-channel measurement when it is about to emit INFO1d.
        # This keeps the integration test deterministic without weakening the
        # production rate gate.
        apply_measured = engine._apply_measured_info1d_offset

        def apply_fixture_probe_quality(actions):
            if any(action.signal == "info1d" for action in actions):
                engine._adapter.signals.probe_quality = {"snr_db": 50.0}
            return apply_measured(actions)

        engine._apply_measured_info1d_offset = apply_fixture_probe_quality

        caller_tx.submit(caller_rx.start(0))
        caller_info0a_done = False
        media_duration = (
            8 * 8000 + caller_to_answerer_delay_samples
            + answerer_to_caller_delay_samples)
        history_samples = max(12 * 8000, media_duration) + max(
            int(local_echo_delay_samples), 320) + 160
        caller_tx_history = np.zeros(history_samples)
        answerer_tx_history = np.zeros_like(caller_tx_history)
        caller_rx_history = np.zeros_like(caller_tx_history)
        caller_rtp_dropped_starts = {
            sample for gap_at, gap_length in caller_rtp_gaps
            for sample in range(gap_at, gap_at + gap_length, 160)
        }
        for start in range(0, media_duration, 160):
            caller_pcm = caller_tx.render(160)
            answerer_wire = engine.tx_g711(160, law)
            answerer_pcm = v34_g711.decode(
                np.frombuffer(answerer_wire, dtype=np.uint8), law).astype(float)
            if start + 160 > len(caller_tx_history):
                raise AssertionError("paired Phase 2 exceeded its media-clock fixture")
            caller_tx_history[start:start + 160] = caller_pcm
            answerer_tx_history[start:start + 160] = answerer_pcm
            caller_rx_pcm = _delayed_echo(
                answerer_tx_history, start, 160,
                delay_samples=answerer_to_caller_delay_samples, gain=1.0)
            caller_rx_pcm += _delayed_echo(
                caller_tx_history, start, 160,
                delay_samples=local_echo_delay_samples,
                gain=local_echo_gain)
            answerer_rx_pcm = _delayed_echo(
                caller_tx_history, start, 160,
                delay_samples=caller_to_answerer_delay_samples, gain=1.0)
            answerer_rx_pcm += _delayed_echo(
                answerer_tx_history, start, 160)
            caller_gap = next((gap_length for gap_at, gap_length in caller_rx_gaps
                               if start == gap_at), 0)
            answerer_gap = next((gap_length for gap_at, gap_length in answerer_rx_gaps
                                 if start == gap_at), 0)
            if caller_gap:
                caller_rx_pcm = np.zeros_like(caller_rx_pcm)
            if answerer_gap:
                answerer_rx_pcm = np.zeros_like(answerer_rx_pcm)
            caller_rx_history[start:start + 160] = caller_rx_pcm
            context = {
                "packet_playout_deadline_monotonic": media_origin + start / 8000,
                "payload_samples": 160,
                "timestamp": start,
            }
            # The paired signal fixture models a lost media interval as
            # silence while preserving the common sample clock. The separate
            # production feed_gap() tests cover the DCE's explicit RTP-gap API.
            engine.feed_g711(bytes(v34_g711.encode(answerer_rx_pcm, law)),
                             law, context)
            now = start + 160

            info0a_end = caller_tx._completion["info0a-complete"]
            if not caller_info0a_done and info0a_end <= now:
                caller_info0a_done = True
                caller_tx.submit(caller_rx.feed_tx_complete(
                    "info0a_complete", media_at=info0a_end, tx_now=now))

            if caller_rtp is not None:
                receiver = caller_rtp._raw_g711_receiver
                receiver.tx_now = now
                if start not in caller_rtp_dropped_starts:
                    packet = struct.pack(
                        "!BBHII", 0x80, caller_rtp._rtp_payload_type,
                        (100 + start // 160) & 0xFFFF,
                        start & 0xFFFFFFFF, 0x12345678)
                    packet += bytes(v34_g711.encode(caller_rx_pcm, law))
                    caller_rtp.handle_rtp_packet(
                        packet, received_at=media_origin + start / 8000)
            else:
                for action in caller_rx.feed(
                        caller_rx_pcm, rx_at=start, tx_now=now):
                    caller_tx.submit([action])

            for action in caller_rx.tick(media_now=now):
                caller_tx.submit([action])

            if caller.done and engine._phase2.done:
                break

        if caller_rtp is not None:
            caller_rtp.stop()
            caller_rtp._sock.close()
        return engine, caller_tx, caller

    def _run_production_paired_phase2(self, law, *, echo_delay_samples,
                                      echo_gain=2.0,
                                      answerer_echo_gain=0.25,
                                      dropped_samples=(), arrival_clock_ppm=0.0,
                                      arrival_jitter_ms=1.5,
                                      phase2_gap_max_samples=320,
                                      phase2_late_l2_echo_policy="bounded",
                                      phase2_late_l2_echo_max_samples=20000,
                                      caller_info0a_fields=None,
                                      peer_info0d_fields=None,
                                      peer_info1d_blocks=None,
                                      continue_v34_fallback=False,
                                      v34_fallback_hold_samples=0,
                                      capture_fallback_pcm=False,
                                      cancel_synthetic_echo_after_fallback=False,
                                      accept_early_peer_recovery=False,
                                      fallback_payloads=None,
                                      fallback_payloads_after_rtp_gap=False,
                                      fallback_payloads_after_lapm_samples=None,
                                      fallback_v42_role_overrides=None,
                                      fallback_force_v42_detection_after_v8=False,
                                      fallback_v42_start_delays_samples=None,
                                      fallback_v42_frame_size=128,
                                      fallback_dce_connect_gate=False,
                                      fallback_tick_hook=None,
                                      fallback_rx_audio_hook=None,
                                      fallback_local_to_remote_delay_samples=0,
                                      fallback_rtp_drop_after_connect_samples=None,
                                      fallback_lapm_trace=False,
                                      fallback_wait_for_lapm_ack=False,
                                      paired_duration_samples=20 * 8000,
                                      peer_v34_mp_request=None,
                                      info1d_timeout_retrain=False,
                                      v90_fallback_retrain=False,
                                      request_preconnect_v90_retrain=False,
                                      request_postconnect_v90_retrain=False,
                                      postconnect_v90_retrain_delay_samples=0,
                                      fallback_retrain_hybrid_echo_gain=0.25,
                                      fallback_retrain_caller_echo_gain=None,
                                      fallback_retrain_answerer_echo_gain=None,
                                      fallback_payloads_after_retrain=None,
                                      drop_answerer_info1d=False,
                                      drop_answerer_phase3_until_caller_retrain=False,
                                      caller_echo_gain_after_phase3=None,
                                      answerer_echo_gain_after_caller_phase3=None,
                                      answerer_echo_delay_samples=320,
                                      answerer_echo_path_after_caller_phase3=None,
                                      answerer_echo_use_configured_delay_all_phases=False,
                                      answerer_echo_gain_during_phase3_phase2_probe=None,
                                      answerer_info1a_timeout_retrain=False,
                                      caller_phase2_retrain_hybrid_echo=False):
        force_detect = fallback_force_v42_detection_after_v8
        if isinstance(force_detect, bool):
            force_detect_by_endpoint = (force_detect, force_detect)
        elif (isinstance(force_detect, tuple) and len(force_detect) == 2
              and all(isinstance(value, bool) for value in force_detect)):
            force_detect_by_endpoint = force_detect
        else:
            raise ValueError(
                "fallback_force_v42_detection_after_v8 must be bool or a "
                "pair of bools ordered caller, answerer")
        profile = {
            "info0a_fields": (caller_info0a_fields or {
                "rate3429": 1, "disallow3429": 1,
            }),
            "info1a_fields": {
                "md_len": 0, "uinfo": 90, "analog_symbol_rate": 5,
                "digital_symbol_rate": 6, "freq_offset": 0,
            },
            "nominal_peak": 1200, "probe_rms": 300,
            "l2_base_samples": 4400, "prefer_v90": True, "v34_mode": 5,
            "phase2_late_l2_echo_policy": phase2_late_l2_echo_policy,
            "phase2_late_l2_echo_max_samples":
                phase2_late_l2_echo_max_samples,
            "phase2_accept_early_peer_recovery": accept_early_peer_recovery,
            "phase2_info1d_timeout_retrain": info1d_timeout_retrain,
            "v90_fallback_retrain": v90_fallback_retrain,
            "postconnect_v90_retrain": request_postconnect_v90_retrain,
            "phase2_rtp_gap_policy": "fill_silence",
            "phase2_rtp_gap_max_samples": phase2_gap_max_samples,
            "phase2_clock_low_delay_percentile": 35,
            "phase2_clock_max_rate_error_ppm": 20000,
            "phase2_clock_anchor_limit": 128,
            "phase2_clock_min_fit_span_samples": 12800,
        }
        media_origin = time.monotonic()
        caller = V90AnalogueCallerPhase2Engine(
            role="analogue-caller",
            negotiation={"pcm_negotiated_bits": V8_PCM_ANALOG},
            media_at=0, media_origin=media_origin, law=law, profile=profile)
        answerer = V90DigitalAnswererPhase2Engine(
            role="digital-answerer",
            # Native V.90 LAPM starts only when V.8 negotiated it (9797c7c4);
            # the paired caller runs LAPM, so the fixture must agree.
            negotiation={"pcm_negotiated_bits": V8_PCM_DIGITAL,
                         "protocols": V8_PROTOCOL_LAPM_V42},
            media_at=0, media_origin=media_origin, law=law,
            info0d_fields=(peer_info0d_fields or {
                "rate3429": 1, "v90_3429_upstream": 1,
            }),
            info1d_fields={"blocks": (peer_info1d_blocks or
                                       ([{"hi_carrier": 0} for _ in range(5)]
                                        + [{"hi_carrier": 0, "max_rate": 12}])),
                           "freq_offset": 0},
            tone_b_peak=800, probe_rms=300,
            probe_builder=v34_call.probing_signal,
            phase3_rate_mask=0xFFFF, phase3_enable_lapm=True,
            phase2_info1a_timeout_retrain=answerer_info1a_timeout_retrain)
        apply_measured = answerer._apply_measured_info1d_offset

        def apply_fixture_probe_quality(actions):
            if any(action.signal == "info1d" for action in actions):
                answerer._adapter.signals.probe_quality = {"snr_db": 50.0}
            return apply_measured(actions)

        answerer._apply_measured_info1d_offset = apply_fixture_probe_quality
        dropped_info1d_ranges = []
        dropped_phase3_ranges = []
        original_submit = answerer._submit_phase2_actions

        def track_info1d(actions):
            original_submit(actions)
            for action in actions:
                if action.signal == "info1d":
                    complete = answerer._renderer._completion["info1d-complete"]
                    dropped_info1d_ranges.append((int(action.at), int(complete)))

        answerer._submit_phase2_actions = track_info1d
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        rtp = OnlineAudioSession(
            sock, ("127.0.0.1", 9), "originate", "answer",
            line_mode="bell103", g711_codec=law)
        rtp._rtp_jitter_samples = 0
        rtp.set_raw_g711_receiver(caller)
        # This helper exercises only synchronous ingress. Keep the regular
        # RTP transmit loop from sending unrelated media to the test socket.
        rtp._stop.set()

        history_samples = int(paired_duration_samples) + max(
            20000, int(echo_delay_samples)) + 160
        caller_tx_history = np.zeros(history_samples)
        answerer_tx_history = np.zeros(history_samples)
        caller_rx_capture = [] if capture_fallback_pcm else None
        answerer_rx_capture = [] if capture_fallback_pcm else None
        caller_tx_capture = [] if capture_fallback_pcm else None
        answerer_tx_capture = [] if capture_fallback_pcm else None
        if fallback_payloads is not None and (
                not isinstance(fallback_payloads, tuple)
                or len(fallback_payloads) != 2):
            raise ValueError("fallback_payloads must be a (caller, answerer) tuple")
        if (not isinstance(postconnect_v90_retrain_delay_samples, int)
                or postconnect_v90_retrain_delay_samples < 0):
            raise ValueError(
                "postconnect_v90_retrain_delay_samples must be a non-negative int")
        caller_payload_rx = bytearray()
        answerer_payload_rx = bytearray()
        fallback_payloads_started = False
        fallback_payloads_after_retrain_started = False
        fallback_v42_started = False
        fallback_v42_started_sample = None
        fallback_v42_start_origin_sample = None
        fallback_v42_endpoint_started = [False, False]
        fallback_dce_connect_emitted = False
        fallback_dce_connect_results = {}
        lapm_connected_sample = None
        fallback_gap_sample = None
        fallback_media_path_entered = False
        lapm_media_trace = []
        last_lapm_state = {}
        peer_v34_mp_request_applied = False
        preconnect_v90_retrain_requested = False
        postconnect_v90_retrain_requested = False
        postconnect_v90_retrain_eligible_sample = None
        postconnect_v90_retrain_request_sample = None
        retained_lapm_links = None
        preconnect_retrain_stage_trace = []
        last_preconnect_retrain_stage = None
        dropped = {sample for start, length in dropped_samples
                   for sample in range(start, start + length, 160)}

        def record_lapm_media_state(sample, injected_gap=False):
            if not fallback_lapm_trace:
                return
            for label, engine in (("caller", caller), ("answerer", answerer)):
                fallback = engine._v34_fallback_engine
                link = None if fallback is None else getattr(fallback, "_v42", None)
                if link is None:
                    continue
                keys = ("fcs_errors", "hdlc_good", "hdlc_bad", "i_tx",
                        "i_rx", "i_resent", "wire_acked", "bytes_in",
                        "bytes_out", "t401_expiries")
                counters = {key: int(link.stats.get(key, 0)) for key in keys}
                tx_state = {
                    "va": int(link.va),
                    "vs": int(link.vs),
                    "unacked_frames": len(link._unacked),
                    "unacked_octets": sum(len(payload)
                                           for _, payload in link._unacked),
                    "ack_pending": int(link._ack_pending),
                    "poll_out": int(link._poll_out),
                    "retx_frames": len(link._retx),
                }
                previous = last_lapm_state.get(label)
                endpoint_gap = injected_gap and label == "caller"
                if endpoint_gap or previous != (link.state, counters, tx_state):
                    lapm_media_trace.append({
                        "endpoint": label,
                        "media_sample": int(sample),
                        "relative_to_lapm_start": (
                            None if fallback_v42_started_sample is None else
                            int(sample - fallback_v42_started_sample)),
                        "injected_gap": bool(endpoint_gap),
                        "state": link.state,
                        **counters,
                        **tx_state,
                    })
                last_lapm_state[label] = (link.state, counters, tx_state)

        for start in range(0, int(paired_duration_samples), 160):
            if (caller._v34_fallback_engine is not None
                    and answerer._v34_fallback_engine is not None):
                # A V.90 retrain may temporarily retire one fallback engine.
                # The paired line's synthetic echo cancellation models the
                # already-connected media path, so do not reintroduce that
                # echo merely because the protocol temporarily left Phase 3.
                fallback_media_path_entered = True
            if fallback_tick_hook is not None:
                fallback_tick_hook(caller, answerer, start)
            if request_preconnect_v90_retrain:
                stage = (
                    None if caller._v34_fallback_engine is None else
                    caller._v34_fallback_engine.answerer.stage,
                    None if answerer._v34_fallback_engine is None else
                    answerer._v34_fallback_engine.caller.stage,
                    None if caller._v34_fallback_engine is None else
                    caller._v34_fallback_engine.connected,
                    None if answerer._v34_fallback_engine is None else
                    answerer._v34_fallback_engine.connected,
                )
                if stage != last_preconnect_retrain_stage:
                    preconnect_retrain_stage_trace.append((start, stage))
                    last_preconnect_retrain_stage = stage
            if (request_preconnect_v90_retrain
                    and not preconnect_v90_retrain_requested
                    and caller._v34_fallback_engine is not None
                    and answerer._v34_fallback_engine is not None
                    and caller._v34_fallback_engine.answerer.stage == "p4"
                    and not caller._v34_fallback_engine.connected):
                # Wait until the analogue caller has validated the complete
                # Phase 3 DIL and entered Phase 4. Requesting earlier truncates
                # DIL and tests descriptor corruption, not Tone B recovery.
                # Phase 4 is still pre-B1/CONNECT, which is the case under test.
                if not answerer.request_retrain(
                        reason="paired pre-CONNECT fallback recovery"):
                    raise AssertionError("digital peer refused pre-CONNECT retrain")
                fallback_media_path_entered = True
                preconnect_v90_retrain_requested = True
            postconnect_retrain_ready = (
                    request_postconnect_v90_retrain
                    and not postconnect_v90_retrain_requested
                    and lapm_connected_sample is not None
                    and caller._v34_fallback_engine is not None
                    and answerer._v34_fallback_engine is not None
                    and fallback_payloads is not None
                    and bytes(caller_payload_rx) == fallback_payloads[1]
                    and bytes(answerer_payload_rx) == fallback_payloads[0])
            if postconnect_retrain_ready:
                left_link = caller._v34_fallback_engine._v42
                right_link = answerer._v34_fallback_engine._v42
                if (left_link is not None and right_link is not None
                        and not left_link._unacked and not right_link._unacked
                        and left_link.stats["wire_acked"] >= len(
                            b"" if fallback_payloads is None else fallback_payloads[0])
                        and right_link.stats["wire_acked"] >= len(
                            b"" if fallback_payloads is None else fallback_payloads[1])):
                    if postconnect_v90_retrain_eligible_sample is None:
                        postconnect_v90_retrain_eligible_sample = int(start)
                    if (start >= postconnect_v90_retrain_eligible_sample
                            + postconnect_v90_retrain_delay_samples):
                        retained_lapm_links = (left_link, right_link)
                        if not answerer.request_retrain(
                                reason="paired post-CONNECT V.90 retrain"):
                            raise AssertionError(
                                "digital peer refused connected V.90 retrain")
                        postconnect_v90_retrain_requested = True
                        postconnect_v90_retrain_request_sample = int(start)
            if (lapm_connected_sample is not None
                    and fallback_rtp_drop_after_connect_samples is not None
                    and fallback_gap_sample is None):
                target = (lapm_connected_sample
                          + int(fallback_rtp_drop_after_connect_samples))
                if start >= target:
                    fallback_gap_sample = start
                    dropped.add(start)
            if (fallback_payloads is not None
                    and fallback_payloads_after_rtp_gap
                    and fallback_gap_sample is not None
                    and start > fallback_gap_sample
                    and not fallback_payloads_started):
                # Keep LAPM idle across the single missing RTP interval, then
                # send fresh data to test whether the link remains usable.
                caller.feed_bytes(fallback_payloads[0])
                answerer.feed_bytes(fallback_payloads[1])
                fallback_payloads_started = True
            if (fallback_payloads is not None
                    and not fallback_payloads_after_rtp_gap
                    and fallback_payloads_after_lapm_samples is not None
                    and lapm_connected_sample is not None
                    and (not fallback_dce_connect_gate
                         or fallback_dce_connect_emitted)
                    and start >= (lapm_connected_sample
                                  + int(fallback_payloads_after_lapm_samples))
                    and not fallback_payloads_started):
                caller.feed_bytes(fallback_payloads[0])
                answerer.feed_bytes(fallback_payloads[1])
                fallback_payloads_started = True
            caller_pcm = np.asarray(caller.tx_samples(160), dtype=float)
            answerer_wire = answerer.tx_g711(160, law)
            answerer_pcm = v34_g711.decode(
                np.frombuffer(answerer_wire, dtype=np.uint8), law).astype(float)
            if (drop_answerer_phase3_until_caller_retrain
                    and caller._phase3_controller is not None
                    and caller._phase3_retrain_attempts == 0):
                # Model a downstream-only Phase-3 outage. Keep the caller's
                # audio and the reverse path intact so its missing-Sd recovery
                # must be detected by the paired digital answerer.
                answerer_pcm.fill(0.0)
                dropped_phase3_ranges.append((int(start), int(start + 160)))
            if (drop_answerer_info1d
                    and any(start < end and start + 160 > begin
                            for begin, end in dropped_info1d_ranges[:1])):
                answerer_pcm.fill(0.0)
            if capture_fallback_pcm:
                caller_tx_capture.append(caller_pcm.astype("<i2"))
                answerer_tx_capture.append(answerer_pcm.astype("<i2"))
            caller_tx_history[start:start + 160] = caller_pcm
            answerer_tx_history[start:start + 160] = answerer_pcm
            v90_retrain_media = (
                answerer._retrain is not None
                if continue_v34_fallback else False)
            if caller_phase2_retrain_hybrid_echo:
                v90_retrain_media = (
                    v90_retrain_media
                    or caller._phase2_retrain_attempts > 0)
            phase3_echo_active = (
                caller._phase3_controller is not None
                or caller._phase3_retrain_attempts > 0)
            if v90_retrain_media:
                # Retrain returns to V.90 Phase 2, whose L2-echo observation
                # is required before INFO1d. Model each local hybrid with a
                # short, ordinary return; do not carry the long initial
                # delayed-echo impairment into both endpoint hybrids.
                caller_echo = _delayed_echo(
                    caller_tx_history, start, 160,
                    delay_samples=320,
                    gain=float(fallback_retrain_hybrid_echo_gain
                               if fallback_retrain_caller_echo_gain is None
                               else fallback_retrain_caller_echo_gain))
            elif (caller_echo_gain_after_phase3 is not None
                    and phase3_echo_active):
                caller_echo = _delayed_echo(
                    caller_tx_history, start, 160,
                    delay_samples=echo_delay_samples,
                    gain=float(caller_echo_gain_after_phase3))
            else:
                caller_echo = _delayed_echo(
                    caller_tx_history, start, 160,
                    delay_samples=echo_delay_samples, gain=echo_gain)
            caller_rx_pcm = answerer_pcm + (
                np.zeros(160) if cancel_synthetic_echo_after_fallback
                and fallback_media_path_entered and not v90_retrain_media
                else caller_echo)
            phase3_echo_gain = (
                answerer_echo_gain_after_caller_phase3
                if answerer_echo_gain_after_caller_phase3 is not None
                and phase3_echo_active
                else answerer_echo_gain)
            if (answerer_echo_gain_during_phase3_phase2_probe is not None
                    and phase3_echo_active
                    and answerer._phase2.state == answerer._phase2.TX_PROBE):
                phase3_echo_gain = (
                    answerer_echo_gain_during_phase3_phase2_probe)
            echo_delay = (
                answerer_echo_delay_samples
                if answerer_echo_use_configured_delay_all_phases
                or (answerer_echo_gain_after_caller_phase3 is not None
                    and phase3_echo_active
                    and answerer._phase2.state == answerer._phase2.COMPLETE)
                else 320)
            answerer_echo = _delayed_echo(
                answerer_tx_history, start, 160, delay_samples=echo_delay,
                gain=float(phase3_echo_gain))
            if (answerer_echo_path_after_caller_phase3 is not None
                    and answerer_echo_gain_after_caller_phase3 is not None
                    and phase3_echo_active
                    and answerer._phase2.state == answerer._phase2.COMPLETE):
                answerer_echo.fill(0.0)
                for relative_delay, path_gain in (
                        answerer_echo_path_after_caller_phase3):
                    answerer_echo += _delayed_echo(
                        answerer_tx_history, start, 160,
                        delay_samples=(answerer_echo_delay_samples
                                       + int(relative_delay)),
                        gain=float(path_gain))
            if (cancel_synthetic_echo_after_fallback
                    and fallback_media_path_entered
                    and not v90_retrain_media):
                answerer_echo = np.zeros(160)
            elif v90_retrain_media:
                answerer_echo = _delayed_echo(
                    answerer_tx_history, start, 160,
                    delay_samples=320,
                    gain=float(fallback_retrain_hybrid_echo_gain
                               if fallback_retrain_answerer_echo_gain is None
                               else fallback_retrain_answerer_echo_gain))
            local_to_remote = caller_pcm
            if (fallback_media_path_entered
                    and int(fallback_local_to_remote_delay_samples) > 0):
                local_to_remote = _delayed_echo(
                    caller_tx_history, start, 160,
                    delay_samples=int(fallback_local_to_remote_delay_samples),
                    gain=1.0)
            answerer_rx_pcm = local_to_remote + answerer_echo
            if fallback_rx_audio_hook is not None:
                fallback_rx_audio_hook(
                    caller, answerer, start, caller_rx_pcm, answerer_rx_pcm)
            if capture_fallback_pcm:
                caller_rx_capture.append(caller_rx_pcm.astype("<i2"))
                answerer_rx_capture.append(answerer_rx_pcm.astype("<i2"))
            context = {
                "packet_playout_deadline_monotonic": media_origin + start / 8000,
                "payload_samples": 160,
                "timestamp": start,
            }
            answerer_output = answerer.feed_g711(
                bytes(v34_g711.encode(answerer_rx_pcm, law)), law, context)
            if (peer_v34_mp_request is not None
                    and not peer_v34_mp_request_applied
                    and answerer._v34_fallback_engine is not None):
                # The captured digital-peer MP describes the receive profile
                # it requests from our analogue-caller TX. Apply it only to
                # that peer instance; the reverse direction stays unchanged.
                request = dict(peer_v34_mp_request)
                if "precoder" in request:
                    request["precoder"] = list(request["precoder"])
                answerer._v34_fallback_engine.caller._local_mp.update(request)
                peer_v34_mp_request_applied = True
            if fallback_payloads_started:
                answerer_payload_rx.extend(answerer_output)

            if start in dropped:
                record_lapm_media_state(start, injected_gap=True)
                continue
            packet = struct.pack("!BBHII", 0x80, rtp._rtp_payload_type,
                                 (100 + start // 160) & 0xFFFF,
                                 start & 0xFFFFFFFF, 0x12345678)
            packet += bytes(v34_g711.encode(caller_rx_pcm, law))
            packet_index = start // 160
            arrival_jitter = ((packet_index % 4) - 1.5) * (
                float(arrival_jitter_ms) / 1000.0)
            caller_output = rtp.handle_rtp_packet(
                packet, received_at=(media_origin
                                     + start / 8000
                                     * (1.0 + float(arrival_clock_ppm) * 1e-6)
                                     + 0.020 + arrival_jitter))
            if fallback_payloads_started:
                caller_payload_rx.extend(caller_output)

            if (fallback_payloads_after_retrain is not None
                    and postconnect_v90_retrain_requested
                    and not fallback_payloads_after_retrain_started
                    and any(name == "retrain-v34-fallback-restarted"
                            for name, _ in answerer.events)
                    and any(name == "postconnect-v90-retrain-v34-resumed"
                            for name, _ in caller.events)
                    and caller._v34_fallback_engine is not None
                    and answerer._v34_fallback_engine is not None
                    and caller._v34_fallback_engine.connected
                    and answerer._v34_fallback_engine.connected):
                if retained_lapm_links != (
                        caller._v34_fallback_engine._v42,
                        answerer._v34_fallback_engine._v42):
                    raise AssertionError(
                        "paired V.90 retrain replaced an established LAPM link")
                caller.feed_bytes(fallback_payloads_after_retrain[0])
                answerer.feed_bytes(fallback_payloads_after_retrain[1])
                fallback_payloads_after_retrain_started = True

            both_fallbacks_started = (
                caller._v34_fallback_engine is not None
                and answerer._v34_fallback_engine is not None)
            if (fallback_payloads is not None
                    and not fallback_v42_started
                    and both_fallbacks_started
                    and caller._v34_fallback_engine.connected
                    and answerer._v34_fallback_engine.connected):
                if fallback_v42_start_delays_samples is None:
                    start_delays = (0, 0)
                else:
                    if (not isinstance(fallback_v42_start_delays_samples, tuple)
                            or len(fallback_v42_start_delays_samples) != 2
                            or any(not isinstance(delay, int) or delay < 0
                                   for delay in fallback_v42_start_delays_samples)):
                        raise ValueError(
                            "fallback_v42_start_delays_samples must be a pair "
                            "of non-negative sample counts")
                    start_delays = fallback_v42_start_delays_samples
                if fallback_v42_start_origin_sample is None:
                    fallback_v42_start_origin_sample = int(start)
                endpoints = (
                    (caller, "originate", "answer"),
                    (answerer, "answer", "originate"),
                )
                for index, (engine, tx_role, rx_role) in enumerate(endpoints):
                    if (not fallback_v42_endpoint_started[index]
                            and start >= (fallback_v42_start_origin_sample
                                          + start_delays[index])):
                        # Start fallback LAPM through the real DCE policy path.
                        # Per-endpoint delays let paired tests reproduce a peer
                        # whose LAPM layer becomes active after V.34 fallback.
                        self._start_fallback_v42_via_dce(
                            engine, tx_role=tx_role, rx_role=rx_role, law=law,
                            role_override=(
                                None if fallback_v42_role_overrides is None
                                else fallback_v42_role_overrides[index]),
                            force_detection_after_v8=force_detect_by_endpoint[index],
                            frame_size=fallback_v42_frame_size,
                            lapm_trace=fallback_lapm_trace)
                        fallback_v42_endpoint_started[index] = True
                if all(fallback_v42_endpoint_started):
                    fallback_v42_started_sample = int(start)
                    fallback_v42_started = True
                    if (fallback_payloads is not None
                            and not fallback_payloads_after_rtp_gap
                            and fallback_payloads_after_lapm_samples is None):
                        caller.feed_bytes(fallback_payloads[0])
                        answerer.feed_bytes(fallback_payloads[1])
                        fallback_payloads_started = True
            record_lapm_media_state(start)
            if (lapm_connected_sample is None
                    and caller._v34_fallback_engine is not None
                    and answerer._v34_fallback_engine is not None):
                caller_link = caller._v34_fallback_engine._v42
                answerer_link = answerer._v34_fallback_engine._v42
                if (caller_link is not None and answerer_link is not None
                        and caller_link.state == "lapm"
                        and answerer_link.state == "lapm"):
                    lapm_connected_sample = int(start)
            if (fallback_dce_connect_gate
                    and lapm_connected_sample is not None
                    and not fallback_dce_connect_emitted):
                # Exercise the production eligibility predicate and Hayes
                # result formatter used by run_pty before paired payload
                # injection. The paired fallback engines remain the live
                # protocol endpoints; these sessions provide their DCE view.
                from modem_dce import AtDce

                for label, engine, tx_role, rx_role in (
                        ("caller", caller, "originate", "answer"),
                        ("answerer", answerer, "answer", "originate")):
                    dce_sock, dce_peer = socket.socketpair(
                        socket.AF_UNIX, socket.SOCK_DGRAM)
                    dce_session = OnlineAudioSession(
                        dce_sock, ("127.0.0.1", 9), tx_role, rx_role,
                        line_mode="chain", g711_codec=law, v42=True,
                        v42_frame_size=fallback_v42_frame_size,
                        v42bis=False)
                    try:
                        chain = dce_session._v22
                        chain.kind = "v90"
                        chain.engine = engine
                        chain.negotiated_protocols = V8_PROTOCOL_LAPM_V42
                        dce_session._want_v42 = True
                        # The paired harness started this exact V.42 link via
                        # OnlineAudioSession.poll() above. Reflect that state
                        # in this DCE-facing session wrapper.
                        dce_session._v42_started = True
                        if not dce_session.negotiation_complete:
                            raise AssertionError(
                                f"{label} DCE CONNECT gate remained closed "
                                f"after paired LAPM establishment")
                        dce = AtDce()
                        dce.verbose = True
                        dce.connect_speed = dce_session.line_speed
                        dce.connect_suffix = dce_session.connect_suffix
                        dce.connect_modulation = dce_session.modulation_name
                        dce.notify_call_progress("CONNECT")
                        result = dce.take_notifications()
                        expected = (
                            f"CONNECT {dce_session.line_speed}"
                            f"{dce_session.connect_suffix}").encode("ascii")
                        if expected not in result:
                            raise AssertionError(
                                f"{label} emitted {result!r}, expected {expected!r}")
                        fallback_dce_connect_results[label] = result
                    finally:
                        dce_session.stop()
                        dce_sock.close()
                        dce_peer.close()
                fallback_dce_connect_emitted = True
            if caller._controller.done and answerer._phase2.done:
                if not continue_v34_fallback:
                    break
                handoff = next((item for item in caller._renderer.handoffs
                                if item.signal == "handoff-v34-answer-phase3"), None)
                hold_complete = (handoff is not None
                                 and start >= handoff.at
                                 + int(v34_fallback_hold_samples))
                if both_fallbacks_started and hold_complete:
                    if fallback_payloads is None:
                        break
                    expected_at_caller = fallback_payloads[1]
                    expected_at_answerer = fallback_payloads[0]
                    if fallback_payloads_after_retrain is not None:
                        expected_at_caller += fallback_payloads_after_retrain[1]
                        expected_at_answerer += fallback_payloads_after_retrain[0]
                    if (bytes(caller_payload_rx) == expected_at_caller
                            and bytes(answerer_payload_rx) == expected_at_answerer
                            and (fallback_payloads_after_retrain is None
                                 or fallback_payloads_after_retrain_started)):
                        caller_link = caller._v34_fallback_engine._v42
                        answerer_link = answerer._v34_fallback_engine._v42
                        all_acked = (
                            caller_link is not None
                            and answerer_link is not None
                            and not caller_link._unacked
                            and not answerer_link._unacked
                            and caller_link.stats["wire_acked"]
                            >= len(fallback_payloads[0])
                            and answerer_link.stats["wire_acked"]
                            >= len(fallback_payloads[1]))
                        if not fallback_wait_for_lapm_ack or all_acked:
                            break
        if fallback_lapm_trace:
            caller._test_v42_media_trace = list(lapm_media_trace)
            answerer._test_v42_media_trace = list(lapm_media_trace)
            caller._test_v42_started_sample = fallback_v42_started_sample
            answerer._test_v42_started_sample = fallback_v42_started_sample
            caller._test_lapm_connected_sample = lapm_connected_sample
            answerer._test_lapm_connected_sample = lapm_connected_sample
            caller._test_fallback_gap_sample = fallback_gap_sample
            answerer._test_fallback_gap_sample = fallback_gap_sample
            caller._test_v42_lapm_bad_frame_trace = {
                label: (None if engine._v34_fallback_engine is None
                        or engine._v34_fallback_engine._v42 is None else
                        engine._v34_fallback_engine._v42.bad_frame_trace_report())
                for label, engine in (("caller", caller), ("answerer", answerer))}
            answerer._test_v42_lapm_bad_frame_trace = (
                caller._test_v42_lapm_bad_frame_trace)
        if request_preconnect_v90_retrain:
            caller._test_preconnect_retrain_stage_trace = preconnect_retrain_stage_trace
        if capture_fallback_pcm:
            caller._test_rx_pcm = np.concatenate(caller_rx_capture)
            answerer._test_rx_pcm = np.concatenate(answerer_rx_capture)
            caller._test_tx_pcm = np.concatenate(caller_tx_capture)
            answerer._test_tx_pcm = np.concatenate(answerer_tx_capture)
            left = caller._v34_fallback_engine
            right = answerer._v34_fallback_engine
            left_rx = (None if left is None else
                       getattr(getattr(left, "answerer", None), "_p3", {})
                       .get("rcv"))
            right_inner = None if right is None else getattr(right, "caller", None)
            right_rx = None if right_inner is None else getattr(right_inner, "_p3_rcv", None)
            caller._test_v34_fallback_symbols = (
                None if left_rx is None else np.asarray(left_rx.idx, dtype=np.int16))
            answerer._test_v34_fallback_symbols = (
                None if right_rx is None else np.asarray(right_rx.idx, dtype=np.int16))
        caller._test_postconnect_retrain_eligible_sample = (
            postconnect_v90_retrain_eligible_sample)
        answerer._test_postconnect_retrain_eligible_sample = (
            postconnect_v90_retrain_eligible_sample)
        caller._test_postconnect_retrain_request_sample = (
            postconnect_v90_retrain_request_sample)
        answerer._test_postconnect_retrain_request_sample = (
            postconnect_v90_retrain_request_sample)
        if fallback_payloads is not None:
            caller._test_fallback_payload_rx = bytes(caller_payload_rx)
            answerer._test_fallback_payload_rx = bytes(answerer_payload_rx)
        if fallback_dce_connect_gate:
            caller._test_dce_connect_result = fallback_dce_connect_results.get(
                "caller")
            answerer._test_dce_connect_result = fallback_dce_connect_results.get(
                "answerer")
        answerer._test_dropped_info1d_ranges = list(dropped_info1d_ranges)
        caller._test_dropped_phase3_ranges = list(dropped_phase3_ranges)
        answerer._test_v90_tx_history = answerer_tx_history
        return caller, answerer, rtp

    def _start_fallback_v42_via_dce(self, engine, *, tx_role, rx_role, law,
                                    role_override=None,
                                    force_detection_after_v8=False,
                                    frame_size=128, lapm_trace=False):
        """Exercise OnlineAudioSession's normal V.42 startup on a live fallback."""
        from v8 import V8_PROTOCOL_LAPM_V42

        sock, peer = socket.socketpair(socket.AF_UNIX, socket.SOCK_DGRAM)
        session = OnlineAudioSession(
            sock, ("127.0.0.1", 9), tx_role, rx_role,
            line_mode="chain", g711_codec=law, v42=True,
            v42_frame_size=frame_size, v42bis=False)
        chain = session._v22
        chain.kind = "v90"
        chain.engine = engine
        chain.negotiated_protocols = V8_PROTOCOL_LAPM_V42
        try:
            env = {}
            if role_override is not None:
                env["V42_ROLE_OVERRIDE"] = role_override
            if force_detection_after_v8:
                env["V42_FORCE_DETECT_AFTER_V8"] = "1"
            if lapm_trace:
                env["V42_LAPM_TRACE"] = "1"
            if env:
                with mock.patch.dict("os.environ", env):
                    session.poll(time.monotonic())
            else:
                session.poll(time.monotonic())
            if not session._v42_started:
                raise AssertionError(
                    f"DCE did not start fallback V.42 for {tx_role} role")
            if force_detection_after_v8:
                fallback = getattr(engine, "_v34_fallback_engine", engine)
                engine._test_v42_force_detect_started = (
                    getattr(fallback, "v42_mode", None) == "detect"
                    and not getattr(fallback, "_v42_direct", True))
        finally:
            # The test owns this engine for the rest of the paired media loop.
            chain.engine = None
            session.stop()
            sock.close()
            peer.close()

    def _run_paired_phase2(self, *, law, hide_post_l2_tone_b_ms=0,
                           probe_offset_hz_at_1050=0.0,
                           caller_late_l2_echo_policy="fail",
                           caller_echo_delay_ms=400):
        caller = V90AnalogueCallerPhase2(
            late_l2_echo_policy=caller_late_l2_echo_policy)
        caller_rx = V90AnaloguePhase2Adapter(
            caller, rx_to_media=lambda sample: sample)

        def caller_probe(duration, rms):
            signal = np.asarray(v34_call.probing_signal(duration, rms), dtype=float)
            if not probe_offset_hz_at_1050:
                return signal
            ratio = 1.0 + probe_offset_hz_at_1050 / 1050.0
            positions = np.arange(len(signal), dtype=float) * ratio
            return np.interp(positions, np.arange(len(signal)), signal,
                             left=signal[0], right=0.0)

        caller_tx = V90AnaloguePhase2Transmitter(
            info0a_fields={"rate3429": 1, "disallow3429": 1},
            info1a_builder=lambda mode: v90.build_info1a_v90(
                md_len=0, uinfo=90, analog_symbol_rate=5, digital_symbol_rate=6,
                freq_offset=0),
            probe_builder=caller_probe,
            nominal_peak=1200, probe_rms=300)

        answerer = V90DigitalPhase2Controller()
        answerer_rx = V90DigitalPhase2Adapter(
            answerer, rx_to_media=lambda sample: sample, trace_events=True)
        answerer_tx = V90DigitalPhase2Transmitter(
            signals=V90DigitalPhase2Signals(
                info0d={"rate3429": 1, "alaw": 0}, amplitude=1200),
            probe_builder=v34_call.probing_signal,
            tone_b_peak=800, probe_rms=300)

        caller_tx.submit(caller_rx.start(0))
        answerer_tx.submit(answerer_rx.start(0))
        caller_info0a_done = caller_echo_done = answerer_echo_done = False
        drop_tone_b_until = None

        for start in range(0, 8 * 8000, 160):
            caller_pcm = caller_tx.render(160)
            answerer_pcm = answerer_tx.render(160)
            now = start + 160

            caller_wire = v34_g711.roundtrip(answerer_pcm, law)
            answerer_wire = v34_g711.roundtrip(caller_pcm, law)
            if drop_tone_b_until is not None and start < drop_tone_b_until:
                caller_wire = np.zeros_like(caller_wire)

            info0a_end = caller_tx._completion["info0a-complete"]
            if not caller_info0a_done and info0a_end <= now:
                caller_info0a_done = True
                caller_tx.submit(caller_rx.feed_tx_complete(
                    "info0a_complete", media_at=info0a_end, tx_now=now))

            for action in caller_rx.feed(caller_wire, rx_at=start, tx_now=now):
                caller_tx.submit([action])
            for action in answerer_rx.feed(answerer_wire, rx_at=start, tx_now=now):
                answerer_tx.submit([action])

            if (hide_post_l2_tone_b_ms and drop_tone_b_until is None
                    and answerer.state == answerer.WAIT_A_AFTER_PROBE):
                drop_tone_b_until = now + hide_post_l2_tone_b_ms * 8

            # This older controller-only impairment fixture keeps explicit
            # measured echo events. Production-engine coverage below uses a
            # PCM return path and detects the echo from samples.
            if (caller._local_l2_start_at is not None and not caller_echo_done
                    and now >= caller._local_l2_start_at
                    + caller_echo_delay_ms * 8):
                caller_echo_done = True
                if caller.state in (caller.TX_LOCAL_PROBES,
                                     caller.WAIT_TONE_B_AFTER_PROBES):
                    caller_rx.feed_l2_echo_complete(
                        rx_at=(caller._local_l2_start_at
                               + caller_echo_delay_ms * 8), tx_now=now)
            if (answerer.l2_started_at is not None and not answerer_echo_done
                    and now >= answerer.l2_started_at + 3200):
                answerer_echo_done = True
                answerer_rx.feed_external_event(
                    "l2_echo_complete", rx_at=answerer.l2_started_at + 3200,
                    tx_now=now)

            for action in caller_rx.tick(media_now=now):
                caller_tx.submit([action])
            # The digital-side controller also owns a bounded recovery timer
            # for a missing caller Tone A. Advance it on the same media clock
            # as the caller; relying only on receive events lets the recovery
            # action arrive late and does not model the production engine's
            # periodic timer service.
            for action in answerer.tick(now):
                answerer_tx.submit([action])

            if caller.done and answerer.done:
                break

        assert caller.state == caller.COMPLETE, (
            caller.state, caller_rx.event_trace[-12:],
            answerer.state, answerer_rx.event_trace[-12:])
        assert answerer.state == answerer.COMPLETE
        assert caller.mode == 6
        assert answerer.info1a["mode"] == 6
        assert any(h.signal == "handoff-v90-phase3" for h in caller_tx.handoffs)
        assert any(h.signal == "handoff-v90-phase3" for h in answerer_tx.handoffs)
        assert caller_echo_done and answerer_echo_done
        assert not caller_tx.late_actions
        assert not answerer_tx.late_actions
        return caller, answerer_rx
