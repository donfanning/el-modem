"""A V.92 analogue caller session (PCM upstream), the counterpart of ``v92_digital_session``: Phase 3 through data.

``feed(samples)`` takes the line from the digital modem as linear PCM, ``read(count)`` returns the samples to
send. Phase 3 is ``v92_phase3_exchange.AnalogueSide``; Phase 4 is ``v92_phase4_controller.analogue_side``, reading the
digital modem's TRN2d from the end of its R-bar-i (found on the way); data carries a V.42 LAPM link
(``V90LapmLink``, calling party) through the PCM-upstream transmitter and the V.90 downstream receiver.

For a test peer or the loopback harness, not for a product: unit-gain line, no equaliser, the assumptions of the offline pieces.
"""

from __future__ import annotations

import math

import numpy as np

import v34_g711
import v92_cp
import v92_data_mode as dm
import v92_fast_parameter_exchange as fpe
import v92_fpe_signals
import v92_ja
import v92_phase3_exchange as p3x
import v92_phase4_controller as p4c
import v92_renegotiation as renegotiation
from v90_lapm import V90LapmLink
from v92_phase3_rx import (PatternTracker, TX_DELAY_BANK, TX_DELAY_BULK,
                           TX_DELAY_HISTORY, TX_DELAY_PHASES, TX_DELAY_TAPS)
import v92_anspcm  # noqa: F401  (kept with the other V.92 pieces)


def _masks():
    return [tuple(range(16 + 16 * i, 32 + 16 * i)) for i in range(6)]


class DownstreamMarkFill:
    """Remove the one-frame runs of marks a digital modem inserts when its data falls behind the downstream rate.

    At 56000 (42-bit frames) Astound's server supplies data at only 50667 bit/s and makes up the rest with 42 ones
    at octet boundaries of its HDLC stream, about 127 times a second, inside frames as well as between them
    (2026-11-20, batch i). Seven ones abort an HDLC frame, so LAPM silently lost every frame one landed in: the XID
    response and the first banner frame on every call, the whole banner on two of three. With the inserted ones
    taken out the same recordings decode every frame with a good FCS. A run of ``fb`` to ``fb + 10`` ones (frame
    data contributes at most five on each side) loses ``fb``; longer runs are idle and pass unchanged.
    """

    SLACK = 10

    def __init__(self) -> None:
        self._ones = 0
        self._passing = False
        self.removed = 0

    def push(self, bits, fb: int) -> list[int]:
        out: list[int] = []
        limit = int(fb) + self.SLACK
        for bit in bits:
            if bit:
                if self._passing:
                    out.append(1)
                    continue
                self._ones += 1
                if self._ones > limit:
                    out.extend([1] * self._ones)
                    self._ones, self._passing = 0, True
                continue
            run, self._ones, self._passing = self._ones, 0, False
            if fb <= run <= limit:
                run -= fb
                self.removed += 1
            out.extend([1] * run)
            out.append(0)
        return out


class V92AnalogueSession:
    # V.92 9.5.2.1.10 (zero-length DIL): after Ri we send CPt until R-bar-i acknowledges it. Astound acknowledges
    # after 6 CPt frames (about 0.75 s), but twice held Ri through ours and gave up after 6 s (2026-11-19, calls
    # d/1 and d/3). After this many (about 2 s) the session fails with ``retrain_requested`` so the engine can
    # retrain before the peer does (9.5.2.2, 9.7.2.1).
    CPT_UNACKNOWLEDGED_FRAMES = 16
    # 9.5.2.2: Tone B (1200 Hz) during Phase 3 is the digital modem asking for a retrain, which the analogue modem
    # answers (9.7.2.2) after detecting it for more than 50 ms. Astound sent it 0.9 s into our Phase 3 and we
    # kept on until the call died (2026-11-20, call 1). Sd and Ri repeat every 6 symbols (1333 Hz), well clear.
    TONE_B_HZ = 1200.0
    TONE_B_BLOCK = 160
    TONE_B_BLOCKS = 3
    TONE_B_SHARE = 0.8
    TONE_B_MIN_RMS = 100.0

    def __init__(self, *, law: str, lu: float = 1000.0, ack_wait: int = 1600,
                 drn_down: int = 2,
                 dil_ucodes: tuple[int, ...] = (),
                 fractional_tx_timing: bool = False,
                 enable_rate_renegotiation: bool = False,
                 enable_fast_parameter_exchange: bool = False,
                 lapm_detect: bool = True,
                 v42bis_compression: tuple[int, int, int] | None = None,
                 v42_compression_recovery: bool = False,
                 trn2u_noise: float = 0.0, pad_aware_upstream: bool = False,
                 downstream_mark_fill: bool = False):
        self.law = law
        self.mark_fill = DownstreamMarkFill() if downstream_mark_fill else None
        # Upstream options passed to every Phase 4 side (see p4c.analogue_side).
        self._upstream_options = {"trn2u_noise": float(trn2u_noise), "pad_aware_upstream": bool(pad_aware_upstream)}
        descriptor = v92_ja.build_ja_descriptor(N=len(dil_ucodes), Ucodes=list(dil_ucodes),
                                                upstream_rates=(28000.0, 32000.0, 36000.0))
        cpt_bits = v92_cp.build_cpu(kind=v92_cp.CPT, drn=2, constellations=_masks(),
                                    interval_sets=(0, 1, 2, 3, 4, 5), alaw=(law == "pcma"))
        self._p3 = p3x.AnalogueSide(
            descriptor, lu=lu, law=law, cpt_bits=cpt_bits,
            fractional_tx_timing=fractional_tx_timing)
        self.profile = dm.downstream_profile_from_cpu(v92_cp.parse_cpu(cpt_bits))
        self._ack_wait, self._drn_down = ack_wait, drn_down
        self._enable_rate_renegotiation = bool(enable_rate_renegotiation)
        self._lapm_detect = bool(lapm_detect)
        if type(enable_fast_parameter_exchange) is not bool:
            raise TypeError("enable_fast_parameter_exchange must be bool")
        self._enable_fpe = enable_fast_parameter_exchange
        self._next_drn_down: int | None = None
        self._reneg: dict | None = None
        self._reneg_tracker = PatternTracker(families=("ru",)) if self._enable_rate_renegotiation else None
        self._reneg_total = 0
        self._reneg_probe = bytearray()
        self._reneg_probe_base = 0
        self._reneg_candidate: int | None = None
        self._fpe_rx_probe = bytearray()
        self._fpe_rx_base = 0
        self._fpe_rx_total = 0
        self._fpe_listener: v92_fpe_signals.RfSequenceDetector | None = None
        self._v42bis_compression = v42bis_compression
        self._v42_compression_recovery = bool(v42_compression_recovery)
        self._p4: p4c.Side | None = None
        self._p4_rx_attached = False
        self._p4_suvu_start_recorded = False
        self._down: list[int] = []                         # codewords received, kept until Phase 4 starts reading
        self._down_base = 0
        self._tx_hist_len = 0
        self._tx_delay_history = np.zeros(TX_DELAY_HISTORY, dtype=float)
        self._tx_delay_samples_seen = 0
        self._tx_delay_command_history = np.zeros(TX_DELAY_BULK, dtype=float)
        self._tx_delay_current = 0.0
        self._tx_delay_target = 0.0
        self._tx_delay_ramp_left = 0
        self._tx_delay_ramp_step = 0.0
        self.stage = "phase3"
        self.retrain_requested = False
        self.retrain_reason: str | None = None
        self._tone_b_pending = np.zeros(0)
        self._tone_b_run = 0
        n = np.arange(self.TONE_B_BLOCK)
        self._tone_b_ref = np.exp(-2j * np.pi * self.TONE_B_HZ * n / 8000.0)
        self.lapm: V90LapmLink | None = None
        self._received = bytearray()
        self._queued = bytearray()
        self.events: list[tuple[str, object]] = []
        self._rt = PatternTracker(families=("ru",))        # Ri / R-bar-i: where the digital Phase 4 starts
        self._rt_origin = 0
        self._ri_flip: int | None = None
        self._rt_fed = 0
        # V.92 uses one analogue transmit level through Phase 3 and Phase 4.
        # The Phase-4 TRN2u table is normalized to LU; resetting it here to a
        # convenient test value makes TRN2u/SUVu several dB hotter than the
        # peer's Phase-3 level estimate.
        self.lu = float(lu)

    @property
    def connected(self) -> bool:
        return self.stage == "data"

    def phase4_diagnostics(self) -> dict | None:
        """Return bounded wire-progress and decoder diagnostics for a live capture."""
        if self._p4 is None:
            return None
        return self._p4.diagnostics()

    def feed(self, samples) -> None:
        x = np.asarray(samples, dtype=float).reshape(-1)
        codes = (v34_g711.encode(np.rint(x).astype(np.int32), self.law).astype(np.uint8) & 0xFF).tolist()
        # What arrives is the digital modem's codewords already decoded by the caller of ``feed``; re-encoding the
        # decoded value returns the same codeword.
        self._down.extend(codes)
        if not self._p4_rx_attached:
            self._p3.feed(x)
            p3 = self._p3
            if (self.stage == "phase3" and p3._ri_seen and not p3._rbar_seen and not p3._nonzero_dil
                    and p3.cpt_frames_sent >= self.CPT_UNACKNOWLEDGED_FRAMES):
                self.stage = "failed"
                self.retrain_requested = True
                self.retrain_reason = "V.92 digital modem did not acknowledge CPt with R-bar-i"
                self.events.append(("v92-phase3-cpt-unacknowledged", {"cpt_frames": p3.cpt_frames_sent}))
                return
            if self.stage in ("phase3", "phase4-tx") and self._tone_b_heard(x):
                self.stage = "failed"
                self.retrain_requested = True
                self.retrain_reason = "V.92 digital modem sent Tone B during Phase 3"
                self.events.append(("v92-phase3-tone-b", {"blocks": self._tone_b_run}))
                return
            for name, at in self._rt.feed(x):
                if self._p3._ri_seen and name.endswith("bar") and self._ri_flip is None:
                    self._ri_flip = at
            # TX follows E1u immediately; the receiver can be attached only
            # after R-bar-i, which may arrive later (especially with nonzero DIL).
            if self._p4 is not None and self._ri_flip is not None:
                self._attach_phase4_rx()
        elif self._p4 is not None:
            if self._reneg is not None and self._reneg.get("fpe"):
                self._feed_fpe_response_probe(x, bytes(codes))
            elif self._reneg is not None and self._reneg.get("rx_started"):
                # After the peer's R-bar is detected, the new Phase 4 receiver
                # owns the incoming TRN2d/control stream even while our outbound
                # Ru pair is still finishing.  Do not send those samples back to
                # the old data decoder during that overlap.
                receive_side = self._reneg.get("new_side", self._p4)
                receive_side.receive(bytes(codes))
                if receive_side is self._p4:
                    self._pump()
            elif self._reneg is not None and self._reneg.get("initiator"):
                self._feed_rate_renegotiation_probe(x, bytes(codes))
            elif self._reneg is not None and self._reneg.get("fpe"):
                self._feed_fpe_response_probe(x, bytes(codes))
            elif (self.stage == "data" and self._reneg is None and self._enable_fpe
                  and self._fpe_listener is not None):
                self._feed_fpe_listener(x, bytes(codes))
            elif self.stage == "data" and self._enable_rate_renegotiation:
                self._feed_rate_renegotiation_probe(x, bytes(codes))
            else:
                self._p4.receive(bytes(codes))
                self._pump()

    def _tone_b_heard(self, x: np.ndarray) -> bool:
        """Whether Tone B has filled TONE_B_BLOCKS consecutive blocks (9.5.2.2)."""
        self._tone_b_pending = np.concatenate((self._tone_b_pending, x))
        block = self.TONE_B_BLOCK
        while len(self._tone_b_pending) >= block:
            seg = self._tone_b_pending[:block]
            self._tone_b_pending = self._tone_b_pending[block:]
            power = float(np.mean(seg ** 2))
            tone = 2.0 * abs(np.dot(seg, self._tone_b_ref)) ** 2 / block ** 2
            if power >= self.TONE_B_MIN_RMS ** 2 and tone >= self.TONE_B_SHARE * power:
                self._tone_b_run += 1
                if self._tone_b_run >= self.TONE_B_BLOCKS:
                    return True
            else:
                self._tone_b_run = 0
        return False

    def _start_phase4_tx(self, e1u_end: int) -> None:
        if self._p4 is not None:
            return
        initial_sign = self._p3._cpt_prev & 1
        self._e1u_sign = initial_sign
        self._p4 = p4c.analogue_side(self.profile, self.law, lu=self.lu, ack_wait=self._ack_wait,
                                     drn=self._drn_down, analogue_trn=12096,
                                     initial_sign=initial_sign, **self._upstream_options)
        self._p4.bit_source = self._tx_bits
        self._tx_after = int(e1u_end)
        trn1u2 = self._p3.trn1u2_start
        self.events.append(("phase4-tx-start", {
            "sample": int(e1u_end),
            "initial_sign": initial_sign,
            "rx_attached_at_tx_start": self._p4_rx_attached,
            "trn1u2_start": trn1u2,
            "offset_mod12": (None if trn1u2 is None else
                             (int(e1u_end) - int(trn1u2)) % 12),
        }))
        self.stage = "phase4-tx"
        if self._ri_flip is not None:
            self._attach_phase4_rx()

    def _attach_phase4_rx(self) -> None:
        if self._p4 is None or self._p4_rx_attached or self._ri_flip is None:
            return
        start = self._ri_flip + 24                    # TRN2d begins after R-bar-i
        self._p4.receive(bytes(self._down[start:]))
        self._p4_rx_attached = True
        self.stage = "phase4"
        self.events.append(("phase4-start", start))

    def _record_phase4_suvu_start(self) -> None:
        if self._p4 is None or self._p4_suvu_start_recorded:
            return
        item = next((entry for entry in self._p4.tx_trace if entry.get("kind") == "SUV"), None)
        if item is None:
            return
        trn1u2 = self._p3.trn1u2_start
        sample = self._tx_after + int(item["unit_at_or_after"])
        self.events.append(("phase4-suvu-start", {
            "sample": sample,
            "trn1u2_start": trn1u2,
            "offset_mod12": (None if trn1u2 is None else (sample - int(trn1u2)) % 12),
        }))
        self._p4_suvu_start_recorded = True

    def _pump(self) -> None:
        p4 = self._p4
        if self.stage == "phase4" and p4.connected and p4.data_rx.qualified:
            self.stage = "data"
            self.events.append(("data", None))
            if self._enable_fpe and self._reneg is None:
                self._arm_fpe_listener(p4)
            if self._reneg is not None and self._reneg.get("round_started"):
                rate = int(self._reneg["drn_down"])
                self._drn_down = rate
                if self.lapm is not None and p4.peer_cp is not None and p4.peer_cp.get("rate"):
                    self.lapm.set_v42_rate(int(p4.peer_cp["rate"]))
                if self._reneg.get("fpe"):
                    self.events.append(("v92-fpe-complete", {
                        "peer_rate": None if p4.peer_cp is None else p4.peer_cp.get("rate")}))
                else:
                    self.events.append(("v92-rate-renegotiated", {"drn_down": rate,
                                                                    "peer_rate": None if p4.peer_cp is None else p4.peer_cp.get("rate")}))
                was_fpe = bool(self._reneg.get("fpe"))
                self._reneg = None
                if was_fpe and self._enable_fpe:
                    self._fpe_rx_probe.clear()
                    self._fpe_rx_base = self._fpe_rx_total = 0
                    self._arm_fpe_listener(p4)
        if self.lapm is not None:
            bits = p4.data_rx.take_bits()
            fb = getattr(p4.data_rx, "fb", None)
            if bits and self.mark_fill is not None and fb:
                bits = self.mark_fill.push(bits, fb)
            if bits:
                self.lapm.feed_line_bits(bits)
                self._received.extend(self.lapm.take_received())

    def read(self, count: int) -> np.ndarray:
        if self._reneg is not None and self._reneg.get("tx_active"):
            out = self._read_rate_renegotiation(count)
            if not self._p3.fractional_tx_timing:
                return out
            return self._apply_tx_fractional_timing(
                out, np.full(len(out), self._p3._tx_fractional_delay, dtype=float))
        if self._p4 is None:
            tx_start = self._p3._pos
            out = self._p3.tx(count)
            delay_trace = (self._p3._last_tx_delay_trace.copy()
                           if self._p3.fractional_tx_timing else None)
            e1u_end = self._p3.e1u_end_sample
            if e1u_end is not None:
                # E1u and Phase 4 are one continuous transmit sequence.  Start
                # TRN2u at the exact sample after E1u, even when that falls
                # inside an RTP block. TX starts independently of R-bar-i:
                # for nonzero DIL, R-bar-i can only follow E1u.
                self._start_phase4_tx(int(e1u_end))
                tail = max(0, min(count, int(e1u_end) - tx_start))
                if tail < count:
                    out[tail:] = np.asarray(
                        self._p4.transmit(count - tail), dtype=float)
                    if delay_trace is not None:
                        delay_trace[tail:] = self._p3._tx_fractional_delay
                    self._record_phase4_suvu_start()
            if delay_trace is None:
                return out
            return self._apply_tx_fractional_timing(out, delay_trace)
        # Phase 3's own transmit has finished (E1u went out); Phase 4's takes over.
        out = np.array(self._p4.transmit(count), dtype=float)
        self._record_phase4_suvu_start()
        if not self._p3.fractional_tx_timing:
            return out
        return self._apply_tx_fractional_timing(
            out, np.full(count, self._p3._tx_fractional_delay, dtype=float))

    def _apply_tx_fractional_timing(self, samples: np.ndarray, delays: np.ndarray) -> np.ndarray:
        """Apply opt-in V.92 timing with fixed bulk latency and a sinc phase bank.

        The 37-tap Kaiser-windowed sinc contributes 18 samples of fixed causal
        latency, which the caller applies from the Phase 2 engine's start. The
        per-sample phase bank adds the accumulated 0..2 sample V.92 delay while
        retaining the 8 kHz cadence. Integer phases use an exact sample shift.
        """
        x = np.asarray(samples, dtype=float).reshape(-1)
        d = np.asarray(delays, dtype=float).reshape(-1)
        if len(x) != len(d):
            raise ValueError("TX samples and timing trace must have equal lengths")
        if not self._p3.fractional_tx_timing:
            return x
        if not np.all(np.isfinite(d)) or np.any(d < 0.0) or np.any(d >= 2.0):
            raise ValueError("fractional TX delay outside supported range [0, 2)")
        if not len(x):
            return x.copy()

        # The filter's fixed bulk latency also delays the sample interval to
        # which a protocol phase change belongs. Delay the control trace by the
        # same bulk so the new phase lands at the delayed waveform boundary.
        commands = np.concatenate((self._tx_delay_command_history, d))
        d = commands[:len(x)]
        self._tx_delay_command_history = commands[len(x):].copy()

        # Move the interpolation phase to a new protocol target over one Su
        # period. A coefficient step at the bar edge rings across the 37-tap
        # window and can hide the Su/Su-bar transition from the digital modem.
        effective = np.empty_like(d)
        current = self._tx_delay_current
        target = self._tx_delay_target
        ramp_left = self._tx_delay_ramp_left
        ramp_step = self._tx_delay_ramp_step
        for i, requested in enumerate(d):
            requested = float(requested)
            if requested != target:
                target = requested
                ramp_left = 6
                ramp_step = (target - current) / 6.0
            if ramp_left:
                current += ramp_step
                ramp_left -= 1
                if not ramp_left:
                    current = target
            effective[i] = current
        self._tx_delay_current = current
        self._tx_delay_target = target
        self._tx_delay_ramp_left = ramp_left
        self._tx_delay_ramp_step = ramp_step
        d = effective

        whole = np.floor(d).astype(np.intp)
        frac = d - whole
        # Avoid routing exact integral phases through floating-point sinc taps.
        integer = frac <= 1e-12
        scaled = frac * TX_DELAY_PHASES
        phase_lo = np.floor(scaled).astype(np.intp)
        phase_weight = scaled - phase_lo
        phase_hi = np.minimum(phase_lo + 1, TX_DELAY_PHASES)
        coefficients = (TX_DELAY_BANK[phase_lo] * (1.0 - phase_weight[:, None])
                        + TX_DELAY_BANK[phase_hi] * phase_weight[:, None])

        history = self._tx_delay_history
        combined = np.concatenate((history, x))
        out = np.empty_like(x)
        positions = np.arange(len(x), dtype=np.intp)
        if np.any(integer):
            source = len(history) + positions[integer] - TX_DELAY_BULK - whole[integer]
            out[integer] = combined[source]
        fractional = ~integer
        if np.any(fractional):
            all_windows = np.lib.stride_tricks.sliding_window_view(
                combined, TX_DELAY_TAPS)
            starts = (len(history) + positions[fractional] - whole[fractional]
                      - (TX_DELAY_TAPS - 1))
            windows = all_windows[starts]
            out[fractional] = np.einsum(
                "ij,ij->i", windows, coefficients[fractional, ::-1],
                optimize=False)
        self._tx_delay_history = combined[-TX_DELAY_HISTORY:].copy()
        self._tx_delay_samples_seen += len(x)
        return out

    def set_next_rate_renegotiation(self, *, drn_down: int) -> None:
        """Choose this caller's CPu rate for the next peer-initiated §9.8 exchange.

        Listening/responding is opt-in because it holds a short receive window while
        distinguishing an R signal from ordinary data. This setting does not initiate a
        renegotiation.
        """
        if not self._enable_rate_renegotiation:
            raise RuntimeError("enable_rate_renegotiation must be set when constructing the session")
        if not isinstance(drn_down, int) or not 1 <= drn_down <= 22:
            raise ValueError("drn_down must be an integer in 1..22; zero is the CPu cleardown value")
        self._next_drn_down = drn_down

    def request_rate_renegotiation(self, *, drn_down: int, silent_period: bool = False) -> None:
        """Initiate V.92 §9.8.2 renegotiation at the next PCM upstream frame boundary."""
        if not self._enable_rate_renegotiation:
            raise RuntimeError("enable_rate_renegotiation must be set when constructing the session")
        if self.stage != "data" or self._p4 is None or self.lapm is None or self._reneg is not None:
            raise RuntimeError("rate renegotiation requires an established idle LAPM data session")
        if isinstance(drn_down, bool) or not isinstance(drn_down, int) or not 1 <= drn_down <= 22:
            raise ValueError("drn_down must be an integer in 1..22; zero is the CPu cleardown value")
        if type(silent_period) is not bool:
            raise TypeError("silent_period must be bool")
        old_side = self._p4
        new_side = p4c.analogue_side(self.profile, self.law, lu=self.lu, ack_wait=self._ack_wait,
                                     drn=drn_down, analogue_trn=12096, initial_sign=self._e1u_sign,
                                     **self._upstream_options)
        new_side.bit_source = self._tx_bits
        if silent_period:
            new_side.set_silent_period_request(True)
        self._reneg = {"phase": "drain", "old_side": old_side, "new_side": new_side,
                       "signal": renegotiation.analogue_r_pair(self.lu), "signal_at": 0,
                       "drn_down": drn_down, "tx_active": True, "rx_started": False,
                       "round_started": False, "initiator": True,
                       "silent_period": silent_period}
        self._reneg_tracker = PatternTracker(families=("ru",))
        self._reneg_total = 0
        self._reneg_probe.clear()
        self._reneg_probe_base = 0
        self._reneg_candidate = None
        self.stage = "renegotiating"
        self.events.append(("v92-rate-renegotiation-start", {
            "drn_down": drn_down, "silent_period": silent_period}))

    def request_fast_parameter_exchange(self) -> None:
        """Initiate opt-in V.92 FPE from an established PCM-upstream data session."""
        if not self._enable_fpe:
            raise RuntimeError("enable_fast_parameter_exchange must be set when constructing the session")
        if self.stage != "data" or self._p4 is None or self.lapm is None or self._reneg is not None:
            raise RuntimeError("fast parameter exchange requires an established idle LAPM data session")
        if not hasattr(self._p4, "make_fpe_rm") or self._p4.own_downstream_profile is None:
            raise RuntimeError("current Phase 4 side does not expose a PCM-upstream FPE profile")
        profile = self._p4.fast_upstream_profile
        if profile is None:
            raise RuntimeError("current Phase 4 side has not established its PCM-upstream data profile")
        self._reneg = {
            "phase": "drain", "signal": None, "signal_at": 0,
            "old_side": self._p4, "new_side": None, "drn_down": self._drn_down,
            "tx_active": True, "rx_started": False, "round_started": False,
            "initiator": True, "fpe": True,
            "rf_detector": v92_fpe_signals.RfSequenceDetector(
                self._p4.own_downstream_profile, self.law),
        }
        self._fpe_rx_probe.clear()
        self._fpe_rx_base = self._fpe_rx_total = 0
        self._fpe_listener = None
        self.stage = "renegotiating"
        self.events.append(("v92-fpe-rm-start", None))

    def _new_fpe_side(self, old_side: p4c.Side) -> p4c.Side:
        new_side = p4c.analogue_side(
            old_side.own_downstream_profile, self.law, lu=self.lu,
            ack_wait=self._ack_wait, drn=self._drn_down, analogue_trn=0,
            fast_upstream_profile=old_side.fast_upstream_profile,
            fast_upstream_scale=getattr(old_side, "upstream_scale", 1.0))
        new_side.bit_source = self._tx_bits
        return new_side

    def _arm_fpe_listener(self, side: p4c.Side) -> None:
        profile = getattr(side, "own_downstream_profile", None)
        if profile is not None:
            self._fpe_listener = v92_fpe_signals.RfSequenceDetector(profile, self.law)

    def _feed_fpe_listener(self, samples: np.ndarray, codes: bytes) -> None:
        side = self._p4
        if side is None or self._fpe_listener is None:
            return
        self._fpe_rx_probe.extend(codes)
        self._fpe_rx_total += len(codes)
        matches = self._fpe_listener.feed(codes)
        if matches:
            positions = dict(matches)
            start = positions["rf"]
            end = positions["rf-bar"] + 2 * 12
            start_offset, end_offset = start - self._fpe_rx_base, end - self._fpe_rx_base
            if start_offset < 0 or end_offset > len(self._fpe_rx_probe):
                self.stage = "failed"
                self.events.append(("v92-fpe-failed", "Rf/Rf-bar receive history expired"))
                return
            if start_offset:
                side.receive(bytes(self._fpe_rx_probe[:start_offset]))
                self._pump()
            new_side = self._new_fpe_side(side)
            new_side.receive(bytes(self._fpe_rx_probe[end_offset:]))
            self._reneg = {
                "phase": "drain", "signal": None, "signal_at": 0,
                "old_side": side, "new_side": new_side, "drn_down": self._drn_down,
                "tx_active": True, "rx_started": True, "round_started": False,
                "responder": True, "fpe": True,
            }
            self.stage = "renegotiating"
            self._fpe_rx_probe.clear()
            self._fpe_rx_base = self._fpe_rx_total
            self._fpe_listener = None
            self.events.append(("v92-fpe-rf-detected", start))
            self.events.append(("v92-fpe-rm-response-start", None))
            self._pump()
            return
        safe = max(self._fpe_rx_base, self._fpe_rx_total - 512)
        release = min(len(self._fpe_rx_probe), max(0, safe - self._fpe_rx_base))
        if release:
            side.receive(bytes(self._fpe_rx_probe[:release]))
            del self._fpe_rx_probe[:release]
            self._fpe_rx_base += release
            self._pump()

    def _feed_fpe_response_probe(self, samples: np.ndarray, codes: bytes) -> None:
        reneg = self._reneg
        if reneg is None or not reneg.get("fpe"):
            return
        if reneg.get("rx_started"):
            reneg["new_side"].receive(codes)
            self._pump()
            return
        self._fpe_rx_probe.extend(codes)
        self._fpe_rx_total += len(codes)
        matches = reneg["rf_detector"].feed(codes)
        if matches:
            positions = dict(matches)
            start, end = positions["rf"], positions["rf-bar"] + 2 * 12
            start_offset, end_offset = start - self._fpe_rx_base, end - self._fpe_rx_base
            if start_offset < 0 or end_offset > len(self._fpe_rx_probe):
                self.stage = "failed"
                self.events.append(("v92-fpe-failed", "Rf/Rf-bar receive history expired"))
                return
            if start_offset:
                reneg["old_side"].receive(bytes(self._fpe_rx_probe[:start_offset]))
                self._pump()
            reneg["new_side"].receive(bytes(self._fpe_rx_probe[end_offset:]))
            reneg["rx_started"] = True
            self._fpe_rx_probe.clear()
            self._fpe_rx_base = self._fpe_rx_total
            self.events.append(("v92-fpe-rf-detected", start))
            self.events.append(("v92-fpe-rf-bar-detected", positions["rf-bar"]))
            self._pump()
            return

        # Keep a guard longer than the complete 408-codeword response, routing
        # only older samples through the still-active old data decoder.
        safe = max(self._fpe_rx_base, self._fpe_rx_total - 512)
        release = min(len(self._fpe_rx_probe), max(0, safe - self._fpe_rx_base))
        if release:
            reneg["old_side"].receive(bytes(self._fpe_rx_probe[:release]))
            del self._fpe_rx_probe[:release]
            self._fpe_rx_base += release
            self._pump()

    def _feed_rate_renegotiation_probe(self, samples, codes: bytes) -> None:
        tracker = self._reneg_tracker
        if tracker is None:
            self._p4.receive(codes)
            self._pump()
            return
        receive_side = (self._reneg["old_side"]
                        if self._reneg is not None and self._reneg.get("initiator")
                        else self._p4)
        base = self._reneg_probe_base
        self._reneg_probe.extend(codes)
        self._reneg_total += len(codes)
        events = tracker.feed(samples)
        pending = getattr(self, "_reneg_bar_pending", None)
        if pending is not None:
            self._reneg_bar_pending = None
            events = [("ru-bar", pending)] + list(events)
        for name, at in events:
            if name == "ru" and self._reneg_candidate is None:
                self._reneg_candidate = at
            elif name == "ru-bar" and self._reneg_candidate is not None:
                candidate = self._reneg_candidate
                trn_start = at + 24
                # Rd samples already handed on went to the old data receiver, which is where they belong.
                candidate_offset = max(0, candidate - base)
                trn_offset = trn_start - base
                if trn_offset > len(self._reneg_probe):
                    # R-bar's last symbols are still to come: finish when they have (live Astound, 2026-11-20,
                    # where this failed the call as "history expired").
                    self._reneg_bar_pending = at
                    return
                if trn_offset < 0:
                    self.stage = "failed"
                    self.events.append(("v92-rate-renegotiation-failed", {
                        "reason": "receive history expired before R-bar", "candidate": candidate,
                        "r_bar_at": at, "probe_base": base, "probe_len": len(self._reneg_probe)}))
                    return
                if self._reneg is not None and self._reneg.get("initiator"):
                    old_side = self._reneg["old_side"]
                    new_side = self._reneg["new_side"]
                    if candidate_offset:
                        old_side.receive(bytes(self._reneg_probe[:candidate_offset]))
                        self._pump()
                    new_side.receive(bytes(self._reneg_probe[trn_offset:]))
                    self._reneg_probe.clear()
                    self._reneg_probe_base = self._reneg_total
                    self._reneg_candidate = None
                    self._reneg["rx_started"] = True
                    self.events.append(("v92-rate-Rd-detected", candidate))
                    self.events.append(("v92-rate-trn2d-start", trn_start))
                    self._pump()
                    return
                if candidate_offset:
                    receive_side.receive(bytes(self._reneg_probe[:candidate_offset]))
                    self._pump()
                drn_down = self._next_drn_down if self._next_drn_down is not None else self._drn_down
                # A renegotiation's TRN2u differential encoder starts, like Phase 4's, from E1u's last sign
                # (8.7.6). From 0 every TRN2u sign was inverted when E1u ended on 1, and Astound's CPd prefilter
                # main tap flipped with it (+0.84 against -0.84 before; call 2, 2026-11-19).
                new_side = p4c.analogue_side(self.profile, self.law, lu=self.lu, ack_wait=self._ack_wait,
                                             drn=drn_down, analogue_trn=12096, initial_sign=self._e1u_sign,
                                             **self._upstream_options)
                new_side.bit_source = self._tx_bits
                old_side = self._p4
                self._p4 = new_side
                self._reneg = {"phase": "drain", "old_side": old_side, "new_side": new_side,
                               "signal": renegotiation.analogue_r_pair(self.lu),
                               "signal_at": 0, "drn_down": drn_down, "tx_active": True,
                               "rx_started": True, "round_started": False}
                self.stage = "renegotiating"
                self._p4.receive(bytes(self._reneg_probe[trn_offset:]))
                self._reneg_probe.clear()
                self._reneg_probe_base = self._reneg_total
                self._reneg_candidate = None
                self._next_drn_down = None
                self.events.append(("v92-rate-Rd-detected", candidate))
                self.events.append(("v92-rate-ru-start", None))
                self._pump()
                return

        # Hold 512 samples so a complete 384T R plus its 24T reversal can be
        # recognized before any part is delivered to the old data decoder.
        safe_end = max(self._reneg_probe_base, self._reneg_total - 512)
        if self._reneg_candidate is not None:
            if self._reneg_total <= self._reneg_candidate + 600:
                safe_end = min(safe_end, self._reneg_candidate)
            else:
                self._reneg_candidate = None
        release = min(len(self._reneg_probe), max(0, safe_end - self._reneg_probe_base))
        if release:
            receive_side.receive(bytes(self._reneg_probe[:release]))
            del self._reneg_probe[:release]
            self._reneg_probe_base += release
            self._pump()

    def _read_rate_renegotiation(self, count: int) -> np.ndarray:
        reneg = self._reneg
        out: list[float] = []
        remaining = count
        if reneg["phase"] == "drain":
            prefix = reneg["old_side"].transmit_to_frame_boundary(remaining, 12)
            out.extend(prefix)
            remaining -= len(prefix)
            if reneg["old_side"].units_sent % 12 == 0:
                if reneg["old_side"]._units_out:
                    self.stage = "failed"
                    self.events.append(("v92-rate-renegotiation-failed", "old PCM mapper crossed the frame boundary"))
                    return np.asarray(out + [0.0] * (count - len(out)), dtype=float)
                if reneg.get("fpe"):
                    signal, old_up = reneg["old_side"].make_fpe_rm()
                    reneg["old_side"].fast_upstream_profile = old_up
                    new_side = reneg.get("new_side") or self._new_fpe_side(reneg["old_side"])
                    reneg["signal"], reneg["new_side"] = signal, new_side
                    reneg["phase"] = "RM"
                else:
                    reneg["phase"] = "Ru"
        if reneg["phase"] in ("Ru", "RM") and remaining:
            signal = reneg["signal"]
            take = min(remaining, len(signal) - reneg["signal_at"])
            start = reneg["signal_at"]
            out.extend(signal[start:start + take])
            reneg["signal_at"] += take
            remaining -= take
            if reneg["signal_at"] == len(signal):
                self._p4 = reneg["new_side"]
                self.stage = "phase4"
                reneg["phase"] = "TRN2"
                reneg["round_started"] = True
                self.events.append(("v92-fpe-suvu-start" if reneg.get("fpe")
                                    else "v92-rate-trn2u-start", None))
        if remaining and reneg["phase"] == "TRN2":
            out.extend(self._p4.transmit(remaining))
        return np.asarray(out, dtype=float)

    def _tx_bits(self, n: int) -> list[int]:
        # The originator's 0.75 s ODP window counts its own transmitted bits, so it starts once the downstream receiver
        # is qualified (an equalised line delays that past the window); until then marks go out.
        if self.lapm is None and self._p4 is not None and self._p4.sent_e and self._p4.data_rx.qualified:
            # The calling side transmits LAPM through the negotiated CPd PCM-upstream
            # bearer. CPt is a training profile and does not describe this data rate.
            peer_rate = self._p4.peer_cp.get("rate") if self._p4.peer_cp is not None else None
            rate = int(round(float(peer_rate if peer_rate else self.profile["rate"])))
            self.lapm = V90LapmLink(
                rate=rate, calling_party=True,
                detect=self._lapm_detect,
                compression=self._v42bis_compression,
                compression_recovery=self._v42_compression_recovery)
            if self._queued:
                self.lapm.feed_bytes(bytes(self._queued))
                self._queued.clear()
        if self.lapm is not None:
            return self.lapm.read_tx_bits(n)
        return [1] * n

    def feed_bytes(self, data: bytes) -> None:
        if self.lapm is not None:
            self.lapm.feed_bytes(bytes(data))
        else:
            self._queued.extend(data)

    def take_received(self) -> bytes:
        data = bytes(self._received)
        self._received.clear()
        return data
