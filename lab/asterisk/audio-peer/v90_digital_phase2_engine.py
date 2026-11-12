"""Opt-in RTP-facing V.90 digital-answerer Phase 2 engine.

This joins the independently tested Phase 2 controller, detectors, PCM
renderer, and exact-codeword G.711 transport. An optional Phase 3 factory can
take the validated INFO1a handoff and route later PCM/codewords through the
Phase 3/4 session, including analogue-upstream V.34 B1/data decoding and an
opt-in LAPM bridge. These remain an experimental integration path: paired
end-to-end waveform qualification, production DTE lifecycle and real-trunk
qualification are still required before enabling a production V.90 profile.
"""

from __future__ import annotations

import os
import threading
import time
from collections import deque

import numpy as np

from g711 import alaw_to_linear, linear_to_alaw, linear_to_ulaw, ulaw_to_linear
from v8 import V8_PCM_DIGITAL, V8_PROTOCOL_LAPM_V42
from v90_digital_phase2 import (
    V90DigitalPhase2Signals,
    constrain_info1d_to_info0a,
    project_info1d_rates_from_probe,
)
from v90_digital_phase2_adapter import V90DigitalPhase2Adapter
from v90_digital_phase2_control import V90DigitalPhase2Controller
from v90_digital_phase2_tx import V90DigitalPhase2Transmitter
from v90_retrain import RetrainPreamble
from v90_media_clock import RtpMediaClock
from v34_call import probing_signal


FS = 8000
_G711_SILENCE = {"pcmu": 0xff, "pcma": 0xd5}
RETRAIN_TIMEOUT_SAMPLES = 20 * FS         # the peer must answer the retrain tones within 20 s


def _write_test_phase3_marker(role: str, media_sample: int) -> None:
    """Emit an opt-in filesystem marker for phase-targeted lab impairment."""
    path = os.environ.get("EL_MODEM_TEST_PHASE3_MARKER")
    if not path:
        return
    try:
        from pathlib import Path
        Path(path).write_text(f"{role} {time.monotonic():.9f} {media_sample}\n")
    except OSError:
        # Test diagnostics must not change modem behavior.
        pass


class V90DigitalAnswererPhase2Engine:
    """Expose Phase 2 through V8Chain's opt-in digital-answerer engine API.

    Construct this only from a test/integration factory that also supplies a
    common `media_at` and monotonic `media_origin`. INFO fields and probe level
    are explicit configuration inputs; no guessed line profile is selected.
    """

    kind = "v90"
    bit_rate = 0

    @property
    def supports_v42(self) -> bool:
        fallback = self._v34_fallback_engine
        if fallback is not None:
            return bool(getattr(fallback, "supports_v42", True))
        return bool(self._phase3_enable_lapm)

    @property
    def v42_managed_by_engine(self) -> bool:
        # The native V.90 Phase 3/4 sessions own their LAPM link. V.34
        # fallback uses the DCE's normal V.42 policy and lifecycle.
        return self._v34_fallback_engine is None

    @property
    def v42_calling_party(self) -> bool:
        # On V.90-to-V.34 fallback, this digital answerer takes the V.34 call
        # modem role complementary to the analogue modem's answer role
        # (§9.2.2.1.9/V.90). V.42 §7.2.1.1 assigns its LAPM role from that
        # modulation handshake, making this side the LAPM originator.
        return True

    def start_v42(self, *args, **kwargs) -> None:
        fallback = self._v34_fallback_engine
        if fallback is None:
            raise RuntimeError("V.90 Phase 4 manages its own LAPM link")
        fallback.start_v42(*args, **kwargs)

    @property
    def _v42(self):
        fallback = self._v34_fallback_engine
        if fallback is not None:
            return getattr(fallback, "_v42", None)
        phase3 = self._phase3
        link = None if phase3 is None else getattr(phase3, "lapm", None)
        return None if link is None else getattr(link, "_v42", None)

    def __init__(self, *, role: str, negotiation: dict, media_at: int,
                 media_origin: float, law: str,
                 info0d_fields: dict, info1d_fields: dict,
                 tone_b_peak: float, probe_rms: float,
                 probe_builder=probing_signal, max_l2_samples: int = 4000,
                 phase3_factory=None, phase3_rate_mask: int | None = None,
                 phase3_16point: bool = False, phase3_span: int = 12,
                 phase3_enable_lapm: bool = False,
                 phase3_upstream_equalizer_mu: float | None = None,
                 phase2_l2_echo_tolerance_samples: int = 0,
                 retrain_on_failure: bool = False, max_retrains: int = 2,
                 phase1_silence_samples: int = 0,
                 phase2_info1a_timeout_retrain: bool = False,
                 phase2_project_info1d_rates_from_probe: bool = True,
                 phase2_clock_offset_step_samples: int = 80,
                 phase4_mp_profile: dict | None = None):
        if role != "digital-answerer":
            raise ValueError("this engine implements only the V.90 digital answerer")
        if int(negotiation.get("pcm_negotiated_bits", 0)) != V8_PCM_DIGITAL:
            raise ValueError("V.8 did not negotiate the local digital PCM role")
        if law not in _G711_SILENCE:
            raise ValueError("law must be 'pcmu' or 'pcma'")
        if (isinstance(phase1_silence_samples, bool)
                or not isinstance(phase1_silence_samples, int)
                or not 0 <= phase1_silence_samples <= FS):
            raise ValueError("Phase 1 silence must be an integer sample count in 0..8000")
        self.law = law
        self._decoder = ulaw_to_linear if law == "pcmu" else alaw_to_linear
        self._encoder = linear_to_ulaw if law == "pcmu" else linear_to_alaw
        self._media_at = int(media_at)
        self._phase1_silence_samples = phase1_silence_samples
        self._clock = RtpMediaClock(
            origin=media_origin, media_at=media_at,
            offset_step_samples=phase2_clock_offset_step_samples)
        self._rx_at = 0
        self._failed = False
        self._closed = False
        self._lock = threading.RLock()
        self._split_fallback_tx_lock = (
            os.environ.get("V90_V34_SPLIT_TX_LOCK") == "1")
        self._fallback_tx_condition = threading.Condition()
        self._fallback_tx_engine = None
        self._fallback_tx_inflight = 0
        self.events: list[tuple[str, object]] = []
        self._queued_dte = bytearray()
        self._queue_limit = 65536
        # INFO1a is decoded only after its CRC arrives, but the analogue
        # caller's Phase 3 S preamble begins shortly after that frame. Keep a
        # small receive tail so a delayed Phase 2 callback can seed Phase 3
        # from the actual INFO1a boundary instead of starting late at the
        # current packet. At 8 kHz this is bounded to two seconds of PCM.
        self._phase2_pcm_history: deque[tuple[int, np.ndarray]] = deque()
        self._phase2_pcm_history_samples = 0
        self._phase2_pcm_history_limit = 2 * FS
        self._reported_done_state: str | None = None
        self._reported_handoffs = 0
        self._reported_deadlines = 0
        self._reported_phase2_failure = False
        if phase3_factory is not None and not callable(phase3_factory):
            raise TypeError("phase3_factory must be callable")
        if phase3_factory is not None and phase3_rate_mask is not None:
            raise ValueError("choose phase3_factory or phase3_rate_mask, not both")
        if phase3_span not in (12, 24):
            raise ValueError("phase3_span must be 12 or 24")
        if type(phase3_enable_lapm) is not bool:
            raise ValueError("phase3_enable_lapm must be boolean")
        if type(retrain_on_failure) is not bool:
            raise ValueError("retrain_on_failure must be boolean")
        if type(phase2_info1a_timeout_retrain) is not bool:
            raise ValueError("phase2_info1a_timeout_retrain must be boolean")
        if type(phase2_project_info1d_rates_from_probe) is not bool:
            raise ValueError("phase2_project_info1d_rates_from_probe must be boolean")
        if (isinstance(phase2_clock_offset_step_samples, bool)
                or not isinstance(phase2_clock_offset_step_samples, int)
                or not 1 <= phase2_clock_offset_step_samples <= FS):
            raise ValueError(
                "phase2_clock_offset_step_samples must be an integer in 1..8000")
        if (isinstance(max_retrains, bool) or not isinstance(max_retrains, int)
                or not 1 <= max_retrains <= 5):
            raise ValueError("max_retrains must be an integer in 1..5")
        if (isinstance(phase2_l2_echo_tolerance_samples, bool)
                or not isinstance(phase2_l2_echo_tolerance_samples, int)
                or not 0 <= phase2_l2_echo_tolerance_samples <= 80):
            raise ValueError("phase2_l2_echo_tolerance_samples must be an integer in 0..80")
        if phase3_upstream_equalizer_mu is not None and not (
                isinstance(phase3_upstream_equalizer_mu, (int, float))
                and not isinstance(phase3_upstream_equalizer_mu, bool)
                and 0.0 < float(phase3_upstream_equalizer_mu) <= 1.0):
            raise ValueError("phase3_upstream_equalizer_mu must be in (0, 1]")
        if phase3_rate_mask is not None:
            from v90_digital_tx import build_jd_bits
            if isinstance(phase3_rate_mask, bool) or not isinstance(phase3_rate_mask, int):
                raise ValueError("phase3_rate_mask must be an integer")
            build_jd_bits(phase3_rate_mask, phase3_16point=bool(phase3_16point))
        self._phase3_factory = phase3_factory
        self._phase3_rate_mask = phase3_rate_mask
        if phase4_mp_profile is not None:
            from v90_mp_policy import validate_answerer_mp_profile
            phase4_mp_profile = validate_answerer_mp_profile(phase4_mp_profile)
        self._phase4_mp_profile = phase4_mp_profile
        self._phase3_16point = bool(phase3_16point)
        self._phase3_span = int(phase3_span)
        self._phase3_upstream_equalizer_mu = (
            None if phase3_upstream_equalizer_mu is None
            else float(phase3_upstream_equalizer_mu))
        # The local profile may implement LAPM, but native V.90 must not
        # start it unless V.8 actually negotiated the common protocol.
        self._v8_protocols = int(negotiation.get("protocols", 0))
        self._v8_lapm_negotiated = bool(
            self._v8_protocols & V8_PROTOCOL_LAPM_V42)
        self._phase3_lapm_configured = bool(phase3_enable_lapm)
        self._phase3_enable_lapm = (
            self._phase3_lapm_configured and self._v8_lapm_negotiated)
        self._phase3_info1d = dict(info1d_fields)
        self._phase2_project_info1d_rates_from_probe = bool(
            phase2_project_info1d_rates_from_probe)
        self._phase3 = None
        self._v34_fallback_engine = None
        info0 = dict(info0d_fields)
        expected_alaw = int(law == "pcma")
        if "alaw" in info0 and int(info0["alaw"]) != expected_alaw:
            raise ValueError("INFO0d law does not match negotiated RTP G.711 law")
        info0["alaw"] = expected_alaw
        # Everything needed to build Phase 2 again after V.90 §9.5 retraining.
        self._phase2_build = dict(
            info0d=info0, info1d=dict(info1d_fields), probe_builder=probe_builder,
            tone_b_peak=float(tone_b_peak), probe_rms=float(probe_rms),
            max_l2_samples=max_l2_samples,
            info1a_timeout_retrain=phase2_info1a_timeout_retrain,
            l2_echo_tolerance_samples=phase2_l2_echo_tolerance_samples)
        self._tx_media = int(media_at)
        self._retrain: RetrainPreamble | None = None
        self._retrain_listener: RetrainPreamble | None = None
        self._retrain_listener_tx_base: int | None = None
        self._retrain_listener_rx_base: int | None = None
        self._retired_phase3: list = []
        self.retrain_count = 0
        self.phase2_history: list[dict] = []     # timing of each Phase 2 that has ended
        self._phase2_start_at = int(media_at)
        self._retrain_on_failure = retrain_on_failure
        self._max_retrains = int(max_retrains)
        self._retrain_wanted = False
        self._retrain_base = 0              # transmit position when the current retrain began
        self._phase3_started_at = 0         # transmit position when the Phase 3 session began
        self._phase2_prebuilt_at: int | None = None
        self._retrain_info0a: dict | None = None
        self._retrain_rx_base = 0
        self._build_phase2(int(media_at), phase1_silence_samples)

    def _build_phase2(self, start_at: int,
                      phase1_silence_samples: int = 0, *,
                      retrain_info0a: dict | None = None,
                      retrain_a_reversal_at: int | None = None) -> None:
        """(Re)create the Phase 2 controller, signals, renderer and adapter at ``start_at``."""
        build = self._phase2_build
        phase2_start_at = int(start_at) + int(phase1_silence_samples)
        self._phase2_start_at = phase2_start_at
        self._phase3_info1d = dict(build["info1d"])
        self._phase2 = V90DigitalPhase2Controller(
            max_l2_samples=build["max_l2_samples"],
            info1a_timeout_retrain=build["info1a_timeout_retrain"],
            l2_echo_tolerance_samples=build["l2_echo_tolerance_samples"])
        self._signals = V90DigitalPhase2Signals(
            info0d=dict(build["info0d"]), info1d=dict(build["info1d"]))
        self._renderer = V90DigitalPhase2Transmitter(
            signals=self._signals, probe_builder=build["probe_builder"],
            tone_b_peak=build["tone_b_peak"], probe_rms=build["probe_rms"],
            # Keep the renderer aligned with the DCE transmit sample clock.
            # The first protocol action is scheduled after the silence; moving
            # the renderer cursor too would transmit that action immediately.
            start_at=int(start_at))
        self._adapter = V90DigitalPhase2Adapter(
            self._phase2, rx_to_media=self._clock.rx_to_media,
            trace_events=os.environ.get("EL_MODEM_V90_PHASE2_TRACE") == "1")
        if retrain_info0a is None:
            self._renderer.submit(self._adapter.start(phase2_start_at))
        else:
            if retrain_a_reversal_at is None:
                raise ValueError("retrain continuation lacks Tone A reversal time")
            # Tone B has been on throughout the digital-initiated retrain.
            # Keep it continuous in the new renderer, then let the controller
            # schedule its §9.2.1.1.3 reversal 40 ± 1 ms after received Tone A.
            from v90_digital_phase2_control import Phase2Action
            self._renderer.submit([Phase2Action(phase2_start_at, "tone_b")])
            self._renderer.submit(self._adapter.start_retrain_continuation(
                phase2_start_at, reversal_at=int(retrain_a_reversal_at),
                info0a=dict(retrain_info0a)))
        self._reported_done_state = None
        self._reported_handoffs = 0
        self._reported_deadlines = 0
        self._reported_phase2_failure = False
        self._phase2_pcm_history.clear()
        self._phase2_pcm_history_samples = 0

    def request_retrain(self, reason: str = "requested") -> bool:
        """Start a V.90 §9.5.1.1 retrain: drop the data link, send silence, then Tone B, and
        run a full Phase 2 again once the peer's Tone A is heard. Returns False if the engine
        is closed or a retrain is already running."""
        with self._lock:
            if self._closed or self._retrain is not None:
                return False
            self._retire_fallback_tx()
            fallback = self._v34_fallback_engine
            if fallback is not None and not fallback.connected:
                # Before B1/CONNECT there is no LAPM state to retain. Discard
                # that incomplete V.34 startup before taking over the media
                # path for the V.90 retrain preamble.
                close = getattr(fallback, "close", None)
                if callable(close):
                    close()
                self._v34_fallback_engine = None
                self._dce_v34_fallback_dsp_capture = None
            self._begin_retrain(initiating=True, reason=reason)
            return True

    @property
    def retraining(self) -> bool:
        return self._retrain is not None

    def _snapshot_phase2(self) -> None:
        p = self._phase2
        if p is None:
            return
        self.phase2_history.append({
            "started_at": self._phase2_start_at, "tx_media": self._tx_media,
            **{name: getattr(p, name, None) for name in (
                "rtd_samples", "b_reversal_at", "l2_started_at", "_second_l2_echo_complete_at")}})

    def _lapm_retrain_snapshot(self) -> dict | None:
        """Capture bounded LAPM state before a failure retrain retires Phase 3."""
        phase3 = self._phase3
        link = getattr(phase3, "lapm", None)
        if link is None and self._v34_fallback_engine is not None:
            link = self._v34_fallback_engine
        if link is None:
            return None
        lapm = getattr(link, "_v42", None)
        if lapm is None:
            lapm = getattr(link, "lapm", None)
        if lapm is None:
            return None
        stats = getattr(lapm, "stats", None)
        events = getattr(lapm, "events", None)
        return {
            "mode": getattr(link, "v42_mode", None),
            "lost": bool(getattr(link, "v42_lost", False)),
            "state": getattr(lapm, "state", None),
            "stats": (dict(stats) if isinstance(stats, dict) else None),
            "t401_expiries": (None if not isinstance(stats, dict)
                              else stats.get("t401_expiries")),
            "t401_bits": getattr(lapm, "_t401_bits", None),
            "tries": getattr(lapm, "_tries", None),
            "bits_sent": getattr(lapm, "bits_sent", None),
            "rate": getattr(lapm, "rate", None),
            "va": getattr(lapm, "va", None),
            "vs": getattr(lapm, "vs", None),
            "vr": getattr(lapm, "vr", None),
            "unacked_frames": len(getattr(lapm, "_unacked", ())),
            "retransmit_queue_frames": len(getattr(lapm, "_retx", ())),
            "poll_out": getattr(lapm, "_poll_out", None),
            "ack_pending": getattr(lapm, "_ack_pending", None),
            "peer_busy": getattr(lapm, "_peer_busy", None),
            "last_events": (list(events[-4:]) if isinstance(events, list) else []),
        }

    def _phase3_retrain_snapshot(self) -> dict | None:
        """Capture the upstream data receiver's state at a failure retrain."""
        phase3 = self._phase3
        if phase3 is None:
            return None
        upstream_data = getattr(phase3, "upstream_data", None)
        upstream = getattr(phase3, "upstream", None)
        receiver = getattr(upstream, "receiver", None)
        equalizer = getattr(receiver, "eq", None)
        timing = getattr(equalizer, "timing_snapshot", None)
        p4 = getattr(phase3, "_p4", None)
        pcm_data = getattr(p4, "data_rx", None)
        pcm_rx = getattr(pcm_data, "rx", None)
        phase4 = getattr(phase3, "phase4", None)
        phase4_tx = getattr(phase4, "transmitter", None)
        phase4_rx = getattr(phase3, "phase4_receiver", None)
        try:
            timing_state = timing() if callable(timing) else None
        except Exception as exc:  # diagnostics must not interrupt call recovery
            timing_state = {"snapshot_error": type(exc).__name__}
        return {
            "state": getattr(phase3, "state", None),
            "stage": getattr(phase3, "stage", None),
            "upstream_phase3_state": getattr(upstream, "state", None),
            "upstream_failure_reason": getattr(upstream, "failure_reason", None),
            "upstream_samples_seen": getattr(upstream, "samples_seen", None),
            # Preserve acquisition diagnostics before retrain retires this
            # Phase 3 session. In particular, an ``acquiring`` state alone
            # cannot distinguish an absent preamble from a rejected timing/
            # correlation candidate.
            "upstream_acquisition": getattr(upstream, "acquisition", None),
            "upstream_training_mse": getattr(upstream, "training_mse", None),
            "upstream_detected_trn_role": getattr(
                upstream, "detected_trn_role", None),
            "upstream_startup_trace": (
                upstream.startup_trace_report()
                if callable(getattr(upstream, "startup_trace_report", None))
                else None),
            "upstream_data_state": getattr(upstream_data, "state", None),
            "upstream_data_failure_reason": getattr(
                upstream_data, "failure_reason", None),
            "upstream_next_symbol": getattr(upstream_data, "next_symbol", None),
            # Preserve the opposite (downstream) Phase 4 boundary as well.
            # A V.90 call can finish the analogue Phase 3 exchange, then
            # retrain because CPt/CP or B1d never completes; without these
            # fields that distinction disappears when the session is retired.
            "phase4_error": getattr(phase3, "phase4_error", None),
            "phase4_state": getattr(phase4, "state", None),
            "phase4_tx_state": getattr(phase4_tx, "state", None),
            "phase4_events": list(getattr(phase4, "events", ())),
            "phase4_tx_output_octets": (None if phase4 is None else
                                        len(getattr(phase4, "_output", ()))),
            "phase4_initial_ri_remaining": getattr(
                phase4, "_initial_ri_remaining", None),
            "phase4_mp_frames_sent": getattr(phase4_tx, "mp_frames_sent", None),
            "phase4_mp_prime_frames_sent": getattr(
                phase4_tx, "mp_prime_frames_sent", None),
            "phase4_rx_bits_seen": getattr(phase4_rx, "bits_seen", None),
            "phase4_profiles": [
                {key: profile.get(key) for key in
                 ("kind", "ack", "drn", "rate", "law", "uinfo")}
                for profile in getattr(phase3, "phase4_profiles", ())[-8:]
            ],
            "phase4_receive_armed": bool(getattr(
                phase3, "_phase4_receive_armed", False)),
            "phase4_cpt_acquired": bool(getattr(
                phase3, "_phase4_cpt_acquired", False)),
            "timing": timing_state,
            "pcm_upstream_rx": (None if pcm_rx is None else {
                **{name: getattr(pcm_rx, name, None) for name in (
                    "state", "frames", "lost_frames", "consecutive_lost_frames",
                    "max_consecutive_lost_frames", "b1_errors",
                    "trellis_violations", "failure")},
                "pending_symbols": len(getattr(pcm_rx, "_pending", ())),
            }),
        }

    def _begin_retrain(self, *, initiating: bool, reason: str,
                       preamble: RetrainPreamble | None = None) -> None:
        self._retire_fallback_tx()
        lapm_snapshot = self._lapm_retrain_snapshot()
        phase3_snapshot = self._phase3_retrain_snapshot()
        self._retrain_info0a = (None if self._phase2.info0a is None else
                                dict(self._phase2.info0a))
        self._snapshot_phase2()
        if self._phase3 is not None:
            self._retired_phase3.append(self._phase3)
            self._phase3 = None
        listener_preamble = (preamble is not None
                             and preamble is self._retrain_listener)
        listener_tx_base = self._retrain_listener_tx_base
        listener_rx_base = self._retrain_listener_rx_base
        self._failed = False
        self._retrain_listener = None
        self._retrain_listener_tx_base = None
        self._retrain_listener_rx_base = None
        self._retrain = preamble or RetrainPreamble(
            "digital", initiating=initiating, peak=self._phase2_build["tone_b_peak"])
        self.retrain_count += 1
        # A passive listener's preamble clocks start when it is first armed,
        # not when its 50 ms detection completes. Preserve those origins so
        # peer-tone timestamps are translated once, rather than adding the
        # detection-time offset a second time.
        self._retrain_base = (
            int(listener_tx_base) if listener_preamble
            and listener_tx_base is not None else self._tx_media)
        self._retrain_rx_base = (
            int(listener_rx_base) if listener_preamble
            and listener_rx_base is not None else self._rx_at)
        self._phase2_prebuilt_at = None
        self.events.append(("retrain-start", {"initiating": initiating, "reason": reason,
                                              "at": self._tx_media,
                                              "preamble_clock_origin": {
                                                  "tx": self._retrain_base,
                                                  "rx": self._retrain_rx_base,
                                                  "listener": listener_preamble,
                                              },
                                              "lapm": lapm_snapshot,
                                              "phase3": phase3_snapshot}))

    def _prebuild_phase2(self) -> None:
        """Start the Phase 2 receive side as soon as the peer's tone is gone.

        The peer's INFO0 is sent once. If it begins a little before our own Phase 2 starts, a
        receive side built only at our start would never see it, so build the objects now,
        scheduled for the packet at which the transmit side will switch (packets are 160
        samples), and keep sending the handshake's silence until then."""
        pre = self._retrain
        scheduled_at = (self._retrain_base
                        + -(-pre.phase2_start_at // 160) * 160)
        # RX is processed after the current TX packet in the RTP loop. Build
        # at the next actual TX cursor if confirmation arrived late, and let
        # the controller/renderer record whether the normative 40 ms response
        # window was missed instead of constructing a renderer in the past.
        at = max(scheduled_at, self._tx_media)
        continuation_info0a = None
        reversal_media_at = None
        if (pre.role == "digital" and pre.peer_ended_at is not None
                and self._retrain_info0a is not None):
            continuation_info0a = dict(self._retrain_info0a)
            reversal_media_at = int(self._clock.rx_to_media(
                self._retrain_rx_base + pre.peer_ended_at))
        self._build_phase2(at, retrain_info0a=continuation_info0a,
                           retrain_a_reversal_at=reversal_media_at)
        self._phase2_prebuilt_at = at
        self.events.append(("retrain-phase2-prebuilt", {
            "start_at": at, "tx_media": self._tx_media,
            "continuation": continuation_info0a is not None,
            "reversal_media_at": reversal_media_at,
            "preamble_peer_reversal_rx": pre.peer_ended_at,
            "preamble_peer_reversal_tx": pre.peer_ended_tx,
            "controller_state": self._phase2.state,
            "b_reversal_at": self._phase2.b_reversal_at,
        }))

    def _finish_retrain(self) -> None:
        self._retrain = None
        if self._phase2_prebuilt_at != self._tx_media:
            self.events.append(("retrain-phase2-rebuild", {
                "prebuilt_at": self._phase2_prebuilt_at,
                "tx_media": self._tx_media,
            }))
            # No early receive side, or the packet grid did not line up: build it now.
            self._build_phase2(self._tx_media)
        self._phase2_prebuilt_at = None
        self.events.append(("retrain-phase2-restart", {"at": self._tx_media}))

    @property
    def connected(self) -> bool:
        if self._v34_fallback_engine is not None:
            return bool(self._v34_fallback_engine.connected)
        phase4 = None if self._phase3 is None else getattr(self._phase3, "phase4", None)
        if phase4 is None:
            return False
        ready = phase4.state == phase4.transmitter.DATA_READY
        if not ready:
            return False
        return not self._phase3_enable_lapm or self.v42_resolved

    @property
    def bit_rate(self) -> int:
        if self._v34_fallback_engine is not None:
            return int(self._v34_fallback_engine.bit_rate)
        phase4 = None if self._phase3 is None else getattr(self._phase3, "phase4", None)
        profile = (None if phase4 is None
                   else phase4.transmitter.cp_profile)
        return 0 if not profile else int(round(float(profile["rate"])))

    @property
    def v42_mode(self) -> str:
        if self._v34_fallback_engine is not None:
            return self._v34_fallback_engine.v42_mode
        link = getattr(self._phase3, "lapm", None)
        return "off" if link is None else link.v42_mode

    @property
    def v42_ever_established(self) -> bool:
        """Preserve native/fallback LAPM history after a later link failure."""
        if self._v34_fallback_engine is not None:
            return bool(getattr(self._v34_fallback_engine,
                                "_v42_ever_established", False))
        lapm = getattr(self._phase3, "lapm", None)
        return bool(getattr(lapm, "_v42_ever_established", False))

    @property
    def v42_lost(self) -> bool:
        if self._v34_fallback_engine is not None:
            return bool(self._v34_fallback_engine.v42_lost)
        link = getattr(self._phase3, "lapm", None)
        lost = False if link is None else link.v42_lost
        if lost and self._can_retrain():
            # Report the link as still alive: the host drops the call the moment this is True,
            # and the transmit path will start the retrain on its next tick.
            self._retrain_wanted = True
            return False
        return lost

    def _can_retrain(self) -> bool:
        return (self._retrain_on_failure and not self._closed
                and (self._retrain is not None or self.retrain_count < self._max_retrains))

    def _data_link_failed(self) -> bool:
        phase3 = self._phase3
        if phase3 is None:
            return False
        link = getattr(phase3, "lapm", None)
        return bool(self._failed or getattr(phase3, "state", None) == "failed"
                    or (link is not None and link.v42_lost))

    @property
    def v42_compression(self):
        if self._v34_fallback_engine is not None:
            return self._v34_fallback_engine.v42_compression
        link = getattr(self._phase3, "lapm", None)
        return None if link is None else link.v42_compression

    @property
    def v42_recovery_stats(self):
        """Expose the active Phase 3 LAPM recovery counters to DCE diagnostics."""
        link = getattr(self._phase3, "lapm", None)
        stats = getattr(link, "_v42_recovery_stats", None)
        return None if stats is None else dict(stats)

    @property
    def v42_resolved(self) -> bool:
        if self._v34_fallback_engine is not None:
            return bool(self._v34_fallback_engine.v42_resolved)
        link = getattr(self._phase3, "lapm", None)
        return not self._phase3_enable_lapm or (
            link is not None and link.v42_resolved)

    @property
    def pending_bits(self) -> int:
        with self._lock:
            if self._v34_fallback_engine is not None:
                return int(self._v34_fallback_engine.pending_bits())
            if self._phase3 is not None and self._phase3_enable_lapm:
                link = getattr(self._phase3, "lapm", None)
                if link is not None:
                    return link.pending_bits()
                return int(getattr(self._phase3, "pending_bits", 0))
            return 8 * len(self._queued_dte)

    @property
    def rx_signal_power_dbm0(self) -> float:
        return -90.0

    @property
    def phase2_state(self) -> str:
        return self._phase2.state

    @property
    def phase3_state(self) -> str | None:
        if self._v34_fallback_engine is not None:
            return str(self._v34_fallback_engine.caller.stage)
        if self._phase3 is None:
            return None
        return self._phase3.state

    @property
    def phase3_handoff(self) -> str | None:
        for item in reversed(self.events):
            if isinstance(item, tuple) and item[0] == "phase3-handoff":
                return str(item[1])
        return None

    def is_settled(self) -> bool:
        return self.connected

    def tx_g711(self, count: int, codec: str) -> bytes:
        if codec != self.law:
            raise ValueError("RTP codec changed after V.8 PCM negotiation")
        if not isinstance(count, int) or count <= 0:
            raise ValueError("G.711 transmit count must be positive")
        if self._split_fallback_tx_lock:
            used, output = self._tx_v34_fallback_without_engine_lock(count)
            if used:
                return output
        with self._lock:
            start = self._tx_media
            output = self._tx_g711_locked(count)
            self._tx_media += count
            return output

    def _publish_fallback_tx(self, fallback) -> None:
        with self._fallback_tx_condition:
            self._fallback_tx_engine = fallback
            self._fallback_tx_condition.notify_all()

    def _retire_fallback_tx(self) -> None:
        """Stop new fast TX leases and drain active renders before lifecycle changes."""
        with self._fallback_tx_condition:
            self._fallback_tx_engine = None
            while self._fallback_tx_inflight:
                self._fallback_tx_condition.wait()

    def _tx_v34_fallback_without_engine_lock(self, count: int) -> tuple[bool, bytes]:
        with self._fallback_tx_condition:
            fallback = self._fallback_tx_engine
            if fallback is None:
                return False, b""
            self._fallback_tx_inflight += 1
        try:
            tx_g711 = getattr(fallback, "tx_g711", None)
            if callable(tx_g711):
                output = tx_g711(count, self.law)
            else:
                samples = np.asarray(fallback.tx_samples(count), dtype=float)
                clipped = np.clip(np.rint(samples), -32768, 32767).astype(np.int32)
                output = bytes(self._encoder(int(sample)) for sample in clipped)
            self._tx_media += count
            return True, output
        finally:
            with self._fallback_tx_condition:
                self._fallback_tx_inflight -= 1
                if not self._fallback_tx_inflight:
                    self._fallback_tx_condition.notify_all()

    def _tx_g711_locked(self, count: int) -> bytes:
        if self._closed:
            return bytes((_G711_SILENCE[self.law],)) * count
        # While the V.90 Phase-2 preamble/continuation is active, it owns the
        # wire even when a connected V.34 fallback object is retained to keep
        # its LAPM state alive.
        if self._v34_fallback_engine is not None and self._retrain is None \
                and self._phase2.done:
            tx_g711 = getattr(self._v34_fallback_engine, "tx_g711", None)
            if callable(tx_g711):
                return tx_g711(count, self.law)
            samples = np.asarray(
                self._v34_fallback_engine.tx_samples(count), dtype=float)
            clipped = np.clip(np.rint(samples), -32768, 32767).astype(np.int32)
            return bytes(self._encoder(int(sample)) for sample in clipped)
        if (self._retrain_on_failure and self._retrain is None and self._phase3 is not None
                and self.retrain_count < self._max_retrains
                and (self._retrain_wanted or self._data_link_failed())):
            phase3_reason = getattr(self._phase3, "failure_reason", None)
            reason = phase3_reason or "data link failed"
            self._retrain_wanted = False
            self._begin_retrain(initiating=True, reason=reason)
        if self._failed:
            return bytes((_G711_SILENCE[self.law],)) * count
        if self._retrain is not None:
            if (not self._retrain.ready
                    and self._retrain.tx_clock >= RETRAIN_TIMEOUT_SAMPLES):
                # The peer never answered the retrain tones: it does not support retrain, or the
                # line is gone. Give up so the host can drop the call.
                self.events.append(("retrain-timeout", {"after": self._retrain.tx_clock}))
                self._retrain = None
                self._failed = True
                return bytes((_G711_SILENCE[self.law],)) * count
            if self._retrain.phase2_due:
                self._finish_retrain()
            else:
                pcm = self._retrain.render(count)
                clipped = np.clip(np.rint(pcm), -32768, 32767).astype(np.int32)
                return bytes(self._encoder(int(sample)) for sample in clipped)
        self._advance_phase2_timer()
        if self._phase3 is not None and self._phase3.transmit_ready:
            try:
                if (self._phase3_enable_lapm
                        and getattr(self._phase3, "lapm", None) is not None):
                    payload = self._phase3.read_g711(
                        count, self.law,
                        data_bit_source=self._phase3.lapm.read_tx_bits)
                else:
                    payload = self._phase3.read_g711(count, self.law)
                if not isinstance(payload, bytes) or len(payload) > count:
                    raise RuntimeError("Phase 3 returned an invalid G.711 payload")
            except (TypeError, ValueError, RuntimeError) as exc:
                self._failed = True
                self.events.append(("phase3-failed", str(exc)))
                return bytes((_G711_SILENCE[self.law],)) * count
            if len(payload) < count:
                payload += bytes((_G711_SILENCE[self.law],)) * (count - len(payload))
            return payload
        pcm = self._renderer.render(count)
        clipped = np.clip(np.rint(pcm), -32768, 32767).astype(np.int32)
        return bytes(self._encoder(int(sample)) for sample in clipped)

    def tx_samples(self, count: int) -> list[int]:
        return [self._decoder(value) for value in self.tx_g711(count, self.law)]

    def feed_g711(self, payload: bytes, codec: str, context: dict) -> bytes:
        if codec != self.law:
            raise ValueError("RTP codec changed after V.8 PCM negotiation")
        data = bytes(payload)
        if not data:
            return b""
        with self._lock:
            if self._closed or self._failed:
                return b""
            try:
                self._clock.observe(self._rx_at, context)
                samples = np.fromiter((self._decoder(sample) for sample in data),
                                      dtype=float, count=len(data))
                if (self._v34_fallback_engine is not None
                        and self._retrain is None and self._phase2.done):
                    feed_g711 = getattr(self._v34_fallback_engine, "feed_g711", None)
                    if callable(feed_g711):
                        result = feed_g711(data, self.law, context)
                        self._rx_at += len(data)
                        return result
                    set_deadline = getattr(
                        self._v34_fallback_engine, "set_rx_media_deadline", None)
                    deadline = context.get("packet_playout_deadline_monotonic")
                    if callable(set_deadline) and deadline is not None:
                        set_deadline(float(deadline))
                    pcm = np.clip(np.rint(samples), -32768, 32767).astype("<i2").tobytes()
                    result = self._v34_fallback_engine.feed_pcm(pcm)
                    self._rx_at += len(data)
                    return result
                if self._retrain is not None:
                    self._retrain.feed(samples)
                    if (self._phase2_prebuilt_at is None and self._retrain.receive_ready
                            and self._retrain.phase2_start_at is not None):
                        self._prebuild_phase2()
                    if self._phase2_prebuilt_at is None:
                        self._rx_at += len(data)
                        return b""
                    # Phase 2 is built and listening; carry on into its receive processing.
                if self._phase3 is not None and self._phase2.done:
                    # Do not acquire from Phase-2 tones/probes: start only
                    # after the local digital Phase-3 transmitter is active.
                    # A sustained peer Tone A while in data mode is the peer's retrain request
                    # (V.90 §9.5.1.2): answer it instead of carrying on with a dead link.
                    if self._retrain_listener is None:
                        self._retrain_listener = RetrainPreamble(
                            "digital", initiating=False,
                            peak=self._phase2_build["tone_b_peak"])
                        self._retrain_listener_tx_base = self._tx_media
                        self._retrain_listener_rx_base = self._rx_at
                    self._retrain_listener.feed(samples)
                    if self._retrain_listener.peer_heard_at is not None:
                        self._begin_retrain(initiating=False, reason="peer tone",
                                            preamble=self._retrain_listener)
                        self._rx_at += len(data)
                        return b""
                    self._phase3.feed_pcm(samples, sample_at=self._rx_at)
                    self._rx_at += len(data)
                    take = getattr(self._phase3, "read_received_data", None)
                    return take() if callable(take) else b""
                self._remember_phase2_pcm(samples)
                now = max(self._clock.tx_now(), self._renderer.cursor)
                actions = self._adapter.feed(samples, rx_at=self._rx_at, tx_now=now)
                self._apply_measured_info1d_offset(actions)
                self._submit_phase2_actions(actions)
                self._advance_phase2_timer(now)
                self._rx_at += len(data)
                while self._reported_deadlines < len(self._adapter.deadline_misses):
                    miss = self._adapter.deadline_misses[self._reported_deadlines]
                    self.events.append(("phase2-deadline-miss", miss))
                    self._reported_deadlines += 1
                if (self._phase2.done
                        and self._reported_done_state != self._phase2.state):
                    self.events.append(("phase2-state", self._phase2.state))
                    self._reported_done_state = self._phase2.state
                self._record_phase2_failure()
                while self._reported_handoffs < len(self._renderer.handoffs):
                    handoff = self._renderer.handoffs[self._reported_handoffs]
                    self.events.append(("phase3-handoff", handoff.signal))
                    self._reported_handoffs += 1
                    if handoff.signal == "handoff-v90-phase3":
                        self._start_phase3(handoff)
                    elif handoff.signal == "handoff-v34-call-phase3":
                        self._start_v34_fallback()
                return b""  # Phase 2 has no DTE payload.
            except (TypeError, ValueError, RuntimeError, KeyError, IndexError) as exc:
                self._failed = True
                if self._phase3 is not None:
                    self.events.append(("phase3-failed", str(exc)))
                else:
                    self._phase2.state = self._phase2.FAILED
                    self.events.append(("phase2-failed", str(exc)))
                    self._reported_phase2_failure = True
                return b""

    def _apply_measured_info1d_offset(self, actions) -> None:
        if not any(action.signal == "info1d" for action in actions):
            return
        info0a = self._phase2.info0a
        if info0a is None:
            raise ValueError("cannot send V.90 INFO1d before receiving INFO0a")
        quality = self._adapter.signals.probe_quality
        # The configured per-rate profile is only a local candidate. V.90
        # Table 9 requires each advertised upstream profile to be consistent
        # with the analogue modem's capabilities from INFO0a. Apply that
        # constraint at call time, after the peer's CRC-checked frame exists.
        candidate = constrain_info1d_to_info0a(
            self._signals.info1d_fields, info0a)
        constrained = (project_info1d_rates_from_probe(candidate, quality)
                       if self._phase2_project_info1d_rates_from_probe
                       else candidate)
        self._signals.info1d_fields = constrained
        self._phase3_info1d = dict(constrained)
        self.events.append(("phase2-info1d-projection", {
            "enabled": self._phase2_project_info1d_rates_from_probe,
            "probe_snr_db": None if quality is None else quality.get("snr_db"),
            "rates": [int(block.get("max_rate", 0))
                      for block in constrained.get("blocks", ())],
        }))
        self.events.append(("phase2-probe-quality",
                            None if quality is None else dict(quality)))
        self.events.append(("phase2-info1d-rate-blocks",
                            [dict(block) for block in constrained["blocks"]]))
        measured_offset = self._adapter.signals.probe_frequency_offset_units
        if measured_offset is not None:
            # INFO1d Table 9 carries the digital receiver's own L2
            # measurement. Prefer the measured value over a static profile
            # field; -512 explicitly means the measurement was not reliable.
            self._signals.info1d_fields["freq_offset"] = int(measured_offset)

    def _submit_phase2_actions(self, actions) -> None:
        """Render actions and arm the INFO1a window from actual INFO1d end."""
        if not actions:
            return
        self._renderer.submit(actions)
        if (self._phase2.info1a_timeout_retrain
                and any(action.signal == "info1d" for action in actions)):
            complete_at = self._renderer._completion.get("info1d-complete")
            if complete_at is None:
                raise RuntimeError("INFO1d renderer did not report completion")
            self._phase2.info1d_transmitted(complete_at)

    def _advance_phase2_timer(self, now: int | None = None) -> None:
        if self._phase3 is not None or self._phase2.done:
            return
        media_now = max(self._clock.tx_now() if now is None else int(now),
                        self._renderer.cursor)
        actions = self._phase2.tick(media_now)
        if actions:
            self._submit_phase2_actions(actions)
        if (self._phase2.done
                and self._reported_done_state != self._phase2.state):
            self.events.append(("phase2-state", self._phase2.state))
            self._reported_done_state = self._phase2.state
        self._record_phase2_failure()

    def _record_phase2_failure(self) -> None:
        if (self._phase2.state == self._phase2.FAILED
                and not self._reported_phase2_failure):
            self.events.append(("phase2-failed", {
                "reason": "V.90 digital Phase 2 failed or timed out",
            }))
            self._reported_phase2_failure = True
            self._failed = True

    def _start_phase3(self, handoff) -> None:
        """Create the V.90 Phase 3 receive/transmit session from validated INFO1a."""
        if self._phase3_factory is None and self._phase3_rate_mask is not None:
            self._phase3_factory = self._default_phase3_factory
        if self._phase3_factory is None:
            self.events.append(("phase3-unavailable", "no Phase 3 factory configured"))
            return
        info1a = dict(self._phase2.info1a or {})
        required = ("v90_uinfo", "v90_analog_symbol_rate", "v90_digital_symbol_rate")
        if any(key not in info1a for key in required):
            raise ValueError("V.90 handoff lacks decoded V.90 INFO1a fields")
        if int(info1a["v90_digital_symbol_rate"]) != 6:
            raise ValueError("V.90 handoff has an invalid digital symbol-rate selector")
        lapm_detect = self._phase3_enable_lapm
        session = self._phase3_factory(
            info1a=info1a, law=self.law, handoff_at=int(handoff.at),
            media_origin=self._clock.origin, media_clock=self._clock,
            lapm_detect=lapm_detect)
        required = ("feed_pcm", "read_g711")
        if (session is None
                or any(not callable(getattr(session, name, None)) for name in required)
                or not isinstance(getattr(session, "transmit_ready", None), bool)
                or not isinstance(getattr(session, "state", None), str)):
            raise TypeError(
                "phase3_factory must return a session with feed_pcm, read_g711, "
                "boolean transmit_ready, and string state")
        self._phase3 = session
        self._phase3_started_at = self._tx_media
        self._replay_phase3_pcm_from_info1a()
        if self._queued_dte:
            feed_bytes = getattr(session, "feed_bytes", None)
            if callable(feed_bytes):
                feed_bytes(bytes(self._queued_dte))
            self._queued_dte.clear()
        self.events.append(("phase3-session-created", {
            "uinfo": int(info1a["v90_uinfo"]),
            "analog_symbol_rate": int(info1a["v90_analog_symbol_rate"]),
        }))

    def _start_v34_fallback(self) -> None:
        """Switch the digital endpoint into the V.34 call-side engine."""
        profile = {} if os.environ.get("V90_FALLBACK_STARTUP_PROFILE") == "1" else None
        profile_started = ((time.perf_counter(), time.thread_time())
                           if profile is not None else None)

        def begin_profile():
            return ((time.perf_counter(), time.thread_time())
                    if profile is not None else None)

        def mark_profile(name: str, started) -> None:
            if profile is not None and started is not None:
                profile[name] = {
                    "wall_ms": round((time.perf_counter() - started[0]) * 1000.0, 3),
                    "thread_cpu_ms": round((time.thread_time() - started[1]) * 1000.0, 3),
                }

        info1a = dict(self._phase2.info1a or {})
        if any(key not in info1a for key in
               ("sym_call", "sym_ans", "max_rate", "hi_carrier")):
            raise ValueError("V.34 fallback handoff lacks decoded INFO1a fields")
        rate = max(2400, int(info1a["max_rate"]) * 2400)
        fallback = self._v34_fallback_engine
        if fallback is not None:
            raise RuntimeError("V.90-to-V.34 fallback restart is outside this profile")
        started = begin_profile()
        from v34_modem import V34Modem
        mark_profile("import_v34_modem", started)
        started = begin_profile()
        fallback = V34Modem(fs=FS, data_rate=rate, min_data_rate=2400,
                            with_v8=False)
        mark_profile("construct_v34_modem", started)
        started = begin_profile()
        fallback.begin_v90_v34_fallback(info1a)
        mark_profile("begin_v34_fallback", started)
        self._v34_fallback_engine = fallback
        started = begin_profile()
        capture = None
        capture_callback = getattr(
            self, '_dce_v34_fallback_capture_callback', None)
        if callable(capture_callback):
            capture = capture_callback(fallback, kind='v90-v34-fallback')
        self._dce_v34_fallback_dsp_capture = capture
        mark_profile("attach_dsp_capture", started)
        started = begin_profile()
        self._replay_v34_fallback_pcm_from_info1a()
        mark_profile("replay_preroll", started)
        if self._queued_dte:
            started = begin_profile()
            fallback.feed_bytes(bytes(self._queued_dte))
            self._queued_dte.clear()
            mark_profile("feed_queued_dte", started)
        started = begin_profile()
        self._publish_fallback_tx(fallback)
        mark_profile("publish_fallback_tx", started)
        event = {
            "sym_call": int(info1a["sym_call"]),
            "sym_ans": int(info1a["sym_ans"]),
            "data_rate_ceiling": rate,
        }
        if profile is not None:
            profile["total"] = {
                "wall_ms": round((time.perf_counter() - profile_started[0]) * 1000.0, 3),
                "thread_cpu_ms": round((time.thread_time() - profile_started[1]) * 1000.0, 3),
            }
            event["startup_profile"] = profile
        self.events.append(("phase3-v34-fallback-started", event))
        _write_test_phase3_marker("answerer", self._tx_media)

    def _replay_v34_fallback_pcm_from_info1a(self) -> None:
        """Backfill the V.34 call-side receiver from the validated INFO1a edge."""
        wall_started = time.perf_counter()
        cpu_started = time.thread_time()
        fallback = self._v34_fallback_engine
        start = self._adapter.info1a_complete_rx_sample
        if fallback is None or start is None or not self._phase2_pcm_history:
            return
        first_start = self._phase2_pcm_history[0][0]
        last_start, last_chunk = self._phase2_pcm_history[-1]
        current_end = last_start + len(last_chunk)
        if start < first_start or start > current_end:
            self.events.append(("phase3-v34-fallback-preroll-unavailable", {
                "info1a_complete_rx_sample": int(start),
                "history_start": int(first_start),
                "history_end": int(current_end),
            }))
            return

        cursor = int(start)
        set_deadline = getattr(fallback, "set_rx_media_deadline", None)
        batch_preroll = os.environ.get("V90_FALLBACK_PREROLL_BATCH") == "1"
        replay_chunks = []
        for chunk_start, chunk in self._phase2_pcm_history:
            chunk_end = chunk_start + len(chunk)
            if chunk_end <= cursor:
                continue
            if chunk_start > cursor:
                raise RuntimeError("V.34 fallback pre-roll history has a receive gap")
            replay = chunk[cursor - chunk_start:]
            if len(replay):
                if batch_preroll:
                    replay_chunks.append(replay)
                    cursor += len(replay)
                    continue
                if callable(set_deadline):
                    set_deadline(self._clock.origin + cursor / FS)
                pcm = np.clip(np.rint(replay), -32768, 32767).astype("<i2")
                capture = getattr(self, '_dce_v34_fallback_dsp_capture', None)
                if capture is None:
                    fallback.feed_pcm(pcm.tobytes())
                else:
                    # INFO1a pre-roll may contain nearly all of Phase 3. Keep
                    # the diagnostic's thread-local hooks active while this
                    # retained audio is replayed, or B1 appears in the engine
                    # report but its waveform/tap evidence is silently lost.
                    capture.feed(pcm.tobytes(), fallback)
                cursor += len(replay)
        if cursor != current_end:
            raise RuntimeError("V.34 fallback pre-roll did not reach the current RX cursor")
        if batch_preroll and replay_chunks:
            # One large feed lets V34_P3_RX_SYMBOL_BUDGET bound total work for
            # this callback. Replaying each 20 ms history chunk separately
            # defeats that budget because every call processes up to one
            # packet's symbols before returning.
            replay = np.concatenate(replay_chunks)
            if callable(set_deadline):
                set_deadline(self._clock.origin + start / FS)
            pcm = np.clip(np.rint(replay), -32768, 32767).astype("<i2")
            capture = getattr(self, '_dce_v34_fallback_dsp_capture', None)
            if capture is None:
                fallback.feed_pcm(pcm.tobytes())
            else:
                capture.feed(pcm.tobytes(), fallback)
        self.events.append(("phase3-v34-fallback-preroll-replayed", {
            "start_rx_sample": int(start),
            "end_rx_sample": int(cursor),
            "samples": int(cursor - start),
            "chunks": len(replay_chunks) if batch_preroll else None,
            "batched": batch_preroll,
            "wall_ms": round((time.perf_counter() - wall_started) * 1000.0, 3),
            "thread_cpu_ms": round((time.thread_time() - cpu_started) * 1000.0, 3),
        }))

    def _remember_phase2_pcm(self, samples: np.ndarray) -> None:
        """Retain bounded decoded PCM while waiting for the INFO1a handoff."""
        if self._phase3 is not None or not len(samples):
            return
        start = int(self._rx_at)
        chunk = np.asarray(samples, dtype=float).copy()
        if len(chunk) > self._phase2_pcm_history_limit:
            drop = len(chunk) - self._phase2_pcm_history_limit
            start += drop
            chunk = chunk[drop:]
        self._phase2_pcm_history.append((start, chunk))
        self._phase2_pcm_history_samples += len(chunk)
        while self._phase2_pcm_history_samples > self._phase2_pcm_history_limit:
            old_start, old = self._phase2_pcm_history.popleft()
            excess = self._phase2_pcm_history_samples - self._phase2_pcm_history_limit
            if excess < len(old):
                self._phase2_pcm_history.appendleft(
                    (old_start + excess, old[excess:]))
                self._phase2_pcm_history_samples -= excess
            else:
                self._phase2_pcm_history_samples -= len(old)

    def _replay_phase3_pcm_from_info1a(self) -> None:
        """Backfill Phase 3 with buffered samples after validated INFO1a."""
        start = self._adapter.info1a_complete_rx_sample
        if start is None or not self._phase2_pcm_history:
            return
        first_start = self._phase2_pcm_history[0][0]
        last_start, last_chunk = self._phase2_pcm_history[-1]
        current_end = last_start + len(last_chunk)
        if start < first_start or start > current_end:
            self.events.append(("phase3-preroll-unavailable", {
                "info1a_complete_rx_sample": int(start),
                "history_start": int(first_start),
                "history_end": int(current_end),
            }))
            return
        cursor = int(start)
        for chunk_start, chunk in self._phase2_pcm_history:
            chunk_end = chunk_start + len(chunk)
            if chunk_end <= cursor:
                continue
            if chunk_start > cursor:
                raise RuntimeError("Phase 3 pre-roll history has a receive gap")
            left = cursor - chunk_start
            replay = chunk[left:]
            if len(replay):
                self._phase3.feed_pcm(replay, sample_at=cursor)
                cursor += len(replay)
        if cursor != current_end:
            raise RuntimeError("Phase 3 pre-roll did not reach the current RX cursor")
        self.events.append(("phase3-preroll-replayed", {
            "start_rx_sample": int(start),
            "end_rx_sample": int(cursor),
            "samples": int(cursor - start),
        }))

    def _default_phase3_factory(self, *, info1a: dict, law: str,
                                handoff_at: int, media_origin: float,
                                media_clock, lapm_detect: bool = True):
        """Build Phase 3 from INFO1a plus explicit local Jd rate capabilities."""
        from v34_tx import CARRIERS, SYMBOL_RATES
        from v90_digital_phase3_session import V90DigitalAnswererPhase3Session

        rate_index = int(info1a["v90_analog_symbol_rate"])
        if not 0 <= rate_index < len(SYMBOL_RATES):
            raise ValueError("V.90 INFO1a analog symbol-rate index is invalid")
        baud = SYMBOL_RATES[rate_index]
        blocks = self._phase3_info1d.get("blocks", [{}] * 6)
        if len(blocks) != 6:
            raise ValueError("V.90 INFO1d profile must contain six rate blocks")
        carrier_index = int(blocks[rate_index].get("hi_carrier", 0))
        carriers = CARRIERS.get(round(baud))
        if carriers is None or not 0 <= carrier_index < len(carriers):
            raise ValueError("V.90 INFO1d carrier selector is invalid")
        carrier = carriers[carrier_index]
        # The rate mask is an explicit product/configuration choice; INFO1a
        # alone does not justify advertising downstream rates to the caller.
        return V90DigitalAnswererPhase3Session(
            baud=baud, carrier=carrier, law=law,
            uinfo=int(info1a["v90_uinfo"]),
            # INFO1d is our own measurement of the analogue modem's L2
            # probing-tone offset. It centers our receive profile; INFO1a's
            # offset was measured by the remote analogue receiver and is not
            # a substitute for this local measurement.
            upstream_probe_frequency_offset_units=int(
                self._phase3_info1d.get("freq_offset", 0)),
            rate_mask=int(self._phase3_rate_mask),
            phase3_16point=self._phase3_16point,
            span=self._phase3_span,
            upstream_equalizer_mu=self._phase3_upstream_equalizer_mu,
            enable_lapm=self._phase3_enable_lapm,
            lapm_detect=lapm_detect,
            phase4_mp_profile=self._phase4_mp_profile,
            round_trip_delay_samples=int(
                getattr(self._phase2, "rtd_samples", 0) or 0))

    def feed_gap(self, samples: int, codec: str, context: dict) -> bytes:
        if codec != self.law:
            raise ValueError("RTP codec changed after V.8 PCM negotiation")
        if samples <= 0:
            return b""
        with self._lock:
            if "packet_playout_deadline_monotonic" in context:
                self._clock.observe(self._rx_at, context)
            if self._v34_fallback_engine is not None and not self._closed:
                self._v34_fallback_engine.note_concealed_rx(int(samples))
                gap_context = dict(context)
                gap_context["payload_samples"] = int(samples)
                sample = self._decoder(_G711_SILENCE[self.law])
                concealed = np.full(int(samples), sample, dtype="<i2")
                deadline = gap_context.get("packet_playout_deadline_monotonic")
                set_deadline = getattr(
                    self._v34_fallback_engine, "set_rx_media_deadline", None)
                if callable(set_deadline) and deadline is not None:
                    set_deadline(float(deadline))
                result = self._v34_fallback_engine.feed_pcm(concealed.tobytes())
                self._rx_at += int(samples)
                return result
            if self._retrain is not None and not self._closed:
                # The handshake is tones and silence: a lost stretch is read as silence and the
                # tone timing does the recovering. Failing here would end the call that the
                # retrain exists to save.
                self._rx_at += int(samples)
                self._retrain.feed(np.zeros(int(samples)))
                self.events.append(("retrain-rx-gap", int(samples)))
                return b""
            if self._phase3 is not None and self._phase2.done and not self._closed:
                # Keep the modem's receive sample clock aligned through an RTP
                # erasure. Feed decoded G.711 silence into the active Phase 3 or
                # upstream data receiver; a packet gap after handoff is not a Phase 2
                # failure. LAPM handles any damaged frames in the usual way.
                silence = self._decoder(_G711_SILENCE[self.law])
                missing = np.full(int(samples), silence, dtype=float)
                if self._retrain_listener is None:
                    self._retrain_listener = RetrainPreamble(
                        "digital", initiating=False,
                        peak=self._phase2_build["tone_b_peak"])
                    self._retrain_listener_tx_base = self._tx_media
                    self._retrain_listener_rx_base = self._rx_at
                self._retrain_listener.feed(missing)
                if self._retrain_listener.peer_heard_at is not None:
                    self._begin_retrain(initiating=False, reason="peer tone",
                                        preamble=self._retrain_listener)
                else:
                    self._phase3.feed_pcm(missing, sample_at=self._rx_at)
                self._rx_at += int(samples)
                self.events.append(("phase3-rx-gap", {
                    "samples": int(samples), "context": dict(context),
                }))
                take = getattr(self._phase3, "read_received_data", None)
                return take() if callable(take) else b""
            if not self._failed and not self._closed:
                self._rx_at += int(samples)
                self._failed = True
                self._phase2.state = self._phase2.FAILED
                self.events.append(("phase2-failed", {
                    "reason": "RTP sample gap during Phase 2",
                    "samples": int(samples), "context": dict(context),
                }))
                self._reported_phase2_failure = True
            return b""

    def feed_pcm(self, pcm: bytes) -> bytes:
        if len(pcm) % 2:
            raise ValueError("linear PCM input must contain complete 16-bit samples")
        samples = np.frombuffer(pcm, dtype="<i2")
        payload = bytes(self._encoder(int(sample)) for sample in samples)
        deadline = self._clock.origin + self._rx_at / FS
        return self.feed_g711(payload, self.law, {
            "packet_playout_deadline_monotonic": deadline,
            "payload_samples": len(payload),
        })

    def feed_bytes(self, data: bytes) -> None:
        payload = bytes(data)
        with self._lock:
            if self._closed:
                raise RuntimeError("V.90 Phase 2 engine is closed")
            if self._v34_fallback_engine is not None:
                self._v34_fallback_engine.feed_bytes(payload)
                return
            if len(self._queued_dte) + len(payload) > self._queue_limit:
                raise OverflowError("DTE data queued before V.90 Phase 4 exceeds 64 KiB")
            if (self._phase3 is not None and self._phase3_enable_lapm):
                self._phase3.feed_bytes(payload)
            else:
                self._queued_dte.extend(payload)

    def close(self) -> None:
        with self._lock:
            self._retire_fallback_tx()
            self._closed = True
            if self._v34_fallback_engine is not None:
                close = getattr(self._v34_fallback_engine, "close", None)
                if callable(close):
                    close()
