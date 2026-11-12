"""Event-driven V.90 analogue-caller Phase 2 procedure.

The controller consumes validated receive events and schedules signal actions
on one absolute 8 kHz sample timeline. A PCM detector and renderer own the
waveforms; this class owns protocol order and the V.90 timing relationships.
"""

from __future__ import annotations

from dataclasses import dataclass


FS = 8000
RETRAIN_TONE_B_WATCHDOG_MS = 5000


@dataclass(frozen=True)
class AnaloguePhase2Action:
    at: int | None
    signal: str
    value: int | None = None
    latest: int | None = None
    after: str | None = None
    tag: str | None = None
    cancel_tag: str | None = None


class V90AnalogueCallerPhase2:
    """Sequence the V.90 analogue-caller half of Phase 2 (§9.2.2).

    Input events are emitted only by external, validated INFO/tone/probe
    detectors. ``start`` is the end of Phase 1 silence. ``feed`` timestamps
    are line-sample positions; ``now`` is the local TX cursor used to expose
    detector-latency overruns rather than silently backdating actions.
    """

    WAIT_INFO0D = "wait-info0d"
    WAIT_TONE_B = "wait-tone-b"
    WAIT_TONE_B_REVERSAL = "wait-tone-b-reversal"
    WAIT_RECOVERY_TONE_B = "wait-recovery-tone-b"
    TX_LOCAL_PROBES = "transmit-local-probes"
    WAIT_TONE_B_AFTER_PROBES = "wait-tone-b-after-probes"
    WAIT_TONE_B_REVERSAL_AFTER_PROBES = "wait-tone-b-reversal-after-probes"
    RX_REMOTE_PROBES = "receive-remote-probes"
    WAIT_INFO1D = "wait-info1d"
    COMPLETE = "complete"
    FAILED = "failed"

    def __init__(self, *, l2_base_samples: int = 4400,
                 prefer_v90: bool = True, v34_mode: int = 5,
                 late_l2_echo_policy: str = "fail",
                 late_l2_echo_max_samples: int = 0,
                 accept_early_peer_recovery: bool = False,
                 info1d_timeout_retrain: bool = False):
        # V.90 §8.2.4 delegates the probing signals to V.34 §10.1.2.4.
        # The caller's L2 duration includes the measured round-trip delay;
        # the remote L2 receive window in §9.2.2.1.8 is separately capped at
        # 500 ms and must not be confused with this transmitted duration.
        if (isinstance(l2_base_samples, bool)
                or not 0 < l2_base_samples <= 550 * FS // 1000):
            raise ValueError("V.90 analogue-caller L2 base must be 1..550 ms")
        if isinstance(v34_mode, bool) or not 0 <= v34_mode <= 5:
            raise ValueError("V.34 fallback mode must be in 0..5")
        if late_l2_echo_policy not in (
                "fail", "bounded", "continue", "advance-after-window"):
            raise ValueError(
                "late L2 echo policy must be fail, bounded, continue or advance-after-window")
        if (isinstance(late_l2_echo_max_samples, bool)
                or not isinstance(late_l2_echo_max_samples, int)
                or late_l2_echo_max_samples < 0
                or late_l2_echo_max_samples > 10 * FS):
            raise ValueError("late L2 echo bound must be an integer in 0..10 seconds")
        if late_l2_echo_policy == "bounded" and late_l2_echo_max_samples < 550 * FS // 1000:
            raise ValueError("bounded late L2 echo policy requires at least 550 ms")
        self.l2_base_samples = int(l2_base_samples)
        self.l2_samples: int | None = None
        self.prefer_v90 = bool(prefer_v90)
        self.v34_mode = int(v34_mode)
        # `continue` is a diagnostic interoperability experiment only. It
        # lets us observe whether a real peer can recover after a delayed
        # echo; it does not change the standards-oriented default.
        self.late_l2_echo_policy = late_l2_echo_policy
        self.late_l2_echo_max_samples = int(late_l2_echo_max_samples)
        self.accept_early_peer_recovery = bool(accept_early_peer_recovery)
        if not isinstance(info1d_timeout_retrain, bool):
            raise ValueError("INFO1d timeout retrain policy must be boolean")
        self.info1d_timeout_retrain = info1d_timeout_retrain
        self._info1d_retrain_attempts = 0
        self.late_l2_echo_samples: int | None = None
        self.l2_echo_advanced_without_echo = False
        self.early_peer_recovery_accepted = False
        self.state = self.WAIT_INFO0D
        self.actions: list[AnaloguePhase2Action] = []
        self.info0d: dict | None = None
        self.info1d: dict | None = None
        self.mode: int | None = None
        self.rtde_samples: int | None = None
        self._started = False
        self._info0a_complete_at: int | None = None
        self._tone_a_reversal_at: int | None = None
        self._tone_b_after_l2 = False
        self._l2_echo_complete = False
        self._tone_a_second_started = False
        self._recovery_after_l2 = False
        self._remote_l1_complete = False
        self._remote_l2_complete = False
        self._remote_probe_end_at: int | None = None
        self._local_l2_start_at: int | None = None
        self._deadline_at: int | None = None
        self._deadline_kind: str | None = None
        self._info1d_deadline_at: int | None = None
        self._tone_b_seen = False
        self._pending_tone_b_at: int | None = None
        self._repeat_info0a_until_info0d = False
        self._recovery_tone_a_start: int | None = None
        self._tone_b_after_l2_at: int | None = None
        self._l2_echo_deadline_at: int | None = None

    @property
    def done(self) -> bool:
        return self.state in (self.COMPLETE, self.FAILED)

    def start(self, at: int) -> list[AnaloguePhase2Action]:
        """Send INFO0a after the V.8/Phase 1 silent interval."""
        if self._started or self.state != self.WAIT_INFO0D:
            raise RuntimeError("analogue Phase 2 already started")
        self._check_sample(at)
        self._started = True
        self.actions.append(AnaloguePhase2Action(at, "info0a", 0))
        self.actions.append(AnaloguePhase2Action(None, "tone_a",
                                                 after="info0a-complete"))
        return self.drain_actions()

    def start_v90_retrain_response(self, at: int, *, info0d: dict
                                   ) -> list[AnaloguePhase2Action]:
        """Resume V.90 Phase 2 after responding to a digital-initiated retrain.

        V.90 §9.5.2.2 ends by continuing at §9.2.2.1.3. Unlike V.92 §9.7,
        this path does not repeat INFO0a/INFO0d. The caller has already
        qualified the digital peer's Tone B for more than 50 ms; it now sends
        70 ms silence, Tone A for at least 50 ms, and the Tone-A reversal that
        begins the normal ranging/probe exchange.
        """
        if self._started or self.state != self.WAIT_INFO0D:
            raise RuntimeError("analogue Phase 2 controller already started")
        self._check_sample(at)
        if not isinstance(info0d, dict) or not info0d:
            raise ValueError("V.90 retrain continuation requires prior INFO0d")
        self._started = True
        self.info0d = dict(info0d)
        self.state = self.WAIT_TONE_B_REVERSAL
        tone_a_at = at + 70 * FS // 1000
        reversal_at = tone_a_at + 50 * FS // 1000
        # This is the start of the resumed Tone-A exchange. Later recovery
        # from a missed peer reversal uses it to enforce the Tone-A minimum.
        self._info0a_complete_at = tone_a_at
        self._tone_a_reversal_at = reversal_at
        self._set_deadline(reversal_at + 2000 * FS // 1000,
                           "tone_b_reversal")
        self.actions.extend((
            AnaloguePhase2Action(at, "silence"),
            AnaloguePhase2Action(tone_a_at, "tone_a"),
            AnaloguePhase2Action(reversal_at, "reverse-tone-a"),
            AnaloguePhase2Action(None, "listen-tone-b-reversal",
                                 after="reverse-tone-a"),
        ))
        return self.drain_actions()

    def start_v90_retrain_initiation(self, at: int, *, info0d: dict
                                     ) -> list[AnaloguePhase2Action]:
        """Initiate a V.90 retrain after a Phase-3/4 failure.

        V.90 §9.5.2.1 requires silence for 70 ± 5 ms, followed by Tone A
        while detecting Tone B. The existing Tone-B transition then enforces
        Tone A's 50 ms minimum before reversing it and resumes the Phase-2
        ranging exchange. The standard leaves the initial Tone-B wait open;
        this controller applies a separate 5 s product watchdog so a dead peer
        reaches the caller's bounded NO CARRIER path.
        """
        if self._started or self.state != self.WAIT_INFO0D:
            raise RuntimeError("analogue Phase 2 controller already started")
        self._check_sample(at)
        if not isinstance(info0d, dict) or not info0d:
            raise ValueError("V.90 retrain initiation requires prior INFO0d")
        self._started = True
        self._reset_probe_exchange_for_tone_b_restart()
        self.info0d = dict(info0d)
        self.state = self.WAIT_TONE_B
        tone_a_at = at + 70 * FS // 1000
        self._info0a_complete_at = tone_a_at
        self._set_deadline(
            tone_a_at + RETRAIN_TONE_B_WATCHDOG_MS * FS // 1000,
            "retrain_tone_b")
        self.actions.extend((
            AnaloguePhase2Action(at, "silence"),
            AnaloguePhase2Action(tone_a_at, "tone_a"),
            AnaloguePhase2Action(None, "listen-tone-b", after="tone_a"),
        ))
        return self.drain_actions()

    def feed(self, event: str, at: int, *, now: int | None = None,
             payload: dict | None = None) -> list[AnaloguePhase2Action]:
        self._check_sample(at)
        tx_now = at if now is None else now
        self._check_sample(tx_now)
        if not self._started:
            raise RuntimeError("start analogue Phase 2 before receive events")
        if self.done:
            return []
        data = payload or {}

        if event == "info0a_complete" and self.state in (
                self.WAIT_INFO0D, self.WAIT_TONE_B,
                self.WAIT_TONE_B_REVERSAL):
            self._info0a_complete_at = at
            if (self.state == self.WAIT_INFO0D
                    and self._repeat_info0a_until_info0d):
                self.actions.extend((
                    AnaloguePhase2Action(None, "info0a", 0,
                                         after="info0a-complete"),
                    AnaloguePhase2Action(None, "tone_a",
                                         after="info0a-complete"),
                    AnaloguePhase2Action(None, "listen-info0d-tone-b",
                                         after="info0a-complete"),
                ))
            if (self.state == self.WAIT_TONE_B
                    and self._pending_tone_b_at is not None):
                # INFO0d and Tone B can arrive while INFO0a is still being
                # sent. Hold that valid Tone B onset until the caller has
                # sent the required first 50 ms of Tone A.
                due = max(at + 50 * FS // 1000,
                          self._pending_tone_b_at, tx_now)
                self.actions.extend((
                    AnaloguePhase2Action(due, "reverse-tone-a"),
                    AnaloguePhase2Action(None, "listen-tone-b-reversal",
                                         after="reverse-tone-a"),
                ))
                self._tone_a_reversal_at = due
                self._pending_tone_b_at = None
                self.state = self.WAIT_TONE_B_REVERSAL
                self._set_deadline(due + 2000 * FS // 1000,
                                   "tone_b_reversal")
        elif event == "info0d" and self.state in (
                self.WAIT_INFO0D, self.WAIT_TONE_B,
                self.WAIT_TONE_B_REVERSAL):
            self.info0d = dict(data)
            repeated = self.state != self.WAIT_INFO0D
            self.state = self.WAIT_TONE_B
            self._repeat_info0a_until_info0d = False
            if repeated or self._tone_b_seen:
                # V.90 §9.2.2.2.1: recover from a premature Tone B or
                # repeated INFO0d by repeating INFO0a with bit 28 set once
                # INFO0d has been validated. Tag the complete retry sequence
                # so a later valid peer response can retire queued repeats.
                self.actions.extend((
                    AnaloguePhase2Action(None, "info0a", 1,
                                         after="info0a-complete",
                                         tag="info0a-retry"),
                    AnaloguePhase2Action(None, "tone_a",
                                         after="info0a-complete",
                                         tag="info0a-retry"),
                    AnaloguePhase2Action(None, "listen-tone-b",
                                         after="info0a-complete",
                                         tag="info0a-retry"),
                ))
                self._tone_b_seen = False
            else:
                self._clear_deadline()
        elif event == "tone_b" and self.state == self.WAIT_INFO0D:
            # Tone B may precede the first valid INFO0d. Remember it and
            # retransmit the initial INFO0a; the bit-28=1 retry is sent only
            # after INFO0d validates.
            self._tone_b_seen = True
            self._repeat_info0a_until_info0d = True
            self.actions.extend((
                AnaloguePhase2Action(None, "info0a", 0,
                                     after="info0a-complete"),
                AnaloguePhase2Action(None, "tone_a",
                                     after="info0a-complete"),
                AnaloguePhase2Action(None, "listen-info0d-tone-b",
                                     after="info0a-complete"),
            ))
        elif event == "tone_b" and self.state == self.WAIT_RECOVERY_TONE_B:
            if self._recovery_tone_a_start is None:
                raise RuntimeError("recovery Tone A start was not recorded")
            self._clear_deadline()
            # The detector timestamps Tone B at its estimated onset, but only
            # reports it after its confirmation windows have arrived. Never
            # backdate a reversal into already-rendered media: doing so can
            # collapse the reversal and following silence onto one TX sample.
            due = max(at, self._recovery_tone_a_start + 50 * FS // 1000,
                      tx_now)
            self.actions.extend((
                AnaloguePhase2Action(due, "reverse-tone-a"),
                AnaloguePhase2Action(due + 10 * FS // 1000, "silence"),
                AnaloguePhase2Action(None, "listen-tone-b-reversal",
                                     after="silence"),
            ))
            self._tone_a_reversal_at = due
            self.state = self.WAIT_TONE_B_REVERSAL
            self._set_deadline(due + 2000 * FS // 1000,
                               "tone_b_reversal")
        elif event == "tone_b" and self.state == self.WAIT_TONE_B:
            if self._deadline_kind == "retrain_tone_b":
                self._clear_deadline()
            if self._info0a_complete_at is None:
                if self._pending_tone_b_at is None:
                    self._pending_tone_b_at = at
                return self.drain_actions()
            earliest = self._info0a_complete_at + 50 * FS // 1000
            # Tone-B onset is itself a detected line-time. If receive and
            # transmit clocks are not perfectly aligned, tx_now can still be
            # behind that timestamp; never reverse Tone A before the peer's
            # Tone B has actually begun.
            due = max(earliest, at, tx_now)
            self.actions.append(AnaloguePhase2Action(due, "reverse-tone-a"))
            self._tone_a_reversal_at = due
            self.state = self.WAIT_TONE_B_REVERSAL
            self._set_deadline(due + 2000 * FS // 1000,
                               "tone_b_reversal")
        elif event == "tone_b_reversal" and self.state == self.WAIT_TONE_B_REVERSAL:
            # The first reversal measures the caller's initial Tone-A to
            # returned Tone-B round trip. A reversal after the post-L2
            # recovery sequence is a different exchange (§9.2.2.2.3): its
            # timing must not be interpreted using the initial 40 ms
            # turnaround or overwrite the RTDE used by the probes.
            if not self._recovery_after_l2:
                self._record_round_trip_delay(at)
            self._clear_deadline()
            if self._recovery_after_l2:
                # §9.2.2.2.3 returns to §9.2.2.1.3. We have already sent
                # the caller's L1/L2; after the recovered Tone B reversal,
                # the next signals are the answerer's L1/L2, which we receive.
                self._recovery_after_l2 = False
                self._remote_l1_complete = False
                self._remote_l2_complete = False
                self._remote_probe_end_at = None
                self.state = self.RX_REMOTE_PROBES
                self._arm_info1d_deadline(at)
                return self.drain_actions()
            self.l2_samples = self.l2_base_samples + self.rtde_samples
            due = at + 40 * FS // 1000
            self.actions.extend((
                AnaloguePhase2Action(due, "reverse-tone-a",
                                     latest=at + 41 * FS // 1000,
                                     cancel_tag="info0a-retry"),
                AnaloguePhase2Action(due + 10 * FS // 1000, "l1",
                                     cancel_tag="info0a-retry"),
                AnaloguePhase2Action(None, "l2", self.l2_samples,
                                     after="l1-complete"),
                AnaloguePhase2Action(None, "listen-tone-b",
                                     after="l2-complete"),
            ))
            self._local_l2_start_at = due + 10 * FS // 1000 + 160 * FS // 1000
            nominal_echo_wait = 550 * FS // 1000 + self.rtde_samples
            if self.late_l2_echo_policy == "bounded":
                echo_wait = max(nominal_echo_wait,
                                self.late_l2_echo_max_samples)
                self._l2_echo_deadline_at = self._local_l2_start_at + echo_wait
            elif self.late_l2_echo_policy == "advance-after-window":
                # If the hybrid echo is delayed beyond its standard window,
                # do not let it postpone Tone A long enough for the digital
                # peer to leave its response state. This is opt-in; the
                # normal policy continues to require the measured echo.
                self._l2_echo_deadline_at = (
                    self._local_l2_start_at + nominal_echo_wait)
            else:
                self._l2_echo_deadline_at = None
            self._set_deadline(self._local_l2_start_at + 600 * FS // 1000
                               + self.rtde_samples, "tone_b_after_l2")
            self.state = self.TX_LOCAL_PROBES
        elif event == "tone_b_after_probes" and self.state in (
                self.TX_LOCAL_PROBES, self.WAIT_TONE_B_AFTER_PROBES):
            if self._local_l2_start_at is None or at < self._local_l2_start_at:
                raise ValueError("Tone B arrived before local L2 began")
            self._clear_deadline()
            self._tone_b_after_l2_at = at
            self._arm_info1d_deadline(at)
            self._tone_b_after_l2 = True
            self.state = self.WAIT_TONE_B_AFTER_PROBES
            self._maybe_send_second_tone_a(tx_now)
        elif (event == "early_peer_recovery_tone_b_reversal"
              and self.state == self.WAIT_TONE_B_AFTER_PROBES
              and self.accept_early_peer_recovery):
            # A V.90 digital peer can take its §9.2.1.2.4 recovery branch
            # when our Tone-A reversal is late. It then sends a Tone-B
            # reversal followed by L1/L2. The captured peer captures contain that
            # ordered sequence before our delayed local-L2 echo completes.
            # The DCE profile enables this standards-defined peer-recovery
            # branch by default, while retaining an explicit disable override
            # for comparison. Its timing is validated by the remote L1/L2
            # receive windows below; it is distinct from accepting a late
            # local hybrid echo.
            self._clear_deadline()
            self._l2_echo_deadline_at = None
            self._remote_l1_complete = False
            self._remote_l2_complete = False
            self._remote_probe_end_at = None
            self.early_peer_recovery_accepted = True
            self.state = self.RX_REMOTE_PROBES
        elif event == "l2_echo_complete" and self.state in (
                self.TX_LOCAL_PROBES, self.WAIT_TONE_B_AFTER_PROBES):
            if self._local_l2_start_at is None:
                raise ValueError("local L2 echo arrived before local L2 was scheduled")
            if at < self._local_l2_start_at:
                raise ValueError("local L2 echo completed before local L2 began")
            echo_deadline = (self._local_l2_start_at + 550 * FS // 1000
                             + (self.rtde_samples or 0))
            if at > echo_deadline:
                self.late_l2_echo_samples = at - echo_deadline
                allowed_deadline = (self._l2_echo_deadline_at
                                    if self._l2_echo_deadline_at is not None
                                    else echo_deadline)
                if (self.late_l2_echo_policy == "fail"
                        or (self.late_l2_echo_policy == "bounded"
                            and at > allowed_deadline)):
                    self.state = self.FAILED
                    self._l2_echo_deadline_at = None
                    self.actions.append(AnaloguePhase2Action(
                        tx_now, "recovery-required"))
                    return self.drain_actions()
                if self.late_l2_echo_policy == "advance-after-window":
                    # The timer normally advances first; this covers a late
                    # echo callback arriving at the same packet boundary.
                    self.l2_echo_advanced_without_echo = True
                    self._l2_echo_deadline_at = None
                    self._maybe_send_second_tone_a(tx_now)
                    return self.drain_actions()
            self._l2_echo_complete = True
            self._l2_echo_deadline_at = None
            # The echo is an external observation; the recovery timer is
            # specifically for missing remote Tone B, so retain it until B.
            self.state = self.WAIT_TONE_B_AFTER_PROBES
            self._maybe_send_second_tone_a(tx_now)
        elif event == "tone_b_reversal_after_probes" and self.state == self.WAIT_TONE_B_REVERSAL_AFTER_PROBES:
            self._clear_deadline()
            self.state = self.RX_REMOTE_PROBES
            origin = (self._tone_b_after_l2_at
                      if self._tone_b_after_l2_at is not None else at)
            self._arm_info1d_deadline(origin)
        elif event == "l1_complete" and self.state == self.RX_REMOTE_PROBES:
            if self._remote_l1_complete:
                raise ValueError("remote L1 completion was already recorded")
            self._remote_l1_complete = True
            # The remote-probe receive window is specified against received
            # line samples.  `at` is normally mapped onto the local media/TX
            # clock for scheduling, whose fitted slope and integer rounding
            # can make an exact 500 ms receive interval appear one sample
            # longer.  The adapter supplies the original RX-clock timestamp;
            # direct controller users may omit it when both clocks coincide.
            self._remote_probe_end_at = self._remote_probe_sample_at(data, at)
            self._maybe_finish_remote_probes(tx_now)
        elif event == "l2_complete" and self.state == self.RX_REMOTE_PROBES:
            if not self._remote_l1_complete:
                raise ValueError("received L2 before the 160 ms L1")
            if self._remote_l2_complete:
                raise ValueError("remote L2 completion was already recorded")
            remote_at = self._remote_probe_sample_at(data, at)
            if (self._remote_probe_end_at is not None
                    and remote_at - self._remote_probe_end_at > 500 * FS // 1000):
                raise ValueError("remote L2 exceeded its 500 ms receive window")
            self._remote_l2_complete = True
            self._remote_probe_end_at = remote_at
            self._maybe_finish_remote_probes(tx_now)
        elif event == "l2_window_end" and self.state == self.RX_REMOTE_PROBES:
            if not self._remote_l1_complete:
                raise ValueError("remote L2 window ended before L1 completed")
            if self._remote_l2_complete:
                raise ValueError("cannot mark L2 absent after receiving L2")
            remote_at = self._remote_probe_sample_at(data, at)
            if (self._remote_probe_end_at is not None
                    and remote_at < self._remote_probe_end_at):
                raise ValueError("remote L2 window ended before L1 completed")
            if (self._remote_probe_end_at is not None
                    and remote_at - self._remote_probe_end_at > 500 * FS // 1000):
                raise ValueError("remote L2 receive window exceeded 500 ms")
            self._remote_l2_complete = True  # completed receive window, L2 absent
            self._remote_probe_end_at = remote_at
            self._maybe_finish_remote_probes(tx_now)
        elif event == "info1d" and self.state == self.WAIT_INFO1D:
            # The protocol deadline is absolute and is also pumped by tick().
            # A frame detector may report INFO1d on the same media packet in
            # which the timer becomes due; do not let callback ordering accept
            # a response that the timeout path would reject at this sample.
            if (self._info1d_deadline_at is not None
                    and at >= self._info1d_deadline_at):
                self._info1d_timeout(tx_now)
                return self.drain_actions()
            self._clear_deadline()
            self._info1d_deadline_at = None
            self._on_info1d(data, tx_now)
        elif event == "tone_b" and self.state == self.WAIT_INFO1D:
            # §9.2.2.2.4 returns to the Phase 2 Tone-B exchange if the
            # peer restarts Phase 2 while INFO1d is expected.
            self._clear_deadline()
            self._info1d_deadline_at = None
            self._reset_probe_exchange_for_tone_b_restart()
            # Tone A begins now; after its required 50 ms minimum, reverse
            # it and resume waiting for the peer's Tone B reversal.
            due = tx_now + 50 * FS // 1000
            self.actions.extend((AnaloguePhase2Action(tx_now, "tone_a"),
                                 AnaloguePhase2Action(due,
                                                      "reverse-tone-a"),
                                 AnaloguePhase2Action(None,
                                     "listen-tone-b-reversal",
                                     after="reverse-tone-a")))
            self._tone_a_reversal_at = due
            self.state = self.WAIT_TONE_B_REVERSAL
            self._set_deadline(due + 2000 * FS // 1000,
                               "tone_b_reversal")
        else:
            raise ValueError(f"event {event!r} is invalid in state {self.state!r}")
        return self.drain_actions()

    def _reset_probe_exchange_for_tone_b_restart(self) -> None:
        """Discard per-probe state before restarting at §9.2.2.1.3."""
        self.info1d = None
        self.mode = None
        self._tone_b_after_l2 = False
        self._l2_echo_complete = False
        self._tone_a_second_started = False
        self._recovery_after_l2 = False
        self._remote_l1_complete = False
        self._remote_l2_complete = False
        self._remote_probe_end_at = None
        self._local_l2_start_at = None
        self._tone_b_seen = False
        self._pending_tone_b_at = None
        self._repeat_info0a_until_info0d = False
        self._recovery_tone_a_start = None
        self._tone_b_after_l2_at = None
        self._l2_echo_deadline_at = None
        self.late_l2_echo_samples = None
        self.l2_echo_advanced_without_echo = False

    @staticmethod
    def _remote_probe_sample_at(payload: dict, fallback: int) -> int:
        """Return a probe timestamp in the receive-sample clock domain."""
        sample_at = payload.get("receive_sample_at", fallback)
        if (isinstance(sample_at, bool) or not isinstance(sample_at, int)
                or sample_at < 0):
            raise ValueError("receive_sample_at must be a nonnegative sample position")
        return sample_at

    def _maybe_send_second_tone_a(self, now: int) -> None:
        echo_condition_met = (self._l2_echo_complete
                              or self.l2_echo_advanced_without_echo)
        if self._tone_a_second_started or not (self._tone_b_after_l2
                                                 and echo_condition_met):
            return
        self._tone_a_second_started = True
        self._l2_echo_deadline_at = None
        # Tone B can be detected while the caller's scheduled L2 is still
        # being emitted: the answerer's response and the returned hybrid echo
        # are on the receive clock, while this transmitter is finishing its
        # own finite probe. Starting Tone A immediately lets that probe's
        # queued end-silence truncate the required 50 ms Tone A interval.
        # Keep the complete L2 on the wire, then begin Tone A and measure its
        # minimum duration from that actual protocol boundary.
        local_l2_end = (self._local_l2_start_at or now) + (self.l2_samples or 0)
        tone_a_at = max(now, local_l2_end)
        reverse_at = tone_a_at + 50 * FS // 1000
        self.actions.extend((
            AnaloguePhase2Action(tone_a_at, "tone_a"),
            AnaloguePhase2Action(reverse_at, "reverse-tone-a"),
            AnaloguePhase2Action(reverse_at + 10 * FS // 1000, "silence"),
            AnaloguePhase2Action(None, "listen-tone-b-reversal",
                                 after="silence"),
        ))
        self.state = self.WAIT_TONE_B_REVERSAL_AFTER_PROBES
        # A lost RTP packet can erase the peer's short phase reversal while
        # leaving the surrounding audio intact. Keep waiting for signal
        # progress, but guarantee a finite retrain handoff if no reversal is
        # detected. The 2 s + RTDE watchdog is a bounded product safety guard,
        # not a separate timeout stated in §9.2.2.
        self._set_deadline(reverse_at + 2000 * FS // 1000
                           + (self.rtde_samples or 0),
                           "tone_b_reversal_after_probes")

    def _maybe_finish_remote_probes(self, now: int) -> None:
        if self._remote_l1_complete and self._remote_l2_complete:
            self.actions.extend((AnaloguePhase2Action(now, "tone_a"),
                                 AnaloguePhase2Action(now, "listen-info1d")))
            self.state = self.WAIT_INFO1D
            # Preserve the absolute limit established at Tone B detection.
            # This is independent of the narrower reversal/echo timers.
            if self._info1d_deadline_at is None:
                origin = (self._tone_b_after_l2_at
                          if self._tone_b_after_l2_at is not None else now)
                self._arm_info1d_deadline(origin)

    def _arm_info1d_deadline(self, origin: int) -> None:
        """Bound the Phase-2 response from its protocol event, not detection completion."""
        self._info1d_deadline_at = (origin + 2000 * FS // 1000
                                    + 2 * (self.rtde_samples or 0))

    def _info1d_timeout(self, now: int) -> None:
        """Take the bounded §9.2.2.2.4 retrain option or fail safely.

        The retry is opt-in because it changes production call behavior and
        has not yet been qualified with an independent digital modem. A
        single retrain prevents a peer that never responds from looping
        forever; the next INFO1d timeout takes the normal NO CARRIER handoff.
        """
        self._info1d_deadline_at = None
        self._clear_deadline()
        self._l2_echo_deadline_at = None
        if (self.info1d_timeout_retrain
                and self._info1d_retrain_attempts < 1):
            self._info1d_retrain_attempts += 1
            self._reset_probe_exchange_for_tone_b_restart()
            self.state = self.WAIT_TONE_B
            # V.90 §9.5.2.1: silence 70 ± 5 ms, Tone A, then detect Tone B.
            # The existing WAIT_TONE_B transition enforces Tone A's 50 ms
            # minimum before reversing it and resumes at §9.2.2.1.3.
            tone_a_at = now + 70 * FS // 1000
            self._info0a_complete_at = tone_a_at
            self.actions.extend((
                AnaloguePhase2Action(now, "silence"),
                AnaloguePhase2Action(tone_a_at, "tone_a"),
                AnaloguePhase2Action(None, "listen-tone-b",
                                     after="tone_a"),
            ))
            return
        self.state = self.FAILED
        self.actions.append(AnaloguePhase2Action(
            now, "handoff-retrain-v90-phase2"))

    def tick(self, now: int) -> list[AnaloguePhase2Action]:
        """Advance protocol recovery timers on the 8 kHz line-sample clock."""
        self._check_sample(now)
        if not self._started:
            raise RuntimeError("start analogue Phase 2 before timer events")
        if self.done:
            return []
        if (self._info1d_deadline_at is not None
                and now >= self._info1d_deadline_at
                and self.state in (self.WAIT_TONE_B_AFTER_PROBES,
                                   self.WAIT_TONE_B_REVERSAL_AFTER_PROBES,
                                   self.RX_REMOTE_PROBES,
                                   self.WAIT_INFO1D)):
            # §9.2.2.2.4 is an absolute Phase-2 response limit. Keep it
            # independent of the state-local reversal and echo deadlines.
            self._info1d_timeout(now)
            return self.drain_actions()
        if (self._l2_echo_deadline_at is not None
                and now >= self._l2_echo_deadline_at
                and not self._l2_echo_complete):
            self._l2_echo_deadline_at = None
            if self.late_l2_echo_policy == "advance-after-window":
                self.l2_echo_advanced_without_echo = True
                self._maybe_send_second_tone_a(now)
                return self.drain_actions()
            self.state = self.FAILED
            self._clear_deadline()
            self.actions.append(AnaloguePhase2Action(now, "recovery-required"))
            return self.drain_actions()
        if self._deadline_at is None or now < self._deadline_at:
            return []
        kind = self._deadline_kind
        self._clear_deadline()
        if kind == "tone_b_reversal" and self.state == self.WAIT_TONE_B_REVERSAL:
            # §9.2.2.2.2: reacquire Tone B and re-enter .1.3.
            self.state = self.WAIT_RECOVERY_TONE_B
            # Tone A has remained on since INFO0a; its 50 ms minimum is
            # already satisfied while the 2 s reversal timer ran.
            self._recovery_tone_a_start = self._info0a_complete_at
            self.actions.extend((AnaloguePhase2Action(now, "listen-tone-b"),))
            # §9.2.2.2.2 requires Tone-B reacquisition but does not specify a
            # further deadline. Bound this wait locally to avoid an endless
            # receive state; the following timeout is a product safety cap.
            self._set_deadline(now + 2000 * FS // 1000
                               + (self.rtde_samples or 0),
                               "recovery_tone_b")
        elif kind == "tone_b_after_l2" and self.state in (
                self.TX_LOCAL_PROBES, self.WAIT_TONE_B_AFTER_PROBES):
            # §9.2.2.2.3: recover the B/Tone-A exchange rather than wait
            # indefinitely for an L2 echo or a tone event.
            self.state = self.WAIT_RECOVERY_TONE_B
            self._l2_echo_deadline_at = None
            self._recovery_tone_a_start = now
            self._recovery_after_l2 = True
            self.actions.extend((AnaloguePhase2Action(now, "tone_a"),
                                 AnaloguePhase2Action(now, "listen-tone-b")))
            # The Recommendation specifies the 600 ms + RTDE trigger and
            # return to §9.2.2.1.3, but no second timeout while reacquiring
            # Tone B. This 2 s + RTDE deadline is a bounded product safety
            # guard, not a normative V.90 timer.
            self._set_deadline(now + 2000 * FS // 1000
                               + (self.rtde_samples or 0),
                               "recovery_tone_b")
        elif (kind == "recovery_tone_b"
              and self.state == self.WAIT_RECOVERY_TONE_B):
            self.state = self.FAILED
            self.actions.append(AnaloguePhase2Action(
                now, "handoff-retrain-v90-phase2"))
        elif kind == "retrain_tone_b" and self.state == self.WAIT_TONE_B:
            self.state = self.FAILED
            self.actions.append(AnaloguePhase2Action(
                now, "handoff-retrain-v90-phase2"))
        elif (kind == "tone_b_reversal_after_probes"
              and self.state == self.WAIT_TONE_B_REVERSAL_AFTER_PROBES):
            self.state = self.FAILED
            self.actions.append(AnaloguePhase2Action(
                now, "handoff-retrain-v90-phase2"))
        return self.drain_actions()

    def _set_deadline(self, at: int, kind: str) -> None:
        self._deadline_at = at
        self._deadline_kind = kind

    def _clear_deadline(self) -> None:
        self._deadline_at = None
        self._deadline_kind = None

    def _record_round_trip_delay(self, at: int) -> None:
        if self._tone_a_reversal_at is None:
            raise ValueError("Tone B reversal arrived before local Tone A reversal")
        elapsed = at - self._tone_a_reversal_at
        propagation = elapsed - 40 * FS // 1000
        if propagation < 0:
            raise ValueError("Tone B reversal precedes the V.90 40 ms turnaround")
        self.rtde_samples = propagation

    def _on_info1d(self, payload: dict, now: int) -> None:
        # INFO1d is the digital modem's V.34-compatible upstream profile. It
        # does not contain a V.90-mode boolean. Whether V.90 was selected is
        # known from Phase 1/V.8; this setting is the local policy choice.
        self.mode = 6 if self.prefer_v90 else self.v34_mode
        self.info1d = dict(payload)
        self.actions.extend((AnaloguePhase2Action(now, "info1a", self.mode),
                             AnaloguePhase2Action(None,
                                 "handoff-v90-phase3" if self.mode == 6
                                 else "handoff-v34-answer-phase3", self.mode,
                                 after="info1a-complete")))
        self.state = self.COMPLETE

    def drain_actions(self) -> list[AnaloguePhase2Action]:
        actions, self.actions = self.actions, []
        return actions

    @staticmethod
    def _check_sample(sample: int) -> None:
        if isinstance(sample, bool) or not isinstance(sample, int) or sample < 0:
            raise ValueError("sample positions must be nonnegative integers")
