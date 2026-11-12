"""Event-driven V.90 digital-modem Phase 2 procedure.

This is the protocol controller between a future line-signal detector and the
INFO/tone/probe transmitters. Events must be detected by the media receiver;
this class does not infer them from PCM. Times use one caller-supplied sample
timeline (8 kHz), so a RTP adapter must map receive detections to that timeline
before forwarding them. Actions describe what the media layer must transmit.

The split is intentional: the paired Gough recordings taught us to establish
channel identity and parse CRC-valid protocol frames before assigning meaning
to spectral features. The controller therefore consumes typed, validated
events and never treats a guessed tone or INFO frame as protocol truth.
"""

from __future__ import annotations

from dataclasses import dataclass

FS = 8000


@dataclass(frozen=True)
class Phase2Action:
    at: int | None
    signal: str
    value: int | None = None
    latest: int | None = None
    after: str | None = None


class V90DigitalPhase2Controller:
    """Sequence the digital-side Phase 2 exchange from validated RX events.

    Event names: ``info0a``, ``tone_a_reversal``, ``l1_complete``,
    ``l2_complete``, ``tone_a``, ``tone_a_reversal_after_l2``,
    ``l2_echo_complete``, ``info1a``. ``start()`` is called at the end of
    V.8's 75 ms silence. Outputs are timestamped actions; the media adapter
    supplies actual waveforms and keeps Tone B continuous between actions.
    """

    WAIT_INFO0A = "wait-info0a"
    WAIT_A_REV2 = "wait-a-reversal-2"
    WAIT_RETRAIN_A_REV2 = "wait-retrain-a-reversal-2"
    RX_PROBE = "receive-first-probe"
    WAIT_A_AFTER_PROBE = "wait-a-after-probe"
    WAIT_A_REV3 = "wait-a-reversal-3"
    TX_PROBE = "send-second-probe"
    WAIT_INFO1A = "wait-info1a"
    WAIT_INFO1A_RETRAIN = "wait-info1a-or-retrain-tone-a"
    WAIT_RETRAIN_TONE_B = "wait-retrain-tone-b"
    WAIT_RETRAIN_A_REVERSAL = "wait-retrain-tone-a-reversal"
    COMPLETE = "complete"
    FAILED = "failed"

    def __init__(self, *, rtd_samples: int = 0, max_l2_samples: int = 4000,
                 info1a_timeout_retrain: bool = False,
                 l2_echo_tolerance_samples: int = 0):
        if rtd_samples < 0 or max_l2_samples < 0 or l2_echo_tolerance_samples < 0:
            raise ValueError("sample durations must be nonnegative")
        self.rtd_samples = int(rtd_samples)
        if not isinstance(info1a_timeout_retrain, bool):
            raise ValueError("INFO1a timeout retrain policy must be boolean")
        self.info1a_timeout_retrain = info1a_timeout_retrain
        # The 550/650 ms limits compare the L2 echo time with the RTD, and both come
        # from separately quantised detections of a stable delay, so they can differ
        # by a sample or more with no change in the line. 0 (the default) keeps the
        # limits exact; a profile may allow a few samples of measurement slack.
        self.l2_echo_tolerance_samples = int(l2_echo_tolerance_samples)
        self.max_l2_samples = min(int(max_l2_samples), int(0.500 * FS))
        self.state = self.WAIT_INFO0A
        self.actions: list[Phase2Action] = []
        self.info0a: dict | None = None
        self.info1a: dict | None = None
        self.b_reversal_at: int | None = None
        self.l2_started_at: int | None = None
        self._a_rev_count = 0
        self._first_probe_l1_done = False
        self._started = False
        self._tone_a_during_second_l2 = False
        self._second_l2_echo_complete_at: int | None = None
        self._tone_a_deadline_at: int | None = None
        self._second_initial_a_reversal_at: int | None = None
        self._missing_tone_a_reversal_deadline_at: int | None = None
        self._info1a_deadline_at: int | None = None
        self._retrain_tone_b_start_at: int | None = None
        self._pending_retrain_a_reversal: tuple[int, dict] | None = None
        self.missing_tone_a_recovery_started = False
        self.retrain_response_started = False

    @property
    def done(self) -> bool:
        return self.state in (self.COMPLETE, self.FAILED)

    def start(self, at: int) -> list[Phase2Action]:
        """Start after Phase 1's required 75 +/- 5 ms silence."""
        if self.state != self.WAIT_INFO0A or self._started:
            raise RuntimeError("Phase 2 already started")
        self._started = True
        self.actions.extend((Phase2Action(at, "info0d", 0),
                             Phase2Action(None, "tone_b", after="info0d-complete")))
        return self.drain_actions()

    def start_retrain_continuation(self, reversal_at: int, *, now: int,
                                   info0a: dict) -> list[Phase2Action]:
        """Resume V.90 at §9.2.1.1.3 after the retrain Tone A reversal.

        §9.5.1.1 directs the digital initiator to §9.2.1.1.3; INFO0 is
        retained across the retrain. The already-detected A reversal therefore
        starts the 40 ms A-to-B reversal response rather than a new INFO0
        exchange.
        """
        if self.state != self.WAIT_INFO0A or self._started:
            raise RuntimeError("Phase 2 already started")
        if not isinstance(info0a, dict) or not info0a:
            raise ValueError("V.90 retrain continuation requires retained INFO0a")
        self.info0a = dict(info0a)
        self._started = True
        return self.feed("tone_a_reversal", int(reversal_at), now=int(now))

    def feed(self, event: str, at: int, *, now: int | None = None,
             payload: dict | None = None) -> list[Phase2Action]:
        """Apply one validated receive event; ``at`` is its estimated line time.

        ``now`` is the transmit cursor. It prevents scheduling an action in the
        past when detector block latency exceeds the standard's response time.
        The adapter must log that lateness; silently moving protocol timestamps
        would hide a real-time failure.
        """
        if at < 0 or (now is not None and now < 0):
            raise ValueError("sample positions must be nonnegative")
        if not self._started:
            raise RuntimeError("start Phase 2 before delivering receive events")
        if self.done:
            return []
        tx_now = at if now is None else now
        if (self.info1a_timeout_retrain
                and self._info1a_deadline_at is not None
                and event == "info1a"
                and self.state == self.WAIT_INFO1A
                and at >= self._info1a_deadline_at):
            # A CRC-valid frame beginning at the limit belongs to the
            # post-deadline receive state. Keep accepting INFO1a there while
            # also listening for the standard Tone-A retrain request.
            self.state = self.WAIT_INFO1A_RETRAIN
        if (event == "tone_a_reversal_after_l2"
                and self._missing_tone_a_reversal_deadline_at is not None
                and at >= self._missing_tone_a_reversal_deadline_at
                and self.state in (self.WAIT_A_AFTER_PROBE,
                                   self.WAIT_A_REV3)):
            # Detection at/after the §9.2.1.2.4 limit belongs to the recovery
            # branch, not the ordinary second-reversal path.
            return self._begin_missing_tone_a_reversal_recovery(tx_now)
        if event == "info0a":
            self._on_info0a(at, payload or {})
        elif (event == "retrain_tone_a_reversal"
              and self.state == self.WAIT_RETRAIN_TONE_B):
            # The analogue responder reverses Tone A after 50 ms, while our
            # §9.5.1.2 response silence lasts 70 ms. Save that early reversal
            # until our Tone B is actually on the transmit timeline.
            if self._pending_retrain_a_reversal is None:
                self._pending_retrain_a_reversal = (at, dict(payload or {}))
        elif (event == "retrain_tone_a_reversal"
              and self.state == self.WAIT_RETRAIN_A_REVERSAL):
            # §9.5.1.2 response ends at §9.2.1.1.3. Seed the ordinary
            # first-reversal handler so its 40 ms Tone-B turnaround and
            # following second-reversal ranging are reused exactly.
            self._a_rev_count = 0
            self.state = self.WAIT_INFO0A
            self._on_a_reversal(at, tx_now)
            self.state = self.WAIT_RETRAIN_A_REV2
        elif event == "tone_a_reversal":
            self._on_a_reversal(at, tx_now)
        elif event == "l1_complete" and self.state == self.RX_PROBE:
            self._first_probe_l1_done = True
        elif event == "l2_complete" and self.state == self.RX_PROBE:
            if not self._first_probe_l1_done:
                raise ValueError("L2 cannot complete before the 160 ms L1")
            self.actions.append(Phase2Action(tx_now, "tone_b"))
            self.state = self.WAIT_A_AFTER_PROBE
        elif event == "tone_a" and self.state == self.WAIT_A_AFTER_PROBE:
            self.state = self.WAIT_A_REV3
        elif event == "tone_a" and self.state == self.TX_PROBE:
            if (self._tone_a_deadline_at is not None
                    and at > self._tone_a_deadline_at):
                self.state = self.FAILED
                self._tone_a_deadline_at = None
                self.actions.append(Phase2Action(tx_now, "retrain-timeout"))
                return self.drain_actions()
            self._tone_a_during_second_l2 = True
            # §9.2.1.2.5 is specifically the deadline for detecting Tone A.
            # Once that event is timely, the 650 ms timer must not fire while
            # the controller waits for the separate L2-echo condition.
            self._tone_a_deadline_at = None
            if self._second_l2_echo_complete_at is not None:
                self._send_info1d(tx_now)
        elif event == "tone_a_reversal_after_l2":
            self._on_l2_a_reversal(at, tx_now)
        elif event == "l2_echo_complete" and self.state == self.TX_PROBE:
            self._second_l2_echo_complete_at = at
            if self._tone_a_during_second_l2:
                self._send_info1d(tx_now)
        elif event == "tone_a" and self.state == self.WAIT_INFO1A_RETRAIN:
            # V.90 §9.2.1.2.6: after the INFO1a deadline, Tone A requests a
            # retrain response under §9.5.1.2. The detector emits this only
            # after qualifying the carrier duration.
            self._info1a_deadline_at = None
            self._a_rev_count = 0
            self.state = self.WAIT_RETRAIN_TONE_B
            self._retrain_tone_b_start_at = tx_now + int(.070 * FS)
            self.retrain_response_started = True
            self.actions.extend((Phase2Action(tx_now, "silence"),
                                 Phase2Action(self._retrain_tone_b_start_at,
                                              "tone_b")))
        elif (event == "info1a" and self.state in
              (self.WAIT_INFO1A, self.WAIT_INFO1A_RETRAIN)):
            if not payload or "mode" not in payload:
                raise ValueError("INFO1a event requires validated mode field")
            mode = int(payload["mode"])
            if not 0 <= mode <= 6:
                raise ValueError("INFO1a mode must be a defined value (0..6)")
            self.info1a = dict(payload)
            self._info1a_deadline_at = None
            self.actions.append(Phase2Action(tx_now, "handoff-v90-phase3" if mode == 6
                                              else "handoff-v34-call-phase3", mode))
            self.state = self.COMPLETE
        else:
            raise ValueError(f"event {event!r} is invalid in state {self.state!r}")
        return self.drain_actions()

    def _on_info0a(self, at: int, payload: dict) -> None:
        if self.state not in (self.WAIT_INFO0A, self.WAIT_A_REV2,
                              self.RX_PROBE, self.WAIT_RETRAIN_A_REV2):
            raise ValueError(f"INFO0a is invalid in state {self.state!r}")
        repeated = self.info0a is not None
        self.info0a = dict(payload)
        # The normal first INFO0a does not interrupt INFO0d/Tone B. Recovery
        # §9.2.1.2.1 repeats INFO0d with acknowledgement after a duplicate
        # INFO0a demonstrates that the first exchange was not synchronized.
        if repeated:
            if self.state in (self.WAIT_A_REV2, self.RX_PROBE):
                # Recovery returns to §9.2.1.1.3: after the acknowledged
                # INFO0d and resumed Tone B, the next Tone-A reversal is the
                # first reversal of a fresh ranging attempt. Do not let the
                # pre-recovery reversal count make it look like the second
                # reversal and skip L1/L2 qualification prematurely.
                self._a_rev_count = 0
                self.b_reversal_at = None
                self._second_initial_a_reversal_at = None
                self._missing_tone_a_reversal_deadline_at = None
                self.rtd_samples = 0
                self._first_probe_l1_done = False
                self.l2_started_at = None
                if self.state == self.RX_PROBE:
                    # A duplicate INFO0a can be the peer's recovery signal
                    # while we have already opened first-probe reception.
                    # Abandon that incomplete ranging attempt and restart the
                    # A/B exchange after the queued INFO0d ACK.
                    self.state = self.WAIT_A_REV2
            self.actions.extend((
                Phase2Action(None, "info0d", 1,
                             after="current-info0d-complete"),
                # The caller restarts its wait for Tone B after the repeated
                # INFO0 exchange. Resume it only when the acknowledged frame
                # actually completes on the media timeline.
                Phase2Action(None, "tone_b", after="info0d-complete"),
            ))

    def _on_a_reversal(self, at: int, now: int) -> None:
        if self.state not in (self.WAIT_INFO0A, self.WAIT_A_REV2,
                              self.WAIT_RETRAIN_A_REV2):
            raise ValueError(f"Tone A reversal is invalid in state {self.state!r}")
        self._a_rev_count += 1
        if self._a_rev_count == 1:
            nominal = at + int(0.040 * FS)
            due = max(now, nominal)
            latest = at + int(0.041 * FS)
            self.actions.extend((Phase2Action(due, "reverse-tone-b", latest=latest),
                                 Phase2Action(due + int(0.010 * FS), "silence")))
            self.b_reversal_at = due
            self.state = self.WAIT_A_REV2
        else:
            # The second A reversal closes the ranging measurement; unlike
            # the first, it is followed by receive L1/L2, not another B reply.
            self.set_rtd_from_second_reversal(at)
            self._second_initial_a_reversal_at = at
            self._missing_tone_a_reversal_deadline_at = (
                at + int(.900 * FS) + self.rtd_samples)
            self.state = self.RX_PROBE

    def set_rtd_from_second_reversal(self, second_a_reversal_at: int) -> int:
        """Update RTD from B-reversal to second A-reversal, less 40 ms."""
        if self.b_reversal_at is None or second_a_reversal_at < self.b_reversal_at:
            raise ValueError("need a valid B reversal and later second A reversal")
        self.rtd_samples = max(0, second_a_reversal_at - self.b_reversal_at - int(.040 * FS))
        return self.rtd_samples

    def _on_l2_a_reversal(self, at: int, now: int) -> None:
        if self.state != self.WAIT_A_REV3:
            raise ValueError("post-probe Tone A reversal is out of sequence")
        self._missing_tone_a_reversal_deadline_at = None
        # Every Phase 2 retry has a new local L2. Do not let the previous
        # attempt's echo timestamp or Tone-A observation qualify this one.
        self._second_l2_echo_complete_at = None
        self._tone_a_during_second_l2 = False
        nominal = at + int(.040 * FS)
        due = max(now, nominal)
        latest = at + int(.041 * FS)
        self.actions.extend((Phase2Action(due, "reverse-tone-b", latest=latest),
                             Phase2Action(due + int(.010 * FS), "l1"),
                             Phase2Action(due + int(.010 * FS) + int(.160 * FS), "l2",
                                          self.max_l2_samples)))
        self.l2_started_at = due + int(.010 * FS) + int(.160 * FS)
        # V.90 §9.2.1.2.5 bounds waiting for Tone A from the beginning of
        # this L2. RTD is updated from Phase 2 ranging.
        self._tone_a_deadline_at = (
            self.l2_started_at + int(.650 * FS) + self.rtd_samples)
        self.state = self.TX_PROBE

    def _begin_missing_tone_a_reversal_recovery(self, now: int) -> list[Phase2Action]:
        """Run V.90 §9.2.1.2.4 after the caller's second reversal is late."""
        deadline = self._missing_tone_a_reversal_deadline_at
        if deadline is None:
            return []
        due = max(int(now), deadline + int(.040 * FS))
        latest = deadline + int(.041 * FS)
        self.actions.extend((
            Phase2Action(due, "reverse-tone-b", latest=latest),
            Phase2Action(due + int(.010 * FS), "l1"),
            Phase2Action(due + int(.010 * FS) + int(.160 * FS),
                         "l2", self.max_l2_samples),
        ))
        self.l2_started_at = due + int(.010 * FS) + int(.160 * FS)
        self._tone_a_deadline_at = (
            self.l2_started_at + int(.650 * FS) + self.rtd_samples)
        self._missing_tone_a_reversal_deadline_at = None
        self._tone_a_during_second_l2 = False
        self._second_l2_echo_complete_at = None
        self.missing_tone_a_recovery_started = True
        self.state = self.TX_PROBE
        return self.drain_actions()

    def tick(self, now: int) -> list[Phase2Action]:
        """Run bounded Phase 2 timers on the shared 8 kHz media timeline."""
        if isinstance(now, bool) or not isinstance(now, int) or now < 0:
            raise ValueError("sample positions must be nonnegative integers")
        if (self.info1a_timeout_retrain
                and self.state == self.WAIT_INFO1A
                and self._info1a_deadline_at is not None
                and now >= self._info1a_deadline_at):
            # V.90 §9.2.1.2.6 begins Tone-A / INFOMARKS monitoring after the
            # response window. INFOMARKSa is not implemented yet; INFO1a
            # remains accepted while the Tone-A retrain response is enabled.
            self.state = self.WAIT_INFO1A_RETRAIN
        if (self.state == self.WAIT_RETRAIN_TONE_B
                and self._retrain_tone_b_start_at is not None
                and now >= self._retrain_tone_b_start_at):
            self.state = self.WAIT_RETRAIN_A_REVERSAL
            self._retrain_tone_b_start_at = None
            if self._pending_retrain_a_reversal is not None:
                reversal_at, payload = self._pending_retrain_a_reversal
                self._pending_retrain_a_reversal = None
                return self.feed("retrain_tone_a_reversal", reversal_at,
                                 now=now, payload=payload)
        if (self._missing_tone_a_reversal_deadline_at is not None
                and self.state in (self.WAIT_A_AFTER_PROBE, self.WAIT_A_REV3)
                and now >= self._missing_tone_a_reversal_deadline_at):
            return self._begin_missing_tone_a_reversal_recovery(now)
        if (self.state == self.TX_PROBE and not self._tone_a_during_second_l2
                and self._tone_a_deadline_at is not None
                and now >= self._tone_a_deadline_at):
            self.state = self.FAILED
            self._tone_a_deadline_at = None
            self.actions.append(Phase2Action(now, "retrain-timeout"))
        return self.drain_actions()

    def info1d_transmitted(self, complete_at: int) -> None:
        """Arm §9.2.1.2.6 from the actual INFO1d transmit completion sample."""
        if (isinstance(complete_at, bool) or not isinstance(complete_at, int)
                or complete_at < 0):
            raise ValueError("INFO1d completion must be a nonnegative sample")
        if not self.info1a_timeout_retrain:
            return
        if self.state != self.WAIT_INFO1A:
            raise RuntimeError("INFO1d completion is only valid while awaiting INFO1a")
        self._info1a_deadline_at = (complete_at + int(.700 * FS)
                                    + self.rtd_samples)

    def _send_info1d(self, at: int) -> None:
        echo_at = self._second_l2_echo_complete_at
        if (self.l2_started_at is None or echo_at is None
                or echo_at < self.l2_started_at or at < self.l2_started_at):
            raise ValueError("L2 echo cannot complete before local L2 begins")
        # The 550 ms limit applies to the local L2 echo observation. Tone A
        # can arrive later while the peer runs its bounded missing-Tone-B
        # recovery; using that later event time here incorrectly rejects a
        # timely echo and prevents the recovery exchange from completing.
        slack = self.l2_echo_tolerance_samples
        if echo_at - self.l2_started_at > int(.550 * FS) + self.rtd_samples + slack:
            self.state = self.FAILED
            self._tone_a_deadline_at = None
            self.actions.append(Phase2Action(at, "retrain-timeout"))
            return
        if at - self.l2_started_at > int(.650 * FS) + self.rtd_samples + slack:
            self.state = self.FAILED
            self._tone_a_deadline_at = None
            self.actions.append(Phase2Action(at, "retrain-timeout"))
            return
        self._tone_a_deadline_at = None
        self.actions.extend((Phase2Action(at, "info1d"),
                             Phase2Action(None, "silence", after="info1d-complete")))
        self.state = self.WAIT_INFO1A

    def drain_actions(self) -> list[Phase2Action]:
        actions, self.actions = self.actions, []
        return actions
