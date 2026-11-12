"""Event-driven V.90 digital-answerer Phase 3 transmitter controller.

This module joins the spec-derived raw G.711 signal generators into the
digital-modem sequence.  It intentionally does not detect the analogue
modem's S transitions or implement the Phase 4 upstream/downstream data
receiver.  A call controller must deliver those events from a qualified
receiver before this can be connected to a live session.
"""

from __future__ import annotations

from collections import deque

from v90_digital_tx import V90DigitalPhase3Transmitter, sd_codewords, sd_prime_codewords

FS = 8000
S_DETECTION_TIMEOUT_MS = 5100


class V90DigitalAnswererPhase3:
    """Sequence digital Phase 3 codewords in response to received S events.

    ``start()`` begins S_d, S'_d, TRN1d and then repeated Jd frames.  The
    first ``notify_s_transition()`` received during Jd completes the current
    Jd frame and emits J'_d. Once J'_d has been sent, a zero-length DIL
    proceeds directly to Phase 4; otherwise DIL starts immediately. A later
    transition stops DIL at a complete segment boundary.

    ``read(count)`` returns up to ``count`` exact negotiated G.711 octets.
    It can return fewer only when it reaches a state that is waiting for an
    input event or when Phase 3 is complete.  It never fabricates silence or
    user data to fill an event wait.
    """

    IDLE = "idle"
    INITIAL = "initial-signals"
    JD = "jd-repeat"
    JPRIME = "jprime"
    DIL = "dil"
    PHASE4_READY = "phase4-ready"
    FAILED = "failed"

    def __init__(self, uinfo: int, law: str, descriptor: dict | None,
                 rate_mask: int, *, phase3_16point: bool = False,
                 renegotiation_16point: bool = False,
                 max_lookahead: int = 1, trn1d_symbols: int = 2040,
                 round_trip_delay_samples: int = 0):
        self.uinfo = int(uinfo)
        self.law = law.lower()
        self.descriptor: dict | None = None
        self.rate_mask = int(rate_mask)
        self.phase3_16point = bool(phase3_16point)
        self.renegotiation_16point = bool(renegotiation_16point)
        self.max_lookahead = int(max_lookahead)
        self.trn1d_symbols = int(trn1d_symbols)
        if (isinstance(round_trip_delay_samples, bool)
                or not isinstance(round_trip_delay_samples, int)
                or round_trip_delay_samples < 0):
            raise ValueError("round-trip delay samples must be a nonnegative integer")
        self.round_trip_delay_samples = round_trip_delay_samples
        self.transmitter = V90DigitalPhase3Transmitter(self.uinfo, self.law)
        self.state = self.IDLE
        self._queue: deque[int] = deque()
        self._jd_event_pending = False
        self._dil = None
        self.s_transitions = 0
        self._samples_emitted = 0
        self._trn1d_start_sample: int | None = None
        self.failure_reason: str | None = None
        if descriptor is not None:
            self.set_dil_descriptor(descriptor)

    @property
    def phase4_ready(self) -> bool:
        return self.state == self.PHASE4_READY

    def start(self) -> None:
        if self.state != self.IDLE:
            raise RuntimeError("V.90 Phase 3 has already started")
        sd = sd_codewords(self.uinfo, self.law)
        sd_prime = sd_prime_codewords(self.uinfo, self.law)
        self._trn1d_start_sample = len(sd) + len(sd_prime)
        initial = sd + sd_prime + self.transmitter.trn1d(self.trn1d_symbols)
        self._queue.extend(initial)
        self.state = self.INITIAL

    def _check_s_timeout(self) -> bool:
        if (self.state == self.JD and not self._jd_event_pending
                and self._trn1d_start_sample is not None
                and self._samples_emitted >= self._trn1d_start_sample
                + S_DETECTION_TIMEOUT_MS * FS // 1000
                + self.round_trip_delay_samples):
            self.state = self.FAILED
            self.failure_reason = (
                "V.90 §9.3.1.5: no analogue S within 5100 ms plus RTD of TRN1d")
            return True
        return False

    def notify_s_transition(self) -> None:
        """Deliver one already-detected analogue S-to-S transition."""
        self._check_s_timeout()
        if self.state == self.IDLE:
            raise RuntimeError("cannot deliver an S transition before Phase 3 starts")
        if self.state in (self.PHASE4_READY, self.FAILED):
            return
        self.s_transitions += 1
        if self.state == self.JD:
            # Finish the in-flight Jd frame before switching to J'_d.
            self._jd_event_pending = True
        elif self.state == self.DIL:
            self._dil.request_stop()

    def set_dil_descriptor(self, descriptor: dict) -> None:
        """Install the parsed caller Ja descriptor before the DIL decision.

        The answerer learns this descriptor from the analogue upstream after
        its own Phase 3 startup has begun; it cannot be required at engine
        construction. It must be present by the end of J-prime, when the
        requested zero/nonzero DIL behavior takes effect.
        """
        if self.state in (self.DIL, self.PHASE4_READY):
            raise RuntimeError("Ja descriptor arrived after the DIL decision")
        if not isinstance(descriptor, dict) or "N" not in descriptor:
            raise ValueError("Ja descriptor must contain its parsed N field")
        if not isinstance(descriptor["N"], int) or descriptor["N"] < 0:
            raise ValueError("Ja descriptor N must be a nonnegative integer")
        self.descriptor = dict(descriptor)

    def _begin_dil_or_finish(self) -> None:
        if self.descriptor is None:
            raise RuntimeError("caller Ja descriptor is required before DIL selection")
        self._dil = self.transmitter.dil_stream(self.descriptor)
        if self._dil is None:
            self.state = self.PHASE4_READY
        else:
            self.state = self.DIL

    def _next_jd(self) -> None:
        if self._jd_event_pending and self.transmitter.jd_frames_sent:
            self._jd_event_pending = False
            self._queue.extend(self.transmitter.jprime_d())
            self.state = self.JPRIME
            return
        self._queue.extend(self.transmitter.jd_frame(
            self.rate_mask,
            phase3_16point=self.phase3_16point,
            renegotiation_16point=self.renegotiation_16point,
            max_lookahead=self.max_lookahead,
        ))

    def read(self, max_samples: int) -> bytes:
        if not isinstance(max_samples, int) or max_samples <= 0:
            raise ValueError("max_samples must be a positive integer")
        if self.state == self.IDLE:
            raise RuntimeError("V.90 Phase 3 has not started")
        if self.state == self.FAILED:
            return b""

        out = bytearray()
        while len(out) < max_samples:
            if self._check_s_timeout():
                break
            until_timeout = None
            if (self.state == self.JD and not self._jd_event_pending
                    and self._trn1d_start_sample is not None):
                deadline = (self._trn1d_start_sample
                            + S_DETECTION_TIMEOUT_MS * FS // 1000
                            + self.round_trip_delay_samples)
                until_timeout = max(0, deadline - self._samples_emitted)
                if until_timeout == 0:
                    self._check_s_timeout()
                    break
            if self._queue:
                take = min(max_samples - len(out), len(self._queue))
                if until_timeout is not None:
                    take = min(take, until_timeout)
                out.extend(self._queue.popleft() for _ in range(take))
                self._samples_emitted += take
                if self.state == self.INITIAL and not self._queue:
                    self.state = self.JD
                elif self.state == self.JPRIME and not self._queue:
                    self._begin_dil_or_finish()
                continue

            if self.state == self.JD:
                self._next_jd()
                continue
            if self.state == self.JPRIME:
                # J-prime has drained. V.90 §9.3.1.5 proceeds directly to
                # Phase 4 for N=0, or §9.3.1.6 starts the requested DIL now.
                self._begin_dil_or_finish()
                continue
            if self.state == self.DIL:
                chunk = self._dil.read(max_samples - len(out))
                out.extend(chunk)
                if not chunk or self._dil.stopped:
                    self.state = self.PHASE4_READY
                    break
                continue
            # No signal is generated while waiting for an S event or after
            # the Phase 4 boundary.  The caller owns media-state handoff.
            break
        return bytes(out)
