"""Automatic line-mode selection (our own code): V.8 first, then common modulation or legacy fallback.

``V8Chain`` presents the same surface as the single engines (tx_samples, feed_pcm, feed_bytes, pending_bits, bit_rate,
is_settled, v42_*...). It runs V.8 (ITU-T V.8) offering V.34 / V.32bis / V.22bis / V.21, and when the V.8 exchange
finishes it builds the engine for the highest modulation both ends named and hands the line over to it, forwarding calls
from then on. Legacy callers recognize V.32 AC or V.22 USB1. Legacy answerers probe V.32 AC
for two seconds after V.8 times out, then fall back to V.22 when it is allowed. The opt-in Bell 103 fallback is separate;
legacy V.21 without a completed V.8 exchange is not selected by this chain.
"""
import threading
import os
import time

from v8 import (V8_MOD_V21, V8_MOD_V22, V8_MOD_V32, V8_MOD_V34,
                V8_PCM_ANALOG, V8_PCM_DIGITAL, V8_PCM_V91,
                V8_PSTN_ACCESS_DIGITAL_NETWORK,
                V8_PROTOCOL_NONE, V8_PROTOCOL_LAPM_V42, V8Negotiator)
from v8 import np
from autodetect import AnsGate


class _LegacyUSB1:
    """Two independent 200 ms blocks of V.22 answer-side unscrambled ones.

    Require a dominant 2250 Hz line, rather than interpreting plain ANS,
    ANSam, V.34 tone A, or Bell's 2225 Hz mark as V.22. This detector only
    runs while the caller is still waiting for a V.8 offer.
    """
    def __init__(self):
        self.pending = []
        self.votes = 0

    def feed(self, samples):
        self.pending.extend(samples)
        while len(self.pending) >= 1600:
            x = np.asarray(self.pending[:1600], dtype=float)
            del self.pending[:1600]
            x -= x.mean()
            power = float(x @ x)
            spectrum = np.abs(np.fft.rfft(x)) ** 2
            # 5 Hz bins; summing the three bins tolerates small frequency
            # offsets without admitting the distinct 2225 Hz signal.
            fraction = float(spectrum[449:452].sum()) / (800 * power + 1e-30)
            self.votes = self.votes + 1 if power / 1600 > 40 ** 2 and fraction > .8 else 0
            if self.votes >= 2:
                return True
        return False


class _LegacyAC:
    """Require both legacy V.32 AC lines in two independent 80 ms blocks."""
    def __init__(self):
        self.pending = []
        self.votes = 0

    def feed(self, samples):
        self.pending.extend(samples)
        while len(self.pending) >= 640:
            x = np.asarray(self.pending[:640], dtype=float)
            del self.pending[:640]
            x -= x.mean()
            power = float(x @ x)
            spectrum = np.abs(np.fft.rfft(x)) ** 2
            lines = [float(spectrum[k-1:k+2].sum()) / (320 * power + 1e-30)
                     for k in (48, 240)]
            present = power / 640 > 40 ** 2 and min(lines) > .15 and sum(lines) > .8
            self.votes = self.votes + 1 if present else 0
            if self.votes >= 2:
                return True
        return False


class V8Chain:
    capture_detect_bits = False

    def __init__(self, calling_party: bool, v34_rate: int = 33600, v32_rates=(14400, 12000, 9600, 7200, 4800),
                 v22_rate: int = 2400, allow=(V8_MOD_V34 | V8_MOD_V32 | V8_MOD_V22), patience: float = 6.0,
                 debug: bool = False, b103_fallback: bool | None = None, v22_attempt_seconds: float | None = None,
                 v22_min_rate: int = 1200, v34_min_rate: int = 2400,
                 protocols: int = V8_PROTOCOL_NONE,
                 pcm_availability: int | bool = 0,
                 v90_engine_factory=None):
        if isinstance(pcm_availability, bool):
            pcm_availability = V8_PCM_ANALOG if pcm_availability else 0
        pcm_availability = int(pcm_availability)
        if pcm_availability & V8_PCM_V91:
            raise ValueError("V.91 PCM capability is not supported by this V.90 engine selector")
        if pcm_availability not in (0, V8_PCM_ANALOG, V8_PCM_DIGITAL):
            raise ValueError("automode PCM availability must select exactly one V.90 role")
        if pcm_availability and v90_engine_factory is None:
            raise ValueError("V.8 PCM availability requires a V.90 engine factory")
        if pcm_availability == V8_PCM_DIGITAL and calling_party:
            raise ValueError("digital V.90 PCM answerer capability requires answer role")
        if pcm_availability == V8_PCM_ANALOG and not calling_party:
            raise ValueError("analogue V.90 PCM caller capability requires caller role")
        self.calling = calling_party
        self.pcm_availability = pcm_availability
        self.v90_engine_factory = v90_engine_factory
        self._failed = False
        self.v34_rate, self.v34_min_rate, self.v32_rates, self.v22_rate = v34_rate, v34_min_rate, tuple(v32_rates), v22_rate
        self.v22_min_rate = v22_min_rate
        self.allow = allow
        self.patience = patience
        self.debug = debug
        self.events: list = []
        # None means V.8 has not completed (or was not used); zero means it
        # completed and the peers did not negotiate a common protocol.
        self.negotiated_protocols: int | None = None
        self.engine = None
        self._engine_events_seen = 0
        self.kind = "v8"
        self._lock = threading.RLock()
        self._rx_lock = threading.RLock()
        self._trace_tx = os.environ.get('EL_MODEM_TIMING_TRACE') == '1'
        self._tx_timing = None
        # A V.90 digital answerer is on a digital network connection; V.8
        # access0.b7 tells the analogue caller so. captured peer sets it, and the
        # SmartLink V.90 caller falls back to V.34 Phase 2 without it.
        self._v8 = V8Negotiator(calling_party, modulations=allow, protocols=protocols,
                                pcm_availability=pcm_availability,
                                pstn_access_bits=(V8_PSTN_ACCESS_DIGITAL_NETWORK
                                                  if pcm_availability == V8_PCM_DIGITAL
                                                  else 0))
        self._t = 0
        # DCE supplies the monotonic deadline associated with each outgoing
        # RTP frame before asking this chain to generate its samples. Keep the
        # sample cursor at that boundary so a later V.8 handoff can anchor the
        # negotiated RTP media clock to the actual transmit timeline rather
        # than to the receive thread's arbitrary selection time.
        self._tx_media_deadline: float | None = None
        self._tx_media_sample = 0
        self._queued = bytearray()
        self._queued_bits: list = []
        self._legacy_probe_at = None
        self._tone_gate: AnsGate | None = None      # set when plain ANS (not ANSam) ended V.8: hide the 2100 Hz tone
        self._choose_t = 0
        self._no_v8_legacy_stage = 0
        self._no_v8_legacy_next_at = None
        self.b103_fallback = b103_fallback if b103_fallback is not None else os.environ.get('V8CHAIN_B103_FALLBACK') == '1'
        self.v22_attempt_seconds = v22_attempt_seconds if v22_attempt_seconds is not None else float(os.environ.get('V8CHAIN_V22_ATTEMPT_S', '8'))
        self._no_common_logged = False
        self._ac = _LegacyAC() if calling_party and allow & V8_MOD_V32 and np is not None else None
        self._usb1 = _LegacyUSB1() if calling_party and allow & V8_MOD_V22 and np is not None else None

    # -- helpers ---------------------------------------------------------------------------------------------------
    def _log(self, text: str) -> None:
        self.events.append((round(self._t / 8000.0, 3), text))

    def _sync_engine_events(self) -> None:
        """Forward the chosen engine's events into our own list so the DCE's poll() can log them.

        Without this, a V.34 (or V.32) handshake after V.8 is invisible in the live log: the engine
        keeps its own ``events`` list and the DCE only reads ``V8Chain.events``.
        """
        eng = self.engine
        if eng is None:
            return
        ev = eng.events
        while self._engine_events_seen < len(ev):
            self.events.append(ev[self._engine_events_seen])
            self._engine_events_seen += 1

    def _check_b103_fallback(self) -> None:
        """Opt-in (``b103_fallback`` / ``V8CHAIN_B103_FALLBACK=1``): a V.22 attempt that has not connected after
        ``v22_attempt_seconds`` gives way to Bell 103 at 300 bit/s. Bell 103 is not a V.8 modulation, so only this step reaches it."""
        if (self.b103_fallback and self.kind == "v22" and self.engine is not None and not self.engine.connected
                and self._t - self._choose_t >= self.v22_attempt_seconds * 8000):
            self._log("V.22 attempt did not connect: falling back to Bell 103 (300 bit/s)")
            self._choose("b103")

    def _choose(self, kind: str, *, probe=False, v90_role: str | None = None,
                v8_result: dict | None = None) -> None:
        if self.engine is not None and hasattr(self.engine, "close"):
            self.engine.close()
        self._legacy_probe_at = self._t if probe else None
        self._choose_t = self._t
        self.kind = kind
        self._log(f"line mode: {kind}")
        calling = self.calling
        if kind == "v90":
            if self.v90_engine_factory is None or v90_role not in (
                    "digital-answerer", "analogue-caller"):
                raise RuntimeError("V.90 selection requires an enabled role engine")
            media_origin = time.monotonic()
            if self._tx_media_deadline is not None:
                media_origin = (self._tx_media_deadline
                                + (self._t - self._tx_media_sample) / 8000.0)
            engine = self.v90_engine_factory(
                role=v90_role, negotiation=dict(v8_result or {}),
                media_at=self._t, media_origin=media_origin)
            required = ("tx_samples", "feed_pcm", "feed_bytes", "tx_g711",
                        "feed_g711", "feed_gap")
            missing = [name for name in required
                       if not callable(getattr(engine, name, None))]
            if missing:
                raise TypeError("V.90 engine factory returned incomplete engine; missing "
                                + ", ".join(missing))
            self.engine = engine
        elif kind == "v34":
            from v34_answer_modem import V34AnswerModem  # noqa: PLC0415
            from v34_modem import V34Modem  # noqa: PLC0415

            self.engine = (V34Modem if calling else V34AnswerModem)(data_rate=self.v34_rate,
                                                                    min_data_rate=self.v34_min_rate, debug=self.debug,
                                                                    with_v8=False)
        elif kind == "v32":
            if calling:
                from v32_stream import V32Modem  # noqa: PLC0415

                self.engine = V32Modem(fs=8000, my_rates=self.v32_rates, debug=self.debug, use_v8=False, echo=False)
            else:
                from v32_answer import V32Answerer  # noqa: PLC0415

                self.engine = V32Answerer(fs=8000, my_rates=self.v32_rates, debug=self.debug, use_v8=False, echo=False)
                self.engine.state, self.engine._lead_left = "gap", 0          # V.8 has just finished
        elif kind == "b103":
            from bell103_engine import Bell103Engine  # noqa: PLC0415

            self.engine = Bell103Engine(calling)
        elif kind == "v21":
            from bell103_engine import Bell103Engine  # noqa: PLC0415

            self.engine = Bell103Engine(calling, standard="v21")
        else:
            from v22 import V22Modem  # noqa: PLC0415

            self.engine = V22Modem(calling, self.v22_rate, min_rate=self.v22_min_rate)
        if self._v8 is not None:
            self._v8.close()
            self._v8 = None
        self._engine_events_seen = 0
        if self._queued and not probe:
            self.engine.feed_bytes(bytes(self._queued))
            self._queued.clear()

    def _advance_no_v8_legacy_fallback(self) -> None:
        """Advance bounded legacy V.34 -> V.32 -> V.22 startup attempts."""
        if not self._no_v8_legacy_stage or self.engine is None:
            return
        if getattr(self.engine, "connected", False):
            self._no_v8_legacy_stage = 0
            self._no_v8_legacy_next_at = None
            return
        if self._no_v8_legacy_next_at is None or self._t < self._no_v8_legacy_next_at:
            return
        if self._no_v8_legacy_stage == 1:
            if self.allow & V8_MOD_V32:
                self._log("legacy V.34 unanswered: falling back to V.32")
                self._choose("v32", probe=not self.calling)
                self._no_v8_legacy_stage = 2
                # Give the legacy V.32 startup a real bounded window. The
                # former two-second slot was shorter than its acquisition
                # sequence and caused an automatic V.22 switch before V.32
                # could complete.
                self._no_v8_legacy_next_at = self._t + 12 * 8000
            elif self.allow & V8_MOD_V22:
                self._log("legacy V.34 unanswered: falling back to V.22/V.22bis")
                self._choose("v22")
                self._no_v8_legacy_stage = 0
                self._no_v8_legacy_next_at = None
            else:
                self._no_v8_legacy_stage = 0
                self._no_v8_legacy_next_at = None
        elif self._no_v8_legacy_stage == 2:
            if self.allow & V8_MOD_V22:
                self._log("legacy V.32 unanswered: falling back to V.22/V.22bis")
                self._choose("v22")
            self._no_v8_legacy_stage = 0
            self._no_v8_legacy_next_at = None

    def _pick(self, mods: int) -> str | None:
        mods &= self.allow
        for bit, kind in ((V8_MOD_V34, "v34"), (V8_MOD_V32, "v32"),
                          (V8_MOD_V22, "v22"), (V8_MOD_V21, "v21")):
            if mods & bit:
                return kind
        return None

    def _step_v8(self) -> None:
        st = self._v8.status
        # Plain ANS can start V.22 at once only when no legacy V.32 AC is possible; otherwise the AC detector
        # keeps listening (a V.32 answerer sends ANS, then AC) and V.22 waits for the patience timeout.
        plain_now = st == 0 and self._v8.plain_ans and self._ac is None
        if st == 2:
            self.negotiated_protocols = int(self._v8.result.get("protocols", V8_PROTOCOL_NONE))
            result = self._v8.result
            kind = self._pick(result["modulations"])
            pcm_role = int(result.get("pcm_negotiated_bits", 0))
            self._log(f"V.8 complete: {result}")
            if pcm_role:
                expected = (V8_PCM_ANALOG if self.calling else V8_PCM_DIGITAL)
                if pcm_role != expected or self.v90_engine_factory is None:
                    # Do not silently start voiceband V.34 after the peers
                    # agreed to a digital PCM role; that would corrupt the
                    # negotiated call rather than provide a valid fallback.
                    self.kind = "v90-unavailable"
                    self._failed = True
                    self._log("V.8 negotiated a PCM role without a matching V.90 engine; refusing V.34 fallback")
                    self._v8.close()
                    self._v8 = None
                    return
                role = "analogue-caller" if self.calling else "digital-answerer"
                self._choose("v90", v90_role=role, v8_result=result)
                return
            if kind is not None:
                self._choose(kind)
            elif self.allow & V8_MOD_V22:
                # No common modulation: fall back to V.22/V.22bis as before, rather than leaving the line silent.
                self._log("V.8 completed without an allowed common modulation: falling back to V.22/V.22bis")
                self._choose("v22")
            elif not self._no_common_logged:
                self._no_common_logged = True       # _step_v8 runs on every receive block; log once
                self._log("V.8 completed without an allowed common modulation and V.22 is not allowed")
        elif st == 4:
            # Status 4 means ANSam was detected and CM was sent, but no JM
            # arrived. This is the one no-completion case where a legacy V.34
            # Phase 1 attempt is justified. A plain ANS/no-ANS timeout is not
            # evidence of a V.34 peer and must retain the legacy V.32/V.22
            # startup order.
            if self.allow & V8_MOD_V34:
                if self._t >= 12 * 8000:
                    self._log("ANSam received but no V.8 menu completed: starting legacy V.34")
                    self._choose("v34")
                    self._no_v8_legacy_stage = 1
                    self._no_v8_legacy_next_at = self._t + 12 * 8000
                return
        elif (st == 3 or plain_now
              or (st == 0 and self._t / 8000.0 > self.patience)):
            # No ANSam or plain V.25 ANS follows the direct legacy path. In
            # particular, don't spend another 12 seconds on undefined V.34
            # startup before the already-supported V.32/V.22 paths.
            if not self.calling and self.allow & V8_MOD_V32:
                self._log("no V.8 partner: probing legacy V.32 AC")
                self._choose("v32", probe=True)
            elif self.allow & V8_MOD_V22:
                if self._v8.plain_ans:
                    # Plain ANS: the answerer is a V.22-era modem. Start V.22/V.22bis now, not after the 12 s V.8 timeout,
                    # and keep the answer tone away from the V.22 engine (a V.22 caller trains on it otherwise).
                    self._log("plain ANS (no V.8 partner): starting V.22/V.22bis")
                    self._tone_gate = AnsGate()
                else:
                    self._log("no V.8 partner: falling back to V.22/V.22bis")
                self._choose("v22")
            elif self.allow & V8_MOD_V32:
                self._choose("v32")

    # -- engine surface --------------------------------------------------------------------------------------------
    @property
    def tx_g711(self):
        """Expose exact-codeword transmit only when the selected engine supports it."""
        engine = self.engine
        method = None if engine is None else getattr(engine, "tx_g711", None)
        return method if callable(method) else None

    @property
    def feed_g711(self):
        """Expose exact-codeword receive only after a raw-G.711 engine is selected."""
        engine = self.engine
        method = None if engine is None else getattr(engine, "feed_g711", None)
        return method if callable(method) else None

    @property
    def feed_gap(self):
        """Expose sample-gap notification only when the selected engine supports it."""
        engine = self.engine
        method = None if engine is None else getattr(engine, "feed_gap", None)
        return method if callable(method) else None

    def tx_samples(self, count: int) -> list[int]:
        started = time.monotonic() if self._trace_tx else None
        cpu_started = time.thread_time() if self._trace_tx else None
        with self._lock:
            acquired = time.monotonic() if self._trace_tx else None
            cpu_acquired = time.thread_time() if self._trace_tx else None
            try:
                self._t += count
                self._advance_no_v8_legacy_fallback()
                if self._failed:
                    return [0] * count
                if self.engine is None:
                    return self._v8.tx(count)
                out = self.engine.tx_samples(count)
                self._sync_engine_events()
                return out
            finally:
                if self._trace_tx:
                    ended, cpu_ended = time.monotonic(), time.thread_time()
                    self._tx_timing = dict(lock_wait_seconds=acquired-started,
                        lock_wait_cpu_seconds=cpu_acquired-cpu_started,
                        engine_seconds=ended-acquired,
                        engine_cpu_seconds=cpu_ended-cpu_acquired)

    def set_tx_media_deadline(self, deadline: float) -> None:
        """Anchor the current generated-sample cursor to an RTP TX deadline."""
        with self._lock:
            self._tx_media_deadline = float(deadline)
            self._tx_media_sample = self._t

    def _note_receive_transport(self, method: str, samples: int) -> None:
        # Notification belongs to the current receive chunk/selected engine.
        # Do not retain V.8-era marks for an engine selected on a later chunk.
        with self._rx_lock:
            with self._lock:
                note = getattr(self.engine, method, None)
                if note is not None:note(samples)

    def note_concealed_rx(self, n: int) -> None:
        self._note_receive_transport('note_concealed_rx', n)

    def note_transport_discontinuity_rx(self, n: int = 0) -> None:
        self._note_receive_transport('note_transport_discontinuity_rx', n)

    def feed_pcm(self, pcm: bytes) -> bytes:
        # Keep receive and shutdown serialized, but do not hold the dispatch
        # lock through V.34's DSP. Match the selected engine's direct-DCE
        # TX/RX concurrency; the caller protects its own waveform splices.
        # Blocking TX here stalls RTP during expensive training.
        with self._rx_lock:
            with self._lock:
                if self._failed:
                    return b""
                engine = self.engine if self.kind == 'v34' else None
            if engine is not None:
                out = engine.feed_pcm(pcm)
                with self._lock:
                    self._sync_engine_events()
                return out
            return self._feed_pcm_negotiating(pcm)

    def _feed_pcm_negotiating(self, pcm: bytes) -> bytes:
        import struct  # noqa: PLC0415

        with self._lock:
            if self.engine is None:
                samples = list(struct.unpack(f"<{len(pcm) // 2}h", pcm))
                self._v8.rx(samples)
                if self._usb1 is not None and self._v8.status == 0 and self._usb1.feed(samples):
                    self._log("legacy V.22 USB1 detected: sustained 2250 Hz")
                    self._choose("v22")
                    return b""
                # ANS phase reversals can look like ANSam. A dominant AC
                # pair can resolve an unanswered CM attempt, but cannot
                # override a completed menu exchange.
                awaiting_legacy = (self._v8.status == 0 or
                                   (self._v8.status == 1 and self._v8._stage == "cm"))
                if self._ac is not None and awaiting_legacy and self._ac.feed(samples):
                    self._log("legacy V.32 AC detected: sustained 600/3000 Hz")
                    self._choose("v32")
                    self.engine.feed_pcm(pcm)
                    self._sync_engine_events()
                    return b""
                self._step_v8()
                return b""
            if self._tone_gate is not None:
                import struct  # noqa: PLC0415

                samples = self._tone_gate.process(list(struct.unpack(f"<{len(pcm) // 2}h", pcm)))
                pcm = struct.pack(f"<{len(samples)}h", *samples)
            out = self.engine.feed_pcm(pcm)
            self._sync_engine_events()
            self._check_b103_fallback()
            if self._legacy_probe_at is not None:
                if self.engine.state not in ("gap", "ac"):
                    self._legacy_probe_at = None
                    self._log("legacy V.32 probe accepted: caller answered AC")
                    if self._queued:
                        self.engine.feed_bytes(bytes(self._queued))
                        self._queued.clear()
                elif self._t - self._legacy_probe_at >= 16000 and self.allow & V8_MOD_V22:
                    self._log("legacy V.32 probe unanswered: falling back to V.22/V.22bis")
                    self._choose("v22")
            return out

    def feed_bytes(self, data: bytes) -> None:
        with self._lock:
            if self._failed:
                return
            if self.engine is None or self._legacy_probe_at is not None:
                self._queued += data
            else:
                self.engine.feed_bytes(data)

    def __getattr__(self, name: str):
        eng = self.__dict__.get("engine")
        if eng is not None:
            return getattr(eng, name)
        defaults = {"connected": False, "bit_rate": 0, "v42_mode": "off", "v42_lost": False, "v42_compression": None,
                    "v42_resolved": False, "rx_signal_power_dbm0": -90.0, "pending_bits": 0}
        if name in defaults:
            value = defaults[name]
            return (lambda: value) if name == "pending_bits" else value
        if name == "is_settled":
            return lambda: False
        raise AttributeError(name)

    def close(self) -> None:
        with self._rx_lock, self._lock:
            if self._v8 is not None:
                self._v8.close()
            if self.engine is not None and hasattr(self.engine, "close"):
                self.engine.close()
