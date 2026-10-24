"""Automatic line-mode selection (our own code): V.8 first, then common modulation or legacy V.32/V.22.

``V8Chain`` presents the same surface as the single engines (tx_samples, feed_pcm, feed_bytes, pending_bits, bit_rate,
is_settled, v42_*...). It runs V.8 (ITU-T V.8) offering V.34 / V.32bis / V.22bis, and when the V.8 exchange finishes it
builds the engine for the highest modulation both ends named and hands the line over to it, forwarding calls from then on.
Legacy callers recognize V.32 AC or V.22 USB1. Legacy answerers probe V.32 AC
for two seconds after V.8 times out, then fall back to V.22 when it is allowed. Bell 103 / V.21 legacy partners are not covered by the chain.
"""
import threading
import os
import time

from v8 import V8_MOD_V22, V8_MOD_V32, V8_MOD_V34, V8Negotiator
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
                 v22_min_rate: int = 1200, v34_min_rate: int = 2400):
        self.calling = calling_party
        self.v34_rate, self.v34_min_rate, self.v32_rates, self.v22_rate = v34_rate, v34_min_rate, tuple(v32_rates), v22_rate
        self.v22_min_rate = v22_min_rate
        self.allow = allow
        self.patience = patience
        self.debug = debug
        self.events: list = []
        self.engine = None
        self._engine_events_seen = 0
        self.kind = "v8"
        self._lock = threading.RLock()
        self._rx_lock = threading.RLock()
        self._trace_tx = os.environ.get('EL_MODEM_TIMING_TRACE') == '1'
        self._tx_timing = None
        self._v8 = V8Negotiator(calling_party, modulations=allow)
        self._t = 0
        self._queued = bytearray()
        self._queued_bits: list = []
        self._legacy_probe_at = None
        self._tone_gate: AnsGate | None = None      # set when plain ANS (not ANSam) ended V.8: hide the 2100 Hz tone
        self._choose_t = 0
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

    def _choose(self, kind: str, *, probe=False) -> None:
        if self.engine is not None and hasattr(self.engine, "close"):
            self.engine.close()
        self._legacy_probe_at = self._t if probe else None
        self._choose_t = self._t
        self.kind = kind
        self._log(f"line mode: {kind}")
        calling = self.calling
        if kind == "v34":
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

    def _pick(self, mods: int) -> str | None:
        mods &= self.allow
        for bit, kind in ((V8_MOD_V34, "v34"), (V8_MOD_V32, "v32"), (V8_MOD_V22, "v22")):
            if mods & bit:
                return kind
        return None

    def _step_v8(self) -> None:
        st = self._v8.status
        # Plain ANS can start V.22 at once only when no legacy V.32 AC is possible; otherwise the AC detector
        # keeps listening (a V.32 answerer sends ANS, then AC) and V.22 waits for the patience timeout.
        plain_now = st == 0 and self._v8.plain_ans and self._ac is None
        if st == 2:
            kind = self._pick(self._v8.result["modulations"])
            self._log(f"V.8 complete: {self._v8.result}")
            if kind is not None:
                self._choose(kind)
            elif self.allow & V8_MOD_V22:
                # No common modulation: fall back to V.22/V.22bis as before, rather than leaving the line silent.
                self._log("V.8 completed without an allowed common modulation: falling back to V.22/V.22bis")
                self._choose("v22")
            elif not self._no_common_logged:
                self._no_common_logged = True       # _step_v8 runs on every receive block; log once
                self._log("V.8 completed without an allowed common modulation and V.22 is not allowed")
        elif st in (3, 4) or plain_now or (st == 0 and self._t / 8000.0 > self.patience):
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
    def tx_samples(self, count: int) -> list[int]:
        started = time.monotonic() if self._trace_tx else None
        cpu_started = time.thread_time() if self._trace_tx else None
        with self._lock:
            acquired = time.monotonic() if self._trace_tx else None
            cpu_acquired = time.thread_time() if self._trace_tx else None
            try:
                self._t += count
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
