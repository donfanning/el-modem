"""ITU-T V.8 start-up negotiation (11/2000 text): ANSam, CI/CM/JM/CJ over V.21 FSK, written from the Recommendation.

``V8Negotiator`` has the surface of the SpanDSP-backed class it replaces (spandsp_v8.V8Negotiator): create it for the
calling or answering side, feed received 8 kHz samples with ``rx`` and take samples to send from ``tx``; ``status``
becomes 2 ("v8-call") when the CM/JM/CJ exchange is complete (``result`` then holds the negotiated call function,
modulation modes and protocols), 3 ("non-v8-call") when the far end does not do V.8, 4 on failure.

Implemented: ANSam with 15 Hz amplitude modulation and 450 ms phase reversals (§7.2); CM/JM/CJ with the §5-6 octet
coding and 2-identical-sequences rules (§7.3, 7.4, 8); CI as an optional call signal (§7.1); the call function,
modulation modes, protocol and PSTN access categories. LAPM is reported only when both peers indicate it. The DCE
uses that common V.8 indication to bypass V.42 ODP/ADP detection and proceed to LAPM/XID; without it, V.42 detection
remains the alternate negotiation path (§6.4 and §7.2.1 of V.8). Not implemented: the PCM modem availability and
non-standard facility categories are carried through (V.90/V.92 flag) but not interpreted, T.66, V.92 quick-connect.

Credit: the signal definitions are ITU-T's; the timings that real answerers tolerate (Te, 75 ms gaps) were checked against
SpanDSP's v8.c (LGPL, Steve Underwood) and the Smart Link modem as black-box peers. No code was copied from either.
"""

from __future__ import annotations

import math
import os

try:
    import numpy as np
except ImportError:
    # numpy ships in the image, but the engines and the test suites must stay importable on a
    # bare checkout: without it this module degrades to "V.8 unavailable" rather than raising,
    # so a caller can ask available() instead of catching ImportError at every import site.
    np = None  # type: ignore[assignment]

FS = 8000
PLAIN_ANS_S = 2.0       # unmodulated 2100 Hz for this long is plain ANS, not ANSam (V.25 tone lasts 2.6 s or more)
BAUD = 300.0
# modulation flags (kept equal to spandsp_v8's constants so the engines need no change)
V8_MOD_V21, V8_MOD_V22, V8_MOD_V23, V8_MOD_V32 = 1 << 1, 1 << 2, 1 << 4, 1 << 9
V8_MOD_V34, V8_MOD_V90 = 1 << 11, 1 << 12
V8_CALL_V_SERIES = 6
V8_PROTOCOL_NONE, V8_PROTOCOL_LAPM_V42 = 0, 1
V8_LAPM_MODULATIONS = V8_MOD_V22 | V8_MOD_V32 | V8_MOD_V34
def available() -> bool:
    return np is not None


STATUS_NAMES = {0: "in-progress", 1: "v8-offered", 2: "v8-call", 3: "non-v8-call", 4: "failed"}

V21_L = (980.0, 1180.0)                       # channel 1 (call): mark, space
V21_H = (1650.0, 1850.0)                      # channel 2 (answer)
SYNC_CM_JM = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
SYNC_CI = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
PREAMBLE = [1] * 10

# call function codes (§6.1): (b5, b6, b7) packed into the option bits; 6 = data (V-series modem)
_CALLF_BITS = {6: 0b110, 1: 0b001, 2: 0b010, 3: 0b100}


def _octet_bits(value: int) -> list[int]:
    return [0] + [(value >> i) & 1 for i in range(8)] + [1]


def call_function_octet(cf: int = V8_CALL_V_SERIES) -> int:
    """Category tag 1 (b0=1), b5..b7 option bits; data = b5 0, b6 1, b7 1 -> 0xC1."""
    opts = {6: (0, 1, 1), 1: (1, 0, 0), 2: (0, 1, 0), 3: (0, 0, 1)}.get(cf, (0, 1, 1))
    return 0x01 | (opts[0] << 5) | (opts[1] << 6) | (opts[2] << 7)


def decode_call_function(octet: int) -> int:
    opts = ((octet >> 5) & 1, (octet >> 6) & 1, (octet >> 7) & 1)
    return {(0, 1, 1): 6, (1, 0, 0): 1, (0, 1, 0): 2, (0, 0, 1): 3}.get(opts, 0)


def modulation_octets(mods: int, pcm: bool = False) -> list[int]:
    """modn0 (tag 5: b5 PCM available, b6 V.34 duplex), modn1 and modn2 extension octets (§6.2, Table 4)."""
    n0 = 0x05 | ((1 if pcm else 0) << 5) | ((1 if mods & V8_MOD_V34 else 0) << 6)
    n1 = 0x10 | (1 if mods & V8_MOD_V32 else 0) | ((1 if mods & V8_MOD_V22 else 0) << 1)
    n2 = 0x10 | ((1 if mods & V8_MOD_V23 else 0) << 2) | ((1 if mods & V8_MOD_V21 else 0) << 7)
    return [n0, n1, n2]


def decode_modulations(octets: list[int]) -> int:
    mods = 0
    for i, o in enumerate(octets):
        if i == 0 and (o & 0x1F) == 0x05:
            mods |= V8_MOD_V34 if o & 0x40 else 0
        elif i == 1:
            mods |= (V8_MOD_V32 if o & 0x01 else 0) | (V8_MOD_V22 if o & 0x02 else 0)
        elif i == 2:
            mods |= (V8_MOD_V23 if o & 0x04 else 0) | (V8_MOD_V21 if o & 0x80 else 0)
    return mods


def build_message(call_function: int, mods: int, protocols: int = 0, pstn_access: bool = False,
                  pcm: int = 0) -> list[int]:
    """One CM/JM sequence: 10 ones, the 10 synchronisation bits, then the information octets."""
    octets = [call_function_octet(call_function)] + modulation_octets(mods, bool(pcm))
    if protocols & V8_PROTOCOL_LAPM_V42:
        octets.append(0x0A | (1 << 5))                              # protocol category, LAPM (b5)
    if pstn_access or pcm:
        octets.append(0x0D | (0 << 5) | (0 << 6) | (0 << 7))        # PSTN access: analogue, not cellular
    if pcm:
        octets.append(0x07 | (1 << 5))                              # PCM modem availability: V.90/V.92 analogue
    bits = list(PREAMBLE) + list(SYNC_CM_JM)
    for o in octets:
        bits += _octet_bits(o)
    return bits


def parse_sequence(bits: list[int]) -> list[int] | None:
    """Octets of one sequence from the bits after the synchronisation word; None if the framing is wrong."""
    octets, i = [], 0
    while i + 10 <= len(bits):
        if bits[i] != 0:                                              # a one where a start bit belongs: next sequence
            break
        if bits[i + 9] != 1:
            return None
        octets.append(sum(bits[i + 1 + j] << j for j in range(8)))
        i += 10
    return octets or None


class _Fsk:
    """Phase-continuous V.21 FSK source at 300 baud with fractional bit timing, fed with a bit generator."""

    def __init__(self, tones: tuple[float, float], amp: float):
        self.tones, self.amp = tones, amp
        self.phase = 0.0
        self.bits: list[int] = []
        self.frac = 0.0
        self.cur = 1
        self.consumed = 0                                              # bits taken from ``bits`` so far

    def read(self, n: int) -> np.ndarray:
        out = np.empty(n)
        spb = FS / BAUD
        i = 0
        while i < n:
            if self.frac <= 0:
                if self.bits:
                    self.cur = self.bits.pop(0)
                    self.consumed += 1
                else:
                    self.cur = 1
                self.frac += spb
            take = min(n - i, max(1, int(math.ceil(self.frac))))
            f = self.tones[0] if self.cur else self.tones[1]
            ph = self.phase + 2 * math.pi * f * np.arange(1, take + 1) / FS
            out[i:i + take] = self.amp * np.sin(ph)
            self.phase = float(ph[-1]) % (2 * math.pi)
            self.frac -= take
            i += take
        return out


class _AnsAm:
    def __init__(self, amp: float):
        self.amp, self.n = amp, 0

    def read(self, n: int) -> np.ndarray:
        t = (self.n + np.arange(n)) / FS
        rev = np.floor(t / 0.450).astype(int) % 2                      # 180 degree phase reversal every 450 ms
        env = 1.0 + 0.2 * np.sin(2 * math.pi * 15.0 * t)               # 0.8 .. 1.2 of the average amplitude
        out = self.amp * env * np.sin(2 * math.pi * 2100.0 * t + math.pi * rev)
        self.n += n
        return out


class _FskRx:
    """Bit-level V.21 receiver: finds CM/JM sequences (or CJ) in a rolling buffer by trying several bit timings."""

    PHASES = 8

    def __init__(self, tones: tuple[float, float]):
        self.tones = tones
        self.buf = np.zeros(0)
        self.base = 0                                                 # absolute index of buf[0]

    def feed(self, x: np.ndarray, keep: int = 3 * FS) -> None:
        self.buf = np.concatenate([self.buf, x])
        if len(self.buf) > keep:
            cut = len(self.buf) - keep
            self.buf = self.buf[cut:]
            self.base += cut

    def _energies(self) -> tuple[np.ndarray, np.ndarray]:
        n = np.arange(len(self.buf)) + self.base
        w = int(round(FS / BAUD))
        k = np.ones(w) / w
        out = []
        for f in self.tones:
            z = self.buf * np.exp(-2j * math.pi * f * n / FS)
            out.append(np.abs(np.convolve(z, k, mode="same")) ** 2)
        return out[0], out[1]

    def bitstreams(self):
        """Yield (phase, bits) for every sampling phase."""
        if len(self.buf) < 2 * FS // 10:
            return
        em, es = self._energies()
        spb = FS / BAUD
        for p in range(self.PHASES):
            t0 = spb / 2 + p * spb / self.PHASES
            idx = (t0 + np.arange(int((len(self.buf) - t0) / spb)) * spb).astype(int)
            idx = idx[idx < len(self.buf)]
            yield p, (em[idx] > es[idx]).astype(int).tolist()

    def sequences(self) -> list[list[int]]:
        """Every complete CM/JM sequence (list of octets) seen at the best phase."""
        best: list[list[int]] = []
        want = PREAMBLE + SYNC_CM_JM
        for _, bits in self.bitstreams():
            found, i = [], 0
            while i + 20 <= len(bits):
                if bits[i:i + 20] == want:
                    j = i + 20
                    octets = parse_sequence(bits[j:])
                    if octets:
                        found.append(octets)
                        i = j + 10 * len(octets)
                        continue
                i += 1
            if len(found) > len(best):
                best = found
        return best

    def cj(self) -> bool:
        """CJ: three consecutive octets of zeros with start and stop bits (§3.5)."""
        want = _octet_bits(0) * 3
        for _, bits in self.bitstreams():
            for i in range(len(bits) - 30):
                if bits[i:i + 30] == want and (i == 0 or bits[i - 1] == 1):
                    return True
        return False


class V8Negotiator:
    def __init__(self, calling_party: bool, modulations: int = V8_MOD_V32, protocols: int = V8_PROTOCOL_NONE,
                 send_ci: bool = False, call_function: int = V8_CALL_V_SERIES, amplitude: float = 2500.0,
                 ans_delay: float = 0.2):
        self.calling = calling_party
        self.mods, self.protocols, self.cf = modulations, protocols, call_function
        if not (self.mods & V8_LAPM_MODULATIONS):
            self.protocols &= ~V8_PROTOCOL_LAPM_V42
        self.send_ci = send_ci
        self.status = 0
        self.result: dict | None = None
        self.single_jm = False
        self.amp = amplitude
        self.t = 0                                                    # samples consumed by tx()
        self.rx_t = 0
        self._am_votes = 0
        self._tx = None                                               # current transmit source
        self._rx = _FskRx(V21_H if calling_party else V21_L)
        self._rx_cj = _FskRx(V21_L)
        self._stage = "wait_ans" if calling_party else "silence"
        self._stage_t = 0
        self._tone: list[float] = []                                  # ANSam analysis blocks (calling side)
        self._tone_n = 0
        self._ans_seen_at: int | None = None
        self._plain_since: int | None = None
        self.plain_ans = False
        self._jm: list[int] | None = None
        self._seqs_checked = 0
        self._ans_delay = ans_delay
        self._cj_poll = 0
        if calling_party:
            self._tx = None                                           # silence until ANSam
        else:
            self._tx = None

    # -- public --------------------------------------------------------------------------------------------
    @property
    def status_name(self) -> str:
        return STATUS_NAMES.get(self.status, str(self.status))

    def close(self) -> None:
        self._tx = None

    def tx(self, count: int) -> list[int]:
        out = np.zeros(count)
        if self._tx is not None and self.status in (0, 1):
            out = self._tx.read(count)
        self.t += count
        self._advance_tx()
        return np.clip(np.round(out), -32768, 32767).astype(int).tolist()

    def rx(self, samples) -> int:
        x = np.asarray(samples, dtype=float)
        self.rx_t += len(x)
        if self.calling:
            self._rx_calling(x)
        else:
            self._rx_answering(x)
        return 0

    # -- transmit sequencing ---------------------------------------------------------------------------------
    def _set(self, stage: str) -> None:
        self._stage, self._stage_t = stage, self.t

    def _advance_tx(self) -> None:
        st = self._stage
        if self.calling:
            if st == "te" and self.t - self._stage_t >= int(0.6 * FS):          # Te >= 0.5 s of silence (§8.1.1)
                src = _Fsk(V21_L, self.amp)
                msg = build_message(self.cf, self.mods, self.protocols)
                src.bits = msg * 400
                self._tx = src
                self._set("cm")
            elif st == "cj_pending" and self._tx is not None and not self._tx.bits:
                self._tx = None
                # V.8 §8.1.2 requires 75 ± 5 ms with no signal after CJ before
                # the selected modulation's call signal. V.34 has its own
                # startup timing, but lower-rate choices such as V.21 do not;
                # keep the gap in V.8 so every selected modulation gets it.
                self._set("post_cj")
            elif st == "post_cj" and self.t - self._stage_t >= int(0.075 * FS):
                self._finish()
            elif st == "wait_ans" and self.send_ci and self._tx is None and self.t > FS:
                src = _Fsk(V21_L, self.amp)
                ci = list(PREAMBLE) + list(SYNC_CI) + _octet_bits(call_function_octet(self.cf))
                src.bits = ci * 4 + [1] * int(0.5 * FS / BAUD * 8)
                self._tx = src
        else:
            if st == "silence" and self.t >= int(self._ans_delay * FS):
                self._tx = _AnsAm(self.amp)
                self._set("ansam")
            elif st == "ansam" and self.t - self._stage_t >= int(5.0 * FS):
                self.status = 3                                                  # no CM within 5 s: not a V.8 call
            elif st == "jm_cj_wait":
                pass
            elif st == "post_cj" and self.t - self._stage_t >= int(0.075 * FS):
                self._finish()

    def _finish(self) -> None:
        self._tx = None
        self.status = 2
        self._set("done")

    # -- calling side -----------------------------------------------------------------------------------------------
    def _ansam_present(self) -> tuple[bool, bool]:
        """(tone present, amplitude modulated) over the recent blocks."""
        blk = 80
        need = blk * 40
        if len(self._rx.buf) < need:
            return False, False
        x = self._rx.buf[-need:]
        n = np.arange(len(x)) + self._rx.base + len(self._rx.buf) - len(x)
        z = x * np.exp(-2j * math.pi * 2100.0 * n / FS)
        env = np.abs(z[:len(z) // blk * blk].reshape(-1, blk).sum(axis=1)) / blk * 2
        tot = np.sqrt(np.mean(x ** 2) * 2) + 1e-9
        tone = float(np.median(env)) / tot > 0.7 and tot > 40
        if not tone:
            self._am_votes = 0
            return False, False
        depth = (float(np.percentile(env, 90)) - float(np.percentile(env, 10))) / (float(np.mean(env)) + 1e-9)
        # A window that straddles silence -> tone looks deeply modulated (p10 near zero) without being AM.
        # ANSam's envelope never falls near zero, so such a window is not a vote for AM.
        onset = float(np.percentile(env, 10)) < 0.3 * float(np.mean(env))
        am_now = depth > 0.15 and not onset
        # ANSam keeps its 15 Hz AM for at least ~2.6 s; a momentary AM-like
        # swell on plain ANS must not trigger V.8. Require several consecutive
        # moderately-spaced positive windows before concluding "V.8 present".
        if am_now:
            self._am_votes += 1
        else:
            self._am_votes = 0
        return True, self._am_votes >= 3

    def _rx_calling(self, x: np.ndarray) -> None:
        self._rx.feed(x)
        self._rx_cj.feed(x)
        st = self._stage
        if st == "wait_ans":
            present, am = self._ansam_present()
            if present and am:
                self._ans_seen_at = self.rx_t
                self._plain_since = None
                self._tx = None
                self.status = 1
                self._set("te")
            elif present:
                # Steady 2100 Hz without AM is plain ANS (V.25): no V.8 partner. ANSam keeps its AM for 5 s, so
                # PLAIN_ANS_S of unmodulated tone is decisive. The chain decides what to do with it (see autochain).
                if self._plain_since is None:
                    self._plain_since = self.rx_t
                elif self.rx_t - self._plain_since >= int(PLAIN_ANS_S * FS):
                    self.plain_ans = True
            else:
                self._plain_since = None
            if self.rx_t > 12 * FS and self.status == 0:
                self.status = 3
        elif st == "cm":
            if self.rx_t - self._cj_poll < FS // 10:
                return
            self._cj_poll = self.rx_t
            seqs = self._rx.sequences()
            for a, b in zip(seqs, seqs[1:]):
                if a == b:                                                     # two identical JM sequences (§8.1.2)
                    self._on_jm(a)
                    return
            if seqs and self._answerer_left_v8():
                self.single_jm = True
                self._on_jm(seqs[-1])
                return
            if self.rx_t - self._stage_t > 0 and self.t - self._stage_t > int(8 * FS):
                self.status = 4

    def _answerer_left_v8(self) -> bool:
        """One valid JM was decoded, but no second, and the answerer now sends V.34 tone A (2400 Hz) instead.

        A pool answerer (dialup.world, 2026-10-07) sent a single JM and went straight to tone A without waiting for
        CJ; waiting for the second identical JM (§8.1.2) then never ends. Treat it as JM received: send CJ and
        carry on. A V.8 answerer that repeats JM is unaffected (two sequences complete it first); V34_V8_SINGLE_JM=0
        disables this.
        """
        if os.environ.get("V34_V8_SINGLE_JM", "1") == "0":
            return False
        n = int(0.4 * FS)
        if len(self._rx.buf) < n:
            return False
        x = self._rx.buf[-n:]
        tone = np.abs(np.sum(x * np.exp(-2j * math.pi * 2400.0 * np.arange(n) / FS))) / n * 2
        rms = math.sqrt(float(np.mean(x ** 2)) * 2)
        return rms > 40 and tone / (rms + 1e-9) > 0.8

    def _on_jm(self, octets: list[int]) -> None:
        cf = decode_call_function(octets[0])
        mods = decode_modulations(octets[1:4]) & self.mods
        peer_protocols = V8_PROTOCOL_LAPM_V42 if any(
            (o & 0x1F) == 0x0A and (o >> 5) & 1 for o in octets
        ) else 0
        # JM describes the common protocol set, not an independent peer offer.
        # Never report LAPM unless this caller also placed it in its CM.
        prot = (self.protocols & peer_protocols
                if mods & V8_LAPM_MODULATIONS else V8_PROTOCOL_NONE)
        pcm = any((o & 0x1F) == 0x07 for o in octets)
        self.result = {"status": 2, "call_function": cf, "modulations": mods, "protocols": prot, "tone": 5,
                       "pcm": pcm}
        # finish the octet in progress (§8.1.2), then CJ = 3 octets of zeros, then 75 ms of silence
        src = self._tx if isinstance(self._tx, _Fsk) else _Fsk(V21_L, self.amp)
        msg_len = len(build_message(self.cf, self.mods, self.protocols))
        pos = src.consumed % msg_len
        src.bits = src.bits[:0 if pos <= 20 else (10 - (pos - 20) % 10) % 10]
        src.bits += _octet_bits(0) * 3
        self._tx = src
        self._set("cj_pending")

    # -- answering side ------------------------------------------------------------------------------------------------
    def _rx_answering(self, x: np.ndarray) -> None:
        self._rx.feed(x)
        st = self._stage
        if st == "ansam":
            if self.rx_t - self._cj_poll < FS // 10:
                return
            self._cj_poll = self.rx_t
            seqs = self._rx.sequences()
            for a, b in zip(seqs, seqs[1:]):
                if a == b:                                                     # two identical CM sequences (§7.4)
                    self._on_cm(a)
                    return
        elif st == "jm":
            if self.rx_t - self._cj_poll < FS // 20:
                return
            self._cj_poll = self.rx_t
            if self._rx.cj():
                self._tx = None
                self._set("post_cj")

    def _on_cm(self, octets: list[int]) -> None:
        cf = decode_call_function(octets[0])
        cm_mods = decode_modulations(octets[1:4])
        common = cm_mods & self.mods
        prot = V8_PROTOCOL_LAPM_V42 if (self.protocols & V8_PROTOCOL_LAPM_V42 and
                                         common & V8_LAPM_MODULATIONS and
                                         any((o & 0x1F) == 0x0A and (o >> 5) & 1 for o in octets)) else 0
        self.result = {"status": 2, "call_function": cf, "modulations": common, "protocols": prot, "tone": 5,
                       "pcm": any((o & 0x1F) == 0x07 for o in octets)}
        src = _Fsk(V21_H, self.amp)
        src.bits = build_message(cf, common, prot) * 400
        self._tx = src
        self.status = 1
        self._rx = _FskRx(V21_L)                                               # now listen for CJ
        self._set("jm")
