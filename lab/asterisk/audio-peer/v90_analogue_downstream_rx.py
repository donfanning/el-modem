"""Streaming receiver for G.711 V.90 digital Phase 3 codewords.

This is the transparent digital-side boundary, not a PCM demodulator. It
recognizes S_d/S'_d, tracks the known TRN1d GPC sequence, and accepts Jd only
after its sync, reserved fields, and CRC all validate. The caller's analogue
DSP still needs to recover exact G.711 octets from RTP/PCM before using it.
"""

from __future__ import annotations

from collections import deque

from v90_dil import g711_ucode_codewords
from v90_digital_tx import sd_codewords, sd_prime_codewords
import v34_info


def parse_jd_bits(bits) -> dict | None:
    """Parse one 72-bit Table 13 Jd frame, or return None when invalid."""
    values = [int(bit) for bit in bits]
    if len(values) != 72 or any(bit not in (0, 1) for bit in values):
        return None
    if values[:17] != [1] * 17 or values[17] != 0:
        return None
    if values[34] != 0 or values[51] != 0 or values[68:72] != [0] * 4:
        return None
    if any(values[41:47]):
        return None
    lookahead = values[49] | (values[50] << 1)
    if not 1 <= lookahead <= 3:
        return None
    information = values[18:34] + values[35:51]
    received_crc = sum(bit << index for index, bit in enumerate(values[52:68]))
    if received_crc != v34_info.crc16(information):
        return None
    mask = sum(bit << index for index, bit in enumerate(values[18:34]))
    mask |= sum(bit << (16 + index)
                for index, bit in enumerate(values[35:47]))
    return {
        "rate_mask": mask,
        "phase3_16point": bool(values[47]),
        "renegotiation_16point": bool(values[48]),
        "max_lookahead": lookahead,
        "bits": tuple(values),
    }


class V90AnalogueDownstreamPhase3Receiver:
    """Consume arbitrary chunks of negotiated-law G.711 octets.

    Event dictionaries include ``event`` and an absolute 8 kHz ``at`` sample.
    ``sd_transition`` marks the first S'_d codeword. ``jd`` carries the parsed
    Jd frame. After the caller reports its S response, the receiver validates
    J'_d and returns exactly the requested number of following DIL octets.
    """

    SEARCH_SD = "search-sd"
    READ_SD_PRIME = "read-sd-prime"
    READ_TRN1D = "read-trn1d"
    JD_FOUND = "jd-found"
    READ_DIL = "read-dil"
    PHASE3_DONE = "phase3-done"
    # Digital Jd can continue for the measured round-trip delay after the
    # analogue caller transmits S. The protocol timer in the caller controller
    # bounds the wait; this receiver must not impose a short frame-count limit.

    def __init__(self, uinfo: int, law: str, *, trn1d_min_symbols: int = 2040):
        if isinstance(trn1d_min_symbols, bool) or trn1d_min_symbols < 2040:
            raise ValueError("TRN1d minimum must be at least 2040 symbols")
        self.uinfo = int(uinfo)
        self.law = law.lower()
        self.state = self.SEARCH_SD
        table = g711_ucode_codewords(self.law)
        self._sign_by_octet = {}
        for code in table:
            code = int(code)
            self._sign_by_octet[code] = 1
            self._sign_by_octet[code ^ 0x80] = 0
        self._sd = sd_codewords(self.uinfo, self.law)
        self._sd_prime = sd_prime_codewords(self.uinfo, self.law)
        self._mark_window: deque[int] = deque(maxlen=len(self._sd))
        self._sd_prime_index = 0
        self._trn_min = int(trn1d_min_symbols)
        self._trn_count = 0
        self._previous_sign: int | None = None
        self._gpc_history = [0] * 23
        self._jd_candidates: list[dict] = []
        self._jd_history: list[int] | None = None
        self._jd_previous_sign: int | None = None
        self._post_jd_bits = 0
        self._caller_s_armed = False
        self._caller_s_sent_at: int | None = None
        self._jd_symbols_to_finish: int | None = None
        self._jprime_zero_count = 0
        self._jprime_frame_bits: list[int] = []
        self._jprime_frames_waited = 0
        self._expected_dil: int | None = None
        self._dil_received = 0
        self._events: list[dict] = []
        self._next_sample: int | None = None
        self.codeword_errors = 0
        # Keep the most recent parser failures for startup qualification. This
        # is bounded so a malformed line cannot grow the diagnostic state
        # without limit, and it does not affect decoding decisions.
        self.codeword_error_events: deque[dict] = deque(maxlen=16)

    def _matches_mark_codeword(self, expected: int, received: int) -> bool:
        """Compare S_d marker octets, allowing PCMU's two zero encodings.

        V.90 defines both +0 and -0 in the S_d/S'_d patterns. A μ-law RTP
        path can canonicalize negative zero (0x7f) to positive zero (0xff)
        without changing the decoded waveform. Keep this tolerance scoped to
        the two known marker patterns; later training/codeword parsing still
        uses the exact negotiated codeword mapping.
        """
        if expected == received:
            return True
        return (self.law == "pcmu"
                and expected in (0x7f, 0xff)
                and received in (0x7f, 0xff))

    def _matches_mark(self, expected: bytes, received: bytes) -> bool:
        return (len(expected) == len(received)
                and all(self._matches_mark_codeword(want, got)
                        for want, got in zip(expected, received)))

    def feed(self, codewords: bytes | bytearray | memoryview, *, at: int,
             sequence_at: int | None = None, on_event=None
             ) -> list[dict]:
        """Feed G.711 octets with protocol time `at` and optional stream index.

        RTP clock fitting can map adjacent receive chunks to media timestamps
        whose integer difference alternates between 159 and 160 samples. That
        is useful for protocol event scheduling, but the octet stream itself
        remains contiguous. `sequence_at` carries its receive-clock position
        for the continuity guard while events continue to use media-clock
        `at`.
        """
        start = int(at)
        stream_start = start if sequence_at is None else int(sequence_at)
        if self._next_sample is not None and stream_start != self._next_sample:
            raise ValueError("G.711 input chunks must be contiguous on the sample clock")
        self._next_sample = stream_start + len(codewords)
        self._events = []
        dil_bytes = bytearray()
        phase4_bytes = bytearray()
        for offset, raw in enumerate(codewords):
            sample = start + offset
            octet = int(raw)
            if self.state == self.PHASE3_DONE:
                # The final caller S/S-bar can end in the middle of an RTP
                # payload. Preserve the remaining exact-law octets for the
                # Phase 4 R-i/MP receiver instead of silently dropping them.
                phase4_bytes.append(octet)
                continue
            if self.state == self.SEARCH_SD:
                self._mark_window.append(octet)
                if (len(self._mark_window) == len(self._sd)
                        and self._matches_mark(self._sd,
                                               bytes(self._mark_window))):
                    self.state = self.READ_SD_PRIME
                    self._sd_prime_index = 0
                    self._emit({"event": "sd_mark",
                                "at": sample + 1 - len(self._sd)}, on_event)
                continue
            if self.state == self.READ_SD_PRIME:
                if not self._matches_mark_codeword(
                        int(self._sd_prime[self._sd_prime_index]), octet):
                    # A malformed mark is not a transition; resume bounded
                    # search from the most recent 384 symbols.
                    self._record_codeword_error(
                        "invalid-sd-prime-codeword", sample,
                        expected=int(self._sd_prime[self._sd_prime_index]),
                        received=octet)
                    self.state = self.SEARCH_SD
                    self._mark_window.clear()
                    self._mark_window.append(octet)
                    continue
                if self._sd_prime_index == 0:
                    self._emit({"event": "sd_transition", "at": sample},
                               on_event)
                self._sd_prime_index += 1
                if self._sd_prime_index == len(self._sd_prime):
                    self.state = self.READ_TRN1D
                continue
            if self.state == self.READ_TRN1D:
                sign = self._sign_by_octet.get(octet)
                if sign is None:
                    self._record_codeword_error(
                        "invalid-trn1d-codeword", sample, received=octet)
                    self._reset_to_search()
                    continue
                # Keep a candidate decoder at each possible TRN1d end. The
                # first Jd bits can resemble continued training, so declaring
                # the boundary from one sign transition is unsafe. A valid
                # sync+CRC frame selects the correct boundary retrospectively.
                if self._trn_count >= self._trn_min:
                    self._jd_candidates.append({
                        "start": sample,
                        "previous_sign": self._previous_sign,
                        "history": self._gpc_history.copy(),
                        "bits": [],
                    })
                    self._advance_jd_candidates(sign, sample, on_event)
                self._consume_training_symbol(sign)
                self._trn_count += 1
                if self._trn_count == self._trn_min:
                    self._emit({"event": "trn1d_min_complete",
                                "at": sample + 1}, on_event)
                continue
            if self.state == self.JD_FOUND:
                self._receive_jd_continuation(octet, sample, on_event)
                continue
            if self.state == self.READ_DIL:
                dil_bytes.append(octet)
                self._dil_received += 1
                if self._dil_received == self._expected_dil:
                    self.state = self.PHASE3_DONE
                continue
        if dil_bytes:
            self._emit({"event": "dil_codewords",
                        "at": start + len(codewords),
                        "codewords": bytes(dil_bytes),
                        "received": self._dil_received,
                        "complete": self.state == self.PHASE3_DONE}, on_event)
        if phase4_bytes:
            self._emit({"event": "phase4_codewords",
                        "at": start + len(codewords),
                        "codewords": bytes(phase4_bytes)}, on_event)
        return list(self._events)

    def _emit(self, event: dict, on_event=None) -> None:
        self._events.append(event)
        if on_event is not None:
            on_event(event)

    def set_expected_dil(self, symbols: int) -> None:
        """Set the caller-requested DIL length before receiving J′d."""
        value = int(symbols)
        if value < 0:
            raise ValueError("expected DIL length cannot be negative")
        if self.state in (self.READ_DIL, self.PHASE3_DONE):
            raise RuntimeError("DIL length must be set before J′d")
        self._expected_dil = value

    def notify_caller_s_sent(self, at: int) -> None:
        """Arm J′d recognition where caller S starts transmitting."""
        if self.state != self.JD_FOUND or self._jd_history is None:
            raise RuntimeError("caller S can only follow a valid Jd")
        if self._caller_s_armed:
            raise RuntimeError("caller S was already reported")
        self._caller_s_armed = True
        self._caller_s_sent_at = int(at)
        phase = self._post_jd_bits % 72
        # The answerer completes its in-flight Jd. At an exact boundary it
        # starts another frame before acting on the S event.
        self._jd_symbols_to_finish = 72 - phase
        self._jprime_zero_count = 0
        self._jprime_frame_bits = []
        self._jprime_frames_waited = 0

    def _receive_jd_continuation(self, octet: int, sample: int,
                                 on_event=None) -> None:
        sign = self._sign_by_octet.get(octet)
        if sign is None or self._jd_history is None:
            self._record_codeword_error(
                "invalid-jd-continuation-codeword", sample, received=octet)
            self._reset_to_search()
            return
        scrambled = (sign if self._jd_previous_sign is None
                     else sign ^ self._jd_previous_sign)
        self._jd_previous_sign = sign
        bit = scrambled ^ self._jd_history[17] ^ self._jd_history[22]
        self._jd_history = [scrambled, *self._jd_history[:22]]
        self._post_jd_bits += 1
        if not self._caller_s_armed:
            return
        if self._jd_symbols_to_finish:
            self._jd_symbols_to_finish -= 1
            return
        # A digital answerer may already have a complete Jd frame buffered
        # beyond the frame intersected by caller S. Continue across valid Jd
        # frames and accept J′d only at a frame boundary; packet scheduling
        # must not make the caller miss its 12-zero terminator.
        if self._jprime_frame_bits:
            self._jprime_frame_bits.append(bit)
            if len(self._jprime_frame_bits) == 72:
                if parse_jd_bits(self._jprime_frame_bits) is None:
                    self._record_codeword_error(
                        "invalid-jd-frame-after-caller-s", sample,
                        frame_bits="".join(map(str, self._jprime_frame_bits)))
                    self._caller_s_armed = False
                    self._jprime_frame_bits = []
                    return
                self._jprime_frames_waited += 1
                self._jprime_frame_bits = []
            return
        if bit == 0:
            self._jprime_zero_count += 1
        else:
            self._jprime_zero_count = 0
            self._jprime_frame_bits = [bit]
        if self._jprime_zero_count < 12:
            return
        self._emit({"event": "jd_prime", "at": sample + 1}, on_event)
        self._caller_s_armed = False
        self._jprime_zero_count = 0
        self._jprime_frame_bits = []
        self._dil_received = 0
        self.state = (self.READ_DIL if self._expected_dil
                      else self.PHASE3_DONE)

    def _consume_training_symbol(self, sign: int) -> None:
        """TRN1d transmits the scrambled sign without Jd differential coding."""
        self._previous_sign = sign
        self._gpc_history = [sign, *self._gpc_history[:22]]

    def _record_codeword_error(self, reason: str, sample: int, **details) -> None:
        self.codeword_errors += 1
        self.codeword_error_events.append({
            "reason": str(reason), "sample": int(sample), **details,
        })

    def _advance_jd_candidates(self, sign: int, sample: int, on_event=None) -> None:
        survivors = []
        for candidate in self._jd_candidates:
            previous = candidate["previous_sign"]
            scrambled = sign if previous is None else sign ^ previous
            candidate["previous_sign"] = sign
            history = candidate["history"]
            bit = scrambled ^ history[17] ^ history[22]
            candidate["history"] = [scrambled, *history[:22]]
            candidate["bits"].append(bit)
            if len(candidate["bits"]) == 72:
                parsed = parse_jd_bits(candidate["bits"])
                if parsed is not None:
                    self._jd_history = candidate["history"]
                    self._jd_previous_sign = candidate["previous_sign"]
                    self._post_jd_bits = 0
                    self.state = self.JD_FOUND
                    self._emit({"event": "jd", "at": sample + 1,
                                "frame_start": candidate["start"],
                                "descriptor": parsed}, on_event)
                    self._jd_candidates.clear()
                    return
            else:
                survivors.append(candidate)
        self._jd_candidates = survivors

    def _reset_to_search(self) -> None:
        self.state = self.SEARCH_SD
        self._mark_window.clear()
        self._sd_prime_index = 0
        self._trn_count = 0
        self._previous_sign = None
        self._gpc_history = [0] * 23
        self._jd_candidates.clear()
        self._jd_history = None
        self._jd_previous_sign = None
        self._post_jd_bits = 0
        self._caller_s_armed = False
        self._jd_symbols_to_finish = None
        self._jprime_zero_count = 0
