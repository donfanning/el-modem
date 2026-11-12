"""Analogue-caller V.90 Phase 4 control-frame symbol encoder.

This maps caller CPt/CP/CP′ information bits to upstream V.34 rotation and
quarter decisions. It is the inverse counterpart to ``V90CpSymbolDecoder``;
sample shaping and the connected modem's carrier/timing loops remain the
responsibility of the caller's waveform transmitter.
"""

from __future__ import annotations

import copy
from collections.abc import Iterable
import math
from numbers import Integral

import v34_tx
from v90_cp import build_cp_frame, parse_cp
from v90_digital_phase4_rx import V90DigitalPhase4Receiver
from v90_digital_phase4 import ri_codewords
from v90_mp import V90MpScanner
from v90_mp_policy import select_analog_upstream_profile
from v90_analogue_upstream_tx import (V90AnalogueUpstreamTransmitter,
                                      upstream_power_estimate_job)
from v90_lapm import V90LapmLink


# R-i acquisition tries all 128 UINFO values. Generating each 24-codeword
# reference inside the packet receive path costs tens of milliseconds, so
# prepare both negotiated G.711 laws once when this caller-side module loads.
_RI_CODEWORD_REFERENCES = {
    law: tuple(ri_codewords(uinfo, law, duration_symbols=24)
               for uinfo in range(128))
    for law in ("pcmu", "pcma")
}


def qam_symbols(decisions: Iterable[tuple[int, int]],
                constellation: int) -> list[complex]:
    """Map analogue GPA decisions onto the V.34 QAM constellation points.

    V.90 supplies the differential rotation and optional quarter index; the
    existing V.34 pulse shaper consumes complex points. This boundary performs
    only the normative point lookup and leaves pulse shaping/carrier timing to
    ``StreamQam``.
    """
    if constellation not in (4, 16):
        raise ValueError("caller Phase 4 constellation must be 4 or 16")
    points = []
    for decision in decisions:
        if (not isinstance(decision, (tuple, list)) or len(decision) != 2):
            raise ValueError("QAM decision must be a (rotation, quarter) pair")
        rotation, quarter = decision
        if (isinstance(rotation, bool) or not isinstance(rotation, Integral)
                or not 0 <= rotation < 4
                or isinstance(quarter, bool) or not isinstance(quarter, Integral)
                or not 0 <= quarter < 4):
            raise ValueError("QAM rotation and quarter must be integers in 0..3")
        if constellation == 4 and quarter != 0:
            raise ValueError("four-point QAM decisions cannot carry a quarter index")
        base = complex(*v34_tx.C16[int(quarter)])
        points.append(base * v34_tx.ROT_CW[int(rotation)])
    return points


class V90AnaloguePhase4Encoder:
    """Encode CP-family control bits with the analogue GPA/differential state.

    Instantiate at the first CPt symbol and preserve one encoder through CP
    and CP′. The selected four- or sixteen-point geometry is fixed for the
    control exchange. ``encode_bits`` returns ``(rotation, quarter)`` pairs
    for the caller's V.34 QAM waveform path.
    """

    def __init__(self, constellation: int):
        if constellation not in (4, 16):
            raise ValueError("caller CP constellation must be 4 or 16")
        self.constellation = int(constellation)
        self._width = 2 if constellation == 4 else 4
        self.reset()

    def reset(self) -> None:
        """Reset GPA and differential memories at the CPt boundary."""
        self._gpa = [0] * 23
        self._rotation = 0

    def encode_bits(self, bits: Iterable[int]) -> list[tuple[int, int]]:
        values = list(bits)
        if any(isinstance(bit, bool) or not isinstance(bit, Integral)
               or bit not in (0, 1) for bit in values):
            raise ValueError("caller Phase 4 input must contain bits")
        values = [int(bit) for bit in values]
        if len(values) % self._width:
            raise ValueError("caller Phase 4 control frame must align to symbol width")
        symbols = []
        for offset in range(0, len(values), self._width):
            scrambled = []
            for bit in values[offset:offset + self._width]:
                coded = bit ^ self._gpa[4] ^ self._gpa[22]
                self._gpa = [coded, *self._gpa[:22]]
                scrambled.append(coded)
            self._rotation = (
                self._rotation + 2 * scrambled[1] + scrambled[0]) % 4
            quarter = (2 * scrambled[3] + scrambled[2]
                       if self.constellation == 16 else 0)
            symbols.append((self._rotation, quarter))
        return symbols

    def encode_cp_frame(self, **fields) -> tuple[list[int], list[tuple[int, int]]]:
        """Build and encode one caller CP-family control frame."""
        bits = build_cp_frame(**fields)
        return bits, self.encode_bits(bits)


class V90AnaloguePhase4Controller:
    """Pair caller CP negotiation with exact-codeword downstream reception.

    The CPt/CP offers are explicit local capability choices. The controller
    checks the answerer's MP against the selected CP's analogue rate mask,
    emits CP and CP′ with continuous caller-side GPA/differential state, then
    qualifies the two ED frames and all 48 B1d frames before exposing
    downstream data bits. Caller upstream B1 and LAPM bits run concurrently
    with answerer B1d on the independent V.34 transmit clock. Sample
    modulation and carrier/timing recovery remain in the caller's waveform/DSP
    layer.
    """

    WAIT_MP = "wait-mp"
    WAIT_MP_PRIME = "wait-mp-prime"
    WAIT_ED = "wait-ed"
    WAIT_B1D = "wait-b1d"
    DATA = "data"
    FAILED = "failed"

    def __init__(self, *, cpt_fields: dict, cp_fields: dict, law: str,
                 constellation: int, baud: float, carrier: float,
                 capture_data_bits: bool = False, lapm_detect: bool = True,
                 data_fault_hold_frames: int = 0):
        self.law = str(law).lower()
        self.lapm_detect = bool(lapm_detect)
        if (isinstance(data_fault_hold_frames, bool)
                or not isinstance(data_fault_hold_frames, int)
                or data_fault_hold_frames < 0):
            raise ValueError("data_fault_hold_frames must be a non-negative integer")
        # 0 keeps the fail-closed default: the first invalid DATA frame ends
        # Phase 4. A positive hold rides through that many consecutive
        # invalid frames (signal loss or a peer retrain) before failing.
        self.data_fault_hold_frames = data_fault_hold_frames
        self._data_fault_run = 0
        self._data_frames = 0
        if self.law not in ("pcmu", "pcma"):
            raise ValueError("caller Phase 4 law must be 'pcmu' or 'pcma'")
        if constellation not in (4, 16):
            raise ValueError("caller CP constellation must be 4 or 16")
        cpt = dict(cpt_fields)
        cpt.update(kind="cpt", ack=0, law=self.law)
        cp = dict(cp_fields)
        cp.update(kind="cp", ack=0, law=self.law)
        self.cpt_bits = build_cp_frame(**cpt)
        self.cp_bits = build_cp_frame(**cp)
        self.cp_prime_bits = build_cp_frame(**{**cp, "ack": 1})
        self.cpt_profile = parse_cp(self.cpt_bits)
        self.cp_profile = parse_cp(self.cp_bits)
        if self.cpt_profile is None or self.cp_profile is None:
            raise ValueError("caller CP capability frame failed local validation")
        if self.cpt_profile["spectral_sign_bits"] != 0:
            raise NotImplementedError("caller Phase 4 currently requires S_r=0 CPt")
        if self.cp_profile["spectral_sign_bits"] != 0:
            raise NotImplementedError("caller Phase 4 currently requires S_r=0 CP")
        if self.cpt_profile["codec"] != self.law or self.cp_profile["codec"] != self.law:
            raise ValueError("caller CP frames must use the negotiated G.711 law")

        self.constellation = int(constellation)
        self.baud = float(baud)
        self.carrier = float(carrier)
        self.capture_data_bits = bool(capture_data_bits)
        self.state = self.WAIT_MP
        self.events: list[tuple[str, object]] = []
        self._started = False
        self._encoder = V90AnaloguePhase4Encoder(self.constellation)
        self._receiver = V90DigitalPhase4Receiver(self.cpt_profile, self.law)
        self._scanner = V90MpScanner()
        self._downstream_pending = bytearray()
        # R-i may repeat while CPt is crossing the analogue channel. Wait for
        # its exact 24T codeword block to stop before treating subsequent
        # codewords as TRN2d/MP. A fixed 24-octet skip only works when the test
        # starts precisely at the post-CPt response boundary.
        self._ri_pending = bytearray()
        self._ri_pattern: bytes | None = None
        self._ri_alignment_end: int | None = None
        self._ri_alignment_phase: int | None = None
        self._trn2d_start: bytes | None = None
        self._awaiting_trn2d = True
        # Ed is recognized as a consecutive zero-frame run under CPt. After
        # the run, keep one CP-decoded boundary frame as receiver history:
        # the GPC descrambler is self-synchronizing, so a boundary frame that
        # straddles the end of Ed may not itself decode to all ones, while the
        # following complete B1d frames can. Never count that transition
        # frame toward the 48 required B1d frames.
        self._ed_zero_run = 0
        self._ed_wait_frames = 0
        # The real captured peer path measured about 0.5 s round-trip delay. Allow
        # three seconds for CP′ to cross the line and Ed to return, while
        # keeping the no-response path bounded. A receive frame is six 8 kHz
        # codewords (0.75 ms), so this deadline is independent of symbol baud.
        self._ed_wait_limit_frames = max(512, math.ceil(8000 * 3.0 / 6.0))
        self._b1d_frames_seen = 0
        self._mp: dict | None = None
        self._failure_reason: str | None = None
        self._failure_code: str | None = None
        self._data_bits: list[int] = []
        self._upstream_b1_started = False
        self.upstream_profile = None
        self.upstream_transmitter: V90AnalogueUpstreamTransmitter | None = None
        self._upstream_power_job = None
        self._upstream_prep_credit = 0
        self.lapm: V90LapmLink | None = None

    def start(self) -> list[tuple[int, int]]:
        """Emit CPt symbols; call once at the V.90 digital Phase 4 boundary."""
        if self.state != self.WAIT_MP or self.events:
            raise RuntimeError("caller Phase 4 controller already started")
        self._started = True
        symbols = self._encoder.encode_bits(self.cpt_bits)
        self.events.append(("cpt-sent", self.cpt_profile))
        return symbols

    def start_qam(self) -> list[complex]:
        """Emit CPt as complex V.34 QAM points for the caller pulse shaper."""
        return qam_symbols(self.start(), self.constellation)

    def repeat_cpt(self) -> list[tuple[int, int]]:
        """Continue CPt while waiting for the answerer's MP response."""
        if not self._started or self.state != self.WAIT_MP:
            raise RuntimeError("caller CPt can repeat only while waiting for MP")
        return self._encoder.encode_bits(self.cpt_bits)

    def repeat_cp(self) -> list[tuple[int, int]]:
        """Continue CP while waiting for the answerer's acknowledged MP′."""
        if not self._started or self.state != self.WAIT_MP_PRIME:
            raise RuntimeError("caller CP can repeat only while waiting for MP-prime")
        return self._encoder.encode_bits(self.cp_bits)

    def repeat_cp_prime(self) -> list[tuple[int, int]]:
        """Continue CP′ while waiting for the answerer's Ed response."""
        if not self._started or self.state != self.WAIT_ED:
            raise RuntimeError("caller CP-prime can repeat only while waiting for Ed")
        return self._encoder.encode_bits(self.cp_prime_bits)

    def feed_downstream(self, codewords: bytes | bytearray) -> list[tuple[int, int]]:
        """Consume exact G.711 octets and return any CP/CP′ response symbols."""
        if not self._started:
            raise RuntimeError("caller Phase 4 controller has not started")
        if self.state in (self.FAILED,):
            return []
        new_codewords = bytes(codewords)
        self._advance_upstream_preparation(len(new_codewords))
        if self._awaiting_trn2d:
            self._ri_pending.extend(new_codewords)
            if self._ri_pattern is None:
                acquired = self._find_ri_pattern(self._ri_pending)
                if acquired is None:
                    # Keep enough history to match a 24-symbol R-i window
                    # that straddles the next RTP packet. R-i is periodic
                    # every six codewords, so acquisition is independent of
                    # the caller's packet boundary and its entry phase.
                    if len(self._ri_pending) > 23:
                        del self._ri_pending[:-23]
                    return []
                offset, phase, pattern = acquired
                self._ri_pattern = pattern
                self._ri_alignment_phase = phase
                # The repeated 24-codeword signature is phase-independent,
                # but the following six-codeword mapper frames are not. Drop
                # through the first R-i frame boundary after acquisition.
                self._ri_alignment_end = offset + 24 + ((-phase) % 6)
            if (self._ri_alignment_end is not None
                    and len(self._ri_pending) < self._ri_alignment_end):
                return []
            if self._ri_alignment_end is not None:
                del self._ri_pending[:self._ri_alignment_end]
                self._ri_alignment_end = None
            # R-i can remain in flight after CPt, and 160-symbol RTP packets
            # split six-symbol mapper frames. Do not consume matching R-i
            # frames one at a time: the last old symbols can combine with the
            # first R-bar-i symbols to form a false R-i frame. Keep a bounded
            # rolling history and detect the R-bar-i transition against the
            # phase boundary acquired from the initial R-i.
            restarted = self._find_rbar_transition(
                self._ri_pending, self._ri_pattern)
            if restarted is None:
                # Preserve 23 symbols of overlap and the six-symbol phase.
                # The initial RI alignment above makes the pending buffer's
                # origin a mapper-frame boundary; dropping whole frames keeps
                # it so and bounds the live receive-path work.
                if len(self._ri_pending) > 53:
                    discard = ((len(self._ri_pending) - 53) // 6) * 6
                    del self._ri_pending[:discard]
                return []
            offset, _phase = restarted
            # The detected sequence is the full 24T R-bar-i burst, so TRN2d
            # starts immediately after it. Keep that reset boundary even when
            # it falls between the six-symbol boundaries used by RTP packets.
            align_end = offset + 24
            if len(self._ri_pending) < align_end:
                return []
            del self._ri_pending[:align_end]
            self._trn2d_start = bytes(self._ri_pending[:24])
            self._downstream_pending.extend(self._ri_pending)
            self._ri_pending.clear()
            self._awaiting_trn2d = False
            self.events.append(("ri-complete", self._ri_pattern))
            new_codewords = b""
        self._downstream_pending.extend(new_codewords)
        output: list[tuple[int, int]] = []
        while len(self._downstream_pending) >= 6:
            frame = bytes(self._downstream_pending[:6])
            del self._downstream_pending[:6]
            try:
                if self.state == self.WAIT_ED:
                    self._ed_wait_frames += 1
                    if self._ed_wait_frames > self._ed_wait_limit_frames:
                        return self._fail(
                            "timed out waiting for V.90 ED after MP-prime",
                            output)
                    candidate = None
                    candidate_bits: list[int] = []
                    if self._ed_zero_run >= 2:
                        # CPt can also map the first B1d frame to all zeroes.
                        # Probe with a fresh CP decoder before asking CPt to
                        # decode a possible B1d frame using the wrong mapper.
                        candidate = V90DigitalPhase4Receiver(
                            self.cp_profile, self.law)
                        try:
                            candidate_bits = candidate.decode_codewords(frame)
                        except ValueError:
                            # decode_codewords buffers octets before validating
                            # them. A rejected speculative CP frame therefore
                            # leaves this receiver dirty; never carry it across
                            # the CPt decision or retry the same frame on it.
                            candidate = None
                        if (candidate_bits
                                and all(bit == 1 for bit in candidate_bits)):
                            self._accept_ed_and_b1d_frame(
                                candidate, candidate_bits, output)
                            continue

                    # Keep CPt active while zero frames continue. Once CPt
                    # leaves the Ed run, the current frame is the CP decoder's
                    # boundary/history frame. It may straddle the transition;
                    # preserve its CP state, but count only following complete
                    # all-ones frames as B1d.
                    try:
                        bits = self._receiver.decode_codewords(frame)
                    except ValueError as exc:
                        if self._ed_zero_run >= 2:
                            if candidate is None:
                                candidate = V90DigitalPhase4Receiver(
                                    self.cp_profile, self.law)
                            if not candidate_bits:
                                try:
                                    candidate_bits = candidate.decode_codewords(frame)
                                except ValueError as cp_exc:
                                    return self._fail(
                                        "V.90 CP transition frame is invalid: "
                                        f"{cp_exc}", output)
                            if candidate_bits and all(
                                    bit == 1 for bit in candidate_bits):
                                self._accept_ed_and_b1d_frame(
                                    candidate, candidate_bits, output)
                            else:
                                self._latch_ed(candidate, output)
                                self.events.append(("b1d-transition-frame", {
                                    "frame_bits": len(candidate_bits),
                                    "ones": sum(candidate_bits),
                                    "reason": "CPt rejected boundary frame",
                                }))
                            continue
                        raise
                    if not any(bits):
                        self._ed_zero_run += 1
                        continue
                    if self._ed_zero_run < 2:
                        # Nonzero mapper frames can occur while MP′/CP′ is
                        # still in flight. Ed must be a consecutive run.
                        self._ed_zero_run = 0
                        continue
                    if candidate is None:
                        # This frame was valid under CPt but not CP. It marks
                        # the transition, but cannot seed CP mapper/GPC state.
                        # Latch a clean CP receiver and start it on the next
                        # complete frame instead of retrying this rejected one.
                        candidate = V90DigitalPhase4Receiver(
                            self.cp_profile, self.law)
                        self._latch_ed(candidate, output)
                        self.events.append(("b1d-transition-frame", {
                            "frame_bits": len(bits),
                            "ones": sum(bits),
                            "reason": "CPt boundary frame is invalid under CP",
                        }))
                        continue
                    if (candidate_bits
                            and all(bit == 1 for bit in candidate_bits)):
                        self._accept_ed_and_b1d_frame(
                            candidate, candidate_bits, output)
                        continue
                    self._latch_ed(candidate, output)
                    self.events.append(("b1d-transition-frame", {
                        "frame_bits": len(candidate_bits),
                        "ones": sum(candidate_bits),
                        "reason": "first CPt nonzero frame",
                    }))
                    continue

                if self.state == self.DATA and self.data_fault_hold_frames:
                    self._data_frames += 1
                    try:
                        bits = self._receiver.decode_codewords(frame)
                    except ValueError as exc:
                        # The rejected frame stays in the receiver's pending
                        # buffer; drop it so the next frame decodes on its own.
                        self._receiver._pending.clear()
                        if not self._data_fault_run:
                            self.events.append(("data-fault-start", {
                                "frame": self._data_frames,
                                "codewords": frame.hex(),
                                "error": str(exc),
                            }))
                        self._data_fault_run += 1
                        if self._data_fault_run > self.data_fault_hold_frames:
                            return self._fail(
                                "V.90 downstream DATA invalid for "
                                f"{self._data_fault_run} frames: {exc}", output)
                        continue
                    if self._data_fault_run:
                        self.events.append(("data-fault-end", {
                            "frame": self._data_frames,
                            "frames": self._data_fault_run,
                        }))
                        self._data_fault_run = 0
                else:
                    bits = (self._decode_b1d_frame(frame)
                            if self.state == self.WAIT_B1D
                            else self._receiver.decode_codewords(frame))
                if self.state in (self.WAIT_MP, self.WAIT_MP_PRIME):
                    for message in self._scanner.feed(bits):
                        response = self._accept_mp(message)
                        if response:
                            output.extend(response)
                elif self.state == self.WAIT_B1D:
                    if any(bit != 1 for bit in bits):
                        return self._fail(
                            "V.90 B1d frame was not all ones", output,
                            failure_code="b1d_not_all_ones")
                    self._b1d_frames_seen += 1
                    if self._b1d_frames_seen == 48:
                        self.state = self.DATA
                        self.events.append(("b1d-qualified", {
                            "frames": self._b1d_frames_seen,
                            "frame_bits": len(bits),
                        }))
                elif self.state == self.DATA:
                    if self.capture_data_bits:
                        self._data_bits.extend(bits)
                    if self.lapm is not None:
                        self.lapm.feed_line_bits(bits)
            except (ValueError, RuntimeError, BufferError) as exc:
                return self._fail(f"{type(exc).__name__}: {exc}", output)
        return output

    def _decode_b1d_frame(self, frame: bytes) -> list[int]:
        """Decode a known B1d frame, resolving only PCMU's zero-sign alias.

        PCMU codewords 0xff and 0x7f both decode to linear zero. If a zero
        Ucode crosses a PCM/analog boundary, its sign is not observable. The
        known all-ones B1d sequence can resolve that one erased sign bit; all
        other symbols and all other decoded bits remain exact requirements.
        """
        receiver = self._receiver
        if not hasattr(receiver, "_codeword_to_point"):
            return receiver.decode_codewords(frame)
        baseline = copy.deepcopy(receiver)
        try:
            decoded = baseline.decode_codewords(frame)
        except ValueError:
            decoded = []
        frame_bits = round(float(self.cp_profile["rate"]) * 6 / 8000)
        expected = [1] * frame_bits
        if decoded == expected:
            self._receiver = baseline
            return decoded

        zero_positions = []
        if self.law == "pcmu":
            zero_position = receiver._codeword_to_point
            for index, codeword in enumerate(frame):
                mapped = zero_position.get(codeword)
                if mapped is not None and mapped[0] == 0:
                    zero_positions.append(index)
        candidates = []
        for mask in range(1, 1 << len(zero_positions)):
            alternate = bytearray(frame)
            for bit, index in enumerate(zero_positions):
                if mask & (1 << bit):
                    alternate[index] ^= 0x80
            trial = copy.deepcopy(receiver)
            try:
                candidate = trial.decode_codewords(alternate)
            except ValueError:
                continue
            if candidate == expected:
                candidates.append(trial)
                if len(candidates) > 1:
                    break
        if len(candidates) == 1:
            self._receiver = candidates[0]
            self.events.append(("b1d-zero-sign-resolved", {
                "positions": zero_positions,
            }))
            return expected

        # Preserve the ordinary decoder's exact result/error behavior when
        # the erased-sign hypothesis does not uniquely recover B1d.
        return receiver.decode_codewords(frame)

    def feed_downstream_qam(self, codewords: bytes | bytearray) -> list[complex]:
        """Consume downstream codewords and return response QAM points."""
        return qam_symbols(self.feed_downstream(codewords), self.constellation)

    def _accept_mp(self, message: dict) -> list[tuple[int, int]]:
        if self.state == self.WAIT_MP:
            if message.get("ack") != 0:
                raise ValueError("expected unacknowledged V.90 MP")
            # Validate that the caller's selected CP can receive the requested
            # upstream profile before committing the CP response.
            policy_mp = {**message, "mp_type": message.get("type")}
            self.upstream_profile = select_analog_upstream_profile(
                self.cp_profile, policy_mp, baud=self.baud, carrier=self.carrier)
            # The mapper's power estimate is about 100 ms of work. Advance it
            # with the downstream feed while CP plays out instead of stalling
            # the media thread here; it is complete by MP′ at the latest.
            self._upstream_power_job = upstream_power_estimate_job(
                self.upstream_profile)
            self.lapm = V90LapmLink(
                rate=self.upstream_profile.rate, calling_party=True,
                detect=self.lapm_detect)
            self._mp = dict(message)
            symbols = self._encoder.encode_bits(self.cp_bits)
            self.state = self.WAIT_MP_PRIME
            self.events.append(("mp-accepted", dict(message)))
            self.events.append(("cp-sent", self.cp_profile))
            return symbols
        if self._mp is not None and self._mp_identity(message) == self._mp_identity(self._mp):
            if self.state == self.WAIT_MP_PRIME and message.get("ack") == 0:
                # MP repeats until the caller's CP reaches the answerer.
                return []
            if (self.state in (self.WAIT_ED, self.WAIT_B1D, self.DATA)
                    and message.get("ack") == 1):
                # MP-prime can repeat while CP-prime is in flight. The ED/B1d
                # transition must happen once; don't restart it on a duplicate.
                return []
        if message.get("ack") != 1 or self._mp is None:
            raise ValueError("expected acknowledged V.90 MP-prime")
        if self._mp_identity(message) != self._mp_identity(self._mp):
            raise ValueError("V.90 MP-prime differs from the accepted MP")
        # CP′ closes profile negotiation. Repeat it while waiting for the
        # answerer's Ed; E and upstream B1 follow only after Ed is qualified.
        self._finish_upstream_preparation()
        symbols = self._encoder.encode_bits(self.cp_prime_bits)
        self.state = self.WAIT_ED
        self._ed_zero_run = 0
        self._ed_wait_frames = 0
        self.events.append(("mp-prime-accepted", dict(message)))
        self.events.append(("cp-prime-sent", self.cp_profile))
        return symbols

    def _latch_ed(self, receiver, output: list[tuple[int, int]]) -> None:
        """Latch Ed and send E at the detected CPt-to-CP boundary."""
        self._receiver = receiver
        self.state = self.WAIT_B1D
        self.events.append(("ed-accepted", None))
        # Acknowledge Ed before the caller's upstream B1 starts.
        output.extend(self._encoder.encode_bits([1] * 20))
        self.events.append(("e-sent", 20))

    def _accept_ed_and_b1d_frame(self, receiver, bits: list[int],
                                 output: list[tuple[int, int]]) -> None:
        """Latch Ed and the first all-ones frame, preserving CP decoder state."""
        self._latch_ed(receiver, output)
        self._b1d_frames_seen += 1
        if self._b1d_frames_seen == 48:
            self.state = self.DATA
            self.events.append(("b1d-qualified", {
                "frames": self._b1d_frames_seen,
                "frame_bits": len(bits),
            }))

    def _find_ri_pattern(self, data: bytearray) -> tuple[int, int, bytes] | None:
        """Find a 24T repeated R-i window at any received codeword offset."""
        if len(data) < 24:
            return None
        for uinfo, reference in enumerate(_RI_CODEWORD_REFERENCES[self.law]):
            rotated = self._find_rotated_pattern(data, reference)
            if rotated is not None:
                offset, phase = rotated
                return offset, phase, reference
        return None

    @staticmethod
    def _find_rotated_pattern(data: bytearray,
                              reference: bytes) -> tuple[int, int] | None:
        """Find the earliest cyclic phase of a repeated 24-codeword signal.

        R-i and R-bar-i have the same six-codeword periodic signature with a
        three-codeword rotation. Searching phase 0 first can therefore match
        inside an R-bar-i burst, three codewords after its true start, and
        lose the transition's alignment. Earliest-offset selection preserves
        the actual burst boundary when both rotations occur in the buffer.
        """
        raw = bytes(data)
        best: tuple[int, int] | None = None
        for phase in range(6):
            pattern = reference[phase:] + reference[:phase]
            offset = raw.find(pattern)
            if offset >= 0 and (best is None or offset < best[0]):
                best = (offset, phase)
        return best

    @staticmethod
    def _find_rbar_transition(data: bytearray,
                              reference: bytes) -> tuple[int, int] | None:
        """Find R-bar-i at the phase opposite the acquired initial R-i.

        With constant UINFO, R-i and R-bar-i are the same six-codeword
        periodic magnitudes with a 3T sign phase difference. The phase-3
        pattern also occurs three symbols into every ordinary R-i repeat, so
        accept it only at an offset that is not the expected phase-3 position
        of the acquired R-i stream. This preserves the actual 24T transition
        start, including when RTP packetization leaves a partial old R-i tail.
        """
        raw = bytes(data)
        pattern = reference[3:] + reference[:3]
        offset = raw.find(pattern)
        while offset >= 0:
            if offset % 6 != 3:
                return offset, 3
            offset = raw.find(pattern, offset + 1)
        return None

    @staticmethod
    def _mp_identity(message: dict) -> dict:
        return {key: value for key, value in message.items()
                if key not in ("ack", "start_bit")}

    def _fail(self, reason: str, output: list[tuple[int, int]], *,
              failure_code: str | None = None) -> list[tuple[int, int]]:
        self.state = self.FAILED
        self._failure_reason = str(reason)
        self._failure_code = failure_code
        self.events.append(("phase4-failed", reason))
        return output

    # One simulated estimate frame (about 1.3 ms) per 32 received codewords
    # (4 ms of media) finishes the 80-frame estimate in about 0.32 s.
    _UPSTREAM_PREP_CODEWORDS_PER_FRAME = 32

    def _advance_upstream_preparation(self, codeword_count: int) -> None:
        if self._upstream_power_job is None:
            return
        self._upstream_prep_credit += int(codeword_count)
        frames, self._upstream_prep_credit = divmod(
            self._upstream_prep_credit, self._UPSTREAM_PREP_CODEWORDS_PER_FRAME)
        if frames and self._upstream_power_job.advance(frames):
            self._build_upstream_transmitter()

    def _finish_upstream_preparation(self) -> None:
        if self._upstream_power_job is None:
            return
        self._upstream_power_job.advance(
            2 * self._upstream_power_job.frames)
        self._build_upstream_transmitter()

    def _build_upstream_transmitter(self) -> None:
        self.upstream_transmitter = V90AnalogueUpstreamTransmitter(
            self.upstream_profile,
            power_estimate=self._upstream_power_job.result)
        self._upstream_power_job = None

    def take_data_bits(self) -> list[int]:
        bits, self._data_bits = self._data_bits, []
        return bits

    def encode_upstream_b1(self) -> list[complex]:
        """Start upstream B1 immediately after caller E, concurrently with B1d."""
        if self.state == self.FAILED:
            reason = self._failure_reason or "unknown Phase 4 failure"
            raise RuntimeError(
                f"V.90 upstream B1 unavailable after Phase 4 failure: {reason}")
        if (self.state not in (self.WAIT_ED, self.WAIT_B1D, self.DATA)
                or self.upstream_transmitter is None
                or not any(event[0] == "e-sent" for event in self.events)):
            raise RuntimeError("V.90 upstream B1 requires transmitted E")
        if self._upstream_b1_started:
            raise RuntimeError("V.90 analogue upstream B1 was already started")
        self._upstream_b1_started = True
        return self.upstream_transmitter.encode_b1()

    def encode_upstream_bits(self, bits: Iterable[int]) -> list[complex]:
        """Map caller/LAPM bits through the negotiated upstream V.34 profile."""
        if (self.state not in (self.WAIT_ED, self.WAIT_B1D, self.DATA)
                or self.upstream_transmitter is None
                or not self.upstream_transmitter.b1_sent):
            raise RuntimeError("V.90 upstream data requires transmitted B1")
        return self.upstream_transmitter.encode_bits(bits)

    def feed_bytes(self, data: bytes) -> None:
        """Queue caller DTE bytes for the negotiated V.42/LAPM link."""
        if self.lapm is None:
            raise RuntimeError("V.90 caller LAPM is unavailable before MP negotiation")
        self.lapm.feed_bytes(bytes(data))

    def take_received(self) -> bytes:
        """Drain caller DTE bytes received through downstream V.90 bits."""
        if self.lapm is None:
            return b""
        return self.lapm.take_received()

    def encode_next_upstream_frame(self) -> list[complex]:
        """Pull one full caller LAPM bit frame and map it into V.34 symbols."""
        if (self.state not in (self.WAIT_ED, self.WAIT_B1D, self.DATA)
                or self.lapm is None
                or self.upstream_transmitter is None
                or not self.upstream_transmitter.b1_sent):
            raise RuntimeError("V.90 upstream LAPM data requires transmitted B1")
        count = self.upstream_transmitter.bits_per_data_frame
        return self.upstream_transmitter.encode_bits(self.lapm.read_tx_bits(count))


class V90AnaloguePhase4MediaTx:
    """Render caller Phase 4 decisions on a continuous 8 kHz V.34 waveform.

    The wrapper owns only the analog transmit pulse-shaper clock. The enclosing
    caller/DCE still owns receive scheduling, RTP pacing, and LAPM. One
    ``StreamQam`` instance is retained across CPt, CP, CP′, B1 and data so
    carrier and pulse history are continuous while the controller waits for
    downstream G.711 input.
    """

    def __init__(self, controller: V90AnaloguePhase4Controller, *,
                 baud: float, carrier: float, rms: float,
                 preemphasis=None, span: int = 8):
        if not isinstance(controller, V90AnaloguePhase4Controller):
            raise TypeError("Phase 4 media transmitter requires its caller controller")
        self.controller = controller
        self.qam = v34_tx.StreamQam(baud, carrier, rms, span=span)
        self.qam.set_constellation(controller.constellation)
        self.qam.preemphasis = preemphasis
        self._started = False
        self.upstream_b1_symbols: list[complex] | None = None

    def start(self, *, decisions=None) -> list[tuple[int, int]]:
        if self._started:
            raise RuntimeError("caller Phase 4 media transmitter already started")
        if decisions is None:
            decisions = self.controller.start()
        elif not self.controller._started:
            raise RuntimeError(
                "precomputed Phase 4 decisions require a started controller")
        self.qam.push(qam_symbols(decisions, self.controller.constellation))
        self._started = True
        return decisions

    def feed_downstream(self, codewords: bytes | bytearray) -> list[tuple[int, int]]:
        if not self._started:
            raise RuntimeError("caller Phase 4 media transmitter has not started")
        decisions = self.controller.feed_downstream(codewords)
        if decisions:
            self.qam.push(qam_symbols(decisions, self.controller.constellation))
        # The caller's E sequence and upstream B1 share one transmit clock.
        # Start B1 at E's boundary; downstream Ed/B1d arrives independently.
        if (self.upstream_b1_symbols is None
                and self.controller.state != self.controller.FAILED
                and any(event[0] == "e-sent" for event in self.controller.events)):
            self.start_upstream_b1()
        return decisions

    def read_pcm(self, sample_count: int):
        if not self._started:
            raise RuntimeError("caller Phase 4 media transmitter has not started")
        # CPt repeats until MP; CP repeats until MP′; CP′ repeats until Ed.
        # Keep the pulse shaper fed
        # far enough ahead for this read, including its lookahead span, while
        # preserving one continuous QAM timeline.
        buffered = len(self.qam.buf)
        chunks = max(0, (max(0, int(sample_count) - buffered)
                         + self.qam.samples_per_chunk - 1)
                     // self.qam.samples_per_chunk)
        required = chunks * self.qam.chunk + self.qam.span
        while (self.controller.state in (
                   self.controller.WAIT_MP,
                   self.controller.WAIT_MP_PRIME,
                   self.controller.WAIT_ED)
               and len(self.qam.queue) < required):
            decisions = (self.controller.repeat_cpt()
                         if self.controller.state == self.controller.WAIT_MP
                        else (self.controller.repeat_cp()
                              if self.controller.state == self.controller.WAIT_MP_PRIME
                              else self.controller.repeat_cp_prime()))
            self.qam.push(qam_symbols(decisions, self.controller.constellation))
        # Upstream data follows caller B1 and runs concurrently with the
        # answerer's downstream B1d. Start LAPM ODP during B1d so V.42 timers
        # advance on the media clock instead of expiring before negotiation.
        while (self.controller.state in (
                   self.controller.WAIT_ED,
                   self.controller.WAIT_B1D,
                   self.controller.DATA)
               and self.controller.lapm is not None
               and self.controller.upstream_transmitter is not None
               and self.controller.upstream_transmitter.b1_sent
               and len(self.qam.queue) < required):
            self.qam.push(self.controller.encode_next_upstream_frame())
        return self.qam.read(sample_count)

    def start_upstream_b1(self) -> list[complex]:
        """Queue the caller's B1 immediately after the queued CP′/E waveform."""
        if self.upstream_b1_symbols is not None:
            raise RuntimeError("caller upstream B1 was already queued")
        symbols = self.controller.encode_upstream_b1()
        # Rescale already queued CP′/E symbols without changing their level,
        # then render the following B1 with the normalized 4-point data mode.
        self.qam.set_constellation(4)  # upstream symbols are normalized to E{|a|²}=2
        self.qam.push(symbols)
        self.upstream_b1_symbols = list(symbols)
        return symbols

    def queue_upstream_bits(self, bits: Iterable[int]) -> list[complex]:
        """Queue complete caller/LAPM bit frames and return their QAM symbols."""
        symbols = self.controller.encode_upstream_bits(bits)
        self.qam.set_constellation(4)
        self.qam.push(symbols)
        return symbols

    def queue_next_upstream_frame(self) -> list[complex]:
        """Queue the next complete upstream LAPM/V.34 frame onto the media clock."""
        symbols = self.controller.encode_next_upstream_frame()
        self.qam.push(symbols)
        return symbols
