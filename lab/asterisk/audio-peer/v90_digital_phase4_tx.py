"""Event-driven exact-codeword transmitter for V.90 digital Phase 4.

This component sequences the digital answerer's transmit side. A separate
receiver/controller must deliver CRC-valid CPt, CP, CP-prime, or E events at
the protocol boundary. It is not yet connected to the production DCE.
"""
from __future__ import annotations

from v90_digital_phase4 import (
    V90DigitalPhase4Encoder,
    rbar_i_codewords,
    ri_codewords,
)
from v90_mp import build_mp


class V90DigitalPhase4Transmitter:
    """Produce Phase 4 startup signals as exact PCMU/PCMA octets.

    Calls are intentionally event-driven so signal generation cannot silently
    guess a CP profile or declare CONNECT. The caller feeds each returned
    burst into its exact-law RTP transmit queue. `next_mp()` is called again
    while the peer has not yet sent CP or CP-prime/E.
    """

    IDLE = "idle"
    WAIT_CPT = "wait-cpt"
    WAIT_CP = "wait-cp"
    WAIT_CP_PRIME = "wait-cp-prime-or-e"
    DATA_READY = "data-ready"

    def __init__(self, *, uinfo: int, law: str, mp: dict | None = None):
        if type(uinfo) is not int or not 0 <= uinfo <= 127:
            raise ValueError("UINFO must be an integer in 0..127")
        law = law.lower()
        if law not in ("pcmu", "pcma"):
            raise ValueError("law must be 'pcmu' or 'pcma'")
        self.uinfo = uinfo
        self.law = law
        self.mp = dict(mp or {})
        self.mp_type = self.mp.pop("mp_type", 0)
        self._mp_fields = self.mp
        self.state = self.IDLE
        self.cpt_profile: dict | None = None
        self.cp_profile: dict | None = None
        self.encoder: V90DigitalPhase4Encoder | None = None
        self.data_encoder: V90DigitalPhase4Encoder | None = None
        self.mp_frames_sent = 0
        self.mp_prime_frames_sent = 0
        self._end_signal: str | None = None

    def start(self) -> bytes:
        """Start Phase 4 with the minimum 192T Ri while waiting for CPt."""
        if self.state != self.IDLE:
            raise RuntimeError("Phase 4 transmitter already started")
        self.state = self.WAIT_CPT
        return ri_codewords(self.uinfo, self.law, duration_symbols=192)

    def repeat_ri(self, duration_symbols: int = 24) -> bytes:
        """Keep transmitting Ri while CPt has not yet arrived."""
        if self.state != self.WAIT_CPT:
            raise RuntimeError("Ri can only repeat while waiting for CPt")
        return ri_codewords(self.uinfo, self.law,
                            duration_symbols=duration_symbols)

    def receive_cpt(self, profile: dict) -> bytes:
        """Accept CRC-valid CPt, then emit R-bar-i(24T)+TRN2d(2040T)."""
        self._validate_profile(profile, expected_kind="cpt")
        if self.state != self.WAIT_CPT:
            raise RuntimeError("CPt is only valid while waiting for the Phase 4 training profile")
        if profile.get("ack") != 0:
            raise ValueError("expected an unacknowledged V.90 CPt profile")
        self.cpt_profile = dict(profile)
        self.encoder = V90DigitalPhase4Encoder(self.cpt_profile, self.law)
        trn2d = self.encoder.trn2d(2040)
        prefix = rbar_i_codewords(self.uinfo, self.law, duration_symbols=24)
        self.state = self.WAIT_CP
        return prefix + trn2d

    def next_mp(self) -> bytes:
        """Emit one MP (or MP′ after CP) through the continuous CPt mapper."""
        if self.state not in (self.WAIT_CP, self.WAIT_CP_PRIME):
            raise RuntimeError("MP is only sent after CPt and before Ed/B1d")
        if self.encoder is None:
            raise RuntimeError("CPt mapper is not initialized")
        fields = dict(self._mp_fields)
        fields["mp_type"] = self.mp_type
        fields["ack"] = int(self.state == self.WAIT_CP_PRIME)
        bits = build_mp(**fields)
        output = self.encoder.encode_bits(bits)
        if self.state == self.WAIT_CP:
            self.mp_frames_sent += 1
        else:
            self.mp_prime_frames_sent += 1
            if self._end_signal is not None:
                output += self._emit_ed_and_b1d()
        return output

    def receive_cp(self, profile: dict) -> None:
        """Latch data-mode CP and switch subsequent MP frames to MP′."""
        self._validate_profile(profile, expected_kind="cp")
        if self.state != self.WAIT_CP:
            raise RuntimeError("CP is only valid while the answerer is sending MP")
        if profile.get("ack") != 0:
            raise ValueError("expected an unacknowledged V.90 CP profile")
        if self.mp_frames_sent == 0:
            raise RuntimeError("cannot acknowledge CP before sending an MP")
        self.cp_profile = dict(profile)
        self.state = self.WAIT_CP_PRIME

    def receive_cp_prime(self, profile: dict) -> bytes:
        """Accept analogue CP′, then emit Ed (2 zero frames) and B1d.

        The CPt encoder's mapper/scrambler state continues from MP/MP′ through
        Ed. B1d uses CP's data-mode profile and starts with reset memories, as
        required by V.90 §8.6.1.
        """
        if self.state != self.WAIT_CP_PRIME:
            raise RuntimeError("CP′/E is only valid after CP")
        if self.encoder is None or self.cpt_profile is None or self.cp_profile is None:
            raise RuntimeError("Phase 4 profiles are incomplete")
        self._validate_profile(profile, expected_kind="cp")
        if profile.get("ack") != 1:
            raise ValueError("expected an acknowledged V.90 CP′ profile")
        if self._without_ack(profile) != self._without_ack(self.cp_profile):
            raise ValueError("CP′ parameters differ from the selected CP data profile")
        return self._latch_end_signal("cp-prime")

    def receive_e(self) -> bytes:
        """Accept an upstream detector-validated 20-bit E and finish Phase 4.

        E bit/symbol detection is a receiver responsibility; this event should
        only be raised after it validates the complete 20-bit sequence.
        """
        if self.state != self.WAIT_CP_PRIME:
            raise RuntimeError("CP′/E is only valid after CP")
        if self.encoder is None or self.cp_profile is None:
            raise RuntimeError("Phase 4 profiles are incomplete")
        return self._latch_end_signal("e")

    def _latch_end_signal(self, signal: str) -> bytes:
        if self._end_signal is not None:
            raise RuntimeError("Phase 4 end signal was already received")
        self._end_signal = signal
        # The response may arrive before the transmitter has finished its
        # first MP′. Latch it and let next_mp() finish the in-flight sequence.
        return self._emit_ed_and_b1d() if self.mp_prime_frames_sent else b""

    def _emit_ed_and_b1d(self) -> bytes:
        """Emit the single Ed sequence and the 48 data-mode B1d frames."""
        assert self.encoder is not None and self.cp_profile is not None
        frame_bits = self.encoder.s + self.encoder.k
        ed = self.encoder.encode_bits([0] * (2 * frame_bits))

        self.data_encoder = V90DigitalPhase4Encoder(self.cp_profile, self.law)
        b1d_bits = [1] * (48 * (self.data_encoder.s + self.data_encoder.k))
        b1d = self.data_encoder.encode_bits(b1d_bits)
        if len(b1d) != 48 * 6:
            raise RuntimeError("B1d encoder did not produce exactly 48 data frames")
        self.state = self.DATA_READY
        return ed + b1d

    def encode_data_bits(self, bits) -> bytes:
        """Map post-B1d user bits with the continuing data-mode encoder state."""
        if self.state != self.DATA_READY or self.data_encoder is None:
            raise RuntimeError("V.90 user data is unavailable before B1d completes")
        return self.data_encoder.encode_bits(bits)

    @staticmethod
    def _without_ack(profile: dict) -> dict:
        # start_bit is scanner metadata, not negotiated signal content.
        return {key: value for key, value in profile.items()
                if key not in ("ack", "start_bit")}

    def _validate_profile(self, profile: dict, *, expected_kind: str) -> None:
        if not isinstance(profile, dict) or profile.get("kind") != expected_kind:
            raise ValueError(f"expected a parsed V.90 {expected_kind.upper()} profile")
        if str(profile.get("codec", "")).lower() != self.law:
            raise ValueError("CP/CPt G.711 law differs from the negotiated RTP law")
        if not profile.get("rate"):
            raise ValueError("CP/CPt profile has no usable digital data rate")
        if len(profile.get("interval_constellations", ())) != 6:
            raise ValueError("CP/CPt profile must select six data-frame interval maps")
