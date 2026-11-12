"""Spec-derived digital-side V.90 Phase 3 downstream codeword signals.

Signals are represented as exact negotiated PCMU/PCMA octets. This module
builds signal primitives; phase-event detection and RTP scheduling belong to
the live call controller.
"""

from __future__ import annotations

import v34_info
from v90_dil import V90DILCodewordStream, g711_ucode_codewords


def _validate_uinfo(uinfo: int) -> int:
    value = int(uinfo)
    # INFO1a requires UINFO > 66. Sd also uses Ucode 16 + UINFO, which must
    # remain in the 0..127 Table 1 range.
    if not 67 <= value <= 111:
        raise ValueError("UINFO must be in 67..111 for V.90 digital Phase 3")
    return value


def build_jd_bits(rate_mask: int, *, phase3_16point: bool = False,
                  renegotiation_16point: bool = False,
                  max_lookahead: int = 1) -> list[int]:
    """Build the 72-bit V.90 Table 13 Jd frame (LSB-first on the wire).

    ``rate_mask`` bit 0 represents Jd bit 18, skipping Jd's start bit at 34.
    The six reserved rates at Jd bits 41..46 must be clear.
    """
    if not isinstance(rate_mask, int) or not 0 <= rate_mask < (1 << 28):
        raise ValueError("rate_mask must be a 28-bit nonnegative integer")
    if rate_mask & (((1 << 6) - 1) << 22):
        raise ValueError("Jd rate-mask bits 41..46 are reserved and must be zero")
    if not 1 <= max_lookahead <= 3:
        raise ValueError("max_lookahead must be 1, 2, or 3")

    bits = [1] * 17 + [0]
    bits.extend((rate_mask >> bit) & 1 for bit in range(16))  # bits 18..33
    bits.append(0)  # bit 34 start
    bits.extend((rate_mask >> (16 + bit)) & 1 for bit in range(12))  # 35..46
    bits.extend((int(phase3_16point), int(renegotiation_16point)))
    bits.extend((max_lookahead >> bit) & 1 for bit in range(2))
    bits.append(0)  # bit 51 start
    information = bits[18:34] + bits[35:51]
    crc = v34_info.crc16(information)
    bits.extend((crc >> bit) & 1 for bit in range(16))
    bits.extend([0] * 4)
    if len(bits) != 72:
        raise AssertionError("Jd frame builder produced the wrong length")
    return bits


def sd_codewords(uinfo: int, law: str) -> bytes:
    """Build the 384-symbol digital S_d mark defined by V.90 §8.4.4."""
    uinfo = _validate_uinfo(uinfo)
    w = uinfo + 16
    levels = ((w, True), (0, True), (w, True), (w, False), (0, False), (w, False))
    table = g711_ucode_codewords(law)
    pattern = bytes(int(table[ucode]) ^ (0 if positive else 0x80)
                    for ucode, positive in levels)
    return pattern * 64


def sd_prime_codewords(uinfo: int, law: str) -> bytes:
    """Build the 48-symbol S'_d mark defined by V.90 §8.4.4."""
    uinfo = _validate_uinfo(uinfo)
    w = uinfo + 16
    levels = ((w, False), (0, False), (w, False), (w, True), (0, True), (w, True))
    table = g711_ucode_codewords(law)
    pattern = bytes(int(table[ucode]) ^ (0 if positive else 0x80)
                    for ucode, positive in levels)
    return pattern * 8


class _GPC:
    """V.34 §7-1 self-synchronizing GPC, initialized to zero for TRN1d."""

    def __init__(self):
        self.history = [0] * 23

    def step(self, bit: int) -> int:
        scrambled = int(bit) ^ self.history[17] ^ self.history[22]
        self.history = [scrambled, *self.history[:22]]
        return scrambled


class V90DigitalPhase3Transmitter:
    """Stateful raw-codeword generator for the digital modem's Phase 3.

    The caller must sequence the methods from the V.90 state machine: S_d,
    S'_d, TRN1d, repeated Jd frames, J'_d, then wait for the subsequent S
    transition before starting DIL if requested. This class preserves GPC and
    binary differential state across TRN1d, Jd, and J'_d.
    """

    def __init__(self, uinfo: int, law: str):
        self.uinfo = _validate_uinfo(uinfo)
        self.law = law.lower()
        self.table = g711_ucode_codewords(self.law)
        self.scrambler = _GPC()
        self.differential_sign: int | None = None
        self.trn1d_sent = False
        self.jd_frames_sent = 0
        self.jprime_sent = False

    def _codeword(self, positive: bool) -> int:
        code = int(self.table[self.uinfo])
        return code if positive else code ^ 0x80

    def trn1d(self, symbols: int = 2040) -> bytes:
        """Generate TRN1d; its length is at least 2040 and a multiple of six."""
        if self.trn1d_sent:
            raise RuntimeError("TRN1d has already been generated")
        if symbols < 2040 or symbols % 6:
            raise ValueError("TRN1d length must be >=2040 and divisible by 6")
        signs = [self.scrambler.step(1) for _ in range(symbols)]
        self.differential_sign = signs[-1]
        self.trn1d_sent = True
        return bytes(self._codeword(bool(sign)) for sign in signs)

    def jd_frame(self, rate_mask: int, *, phase3_16point: bool = False,
                 renegotiation_16point: bool = False,
                 max_lookahead: int = 1) -> bytes:
        """Scramble/differentially encode one Jd frame as UINFO codewords."""
        if not self.trn1d_sent or self.differential_sign is None:
            raise RuntimeError("TRN1d must precede Jd")
        if self.jprime_sent:
            raise RuntimeError("Jd cannot follow J'_d")
        bits = build_jd_bits(rate_mask, phase3_16point=phase3_16point,
                             renegotiation_16point=renegotiation_16point,
                             max_lookahead=max_lookahead)
        signs = []
        for bit in bits:
            scrambled = self.scrambler.step(bit)
            self.differential_sign ^= scrambled
            signs.append(self.differential_sign)
        self.jd_frames_sent += 1
        return bytes(self._codeword(bool(sign)) for sign in signs)

    def jprime_d(self) -> bytes:
        """Emit the 12-zero J'_d terminator after at least one Jd frame."""
        if self.jd_frames_sent < 1:
            raise RuntimeError("at least one Jd frame must precede J'_d")
        if self.jprime_sent or self.differential_sign is None:
            raise RuntimeError("J'_d has already been generated")
        signs = []
        for _ in range(12):
            scrambled = self.scrambler.step(0)
            self.differential_sign ^= scrambled
            signs.append(self.differential_sign)
        self.jprime_sent = True
        return bytes(self._codeword(bool(sign)) for sign in signs)

    def dil_stream(self, descriptor: dict) -> V90DILCodewordStream | None:
        """Start the requested DIL after J'_d; N=0 means no DIL phase."""
        if not self.jprime_sent:
            raise RuntimeError("J'_d must precede DIL")
        if int(descriptor["N"]) == 0:
            return None
        return V90DILCodewordStream(descriptor, self.law)
