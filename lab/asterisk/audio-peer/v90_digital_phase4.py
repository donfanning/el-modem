"""Digital-modem V.90 Phase 4 signal primitives.

The functions here emit exact PCMU/PCMA codewords. They are deliberately
standalone: the production DCE must not advertise or enter V.90 until the full
Phase 4 controller, data encoder, receiver feedback, and payload path qualify.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

import v34_g711 as g711
from v90_dil import g711_ucode_codewords


_R_I_SIGNS = (1, 1, 1, 0, 0, 0)


def _check_law(law: str) -> str:
    normalized = law.lower()
    if normalized not in ("pcmu", "pcma"):
        raise ValueError("law must be 'pcmu' or 'pcma'")
    return normalized


def _r_signal(six_ucodes: Sequence[int], law: str, *, r_bar: bool = False) -> bytes:
    """Encode four repetitions of one six-codeword R/R-bar sequence.

    The six Ucodes define the positive magnitudes; R begins with signs +++---
    and R-bar uses the complementary ---+++. `Ri` is R with UINFO magnitudes;
    the bar variant is a polarity-independent detection partner, not a distinct
    `Rii` codeword profile.
    """
    if len(six_ucodes) != 6:
        raise ValueError("an R signal requires six interval Ucodes")
    if any(not isinstance(value, (int, np.integer)) or not 0 <= int(value) <= 127
           for value in six_ucodes):
        raise ValueError("R Ucodes must be integers in 0..127")
    law = _check_law(law)
    table = g711_ucode_codewords(law)
    signs = tuple(1 - bit for bit in _R_I_SIGNS) if r_bar else _R_I_SIGNS
    frame = bytes(int(table[int(ucode)]) ^ (0 if positive else 0x80)
                  for ucode, positive in zip(six_ucodes, signs))
    return frame * 4


def ri_codewords(uinfo: int, law: str, *, duration_symbols: int = 192) -> bytes:
    """Return R-i for a duration that is a positive multiple of 24 symbols.

    V.90 startup uses R-i for at least 192T while awaiting CPt, then for 24T
    after CPt. This helper defaults to the 192T minimum; callers may request
    24 for the post-CPt interval. UINFO is repeated in all six intervals.
    """
    if not isinstance(uinfo, (int, np.integer)) or not 0 <= int(uinfo) <= 127:
        raise ValueError("UINFO must be an integer in 0..127")
    if duration_symbols <= 0 or duration_symbols % 24:
        raise ValueError("R-i duration must be a positive multiple of 24 symbols")
    one = _r_signal((int(uinfo),) * 6, law)
    return one * (duration_symbols // 24)


def rbar_i_codewords(uinfo: int, law: str, *, duration_symbols: int = 24) -> bytes:
    """Return the complementary R-bar-i sign pattern using UINFO magnitudes."""
    if not isinstance(uinfo, (int, np.integer)) or not 0 <= int(uinfo) <= 127:
        raise ValueError("UINFO must be an integer in 0..127")
    if duration_symbols <= 0 or duration_symbols % 24:
        raise ValueError("R-bar-i duration must be a positive multiple of 24 symbols")
    one = _r_signal((int(uinfo),) * 6, law, r_bar=True)
    return one * (duration_symbols // 24)


def _highest_power_interval_ucodes(profile: dict, law: str) -> tuple[int, ...]:
    """Select each interval's highest-power Ucode from a parsed CP/CPt profile."""
    law = _check_law(law)
    selectors = tuple(int(value) for value in profile["interval_constellations"])
    transmitter_maps = profile.get("constellations")
    codec_maps = profile.get("codec_constellations")
    # CP bit 128 explicitly supplies the post-codec maps when the transmitter
    # constellation and codec-output constellation differ.
    maps = codec_maps if codec_maps is not None else transmitter_maps
    if maps is None:
        raise ValueError("CP profile has no usable constellation maps")
    table = g711_ucode_codewords(law)
    levels = g711.decode(table, law).astype(np.int64)
    selected = []
    for index in selectors:
        if not 0 <= index < len(maps):
            raise ValueError("CP profile selects a missing constellation")
        mask = tuple(int(value) for value in maps[index])
        if not mask or any(not 0 <= value <= 127 for value in mask):
            raise ValueError("CP profile has an empty or invalid Ucode mask")
        selected.append(max(mask, key=lambda ucode: abs(int(levels[ucode]))))
    return tuple(selected)


def rt_codewords(cpt_profile: dict, law: str, *, duration_symbols: int = 24) -> bytes:
    """Return R-t using each CPt training interval's highest-power Ucode."""
    if duration_symbols <= 0 or duration_symbols % 24:
        raise ValueError("R-t duration must be a positive multiple of 24 symbols")
    ucodes = _highest_power_interval_ucodes(cpt_profile, law)
    one = _r_signal(ucodes, law)
    return one * (duration_symbols // 24)


def rd_codewords(cp_profile: dict, law: str, *, duration_symbols: int = 24) -> bytes:
    """Return R-d using each CP data interval's highest-power Ucode."""
    if duration_symbols <= 0 or duration_symbols % 24:
        raise ValueError("R-d duration must be a positive multiple of 24 symbols")
    ucodes = _highest_power_interval_ucodes(cp_profile, law)
    one = _r_signal(ucodes, law)
    return one * (duration_symbols // 24)


class _Gpc:
    """V.34/V.90 self-synchronizing GPC recurrence, initialized to zero."""

    def __init__(self) -> None:
        self.state = 0

    def step(self, bit: int) -> int:
        state = self.state
        scrambled = int(bit) ^ ((state >> 17) & 1) ^ ((state >> 22) & 1)
        self.state = ((state << 1) | scrambled) & 0x7fffff
        return scrambled


class V90DigitalPhase4Encoder:
    """V.90 §5.4 mapper/sign encoder for Phase 4 training.

    CPt supplies the Phase 4 training rate and each interval's positive Ucode
    set. The implementation uses the normative mixed-radix modulus mapping,
    descending-Ucode mapper labels, GPC-scrambled ones, and cross-frame sign
    differential rule. The currently implemented shaper is the normative
    S_r=1, look-ahead-depth-zero path; unsupported profiles fail closed.
    """

    def __init__(self, cpt_profile: dict, law: str):
        self.law = _check_law(law)
        self.sr = int(cpt_profile["spectral_sign_bits"])
        self.lookahead = int(cpt_profile["lookahead"])
        if self.sr not in (0, 1):
            raise NotImplementedError("Phase 4 supports S_r=0 or S_r=1 only")
        if self.sr == 1 and self.lookahead != 0:
            raise NotImplementedError("S_r=1 shaping currently requires look-ahead depth zero")
        self.s = 6 - self.sr
        rate = float(cpt_profile["rate"])
        total_bits = round(rate * 6 / 8000)
        if abs(rate - total_bits * 8000 / 6) > 1e-6:
            raise ValueError("CPt training rate is not on the V.90 frame-rate grid")
        self.k = total_bits - self.s
        if not 1 <= self.k <= 39:
            raise ValueError("CPt training rate gives an unsupported modulus width")

        selectors = tuple(int(v) for v in cpt_profile["interval_constellations"])
        maps = (cpt_profile.get("codec_constellations")
                or cpt_profile.get("constellations"))
        if maps is None or len(selectors) != 6:
            raise ValueError("CPt profile must define six selected constellations")
        self.code_sets = []
        for index in selectors:
            if not 0 <= index < len(maps):
                raise ValueError("CPt selects a missing training constellation")
            points = sorted({int(v) for v in maps[index]}, reverse=True)
            if not points or any(not 0 <= point <= 127 for point in points):
                raise ValueError("CPt contains an empty or invalid Ucode set")
            self.code_sets.append(tuple(points))
        product = 1
        for points in self.code_sets:
            product *= len(points)
        if (1 << self.k) > product:
            raise ValueError("CPt constellation product cannot represent the training rate")

        self.gpc = _Gpc()
        self.previous_sign5 = 0
        self.ucode_octets = g711_ucode_codewords(self.law)
        self.levels = g711.decode(self.ucode_octets, self.law).astype(np.float64)
        self.shape_filter = tuple(float(v) for v in cpt_profile.get("shape", ()))
        if self.sr == 1 and len(self.shape_filter) != 4:
            raise ValueError("S_r=1 CPt profile must provide a1,a2,b1,b2")
        self.shape_q = 0
        self.shape_p_prime_5 = 0
        self.shape_t_previous = [0] * 6
        self.shape_filter_state = (0.0, 0.0, 0.0)
        self._pending_bits: list[int] = []

    def reset(self) -> None:
        """Reset mapper memories for a new V.90 training/data block."""
        self.gpc = _Gpc()
        self.previous_sign5 = 0
        self.shape_q = 0
        self.shape_p_prime_5 = 0
        self.shape_t_previous = [0] * 6
        self.shape_filter_state = (0.0, 0.0, 0.0)
        self._pending_bits.clear()

    def _shape_rule(self, t: Sequence[int], magnitudes: Sequence[float]):
        """Choose the minimum-metric legal rule for one Sr=1 frame (ld=0)."""
        a1, a2, b1, b2 = self.shape_filter
        # Figure 2 state transitions: A/B leave Q0 for Q0/Q1; C/D leave
        # Q1 for Q0/Q1. Rule order is also the deterministic tie-break order.
        rules = (("A", self.shape_q, lambda k: t[k]),
                 ("B", 1, lambda k: t[k] ^ 1)) if self.shape_q == 0 else (
                 ("C", 0, lambda k: t[k] ^ (k % 2 == 0)),
                 ("D", 1, lambda k: t[k] ^ (k % 2 == 1)))
        x_prev, y_prev, v_prev = self.shape_filter_state
        scored = []
        for name, next_q, sign_fn in rules:
            x1, y1, v1 = x_prev, y_prev, v_prev
            energy = 0.0
            signs = [int(sign_fn(k)) for k in range(6)]
            for sign, magnitude in zip(signs, magnitudes):
                x = float(magnitude) if sign else -float(magnitude)
                y = x - b1 * x1 + a1 * y1
                v = y - b2 * y1 + a2 * v1
                energy += v * v
                x1, y1, v1 = x, y, v
            scored.append((energy, name, next_q, signs, (x1, y1, v1)))
        energy, name, self.shape_q, signs, self.shape_filter_state = min(scored)
        return signs

    def _encode_frame(self, input_bits: Sequence[int]) -> bytes:
        """Map one frame of D=S+K input bits to six G.711 codewords."""
        if len(input_bits) != self.s + self.k:
            raise ValueError("V.90 frame needs exactly S+K input bits")
        if any(bit not in (0, 1) for bit in input_bits):
            raise ValueError("V.90 encoder input must contain only bits")
        d = [self.gpc.step(int(bit)) for bit in input_bits]
        sign_inputs = d[:self.s]
        modulus_bits = d[self.s:]
        value = sum(bit << index for index, bit in enumerate(modulus_bits))
        mapped_indices = []
        for points in self.code_sets:
            index = value % len(points)
            value //= len(points)
            mapped_indices.append(index)

        if self.sr == 0:
            signs = [sign_inputs[0] ^ self.previous_sign5]
            for bit in sign_inputs[1:]:
                signs.append(int(bit) ^ signs[-1])
            self.previous_sign5 = signs[-1]
        else:
            # Tables 3/4: parse five input sign bits, apply odd-bit
            # differential coding, then the second differential stage.
            p = [0, *map(int, sign_inputs)]
            p_prime = [0] * 6
            p_prime[1] = p[1] ^ self.shape_p_prime_5
            p_prime[2] = p[2]
            p_prime[3] = p[3] ^ p_prime[1]
            p_prime[4] = p[4]
            p_prime[5] = p[5] ^ p_prime[3]
            t = [p_prime[k] ^ self.shape_t_previous[k] for k in range(6)]
            self.shape_p_prime_5 = p_prime[5]
            self.shape_t_previous = t
            signs = self._shape_rule(t, [self.levels[points[index]]
                                         for points, index in zip(self.code_sets, mapped_indices)])
        return bytes(int(self.ucode_octets[points[index]]) ^ (0 if sign else 0x80)
                     for points, index, sign in zip(self.code_sets, mapped_indices, signs))

    def encode_bits(self, bits: Sequence[int]) -> bytes:
        """Stream arbitrary V.90 input bits as complete six-symbol frames.

        Each frame consumes ``S+K`` serialized input bits and emits six 8 kHz
        G.711 codewords. Incomplete trailing bits are retained for the next
        call, which lets MP frames cross mapper-frame boundaries without
        inserting non-standard padding. Use :meth:`reset` at a standard-defined
        block reset.
        """
        if any(bit not in (0, 1) for bit in bits):
            raise ValueError("V.90 encoder input must contain only bits")
        self._pending_bits.extend(int(bit) for bit in bits)
        frame_bits = self.s + self.k
        complete = len(self._pending_bits) // frame_bits
        count = complete * frame_bits
        encoded = b"".join(self._encode_frame(self._pending_bits[offset:offset + frame_bits])
                           for offset in range(0, count, frame_bits))
        del self._pending_bits[:count]
        return encoded

    def trn2d(self, symbols: int = 2040) -> bytes:
        """Generate TRN2d: at least 2040 symbols, in six-symbol frames."""
        if symbols < 2040 or symbols % 6:
            raise ValueError("TRN2d length must be >=2040 and divisible by six")
        self.reset()
        frame = [1] * (self.s + self.k)
        return b"".join(self._encode_frame(frame) for _ in range(symbols // 6))
