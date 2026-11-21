"""V.92 analogue-modem Phase 3 transmit, first half (clause 9.5.2.1.1 to 9.5.2.1.3).

After INFO1a the analogue modem sends 70 +- 5 ms of silence, Ru for 384T and Ru-bar for 24T, then
(if INFO1a asked for MD) MD, Ru, Ru-bar again, then TRN1u for at least 2040T, then Ja. All are
one PCM sample per symbol (8000 symbols/s) at the level LU. This builds that stretch as a
sample-exact plan with the segment boundaries, so a controller can schedule it on the RTP clock
and the digital side can be tested against it. It ends where Ja ends; the rest of Phase 3 (stop Ja
at the Sd-to-Sd-bar transition, Su, wait for Jp/Jp', TRN1u, CPt, E1u) reacts to the digital modem
and is not here.

Ja (8.5.4): 24 binary ones, then repetitions of the DIL descriptor, scrambled and differentially
encoded (present bit XOR previously transmitted bit); the differential memory starts as the final
TRN1u symbol; the scrambler carries on from TRN1u. A transmitted 1 is -LU.

Not implemented: MD (V.34 10.1.3.5), so INFO1a with a non-zero MD length is refused.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import v34_tx
import v92_phase3

SILENCE_SAMPLES = 560           # 70 ms
RU_SYMBOLS = 384
RU_BAR_SYMBOLS = 24
TRN1U_MIN = 2040
DEFAULT_TRN1U = 2052            # 171 x 12: the first multiple of 12 above 2040
DEFAULT_JA_BITS = 24 + 276 * 2


@dataclass(frozen=True)
class Phase3Plan:
    samples: np.ndarray                 # at LU = the requested level
    boundaries: dict                    # name -> (start, end) sample index
    ja_bits: tuple                      # the bits Ja carried (before scrambling)

    @property
    def length(self) -> int:
        return len(self.samples)


def ja_bit_stream(descriptor, bits: int) -> list[int]:
    """24 ones then the descriptor repeated, cut to ``bits`` (a multiple of 12)."""
    if bits % 12 or bits < 24 + 12:
        raise ValueError("Ja length must be a multiple of 12 bits and longer than its preamble")
    descriptor = [int(b) & 1 for b in descriptor]
    if not descriptor:
        raise ValueError("Ja needs a DIL descriptor")
    out = [1] * 24
    while len(out) < bits:
        out.extend(descriptor)
    return out[:bits]


def build(descriptor, *, lu: float = 1000.0, md_symbols: int = 0,
          trn1u_symbols: int = DEFAULT_TRN1U, ja_bits: int = DEFAULT_JA_BITS,
          silence_samples: int = SILENCE_SAMPLES) -> Phase3Plan:
    if md_symbols:
        raise NotImplementedError("MD (V.34 10.1.3.5) is not implemented; INFO1a must ask for 0")
    if lu <= 0:
        raise ValueError("LU must be positive")
    if trn1u_symbols < TRN1U_MIN or trn1u_symbols % 12:
        raise ValueError("TRN1u is at least 2040 symbols and a multiple of 12")
    if not 0 <= silence_samples:
        raise ValueError("silence must not be negative")
    parts: list[tuple[str, np.ndarray]] = [("silence", np.zeros(silence_samples))]
    parts.append(("ru", np.array(v92_phase3.ru(RU_SYMBOLS // 6)) * lu))
    parts.append(("ru-bar", np.array(v92_phase3.ru(RU_BAR_SYMBOLS // 6, bar=True)) * lu))

    scrambler = v34_tx.Scrambler("answer")
    trn_bits = [scrambler.step(1) for _ in range(trn1u_symbols)]
    parts.append(("trn1u", np.array([-lu if b else lu for b in trn_bits])))

    previous = trn_bits[-1]                    # differential memory: last TRN1u symbol
    bits = ja_bit_stream(descriptor, ja_bits)
    ja_symbols = []
    for bit in bits:
        previous ^= scrambler.step(bit)
        ja_symbols.append(-lu if previous else lu)
    parts.append(("ja", np.array(ja_symbols)))

    boundaries, at = {}, 0
    for name, piece in parts:
        boundaries[name] = (at, at + len(piece))
        at += len(piece)
    return Phase3Plan(np.concatenate([p for _, p in parts]), boundaries, tuple(bits))


def recover_ja_bits(symbols, previous_symbol_bit: int, scrambler_history) -> list[int]:
    """Inverse of the Ja coding, for tests and for a receiver: symbols (sign only) back to
    the Ja bits. ``scrambler_history`` is the scrambler output sequence (TRN1u) before Ja."""
    tap, reg = 5, list(reversed([int(b) for b in scrambler_history[-23:]]))
    out, previous = [], int(previous_symbol_bit)
    for s in symbols:
        transmitted = 1 if s < 0 else 0
        scrambled = transmitted ^ previous
        previous = transmitted
        out.append(scrambled ^ reg[tap - 1] ^ reg[22])
        reg = [scrambled] + reg[:-1]
    return out
