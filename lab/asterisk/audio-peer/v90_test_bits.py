"""Independent test-only V.34 modulo-4 differential/descrambler helper."""
from __future__ import annotations

import numpy as np


def symbols_to_bits(symbols: np.ndarray, trn_end: int, role: str) -> list[int]:
    values = np.asarray(symbols, dtype=np.int64)
    rotated = values.copy()
    if trn_end < 0 or trn_end > len(values):
        raise ValueError("TRN boundary outside symbol stream")
    if trn_end < len(values):
        prior = int(values[trn_end - 1]) if trn_end else 0
        rotated[trn_end] = (int(values[trn_end]) - prior) % 4
        for k in range(trn_end + 1, len(values)):
            rotated[k] = (int(values[k]) - int(values[k - 1])) % 4
    scrambled = [bit for value in rotated
                 for bit in (int(value) & 1, (int(value) >> 1) & 1)]
    if role not in ("answer", "call"):
        raise ValueError("invalid V.34 role")
    tap = 5 if role == "answer" else 18
    history = [0] * 23
    plain = []
    for received in scrambled:
        plain.append(received ^ history[tap - 1] ^ history[22])
        history = [received, *history[:22]]
    return plain
