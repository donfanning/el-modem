"""Incremental V.90 Ja descriptor framing for a descrambled bit stream.

This parser is independent of the analogue waveform/equalizer.  A Phase 3
receiver supplies bits after the known-one TRN boundary; only complete,
CRC-valid descriptors with valid start/reserved fields are emitted.
"""

from __future__ import annotations

from v90_info_reference import parse_ja_descriptor


JA_SYNC = (1,) * 17 + (0,)


def ja_descriptor_bits_required(header) -> int | None:
    """Return the complete Ja descriptor plus two fill bits from its 51-bit header."""
    data = [int(bit) for bit in header]
    if len(data) < 51 or tuple(data[:18]) != JA_SYNC:
        return None
    if any(bit not in (0, 1) for bit in data[:51]):
        return None
    if any(data[index] for index in (17, 34, 50)):
        return None
    n = _value(data, 18, 25)
    lsp = _value(data, 35, 41) + 1
    ltp = _value(data, 43, 49) + 1
    alpha = ((lsp + 15) // 16) * 17
    beta = ((ltp + 15) // 16) * 17
    terminal = 51 + alpha + beta + 4 * 17 + 4 * 17 + ((n + 1) // 2) * 17
    return terminal + 1 + 16 + 2


class V90JaScanner:
    """Find complete Ja descriptors across arbitrarily split bit chunks."""

    SYNC = JA_SYNC

    def __init__(self):
        self._bits: list[int] = []
        self._cursor = 0

    @property
    def bits_seen(self) -> int:
        return len(self._bits)

    def feed(self, bits) -> list[dict]:
        chunk = [int(bit) for bit in bits]
        if any(bit not in (0, 1) for bit in chunk):
            raise ValueError("Ja scanner input must contain only bits")
        self._bits.extend(chunk)
        found = []

        while self._cursor + len(self.SYNC) <= len(self._bits):
            if tuple(self._bits[self._cursor:self._cursor + len(self.SYNC)]) != self.SYNC:
                self._cursor += 1
                continue

            start = self._cursor
            # The descriptor length fields are available through bit 50.
            if start + 51 > len(self._bits):
                break
            required = ja_descriptor_bits_required(self._bits[start:start + 51])
            if required is None:
                self._cursor += 1
                continue
            required += start
            if required > len(self._bits):
                break

            descriptor = parse_ja_descriptor(self._bits[start:required])
            if (descriptor is not None and descriptor["valid"]
                    and descriptor["crc_valid"] and descriptor["reserved_bits_zero"]):
                descriptor["start_bit"] = start
                found.append(descriptor)
                self._cursor = start + descriptor["descriptor_bits"]
            else:
                # A sync-like sequence with a bad CRC is not framing evidence;
                # continue looking one bit later for a real descriptor.
                self._cursor += 1

        # Keep the possible partial 18-bit sync suffix available next call.
        self._cursor = min(self._cursor, max(0, len(self._bits) - len(self.SYNC) + 1))
        return found


def _value(bits: list[int], lo: int, hi: int) -> int:
    return sum(int(bits[lo + i]) << i for i in range(hi - lo + 1))
