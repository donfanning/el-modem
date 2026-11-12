"""CRC-checked V.90 Phase 4 CP/CPt parameter parser (Recommendation §8.5.2).

CP carries the analogue modem's selected digital-downstream constellation and
rate parameters. This parser validates the variable-length frame and exposes
the Ucode masks without making a policy decision about unsupported profiles.
"""

from __future__ import annotations

from numbers import Integral

from v34_info import crc16

SYNC = (1,) * 17 + (0,)
SYNC_BYTES = bytes(SYNC)


def _value(bits, lo: int, width: int) -> int:
    return sum(int(bits[lo + offset]) << offset for offset in range(width))


def _signed(value: int, width: int) -> int:
    return value - (1 << width) if value & (1 << (width - 1)) else value


def _mask(bits, start: int) -> tuple[int, ...]:
    """Read one 128-bit Ucode mask from a block with 16-bit groups."""
    return tuple(group * 16 + bit
                 for group in range(8) for bit in range(16)
                 if bits[start + 1 + group * 17 + bit])


def cp_frame_length(header) -> int | None:
    """Return the CP frame length once its fixed 129-bit header is available."""
    data = [int(bit) for bit in header]
    if len(data) < 129 or data[:18] != list(SYNC):
        return None
    if any(bit not in (0, 1) for bit in data[:129]):
        return None
    if data[18] or any(data[index] for index in range(25, 30)):
        return None
    if any(data[index] for index in (34, 51, 68, 85, 102, 119)):
        return None
    if _value(data, 20, 5) > 22:
        return None
    maximum = max(_value(data, lo, 4)
                  for lo in (103, 107, 111, 115, 120, 124))
    if maximum > 5:
        return None
    gamma = 136 * maximum
    delta = 2 * gamma + 136 if data[128] else gamma
    return 272 + delta + 20


def parse_cp(bits) -> dict | None:
    """Parse one complete CPt/CP frame beginning at its first sync bit.

    Returns None for bad framing, CRC, reserved bits, start bits, or invalid
    constellation selections. The parsed rate is in bit/s; ``constellations``
    and optional ``codec_constellations`` are tuples of enabled Ucode values.
    """
    data = [int(bit) for bit in bits]
    if len(data) < 292 or any(bit not in (0, 1) for bit in data[:292]):
        return None
    if data[:17] != [1] * 17 or data[17] != 0:
        return None
    if data[18] != 0 or any(data[index] for index in range(25, 30)):
        return None
    if data[34] or data[51] or data[68] or data[85] or data[102] or data[119]:
        return None
    if data[129:136] != [0] * 7:
        return None

    cp_type = data[19]
    drn = _value(data, 20, 5)
    if drn > 22:
        return None
    # DRN zero is the protocol's cleardown indication, not a low data rate.
    rate = (drn + (8 if cp_type == 0 else 20)) * 8000 / 6 if drn else None
    sr = _value(data, 31, 2)
    codec = "pcma" if data[35] else "pcmu"
    # Four interval selectors precede the 119 start bit; the final two follow.
    max_constellation = max(
        *(_value(data, lo, 4) for lo in (103, 107, 111, 115, 120, 124)))
    if max_constellation > 5:
        return None

    codec_masks_present = bool(data[128])
    constellation_count = max_constellation + 1
    gamma = 136 * max_constellation
    # δ includes the base codec constellation as well as additional maps.
    # With bit 128 clear, only the extra transmitter masks follow.
    expected_length = cp_frame_length(data[:129])
    if expected_length is None:
        return None
    crc_start = expected_length - 20
    if len(data) < expected_length:
        return None
    if any(bit not in (0, 1) for bit in data[:expected_length]):
        return None

    starts = {34, 51, 68, 85, 102, 119, 136}
    for block_start in range(272, 272 + gamma, 136):
        starts.add(block_start)
    if codec_masks_present:
        codec_start = 272 + gamma
        starts.add(codec_start)
        for block_start in range(codec_start, codec_start + gamma + 136, 136):
            starts.add(block_start)
    for block_start in range(136, 272 + gamma, 136):
        starts.update(block_start + 17 * group for group in range(1, 8))
    if codec_masks_present:
        for block_start in range(272 + gamma, 272 + 2 * gamma + 136, 136):
            starts.update(block_start + 17 * group for group in range(1, 8))
    starts.add(crc_start)
    if any(data[index] for index in starts):
        return None
    if any(data[index] for index in range(expected_length - 3, expected_length)):
        return None

    information = [bit for index, bit in enumerate(data[18:crc_start], start=18)
                   if index not in starts]
    if _value(data, crc_start + 1, 16) != crc16(information):
        return None

    interval_indexes = tuple(_value(data, lo, 4)
                             for lo in (103, 107, 111, 115, 120, 124))
    if any(index > 5 for index in interval_indexes):
        return None
    constellations = tuple(
        tuple(_mask(data, 136 + index * 136))
        for index in range(constellation_count))
    codec_constellations = None
    if codec_masks_present:
        codec_constellations = tuple(
            tuple(_mask(data, 272 + gamma + index * 136))
            for index in range(constellation_count))
    if any(not constellations[index] for index in interval_indexes):
        return None
    if (codec_constellations is not None
            and any(not codec_constellations[index] for index in interval_indexes)):
        return None

    return {
        "kind": "cpt" if cp_type == 0 else "cp",
        "drn": drn,
        "rate": rate,
        "cleardown": drn == 0,
        "silent_requested": bool(data[30]),
        "spectral_sign_bits": sr,
        "ack": data[33],
        "codec": codec,
        "analog_rate_mask": _value(data, 36, 13),
        "lookahead": _value(data, 49, 2),
        "gain": _value(data, 52, 16) / 8192.0,
        "shape": tuple(_signed(_value(data, lo, 8), 8) / 64.0
                       for lo in (69, 77, 86, 94)),
        "interval_constellations": interval_indexes,
        "codec_constellations_present": codec_masks_present,
        "constellations": constellations,
        "codec_constellations": codec_constellations,
        "frame_bits": expected_length,
    }


def build_cp_frame(*, kind: str, drn: int, selectors, tx_masks,
                   codec_masks=None, law: str = "pcmu", ack: int = 1,
                   spectral_sign_bits: int = 1,
                   analog_rate_mask: int = 0, lookahead: int = 0,
                   gain_q13: int = 0x4000,
                   shape_q6=(0x10, 0xF0, 0x08, 0xE8)) -> list[int]:
    """Build a CRC-protected CPt/CP information frame (V.90 §8.5.2).

    Masks are ordered by constellation index and contain enabled Ucodes in
    ``0..127``. This is a wire-format encoder, not a policy for choosing the
    caller's downstream rate or constellation. ``gain_q13`` and
    ``shape_q6`` are the encoded integer fields, avoiding floating-point
    rounding at the protocol boundary.
    """
    if kind not in ("cpt", "cp"):
        raise ValueError("CP frame kind must be 'cpt' or 'cp'")
    integer_fields = {
        "drn": (drn, 0, 22),
        "ack": (ack, 0, 1),
        "spectral_sign_bits": (spectral_sign_bits, 0, 3),
        "analog_rate_mask": (analog_rate_mask, 0, 0x1FFF),
        "lookahead": (lookahead, 0, 3),
        "gain_q13": (gain_q13, 0, 0xFFFF),
    }
    for name, (value, low, high) in integer_fields.items():
        if isinstance(value, bool) or not isinstance(value, Integral) or not low <= value <= high:
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if law not in ("pcmu", "pcma"):
        raise ValueError("law must be 'pcmu' or 'pcma'")
    selected = tuple(selectors)
    if len(selected) != 6 or any(
            isinstance(value, bool) or not isinstance(value, Integral)
            or not 0 <= value <= 5 for value in selected):
        raise ValueError("CP requires six constellation selectors in 0..5")
    maximum = max(selected)

    def validate_masks(name, masks):
        if len(masks) != maximum + 1:
            raise ValueError(f"{name} must define masks 0..{maximum}")
        result = []
        for index, mask in enumerate(masks):
            values = tuple(mask)
            if (not values or any(isinstance(value, bool)
                                  or not isinstance(value, Integral)
                                  or not 0 <= value <= 127 for value in values)
                    or len(set(values)) != len(values)):
                raise ValueError(f"{name}[{index}] must be unique nonempty Ucodes in 0..127")
            result.append(values)
        if any(not result[index] for index in selected):
            raise ValueError(f"{name} is empty for a selected constellation")
        return tuple(result)

    transmitter_masks = validate_masks("tx_masks", tx_masks)
    codec_mask_values = (None if codec_masks is None
                         else validate_masks("codec_masks", codec_masks))
    shape = tuple(shape_q6)
    if len(shape) != 4 or any(
            isinstance(value, bool) or not isinstance(value, Integral)
            or not -128 <= value <= 127 for value in shape):
        raise ValueError("shape_q6 must contain four signed 8-bit integers")

    gamma = 136 * maximum
    delta = 2 * gamma + 136 if codec_mask_values is not None else gamma
    crc_start = 272 + delta
    bits = [0] * (crc_start + 20)
    bits[:17] = [1] * 17
    bits[19] = int(kind == "cp")

    def set_field(start, width, value):
        encoded = int(value) & ((1 << width) - 1)
        for offset in range(width):
            bits[start + offset] = (encoded >> offset) & 1

    set_field(20, 5, drn)
    bits[31:33] = [(spectral_sign_bits >> bit) & 1 for bit in range(2)]
    bits[33] = int(ack)
    bits[35] = int(law == "pcma")
    set_field(36, 13, analog_rate_mask)
    set_field(49, 2, lookahead)
    set_field(52, 16, gain_q13)
    for start, value in zip((69, 77, 86, 94), shape):
        set_field(start, 8, value)
    for start, value in zip((103, 107, 111, 115, 120, 124), selected):
        set_field(start, 4, value)
    bits[128] = int(codec_mask_values is not None)

    def put_mask(block_start, codes):
        for code in codes:
            group, bit = divmod(int(code), 16)
            bits[block_start + 1 + 17 * group + bit] = 1

    for index, codes in enumerate(transmitter_masks):
        put_mask(136 + index * 136, codes)
    if codec_mask_values is not None:
        codec_start = 272 + gamma
        for index, codes in enumerate(codec_mask_values):
            put_mask(codec_start + index * 136, codes)

    starts = {34, 51, 68, 85, 102, 119, 136, crc_start}
    starts.update(range(272, 272 + gamma, 136))
    starts.update(range(272 + gamma, 272 + 2 * gamma + 136, 136))
    for block_start in range(136, 272 + gamma, 136):
        starts.update(block_start + 17 * group for group in range(1, 8))
    if codec_mask_values is not None:
        for block_start in range(272 + gamma, 272 + 2 * gamma + 136, 136):
            starts.update(block_start + 17 * group for group in range(1, 8))
    information = [bit for index, bit in enumerate(bits[18:crc_start], 18)
                   if index not in starts]
    set_field(crc_start + 1, 16, crc16(information))

    # Fail closed if encoder layout changes without a matching parser update.
    parsed = parse_cp(bits)
    if parsed is None or parsed["kind"] != kind:
        raise RuntimeError("generated CP frame did not pass the local parser")
    return bits


class V90CpScanner:
    """Incrementally find CRC-valid CP/CPt frames in descrambled bits."""

    def __init__(self, *, max_buffer_bits: int = 8192):
        if max_buffer_bits < 1788:
            raise ValueError("CP scanner buffer must fit the largest legal frame")
        self.max_buffer_bits = int(max_buffer_bits)
        # The scanner is fed in very small chunks by every speculative CPt
        # timing candidate. Store bits as bytes so the hot sync comparison is
        # performed by the C byte-sequence comparator instead of allocating a
        # list slice and tuple for every possible bit offset.
        self._bits = bytearray()
        self._cursor = 0
        self._base_offset = 0

    @property
    def bits_seen(self) -> int:
        return self._base_offset + len(self._bits)

    def feed(self, bits) -> list[dict]:
        chunk = [int(bit) for bit in bits]
        if any(bit not in (0, 1) for bit in chunk):
            raise ValueError("CP scanner input must contain only bits")
        return self._feed_validated(chunk)

    def _feed_validated(self, bits, *, empty_none: bool = False) -> list[dict] | None:
        """Scan bits produced internally by the checked symbol decoder."""
        if not bits:
            return None if empty_none else []
        chunk = bits
        self._bits.extend(chunk)
        found = None

        while self._cursor + len(SYNC) <= len(self._bits):
            # The speculative candidate bank runs many independent scanners
            # on short chunks. Search the byte-per-bit stream in C instead of
            # checking each possible sync offset in Python.
            at = self._bits.find(SYNC_BYTES, self._cursor)
            if at < 0:
                self._cursor = max(
                    self._cursor,
                    len(self._bits) - len(SYNC) + 1,
                )
                break
            self._cursor = at
            if at + 129 > len(self._bits):
                break
            frame_length = cp_frame_length(self._bits[at:at + 129])
            if frame_length is None:
                self._cursor += 1
                continue
            if at + frame_length > len(self._bits):
                break
            parsed = parse_cp(self._bits[at:at + frame_length])
            if parsed is None:
                self._cursor += 1
                continue
            parsed["start_bit"] = self._base_offset + at
            if found is None:
                found = []
            found.append(parsed)
            self._cursor = at + frame_length

        # Retain only an unconsumed partial frame or possible sync suffix.
        self._cursor = min(self._cursor, max(0, len(self._bits) - len(SYNC) + 1))
        if self._cursor:
            del self._bits[:self._cursor]
            self._base_offset += self._cursor
            self._cursor = 0
        if len(self._bits) > self.max_buffer_bits:
            raise BufferError("CP scanner retained more than its configured bound")
        return ([] if found is None and not empty_none else found)


class V90CpSymbolDecoder:
    """Descramble V.90 analogue CP symbol decisions and scan for CP/CPt.

    Input symbols are ``(rotation, quarter)`` pairs from a trained V.34-style
    phase-four slicer. For four-point signaling, quarter must be zero. The
    analogue modem uses the V.34 GPA polynomial; the differential encoder and
    scrambler are both initialized to zero before the first CPt frame. Keep one
    instance for the entire CPt/CP/CP′ exchange so their state remains
    continuous across frames.

    This starts at the first symbol of the first CPt stream. It does not find
    symbol timing, carrier phase, the CPt onset, or a constellation itself.
    """

    SCRAMBLER_TAP = 5  # GPA, V.90 §6.5 / V.34 §7

    def __init__(self, constellation: int):
        if constellation not in (4, 16):
            raise ValueError("CP constellation must be 4 or 16 points")
        self.constellation = constellation
        self._previous_rotation = 0
        self._scrambled_history = 0
        self.scanner = V90CpScanner()
        self.last_descrambled_bits: tuple[int, ...] = ()

    @property
    def bits_seen(self) -> int:
        """Absolute count of descrambled bits emitted so far."""
        return self.scanner.bits_seen

    def feed(self, symbols) -> list[dict]:
        decoded = []
        for rotation, quarter in symbols:
            if (not isinstance(rotation, Integral) or isinstance(rotation, bool)
                    or not isinstance(quarter, Integral) or isinstance(quarter, bool)
                    or not 0 <= rotation < 4 or not 0 <= quarter < 4):
                raise ValueError("CP symbol decisions must be integer rotation/quarter indices in 0..3")
            rotation, quarter = int(rotation), int(quarter)
            if self.constellation == 4 and quarter != 0:
                raise ValueError("four-point CP decisions cannot carry a quarter index")
            differential = (rotation - self._previous_rotation) % 4
            self._previous_rotation = rotation
            raw = [differential & 1, differential >> 1]
            if self.constellation == 16:
                raw.extend((quarter & 1, quarter >> 1))
            for bit in raw:
                plain = bit ^ ((self._scrambled_history >> 4) & 1) ^ ((self._scrambled_history >> 22) & 1)
                self._scrambled_history = ((self._scrambled_history << 1) | bit) & 0x7FFFFF
                decoded.append(plain)
        self.last_descrambled_bits = tuple(decoded)
        return self.scanner.feed(decoded)

    def feed_symbol(self, rotation: int, quarter: int) -> list[dict]:
        """Decode one already-sliced symbol without singleton-list overhead."""
        if (not isinstance(rotation, Integral) or isinstance(rotation, bool)
                or not isinstance(quarter, Integral) or isinstance(quarter, bool)
                or not 0 <= rotation < 4 or not 0 <= quarter < 4
                or (self.constellation == 4 and quarter != 0)):
            raise ValueError("CP symbol decisions must be integer rotation/quarter indices in 0..3")
        rotation, quarter = int(rotation), int(quarter)
        return self._feed_symbol_unchecked(rotation, quarter)

    def _feed_symbol_unchecked(self, rotation: int, quarter: int, *,
                               capture_last_bits: bool = True,
                               empty_none: bool = False) -> list[dict] | None:
        """Decode decisions already validated by the modem's internal slicer."""
        decoded = self._descramble_symbol_unchecked(rotation, quarter)
        if capture_last_bits and decoded:
            bits_per_symbol = 4 if self.constellation == 16 else 2
            self.last_descrambled_bits = tuple(decoded[-bits_per_symbol:])
        return self.scanner._feed_validated(decoded, empty_none=empty_none)

    def _descramble_symbol_unchecked(self, rotation: int,
                                     quarter: int) -> tuple[int, ...]:
        """Update differential/GPA state and return this symbol's bits."""
        differential = (rotation - self._previous_rotation) % 4
        self._previous_rotation = rotation
        history = self._scrambled_history
        bit0 = differential & 1
        bit1 = differential >> 1
        tap = ((history >> 4) ^ (history >> 22)) & 1
        plain0 = bit0 ^ tap
        history = ((history << 1) | bit0) & 0x7FFFFF
        tap = ((history >> 4) ^ (history >> 22)) & 1
        plain1 = bit1 ^ tap
        history = ((history << 1) | bit1) & 0x7FFFFF
        if self.constellation == 16:
            bit2 = quarter & 1
            bit3 = quarter >> 1
            tap = ((history >> 4) ^ (history >> 22)) & 1
            plain2 = bit2 ^ tap
            history = ((history << 1) | bit2) & 0x7FFFFF
            tap = ((history >> 4) ^ (history >> 22)) & 1
            plain3 = bit3 ^ tap
            history = ((history << 1) | bit3) & 0x7FFFFF
            decoded = (plain0, plain1, plain2, plain3)
        else:
            decoded = (plain0, plain1)
        self._scrambled_history = history
        return decoded

    def _feed_symbols_unchecked(self, symbols, *,
                                capture_last_bits: bool = False,
                                empty_none: bool = False) -> list[dict] | None:
        """Decode a small symbol batch, scanning once instead of per symbol."""
        previous_rotation = self._previous_rotation
        history = self._scrambled_history
        decoded = bytearray()
        if self.constellation == 16:
            for rotation, quarter in symbols:
                differential = (rotation - previous_rotation) & 3
                previous_rotation = rotation
                bit0 = differential & 1
                bit1 = differential >> 1
                decoded.append(bit0 ^ (((history >> 4) ^ (history >> 22)) & 1))
                history = ((history << 1) | bit0) & 0x7FFFFF
                decoded.append(bit1 ^ (((history >> 4) ^ (history >> 22)) & 1))
                history = ((history << 1) | bit1) & 0x7FFFFF
                bit2 = quarter & 1
                decoded.append(bit2 ^ (((history >> 4) ^ (history >> 22)) & 1))
                history = ((history << 1) | bit2) & 0x7FFFFF
                bit3 = quarter >> 1
                decoded.append(bit3 ^ (((history >> 4) ^ (history >> 22)) & 1))
                history = ((history << 1) | bit3) & 0x7FFFFF
        else:
            for rotation, _quarter in symbols:
                differential = (rotation - previous_rotation) & 3
                previous_rotation = rotation
                bit0 = differential & 1
                bit1 = differential >> 1
                decoded.append(bit0 ^ (((history >> 4) ^ (history >> 22)) & 1))
                history = ((history << 1) | bit0) & 0x7FFFFF
                decoded.append(bit1 ^ (((history >> 4) ^ (history >> 22)) & 1))
                history = ((history << 1) | bit1) & 0x7FFFFF
        self._previous_rotation = previous_rotation
        self._scrambled_history = history
        if capture_last_bits:
            self.last_descrambled_bits = tuple(decoded)
        return self.scanner._feed_validated(decoded, empty_none=empty_none)
