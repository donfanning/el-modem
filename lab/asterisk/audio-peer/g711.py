"""Small dependency-free G.711 PCMU/PCMA primitives for the live audio path."""


def ulaw_to_linear(sample: int) -> int:
    value = (~sample) & 0xFF
    magnitude = ((value & 0x0F) << 3) + 0x84
    magnitude <<= (value & 0x70) >> 4
    return 0x84 - magnitude if value & 0x80 else magnitude - 0x84


def linear_to_ulaw(sample: int) -> int:
    sign = 0x80 if sample < 0 else 0
    magnitude = min(abs(int(sample)), 32635) + 0x84
    exponent = 0
    for threshold in (0xFF, 0x1FF, 0x3FF, 0x7FF, 0xFFF, 0x1FFF, 0x3FFF):
        if magnitude <= threshold:
            break
        exponent += 1
    mantissa = (magnitude >> (exponent + 3)) & 0x0F
    return (~(sign | (exponent << 4) | mantissa)) & 0xFF


_ALAW_SEGMENT_END = (0xFF, 0x1FF, 0x3FF, 0x7FF, 0xFFF, 0x1FFF, 0x3FFF, 0x7FFF)


def alaw_to_linear(sample: int) -> int:
    """Decode one ITU-T G.711 A-law codeword to signed linear PCM."""
    code = int(sample) & 0xFF
    value = code ^ 0x55
    segment = (value & 0x70) >> 4
    magnitude = (value & 0x0F) << 4
    if segment == 0:
        magnitude += 8
    elif segment == 1:
        magnitude += 0x108
    else:
        magnitude = (magnitude + 0x108) << (segment - 1)
    return magnitude if code & 0x80 else -magnitude


def linear_to_alaw(sample: int) -> int:
    """Encode one signed linear PCM sample as an ITU-T G.711 A-law codeword."""
    value = max(-32768, min(32767, int(sample)))
    if value >= 0:
        mask = 0xD5
    else:
        mask = 0x55
        value = -value - 1
    segment = next((i for i, end in enumerate(_ALAW_SEGMENT_END) if value <= end), 8)
    if segment >= 8:
        return 0x7F ^ mask
    code = segment << 4
    code |= ((value >> 4) & 0x0F) if segment < 2 else ((value >> (segment + 3)) & 0x0F)
    return code ^ mask


__all__ = ["linear_to_ulaw", "ulaw_to_linear", "linear_to_alaw", "alaw_to_linear"]
