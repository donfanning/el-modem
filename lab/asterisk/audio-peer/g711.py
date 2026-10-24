"""Small dependency-free G.711 PCMU primitives for the live audio path."""


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


__all__ = ["linear_to_ulaw", "ulaw_to_linear"]
