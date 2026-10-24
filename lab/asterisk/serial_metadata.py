"""Versioned metadata framing for optional byte-serial side channels."""

from __future__ import annotations

import struct


_BAUD_FRAME = struct.Struct("!4sBI")
_FORMAT_FRAME = struct.Struct("!4sBBIB")
_MAGIC = b"ELSB"
VERSION = 1
FORMAT_VERSION = 2
_KIND_BAUD = 1
_KIND_FORMAT = 2


def encode_baud(rate: int) -> bytes:
    """Encode one nominal DTE baud update."""
    value = int(rate)
    if not 1 <= value <= 4_000_000:
        raise ValueError("baud metadata outside supported framing range")
    return _BAUD_FRAME.pack(_MAGIC, VERSION, value)


def encode_character_format(format_code: int, parity_code: int) -> bytes:
    """Encode a V.250 +ICF framing update (format code and parity code)."""
    fmt, parity = int(format_code), int(parity_code)
    if fmt not in range(1, 7) or parity not in range(4):
        raise ValueError("invalid character-format metadata")
    return _FORMAT_FRAME.pack(_MAGIC, FORMAT_VERSION, _KIND_FORMAT, fmt, parity)


def take_metadata_frames(buffer: bytearray) -> list[tuple[str, int, int]]:
    """Consume complete frames, returning (kind, value, auxiliary) triples."""
    frames = []
    while len(buffer) >= 5:
        magic, version = struct.unpack("!4sB", buffer[:5])
        if magic != _MAGIC:
            del buffer[0]
            continue
        size = _BAUD_FRAME.size if version == VERSION else _FORMAT_FRAME.size if version == FORMAT_VERSION else 0
        if not size:
            del buffer[:5]
            continue
        if len(buffer) < size:
            break
        if version == VERSION:
            _magic, _version, value = _BAUD_FRAME.unpack(bytes(buffer[:size]))
            frames.append(("baud", value, 0))
        else:
            _magic, _version, kind, value, aux = _FORMAT_FRAME.unpack(bytes(buffer[:size]))
            if kind == _KIND_FORMAT:
                frames.append(("format", value, aux))
        del buffer[:size]
    return frames


def take_baud_frames(buffer: bytearray) -> list[tuple[int, int]]:
    """Backward-compatible baud-only parser."""
    result = []
    for kind, value, _aux in take_metadata_frames(buffer):
        if kind == "baud":
            result.append((VERSION, value))
    return result
