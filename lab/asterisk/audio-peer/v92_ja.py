"""V.92 Ja DIL descriptor (ITU-T V.92 clause 8.5.4, Table 20).

The V.92 descriptor is the V.90 descriptor (V.90 8.3.1) with the analogue modem's
data-rate capability mask inserted between the terminal start bit and the CRC:

    [V.90 descriptor through its terminal start bit, index T]
    T+1..T+16   mask word 1: bit i = 24,000 + i*8000/6 bit/s, i = 0..15 (24,000..44,000)
    T+17        start bit 0
    T+18..T+33  mask word 2: bit 0..2 = 45,333 / 46,667 / 48,000; bits 3..15 reserved 0
    T+34        start bit 0
    T+35..T+50  CRC (V.34 CRC-16 over every data bit after the sync, start bits excluded)
    T+51        fill bit 0, then fill bits to a multiple of 12 bits

with T = 187 + beta + ceil(N/2)*17 as in V.90. For N = 0 the descriptor ends at bit 272 and is
padded to 276 bits, which is the length the standard states. The common V.90
descriptor CRC excludes frame sync, start, and fill bits; reserved information
bits (including bit 50) remain covered.
"""

from __future__ import annotations

import v34_info as vi
import v90_info_reference as v90

RATE_STEP = 8000 / 6
RATE_BITS = 19                      # 24,000 .. 48,000 in 8000/6 steps
MASK_WORD_BITS = 16


def descriptor_bits_from_prefix(bits) -> int | None:
    """Return the padded V.92 Ja descriptor length once its size fields are present.

    Ja repeats descriptors back-to-back, so a receiver must know the negotiated
    ``N`` before deciding how many bits belong to this descriptor. ``None`` means
    the prefix does not yet contain the fields needed to determine that length.
    """
    b = [int(x) for x in bits]
    if len(b) < 50 or b[:17] != [1] * 17:
        return None
    n = v90._value(b, 18, 25)
    lsp = v90._value(b, 35, 41) + 1
    ltp = v90._value(b, 43, 49) + 1
    terminal, _ = _geometry(n, lsp, ltp)
    end = terminal + 52
    return -(-end // 12) * 12


def descriptor_repeat_lengths_from_prefix(bits) -> tuple[int, ...] | None:
    """Return supported JA repetition lengths for a decoded descriptor prefix.

    The canonical descriptor is padded to a 12-bit boundary.  Some deployed
    peers repeat the descriptor through its mandatory fill bit and one
    additional zero, without repeating the remaining boundary padding.  That
    shorter framing is accepted only by :func:`parse_repeated_ja_descriptor`,
    which still requires a CRC-valid majority across multiple copies.
    """
    canonical = descriptor_bits_from_prefix(bits)
    if canonical is None:
        return None
    b = [int(x) for x in bits]
    n = v90._value(b, 18, 25)
    lsp = v90._value(b, 35, 41) + 1
    ltp = v90._value(b, 43, 49) + 1
    terminal, _ = _geometry(n, lsp, ltp)
    compact = terminal + 53
    return tuple(sorted({canonical, compact} if compact <= canonical else {canonical}))


def parse_repeated_ja_descriptor(bits, *, repeat_bits: int,
                                 repetitions: int) -> dict | None:
    """Decode repeated JA copies by strict bit majority, then apply the normal parser.

    Each copy must begin with the 17-bit descriptor sync (up to six errors in
    that sync are tolerated per copy).  The consensus is zero-padded only for
    the compact repetition form.  The existing parser then checks all start,
    reserved, fill, geometry and CRC conditions without relaxation.
    """
    b = [int(x) for x in bits]
    if any(x not in (0, 1) for x in b) or repetitions < 3 or repetitions % 2 == 0:
        return None
    if repeat_bits <= 0 or len(b) < repeat_bits * repetitions or len(b) < 50:
        return None
    canonical = descriptor_bits_from_prefix(b[:50])
    periods = descriptor_repeat_lengths_from_prefix(b[:50])
    if canonical is None or periods is None or repeat_bits not in periods:
        return None
    if repeat_bits > canonical:
        return None
    frames = [b[i * repeat_bits:(i + 1) * repeat_bits] for i in range(repetitions)]
    if any(sum(bit != 1 for bit in frame[:17]) > 6 for frame in frames):
        return None
    votes = [sum(frame[i] for frame in frames) for i in range(repeat_bits)]
    # An odd copy count has no ties; keep the strict test explicit so this
    # remains safe if the accepted repetition policy changes later.
    if any(vote * 2 == repetitions for vote in votes):
        return None
    consensus = [int(vote * 2 > repetitions) for vote in votes]
    if repeat_bits < canonical:
        # All transmitted bits after the CRC/fill payload in the compact form
        # must be zero.  The omitted boundary padding is synthesized as zero.
        n = v90._value(consensus, 18, 25)
        lsp = v90._value(consensus, 35, 41) + 1
        ltp = v90._value(consensus, 43, 49) + 1
        terminal, _ = _geometry(n, lsp, ltp)
        end = terminal + 52
        if repeat_bits < end or any(consensus[end:]):
            return None
        consensus.extend([0] * (canonical - repeat_bits))
    parsed = parse_ja_descriptor(consensus)
    if parsed is None:
        return None
    parsed = dict(parsed)
    parsed["repetitions"] = repetitions
    parsed["repeat_bits"] = repeat_bits
    return parsed


def _geometry(n: int, lsp: int, ltp: int):
    alpha = ((lsp + 15) // 16) * 17
    beta = alpha + ((ltp + 15) // 16) * 17
    u_start = 187 + beta
    terminal = u_start + ((n + 1) // 2) * 17
    # Bit 50 is reserved information, not a start bit.  It is set to zero,
    # validated as reserved below, and included in the CRC input.
    starts = {17, 34}
    starts.update(51 + g * 17 for g in range((lsp + 15) // 16))
    starts.update(51 + alpha + g * 17 for g in range((ltp + 15) // 16))
    starts.update(51 + beta + g * 17 for g in range(4))
    starts.update(51 + beta + 68 + g * 17 for g in range(4))
    starts.update(u_start + g * 17 for g in range((n + 1) // 2))
    return terminal, starts


def rate_mask(rates) -> int:
    """Capability mask (bit 0 = 24,000 bit/s) for a collection of upstream rates."""
    mask = 0
    for rate in rates:
        k = round((rate - 24000) / RATE_STEP)
        if abs(24000 + k * RATE_STEP - rate) > 1e-6 or not 0 <= k < RATE_BITS:
            raise ValueError(f"{rate} is not a V.92 upstream rate (24,000..48,000 by 8000/6)")
        mask |= 1 << k
    return mask


def rates_from_mask(mask: int) -> list[float]:
    return [24000 + k * RATE_STEP for k in range(RATE_BITS) if mask >> k & 1]


def build_ja_descriptor(*, upstream_rates=(), **fields) -> list[int]:
    """V.92 descriptor. ``fields`` are the V.90 builder's (N, LSP, LTP, SP, TP, H, REF, Ucodes)."""
    n = int(fields.get("N", 0))
    lsp = int(fields.get("LSP", 1))
    ltp = int(fields.get("LTP", 1))
    mask = rate_mask(upstream_rates)
    base = v90.build_ja_descriptor(**fields)
    terminal, starts = _geometry(n, lsp, ltp)
    body = list(base[:terminal])
    information = [body[i] for i in range(18, terminal) if i not in starts]
    word1 = [(mask >> i) & 1 for i in range(MASK_WORD_BITS)]
    word2 = [(mask >> (MASK_WORD_BITS + i)) & 1 for i in range(MASK_WORD_BITS)]
    information += word1 + word2
    out = body + [0] + word1 + [0] + word2 + [0] + vi._field(vi.crc16(information), 16) + [0]
    while len(out) % 12:
        out.append(0)
    return out


def parse_ja_descriptor(bits) -> dict | None:
    """Parse a V.92 descriptor; None for any framing, start-bit, reserved-bit or CRC fault."""
    b = [int(x) for x in bits]
    if len(b) < 60 or b[:17] != [1] * 17 or any(x not in (0, 1) for x in b):
        return None
    n = v90._value(b, 18, 25)
    lsp = v90._value(b, 35, 41) + 1
    ltp = v90._value(b, 43, 49) + 1
    terminal, core_starts = _geometry(n, lsp, ltp)
    end = terminal + 52
    descriptor_bits = -(-end // 12) * 12
    if len(b) != descriptor_bits:
        return None
    starts = core_starts | {terminal, terminal + 17, terminal + 34}
    if any(b[i] for i in starts) or b[terminal + 51]:
        return None
    if any(b[end:]):
        return None
    word1 = b[terminal + 1:terminal + 17]
    word2 = b[terminal + 18:terminal + 34]
    if any(word2[RATE_BITS - MASK_WORD_BITS:]):
        return None                                   # reserved mask bits
    information = [b[i] for i in range(18, terminal) if i not in starts]
    information += word1 + word2
    crc = v90._value(b, terminal + 35, terminal + 50)
    if crc != vi.crc16(information):
        return None
    mask = sum(bit << i for i, bit in enumerate(word1 + word2[:RATE_BITS - MASK_WORD_BITS]))
    # The common V.90 descriptor fields define DIL's SP/TP, H, REF and Ucode
    # segments. Recompute its own CRC for the common prefix so the established
    # V.90 parser can decode those fields independently of V.92's appended mask.
    core = b[:terminal + 1]
    core_info = [core[i] for i in range(18, terminal) if i not in core_starts]
    core += vi._field(vi.crc16(core_info), 16) + [0]
    while len(core) % 12:
        core.append(0)
    dil = v90.parse_ja_descriptor(core)
    if dil is None or not dil["valid"]:
        return None
    return {"N": n, "LSP": lsp, "LTP": ltp, "upstream_rate_mask": mask,
            "upstream_rates": rates_from_mask(mask), "bits": descriptor_bits,
            "SP": dil["SP"], "TP": dil["TP"], "H": dil["H"],
            "REF": dil["REF"], "Ucodes": dil["Ucodes"]}
