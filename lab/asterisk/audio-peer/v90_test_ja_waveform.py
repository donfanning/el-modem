#!/usr/bin/env python3
"""Exercise the phase-3 Ja receive path with an independently built waveform.

This is a diagnostic control, not a V.90 conformance test or a production
transmitter. The transmit scrambler, differential mapper, and descriptor CRC
below are separately implemented from the V.34/V.90 equations. A simple RRC
pulse/carrier model creates an idealized test signal; it is not asserted to be
the exact normative transmit filter. The receiver deliberately reuses
El-Modem's phase-3 equalizer, then the standalone bit audit. It reports both
the receiver's TRN-end estimate and the known synthetic boundary so boundary
error cannot be mistaken for a bit/CRC failure.

V.90 §9.3.2.4 puts analogue Ja immediately after TRN.  No S/S-bar/SCR block is
inserted between them.  This tool does not model the peer digital modem, DIL,
Phase 4, or a real channel.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "audio-peer"))
import v34_rx
import v34_tx
from v90_test_bits import symbols_to_bits
from v90_info_reference import find_ja_descriptors


FS = 8000
BAUD = 2400.0 * 10 / 7
# V.34 Table 2 defines carrier as d/e times the selected symbol rate.  Using
# 2400 here (the rate-grid base) produced 1371 Hz instead of the 1959 Hz
# carrier selected for 3429 baud, invalidating the earlier synthetic B1 probe.
CARRIER = v34_tx.CARRIERS[round(BAUD)][0]
ROLE = "answer"
N_TRN = 1400


def _field(value: int, width: int) -> list[int]:
    return [(value >> bit) & 1 for bit in range(width)]


def _scramble(bits: list[int], state: list[int] | None = None) -> tuple[list[int], list[int]]:
    """GPA self-synchronizing recurrence, independently implemented."""
    history = [0] * 23 if state is None else list(state)
    output = []
    for bit in bits:
        coded = int(bit) ^ history[4] ^ history[22]
        output.append(coded)
        history = [coded, *history[:22]]
    return output, history


def _descriptor_crc_bits(information: list[int]) -> list[int]:
    """V.34 x^16+x^12+x^5+1 CRC as output bits in descriptor order."""
    register = 0xFFFF
    for bit in information:
        feedback = ((register >> 15) & 1) ^ int(bit)
        register = (register << 1) & 0xFFFF
        if feedback:
            register ^= 0x1021
    # The CRC register output is presented starting at its least-significant
    # position in the polynomial definition, then placed in the Table 12
    # low-index-first field.  This is the resulting wire-time bit order.
    return [(register >> bit) & 1 for bit in range(15, -1, -1)]


def _ja_descriptor(n: int, lsp: int, ltp: int) -> list[int]:
    """Build a legal-format Ja descriptor with deterministic field contents."""
    if not 0 <= n <= 255 or not 1 <= lsp <= 128 or not 1 <= ltp <= 128:
        raise ValueError("Ja N/LSP/LTP outside V.90 Table 12 field limits")
    # Preserve the original vector's field contents and extend only the field
    # whose length exceeds that vector. This makes one-field-at-a-time runs
    # comparable while keeping every generated entry deterministic.
    base_sp = [1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1]
    base_tp = [0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0,
               1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1]
    sp = base_sp[:lsp] + [((index * 3 + index // 4) & 1)
                          for index in range(len(base_sp), lsp)]
    tp = base_tp[:ltp] + [((index * 5 + index // 3 + 1) & 1)
                          for index in range(len(base_tp), ltp)]
    h = [1, 2, 0, 3, 1, 2, 0, 1]
    ref = [0, 10, 20, 30, 40, 50, 60, 70]
    base_ucodes = [1, 17, 33, 49, 65, 81, 97]
    ucodes = base_ucodes[:n] + [((index * 19 + 7) % 128)
                                for index in range(len(base_ucodes), n)]
    alpha = ((lsp + 15) // 16) * 17
    beta = alpha + ((ltp + 15) // 16) * 17
    u_start = 187 + beta
    terminal = u_start + ((n + 1) // 2) * 17
    crc_start = terminal + 1
    frame = [0] * (crc_start + 18)
    frame[:17] = [1] * 17
    frame[18:26] = _field(n, 8)
    frame[35:42] = _field(lsp - 1, 7)
    frame[43:50] = _field(ltp - 1, 7)
    # Table 12 bit 50 is an ITU-reserved information bit, not a start bit;
    # V.34's CRC excludes framing markers but includes this zero-valued bit.
    starts = {17, 34, terminal}

    def put_pattern(start: int, values: list[int]) -> None:
        for group in range((len(values) + 15) // 16):
            position = start + 17 * group
            starts.add(position)
            chunk = values[group * 16:(group + 1) * 16]
            frame[position + 1:position + 1 + len(chunk)] = chunk

    put_pattern(51, sp)
    put_pattern(51 + alpha, tp)
    h_start = 51 + beta
    for group in range(4):
        position = h_start + 17 * group
        starts.add(position)
        frame[position + 1:position + 8] = _field(h[2 * group], 7)
        frame[position + 9:position + 16] = _field(h[2 * group + 1], 7)
    ref_start = h_start + 68
    for group in range(4):
        position = ref_start + 17 * group
        starts.add(position)
        frame[position + 1:position + 8] = _field(ref[2 * group], 7)
        frame[position + 9:position + 16] = _field(ref[2 * group + 1], 7)
    for group in range((n + 1) // 2):
        position = u_start + 17 * group
        starts.add(position)
        frame[position + 1:position + 8] = _field(ucodes[2 * group], 7)
        if 2 * group + 1 < n:
            frame[position + 9:position + 16] = _field(ucodes[2 * group + 1], 7)

    information = [frame[position] for position in range(18, terminal)
                   if position not in starts]
    frame[crc_start:crc_start + 16] = _descriptor_crc_bits(information)
    if len(frame) & 1:
        frame.append(0)
    return frame


def _differential(bits: list[int], initial_rotation: int,
                  state: list[int]) -> tuple[np.ndarray, list[int]]:
    coded, state = _scramble(bits, state)
    points = (1 + 1j) * np.asarray([1, -1j, -1, 1j])
    symbols = []
    rotation = initial_rotation
    for offset in range(0, len(coded), 2):
        i1, i2 = coded[offset:offset + 2]
        rotation = (rotation + 2 * i2 + i1) % 4
        symbols.append(points[rotation])
    return np.asarray(symbols), state


def _rrc(time_s: np.ndarray, symbol_s: float, rolloff: float = 0.15) -> np.ndarray:
    x = time_s / symbol_s
    result = np.empty_like(x)
    near_zero = np.abs(x) < 1e-10
    result[near_zero] = 1 - rolloff + 4 * rolloff / math.pi
    singular = np.abs(np.abs(x) - 1 / (4 * rolloff)) < 1e-8
    result[singular] = rolloff / math.sqrt(2) * (
        (1 + 2 / math.pi) * math.sin(math.pi / (4 * rolloff))
        + (1 - 2 / math.pi) * math.cos(math.pi / (4 * rolloff)))
    regular = ~(near_zero | singular)
    xr = x[regular]
    result[regular] = (np.sin(math.pi * xr * (1 - rolloff))
                       + 4 * rolloff * xr * np.cos(math.pi * xr * (1 + rolloff))) / (
                           math.pi * xr * (1 - (4 * rolloff * xr) ** 2))
    return result


def _modulate(symbols: np.ndarray) -> np.ndarray:
    symbol_s = 1 / BAUD
    span = 8
    count = math.ceil((len(symbols) + 2 * span) * FS / BAUD)
    time_s = np.arange(count) / FS - span * symbol_s
    baseband = np.zeros(count, dtype=np.complex128)
    for index, symbol in enumerate(symbols):
        relative = time_s - index * symbol_s
        active = np.abs(relative) <= (span + 1) * symbol_s
        baseband[active] += symbol * _rrc(relative[active], symbol_s)
    return (baseband * np.exp(2j * math.pi * CARRIER * np.arange(count) / FS)).real * 0.1


def _preamble_and_trn() -> tuple[np.ndarray, list[int], list[int]]:
    p0 = 1 + 1j
    s = np.asarray([p0, -1 + 1j] * 64)
    s_bar = np.asarray([-p0, 1 - 1j] * 8)
    pp = []
    for k in range(72):
        for i in range(4):
            angle = (math.pi * (k * i + 4) / 6 if k % 3 == 1
                     else math.pi * k * i / 6)
            pp.append(complex(math.cos(angle), math.sin(angle)))
    scrambled, state = _scramble([1] * (2 * N_TRN))
    trn = []
    for offset in range(0, len(scrambled), 2):
        i1, i2 = scrambled[offset:offset + 2]
        rotation = (2 * i2 + i1) % 4
        trn.append(p0 * [1, -1j, -1, 1j][rotation])
    sequence = np.concatenate([s, s_bar, np.asarray(pp), np.asarray(trn)])
    final_rotation = (2 * scrambled[-1] + scrambled[-2]) % 4
    return sequence, state, [final_rotation]


def analyze(n: int, lsp: int, ltp: int) -> dict:
    prefix, scrambler_state, rotation_state = _preamble_and_trn()
    descriptor = _ja_descriptor(n, lsp, ltp)
    source_parse = find_ja_descriptors(descriptor, include_bad_crc=True)
    ja = np.tile(np.asarray(descriptor, dtype=np.int8), 3).tolist()
    ja_symbols, _ = _differential(ja, rotation_state[0], scrambler_state)
    source_symbol_ids = np.mod(
        np.rint(-np.angle(ja_symbols / (1 + 1j)) / (np.pi / 2)).astype(np.int64), 4
    )
    samples = _modulate(np.concatenate([prefix, ja_symbols]))

    frontend = v34_rx.Phase3Rx(BAUD, CARRIER, "answer")
    acquired = frontend.acquire(samples)
    if not acquired:
        return {"acquired": False}
    train_mse, role = frontend.train_trn_auto()
    follower = v34_rx.Phase3Receiver(frontend, role=role)
    follower.feed()
    symbols = np.asarray(follower.idx)
    estimated_end = follower._end_est
    expected_end = N_TRN

    def decode(boundary: int) -> tuple[list[int], list[dict]]:
        decoded_bits = symbols_to_bits(symbols, boundary, "answer")
        frames = find_ja_descriptors(decoded_bits, include_bad_crc=True)
        rows = []
        for item in frames:
            start = item["start_bit"]
            received = decoded_bits[start:start + len(descriptor)]
            error_offsets = [offset for offset, (actual, expected) in
                             enumerate(zip(received, descriptor)) if actual != expected]
            symbol_offset = start // 2 - expected_end
            symbol_stop = symbol_offset + (len(descriptor) + 1) // 2
            actual_symbol_ids = symbols[expected_end + symbol_offset:
                                        expected_end + symbol_stop]
            expected_symbol_ids = source_symbol_ids[symbol_offset:symbol_stop]
            symbol_error_offsets = [offset for offset, (actual, expected) in
                                    enumerate(zip(actual_symbol_ids, expected_symbol_ids))
                                    if int(actual) != int(expected)]
            rows.append({"start_bit": start, "crc_valid": item["crc_valid"],
                         "bit_errors_vs_source": len(error_offsets),
                         "bit_error_offsets": error_offsets,
                         "equalizer_symbol_errors_vs_source": len(symbol_error_offsets),
                         "equalizer_symbol_error_offsets": symbol_error_offsets})
        return decoded_bits, rows

    estimate_bits, estimate_frames = decode(estimated_end)
    oracle_bits, oracle_frames = decode(expected_end)
    trn_bits, _ = decode(expected_end)
    trn_known = trn_bits[23:2 * N_TRN]
    return {
        "acquired": True,
        "acquisition_metric": acquired["metric"],
        "train_mse": train_mse,
        "scrambler_role": role,
        "trn_symbols_expected": expected_end,
        "trn_end_estimated": estimated_end,
        "descrambled_bits_changed_by_known_boundary": sum(
            left != right for left, right in zip(estimate_bits, oracle_bits)),
        "trn_known_one_errors_after_warmup": sum(bit != 1 for bit in trn_known),
        "descriptor_bits": len(descriptor),
        "descriptor_shape": {"N": n, "LSP": lsp, "LTP": ltp},
        "source_descriptor_crc_valid": bool(source_parse and source_parse[0]["crc_valid"]),
        "descriptor_frames_using_estimated_boundary": estimate_frames,
        "descriptor_frames_using_known_synthetic_boundary": oracle_frames,
        "interpretation": "synthetic known-channel control only; not interoperability or conformance",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="optional JSON report path")
    parser.add_argument("--n", type=int, default=7, help="Ja descriptor N field (0..255)")
    parser.add_argument("--lsp", type=int, default=19, help="Ja descriptor LSP field (1..128)")
    parser.add_argument("--ltp", type=int, default=31, help="Ja descriptor LTP field (1..128)")
    args = parser.parse_args()
    report = analyze(args.n, args.lsp, args.ltp)
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    if not report.get("acquired"):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
