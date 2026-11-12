import unittest

from v34_info import crc16
from v90_cp import V90CpScanner, V90CpSymbolDecoder, cp_frame_length, parse_cp


def _set(bits, start, width, value):
    for offset in range(width):
        bits[start + offset] = (value >> offset) & 1


def _frame(*, kind, drn, selectors, tx_masks, codec_masks=None, law=0, ack=1,
           spectral_sign_bits=1):
    maximum = max(selectors)
    gamma = 136 * maximum
    delta = 2 * gamma + 136 if codec_masks is not None else gamma
    crc_start = 272 + delta
    bits = [0] * (crc_start + 20)
    bits[:17] = [1] * 17
    _set(bits, 20, 5, drn)
    bits[19] = int(kind == "cp")
    bits[35] = law
    _set(bits, 36, 13, (1 << 12) | (1 << 7) | 1)
    _set(bits, 31, 2, spectral_sign_bits)
    bits[33] = ack
    _set(bits, 49, 2, 1)
    _set(bits, 52, 16, 0x4000)
    for offset, value in zip((69, 77, 86, 94), (0x10, 0xF0, 0x08, 0xE8)):
        _set(bits, offset, 8, value)
    for offset, value in zip((103, 107, 111, 115, 120, 124), selectors):
        _set(bits, offset, 4, value)
    bits[128] = int(codec_masks is not None)

    def put_mask(block_start, codes):
        for code in codes:
            group, bit = divmod(code, 16)
            bits[block_start + 1 + 17 * group + bit] = 1

    for index, codes in enumerate(tx_masks):
        put_mask(136 + index * 136, codes)
    if codec_masks is not None:
        for index, codes in enumerate(codec_masks):
            put_mask(272 + gamma + index * 136, codes)

    starts = {34, 51, 68, 85, 102, 119, 136, crc_start}
    starts.update(range(272, 272 + gamma, 136))
    starts.update(range(272 + gamma, 272 + 2 * gamma + 136, 136))
    for block_start in range(136, 272 + gamma, 136):
        starts.update(block_start + 17 * group for group in range(1, 8))
    if codec_masks is not None:
        for block_start in range(272 + gamma, 272 + 2 * gamma + 136, 136):
            starts.update(block_start + 17 * group for group in range(1, 8))
    info = [bit for index, bit in enumerate(bits[18:crc_start], 18)
            if index not in starts]
    _set(bits, crc_start + 1, 16, crc16(info))
    return bits


def _encode_symbols(bits, *, constellation):
    """Independent test encoder for the V.34 GPA/differential decision boundary."""
    reg = [0] * 23
    rotation = 0
    width = 4 if constellation == 16 else 2
    padded = list(bits)
    padded += [0] * ((-len(padded)) % width)
    symbols = []
    for start in range(0, len(padded), width):
        scrambled = []
        for bit in padded[start:start + width]:
            value = bit ^ reg[4] ^ reg[22]
            reg = [value] + reg[:-1]
            scrambled.append(value)
        increment = 2 * scrambled[1] + scrambled[0]
        rotation = (rotation + increment) % 4
        quarter = (2 * scrambled[3] + scrambled[2]) if width == 4 else 0
        symbols.append((rotation, quarter))
    return symbols


class V90ControlParameterTests(unittest.TestCase):
    def test_cpt_variable_length_fields_and_ucode_masks(self):
        bits = _frame(kind="cpt", drn=7, selectors=(0, 1, 2, 0, 1, 2),
                      tx_masks=((0, 15), (16, 47), (80, 127)))
        parsed = parse_cp(bits)
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["kind"], "cpt")
        self.assertEqual(parsed["rate"], 20000)
        self.assertEqual(parsed["codec"], "pcmu")
        self.assertEqual(parsed["interval_constellations"], (0, 1, 2, 0, 1, 2))
        self.assertEqual(parsed["constellations"], ((0, 15), (16, 47), (80, 127)))
        self.assertIsNone(parsed["codec_constellations"])

    def test_cp_codec_maps_and_pcma_profile(self):
        bits = _frame(kind="cp", drn=4, selectors=(0, 1, 2, 0, 1, 2),
                      tx_masks=((0, 1), (20, 21), (100, 101)),
                      codec_masks=((2, 3), (10, 11), (110, 111)), law=1)
        parsed = parse_cp(bits)
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["kind"], "cp")
        self.assertEqual(parsed["rate"], 32000)
        self.assertEqual(parsed["codec"], "pcma")
        self.assertEqual(parsed["codec_constellations"],
                         ((2, 3), (10, 11), (110, 111)))
        self.assertEqual(parsed["shape"], (0.25, -0.25, 0.125, -0.375))

    def test_codec_map_frame_length_includes_base_codec_map(self):
        bits = _frame(kind="cp", drn=0, selectors=(0, 0, 0, 0, 0, 0),
                      tx_masks=((7,),), codec_masks=((9,),))
        parsed = parse_cp(bits)
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["frame_bits"], 428)
        self.assertIsNone(parsed["rate"])
        self.assertTrue(parsed["cleardown"])
        self.assertEqual(parsed["codec_constellations"], ((9,),))

    def test_scanner_handles_split_frames_and_bounded_retention(self):
        first = _frame(kind="cpt", drn=7, selectors=(0, 1, 2, 0, 1, 2),
                       tx_masks=((0,), (16,), (80,)))
        second = _frame(kind="cp", drn=4, selectors=(0, 1, 2, 0, 1, 2),
                        tx_masks=((1,), (21,), (101,)),
                        codec_masks=((2,), (10,), (110,)))
        scanner = V90CpScanner()
        prefix = [0, 1, 0, 1, 1]
        self.assertIsNone(cp_frame_length(first[:128]))
        self.assertEqual(cp_frame_length(first[:129]), len(first))
        self.assertEqual(scanner.feed(prefix + first[:400]), [])
        self.assertEqual(scanner.feed(first[400:] + second[:500]), [
            {**parse_cp(first), "start_bit": len(prefix)}])
        parsed = scanner.feed(second[500:])
        self.assertEqual(parsed, [{**parse_cp(second),
                                   "start_bit": len(prefix) + len(first)}])
        self.assertEqual(scanner.bits_seen, len(prefix) + len(first) + len(second))
        self.assertLessEqual(len(scanner._bits), scanner.max_buffer_bits)

    def test_symbol_decoder_recovers_cp_and_cpt_across_chunks(self):
        cpt = _frame(kind="cpt", drn=7, selectors=(0, 1, 2, 0, 1, 2),
                     tx_masks=((0, 15), (16, 47), (80, 127)))
        cp = _frame(kind="cp", drn=4, selectors=(0, 1, 2, 0, 1, 2),
                    tx_masks=((0, 1), (20, 21), (100, 101)),
                    codec_masks=((2, 3), (10, 11), (110, 111)), law=1)
        wire = _encode_symbols(cpt + cp, constellation=16)
        decoder = V90CpSymbolDecoder(16)
        self.assertEqual(decoder.feed(wire[:97]), [])
        frames = decoder.feed(wire[97:])
        self.assertEqual([frame["kind"] for frame in frames], ["cpt", "cp"])
        self.assertEqual(frames[0]["start_bit"], 0)
        self.assertEqual(frames[1]["start_bit"], len(cpt))
        self.assertEqual(frames[1]["codec"], "pcma")

    def test_single_symbol_fast_path_matches_chunk_decoder(self):
        cpt = _frame(kind="cpt", drn=7, selectors=(0, 1, 2, 0, 1, 2),
                     tx_masks=((0,), (16,), (80,)))
        wire = _encode_symbols(cpt, constellation=16)
        chunked = V90CpSymbolDecoder(16)
        incremental = V90CpSymbolDecoder(16)
        internal = V90CpSymbolDecoder(16)
        chunk_frames = chunked.feed(wire)
        single_frames = []
        internal_frames = []
        for rotation, quarter in wire:
            single_frames.extend(incremental.feed_symbol(rotation, quarter))
            internal_frames.extend(internal._feed_symbol_unchecked(
                rotation, quarter))
        self.assertEqual(single_frames, chunk_frames)
        self.assertEqual(internal_frames, chunk_frames)
        batched = V90CpSymbolDecoder(16)
        batch_frames = []
        for offset in range(0, len(wire), 4):
            batch_frames.extend(batched._feed_symbols_unchecked(
                wire[offset:offset + 4], empty_none=True) or ())
        self.assertEqual(batch_frames, chunk_frames)
        self.assertEqual(batched._previous_rotation,
                         chunked._previous_rotation)
        self.assertEqual(batched._scrambled_history,
                         chunked._scrambled_history)
        self.assertEqual(batched.bits_seen, chunked.bits_seen)
        self.assertEqual(batched.scanner._bits, chunked.scanner._bits)
        self.assertEqual(batched.scanner._cursor, chunked.scanner._cursor)
        self.assertEqual(incremental.last_descrambled_bits,
                         chunked.last_descrambled_bits[-4:])
        self.assertEqual(incremental.bits_seen, chunked.bits_seen)
        self.assertEqual(internal.bits_seen, chunked.bits_seen)
        with self.assertRaises(ValueError):
            incremental.feed_symbol(0, True)
        speculative = V90CpSymbolDecoder(16)
        self.assertIsNone(speculative._feed_symbol_unchecked(
            0, 0, capture_last_bits=False, empty_none=True))
        self.assertEqual(speculative.last_descrambled_bits, ())

    def test_symbol_decoder_handles_four_point_and_rejects_bad_decisions(self):
        frame = _frame(kind="cpt", drn=7, selectors=(0, 1, 2, 0, 1, 2),
                       tx_masks=((0,), (16,), (80,)))
        decoder = V90CpSymbolDecoder(4)
        self.assertEqual([item["kind"] for item in decoder.feed(
            _encode_symbols(frame, constellation=4))], ["cpt"])
        with self.assertRaises(ValueError):
            decoder.feed([(0, 1)])
        with self.assertRaises(ValueError):
            V90CpSymbolDecoder(16).feed([(4, 0)])

    def test_rejects_bad_crc_reserved_fields_and_incomplete_variable_frame(self):
        args = dict(kind="cpt", drn=3, selectors=(0, 1, 0, 1, 0, 1),
                    tx_masks=((0,), (1,)))
        good = _frame(**args)
        damaged = list(good)
        damaged[34] = 1
        self.assertIsNone(parse_cp(damaged))
        damaged = list(good)
        damaged[18] = 1
        self.assertIsNone(parse_cp(damaged))
        damaged = list(good)
        damaged[-19] ^= 1
        self.assertIsNone(parse_cp(damaged))
        damaged = list(good)
        damaged[-1] = 1
        self.assertIsNone(parse_cp(damaged))
        self.assertIsNone(parse_cp(good[:-1]))


if __name__ == "__main__":
    unittest.main()
