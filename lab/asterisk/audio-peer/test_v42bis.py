#!/usr/bin/env python3
"""V.42bis: our encoder against our decoder, and against SpanDSP's implementation as an independent reference."""
import ctypes
import os
import random
import unittest

import v42bis as bis

TEXT = (b"The quick brown fox jumps over the lazy dog. " * 30 + b"pack my box with five dozen liquor jugs. " * 20)


def roundtrip(data, n2=1024, n7=8, chunk=None, flush_every=None, seed=1, auto=True):
    enc, dec = bis.Encoder(n2, n7, auto), bis.Decoder(n2, n7)
    rnd = random.Random(seed)
    wire = bytearray()
    out = bytearray()
    i = 0
    while i < len(data):
        n = rnd.randrange(1, 200) if chunk is None else chunk
        piece = data[i:i + n]
        i += n
        w = enc.compress(piece)
        if rnd.random() < (flush_every or 0):
            w += enc.flush()
        wire += w
        out += dec.decompress(w)
    w = enc.flush()
    wire += w
    out += dec.decompress(w)
    return bytes(out), bytes(wire)


class SelfTests(unittest.TestCase):
    def test_text_roundtrips_and_shrinks(self):
        out, wire = roundtrip(TEXT, flush_every=0.2)
        self.assertEqual(out, TEXT)
        self.assertLess(len(wire), len(TEXT) // 3)

    def test_random_binary_survives_mode_switching(self):
        rnd = random.Random(5)
        data = b"".join(rnd.choice([TEXT[:500], bytes(rnd.randrange(256) for _ in range(700)), bytes(300), b"\x00\x33\x66\x99" * 80])
                        for _ in range(40))
        out, wire = roundtrip(data, flush_every=0.3)
        self.assertEqual(out, data)

    def test_escape_character_cycling_in_both_modes(self):
        data = bytes([0, 51, 102, 153, 204, 255, 50, 0, 0, 0]) * 60 + TEXT + bytes(range(256)) * 3
        for auto in (True, False):
            out, _ = roundtrip(data, flush_every=0.1, auto=auto)
            self.assertEqual(out, data)

    def test_dictionary_wraps_and_codeword_size_grows(self):
        rnd = random.Random(8)
        words = [bytes(rnd.randrange(32, 127) for _ in range(rnd.randrange(3, 9))) for _ in range(3000)]
        data = b" ".join(rnd.choice(words) for _ in range(6000))
        for n2, n7 in ((512, 6), (1024, 8), (2048, 32), (4096, 250)):
            out, wire = roundtrip(data, n2=n2, n7=n7, flush_every=0.05)
            self.assertEqual(out, data, (n2, n7))

    def test_decoder_rejects_a_reserved_command_code(self):
        dec = bis.Decoder()
        with self.assertRaises(bis.V42bisError):
            dec.decompress(bytes([0, 9]))                   # escape character 0 followed by reserved code 9


# ---- independent reference: SpanDSP's v42bis (test only) -------------------------------------------------------
def _spandsp():
    try:
        lib = ctypes.CDLL(os.environ.get("SPANDSP_LIB", "libspandsp.so.4"))
        return lib if hasattr(lib, "v42bis_init") else None
    except OSError:
        return None


PUT = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int)


class Ref:
    """SpanDSP v42bis context: .compress(data)->bytes, .decompress(bytes)->bytes (both directions on one object)."""

    def __init__(self, lib, n2=1024, n7=8):
        self.lib, self.tx, self.rx = lib, bytearray(), bytearray()
        self._cbs = (PUT(lambda u, b, n: self.tx.extend(ctypes.string_at(b, n)) if n > 0 else None),
                     PUT(lambda u, b, n: self.rx.extend(ctypes.string_at(b, n)) if n > 0 else None))
        lib.v42bis_init.restype = ctypes.c_void_p
        lib.v42bis_init.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, PUT, ctypes.c_void_p,
                                    ctypes.c_int, PUT, ctypes.c_void_p, ctypes.c_int]
        self.s = lib.v42bis_init(None, 3, n2, n7, self._cbs[0], None, 256, self._cbs[1], None, 1024)
        for f in ("v42bis_compress", "v42bis_decompress"):
            getattr(lib, f).argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]
        lib.v42bis_compress_flush.argtypes = [ctypes.c_void_p]
        lib.v42bis_decompress_flush.argtypes = [ctypes.c_void_p]

    def compress(self, data):
        self.tx.clear()
        self.lib.v42bis_compress(self.s, data, len(data))
        self.lib.v42bis_compress_flush(self.s)
        return bytes(self.tx)

    def decompress(self, data):
        self.rx.clear()
        self.lib.v42bis_decompress(self.s, data, len(data))
        self.lib.v42bis_decompress_flush(self.s)            # the reference buffers output until asked
        return bytes(self.rx)


@unittest.skipUnless(_spandsp(), "libspandsp (reference implementation) not available")
class InteropTests(unittest.TestCase):
    def data(self):
        rnd = random.Random(11)
        return [TEXT, bytes(rnd.randrange(256) for _ in range(2000)), TEXT[::-1] + bytes(range(256)) * 4,
                b"".join(rnd.choice([TEXT[:300], bytes(rnd.randrange(256) for _ in range(400)), bytes(200)])
                         for _ in range(30))]

    def test_our_encoder_into_the_reference_decoder(self):
        for n2, n7 in ((1024, 8), (512, 6), (2048, 32)):
            for data in self.data():
                ref, enc = Ref(_spandsp(), n2, n7), bis.Encoder(n2, n7)
                got = bytearray()
                for i in range(0, len(data), 97):
                    got += ref.decompress(enc.compress(data[i:i + 97]) + enc.flush())
                self.assertEqual(bytes(got), data, (n2, n7))

    def test_the_reference_encoder_into_our_decoder(self):
        for n2, n7 in ((1024, 8), (512, 6), (2048, 32)):
            for data in self.data():
                ref, dec = Ref(_spandsp(), n2, n7), bis.Decoder(n2, n7)
                got = bytearray()
                for i in range(0, len(data), 97):
                    got += dec.decompress(ref.compress(data[i:i + 97]))
                self.assertEqual(bytes(got), data, (n2, n7))


if __name__ == "__main__":
    unittest.main()
