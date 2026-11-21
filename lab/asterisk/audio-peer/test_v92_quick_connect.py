import unittest

import numpy as np

import v8
import v92_quick_connect as qc


def fsk_bits(bits, tones, lead=24, trail=24):
    """Modulate with the repo's V.21 source and read the bit decisions back at the best sampling phase."""
    src = v8._Fsk(tones, 2000.0)
    src.bits = [1] * lead + list(bits) + [1] * trail
    x = src.read(int(len(src.bits) * v8.FS / v8.BAUD) + 40)
    rx = v8._FskRx(tones)
    rx.feed(x)
    want = list(bits)
    for _, decided in rx.bitstreams():
        for i in range(len(decided) - len(want) + 1):
            if decided[i:i + len(want)] == want:
                return decided[i:i + len(want)]
    return None


class QuickConnectTests(unittest.TestCase):
    def test_lengths_and_layout_match_the_tables(self):
        qc1a = qc.build("QC1a", lapm=True, code=0b0101)
        self.assertEqual(len(qc1a), 60)
        self.assertEqual(qc1a[20:30], [0, 0, 0, 1, 0, 0, 1, 0, 1, 1])      # 000PW0XYZ1 with P=1, WXYZ=0101
        self.assertEqual(qc1a[50:60], qc1a[20:30])
        qca1a = qc.build("QCA1a", lapm=False, code=0b1111)
        self.assertEqual(len(qca1a), 70)
        self.assertEqual(qca1a[20:30], [0, 0, 1, 0, 1, 0, 1, 1, 1, 1])     # 001PW0XYZ1, W=X=Y=Z=1
        self.assertEqual(qca1a[60:], [1] * 10)
        qc1d = qc.build("QC1d", lapm=True, code=0b10)
        self.assertEqual(qc1d[20:30], [0, 1, 0, 1, 0, 0, 0, 1, 0, 1])      # 010P000LM1 with L=1 M=0
        qca1d = qc.build("QCA1d", code=0b11)
        self.assertEqual(qca1d[20:30], [0, 1, 1, 0, 0, 0, 0, 1, 1, 1])
        self.assertEqual(len(qca1d), 70)

    def test_every_code_round_trips(self):
        for code in range(16):
            for kind in ("QC1a", "QCA1a"):
                for lapm in (False, True):
                    got = qc.parse(qc.build(kind, lapm=lapm, code=code))
                    self.assertEqual((got["kind"], got["uqts_code"], got["lapm"]), (kind, code, lapm))
                    self.assertEqual(got["ucode"], qc.UQTS_UCODES.get(code))
                    self.assertEqual(got["cleardown"], code == 0b1111)
        for code in range(4):
            for kind in ("QC1d", "QCA1d"):
                got = qc.parse(qc.build(kind, lapm=True, code=code))
                self.assertEqual((got["kind"], got["level_code"], got["lapm"]), (kind, code, True))
                self.assertEqual(got["level_dbm0"], qc.LEVELS_DBM0[code])

    def test_the_uqts_table(self):
        self.assertEqual([qc.UQTS_UCODES[c] for c in range(15)],
                         [61, 62, 63, 66, 67, 70, 71, 74, 75, 78, 79, 82, 83, 86, 87])

    def test_damaged_or_inconsistent_sequences_are_refused(self):
        good = qc.build("QC1a", code=3)
        for index in (0, 12, 25, 35, 45, 55, 58):
            bad = list(good)
            bad[index] ^= 1
            self.assertIsNone(qc.parse(bad), index)
        bad = list(good)
        bad[24] ^= 1                                      # the first copy of W changes, the repeat does not
        self.assertIsNone(qc.parse(bad))
        self.assertIsNone(qc.parse(good[:59]))
        self.assertIsNone(qc.parse(qc.build("QCA1d", code=1)[:60]))   # a QCA without its closing ones

    def test_the_signals_survive_v21_fsk_in_the_right_channel(self):
        cases = (("QC1a", v8.V21_L, 9), ("QCA1a", v8.V21_H, 4), ("QC1d", v8.V21_L, 2), ("QCA1d", v8.V21_H, 1))
        for kind, tones, code in cases:
            with self.subTest(kind=kind):
                bits = qc.build(kind, lapm=True, code=code)
                recovered = fsk_bits(bits, tones)
                self.assertIsNotNone(recovered)
                parsed = qc.parse(recovered)
                self.assertEqual((parsed["kind"], parsed["lapm"]), (kind, True))

    def test_the_v8bis_identification_fields(self):
        f = qc.identification_field("QC2a", lapm=True, code=0b1010)
        self.assertEqual(f, [1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0])
        self.assertEqual(qc.identification_field("QCA2a", code=0)[14:], [1, 0])        # QCA identifier, analogue
        d = qc.identification_field("QC2d", lapm=False, code=0b01)
        self.assertEqual(d[8:10], [0, 1])
        self.assertEqual(d[14:], [0, 1])                                              # QC identifier, digital
        self.assertEqual(qc.identification_field("QCA2d", code=3)[14:], [1, 1])
        with self.assertRaises(ValueError):
            qc.identification_field("QC1a")
        with self.assertRaises(ValueError):
            qc.build("QC1d", code=4)


if __name__ == "__main__":
    unittest.main()
