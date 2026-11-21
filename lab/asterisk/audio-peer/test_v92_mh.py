"""V.92 modem-on-hold MH sequences (Tables 32 and 33)."""

import unittest

try:
    import numpy as np  # noqa: F401
except ImportError:
    np = None

if np is not None:
    import v92_mh as mh


@unittest.skipIf(np is None, "numpy not installed")
class MhTests(unittest.TestCase):
    def test_frame_layout_matches_table_32(self):
        bits = mh.build_mh("MHreq")
        self.assertEqual(len(bits), 40)
        self.assertEqual(bits[:4], [1, 1, 1, 1])
        self.assertEqual(bits[4:12], [0, 1, 1, 1, 0, 0, 1, 0])      # sync 01110010
        self.assertEqual(bits[36:], [1, 1, 1, 1])
        # MHreq = 0011 (LSB first on the wire, so 1,1,0,0), repeated in the info bits.
        self.assertEqual(bits[12:20], [1, 1, 0, 0, 1, 1, 0, 0])

    def test_every_signal_round_trips(self):
        for signal in ("MHreq", "MHnack", "MHcda", "MHfrr"):
            self.assertEqual(mh.parse_mh(mh.build_mh(signal)), {"signal": signal})
        ack = mh.parse_mh(mh.build_mh("MHack", timeout_code=5))
        self.assertEqual((ack["signal"], ack["timeout_seconds"]), ("MHack", 60))
        self.assertIsNone(mh.parse_mh(mh.build_mh("MHack", timeout_code=13))["timeout_seconds"])
        clrd = mh.parse_mh(mh.build_mh("MHclrd", cleardown_reason="outgoing call"))
        self.assertEqual(clrd["cleardown_reason"], "outgoing call")

    def test_timeout_table_matches_table_33(self):
        expected = {1: 10, 2: 20, 3: 30, 4: 40, 5: 60, 6: 120, 7: 180, 8: 240, 9: 360,
                    10: 480, 11: 720, 12: 960}
        for code, seconds in expected.items():
            self.assertEqual(mh.parse_mh(mh.build_mh("MHack", timeout_code=code))
                             ["timeout_seconds"], seconds)
        for code in (0, 14, 15):
            with self.assertRaises(ValueError):
                mh.build_mh("MHack", timeout_code=code)

    def test_faults_and_undefined_codes_are_ignored(self):
        good = mh.build_mh("MHreq")
        for index in (0, 5, 13, 17, 25, 38):
            bad = list(good)
            bad[index] ^= 1
            self.assertIsNone(mh.parse_mh(bad), index)
        # An undefined indication code with a valid CRC must be ignored (Note 1).
        import v34_info as vi
        undefined = vi._frame([0, 0, 0, 0, 0, 0, 0, 0])
        self.assertIsNone(mh.parse_mh(undefined))
        with self.assertRaises(ValueError):
            mh.build_mh("MHreq", timeout_code=3)
        with self.assertRaises(ValueError):
            mh.build_mh("nonsense")


if __name__ == "__main__":
    unittest.main()
