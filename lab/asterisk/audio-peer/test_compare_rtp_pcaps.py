import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).parent/'tools'))
from compare_rtp_pcaps import compare, deduplicate


def packet(ssrc, sequence, timestamp, arrival, payload=b'pcmu'):
    return dict(ssrc=ssrc, sequence=sequence, timestamp=timestamp,
                arrival_wall_time=arrival, pcmu_payload_hex=payload.hex())


class RtpCaptureComparisonTests(unittest.TestCase):
    def test_constant_clock_offset_cancels_and_added_burstiness_is_measured(self):
        a = deduplicate([packet(7, 65534, 1000, 10.0),
                         packet(7, 65535, 1160, 10.020),
                         packet(7, 0, 1320, 10.040),
                         packet(7, 1, 1480, 10.060)])
        b = deduplicate([packet(7, 65534, 1000, 80.0),
                         packet(7, 65535, 1160, 80.025),
                         packet(7, 0, 1320, 80.070),
                         packet(7, 1, 1480, 80.090)])
        result = compare(a, b)
        self.assertEqual(result['matched_packets'], 4)
        self.assertEqual(result['only_a'], 0)
        self.assertEqual(result['only_b'], 0)
        self.assertEqual(result['payload_mismatches'], 0)
        self.assertAlmostEqual(result['interval_extra_at_b_minus_a_ms']['mean_ms'], 10.0, places=6)

    def test_missing_and_changed_payload_packets_are_reported(self):
        a = deduplicate([packet(3, 1, 0, 1.0), packet(3, 2, 160, 1.02),
                         packet(3, 3, 320, 1.04)])
        b = deduplicate([packet(3, 1, 0, 2.0), packet(3, 3, 320, 2.04, b'changed')])
        result = compare(a, b)
        self.assertEqual((result['matched_packets'], result['only_a'], result['only_b']), (2, 1, 0))
        self.assertEqual(result['payload_mismatches'], 1)

    def test_duplicate_identity_keeps_earliest_copy_and_detects_conflicts(self):
        result = deduplicate([packet(5, 9, 1440, 2.0),
                              packet(5, 9, 1440, 1.0),
                              packet(5, 9, 1440, 1.5, b'changed')])
        self.assertEqual(result['duplicate_copies'], 2)
        self.assertEqual(result['conflicting_duplicate_identities'], 1)
        row = next(iter(result['packets'].values()))
        self.assertEqual(row['arrival_wall_time'], 1.0)


if __name__ == '__main__':
    unittest.main()
