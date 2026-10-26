import io
import struct
import unittest

from tools.capture_npcap import device_matches, write_pcap_header, write_pcap_packet


class NpcapCaptureUtilityTests(unittest.TestCase):
    def test_pcap_header_is_classic_little_endian_ethernet(self):
        output = io.BytesIO()
        write_pcap_header(output, 65535, 1)
        self.assertEqual(output.getvalue(), struct.pack("<IHHIIII", 0xa1b2c3d4, 2, 4, 0, 0,
                                                        65535, 1))

    def test_packet_record_preserves_timestamp_lengths_and_bytes(self):
        output = io.BytesIO()
        write_pcap_packet(output, 123, 456789, 3, 5, b"abc")
        self.assertEqual(output.getvalue(), struct.pack("<IIII", 123, 456789, 3, 5)+b"abc")

    def test_device_guid_match_is_case_insensitive_and_brace_tolerant(self):
        self.assertTrue(device_matches(r"\\Device\\NPF_{E533A8A7-B730-4511-9E74-06230B0D3119}",
                                       "e533a8a7-b730-4511-9e74-06230b0d3119"))
        self.assertFalse(device_matches(r"\\Device\\NPF_{OTHER}", "e533a8a7"))


if __name__ == "__main__":
    unittest.main()
