import struct
import unittest

from runtime_requirements import floor, macho_slices, native_format, needed_versions


def thin(cpu, minimum, legacy=False):
    packed = (minimum[0] << 16) | (minimum[1] << 8) | minimum[2]
    command = (struct.pack('<IIII', 0x24, 16, packed, packed) if legacy else
               struct.pack('<IIIIII', 0x32, 24, 1, packed, packed, 0))
    return struct.pack('<IIIIIIII', 0xFEEDFACF, cpu, 0, 6, 1, len(command), 0, 0) + command


class RuntimeRequirementsTests(unittest.TestCase):
    def test_glibc_needed_not_exports_or_libstdcpp(self):
        names = needed_versions('''Version definition section '.gnu.version_d':
  Name: GLIBC_99.0
Version needs section '.gnu.version_r':
  Name: GLIBC_2.9
  Name: GLIBC_2.38
  Name: GLIBCXX_3.4.99
''')
        result = floor([dict(name='library', version_requirements=names)], 'linux-x86_64')
        self.assertEqual(result['version'], '2.38')
        self.assertNotIn('GLIBC_99.0', names)

    def test_fat_uses_active_architecture(self):
        intel, arm = thin(0x1000007, (10, 15, 0)), thin(0x100000C, (11, 1, 0))
        offset = 8 + 40
        blob = (struct.pack('>II', 0xCAFEBABE, 2) +
                struct.pack('>IIIII', 0x1000007, 0, offset, len(intel), 0) +
                struct.pack('>IIIII', 0x100000C, 0, offset + len(intel), len(arm), 0) + intel + arm)
        records = [dict(name='universal', macho=macho_slices(blob))]
        self.assertEqual(floor(records, 'macos-x86_64')['version'], '10.15.0')
        self.assertEqual(floor(records, 'macos-arm64')['version'], '11.1.0')

    def test_legacy_minimum_command(self):
        rows = macho_slices(thin(0x1000007, (10, 13, 0), True))
        self.assertEqual(rows[0]['minimums'][0]['command'], 'LC_VERSION_MIN_MACOSX')
        self.assertEqual(rows[0]['minimums'][0]['minimum'], (10, 13, 0))

    def test_reject_truncated_load_command(self):
        with self.assertRaises(ValueError):
            macho_slices(thin(0x100000C, (11, 1, 0))[:-1])

    def test_fake_mz_data_is_not_pe(self):
        self.assertIsNone(native_format(b'MZ' + b'\0' * 100))
        data = bytearray(100)
        data[:2] = b'MZ';struct.pack_into('<I', data, 60, 80);data[80:84] = b'PE\0\0'
        self.assertEqual(native_format(data), 'PE')

    def test_no_version_is_not_a_compatibility_claim(self):
        self.assertIsNone(floor([], 'macos-arm64'))


if __name__ == '__main__':
    unittest.main()
