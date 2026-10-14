import unittest
from inventory_portable import is_native


class Classify(unittest.TestCase):
    def test_magic(self):
        self.assertTrue(is_native('a.dll', b'MZ\x90\x00'))
        self.assertTrue(is_native('libx', b'\x7fELF\x02'))
        self.assertTrue(is_native('Python', b'\xcf\xfa\xed\xfe....'))
        self.assertTrue(is_native('fat', b'\xca\xfe\xba\xbe....'))

    def test_data_files_are_not_native(self):
        for name, data in (('textual\\tree-sitter\\highlights\\bash.scm', b'(comment)'),
                           ('base_library.zip', b'PK\x03\x04'), ('el-modem-build.json', b'{"a":1}'),
                           ('textual-0.89.1.dist-info\\LICENSE', b'MIT'), ('', b'')):
            self.assertFalse(is_native(name, data))


if __name__ == '__main__':
    unittest.main()
