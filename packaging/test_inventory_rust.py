import tempfile
import unittest
from pathlib import Path

from inventory_rust import crate_notices


class RustNoticeTests(unittest.TestCase):
    def test_retains_all_root_license_forms_and_ignores_sources(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            for name in ('LICENSE', 'LICENSE-MIT', 'LICENSE-UNICODE', 'COPYING.txt',
                         'NOTICE', 'COPYRIGHT', 'src.rs', 'README.md'):
                (root/name).write_text(name)
            self.assertEqual([path.name for path in crate_notices(root)],
                             ['COPYING.txt', 'COPYRIGHT', 'LICENSE', 'LICENSE-MIT',
                              'LICENSE-UNICODE', 'NOTICE'])


if __name__ == '__main__':
    unittest.main()
