import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from python_component_notices import retain


class ComponentNoticeTests(unittest.TestCase):
    def test_exact_upstream_notice_is_retained(self):
        data = (Path(__file__).parent / 'third-party/CPython-SHA3.LICENSE').read_bytes()
        with tempfile.TemporaryDirectory() as folder, patch('python_component_notices.urlopen', return_value=io.BytesIO(data)):
            root = Path(folder)
            result = retain(root, (3, 11, 9))
            self.assertEqual((root / result['path']).read_bytes(), data)
            self.assertIn('/v3.11.9/', result['source'])

    def test_unreviewed_notice_fails_without_output(self):
        with tempfile.TemporaryDirectory() as folder, patch('python_component_notices.urlopen', return_value=io.BytesIO(b'changed')):
            with self.assertRaises(ValueError):
                retain(Path(folder), (3, 11, 17))
            self.assertEqual(list(Path(folder).rglob('*')), [])

    def test_other_python_minor_requires_review(self):
        with self.assertRaises(ValueError):
            retain(Path('/unused'), (3, 12, 0))


if __name__ == '__main__':
    unittest.main()
