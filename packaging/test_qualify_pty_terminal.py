import hashlib
from pathlib import Path
import tempfile
import unittest

from qualify_pty_terminal import qualify


class PtyArtifactBindingTests(unittest.TestCase):
    def test_candidate_hash_is_checked_before_platform_qualification(self):
        with tempfile.TemporaryDirectory() as folder:
            executable = Path(folder) / 'el-modem'
            executable.write_bytes(b'candidate executable')
            digest = hashlib.sha256(executable.read_bytes()).hexdigest()
            with self.assertRaisesRegex(RuntimeError, 'expected SHA-256'):
                qualify(executable, 'macos-arm64', Path(folder) / 'reports', '0' * 64)
            with self.assertRaisesRegex(ValueError, '64 hexadecimal'):
                qualify(executable, 'macos-arm64', Path(folder) / 'reports', 'not-a-hash')
            # The exact build hash clears the artifact-binding check, then the
            # host-platform guard stops this Linux-only unit test.
            with self.assertRaisesRegex(RuntimeError, 'Actual macOS required'):
                qualify(executable, 'macos-arm64', Path(folder) / 'reports', digest)


if __name__ == '__main__':
    unittest.main()
