import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from public_manifest import NAME, check, refresh


class PublicManifestTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.git('init', '-q')
        self.git('config', 'user.name', 'Manifest test')
        self.git('config', 'user.email', 'test@example.invalid')
        (self.root / 'source.py').write_text('value = 1\n')
        (self.root / NAME).write_text(json.dumps(dict(source_commit='historical', files={})))
        self.git('add', 'source.py', NAME)
        self.git('commit', '-qm', 'fixture')

    def git(self, *args):
        subprocess.run(['git', '-C', str(self.root), *args], check=True, capture_output=True)

    def test_complete_snapshot_excludes_itself_and_untracked_files(self):
        (self.root / 'untracked.txt').write_text('excluded')
        self.assertTrue(refresh(self.root)['valid'])
        report = json.loads((self.root / NAME).read_text())
        self.assertEqual(report['baseline_private_source_commit'], 'historical')
        self.assertEqual(set(report['files']), {'source.py'})
        self.assertTrue(check(self.root)['valid'])

    def test_changed_and_new_tracked_files_fail(self):
        refresh(self.root)
        (self.root / 'source.py').write_text('value = 2\n')
        (self.root / 'new.py').write_text('value = 3\n')
        self.git('add', 'new.py')
        report = check(self.root)
        self.assertFalse(report['valid'])
        self.assertEqual(report['changed'], ['source.py'])
        self.assertEqual(report['missing'], ['new.py'])

    def test_removed_tracked_file_fails(self):
        refresh(self.root)
        self.git('rm', 'source.py')
        self.assertEqual(check(self.root)['extra'], ['source.py'])


if __name__ == '__main__':
    unittest.main()
