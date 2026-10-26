import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile
from unittest.mock import patch
from package_native_candidate import assemble

class CandidatePackageTests(unittest.TestCase):
    def fixture(self,root):
        (root/'packaging').mkdir();(root/'packaging/el_modem.spec').write_bytes(b'spec')
        for name in ('LICENSE','NOTICE','CREDITS.md','docs/AT_COMMANDS.md','docs/DESKTOP.md','docs/SERIAL.md','packaging/public/REDISTRIBUTION.md','packaging/serial/README.md','packaging/serial/build-driver.sh','packaging/serial/serialsim-flow.patch','packaging/serial/COPYING','packaging/com0com/README.md'):
            p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('original documentation')
        exe=root/'el-modem';exe.write_bytes(b'executable')
        reports=root/'reports';reports.mkdir();notices=root/'notices';notices.mkdir()
        digest=hashlib.sha256(exe.read_bytes()).hexdigest()
        record=dict(status='PASS',artifact_sha256=digest,serial_at_smoke='PASS',payload_requested=True,
            build_manifest=dict(local_modules={'NOTICE':hashlib.sha256((root/'NOTICE').read_bytes()).hexdigest()},specification_sha256=hashlib.sha256(b'spec').hexdigest()))
        for name in ('reference.json','native.json'):(reports/name).write_text(json.dumps(record))
        (reports/'desktop.json').write_text(json.dumps(dict(status='PASS')))
        (notices/'inventory.json').write_text(json.dumps(dict(artifact_sha256=digest,review_gaps=['test review gap'])))
        return exe,reports,notices

    def test_archive_hashes_modes_and_negative_binding_controls(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);exe,reports,notices=self.fixture(root)
            def pack(output,target='linux-x86_64'):
                return assemble(exe,root,reports,notices,output,target,'0.1.0-rc1','a'*40)
            with patch('package_native_candidate.platform.system',return_value='Linux'),patch('package_native_candidate.platform.machine',return_value='x86_64'):
                archive=root/'candidate.zip';result=pack(archive)
                self.assertEqual(result['status'],'REVIEW_REQUIRED')
                with zipfile.ZipFile(archive) as data:
                    self.assertEqual(data.getinfo('el-modem-0.1.0-rc1-linux-x86_64/el-modem').external_attr>>16,0o100755)
                    readme=data.read('el-modem-0.1.0-rc1-linux-x86_64/README.txt').decode()
                    self.assertIn('IAX2 calling',readme)
                    self.assertIn('Experimental pre-release build',readme)
                    self.assertIn('docs/REDISTRIBUTION.md',readme)
                    self.assertEqual(data.read('el-modem-0.1.0-rc1-linux-x86_64/docs/REDISTRIBUTION.md'),b'original documentation')
                    self.assertNotIn('not cleared for public binary redistribution',readme)
                    self.assertNotIn('IAX calling\nis not implemented',readme)
                with self.assertRaises(ValueError):pack(archive)
                with self.assertRaises(ValueError):pack(root/'wrong.zip','windows-x86_64')
                exe.write_bytes(b'different executable')
                with self.assertRaises(ValueError):pack(root/'changed.zip')
                exe.write_bytes(b'executable');(root/'NOTICE').write_bytes(b'changed source')
                with self.assertRaises(ValueError):pack(root/'changed-source.zip')

if __name__=='__main__':unittest.main()
