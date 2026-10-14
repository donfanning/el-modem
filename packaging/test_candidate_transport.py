import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from candidate_transport import encrypt_candidate

@unittest.skipUnless(shutil.which('age') and shutil.which('age-keygen'),'age test tools unavailable')
class CandidateTransportTests(unittest.TestCase):
    def test_only_matching_private_identity_decrypts_and_overwrite_is_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);identity=root/'identity';wrong=root/'wrong'
            for key in (identity,wrong):subprocess.run(['age-keygen','-o',str(key)],check=True,capture_output=True)
            recipient=subprocess.check_output(['age-keygen','-y',str(identity)],text=True).strip()
            original=root/'candidate';original.write_bytes(b'private review binary'*100)
            encrypted=root/'candidate.age';report=encrypt_candidate(original,encrypted,recipient,Path(shutil.which('age')))
            self.assertNotIn(original.read_bytes(),encrypted.read_bytes())
            result=subprocess.run(['age','-d','-i',str(wrong),str(encrypted)],capture_output=True)
            self.assertNotEqual(result.returncode,0)
            self.assertEqual(subprocess.check_output(['age','-d','-i',str(identity),str(encrypted)]),original.read_bytes())
            self.assertEqual(report['recipient'],recipient)
            with self.assertRaises(ValueError):encrypt_candidate(original,encrypted,recipient,Path(shutil.which('age')))

if __name__=='__main__':unittest.main()
