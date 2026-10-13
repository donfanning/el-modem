from pathlib import Path
import tempfile
import unittest
import json
from build_with_lineage import record_processing,digest

class LineageTests(unittest.TestCase):
    def test_records_exact_actual_input_output_and_preserves_call(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=root/'original';source.write_bytes(b'original library')
            result=root/'processed';calls=[]
            def upstream(src_name,dest_name,use_strip=False,use_upx=False,target_arch=None,codesign_identity=None):
                calls.append((src_name,dest_name,use_upx));result.write_bytes(b'loader changed library');return str(result)
            journal=root/'lineage.jsonl';wrapped=record_processing(upstream,journal)
            self.assertEqual(wrapped(str(source),'library.dll',use_upx=True),str(result))
            self.assertEqual(calls,[(str(source),'library.dll',True)])
            record=json.loads(journal.read_text());self.assertEqual(record['source_sha256'],digest(source))
            self.assertEqual(record['processed_sha256'],digest(result));self.assertTrue(record['upx'])
            self.assertNotEqual(record['source_sha256'],record['processed_sha256'])

if __name__=='__main__':unittest.main()
