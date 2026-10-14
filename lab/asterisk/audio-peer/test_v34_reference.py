"""Small real-waveform regressions, independent of the external corpus directory."""
import hashlib
import json
from pathlib import Path
import unittest
import numpy as np
from v34_info import InfoDemodulator
from v34_ans import ToneTracker

FIXTURES=Path(__file__).resolve().parent/'fixtures'/'v34-reference'
MANIFEST=json.loads((FIXTURES/'manifest.json').read_text())

class ReferenceWaveformTests(unittest.TestCase):
    def test_crc_checked_parameter_frames_from_three_real_modems(self):
        for fixture in MANIFEST['fixtures']:
            if fixture['kind']=='tone_reversal':continue
            with self.subTest(source=fixture['source']):
                raw=(FIXTURES/fixture['file']).read_bytes()
                self.assertEqual(hashlib.sha256(raw).hexdigest(),fixture['sha256'])
                x=np.frombuffer(raw,dtype='<i2').astype(float)
                got=InfoDemodulator(fixture['sender'],phases=8).decode(x,fixture['kind'])
                frames=[{k:v for k,v in f.items() if not k.startswith('_')} for f in got]
                self.assertIn(fixture['expected'],frames)

    def test_real_short_b_reversal_is_detected_before_carrier_disappears(self):
        fixture=next(f for f in MANIFEST['fixtures'] if f['kind']=='tone_reversal')
        x=np.frombuffer((FIXTURES/fixture['file']).read_bytes(),dtype='<i2').astype(float)
        # Independently check the waveform before testing the detector's timestamp.
        def coherent(lo,hi):
            t=np.arange(lo,hi)
            return np.sum(x[lo:hi]*np.exp(-2j*np.pi*1200*t/8000))/(hi-lo)
        pre,post=coherent(880,940),coherent(976,1020)
        self.assertGreater(abs(pre),500);self.assertGreater(abs(post),500)
        self.assertLess((pre*np.conj(post)).real/(abs(pre)*abs(post)),-.9)
        tracker=ToneTracker(1200);tracker.feed(np.concatenate([x,np.zeros(400)]))
        self.assertEqual(len(tracker.reversals),1)
        self.assertAlmostEqual(tracker.reversals[0],fixture['expected_reversal_sample'],delta=8)

if __name__=='__main__':unittest.main()
