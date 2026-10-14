"""Original recorded PCM through downconversion/interpolation/equalization.

The acquisition parameters are frozen from full replays. These short regions
exercise the physical receive path; they do not replace full startup gates.
"""
import hashlib
import json
from pathlib import Path
import unittest
import numpy as np
from v34_rx import Phase3Rx, interp
from v34_receive import Receiver
from v34_negotiation import DirectionParameters

FIXTURES=Path(__file__).parent/'fixtures/v34-reference/b1-pcm'

class B1PCMTests(unittest.TestCase):
    def test_raw_pcm_is_chunk_invariant_and_preserves_failures(self):
        for case in json.loads((FIXTURES/'manifest.json').read_text())['fixtures']:
            with self.subTest(recording=case['recording'],role=case['role']):
                source=FIXTURES/case['file']
                self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(),case['sha256'])
                z=np.load(source);raw=z['raw'].astype(float)
                fc=float(z['frequency']);baud=float(z['baud']);spl=8000/baud
                span=len(z['taps_before'])//2
                centers=z['centers']-float(z['source_offset'])-float(z['baseband_offset'])
                positions=centers[:,None]+(np.arange(2*span)-span)*spl/2
                phase=float(z['phase'])+(np.arange(len(centers))-int(z['margin']))*float(z['frequency_loop'])
                for chunk in (80,160,320):
                    r=Phase3Rx(baud,fc)
                    for end in range(chunk,len(raw)+chunk,chunk):
                        baseband=r.update_baseband(raw[:min(end,len(raw))])
                    baseband=baseband*np.exp(-2j*np.pi*fc*float(z['baseband_offset'])/8000)
                    windows=interp(baseband,positions.ravel()).reshape(positions.shape)*np.exp(-1j*phase[:,None])
                    observed=windows@z['taps_before']
                    np.testing.assert_allclose(observed,z['observed_before'],rtol=1e-8,atol=1e-7)
                    if 'observed_after' not in z or 'bit_errors' not in case['qualification']:continue
                    q=case['qualification'];shift=int(z['chosen_shift'])
                    receiver=Receiver(DirectionParameters(**q['profile']),
                        'answer' if case['role']=='call' else 'call',q['initial_trellis_state'])
                    corrected=windows[shift:shift+len(receiver.linear)]@z['taps_after']
                    np.testing.assert_allclose(corrected,z['observed_after'],rtol=1e-8,atol=1e-7)
                    receiver.qualify(corrected)
                    self.assertEqual(receiver.diagnostics['bit_errors'],q['bit_errors'])
                    self.assertEqual(receiver.diagnostics['qualified'],q['qualified'])

if __name__=='__main__':unittest.main()
