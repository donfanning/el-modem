"""Bounded observation must preserve decoding and the interpolation operator."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import v34_data as vd
import v34_data_trace as trace
from v34_call import V34Caller
from v34_datarx import DataRx
from v34_negotiation import DirectionParameters
from v34_rx import SymbolEq
from v34_viterbi import TrellisDetector


class DataTraceTests(unittest.TestCase):
    def test_hook_copies_frames_without_changing_decode_or_windows(self):
        window, grid = SymbolEq.window, SymbolEq._trajectory_window
        trace.install()
        self.assertIs(SymbolEq.window, window)
        self.assertIs(SymbolEq._trajectory_window, grid)
        tx = vd.DataTx(33600, 3429)
        rx = DataRx(33600, 3429)
        baseline = copy.deepcopy(rx)
        vit = TrellisDetector(4*rx.M*2**rx.q, rx.J, rx.P, backend='python')
        vit.emit_states = True
        engine = V34Caller.__new__(V34Caller)
        engine._rx_state = dict(drx=rx, vit=vit, hist=[1]*23,
            receiver=SimpleNamespace(profile=DirectionParameters(33600,3429,1959)))
        engine._p4rcv = SimpleNamespace(eq=SimpleNamespace(k=0,h=[1+0j],timing_mode='gradient',
            _traj=True,_native_feedback=False))
        with tempfile.TemporaryDirectory() as directory:
            recorder = trace.DataTrace(Path(directory)/'trace.json',limit=2)
            engine._bounded_data_trace = recorder
            trace._context.engine = engine
            try:
                for _ in range(3):
                    symbols = tx.data_frame([1]*tx.bits_per_data_frame)
                    self.assertEqual(rx.decode_frame(symbols),baseline.decode_frame(symbols))
                saved = recorder.frames[0]['points'][0][:]
                symbols[0] = 123+456j
                self.assertEqual(recorder.frames[0]['points'][0],saved)
                recorder.close()
                result = json.loads(recorder.path.read_text())
                self.assertEqual(len(result['frames']),2)
                self.assertTrue(result['complete'])
                self.assertFalse(result['operator_replaced'])
                self.assertFalse(result['receiver_given_payload_truth'])
                self.assertEqual(result['header']['detector']['pending_intervals'],0)
            finally:
                trace._context.engine = None

    def test_partial_trace_is_explicit_on_close(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = trace.DataTrace(Path(directory)/'trace.json')
            recorder.close()
            result = json.loads(recorder.path.read_text())
            self.assertFalse(result['complete'])
            self.assertEqual(result['frames'],[])


if __name__ == '__main__':unittest.main()
