"""The opt-in V.34 post-B1 trace is bounded and contains no payload/audio."""
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import v34_datarx


class DataStateTraceTests(unittest.TestCase):
    def test_disabled_is_noop(self):
        core = SimpleNamespace(rx_t=123)
        with patch.object(v34_datarx, 'DATA_STATE_TRACE_ENABLED', False):
            v34_datarx.mark_data_state_gap(core, 100, 160)
            self.assertIsNone(v34_datarx.begin_data_state_frame(
                core, SimpleNamespace(k=3), SimpleNamespace(frame_pos=4), None))
        self.assertFalse(hasattr(core, '_data_state_trace'))

    def test_gap_keeps_pre_and_bounded_post_frames_without_payload(self):
        core = SimpleNamespace(rx_t=8000)
        eq = SimpleNamespace(k=100, timing_snapshot=lambda: dict(symbol=100, slips=0))
        drx = SimpleNamespace(frame_pos=20, m4=40, z=0, x_hist=[0j, 0j, 0j],
                              conv=SimpleNamespace(d=[0, 0, 0, 0]), errors=0)
        vit = SimpleNamespace(m4=40, metric=np.array([0.0, 1.0]))
        with patch.object(v34_datarx, 'DATA_STATE_TRACE_ENABLED', True):
            for i in range(40):
                eq.k = i
                token = v34_datarx.begin_data_state_frame(core, eq, drx, [0, 1, 2, 3])
                v34_datarx.end_data_state_frame(core, token, drx)
            v34_datarx.mark_data_state_gap(core, 7990, 320)
            for i in range(v34_datarx._DataStateTrace.POST_FRAMES + 10):
                eq.k = 100 + i
                token = v34_datarx.begin_data_state_frame(core, eq, drx, [0, 1, 2, 3], [
                    dict(decision_interval=i*4+j, lookahead_interval=i*4+j+31,
                         survivor_gap=1.0, metric_spread=2.0) for j in range(4)])
                drx.errors += int(i == 0)
                v34_datarx.end_data_state_frame(core, token, drx)
        report = core._data_state_trace.report()
        self.assertEqual(len(report['gaps']), 1)
        gap = report['gaps'][0]
        self.assertEqual(gap['concealed_samples'], 320)
        self.assertEqual(len(gap['pre_frames']), 32)
        self.assertEqual(len(gap['post_frames']), 256)
        self.assertEqual(gap['pre_frames'][-1]['symbol'], 39)
        self.assertEqual(gap['post_frames'][0]['mapper_errors_after'] -
                         gap['post_frames'][0]['mapper_errors_before'], 1)
        self.assertFalse(report['payload_or_audio_captured'])
        self.assertNotIn('bits', gap['post_frames'][0])
        self.assertNotIn('samples', gap['post_frames'][0])
        self.assertEqual(gap['post_frames'][0]['viterbi_emission_quality'][0]['survivor_gap'], 1.0)
        self.assertEqual(gap['post_frames'][0]['equalizer_state']['slips'], 0)
        json.dumps(report)


if __name__ == '__main__':
    unittest.main()
