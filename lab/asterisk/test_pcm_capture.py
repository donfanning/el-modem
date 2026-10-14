#!/usr/bin/env python3
"""Tests for the per-call audio recorder (pcm_capture.py)."""

import os
import struct
import tempfile
import unittest
import threading
import time
import json
from unittest.mock import patch

from pcm_capture import PcmCapture, capture_from_env


class CaptureTests(unittest.TestCase):
    def test_async_disk_stall_preserves_duplex_pcm_enqueue_times_and_close_drain(self):
        with patch.dict(os.environ,{'MODEM_CAPTURE_ASYNC':'1','MODEM_CAPTURE_TIMELINE':'1'}):
            cap=PcmCapture(self.prefix,'async-stall')
        entered,resume=threading.Event(),threading.Event()
        original=cap._timeline
        class PausedWriter:
            def write(self,text):
                entered.set()
                if not resume.wait(2):raise RuntimeError('write not resumed')
                return original.write(text)
            def close(self):original.close()
        cap._timeline=PausedWriter()
        context={'nested':{'value':1}}
        cap.rx(self._pcm([42]),context)
        self.assertTrue(entered.wait(1))
        returned=threading.Event()
        def transmit():
            cap.tx(self._pcm([7,8]));returned.set()
        tx=threading.Thread(target=transmit);tx.start()
        closer=threading.Thread(target=cap.close)
        try:
            self.assertTrue(returned.wait(.2),'media must not wait for timeline disk write')
            context['nested']['value']=2
            closer.start();closer.join(.05)
            self.assertTrue(closer.is_alive(),'close must drain the recording')
        finally:
            released=time.monotonic();resume.set();tx.join(2)
            if closer.ident is not None:closer.join(2)
            cap.close()
        self.assertTrue(cap.report()['complete'])
        self.assertFalse(cap._writer.is_alive())
        with open(cap.stem+'.rx.raw','rb') as f:self.assertEqual(f.read(),self._pcm([42]))
        with open(cap.stem+'.tx.raw','rb') as f:self.assertEqual(f.read(),self._pcm([7,8]))
        with open(cap.stem+'.timeline.jsonl') as f:rows=[json.loads(x) for x in f]
        self.assertEqual([r['direction'] for r in rows],['rx','tx'])
        self.assertTrue(all(r['monotonic']<released for r in rows))
        self.assertEqual(rows[0]['context'],{'nested':{'value':1}})

    def test_async_queue_overflow_is_explicit_and_never_a_complete_capture(self):
        with patch.dict(os.environ,{'MODEM_CAPTURE_ASYNC':'1'}):
            cap=PcmCapture(self.prefix,'async-overflow',queue_frames=1)
        entered,resume=threading.Event(),threading.Event()
        original=cap._rx
        class PausedWriter:
            def write(self,pcm):
                entered.set();resume.wait(2)
                return original.write(pcm)
            def close(self):original.close()
        cap._rx=PausedWriter()
        try:
            cap.rx(self._pcm([1]));self.assertTrue(entered.wait(1))
            cap.tx(self._pcm([2]))
            cap.rx(self._pcm([3]))
            self.assertEqual(cap.report()['error'],'capture queue overflow')
        finally:resume.set();cap.close()
        self.assertFalse(cap.report()['complete'])
        self.assertTrue(cap.truncated)
        with open(cap.stem+'.txt') as f:self.assertIn('truncated True',f.read())

    def test_async_write_error_is_reported_and_shutdown_finishes(self):
        with patch.dict(os.environ,{'MODEM_CAPTURE_ASYNC':'1'}):
            cap=PcmCapture(self.prefix,'async-error')
        original=cap._rx
        class FailedWriter:
            def write(self,pcm):raise OSError('injected disk error')
            def close(self):original.close()
        cap._rx=FailedWriter()
        cap.rx(self._pcm([1]));cap.close()
        report=cap.report()
        self.assertFalse(report['complete'])
        self.assertIn('injected disk error',report['error'])
        self.assertEqual(report['written_bytes']['rx'],0)

    def setUp(self):
        self._env = os.environ.pop("MODEM_CAPTURE", None)
        self._dir = tempfile.mkdtemp(prefix="cap")
        self.prefix = os.path.join(self._dir, "call")

    def tearDown(self):
        if self._env is not None:
            os.environ["MODEM_CAPTURE"] = self._env
        for name in os.listdir(self._dir):
            os.unlink(os.path.join(self._dir, name))
        os.rmdir(self._dir)

    @staticmethod
    def _pcm(values):
        return struct.pack(f"<{len(values)}h", *values)

    def test_both_directions_land_in_separate_files_with_the_samples_given(self):
        cap = PcmCapture(self.prefix, "originate-ms")
        cap.rx(self._pcm([1, -2, 3]))
        cap.tx(self._pcm([4, 5]))
        cap.close()
        with open(f"{cap.stem}.rx.raw", "rb") as fh:
            self.assertEqual(struct.unpack("<3h", fh.read()), (1, -2, 3))
        with open(f"{cap.stem}.tx.raw", "rb") as fh:
            self.assertEqual(struct.unpack("<2h", fh.read()), (4, 5))
        self.assertTrue(os.path.exists(f"{cap.stem}.txt"))

    def test_records_the_offset_between_the_two_file_origins(self):
        """Cross-direction timing needs the skew between the two files' origins.

        rx gets its first sample when media arrives, tx when the send loop starts, and those are
        not the same instant. Without the skew recorded, an index into .tx.raw compared against an
        index in .rx.raw silently mixes two timelines.
        """
        cap = PcmCapture(self.prefix, "originate-ms")
        cap.tx(self._pcm([4, 5]))
        cap.rx(self._pcm([1, -2, 3]))
        cap.close()
        with open(f"{cap.stem}.txt") as fh:
            meta = {}
            for line in fh.read().splitlines():
                parts = line.split(maxsplit=1)
                if len(parts) == 2:          # "note " with an empty value splits to one part
                    meta[parts[0]] = parts[1]
        self.assertNotEqual(meta["tx_t0_monotonic"], "unknown")
        self.assertNotEqual(meta["rx_t0_monotonic"], "unknown")
        self.assertNotEqual(meta["tx_minus_rx_origin_s"], "unknown")
        # tx was written first here, so its origin comes first. Compare against the two recorded
        # origins rather than asserting a sign: the two writes can land in the same clock tick, and
        # what matters is that the skew is *consistent with* them, not that it is visibly negative.
        tx_t0 = float(meta["tx_t0_monotonic"])
        rx_t0 = float(meta["rx_t0_monotonic"])
        self.assertLessEqual(tx_t0, rx_t0)
        self.assertAlmostEqual(float(meta["tx_minus_rx_origin_s"]), tx_t0 - rx_t0, places=3)

    def test_skew_is_reported_as_unknown_when_a_direction_never_ran(self):
        """A one-sided capture must say 'unknown', not report 0.0 -- which would read as
        'the two files share an origin' and invite exactly the mixed-timeline comparison."""
        cap = PcmCapture(self.prefix, "originate-ms")
        cap.rx(self._pcm([1, -2, 3]))
        cap.close()
        with open(f"{cap.stem}.txt") as fh:
            meta = {}
            for line in fh.read().splitlines():
                parts = line.split(maxsplit=1)
                if len(parts) == 2:          # "note " with an empty value splits to one part
                    meta[parts[0]] = parts[1]
        self.assertEqual(meta["tx_minus_rx_origin_s"], "unknown")

    def test_each_call_gets_its_own_pair_of_files(self):
        """Two calls in one modem process must not append over each other: a capture that mixed
        two calls would be worse than none when diagnosing an interop failure."""
        first = PcmCapture(self.prefix, "originate-ms")
        first.rx(self._pcm([7]))
        first.close()
        second = PcmCapture(self.prefix, "originate-ms")
        second.rx(self._pcm([8]))
        second.close()
        self.assertNotEqual(first.stem, second.stem)
        with open(f"{first.stem}.rx.raw", "rb") as fh:
            self.assertEqual(struct.unpack("<1h", fh.read()), (7,))
        with open(f"{second.stem}.rx.raw", "rb") as fh:
            self.assertEqual(struct.unpack("<1h", fh.read()), (8,))

    def test_a_long_call_is_bounded_and_says_so(self):
        """An engine that wedges mid-call would otherwise fill the disk for the life of the
        process, so each direction stops at the cap and the note records that it did."""
        cap = PcmCapture(self.prefix, "originate-v32", max_seconds=0.01)   # 80 samples/direction
        for _ in range(20):
            cap.rx(self._pcm([0] * 16))
            cap.tx(self._pcm([0] * 16))
        cap.close()
        self.assertTrue(cap.truncated)
        self.assertEqual(os.path.getsize(f"{cap.stem}.rx.raw"), 80 * 2)
        self.assertEqual(os.path.getsize(f"{cap.stem}.tx.raw"), 80 * 2)
        with open(f"{cap.stem}.txt") as fh:
            note = fh.read()
        self.assertIn("truncated True", note)
        self.assertIn("rx_seconds 0.01", note)

    def test_disabled_unless_the_prefix_is_set(self):
        self.assertIsNone(capture_from_env("originate-ms"))
        os.environ["MODEM_CAPTURE"] = self.prefix
        cap = capture_from_env("originate-ms")
        try:
            self.assertIsNotNone(cap)
            cap.close()
        finally:
            del os.environ["MODEM_CAPTURE"]

    def test_late_media_and_repeated_close_preserve_final_capture(self):
        cap = PcmCapture(self.prefix, "late-media")
        cap.rx(self._pcm([7]))
        cap.close()
        cap.rx(self._pcm([8]))
        cap.tx(self._pcm([9]))
        cap.close()
        with open(f"{cap.stem}.rx.raw", "rb") as fh:
            self.assertEqual(fh.read(), self._pcm([7]))
        with open(f"{cap.stem}.tx.raw", "rb") as fh:
            self.assertEqual(fh.read(), b"")
        with open(f"{cap.stem}.txt") as fh:
            metadata = fh.read()
        self.assertEqual(metadata.count("rx_seconds "), 1)
        self.assertIn("tx_t0_monotonic unknown", metadata)

    def test_close_waits_for_inflight_media_write(self):
        cap = PcmCapture(self.prefix, "concurrent-close")
        entered, resume, closing = threading.Event(), threading.Event(), threading.Event()
        original = cap._rx
        class PausedWriter:
            def write(self, pcm):
                entered.set()
                if not resume.wait(2):
                    raise RuntimeError("write was not resumed")
                return original.write(pcm)
            def close(self):
                original.close()
        cap._rx = PausedWriter()
        failures = []
        def write():
            try:
                cap.rx(self._pcm([42]))
            except Exception as exc:
                failures.append(exc)
        def close():
            closing.set()
            cap.close()
        writer = threading.Thread(target=write)
        closer = threading.Thread(target=close)
        writer.start()
        try:
            self.assertTrue(entered.wait(1))
            closer.start()
            self.assertTrue(closing.wait(1))
            closer.join(.05)
            self.assertTrue(closer.is_alive(), "close must wait for the active writer")
        finally:
            resume.set()
            writer.join(2)
            if closer.ident is not None:
                closer.join(2)
            cap.close()
        self.assertFalse(writer.is_alive())
        self.assertFalse(closer.is_alive())
        self.assertEqual(failures, [])
        with open(f"{cap.stem}.rx.raw", "rb") as fh:
            self.assertEqual(fh.read(), self._pcm([42]))
        self.assertEqual(cap._rx_bytes, 2)


if __name__ == "__main__":
    unittest.main()
