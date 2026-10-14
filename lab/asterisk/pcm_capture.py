"""Record both directions of a call's linear audio to disk, for offline analysis.

A real-PSTN interop failure is usually only diagnosable from the audio: which side stopped
talking, when the training sequence diverged, whether the slicer or the timing recovery lost
it. This sits in the media path rather than inside an engine, so it records whichever engine is
loaded and needs no per-modulation opt-in.

Set ``MODEM_CAPTURE=<prefix>`` to enable it. Each call writes::

    <prefix>.<label>.<stamp>.rx.raw    linear samples we received  (int16 LE, 8 kHz)
    <prefix>.<label>.<stamp>.tx.raw    linear samples we transmitted (int16 LE, 8 kHz)
    <prefix>.<label>.<stamp>.txt      what was configured, and how much audio was kept

The ``.raw`` files are exactly what ``audio-peer/replay_v32_capture.py`` consumes, so a
captured call can be replayed through the receive state machine without dialling again.
"""
import os
import time
import threading
import json
import copy
from queue import Queue, Full, Empty

SAMPLE_RATE = 8000
DEFAULT_MAX_SECONDS = 300.0          # per call, per direction: 4.8 MB, bounded even on a stuck call


class PcmCapture:
    """Appends linear PCM to a per-call pair of files. Cheap enough to leave switched on."""

    def __init__(self, prefix: str, label: str, max_seconds: float = DEFAULT_MAX_SECONDS,
                 note: str = "", *, queue_frames: int = 1024):
        self._max_bytes = int(max_seconds * SAMPLE_RATE * 2)
        self._rx_left = self._max_bytes
        self._tx_left = self._max_bytes
        self._rx_bytes = self._tx_bytes = 0
        self._lock = threading.Lock()
        self._closed = False
        self._close_done = threading.Event()
        self._async = os.environ.get('MODEM_CAPTURE_ASYNC') == '1'
        if self._async and queue_frames<1:raise ValueError('capture queue must be positive')
        self._failure = None
        self._written = {'rx':0,'tx':0}
        self._queue_peak = 0
        # Wall clock at each direction's first sample. The two .raw files have no shared origin:
        # the send loop starts when the DCE starts, but rx only gets its first sample when media
        # arrives, which is later by however long SIP/RTP setup took. Without this, an index into
        # .rx.raw and an index into .tx.raw are two different timelines and any cross-direction
        # timing taken from them is meaningless -- which is exactly how a 3 s "we answered too
        # late" reading came out of one of these files. See README on cross-direction timing.
        self._rx_t0: float | None = None
        self._tx_t0: float | None = None
        stamp = time.strftime("%Y%m%dT%H%M%S")
        os.makedirs(os.path.dirname(prefix) or ".", exist_ok=True)
        # Back-to-back calls (probe, hang up, dial again) land inside the same second, and a
        # colliding name would silently overwrite the previous call's audio, so disambiguate.
        stem = f"{prefix}.{label}.{stamp}"
        n = 1
        while os.path.exists(f"{stem}.rx.raw") or os.path.exists(f"{stem}.txt"):
            stem = f"{prefix}.{label}.{stamp}-{n}"
            n += 1
        self.stem = stem
        self._rx = open(f"{stem}.rx.raw", "wb")
        self._tx = open(f"{stem}.tx.raw", "wb")
        self._timeline = (open(f"{stem}.timeline.jsonl", "w")
                          if os.environ.get('MODEM_CAPTURE_TIMELINE')=='1' else None)
        self._started = time.time()
        with open(f"{stem}.txt", "w") as fh:
            fh.write(f"label {label}\nstarted {stamp}\nnote {note}\n"
                     f"max_seconds {max_seconds}\nsample_rate {SAMPLE_RATE}\n"
                     f"argv {' '.join(os.sys.argv)}\n")
        self.truncated = False
        if self._async:
            self._queue = Queue(maxsize=queue_frames)
            self._writer_wake = threading.Event()
            self._writer = threading.Thread(target=self._drain,daemon=True,name='pcm-recorder')
            self._writer.start()

    def rx(self, pcm: bytes, context=None) -> None:
        self._write(self._rx, pcm, "_rx", context)

    def tx(self, pcm: bytes, context=None) -> None:
        self._write(self._tx, pcm, "_tx", context)

    def _write(self, fh, pcm: bytes, which: str, context=None) -> None:
        # Media reception and session shutdown can run on different threads.
        # A reference to this recorder can survive the session's close call.
        with self._lock:
            if self._closed:
                return
            if self._async:
                self._enqueue(fh,pcm,which,context)
                return
            self._write_open(fh, pcm, which, context)

    def _enqueue(self,fh,pcm,which,context):
        if self._failure is not None:return
        room = self._rx_left if which=='_rx' else self._tx_left
        if room<=0:
            self.truncated=True
            return
        if len(pcm)>room:
            pcm=pcm[:room]
            self.truncated=True
        direction=which[1:]
        offset=(self._rx_bytes if which=='_rx' else self._tx_bytes)//2
        now=time.monotonic()
        row=dict(direction=direction,sample_offset=offset,samples=len(pcm)//2,
                 monotonic=now,wall_time=time.time(),context=copy.deepcopy(context or {}))
        try:
            self._queue.put_nowait((fh,bytes(pcm),row))
        except Full:
            self._failure='capture queue overflow'
            self.truncated=True
            return
        self._queue_peak=max(self._queue_peak,self._queue.qsize())
        if which=='_rx':
            if self._rx_t0 is None and pcm:self._rx_t0=now
            self._rx_left-=len(pcm);self._rx_bytes+=len(pcm)
        else:
            if self._tx_t0 is None and pcm:self._tx_t0=now
            self._tx_left-=len(pcm);self._tx_bytes+=len(pcm)

    def _drain(self):
        while True:
            # Drain in bursts rather than wake and contend for the GIL per frame.
            self._writer_wake.wait(.050)
            self._writer_wake.clear()
            while True:
                try:item=self._queue.get_nowait()
                except Empty:break
                if item is None:
                    self._queue.task_done()
                    return
                fh,pcm,row=item
                try:
                    fh.write(pcm)
                    if self._timeline is not None:self._timeline.write(json.dumps(row)+'\n')
                    with self._lock:self._written[row['direction']]+=len(pcm)
                except Exception as exc:
                    with self._lock:
                        self._failure=f'capture writer failed: {type(exc).__name__}: {exc}'
                        self.truncated=True
                finally:self._queue.task_done()

    def report(self):
        with self._lock:
            written=dict(self._written) if self._async else dict(rx=self._rx_bytes,tx=self._tx_bytes)
            accepted=dict(rx=self._rx_bytes,tx=self._tx_bytes)
            return dict(mode='async' if self._async else 'sync',
                closed=self._close_done.is_set(),truncated=self.truncated,error=self._failure,
                complete=(not self.truncated and self._failure is None and written==accepted)
                         if self._close_done.is_set() else None,
                accepted_bytes=accepted,written_bytes=written,
                queue_capacity=self._queue.maxsize if self._async else 0,
                queue_peak=self._queue_peak,
                writer_interval_seconds=.050 if self._async else 0.,
                timestamp_source='capture enqueue' if self._async else 'synchronous write')

    def _write_open(self, fh, pcm: bytes, which: str, context=None) -> None:
        if which == "_rx":
            if self._rx_t0 is None and pcm:
                self._rx_t0 = time.monotonic()
        elif self._tx_t0 is None and pcm:
            self._tx_t0 = time.monotonic()
        room = self._rx_left if which == "_rx" else self._tx_left
        if room <= 0:
            self.truncated = True
            return
        if len(pcm) > room:
            pcm = pcm[:room]
            self.truncated = True
        offset = (self._rx_bytes if which=='_rx' else self._tx_bytes)//2
        fh.write(pcm)
        if self._timeline is not None:
            self._timeline.write(json.dumps(dict(direction=which[1:], sample_offset=offset,
                samples=len(pcm)//2, monotonic=time.monotonic(), wall_time=time.time(),
                context=context or {}))+'\n')
        if which == "_rx":
            self._rx_left -= len(pcm)
            self._rx_bytes += len(pcm)
        else:
            self._tx_left -= len(pcm)
            self._tx_bytes += len(pcm)

    def close(self) -> None:
        if self._async:
            with self._lock:
                closing=self._closed
                self._closed=True
            if closing:
                self._close_done.wait()
                return
            try:
                # Shutdown may wait for disk; media paths never wait for writes.
                self._writer_wake.set()
                self._queue.put(None)
                self._writer.join()
                self._close_open()
            finally:self._close_done.set()
            return
        with self._lock:
            if self._closed:
                return
            self._closed = True
            try:self._close_open()
            finally:self._close_done.set()

    def _close_open(self) -> None:
        for fh in (self._rx, self._tx, self._timeline):
            if fh is None:
                continue
            try:
                fh.close()
            except OSError as exc:
                self._failure=f'capture close failed: {exc}'
                self.truncated=True
        rx_s = self._rx_bytes / 2 / SAMPLE_RATE
        tx_s = self._tx_bytes / 2 / SAMPLE_RATE
        skew = (self._tx_t0 - self._rx_t0) if (self._rx_t0 and self._tx_t0) else None
        with open(f"{self.stem}.txt", "a") as fh:
            fh.write(f"recording_mode {'async' if self._async else 'sync'}\n"
                     f"capture_error {self._failure or 'none'}\n"
                     f"queue_peak {self._queue_peak}\n")
            fh.write(f"rx_seconds {rx_s:.2f}\ntx_seconds {tx_s:.2f}\n"
                     f"truncated {self.truncated}\nwall_seconds {time.time() - self._started:.2f}\n")
            # To line up an index in one .raw against an index in the other, add this skew to the
            # tx index: tx_index_in_rx_time = tx_index + skew. Written whether or not it is known,
            # so a reader can tell "no skew" from "skew unmeasured".
            fh.write("rx_t0_monotonic {}\n"
                     "tx_t0_monotonic {}\n"
                     "tx_minus_rx_origin_s {}\n".format(
                         f"{self._rx_t0:.6f}" if self._rx_t0 else "unknown",
                         f"{self._tx_t0:.6f}" if self._tx_t0 else "unknown",
                         f"{skew:+.4f}" if skew is not None else "unknown"))
        print(f"[capture] {self.stem} rx={rx_s:.1f}s tx={tx_s:.1f}s"
              f"{' TRUNCATED' if self.truncated else ''}"
              f"{'' if skew is None else f' tx-rx origin {skew:+.3f}s'}", flush=True)


def capture_from_env(label: str, note: str = "") -> PcmCapture | None:
    """Builds the recorder ``MODEM_CAPTURE`` asks for, or None when it is not set."""
    prefix = os.environ.get("MODEM_CAPTURE")
    if not prefix:
        return None
    try:
        cap = PcmCapture(prefix, label, float(os.environ.get("MODEM_CAPTURE_SECONDS", DEFAULT_MAX_SECONDS)), note)
    except OSError as exc:
        if os.environ.get('MODEM_CAPTURE_REQUIRED')=='1':
            raise
        print(f"[capture] disabled, cannot write {prefix}: {exc}", flush=True)
        return None
    print(f"[capture] recording to {cap.stem}", flush=True)
    return cap
