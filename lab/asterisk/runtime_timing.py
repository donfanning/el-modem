"""Opt-in bounded transport trace. Never changes a modem or RTP sample clock.

Only metadata is retained; no PCM, DTE bytes, credentials, or per-symbol state.
First and worst outliers retain preceding context even after a long cascade.
Thresholds trigger recording, not acceptance; existing qualification gates apply.
"""
from collections import deque
import copy
import threading


class TimingTrace:
    LIMIT = 64
    KINDS = frozenset(('queue', 'queue_data', 'queue_startup', 'rx_processing', 'tx_processing', 'sender_late',
                       'main_control', 'main_active', 'rtp_arrival_step',
                       'rtp_decode_step', 'clock_pair_span', 'clock_offset_step',
                       'tx_watchdog_startup', 'tx_watchdog_data'))
    THRESHOLDS = dict(queue_data=.050, queue_startup=.250, rx_processing=.020,
                      tx_processing=.020, sender_late=.020, main_control=.020,
                      main_active=.040, clock_pair_span=.001, clock_offset_step=.002)

    def __init__(self):
        self._lock = threading.RLock()
        self._recent = {key: deque(maxlen=self.LIMIT) for key in ('rx', 'tx', 'main')}
        self._counts = dict.fromkeys(self._recent, 0)
        self._events = {}
        self._last_packet = None
        self._clock_offset = None

    def record(self, stream, row):
        with self._lock:
            self._counts[stream] += 1
            self._recent[stream].append(dict(row))

    def event(self, kind, score, row):
        if kind not in self.KINDS:
            raise ValueError('unknown timing event')
        with self._lock:
            current = self._events.get(kind)
            capture = current is None or score > current['worst']['score']
            if capture:
                snapshot = dict(score=score, row=dict(row),
                    preceding={key: list(rows)[-32:] for key, rows in self._recent.items()})
            if current is None:
                self._events[kind] = dict(count=1, first=snapshot, worst=snapshot)
            else:
                current['count'] += 1
                if capture:
                    current['worst'] = snapshot

    def threshold(self, kind, score, row, *, data=False):
        key = ('queue_data' if data else 'queue_startup') if kind == 'queue' else kind
        if score > self.THRESHOLDS[key]:
            self.event(kind, score, row)
            if kind == 'queue':
                # Preserve each regime's worst queue separately; a longer
                # startup stall must not hide the data-phase acceptance failure.
                self.event(key, score, row)

    def packet(self, row):
        with self._lock:
            self.record('rx', row)
            previous = self._last_packet
            if previous is not None and row['ssrc'] == previous['ssrc']:
                seq_step = (row['sequence'] - previous['sequence']) & 0xffff
                ts_step = (row['timestamp'] - previous['timestamp']) & 0xffffffff
                if seq_step != 1 or ts_step != 160:
                    self.event('rtp_arrival_step', max(seq_step, ts_step),
                               dict(row, sequence_step=seq_step, timestamp_step=ts_step))
            self._last_packet = row
            if 'clock_mono_before' in row:
                span = row['clock_mono_after'] - row['clock_mono_before']
                self.threshold('clock_pair_span', span, row)
                offset = row['clock_wall'] - row['clock_mono_before']
                if self._clock_offset is not None:
                    self.threshold('clock_offset_step', abs(offset - self._clock_offset),
                                   dict(row, offset_step=offset - self._clock_offset))
                self._clock_offset = offset

    def report(self):
        with self._lock:
            return copy.deepcopy(dict(enabled=True, schema=1, capacity_per_stream=self.LIMIT,
                thresholds_seconds=self.THRESHOLDS, counts=self._counts,
                recent={key: list(rows) for key, rows in self._recent.items()}, events=self._events,
                scope='metadata only; first/worst outliers with preceding context; arrival steps may include reordering'))
