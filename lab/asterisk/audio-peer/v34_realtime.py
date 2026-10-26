"""Bounded per-call timing measurements, separate from modem sample clocks.

Samples are kept in fixed-size numeric rings so packet recording does not grow
Python containers or allocate one object per observation. CPU-time sampling is
opt-in because it adds two clock reads per measured packet.
"""
import os
import time
from array import array


class _Samples:
    __slots__ = ('values', 'size', 'next', 'count', 'total', 'maximum')

    def __init__(self, capacity):
        self.values = array('d', [0.0]) * capacity
        self.size = 0
        self.next = 0
        self.count = 0
        self.total = 0.0
        self.maximum = 0.0

    def add(self, value):
        value = float(value)
        self.values[self.next] = value
        self.next = (self.next + 1) % len(self.values)
        if self.size < len(self.values):
            self.size += 1
        self.count += 1
        self.total += value
        if value > self.maximum:
            self.maximum = value

    def report(self):
        if not self.size:
            return dict(count=0, mean=0.0, p99=0.0, maximum=0.0, p995=0.0,
                        percentile_scope='last at most 32768 observations')
        # `next` is the oldest slot once the ring has wrapped.
        if self.size == len(self.values):
            ordered = sorted(self.values[self.next:] + self.values[:self.next])
        else:
            ordered = sorted(self.values[:self.size])
        n = len(ordered)
        return dict(count=self.count, mean=self.total / self.count,
                    p99=ordered[min(n-1, int(.99*n))], maximum=self.maximum,
                    p995=ordered[min(n-1, int(.995*n))],
                    percentile_scope='last at most 32768 observations')


class RealtimeStats:
    CAPACITY = 32768

    def __init__(self, *, thread_timing=None):
        self.values = {}
        self.started = time.monotonic()
        self.data_started = None
        if thread_timing is None:
            thread_timing = os.environ.get('EL_MODEM_THREAD_TIMING') == '1'
        if thread_timing and not callable(getattr(time, 'thread_time', None)):
            raise RuntimeError('thread CPU timing requested but time.thread_time is unavailable')
        self.thread_timing = bool(thread_timing)

    def add(self, name, value, *, data=False):
        key = ('data_' if data else 'startup_') + name
        samples = self.values.get(key)
        if samples is None:
            samples = self.values[key] = _Samples(self.CAPACITY)
        samples.add(value)
        if data and self.data_started is None:
            self.data_started = time.monotonic()

    def report(self):
        rows = {name: samples.report() for name, samples in self.values.items()}
        tx=rows.get('data_tx_seconds',{});rx=rows.get('data_rx_seconds',{})
        combined=dict(mean=tx.get('mean',0)+rx.get('mean',0),
                      p99_upper_budget=tx.get('p995',0)+rx.get('p995',0),
                      percentile_scope='sum of marginal TX/RX p99.5 budgets; no independence assumed')
        return dict(elapsed_seconds=time.monotonic()-self.started,measurements=rows,
                    combined_data_processing_seconds=combined,
                    thread_cpu_timing_enabled=self.thread_timing,
                    queue_age_scope='receive arrival to DSP, less declared jitter buffer')
