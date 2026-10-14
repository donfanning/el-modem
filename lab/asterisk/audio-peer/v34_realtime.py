"""Bounded per-call timing measurements, separate from modem sample clocks."""
from collections import deque
import time


class RealtimeStats:
    def __init__(self):
        self.values={};self.counts={};self.sums={};self.maxima={}
        self.started=time.monotonic();self.data_started=None

    def add(self,name,value,*,data=False):
        key=('data_' if data else 'startup_')+name
        self.values.setdefault(key,deque(maxlen=32768)).append(float(value))
        self.counts[key]=self.counts.get(key,0)+1
        self.sums[key]=self.sums.get(key,0.)+value
        self.maxima[key]=max(self.maxima.get(key,0.),value)
        if data and self.data_started is None:self.data_started=time.monotonic()

    def report(self):
        rows={}
        for name,values in list(self.values.items()):
            ordered=sorted(values);n=len(ordered)
            rows[name]=dict(count=self.counts[name],mean=self.sums[name]/self.counts[name],
                p99=ordered[min(n-1,int(.99*n))],maximum=self.maxima[name],
                p995=ordered[min(n-1,int(.995*n))],
                percentile_scope='last at most 32768 observations')
        tx=rows.get('data_tx_seconds',{});rx=rows.get('data_rx_seconds',{})
        combined=dict(mean=tx.get('mean',0)+rx.get('mean',0),
                      p99_upper_budget=tx.get('p995',0)+rx.get('p995',0),
                      percentile_scope='sum of marginal TX/RX p99.5 budgets; no independence assumed')
        return dict(elapsed_seconds=time.monotonic()-self.started,measurements=rows,
                    combined_data_processing_seconds=combined,
                    queue_age_scope='receive arrival to DSP, less declared jitter buffer')
