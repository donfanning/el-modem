"""Lab-only sample-preserving post-connect fade; truth never enters the receiver."""
import numpy as np


class DataFade:
    def __init__(self, enabled=False, *, delay=4000, edge=160, silence=640, recovery=4000):
        if min(delay, silence, recovery) < 0 or edge < 2:
            raise ValueError('invalid fade support')
        self.enabled = enabled
        self.delay, self.edge, self.silence, self.recovery = delay, edge, silence, recovery
        self.start = None
        self.samples_affected = self.zero_samples = 0

    def arm(self, sample):
        if self.start is None:
            self.start = int(sample) + self.delay

    @property
    def end(self):
        return None if self.start is None else self.start + 2*self.edge + self.silence

    def ready_for_payload(self, sample, protocol):
        # Raw mode verifies later recovery, not delivery of erased bytes. LAPM
        # queues data immediately so the gap must be survived during transfer.
        return not self.enabled or protocol == 'lapm' or (
            self.end is not None and sample >= self.end + self.recovery)

    def process(self, samples, origin):
        x = np.asarray(samples)
        if not self.enabled or self.start is None:
            return x.copy()
        pos = np.arange(len(x)) + origin - self.start
        gain = np.ones(len(x))
        down = (pos >= 0) & (pos < self.edge)
        mute = (pos >= self.edge) & (pos < self.edge + self.silence)
        up = (pos >= self.edge + self.silence) & (pos < 2*self.edge + self.silence)
        gain[down] = (self.edge-1-pos[down])/(self.edge-1)
        gain[mute] = 0
        gain[up] = (pos[up]-self.edge-self.silence)/(self.edge-1)
        self.samples_affected += int(np.count_nonzero(gain < 1))
        self.zero_samples += int(np.count_nonzero(gain == 0))
        return np.rint(x*gain).astype('<i2')

    def report(self):
        return dict(enabled=self.enabled, start_sample=self.start, end_sample=self.end,
                    edge_samples=self.edge, silence_samples=self.silence,
                    recovery_wait_samples=self.recovery, samples_affected=self.samples_affected,
                    zero_gain_samples=self.zero_samples, sample_positions_preserved=True,
                    impairment_truth_given_to_receiver=False)
