"""V.32bis section-7 sustained AA/AC request detector, independent of data bits.

Four 20-ms windows conservatively exceed the 128-symbol request duration.
Short section-8 rate-renegotiation preambles are not full-retrain requests.
"""
import numpy as np


class RequestTone:
    def __init__(self, role, fs=8000):
        if role not in ('call', 'answer') or fs <= 6000 or fs % 50:
            raise ValueError('require call/answer role and a supported 20-ms PCM clock')
        self.frequencies = (600, 3000) if role == 'call' else (1800,)
        self.size = fs // 50
        self.basis = np.exp(-2j*np.pi*np.array(self.frequencies)[:, None]
                            * np.arange(self.size)/fs)
        self.pending = np.empty(0)
        self.samples = 0
        self.run = 0
        self.latched = False

    def feed(self, samples):
        self.pending = np.concatenate((self.pending, np.asarray(samples, dtype=float)))
        events = []
        while len(self.pending) >= self.size:
            frame, self.pending = self.pending[:self.size], self.pending[self.size:]
            energy = np.dot(frame, frame)
            power = 2*np.sum(abs(self.basis @ frame)**2)/self.size
            fraction = float(power/energy) if energy > 0 else 0.
            self.samples += self.size
            if fraction >= .85:
                self.run += 1
                if self.run >= 4 and not self.latched:
                    events.append(dict(start_sample=self.samples-self.run*self.size,
                                       detected_sample=self.samples,
                                       frequencies=list(self.frequencies),
                                       last_window_fraction=fraction))
                    self.latched = True
            else:
                self.run = 0
                self.latched = False
        return events

