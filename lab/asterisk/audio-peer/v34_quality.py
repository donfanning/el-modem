"""Bounded data-only observations; these are not a rate-selection rule.

The detector's accumulated minimum survivor cost is in squared forward-waveform
grid units, using its own decision-directed precoder history. It is not error
probability or a residual against known transmitted symbols. A wrong surviving
path can have a small cost. Divide by two symbols per 4D interval, not by signal
energy: the latter would reintroduce the density-dependent NMSE comparison.
Nonlinear profiles have nonuniform waveform spacing; no universal threshold is
implied by this report. Sample once per receive batch, without tracing symbols.
"""
from collections import deque
import math


class DataQuality:
    def __init__(self, detector, profile, baud, min_symbols=512, history=32, held_symbols=0):
        if min_symbols < 2 or history < 1:
            raise ValueError('positive batch size and history required')
        self.detector = detector
        self.profile = dict(rate=profile.rate, baud=baud,
                            nonlinear=bool(profile.nonlinear),
                            precoded=bool(any(detector.h)),
                            constellation_points=len(detector.pts))
        self.min_symbols = min_symbols
        self.windows = deque(maxlen=history)
        self._interval = detector.m4
        self._cost = detector.metric_offset
        self.symbols = self.window_count = self.discarded_symbols = 0
        self.invalid_batches = 0
        self.total_cost = 0.0
        self._held_symbols = held_symbols

    def sample(self, *, suspended=False, held_symbols=None):
        """Observe accumulated data cost. Holds discard the whole pending batch.

        A new detector requires a new collector, as production initialization
        does on every B1/core/rate epoch. Never difference costs across detectors.
        """
        interval = self.detector.m4
        cost = float(self.detector.metric_offset)
        symbols = 2 * (interval - self._interval)
        loss = cost - self._cost
        if held_symbols is not None:
            suspended = suspended or held_symbols != self._held_symbols
            self._held_symbols = held_symbols
        if symbols < 0 or not math.isfinite(loss) or loss < -1e-8:
            self.invalid_batches += 1
            self._interval, self._cost = interval, cost
            return None
        if suspended:
            self.discarded_symbols += symbols
            self._interval, self._cost = interval, cost
            return None
        if symbols < self.min_symbols:
            return None
        loss = max(0.0, loss)
        row = dict(end_interval=interval, symbols=symbols,
                   path_cost_per_symbol=loss / symbols)
        self.windows.append(row)
        self.window_count += 1
        self.symbols += symbols
        self.total_cost += loss
        self._interval, self._cost = interval, cost
        return row

    def report(self):
        return dict(domain='squared forward-waveform grid units per symbol',
                    reference='minimum survivor path; decision-directed feedback',
                    selection_enabled=False, profile=dict(self.profile),
                    symbols=self.symbols, windows=self.window_count,
                    discarded_symbols=self.discarded_symbols,
                    invalid_batches=self.invalid_batches,
                    path_cost_per_symbol=(self.total_cost / self.symbols
                                          if self.symbols else None),
                    recent=[dict(row) for row in self.windows])


def observe(engine, state, eq):
    monitor = state.get('quality')
    if monitor is not None:
        guard = getattr(eq, '_input_guard', None)
        monitor.sample(suspended=bool(getattr(guard, 'holding', False)),
                       held_symbols=getattr(guard, 'held_symbols', 0))
