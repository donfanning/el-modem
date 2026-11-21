"""Digital-modem detection of the V.92 Phase 3 training patterns Ru, Ru-bar, Su and Su-bar.

The analogue modem sends these as one PCM sample per symbol (8000 symbols/s), repeating a
6-symbol pattern (clause 8.5.5, 8.5.6). The digital modem sees them after the line, the
analogue-to-digital conversion and G.711, with an unknown gain, so detection here is by
gain-invariant correlation against the four patterns at each of the six symbol phases,
on linear PCM.

``PatternTracker.feed`` takes linear samples in order and returns events::

    ("ru" | "ru-bar" | "su" | "su-bar", start_sample)

An event fires when the same pattern has matched, at one consistent phase, for ``confirm``
consecutive periods; ``start_sample`` is the first sample of the first matching period.
A change of pattern (Ru to Ru-bar is a sign flip) is a new event. Silence and noise
produce nothing.

Ru and Su are antisymmetric over half a period, so Ru delayed by three samples is Ru-bar. The
detector therefore names the first pattern it locks to as the plain one and reports "bar" only
for a flip against that phase, which is what clause 9.5.1.1.1 asks of the digital modem (detect
Ru and the Ru-to-Ru-bar transition). ``start_sample`` of the first event is accurate to the
three-sample ambiguity; a flip is exact to within the period.

Ru and Su correlate at 0.82, so on a distorted line the best-scoring template can be the wrong
family (the Gough recordings name many Sd runs "ru"). The call state knows which one it is waiting for, so ``families`` restricts the tracker to
the expected family ("ru" or "su"); with both allowed (the default) the name is a best guess.

Known limit: Ru's fundamental is 8000/6 = 1333 Hz and a sampled sine at that frequency correlates
0.94 with it, so a pure 1333 Hz tone is reported as Ru. The controller listens for Ru only at the
point of the handshake where the peer sends it, so this is not a path the call can reach.

Validation status: synthetic Ru/Su through G.711 (both laws), gain 60..3000, noise and a one-pole
line, any chunking; and the Gough recordings (42 files, mu-law, both channels). There, "ru" followed
by "ru-bar" occurs 53 times; the spacing between the two events clusters at 378-384 samples (20 of
53, i.e. 384T) and at 750-768 samples (about 13, two such runs), with the rest scattered. So the
detector does see real 384T pattern/flip pairs. Restricting it to one family on the left channel of
five V.90 recordings gives hits only as Ru-shaped ({+++---}), none as Su-shaped, so those pairs are
not the digital modem's Sd as I first guessed (Sd has the Su shape). What signal they are is open: a
pattern of Ru's shape and length in V.90 calls that I have not identified. Treat a hit as "a 384T
Ru-shaped pattern with a flip", and let the call state say which signal it is.
"""

from __future__ import annotations

import math

import numpy as np

PERIOD = 6
_A = math.sqrt(1.5)
PATTERNS = {
    "ru": np.array([1.0, 1.0, 1.0, -1.0, -1.0, -1.0]),
    "ru-bar": np.array([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0]),
    "su": np.array([_A, 0.0, _A, -_A, 0.0, -_A]),
    "su-bar": np.array([-_A, 0.0, -_A, _A, 0.0, _A]),
}
_NAMES = tuple(PATTERNS)
# Unit-norm, zero-mean templates. Ru and Su sum to zero already, so the mean is removed only
# from the received block.
_TEMPLATES = np.stack([PATTERNS[n] / np.linalg.norm(PATTERNS[n]) for n in _NAMES])
TX_DELAY_TAPS = 37
TX_DELAY_BULK = (TX_DELAY_TAPS - 1) // 2
TX_DELAY_PHASES = 1024
TX_DELAY_BETA = 8.0
TX_DELAY_HISTORY = TX_DELAY_TAPS + 1  # one extra sample for whole-delay carry


def _build_tx_delay_bank() -> np.ndarray:
    """Windowed-sinc fractional-delay bank shared by TX and the Su detector."""
    n = np.arange(TX_DELAY_TAPS, dtype=float)
    window = np.kaiser(TX_DELAY_TAPS, TX_DELAY_BETA)
    bank = np.empty((TX_DELAY_PHASES + 1, TX_DELAY_TAPS), dtype=float)
    for phase in range(TX_DELAY_PHASES + 1):
        delay = TX_DELAY_BULK + phase / TX_DELAY_PHASES
        taps = np.sinc(n - delay) * window
        taps /= np.sum(taps)
        bank[phase] = taps
    bank.setflags(write=False)
    return bank


TX_DELAY_BANK = _build_tx_delay_bank()


class PatternTracker:
    def __init__(self, *, confirm: int = 8, min_correlation: float = 0.8,
                 min_rms: float = 20.0, flip_confirm: int = 3,
                 fractional_timing: bool = False,
                 families: tuple = ("ru", "su")):
        if confirm < 2:
            raise ValueError("confirm must be at least 2 periods")
        if not 0.0 < min_correlation <= 1.0:
            raise ValueError("min_correlation must be in (0, 1]")
        if flip_confirm < 2 or flip_confirm > confirm:
            raise ValueError("flip_confirm must be between 2 and confirm")
        if not families or any(f not in ("ru", "su") for f in families):
            raise ValueError("families must be a non-empty subset of ('ru', 'su')")
        self._rows = [i for i, n in enumerate(_NAMES) if n.split("-")[0] in families]
        self.confirm = int(confirm)
        self.flip_confirm = int(flip_confirm)
        self.min_correlation = float(min_correlation)
        self.min_rms = float(min_rms)
        if type(fractional_timing) is not bool:
            raise TypeError("fractional_timing must be bool")
        self.fractional_timing = fractional_timing
        self._templates = _TEMPLATES[self._rows]
        self._template_ids = np.asarray(self._rows, dtype=np.intp)
        self._fractional_templates = None
        self._fractional_template_ids = None
        if self.fractional_timing:
            # Match the waveform generated by the TX sinc bank at a sub-sample
            # delay. The 18-sample bulk delay is three periods and therefore
            # does not change the six-sample pattern phase.
            phase_indices = np.linspace(0, TX_DELAY_PHASES, 33, endpoint=False).astype(np.intp)
            symbols = np.arange(PERIOD, dtype=np.intp)[:, None]
            tap_indices = np.arange(TX_DELAY_TAPS, dtype=np.intp)[None, :]
            rows = []
            ids = []
            for row in self._rows:
                pattern = PATTERNS[_NAMES[row]]
                templates = []
                for phase in phase_indices:
                    taps = TX_DELAY_BANK[phase]
                    source_indices = (symbols - TX_DELAY_BULK - tap_indices) % PERIOD
                    delayed = np.sum(pattern[source_indices] * taps[None, :], axis=1)
                    delayed -= float(np.mean(delayed))
                    templates.append(delayed / max(float(np.linalg.norm(delayed)), 1e-12))
                rows.append(np.asarray(templates))
                ids.extend([row] * len(phase_indices))
            self._fractional_templates = np.concatenate(rows, axis=0)
            self._fractional_template_ids = np.asarray(ids, dtype=np.intp)
        self._buf = np.empty(0)
        self._base = 0                 # sample index of _buf[0]
        self._scan = 0                 # next absolute sample to start a period at
        self.current: str | None = None
        self.current_phase: int | None = None
        self._run: list[tuple[int, str]] = []       # consecutive matching (start, name)
        self._misses = 0
        self.events: list[tuple[str, int]] = []

    def _classify(self, block: np.ndarray) -> str | None:
        block = block - block.mean()
        norm = float(np.linalg.norm(block))
        if norm / math.sqrt(PERIOD) < self.min_rms:
            return None
        scores = _TEMPLATES[self._rows] @ block / norm
        best = int(np.argmax(scores))
        return _NAMES[self._rows[best]] if scores[best] >= self.min_correlation else None

    def _classify_many(self, blocks: np.ndarray) -> np.ndarray:
        """Classify overlapping six-sample windows in one NumPy operation.

        The streaming state machine still consumes results in sample order,
        but doing a Python-to-NumPy mean/norm/argmax call for every candidate
        window made Phase 3 detection needlessly expensive during acquisition.
        A result is a ``_NAMES`` index, or -1 for an unclassified window.
        """
        blocks = np.asarray(blocks, dtype=float).reshape(-1, PERIOD)
        if not len(blocks):
            return np.empty(0, dtype=np.intp)
        centered = blocks - blocks.mean(axis=1, keepdims=True)
        norms = np.linalg.norm(centered, axis=1)
        templates = self._fractional_templates if self.fractional_timing else self._templates
        ids_by_template = (self._fractional_template_ids if self.fractional_timing
                           else self._template_ids)
        scores = centered @ templates.T
        np.divide(scores, norms[:, None], out=scores,
                  where=norms[:, None] > 0.0)
        best = np.argmax(scores, axis=1)
        best_score = scores[np.arange(len(blocks)), best]
        ids = ids_by_template[best]
        valid = ((norms / math.sqrt(PERIOD) >= self.min_rms)
                 & (best_score >= self.min_correlation))
        return np.where(valid, ids, -1).astype(np.intp, copy=False)

    def feed(self, samples) -> list[tuple[str, int]]:
        x = np.asarray(samples, dtype=float).reshape(-1)
        if not np.all(np.isfinite(x)):
            raise ValueError("samples must be finite")
        self._buf = np.concatenate((self._buf, x))
        out: list[tuple[str, int]] = []
        available = self._base + len(self._buf) - self._scan - PERIOD + 1
        first = self._scan - self._base
        if available > 0:
            blocks = np.lib.stride_tricks.sliding_window_view(
                self._buf[first:], PERIOD)[:available]
            classified = self._classify_many(blocks)
        else:
            classified = np.empty(0, dtype=np.intp)
        scan_start = self._scan
        while self._scan + PERIOD <= self._base + len(self._buf):
            name_id = int(classified[self._scan - scan_start])
            name = None if name_id < 0 else _NAMES[name_id]
            locked = self.current_phase is not None
            if not locked and name is not None and name.endswith("-bar"):
                # Ru shifted by three samples is Ru-bar (likewise Su), so the sign of a pattern
                # means something only against the phase first locked to. A first lock is always
                # taken on the plain alignment (three samples away); bar then means "flipped".
                name = None
            if name is None:
                self._run.clear()
                if locked:
                    # One bad period (a pattern change smears across the line) is not a loss;
                    # three in a row is.
                    self._misses += 1
                    if self._misses >= 3:
                        self.current, self.current_phase, self._misses = None, None, 0
                    self._scan += PERIOD
                else:
                    self._scan += 1                       # hunting: slide one sample
                continue
            self._misses = 0
            if self._run and self._run[-1][1] != name:
                self._run.clear()
            self._run.append((self._scan, name))
            self._scan += PERIOD
            # Ru-bar lasts only 24T (4 periods) before TRN1u, so a flip against the pattern
            # already locked needs fewer matching periods than a first detection.
            flip = (self.current is not None and name != self.current
                    and name.rstrip("-bar") == self.current.rstrip("-bar"))
            if len(self._run) >= (self.flip_confirm if flip else self.confirm):
                start = self._run[0][0]
                if name != self.current:
                    self.current, self.current_phase = name, start % PERIOD
                    event = (name, start)
                    self.events.append(event)
                    out.append(event)
                self._run = self._run[-1:]
        keep = max(0, self._scan - self._base)
        if keep:
            self._buf = self._buf[keep:]
            self._base += keep
        return out
