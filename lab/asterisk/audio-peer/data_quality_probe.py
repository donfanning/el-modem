#!/usr/bin/env python3
"""Controlled symbol-domain AWGN development probe; not independent live gates.

Uses actual data mapping/trellis and expected transmit points only for scoring.
SNR is known injected symbol-domain AWGN referred to this generated burst's
mean complex-symbol power, before noise. Reports nearest-point slicer and
trellis outcomes against ground truth plus detector path-cost/survivor-gap
statistics. This checks whether those diagnostics track controlled quality;
it is not a live-SNR estimator. No frontend, RTP, timing drift, nonlinear
projection or precoder is exercised here.
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import v34_data as data
from v34_quality import DataQuality
from v34_viterbi import TrellisDetector
from types import SimpleNamespace


def measure(rate, snr_db, seed, frames, backend):
    baud = 24000 / 7
    tx = data.DataTx(rate, baud, False)
    bits = random.Random(seed)
    symbols = []
    for _ in range(frames):
        symbols.extend(tx.data_frame([bits.randrange(2) for _ in range(tx.bits_per_data_frame)]))
    signal = np.asarray(symbols)
    power = float(np.vdot(signal, signal).real / len(signal))
    sigma = (power / (2 * 10**(snr_db/10)))**.5
    rng = np.random.default_rng(seed + 10000)
    noise = rng.normal(0, sigma, len(signal)) + 1j*rng.normal(0, sigma, len(signal))
    observed = signal + noise
    detector = TrellisDetector(4*tx.M*2**tx.q, tx.J, tx.P, backend=backend)
    detector.emit_metrics = True
    monitor = DataQuality(detector, SimpleNamespace(rate=rate, nonlinear=False), baud,
                          min_symbols=len(signal))
    for i in range(0, len(signal), 2):
        detector.feed(*observed[i:i+2])
    # Sample before flush: drain changes decisions, not received cost.
    monitor.sample()
    detector.flush()
    decoded = np.asarray([s for pair in detector.decided for s in pair])
    slicer = detector.pts[np.argmin(abs(observed[:, None] - detector.pts[None, :])**2, axis=1)]
    trellis_errors = int(np.count_nonzero(abs(decoded - signal) > 1e-9))
    slicer_errors = int(np.count_nonzero(abs(slicer - signal) > 1e-9))
    disagreements = int(np.count_nonzero(abs(slicer - decoded) > 1e-9))
    gaps = np.asarray([row['survivor_gap'] for row in detector.decided_metrics], dtype=float)
    interval_errors = np.any((abs(decoded - signal) > 1e-9).reshape(-1, 2), axis=1)
    if len(gaps) != len(interval_errors):
        raise RuntimeError('survivor metrics do not align with decoded trellis intervals')
    def gap_summary(values):
        if not len(values):
            return dict(intervals=0, p10=None, median=None, p90=None)
        return dict(intervals=int(len(values)), p10=float(np.percentile(values, 10)),
                    median=float(np.median(values)), p90=float(np.percentile(values, 90)))
    return dict(rate=rate, snr_db=snr_db, seed=seed, symbols=len(signal),
                transmit_power_grid_squared=power, injected_noise_grid_squared=float(np.mean(abs(noise)**2)),
                slicer_symbol_errors=slicer_errors,
                slicer_symbol_error_fraction=slicer_errors/len(signal),
                trellis_symbol_errors=trellis_errors,
                trellis_symbol_error_fraction=trellis_errors/len(signal),
                slicer_trellis_disagreements=disagreements,
                slicer_trellis_disagreement_fraction=disagreements/len(signal),
                survivor_gap_per_interval=gap_summary(gaps),
                survivor_gap_correct_intervals=gap_summary(gaps[~interval_errors]),
                survivor_gap_error_intervals=gap_summary(gaps[interval_errors]),
                path_cost_per_symbol=monitor.report()['path_cost_per_symbol'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rates', type=int, nargs='+', default=[24000, 31200, 33600])
    parser.add_argument('--snr-db', type=float, nargs='+', default=[40, 32, 28, 24])
    parser.add_argument('--seeds', type=int, nargs='+', default=[3519, 3520, 3521])
    parser.add_argument('--frames', type=int, default=16)
    parser.add_argument('--backend', choices=['python', 'numba'], default='numba')
    args = parser.parse_args()
    if args.frames < 1:
        parser.error('frames must be positive')
    rows = [measure(rate, snr, seed, args.frames, args.backend)
            for rate in args.rates for snr in args.snr_db for seed in args.seeds]
    args.output.write_text(json.dumps(dict(domain='symbol-domain AWGN; linear unprecoded',
        backend=args.backend, frames=args.frames, results=rows), indent=2)+'\n')
    print(f'{len(rows)} development cases written to {args.output}')


if __name__ == '__main__':
    main()
