#!/usr/bin/env python3
"""Controlled symbol-domain AWGN development probe; not independent live gates.

Uses actual data mapping/trellis and expected transmit points only for scoring.
The monitor receives only the detector's own accumulated cost. SNR is referred
to this generated burst's mean complex-symbol power, before noise. No frontend,
RTP, timing drift, nonlinear projection or precoder is exercised here.
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
    monitor = DataQuality(detector, SimpleNamespace(rate=rate, nonlinear=False), baud,
                          min_symbols=len(signal))
    for i in range(0, len(signal), 2):
        detector.feed(*observed[i:i+2])
    # Sample before flush: drain changes decisions, not received cost.
    monitor.sample()
    detector.flush()
    decoded = np.asarray([s for pair in detector.decided for s in pair])
    errors = int(np.count_nonzero(abs(decoded - signal) > 1e-9))
    return dict(rate=rate, snr_db=snr_db, seed=seed, symbols=len(signal),
                transmit_power_grid_squared=power, injected_noise_grid_squared=float(np.mean(abs(noise)**2)),
                symbol_errors=errors, symbol_error_fraction=errors/len(signal),
                quality=monitor.report()['path_cost_per_symbol'])


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
