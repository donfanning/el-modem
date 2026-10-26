"""Measured DSP facts for ATI reports; no conversion of path cost into SNR."""
import copy
import math


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def line_metrics(engine):
    """Snapshot an active engine. Missing measurements stay absent."""
    stats = {}
    if engine is None:
        return stats
    if getattr(engine, 'is_v32', False):
        stats['tx_rate'] = getattr(getattr(engine, 'tx', None), 'data_rate', None)
        stats['rx_rate'] = getattr(engine, 'bit_rate', None)
        stats['symbol_rate'] = 2400  # V.32 / V.32bis line symbol rate.
        if getattr(engine, '_sym_count', 0) > 0 and finite(getattr(engine, '_mse', None)):
            stats['decision_mse'] = engine._mse
            stats['decision_symbols'] = engine._sym_count
    else:
        stats['tx_rate'] = getattr(engine, 'bit_rate', None)
        stats['rx_rate'] = getattr(engine, 'rx_bit_rate', stats['tx_rate'])
    for part in ('caller', 'answerer'):
        core = getattr(engine, part, None)
        probe = getattr(core, 'probe_result', None)
        if isinstance(probe, dict) and finite(probe.get('snr_db')):
            stats['probe_snr_db'] = probe['snr_db']
        if core is not None:
            # This is the startup line-probe estimate, not ongoing data SNR.
            stats['symbol_rate'] = getattr(core, 'call_baud', None)
    if hasattr(engine, '_retrains'):
        stats['retrain_attempts'] = engine._retrains
        stats['retraining'] = bool(getattr(engine, '_retraining', False))
    stats['line_state'] = getattr(engine, 'state', None)
    return copy.deepcopy(stats)
