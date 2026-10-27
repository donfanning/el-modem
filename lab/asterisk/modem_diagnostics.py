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
    retrain = getattr(engine, 'retrain_stats', None)
    if isinstance(retrain, dict):
        stats['retrain'] = retrain
        stats['retrain_attempts'] = retrain.get('attempts')
        stats['retraining'] = bool(retrain.get('active', False))
    elif hasattr(engine, '_retrains'):
        stats['retrain_attempts'] = engine._retrains
        stats['retraining'] = bool(getattr(engine, '_retraining', False))
    stats['line_state'] = getattr(engine, 'state', None)
    return copy.deepcopy(stats)


def pcm_dbfs(sum_squares, samples):
    """RMS level relative to a full-scale sine in decoded signed-16 PCM."""
    if not samples or not finite(sum_squares) or sum_squares <= 0:
        return None
    reference = (32768.0 ** 2) / 2.0
    return 10.0 * math.log10((sum_squares / samples) / reference)


def recorded_fallbacks(engine, online_control=None):
    """Return observed post-connect fallback events; never infer from negotiated rate alone."""
    events = []
    seen = set()
    tracks_bell = hasattr(engine, 'bell103_fallback')
    if tracks_bell and getattr(engine, 'bell103_fallback', False):
        events.append(dict(kind='bell212a-to-bell103', source='engine-state'))
    retrain = getattr(engine, 'retrain_stats', None)
    tracks_v32 = bool(getattr(engine, 'is_v32', False) and isinstance(retrain, dict))
    if tracks_v32:
        for event in retrain.get('events', ()):
            old, new = event.get('previous_rate'), event.get('negotiated_rate')
            if isinstance(old, int) and isinstance(new, int) and new < old:
                key = ('rate-downshift', old, new)
                if key not in seen:
                    events.append(dict(kind='retrain-rate-downshift', from_rate=old,
                                       to_rate=new, source='retrain-history'))
                    seen.add(key)
    if isinstance(online_control, dict):
        for event in online_control.get('transactions', ()):
            if event.get('status') == 'fallback-retrained':
                old, new = event.get('initial_tx_rate'), event.get('tx_rate')
                key = ('rate-downshift', old, new)
                if key not in seen:
                    events.append(dict(kind='online-control-fallback-retrain',
                                       from_rate=old, to_rate=new, source='online-control'))
                    seen.add(key)
    if tracks_bell or tracks_v32 or isinstance(online_control, dict):
        return events
    return None
