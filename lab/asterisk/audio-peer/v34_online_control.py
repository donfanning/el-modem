"""DCE online controls executed by the receive/serial owner, serialized with TX.

Uses the documented MultiTech O1/O3/O4/O5 convention; O2 fast retrain is not
implemented. Results reflect physical completion, not merely request acceptance.
"""
import copy
from contextlib import nullcontext
import threading
import time
from collections import deque


class OnlineControl:
    @staticmethod
    def request_link_snapshot(eng):
        """Bounded owner-side evidence, before restart changes line DSP state."""
        with getattr(eng, '_v42_lock', nullcontext()):
            link = getattr(eng, '_v42', None)
            baseband = getattr(eng, 'bb', None)
            snapshot = dict(monotonic=time.monotonic(),
                            tx_samples=getattr(eng, '_txn', None),
                            rx_samples=getattr(baseband, 'n', None),
                            v42_mode=getattr(eng, 'v42_mode', None))
            if baseband is not None:
                snapshot['v32_receive'] = dict(
                    next_half_symbol_sample=float(baseband.pos),
                    half_symbol_step_samples=float(baseband.step),
                    next_symbol_index=int(eng._k),
                    carrier_phase_radians=float(eng._ph_acc),
                    frequency_loop=float(eng._freq),
                    equalizer_window_offset_half_symbols=int(eng._cofs))
            if link is None or not hasattr(link, '_unacked'):
                snapshot['lapm'] = None
                return snapshot
            snapshot['lapm'] = dict(state=link.state, vs=link.vs, va=link.va, vr=link.vr,
                outstanding_sequences=[int(item[0]) for item in link._unacked],
                queued_application_bytes=len(link._txq), queued_wire_bits=len(link._tx),
                retransmit_frames=len(link._retx), queued_compressed_octets=len(link._cq),
                poll_out=link._poll_out, peer_busy=link._peer_busy,
                ack_pending=link._ack_pending, t401_at=link._t401_at,
                layer_hold_bytes=len(getattr(eng, '_v42_hold', ())))
            return snapshot

    def __init__(self, engine, *, allow_v34=True, allow_v32_rate=False):
        self.engine = engine  # supplier: V.8 may select a different engine
        self.allow_v34 = allow_v34
        self.allow_v32_rate = allow_v32_rate
        self.lock = threading.RLock()
        self.pending = None
        self.history = deque(maxlen=16)

    @staticmethod
    def core(engine):
        return getattr(engine, 'caller', None) or getattr(engine, 'answerer', None)

    def request(self, parameter, *, origin='host', rates=None):
        with self.lock:
            eng = self.engine()
            if getattr(eng, 'is_v32', False):
                return self._request_v32(eng, parameter, origin, rates)
            if getattr(eng, 'is_v22', False):
                if (rates is not None or origin != 'host' or self.pending is not None
                        or getattr(eng, 'profile', None) != 'v22bis'
                        or not eng.connected or getattr(eng, '_state', None) != 'bis'
                        or parameter not in (4, 5)):
                    return False
                target = 1200 if parameter == 4 else 2400
                if eng.bit_rate == target:
                    return False
                entry = dict(parameter=parameter, requested_cap=target, origin=origin,
                             started_monotonic=time.monotonic(), initial_tx_rate=eng.bit_rate,
                             initial_rx_rate=eng.bit_rate, procedure='V22bis-section-6.6',
                             event_index=len(eng.events), status='pending')
                try:
                    eng.request_rate_change(target)
                except (RuntimeError, ValueError):
                    return False
                self.pending = (eng, eng, 0, entry)
                return True
            if not self.allow_v34:
                return False
            core = self.core(eng)
            if (self.pending is not None or core is None or not eng.connected
                    or getattr(eng, '_retraining', False)):
                return False
            rate = getattr(core, '_connected_rate', None)
            if rate is not None and rate.exchange is not None:
                return False
            ceiling = max(core._local_mp['max_c2a'], core._local_mp['max_a2c']) * 2400
            cap = None
            if rates is not None:
                if parameter not in (4, 5):return False
            elif parameter in (4, 5):
                cap = eng.bit_rate + (-2400 if parameter == 4 else 2400)
                mask = core._local_mp['mask']
                while 2400 <= cap <= ceiling and not mask & (1 << (cap // 2400 - 1)):
                    cap += -2400 if parameter == 4 else 2400
                if not 2400 <= cap <= ceiling:
                    return False
            elif parameter not in (1, 3):
                return False
            entry = dict(parameter=parameter, requested_cap=cap, origin=origin,
                         started_monotonic=time.monotonic(), initial_tx_rate=eng.bit_rate,
                         initial_rx_rate=eng.rx_bit_rate, status='pending')
            try:
                if parameter == 1:
                    eng.request_retrain()
                else:
                    if rates is None:
                        eng.request_rate_change(cap, resync=parameter == 3)
                    else:
                        if origin=='automatic':
                            eng._rate_fallback_rates = tuple(rates) if parameter==4 else (eng.bit_rate,eng.rx_bit_rate)
                        eng.request_rate_change(rates=rates)
                        entry['requested_rates'] = list(rates)
            except (RuntimeError, ValueError):
                if origin=='automatic' and hasattr(eng,'_rate_fallback_rates'):
                    del eng._rate_fallback_rates
                return False
            self.pending = (eng, core, getattr(core, '_rate_epoch', 0), entry)
            return True

    def activate(self):
        """Receive/control owner: enable passive V32 preamble tracking after LAPM."""
        with self.lock:
            eng = self.engine()
            if (self.allow_v32_rate and getattr(eng, 'is_v32', False)
                    and not getattr(eng, '_rate_changes_enabled', False)
                    and eng.connected and eng.state == 'data' and eng.v42_mode == 'lapm'):
                eng.enable_rate_changes()

    def _request_v32(self, eng, parameter, origin, rates):
        if (rates is not None or origin != 'host' or self.pending is not None
                or not eng.connected or eng.state != 'data'
                or getattr(eng, '_retraining', False)
                or getattr(getattr(eng, '_rrp', None), 'active', False)):
            return False
        cap = None
        if parameter != 1:
            if not self.allow_v32_rate or parameter not in (3, 4, 5):
                return False
            if eng.v42_mode != 'lapm':
                return False
            configured = sorted(eng._configured_rates)
            choices = [r for r in configured if
                       (r < eng.bit_rate if parameter == 4 else r > eng.bit_rate)]
            if parameter == 3:
                cap = eng.bit_rate
            elif not choices:
                return False
            else:
                cap = max(choices) if parameter == 4 else min(choices)
        entry = dict(parameter=parameter, requested_cap=cap, origin=origin,
                     started_monotonic=time.monotonic(),
                     initial_tx_rate=eng.tx.data_rate, initial_rx_rate=eng.bit_rate,
                     procedure='V32bis-section-7' if parameter == 1 else 'V32bis-section-8',
                     status='pending')
        if parameter == 1:
            with getattr(eng, '_line_lock', nullcontext()):
                entry['request_link_snapshot'] = self.request_link_snapshot(eng)
                if not eng.request_retrain():
                    return False
            old = eng
        else:
            if not eng.enable_rate_changes() or not eng.request_rate_change(cap):
                return False
            old = eng._rrp
            # Retain the actual transaction, not an index into bounded history.
            entry['_transaction'] = old.events[-1]
        self.pending = (eng, old, eng.retrain_stats['attempts'], entry)
        return True

    def _poll_v32_rate(self, eng, old, epoch, entry):
        transaction = entry['_transaction']
        recovery = eng.retrain_stats
        if eng.state in ('failed', 'down') or not eng.connected:
            entry.update(status='failed', reason=eng.failed_reason or 'carrier lost')
        elif recovery['attempts'] != epoch:
            if recovery['active']:
                return None
            event = recovery['events'][-1] if recovery['events'] else {}
            if 'completed_seconds' not in event or event.get('failure'):
                entry.update(status='failed', reason='superseding full retrain did not complete')
            else:
                entry.update(status='fallback-retrained', retrain=copy.deepcopy(event))
        elif eng._rrp is not old:
            entry.update(status='failed', reason='rate controller replaced before completion')
        elif transaction.get('status') == 'complete':
            if old.active:
                return None  # A following peer exchange must settle before CONNECT.
            entry.update(status='rate-exchanged', exchange=copy.deepcopy(transaction))
            if old.events[-1] is not transaction:
                entry['following_exchange'] = copy.deepcopy(old.events[-1])
        elif transaction.get('status') in ('failed', 'cleared', 'interrupted'):
            entry.update(status='failed', reason=transaction.get('reason', transaction['status']))
        else:
            return None
        del entry['_transaction']
        entry.update(completed_monotonic=time.monotonic(),
                     tx_rate=eng.tx.data_rate, rx_rate=eng.bit_rate)
        self.history.append(entry)
        self.pending = None
        return copy.deepcopy(entry)

    def poll(self):
        """Return a terminal record once; no mutation of the live DSP here."""
        with self.lock:
            if self.pending is None:
                return None
            eng, old, epoch, entry = self.pending
            current = self.core(eng)
            if entry.get('procedure') == 'V22bis-section-6.6':
                events = eng.events[entry['event_index']:]
                success = next((rate for name, rate in events if name == 'RATE_CHANGE_SUCCEEDED'), None)
                failure = next((rate for name, rate in events if name == 'RATE_CHANGE_FAILED'), None)
                if failure is not None:
                    entry.update(status='failed', reason=f'V.22bis rate change to {failure} failed')
                elif success is not None:
                    entry.update(status='rate-exchanged', tx_rate=success, rx_rate=success)
                else:
                    return None
                entry.update(completed_monotonic=time.monotonic())
                self.history.append(entry)
                self.pending = None
                return copy.deepcopy(entry)
            if entry.get('procedure') == 'V32bis-section-8':
                return self._poll_v32_rate(eng, old, epoch, entry)
            if getattr(old, 'is_v32', False):
                recovery = eng.retrain_stats
                events = recovery['events']
                if eng.state in ('failed', 'down') or not eng.connected:
                    entry.update(status='failed', reason=getattr(eng, 'failed_reason', '') or 'carrier lost')
                elif recovery['attempts'] != epoch:
                    entry.update(status='failed', reason='full retrain superseded before online command completion')
                elif recovery['active'] or not events:
                    return None
                elif 'completed_seconds' not in events[-1] or events[-1].get('failure'):
                    return None
                else:
                    entry.update(status='retrained', retrain=copy.deepcopy(events[-1]))
                entry.update(completed_monotonic=time.monotonic(),
                             tx_rate=eng.tx.data_rate, rx_rate=eng.bit_rate)
                self.history.append(entry)
                self.pending = None
                return copy.deepcopy(entry)
            if eng.state == 'failed':
                entry.update(status='failed', reason=eng.failed_reason)
            elif current is not old:
                if getattr(eng, '_retraining', False) or getattr(current, '_rx_state', None) is None:
                    return None
                entry['status'] = 'retrained' if entry['parameter'] == 1 else 'fallback-retrained'
            elif entry['parameter'] != 1 and getattr(current, '_rate_epoch', 0) > epoch:
                entry['status'] = 'rate-exchanged'
                entry['exchange'] = copy.deepcopy(current._connected_rate.completed[-1])
            else:
                return None
            entry.update(completed_monotonic=time.monotonic(), tx_rate=eng.bit_rate,
                         rx_rate=eng.rx_bit_rate, qualification=copy.deepcopy(getattr(current, 'rx_qualification', {})))
            self.history.append(entry)
            if entry.get('origin')=='automatic' and hasattr(eng,'_rate_fallback_rates'):
                del eng._rate_fallback_rates
            self.pending = None
            return copy.deepcopy(entry)

    def report(self):
        with self.lock:
            return dict(convention='O1-retrain/O3-resync/O4-down/O5-up',
                        pending=copy.deepcopy({k:v for k,v in self.pending[-1].items()
                                               if k != '_transaction'}) if self.pending else None,
                        transactions=copy.deepcopy(list(self.history)))
