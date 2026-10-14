"""Opt-in connected V.34 exchange on the existing channel/equalizer.

V34_CONNECTED_RATE=1 enables the bounded S/S-bar watcher. No startup fitting,
application reset or DCE reconnect is used. Independent-peer qualification is
still required before enabling this path by default.
"""
import copy
import math
import sys
from collections import deque
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np

import v34_datarx
import v34_negotiation
import v34_receive
import v34_rx
import v34_tx as tx
from v34_rate_change import RateExchange, RateQamBridge
from v34_streaming import update_data_baseband, trim_data_pcm


class ConnectedRate:
    LOOKAHEAD = 160

    def __init__(self, engine, role):
        self.engine, self.role = engine, role
        self.eq = engine._p4rcv.eq
        self.source = engine._p4_x0 if role == 'call' else engine._p3['x0']
        self.exchange = None
        self.stop = self.origin = None
        self.boundary_gain = None
        self.phase = 'idle'
        self.completed = deque(maxlen=8)
        self.mp_spans = deque(maxlen=32)
        self._guard()

    def _guard(self):
        self.eq._filter_future = 40 + math.ceil(self.LOOKAHEAD * self.eq.R.spl)

    def request(self, cap=None, *, resync=False, rates=None):
        if self.exchange is not None:
            raise RuntimeError('rate exchange already active')
        self._start(True, cap, resync=resync, rates=rates)

    def _start(self, initiating, cap=None, *, resync=False, rates=None):
        e = self.engine
        local = copy.deepcopy(e._local_mp)
        if sum((cap is not None, resync, rates is not None)) > 1:
            raise ValueError('choose scalar cap, resync, or directional rates')
        if cap is not None:
            requested = v34_negotiation.offer(cap)
            for key in ('max_c2a', 'max_a2c', 'mask'):
                local[key] = min(local[key], requested[key]) if key != 'mask' else requested[key] & local[key]
            if cap > max(e._local_mp['max_c2a'], e._local_mp['max_a2c']) * 2400:
                raise ValueError('requested rate exceeds original offer')
        if rates is not None:
            tx_rate, rx_rate = rates
            local = v34_negotiation.directional_offer(local,self.role,rates)
            p = e.negotiated
            peer = e.mp_ans if self.role == 'call' else e.mp_call
            if (not v34_negotiation.supported(tx_rate, p.tx.baud, peer.get('shaping', 0))
                    or not v34_negotiation.supported(rx_rate, p.rx.baud, local.get('shaping', 0))):
                raise ValueError('directional rate unsupported at current symbol rate')
        if resync:
            if cap is not None:
                raise ValueError('resync cannot also request a new ceiling')
            p = e.negotiated
            call_rate, answer_rate = ((p.tx.rate, p.rx.rate) if self.role == 'call'
                                     else (p.rx.rate, p.tx.rate))
            local['max_c2a'], local['max_a2c'] = call_rate // 2400, answer_rate // 2400
            local['mask'] &= v34_negotiation.offer(max(call_rate, answer_rate))['mask']
        peer = e.mp_ans if self.role == 'call' else e.mp_call
        p = e.negotiated
        self.exchange = RateExchange(local, peer, role=self.role, initiating=initiating,
            tx_baud=p.tx.baud, rx_baud=p.rx.baud, tx_carrier=p.tx.carrier,
            rx_carrier=p.rx.carrier, rtd=(e.rtd_samples or 0) / tx.FS,
            cme=bool((getattr(e, 'info0a' if self.role == 'call' else 'info0c', None) or {}).get('cme')))
        bridge = RateQamBridge(e._qam, self.exchange, self._fill_data)
        e._src = bridge
        e._queue = []
        self.phase = 'await_s'
        e._ev('connected rate exchange started (11.6)')

    def _fill_data(self, qam, exchange):
        e = self.engine
        e._dtx, e._dsc = exchange.data_tx, exchange.data_sc
        e._dscale = math.sqrt(qam.power / e._dtx.avg_power)
        e._fill_data(qam)

    def limit(self, available):
        return min(available, self.stop) if self.phase == 'drain' else available

    def _boundary(self):
        e, eq = self.engine, self.eq
        origin = getattr(e, '_rx_origin', 0)
        lo = max(origin, e.rx_t - 800)
        result = v34_rx.detect_s_flip(e._rx[lo-origin:], eq.R.fc, eq.R.baud)
        if result is None or result[0] != 'flip':
            return None
        update_data_baseband(e, eq.R, self.source)
        base = eq.R.lag + (eq.R.LEAD + eq.R.delay) * eq.R.spl
        estimate = round((lo + result[1] - self.source - base - eq.dt) / eq.R.spl)
        template = np.r_[tx.s_signal(16), tx.s_signal(16, True)]
        best = None
        for k in range(estimate - 8, estimate + 9):
            indices = np.arange(k - 16, k + 16)
            centers = base + indices * eq.R.spl + eq.dt + (indices - eq.k) * eq.vel
            positions = centers[:, None] + eq._offs
            bb_origin = getattr(eq.R, '_bb_origin', 0)
            if positions.min() - 6 < bb_origin or positions.max() + 46 >= bb_origin + len(eq.R._z):
                continue
            windows = v34_rx.interp(eq.R._z, positions.ravel() - bb_origin).reshape(len(indices), eq.L)
            phase = eq.phase + (indices - eq.k) * eq.freq
            observed = (windows @ eq.h) * np.exp(-1j * phase)
            gain = np.vdot(template, observed) / np.vdot(template, template).real
            error = float(np.sum(abs(observed - gain * template)**2) / max(float(np.sum(abs(observed)**2)), 1e-30))
            if best is None or error < best[0]:
                best = error, k, gain
        if best is None or best[0] > .15 or abs(best[2]) < 1e-9:
            return None
        return best

    def _flush_old(self):
        e, st = self.engine, self.engine._rx_state
        vit = st['vit']
        vit.flush()
        while vit.decided:
            pair = vit.decided.pop(0)
            if vit.emit_states:
                st.setdefault('pend_states', []).append(vit.decided_states.pop(0))
            st['pend'].extend(pair)
            if len(st['pend']) == 8:
                states = st.pop('pend_states', None)
                bits, _ = (st['drx'].decode_frame(st['pend'], trellis_states=states) if states is not None
                           else st['drx'].decode_frame(st['pend']))
                raw = v34_datarx.descramble_stream(bits, st['hist'][-23:], 'answer' if self.role == 'call' else 'call')
                st['hist'] = (st['hist'] + bits)[-23:]
                if e.bit_sink is not None:
                    e.bit_sink(raw)
                else:
                    st['bits'].extend(raw)
                st['pend'] = []
        e._async_bytes(st)

    def _enter_mp(self):
        e, old = self.engine, self.eq
        self._flush_old()
        eq = copy.copy(old)
        eq.h = old.h.copy() / self.boundary_gain
        delta = self.origin - old.k
        eq.k = self.origin
        eq.dt += delta * eq.vel
        eq.phase += delta * eq.freq
        eq._grid = None
        eq._g_zlen = -1
        eq._gradient_rows = []
        eq._filter_future = 40
        eq.adapt = True
        self.eq = eq
        e._p4rcv = SimpleNamespace(eq=eq, k_trn=self.origin)
        e._rate_capture_epoch = getattr(e, '_rate_capture_epoch', 0) + 1
        cap = self._capture()
        if cap is not None:
            cap.adopt_core(e)
        e._rate_local_mp = copy.deepcopy(self.exchange.local)
        self.mp_spans.clear()
        e._rx_state = None
        e.rx_qualification = {}
        e._connected_rate_b1 = True
        self.phase = 'mp'

    @staticmethod
    def _capture():
        module = sys.modules.get('v34_diagnostic')
        return module._current() if module is not None else None

    def _failed(self, reason):
        # The wrapper owns bounded retrains and retained link-layer state.
        # Do not turn an 11.6 failure into terminal startup failure.
        self.phase = 'failed'
        if self.exchange is not None:
            self.exchange.failed = reason
        self.engine._rate_retrain_reason = reason
        self.engine.failed_reason = None
        return True

    def poll(self):
        with getattr(self.engine, '_tx_lock', nullcontext()):
            return self._poll()

    def _poll(self):
        e, eq = self.engine, self.eq
        if self.phase not in ('idle', 'await_s'):
            update_data_baseband(e, eq.R, self.source)
        if self.phase in ('idle', 'await_s'):
            found = self._boundary()
            if found is not None:
                score, sbar, self.boundary_gain = found
                self.stop, self.origin = sbar - 128, sbar + 16
                if self.stop < eq.k:
                    return self._failed('connected-rate S was detected after data consumed its boundary')
                if self.exchange is None:
                    self._start(False)
                self.exchange.receive_s()
                self.phase = 'drain'
                e._ev(f'connected rate S-bar at symbol {sbar}, fit {score:.6g}')
        if self.exchange is not None and self.exchange.failed:
            return self._failed(self.exchange.failed)
        if self.phase == 'drain':
            if eq.k < self.stop:
                return False
            self._enter_mp()
            eq = self.eq
        if self.phase == 'mp':
            cap = self._capture()
            with cap.phase('phase4') if cap is not None else nullcontext():
                while eq.k < eq.available():
                    y = eq.step(decide=v34_rx.decide4)
                    events = self.exchange.feed_rotation(tx.rotation_of(v34_rx.decide4(y)))
                    self.mp_spans.extend(dict(start_bit=event['start'], end_bit=event['end'], fields=event['mp'])
                        for event in events if event['kind'] == 'mp')
                    if self.exchange.failed:
                        return self._failed(self.exchange.failed)
                    if self.exchange.peer_e:
                        if self.exchange.scanner.e_end % 2:
                            return self._failed('connected rate E ends between symbols')
                        e._rx_b1 = self.origin + self.exchange.scanner.e_end // 2
                        e.negotiated = self.exchange.negotiated
                        if cap is not None:
                            end = self.exchange.scanner.e_end
                            cap.phase4 = dict(k_trn=self.origin, trn_end=None, b1_symbol=e._rx_b1,
                                constellation=4, mp_spans=list(self.mp_spans),
                                e_span=dict(start_bit=end-20, end_bit=end, b1_symbol=e._rx_b1,
                                            processing_symbol=eq.k), procedure='connected-rate-11.6')
                        self.phase = 'b1'
                        break
        if self.phase == 'b1':
            if v34_receive.initialize(e, eq, 'answer' if self.role == 'call' else 'call', eq.R.baud):
                self.exchange.receive_b1(qualified=True)
                self.phase = 'ready'
            elif getattr(e, 'failed_reason', None):
                reason = e.failed_reason
                self.exchange.receive_b1(qualified=False)
                return self._failed(reason)
        if self.phase == 'ready' and self.exchange.tx_ready:
            p = self.exchange.negotiated
            e.data_rate, e.rx_rate = p.tx.rate, p.rx.rate
            e._rate_epoch = getattr(e, '_rate_epoch', 0) + 1
            if self.role == 'call':
                e.mp_ans = copy.deepcopy(self.exchange.peer)
            else:
                e.mp_call = copy.deepcopy(self.exchange.peer)
            e._connected_rate_b1 = False
            self.completed.append(dict(tx_rate=p.tx.rate, rx_rate=p.rx.rate,
                b1_symbol=e._rx_b1, local_mp=copy.deepcopy(self.exchange.local),
                peer_mp=copy.deepcopy(self.exchange.peer), qualification=copy.deepcopy(e.rx_qualification)))
            e._ev(f'connected rate exchange complete: tx={p.tx.rate} rx={p.rx.rate}')
            self.exchange = None
            self.phase = 'idle'
            self.stop = self.origin = None
            self._guard()
            return False
        if self.phase == 'mp':
            trim_data_pcm(e, eq, self.source, tracking=True)
        return self.phase not in ('idle', 'await_s', 'drain')


def controller(engine, role):
    current = getattr(engine, '_connected_rate', None)
    if current is None and getattr(engine, '_rx_state', None) is not None:
        current = engine._connected_rate = ConnectedRate(engine, role)
    return current


def request(engine, role, cap=None, *, resync=False, rates=None):
    if not getattr(engine, '_connected_rate_enabled', False):
        raise RuntimeError('connected rate changes are opt-in')
    current = controller(engine, role)
    if current is None:
        raise RuntimeError('not in qualified data mode')
    with getattr(engine, '_tx_lock', nullcontext()):
        current.request(cap, resync=resync, rates=rates)
