"""Opt-in bounded data trace. Observes the live operator; never replaces windows."""
from __future__ import annotations

import atexit
from dataclasses import asdict
import functools
import json
import os
from pathlib import Path
import threading

_context = threading.local()
_traces = []
_installed = False


def pairs(values):
    return [[float(complex(v).real), float(complex(v).imag)] for v in values]


class DataTrace:
    def __init__(self, path, limit=256, start_interval=0, start_on_erasure=False,
            trellis_only=False):
        self.path = Path(path)
        self.limit = limit
        self.start_interval = start_interval
        self.start_on_erasure = start_on_erasure
        self.trellis_only = trellis_only
        self.symbols, self.intervals, self.frames, self.decisions = [], [], [], []
        self.header = None
        self.done = False
        self.writer = None
        self.error = None

    def initialize(self, engine):
        st = engine._rx_state
        eq, d, vit = engine._p4rcv.eq, st['drx'], st['vit']
        self.header = dict(role='call' if engine.__class__.__name__ == 'V34Caller' else 'answer',
            profile=asdict(st['receiver'].profile), hist=list(st['hist']),
            mapper=dict(z=d.z, m4=d.m4, frame_pos=d.frame_pos, conv=list(d.conv.d), x_hist=pairs(d.x_hist)),
            detector=dict(L=len(vit.pts), J=vit.J, P=vit.P, depth=vit.depth, m4=vit.m4,
                metric=vit.metric.tolist(), x_hist=pairs(vit.x_hist), precoder=pairs(vit.h),
                w=vit.w, nonlinear=bool(vit.nonlinear), energy=float(vit.energy),
                backend=vit.backend, emit_states=bool(vit.emit_states), pending_intervals=len(vit.hist),
                pending_history=[dict(interval=int(h['interval']), prev=h['prev'].tolist(),
                    s0=h['s0'].tolist(), s1=h['s1'].tolist(), p0=h['p0'].tolist(), U=h['U'].tolist())
                    for h in vit.hist]),
            eq=dict(symbol=eq.k, taps=pairs(eq.h), timing_mode=eq.timing_mode,
                trajectory_requested=bool(eq._traj), native_feedback=bool(eq._native_feedback),
                native_interpolation=bool(getattr(eq, '_native_interp', False))))
        self.header['trace_mode'] = 'trellis-only' if self.trellis_only else 'full'

    def publish(self, complete=True):
        if self.done:
            return
        self.done = True
        payload = dict(header=self.header, symbols=self.symbols, intervals=self.intervals,
            frames=self.frames, decisions=self.decisions, complete=complete, frame_limit=self.limit,
            requested_start_interval=self.start_interval,
            requested_start_on_erasure=self.start_on_erasure,
            operator_replaced=False, receiver_given_payload_truth=False,
            production_snapshots_enabled=False)
        def write():
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self.path.write_text(json.dumps(payload, separators=(',', ':')) + '\n')
            except Exception as error:
                self.error = repr(error)
        # Serialize once off the RX thread. No I/O per symbol/frame.
        self.writer = threading.Thread(target=write, daemon=True)
        self.writer.start()

    def close(self):
        self.publish(complete=False)
        if self.writer:
            self.writer.join()
        if self.error:
            raise RuntimeError('V.34 data trace write failed: ' + self.error)


def _current(*, trigger=False):
    engine = getattr(_context, 'engine', None)
    if engine is None or getattr(engine, '_rx_state', None) is None:
        return None, None
    trace = engine._bounded_data_trace
    if trace.done:
        return None, None
    if trace.header is None:
        if trace.start_on_erasure and not trigger:
            return engine, None
        # A data-mode capture can be armed well before a late impairment. Keep
        # the observer dormant until the live trellis reaches the requested
        # interval, then snapshot the actual state at that point.
        vit = engine._rx_state['vit']
        if vit.m4 < trace.start_interval:
            return engine, None
        trace.initialize(engine)
    return engine, trace


def install():
    global _installed
    if _installed:
        return
    _installed = True
    from v34_call import V34Caller
    from v34_ans import V34Answerer
    from v34_rx import SymbolEq
    from v34_datarx import DataRx, descramble_stream
    from v34_viterbi import TrellisDetector

    for cls in (V34Caller, V34Answerer):
        original = cls._data_rx
        def wrap(original):
            @functools.wraps(original)
            def receive(engine, *args, **kwargs):
                path = os.environ.get('V34_DATA_TRACE')
                if not path:
                    return original(engine, *args, **kwargs)
                if not hasattr(engine, '_bounded_data_trace'):
                    start_interval = max(0, int(os.environ.get('V34_DATA_TRACE_FROM_INTERVAL', '0')))
                    limit = max(1, int(os.environ.get('V34_DATA_TRACE_LIMIT', '256')))
                    start_on_erasure = os.environ.get('V34_DATA_TRACE_ON_ERASURE') == '1'
                    trellis_only = os.environ.get('V34_DATA_TRACE_TRELLIS_ONLY') == '1'
                    engine._bounded_data_trace = DataTrace(path, limit=limit,
                        start_interval=start_interval, start_on_erasure=start_on_erasure,
                        trellis_only=trellis_only)
                    _traces.append(engine._bounded_data_trace)
                previous = getattr(_context, 'engine', None)
                _context.engine = engine
                try:
                    return original(engine, *args, **kwargs)
                finally:
                    _context.engine = previous
            return receive
        cls._data_rx = wrap(original)

    original_step = SymbolEq.step
    @functools.wraps(original_step)
    def step(eq, *args, **kwargs):
        engine = getattr(_context, 'engine', None)
        existing = getattr(engine, '_bounded_data_trace', None) if engine is not None else None
        if existing and existing.trellis_only:
            return original_step(eq, *args, **kwargs)
        engine, trace = _current()
        row = None
        if trace and eq is engine._p4rcv.eq and len(trace.symbols) < 8*trace.limit + 64:
            k = eq.k
            row = dict(k=k, sample_position=float(eq.R.lag+(eq.R.LEAD+k+eq.R.delay)*eq.R.spl+eq.dt),
                dt=float(eq.dt), vel=float(eq.vel), phase=float(eq.phase), freq=float(eq.freq))
        y = original_step(eq, *args, **kwargs)
        if row is not None:
            row.update(y=pairs([y])[0], trajectory_used=getattr(eq, '_tj_k', -1) == row['k'],
                dt_after=float(eq.dt), vel_after=float(eq.vel), phase_after=float(eq.phase))
            trace.symbols.append(row)
        return y
    SymbolEq.step = step

    original_feed = TrellisDetector.feed
    @functools.wraps(original_feed)
    def feed(det, y0, y1, *args, **kwargs):
        erasure=bool(kwargs.get('erasure',False))
        engine, trace = _current(trigger=erasure)
        active = trace and det is engine._rx_state['vit'] and len(trace.intervals) < 4*trace.limit+32
        m4 = det.m4
        result = original_feed(det, y0, y1, *args, **kwargs)
        if active:
            trace.intervals.append(dict(m4=m4, y=pairs([y0, y1]), metric=det.metric.tolist(),
                erasure=erasure, depth=int(det.depth),
                hold_active=bool(getattr(det, '_gap_hold_active', False)),
                hold_erased_intervals=int(getattr(det, '_gap_erased_intervals', 0)),
                hold_clean_intervals=int(getattr(det, '_gap_clean_intervals', 0))))
            if trace.trellis_only and len(trace.intervals) >= trace.limit:
                trace.publish()
        return result
    TrellisDetector.feed = feed

    # Observe the exact survivor choice at the production traceback boundary.
    # Do not enable emit_states/emit_metrics mid-stream: their pending lists are
    # part of the frame interface and a runtime toggle can mismatch frame size.
    original_path = TrellisDetector._path
    @functools.wraps(original_path)
    def path(det, steps, with_states=False):
        engine, trace = _current()
        active = trace and det is engine._rx_state['vit']
        result = original_path(det, steps, with_states=with_states)
        if active and with_states:
            h, s0, s1, p0, u0, before, after = result
            trace.decisions.append(dict(interval=int(h['interval']), s0=int(s0), s1=int(s1),
                p0=int(p0), U0=int(u0), state_before=int(before), state_after=int(after)))
        return result
    TrellisDetector._path = path

    original_decode = DataRx.decode_frame
    @functools.wraps(original_decode)
    def decode(drx, symbols, *, trellis_states=None):
        engine = getattr(_context, 'engine', None)
        existing = getattr(engine, '_bounded_data_trace', None) if engine is not None else None
        if existing and existing.trellis_only:
            return original_decode(drx, symbols, trellis_states=trellis_states)
        engine, trace = _current()
        active = trace and drx is engine._rx_state['drx']
        if active:
            hist = list(engine._rx_state['hist'][-23:])
            before = dict(z=drx.z, m4=drx.m4, frame_pos=drx.frame_pos, conv=list(drx.conv.d))
        result = original_decode(drx, symbols, trellis_states=trellis_states)
        if active:
            role = 'answer' if trace.header['role'] == 'call' else 'call'
            trace.frames.append(dict(before=before, points=pairs(symbols),
                states=None if trellis_states is None else [int(s) for s in trellis_states],
                scrambled=list(result[0]), raw=descramble_stream(result[0], hist, role), worst=float(result[1])))
            if len(trace.frames) == trace.limit:
                trace.publish()
        return result
    DataRx.decode_frame = decode
    atexit.register(lambda: [trace.close() for trace in _traces])
