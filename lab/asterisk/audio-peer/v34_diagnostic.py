"""Opt-in observation of the live receiver; never selects or changes a DSP model.

Collection happens inside the production call on its receive thread. Copies of
the interpolation source are taken when the grid is built, because subsequent
baseband updates revise its trailing edge. Qualification is run exactly once.
"""
from contextlib import contextmanager
import copy
from dataclasses import asdict, is_dataclass
import json
import math
import os
from pathlib import Path
import threading
import time

import numpy as np

_active = threading.local()
_install_lock = threading.Lock()
_installed = False
PHASES = {'phase3': 1, 'phase4': 2, 'b1': 3, 'initial-data': 4, 'receiver-transition': 5}
MAX_ROWS = 32768
INITIAL_DATA_SYMBOLS = 1024
MAX_EPOCHS = 8


def owned_support(source, start, width):
    """Snapshot contiguous interpolation support, including clipped endpoints."""
    if 0 <= start and start + width <= len(source):
        return source[start:start+width].copy()
    # Advanced indexing already owns its result. Keep the original clipping
    # operator when support extends beyond either end of the source.
    return source[np.clip(np.arange(start, start+width), 0, len(source)-1)]


def json_value(value):
    if is_dataclass(value):
        return json_value(asdict(value))
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, complex):
        return [value.real, value.imag]
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _current():
    return getattr(_active, 'capture', None)


def _install_hooks():
    """Install once; thread-local ownership keeps simultaneous calls separate."""
    global _installed
    with _install_lock:
        if _installed:
            return
        import v34_rx as rx
        import v34_receive as receive
        train, grid, step = rx.Phase3Rx.train_known, rx.SymbolEq._build_grid, rx.SymbolEq.step
        phase3, phase4 = rx.Phase3Receiver.feed, rx.Phase4Receiver.feed
        initialize, qualify = receive.initialize, receive.Receiver.qualify
        refine = receive.refine_taps

        def refine_hook(*args, **kwargs):
            cap = _current()
            if cap is None:
                return refine(*args, **kwargs)
            with cap.measure('b1_refine_taps'):
                return refine(*args, **kwargs)

        def train_hook(R, known, k_start, *args, **kwargs):
            cap = _current()
            if cap is None:
                return train(R, known, k_start, *args, **kwargs)
            with cap.measure('phase3_train'):
                score = train(R, known, k_start, *args, **kwargs)
            with cap.measure('capture_training_rows'):
                cap.training(R, known, k_start, score)
            return score

        def grid_hook(eq, *args, **kwargs):
            cap = _current()
            if cap is not None and cap.requires_exact_windows():
                # Every retained row must use the exact interpolation operator.
                # After the bounded data capture, honor the requested live grid.
                args, kwargs = (), {}
            result = grid(eq, *args, **kwargs)
            if cap is not None and not eq._g_traj:
                # Store only the source support used by this interpolation batch.
                half = eq.R.spl / 2
                center = eq.R.lag + (eq.R.LEAD + eq._g_k0 + eq.R.delay) * eq.R.spl + eq._g_dt
                start = center + (-eq.MARGIN - eq.span) * half
                end = start + (len(eq._grid) - 1) * half
                lo = int(np.floor(start)) - 8
                hi = int(np.ceil(end)) + 9
                origin = getattr(eq.R, '_bb_origin', 0)
                positions=center+(np.arange(len(eq._grid))-eq.MARGIN-eq.span)*half
                eq._diagnostic_grid_source = (lo, owned_support(eq.R._z,lo-origin,hi-lo),positions)
            return result

        def step_hook(eq, desired=None, decide=None):
            cap = _current()
            # Once the bounded initial-data capture is full, before_symbol
            # returns None and trajectory mode is preserved. Avoid its target
            # list and try/finally on every remaining production symbol.
            if cap is None or not cap.requires_exact_windows():
                return step(eq, desired=desired, decide=decide)
            requested_trajectory = eq._traj
            if cap.requires_exact_windows():
                eq._traj = False
            try:
                window = eq.window(eq.k)
                row = cap.before_symbol(eq, desired is not None, window=window)
                target = [desired]
                callback = decide
                if row is not None and desired is None and decide is not None:
                    def callback(y):
                        target[0] = decide(y)
                        return target[0]
                y = step(eq, desired=desired, decide=callback, _window=window)
                if row is not None:
                    cap.after_symbol(eq, row, y, target[0])
                return y
            finally:
                eq._traj = requested_trajectory

        def phase3_hook(receiver, *args, **kwargs):
            cap = _current()
            if cap is None:
                return phase3(receiver, *args, **kwargs)
            with cap.measure('phase3_feed'), cap.phase('phase3'):
                return phase3(receiver, *args, **kwargs)

        def phase4_hook(receiver, *args, **kwargs):
            cap = _current()
            if cap is None:
                return phase4(receiver, *args, **kwargs)
            with cap.measure('phase4_feed'), cap.phase('phase4'):
                result = phase4(receiver, *args, **kwargs)
            cap.phase4 = dict(k_trn=receiver.k_trn, trn_end=receiver.trn_end,
                             b1_symbol=receiver.b1_symbol, e_span=receiver.e_span,
                             mp_spans=receiver.mp_spans, constellation=16 if receiver.c16 else 4)
            return result

        def initialize_hook(engine, eq, *args, **kwargs):
            cap = _current()
            if cap is None:
                return initialize(engine, eq, *args, **kwargs)
            cap.adopt_core(engine)
            with cap.measure('b1_initialize'), cap.phase('b1'):
                try:
                    return initialize(engine, eq, *args, **kwargs)
                finally:
                    inputs = getattr(engine, '_b1_inputs', None)
                    if inputs is not None:
                        cap.b1_inputs = {k: v.copy() if isinstance(v, np.ndarray) else v
                                         for k, v in inputs.items()}

        def qualify_hook(receiver, observed, *, trace=False):
            cap = _current()
            if cap is None:
                return qualify(receiver, observed, trace=trace)
            # Extra trace storage only; no second feed, calibration or flush.
            try:
                with cap.measure('b1_qualify'):
                    g, ref = qualify(receiver, observed, trace=True)
                cap.qualification_role = receiver.role
                cap.reference = dict(observed=np.asarray(observed).copy(),
                    projected_reference=np.asarray(ref).copy(),
                    expected_mapper=receiver.expected_mapper.copy(),
                    expected_bits=np.asarray(receiver.expected, dtype=np.uint8),
                    decoded_bits=np.asarray(receiver.bits, dtype=np.uint8),
                    gain=np.asarray(g), energy=np.asarray(receiver.energy))
                return g, ref
            finally:
                detector = getattr(receiver, 'b1_detector', None)
                if detector is not None:
                    with cap.measure('capture_b1_trace'):
                        # Own the complete B1 snapshot now: streaming can pop
                        # decisions and mutate/reuse trellis buffers afterwards.
                        # Keep arrays as arrays until close(), outside receive.
                        cap.detector_trace = copy.deepcopy(dict(trace=detector.trace,
                            snapshots=detector.snapshots, decided=detector.decided))

        rx.Phase3Rx.train_known = train_hook
        rx.SymbolEq._build_grid, rx.SymbolEq.step = grid_hook, step_hook
        rx.Phase3Receiver.feed, rx.Phase4Receiver.feed = phase3_hook, phase4_hook
        receive.initialize, receive.Receiver.qualify = initialize_hook, qualify_hook
        receive.refine_taps = refine_hook
        _installed = True


class DspCapture:
    def __init__(self, prefix, engine):
        self.training_stride = int(os.environ.get('EL_MODEM_DSP_TRAINING_STRIDE', '1'))
        if self.training_stride not in (1, 2, 4, 8):
            raise ValueError('DSP training stride must be 1, 2, 4 or 8')
        self._training_calls = {}
        self.training_rows_skipped = 0
        self.prefix = Path(prefix)
        self.prefix.parent.mkdir(parents=True, exist_ok=True)
        # Fail before starting a receiver instead of silently losing its evidence.
        self.ready_path = Path(str(self.prefix) + '.ready.json')
        self.ready_path.write_text(json.dumps(dict(enabled=True, pid=os.getpid(),
            started_monotonic=time.monotonic(), max_rows=MAX_ROWS, training_stride=self.training_stride)) + '\n')
        self.engine = engine
        self.core = getattr(engine, 'caller', None) or getattr(engine, 'answerer', None)
        if self.core is None:
            raise ValueError('DSP capture requires the original V.34 engine')
        self.core._receive_trace = True
        self.groups = {}
        self.fits = []
        self.received = self.row_count = self.data_rows = 0
        self.errors = []
        self.truncated = self.closed = False
        self.stage = 'initial-data'
        self.phase4 = {}
        self.b1_inputs = self.reference = self.detector_trace = None
        self.qualification_role = None
        self.epochs = []
        self.epoch = -1
        self._epoch_core = None
        self.adopt_core(self.core)
        _install_hooks()

    def _protocol(self, names):
        result = {name:getattr(self.core,name,None) for name in names}
        if getattr(self.core, '_rate_local_mp', None) is not None:
            result['original_local_mp'] = result['_local_mp']
            result['_local_mp'] = self.core._rate_local_mp
        return result

    def adopt_core(self, core):
        """A new retrain core owns a new diagnostic epoch, including B1."""
        generation = getattr(core, '_rate_capture_epoch', 0)
        if core is self._epoch_core and generation == getattr(self, '_epoch_generation', None):
            return
        if self._epoch_core is not None and self.epoch < MAX_EPOCHS:
            self._finish_epoch()
        self.core = self._epoch_core = core
        self._epoch_generation = generation
        self.epoch += 1
        core._receive_trace = True
        self.data_rows = 0
        self.phase4 = {}
        self.b1_inputs = self.reference = self.detector_trace = None
        self.qualification_role = None
        if self.epoch >= MAX_EPOCHS:
            self.truncated = True
            return
        self.epochs.append(dict(epoch=self.epoch, started_monotonic=time.monotonic(),
            capture_received_samples=self.received, core_rx_samples=getattr(core,'rx_t',0),
            initial_data_symbols=0, timings={}))

    def _finish_epoch(self):
        item = self.epochs[self.epoch]
        item.update(ended_monotonic=time.monotonic(), initial_data_symbols=self.data_rows,
            last_core_stage=getattr(self.core,'stage',None),
            qualification=getattr(self.core,'rx_qualification',None),
            phase4=dict(self.phase4), qualification_remote_role=self.qualification_role,
            negotiated=json_value(getattr(self.core,'negotiated',None)),
            protocol=json_value(self._protocol(('mp_ans','mp_call','_local_mp'))))

    @contextmanager
    def measure(self, name):
        # Only feed/stage boundaries, never per-symbol stack walks or file I/O.
        epoch = self.epoch
        core = self.core
        started = time.monotonic()
        before = getattr(core,'rx_t',None)
        progress_before = self.progress()
        failed = True
        try:
            yield
            failed = False
        finally:
            ended = time.monotonic()
            if epoch < MAX_EPOCHS:
                timings = self.epochs[epoch]['timings']
                record = timings.setdefault(name,dict(count=0,total_seconds=0.,maximum_seconds=0.))
                duration = ended-started
                record['count'] += 1
                record['total_seconds'] += duration
                if 'first' not in record or duration > record['maximum_seconds']:
                    event = dict(started_monotonic=started,ended_monotonic=ended,
                        core_rx_before=before,core_rx_after=getattr(core,'rx_t',None),
                        capture_received_samples=self.received,core_stage=getattr(core,'stage',None),
                        failed=failed,progress_before=progress_before,progress_after=self.progress())
                    if 'first' not in record:record['first'] = event
                    record['worst'] = event
                    record['maximum_seconds'] = duration

    def progress(self):
        p4 = getattr(self.core,'_p4rcv',None)
        eq = getattr(p4,'eq',None)
        if eq is None:
            return None
        # Same absolute next-symbol center used by production PCM trimming.
        center = (self.source_offset(eq.R)+eq.R.lag
                  +(eq.R.LEAD+eq.k+eq.R.delay)*eq.R.spl+eq.dt)
        available = eq.available()
        b1 = getattr(p4,'b1_symbol',None)
        return dict(next_symbol=eq.k,next_symbol_pcm_center=center,
            received_minus_next_symbol_center_samples=self.core.rx_t-center,
            available_symbol=available,trn_end=getattr(p4,'trn_end',None),b1_symbol=b1,
            available_minus_b1_symbols=None if b1 is None else available-b1)

    @contextmanager
    def phase(self, name):
        previous = self.stage
        self.stage = name
        try:
            yield
        finally:
            self.stage = previous

    def feed(self, pcm, receiver):
        core = getattr(self.engine,'caller',None) or getattr(self.engine,'answerer',None)
        self.adopt_core(core)
        self.received += len(pcm)//2
        previous = _current()
        _active.capture = self
        try:
            return receiver.feed_pcm(pcm)
        finally:
            _active.capture = previous
            # The wrapper can replace its caller inside feed_pcm when retrain
            # is recognized. Arm capture before the next training feed arrives.
            core = getattr(self.engine,'caller',None) or getattr(self.engine,'answerer',None)
            self.adopt_core(core)

    def source_offset(self, R):
        if R is getattr(self.core, '_p4R', None):
            return getattr(self.core, '_p4_x0', 0)
        phase3=getattr(self.core,'_p3',None)
        if isinstance(phase3,dict):
            return phase3.get('x0',0)
        return getattr(self.core,'info1c_sent_t',0)

    def training(self, R, known, k_start, score):
        if len(self.fits) >= 16:
            self.truncated = True
            return
        self.fits.append(dict(epoch=self.epoch, role=R.role, start=k_start, nmse=score,
            baud=R.baud, carrier=R.fc, span=R._span,
            matrix=R._design(k_start, len(known)).copy(),
            known=np.asarray(known).copy(), taps=R.taps.copy()))

    def requires_exact_windows(self):
        return (self.stage != 'initial-data'
                or getattr(self.core, '_rx_state', None) is None
                or self.data_rows < INITIAL_DATA_SYMBOLS)

    def before_symbol(self, eq, supervised, *, window=None):
        stage=self.stage
        if stage=='initial-data' and getattr(self.core,'_rx_state',None) is None:
            stage='receiver-transition'
        if stage == 'initial-data':
            if self.data_rows >= INITIAL_DATA_SYMBOLS:
                return None
            self.data_rows += 1
        if stage in ('phase3', 'phase4'):
            key = (id(eq), stage)
            ordinal = self._training_calls.get(key, 0)
            self._training_calls[key] = ordinal+1
            if ordinal % self.training_stride:
                self.training_rows_skipped += 1
                return None
        if self.row_count >= MAX_ROWS:
            self.truncated = True
            return None
        self.row_count += 1
        w = (eq.window(eq.k) if window is None else window).copy()
        center = eq.R.lag + (eq.R.LEAD+eq.k+eq.R.delay)*eq.R.spl + eq.dt
        positions = center + (np.arange(eq.L)-eq.span)*eq.R.spl/2
        source_origin = getattr(eq.R, '_bb_origin', 0)
        source = eq.R._z
        direct = (abs(eq.dt - (eq._g_dt+eq._g_shift*eq.R.spl/2)) > 1e-12
                  or eq.R.spl != eq._g_spl)
        if not direct:
            source_origin, source, grid_positions = eq._diagnostic_grid_source
            first=2*(eq.k-eq._g_k0)+eq._g_shift+eq.MARGIN
            positions=grid_positions[first:first+eq.L]
        lo = int(np.floor(positions.min())) - 8
        width = int(np.ceil((eq.L-1)*eq.R.spl/2)) + 19
        support = owned_support(source,lo-source_origin,width)
        group = self.groups.setdefault(id(eq), dict(eq=eq, epoch=self.epoch, role=eq.R.role,
            baud=eq.R.baud, carrier=eq.R.fc, span=eq.span, rows=[]))
        rotation = complex(math.cos(-eq.phase), math.sin(-eq.phase))
        return group, dict(phase=PHASES[stage], symbol=eq.k,
            position=center, rx_pcm_position=center+self.source_offset(eq.R)+self.received-self.core.rx_t,
            timing=[eq.dt, eq.vel, eq.phase, eq.freq, eq.R.spl],
            cache=[eq._g_dt, eq._g_shift, eq._g_spl, eq._g_k0, getattr(eq, '_g_origin', 0)],
            adaptation=[eq.adapt, eq.adapt_nlms, supervised],
            window=w*rotation, taps=eq.h.copy(), source=support,
            positions=positions-lo, source_origin=lo, rotation=rotation,
            execution=self.row_count-1)

    def after_symbol(self, eq, item, observed, target):
        group, row = item
        row.update(observed=observed, target=target,
            timing_after=[eq.dt, eq.vel, eq.phase, eq.freq, eq.R.spl])
        group['rows'].append(row)

    def report(self):
        return dict(enabled=True, prefix=str(self.prefix), rows=self.row_count,
                    training_stride=self.training_stride,
                    training_rows_skipped=self.training_rows_skipped,
                    sampling_scope='phase3/phase4 per equalizer execution; fit matrices, B1 and initial data retained in full',
                    initial_data_symbols=self.data_rows, truncated=self.truncated,
                    errors=self.errors, closed=self.closed,
                    current_epoch=self.epoch, epochs=json_value(self.epochs),
                    timing_scope='inclusive stage wall time; nested totals must not be added',
                    evidence_class='instrumented live receiver; DSP parameters unchanged')

    def close(self):
        if self.closed:
            return
        if self.epoch < MAX_EPOCHS:self._finish_epoch()
        arrays = {}
        groups = []
        for i, group in enumerate(self.groups.values()):
            rows = group['rows']
            groups.append({k: v for k, v in group.items() if k not in ('eq', 'rows')} | dict(id=i, rows=len(rows)))
            for key in rows[0] if rows else []:
                arrays[f'eq{i}_{key}'] = np.asarray([r[key] for r in rows])
        fits = []
        for i, fit in enumerate(self.fits):
            fits.append({k: v for k, v in fit.items() if not isinstance(v, np.ndarray)})
            for k, v in fit.items():
                if isinstance(v, np.ndarray):
                    arrays[f'fit{i}_{k}'] = v
        b1_meta = {}
        for k, v in (self.b1_inputs or {}).items():
            if isinstance(v, np.ndarray):
                arrays['b1_'+k] = v
            else:
                b1_meta[k] = v
        for k, v in (self.reference or {}).items():
            arrays['qualified_'+k] = v
        np.savez(str(self.prefix)+'.npz', **arrays)
        if self.detector_trace is not None:
            Path(str(self.prefix)+'.detector.json').write_text(json.dumps(json_value(self.detector_trace))+'\n')
        self.closed = True
        info = self.report() | dict(schema=1, groups=groups, training_fits=fits,
            phases=PHASES, phase4=self.phase4, b1=b1_meta,
            domains=dict(window='phase-rotated complex baseband interpolation row',
                observed='equalizer output before B1 normalization',
                qualified_projected_reference='forward waveform after nonlinear projection',
                qualified_expected_mapper='mapper constellation coordinates'),
            sampling_source='copied at interpolation batch construction; direct windows use live source',
            initial_data_capture_limit=INITIAL_DATA_SYMBOLS,
            interpolation_policy='exact retained rows; requested trajectory grid resumes after data limit',
            negotiated=getattr(self.core,'negotiated',None),
            qualification=getattr(self.core,'rx_qualification',None),
            qualification_remote_role=self.qualification_role,
            failed_reason=getattr(self.engine,'failed_reason',''),
            last_core_stage=getattr(self.core,'stage',None),
            events=getattr(self.engine,'events',[]),
            protocol=self._protocol(('info0_ans','info0_call','info1a','info1c','mp_ans','mp_call','_local_mp')))
        Path(str(self.prefix)+'.json').write_text(json.dumps(json_value(info),indent=2)+'\n')
        self.groups.clear()
        self.fits.clear()


def configured(engine):
    prefix = os.environ.get('EL_MODEM_DSP_CAPTURE')
    return DspCapture(prefix, engine) if prefix else None
