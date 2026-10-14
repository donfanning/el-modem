"""Causal real-PCM cancellation model; coefficients selected on guarded TRN."""
import numpy as np
import v34_rx as rx

FIT=(1400,2400)
HOLD=(3400,4400)
MAX_DELAY=1600
TAPS=32
RIDGE=1e-3
SHORTLIST=8

def pcm_filter(source, delay, taps):
    """Causal real PCM echo; float output avoids an extra quantizer/clipping."""
    if delay < 0 or np.iscomplexobj(taps):
        raise ValueError('expected causal real passband taps')
    out = np.zeros(len(source))
    if delay < len(source):
        out[delay:] = np.convolve(source, taps, mode='full')[:len(source)-delay]
    return out


def operator(z, centers, h, spl, rotation):
    offsets = (np.arange(len(h))-len(h)//2)*spl/2
    positions = np.asarray(centers)[:, None]+offsets
    return (rx.interp(z, positions.ravel()).reshape(positions.shape)@h)*rotation


def echo_operator(centers, h, spl):
    """Collapse equalizer and sinc weights once; integer FIR delays reuse them."""
    centers=np.asarray(centers)
    base=np.floor(centers).astype(np.int64)
    pos=centers[:,None]+(np.arange(len(h))-len(h)//2)*spl/2
    whole=np.floor(pos).astype(np.int64)
    m=np.arange(-5,7)
    relative=whole[:,:,None]-base[:,None,None]+m
    first=int(relative.min());last=int(relative.max())
    d=m-(pos-whole)[:,:,None]
    weights=np.sinc(d)*(.5+.5*np.cos(np.pi*d/6))*h[None,:,None]
    combined=np.zeros((len(centers),last-first+1),complex)
    rows=np.broadcast_to(np.arange(len(centers))[:,None,None],relative.shape)
    np.add.at(combined,(rows.ravel(),(relative-first).ravel()),weights.ravel())
    return base, np.arange(first,last+1), combined


def echo_features(z, centers, h, spl, rotation, fc, delay, count=TAPS, prepared=None):
    """Mixer shift identity: BB{s[n-d]}=BB{s}[n-d] exp(-i*w*d)."""
    base, offsets, weights=echo_operator(centers,h,spl) if prepared is None else prepared
    columns=[]
    for j in range(count):
        idx=np.clip(base[:,None]+offsets[None,:]-delay-j,0,len(z)-1)
        columns.append(np.einsum('ij,ij->i',z[idx],weights)*rotation
            *np.exp(-2j*np.pi*fc*(delay+j)/8000))
    return np.column_stack(columns)


def own_design(known, indices):
    # Short known-symbol ISI nuisance; no B1 reference or dense mapper points.
    return np.column_stack([known[indices+j] for j in range(-4, 5)])


def stack_real(a):
    return np.concatenate((a.real, a.imag), axis=0)


def fit_real_echo(nuisance, features, y):
    # Partial out own-channel structure before attributing energy to leakage.
    own = np.linalg.lstsq(nuisance, y, rcond=1e-10)[0]
    base = y-nuisance@own
    projected = features-nuisance@np.linalg.lstsq(nuisance, features, rcond=1e-10)[0]
    a, b = stack_real(projected), stack_real(base)
    scale = np.linalg.norm(a, ord=2)**2
    if scale <= 1e-20:
        return np.zeros(features.shape[1]), own, float(np.mean(abs(base)**2))
    taps = np.linalg.solve(a.T@a+RIDGE*scale*np.eye(a.shape[1]), a.T@b)
    # Training ranking uses the joint model; validation checks echo-only
    # subtraction against the SAME own-channel model, not a favorable refit.
    score = float(np.mean(abs(base-projected@taps)**2))
    return taps, own, score


def bootstrap_benefit(benefit):
    benefit = np.asarray(benefit)
    intervals = {}
    for width in (8, 16, 32):
        rng = np.random.default_rng(340170+width)
        # Preserve the seeded draw order and each draw's reduction order.
        # Bounded batches avoid 1200 Python/NumPy dispatches during startup,
        # without changing the resampling scheme or confidence policy.
        draws = np.empty(400)
        blocks = int(np.ceil(len(benefit)/width))
        offsets = np.arange(width)
        for first in range(0, 400, 32):
            count = min(32, 400-first)
            starts = rng.integers(0, len(benefit), (count, blocks))
            samples = ((starts[:, :, None]+offsets) % len(benefit)).reshape(count, -1)
            draws[first:first+count] = np.mean(benefit[samples[:, :len(benefit)]], axis=1)
        intervals[str(width)] = np.percentile(draws, [1, 99]).tolist()
    return intervals


def train_model(source, trace, known, fc, spl, h, offset, *, minimum_delay=0):
    if not isinstance(minimum_delay, (int, np.integer)) or not 0 <= minimum_delay <= MAX_DELAY:
        raise ValueError('no supported causal delay range')
    indices = trace['symbol'].astype(int)
    selected = ((indices >= FIT[0]) & (indices < FIT[1])) | ((indices >= HOLD[0]) & (indices < HOLD[1]))
    indices = indices[selected]
    centers, rotation, y = (trace[k][selected] for k in ('center', 'rotation', 'observed'))
    if indices.min()<4 or indices.max()+4>=len(known):
        raise ValueError('nuisance reference extends past known TRN')
    fit = (indices >= FIT[0]) & (indices < FIT[1])
    hold = (indices >= HOLD[0]) & (indices < HOLD[1])
    if fit.sum()!=FIT[1]-FIT[0] or hold.sum()!=HOLD[1]-HOLD[0]:
        raise ValueError('incomplete predeclared TRN blocks')
    support = len(h)*spl/4+46
    gap = float(centers[hold].min()-centers[fit].max()-MAX_DELAY-TAPS-2*support)
    if gap <= 0:
        raise ValueError('training/validation PCM support overlap')
    if (centers[fit].min()-MAX_DELAY-TAPS-support<0
            or centers[hold].max()+support-minimum_delay>=len(source)):
        raise ValueError('insufficient opposite PCM context')
    z = rx.to_baseband(source, fc, max(1700., .62*8000/spl))
    # Source and target mixers use different origins. Restore the target's
    # local phase origin exactly before applying its PLL rotation.
    rot = rotation*np.exp(2j*np.pi*fc*offset/8000)
    n = own_design(known, indices)
    own = np.linalg.lstsq(n[fit], y[fit], rcond=1e-10)[0]
    residual = y[fit]-n[fit]@own
    # Cheap delay screening is FIT-only. Its interpolation approximation is
    # never used for coefficients, validation or cancellation.
    offs = (np.arange(len(h))-len(h)//2)*spl/2
    m = np.arange(-int(np.ceil(abs(offs).max()))-6, int(np.ceil(abs(offs).max()))+7)
    kernel = np.zeros(len(m), complex)
    for off, tap in zip(offs, h):
        d = m-off
        kernel += tap*np.where((d > -6) & (d < 6), np.sinc(d)*(.5+.5*np.cos(np.pi*d/6)), 0)
    smooth = np.convolve(z, kernel[::-1], mode='same')
    scores = []
    first_delay = ((minimum_delay + 15) // 16) * 16
    for delay in range(first_delay, MAX_DELAY+1, 16):
        feature = rx.interp(smooth, centers[fit]-delay)*rot[fit]*np.exp(-2j*np.pi*fc*delay/8000)
        # Magnitude screens possible filtered echoes, including either sign.
        scores.append((float(abs(np.vdot(feature, residual))**2/max(np.vdot(feature, feature).real, 1e-30)), delay))
    # Coarse correlation names the vicinity of the impulse, not the first
    # causal tap. Bracket each peak so a peak just AFTER the true delay cannot
    # exclude that impulse from its forward-only FIR window.
    candidates = [(score, max(minimum_delay, delay-TAPS//2))
                  for score, delay in sorted(scores, reverse=True)[:SHORTLIST]]
    fitted = []
    prepared=echo_operator(centers[fit],h,spl)
    for _, delay in candidates:
        features = echo_features(z, centers[fit], h, spl, rot[fit], fc, delay,prepared=prepared)
        taps, own, score = fit_real_echo(n[fit], features, y[fit])
        fitted.append((score, delay, taps, own))
    score, delay, taps, own = min(fitted, key=lambda row: row[0])
    fh = echo_features(z, centers[hold], h, spl, rot[hold], fc, delay)
    base = y[hold]-n[hold]@own
    after = base-fh@taps
    before_mse, after_mse = float(np.mean(abs(base)**2)), float(np.mean(abs(after)**2))
    intervals = bootstrap_benefit(abs(base)**2-abs(after)**2)
    improvement = (before_mse-after_mse)/max(before_mse, 1e-30)
    accepted = bool(before_mse>1e-14 and improvement>=.1 and all(v[0]>0 for v in intervals.values()))
    return dict(delay_samples=int(delay), taps=taps.tolist(), accepted=accepted,
        fit_symbols=list(FIT), validation_symbols=list(HOLD), fit_mse=score,
        validation_baseline_mse=before_mse, validation_cancelled_mse=after_mse,
        improvement_fraction=improvement, benefit_block_bootstrap_1_99=intervals,
        pcm_support_gap_samples=gap, support_half_samples=support,
        activation_sample=int(np.ceil(centers[hold].max()+support))+1,
        lag_range_samples=[int(minimum_delay), MAX_DELAY], real_fir_taps=TAPS, ridge=RIDGE,
        screening_shortlist=[d for _, d in candidates], uses_b1_labels=False,
        coefficient_units='opposite PCM units to receive PCM units; real amplitude ratio',
        selection='FIT-only screening and coefficients; one fixed candidate gated on later TRN')
