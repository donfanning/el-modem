"""Bound unstable retrain loops; renew only after qualified LAPM delivery.

This is a session retry guard, not a line-quality or rate-selection rule.
Ten seconds, 1024 delivered bytes and a one-second delivery gap are engineering guards,
not V.34 constants. Raw/unprotected delivery cannot replenish the budget.
"""


class RetrainBudget:
    def __init__(self, limit=2, stable_seconds=10., minimum_bytes=1024):
        self.limit = limit
        self.stable_seconds = stable_seconds
        self.minimum_bytes = minimum_bytes
        self.used = self.total = self.rearms = 0
        self.samples = self.bytes = self.delivery_gap_samples = 0
        self._core = None
        self._bad = None

    def take(self):
        if self.used >= self.limit:
            return False
        self.used += 1
        self.total += 1
        self.samples = self.bytes = self.delivery_gap_samples = 0
        return True

    def observe(self, *, core, samples, fs, healthy, delivered, bad_frames):
        if core is not self._core:
            self._core = core
            self.samples = self.bytes = self.delivery_gap_samples = 0
            self._bad = bad_frames
        changed = bad_frames != self._bad
        self._bad = bad_frames
        if not healthy or changed:
            self.samples = self.bytes = self.delivery_gap_samples = 0
            return False
        if not self.used:
            return False
        self.delivery_gap_samples = (0 if delivered else min(fs + 1, self.delivery_gap_samples + samples))
        if self.delivery_gap_samples > fs:
            self.samples = self.bytes = 0
            return False
        self.samples += samples
        self.bytes += delivered
        if self.samples >= self.stable_seconds * fs and self.bytes >= self.minimum_bytes:
            self.used = 0
            self.rearms += 1
            self.samples = self.bytes = self.delivery_gap_samples = 0
            return True
        return False

    def report(self):
        return dict(limit=self.limit, used=self.used, total=self.total, rearms=self.rearms,
                    stable_seconds_required=self.stable_seconds,
                    minimum_lapm_bytes=self.minimum_bytes, maximum_lapm_delivery_gap_seconds=1.,
                    delivery_gap_samples=self.delivery_gap_samples,
                    stable_receive_samples=self.samples, validated_bytes=self.bytes)


def observe(modem, core, samples, delivered, *, concealed=False):
    """Use actual received samples and delivered LAPM bytes, never TX time."""
    q = getattr(core, 'rx_qualification', None) or {}
    eq = getattr(getattr(core, '_p4rcv', None), 'eq', None)
    guard = getattr(eq, '_input_guard', None)
    link = getattr(modem, '_v42', None)
    exchange = getattr(getattr(core, '_connected_rate', None), 'exchange', None)
    bad = (getattr(link, 'stats', None) or {}).get('hdlc_bad')
    healthy = (modem.connected and not modem._retraining and not concealed and exchange is None
               and core.stage == 'data' and getattr(core, '_rx_state', None) is not None
               and q.get('standard_qualification') is True and q.get('bit_errors') == 0
               and not getattr(core, 'lost_too_long', False)
               and not getattr(core, '_freeze_held', False)
               and not getattr(guard, 'holding', False)
               and modem.v42_mode == 'lapm' and getattr(link, 'state', None) == 'lapm'
               and bad is not None)
    if modem._retrain_budget.observe(core=getattr(core, '_rx_state', None), samples=samples, fs=modem.fs,
                                    healthy=healthy, delivered=len(delivered), bad_frames=bad):
        modem._ev('retrain budget renewed after qualified data and LAPM delivery')
