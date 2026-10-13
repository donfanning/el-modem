"""Symbol-domain V.34 section 11.6 exchange; not yet attached to the live DCE.

The frontend must detect S/S-bar, supply normalized four-point rotation indices
after S-bar, and acknowledge units only after their final symbol is transmitted.
It must qualify the new received B1 before calling receive_b1(). No application
or link-layer queues are owned or reset here.
"""
from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import dataclass
import math

import numpy as np

import v34_data
import v34_mp
import v34_negotiation
import v34_tx as tx


class MPScanner:
    """Bounded streaming differential decoder/descrambler and CRC MP scanner.

    TRN need not have a known length. Treating TRN as differential data corrupts
    its decoded bits and the first 23 MP bits; repeated MP self-synchronizes.
    Global quarter-turn ambiguity also washes out after 23 bits. E detection
    requires a valid MP and the caller's explicit acknowledgement permission.
    Offsets count bits from the first supplied post-S-bar symbol.
    """
    def __init__(self, peer_role):
        if peer_role not in ('call', 'answer'):
            raise ValueError('invalid peer role')
        self.tap = 18 if peer_role == 'call' else 5
        self.history = deque([0] * 23, maxlen=23)
        self.buffer = deque()
        self.previous = 0
        self.bits = 0
        self.start = 0
        self.have_mp = False
        self.e_end = None

    def feed_rotation(self, rotation, *, allow_e=False):
        if rotation not in (0, 1, 2, 3):
            raise ValueError('rotation must be 0..3')
        if self.e_end is not None:
            return []
        delta = (rotation - self.previous) % 4
        self.previous = rotation
        events = []
        for raw in (delta & 1, delta >> 1):
            bit = raw ^ self.history[self.tap - 1] ^ self.history[22]
            self.history.appendleft(raw)
            self.buffer.append(bit)
            self.bits += 1
            events.extend(self._scan(allow_e))
        return events

    def _drop(self, n):
        for _ in range(n):
            self.buffer.popleft()
        self.start += n

    def _scan(self, allow_e):
        events = []
        while len(self.buffer) >= 20:
            b = list(self.buffer)
            if b[:18] == [1] * 17 + [0]:
                n = 188 if b[18] else 88
                if len(b) < n:
                    break
                mp = v34_mp.parse_mp(b[:n])
                if mp is not None:
                    events.append(dict(kind='mp', start=self.start,
                                       end=self.start + n, mp=mp))
                    self.have_mp = True
                    self._drop(n)
                    continue
            if allow_e and self.have_mp and b[:20] == [1] * 20:
                self.e_end = self.start + 20
                events.append(dict(kind='e', start=self.start, end=self.e_end))
                self._drop(20)
                break
            self._drop(1)
        return events


@dataclass(frozen=True)
class TransmitUnit:
    kind: str
    symbols: np.ndarray


class RateExchange:
    """Unit-level duplex exchange with explicit physical completion barriers.

    next_unit() reserves whole units for pulse-shaper lookahead; transmitted()
    commits them in wire order. Queued MP' does not satisfy the acknowledgement
    barrier prematurely. At most 32 units can be outstanding. Time is
    supplied by the frontend in seconds on its PCM clock, not by wall time.
    The frontend still owns S detection, sampling, B1 qualification and fallback.
    """
    def __init__(self, local, previous_peer, *, role, initiating, tx_baud,
                 rx_baud, tx_carrier=0., rx_carrier=0., rtd=0., cme=False,
                 trn_symbols=1024):
        if role not in ('call', 'answer') or not math.isfinite(rtd) or rtd < 0:
            raise ValueError('invalid role or round trip delay')
        if not isinstance(trn_symbols, int) or not 0 < trn_symbols <= (2 + (rtd if initiating else 0)) * tx_baud:
            raise ValueError('transmitted TRN must be positive and within section 11.6 limit')
        self.role, self.initiating = role, bool(initiating)
        self.local = deepcopy(local)
        self.previous_peer = deepcopy(previous_peer)
        self.params = (tx_baud, rx_baud, tx_carrier, rx_carrier)
        # Validate supported offers before putting anything on the wire.
        v34_negotiation.resolve(self.local, self.previous_peer, role, *self.params)
        self.timeout = 30. if cme else 2.5 + (2 if initiating else 3) * rtd
        self.trn, self.sc = tx.trn_symbols(trn_symbols, role, constellation=4)
        self.z = tx.trn_last_rotation(self.trn, 4)
        self.scanner = MPScanner('answer' if role == 'call' else 'call')
        self.peer = None
        self.negotiated = None
        self.peer_prime = self.peer_e = self.prime_sent = False
        self.tx_ready = False
        # An initiator continues receiving old data until the peer sends S.
        # A responder is constructed only after S has already been detected.
        self.rx_clamped = not self.initiating
        self.s_received = not self.initiating
        self.failed = None
        self.transition_at = None
        self.stage = 's'
        self.pending = deque()
        self.data_tx = self.data_sc = None

    @staticmethod
    def _coding(mp):
        return {k: v for k, v in mp.items() if k not in ('ack', 'type')}

    def feed_rotation(self, rotation):
        if self.failed:
            return []
        if not self.s_received:
            raise RuntimeError('post-S-bar symbols supplied before detecting peer S')
        events = self.scanner.feed_rotation(rotation, allow_e=self.prime_sent)
        for event in events:
            if event['kind'] == 'mp':
                peer = deepcopy(event['mp'])
                # Type 0 leaves the previous precoding coefficients unaffected.
                if peer['type'] == 0:
                    peer['precoder'] = deepcopy(self.previous_peer.get('precoder', [0] * 6))
                if self.peer is not None and self._coding(peer) != self._coding(self.peer):
                    self.failed = 'peer changed coding fields during exchange'
                    break
                try:
                    negotiated = v34_negotiation.resolve(self.local, peer, self.role, *self.params)
                except ValueError as exc:
                    self.failed = str(exc)
                    break
                self.peer, self.negotiated = peer, negotiated
                self.peer_prime |= bool(peer['ack'])
            else:
                self.peer_e = True
        return events

    def receive_s(self):
        self.s_received = True
        self.rx_clamped = True

    def next_unit(self):
        if self.failed or self.stage == 'data':
            return None
        if len(self.pending) >= 32:
            raise RuntimeError('transmit lookahead limit exceeded')
        if self.stage == 's':
            unit = TransmitUnit('s', tx.s_signal(128))
        elif self.stage == 'sbar':
            unit = TransmitUnit('sbar', tx.s_signal(16, True))
        elif self.stage == 'trn':
            unit = TransmitUnit('trn', self.trn.copy())
        elif self.stage == 'b1':
            p = self.negotiated.tx
            self.data_tx = v34_data.DataTx(p.rate, p.baud, expanded=p.shaping,
                precoder=v34_data.precoder_from_mp(p.precoder), nonlinear=p.nonlinear, trellis=p.trellis)
            self.data_sc = tx.Scrambler(self.role)
            bits = [self.data_sc.step(1) for _ in range(self.data_tx.bits_per_data_frame)]
            symbols = np.asarray(self.data_tx.data_frame(bits)) * math.sqrt(2 / self.data_tx.avg_power)
            unit = TransmitUnit('b1', symbols)
        else:
            if self.prime_sent and (self.peer_prime or self.peer_e):
                kind, bits = 'e', [1] * 20
            else:
                kind = 'mp_prime' if self.peer is not None and self.stage != 'first_mp' else 'mp'
                bits = v34_mp.build_mp(1, ack=int(kind == 'mp_prime'), **self.local)
            symbols, self.z = tx.bit_symbols(bits, self.role, self.sc, self.z, constellation=4)
            unit = TransmitUnit(kind, symbols)
        self.pending.append(unit)
        # Reservation advances generation, never the physical acknowledgement
        # or Circuit 106. Extra MP' may be reserved while the first is in flight.
        if unit.kind == 's':
            self.stage = 'sbar'
        elif unit.kind == 'sbar':
            self.stage = 'trn'
        elif unit.kind == 'trn':
            self.stage = 'first_mp'
        elif unit.kind in ('mp', 'mp_prime'):
            self.stage = 'mp'
        elif unit.kind == 'e':
            self.stage = 'b1'
        elif unit.kind == 'b1':
            self.stage = 'data'
        return unit

    def transmitted(self, unit):
        if not self.pending or unit is not self.pending[0]:
            raise RuntimeError('wrong transmit completion')
        self.pending.popleft()
        if unit.kind in ('mp', 'mp_prime'):
            self.prime_sent |= unit.kind == 'mp_prime'
        elif unit.kind == 'b1':
            self.tx_ready = True

    def mark_sbar_start(self, seconds):
        if self.transition_at is not None:
            raise RuntimeError('S/S-bar transition already marked')
        self.transition_at = float(seconds)

    def check_timeout(self, seconds):
        if not self.peer_e and self.transition_at is not None and seconds >= self.transition_at + self.timeout:
            self.failed = 'E timeout; full retrain required'
        return self.failed

    def receive_b1(self, *, qualified):
        if self.failed:
            raise RuntimeError('failed exchange requires full retrain')
        if not self.peer_e or self.negotiated is None:
            raise RuntimeError('B1 before negotiated E')
        if not qualified:
            self.failed = 'B1 qualification failed; full retrain required'
        else:
            self.rx_clamped = False


class RateQamBridge:
    """Drain queued data through the existing pulse shaper before sending S.

    All subsequent reads must go through this bridge. Generation uses the same
    constellation power, carrier, pulse history and pre-emphasis state; temporary
    four-point units are rescaled instead of changing the shaper's normalization.
    ``data_fill(qam, exchange)`` must use exchange.data_tx/data_sc, which already
    contain B1's encoder history, and preserve the existing application queues.
    This is a transmit adapter, not receive-side or live-DCE integration.
    """
    def __init__(self, qam, exchange, data_fill):
        if qam.baud != exchange.params[0] or qam.fc != exchange.params[2]:
            raise ValueError('connected rate change must retain baud and carrier')
        self.qam, self.exchange, self.data_fill = qam, exchange, data_fill
        self.markers = deque()
        self.sbar_sample = None
        self.scale = math.sqrt(qam.power / 2)
        self.emitted = qam.rendered_chunks * qam.samples_per_chunk - len(qam.buf)
        qam.fill = self._fill

    def _fill(self, qam):
        unit = self.exchange.next_unit()
        if unit is None:
            if self.exchange.failed:
                raise RuntimeError(self.exchange.failed)
            before = len(qam.queue)
            self.data_fill(qam, self.exchange)
            if len(qam.queue) <= before:
                raise RuntimeError('data fill supplied no symbols')
            return
        start = qam.rendered_symbols + len(qam.queue)
        qam.push(unit.symbols * self.scale)
        end = qam.rendered_symbols + len(qam.queue)
        self.markers.append((end * tx.FS / qam.baud, unit))
        if unit.kind == 'sbar':
            self.sbar_sample = start * tx.FS / qam.baud

    def read(self, n):
        # RX will hand the failed exchange to the wrapper's retrain owner.
        # Until then stop old data without crashing the real-time TX thread.
        if self.exchange.failed:
            return np.zeros(n)
        out = self.qam.read(n)
        self.emitted += len(out)
        if self.sbar_sample is not None and self.exchange.transition_at is None and self.emitted >= self.sbar_sample:
            self.exchange.mark_sbar_start(self.sbar_sample / tx.FS)
        while self.markers and self.emitted >= self.markers[0][0]:
            _, unit = self.markers.popleft()
            self.exchange.transmitted(unit)
        self.exchange.check_timeout(self.emitted / tx.FS)
        return out
