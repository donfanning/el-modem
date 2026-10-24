"""Duplex MP resolution, V.34 (02/98) 11.4.1.1.3 / 11.4.1.2.3.

An MP's trellis/nonlinear/shaping/precoder fields describe what its sender's
RECEIVER requests from the remote transmitter, not its own transmitted format.
"""
from dataclasses import dataclass
import copy

from v34_data import frame_params, mapping_params


@dataclass(frozen=True)
class DirectionParameters:
    rate: int
    baud: float
    carrier: float
    trellis: int = 0
    shaping: bool = False
    nonlinear: bool = False
    precoder: tuple[int, ...] = (0, 0, 0, 0, 0, 0)


@dataclass(frozen=True)
class NegotiatedParameters:
    tx: DirectionParameters
    rx: DirectionParameters
    asymmetric: bool


def supported(rate, baud, shaping=False):
    try:
        fp = frame_params(rate, baud)
        mp = mapping_params(fp['b'], shaping)
        return fp['b'] >= 8 and mp['L'] <= 1664
    except (KeyError, ValueError):
        return False


def directional_offer(original, role, rates):
    """Temporary directional limits; never enlarge or mutate the initial MP."""
    if role not in ('call','answer'):raise ValueError('unknown V.34 role')
    tx_rate,rx_rate=rates
    c2a,a2c=((tx_rate,rx_rate) if role=='call' else (rx_rate,tx_rate))
    local=copy.deepcopy(original)
    for key,rate in (('max_c2a',c2a),('max_a2c',a2c)):
        if (not isinstance(rate,int) or rate%2400 or not 2400<=rate<=original[key]*2400
                or not original['mask'] & (1 << (rate//2400-1))):
            raise ValueError('directional rate outside original offer')
        local[key]=rate//2400
    local['mask'] &= (1 << (max(rates)//2400))-1
    return local


def offer(cap, receive=None, minimum=2400):
    if cap % 2400 or not 2400 <= cap <= 33600:
        raise ValueError('V.34 cap must be 2400..33600 in steps of 2400')
    if minimum % 2400 or not 2400 <= minimum <= cap:
        raise ValueError('V.34 minimum must be 2400..cap in steps of 2400')
    if receive is None:
        import os
        receive = dict(nonlinear=int(os.environ.get('V34_RX_NONLINEAR','0')),
                       shaping=int(os.environ.get('V34_RX_SHAPING','0')),
                       precoder=[int(x) for x in os.environ.get('V34_RX_PRECODER','0,0,0,0,0,0').split(',')])
    if receive.get('trellis',0) != 0:
        raise ValueError('only 16-state receive trellis is supported')
    if any(receive.get(k,0) not in (0,1,False,True) for k in ('nonlinear','shaping')):
        raise ValueError('receive nonlinear/shaping flags must be boolean')
    coeff = receive.get('precoder',[0]*6)
    if len(coeff)!=6 or any(not isinstance(x,int) or not -32768<=x<=32767 for x in coeff):
        raise ValueError('receive precoder requires six signed 16-bit coefficients')
    n = cap // 2400
    low = minimum // 2400
    mask = ((1 << n) - 1) ^ ((1 << (low - 1)) - 1)
    return dict(max_c2a=n, max_a2c=n, mask=mask, asym=1,
                trellis=0, nonlinear=int(receive.get("nonlinear",0)), shaping=int(receive.get("shaping",0)), precoder=list(coeff))


def resolve(local, peer, role, tx_baud, rx_baud, tx_carrier=0.0, rx_carrier=0.0):
    if role not in ('call', 'answer'):
        raise ValueError('V.34 role must be call or answer')
    for mp in (local, peer):
        if any(not 1 <= mp[k] <= 14 for k in ('max_c2a', 'max_a2c')):
            raise ValueError('reserved or zero MP directional rate')
        if mp.get('trellis', 0) not in (0, 1, 2):
            raise ValueError('reserved MP trellis selection')
        if len((mp.get('precoder') or [0] * 6)) != 6:
            raise ValueError('MP precoder must contain six coefficients')
    mask = local['mask'] & peer.get('mask', local['mask']) & 0x3fff
    asym = bool(local.get('asym') and peer.get('asym'))
    c2a = min(local['max_c2a'], peer['max_c2a'])
    a2c = min(local['max_a2c'], peer['max_a2c'])
    tx_cap, rx_cap = (c2a, a2c) if role == 'call' else (a2c, c2a)
    def params(rate, baud, carrier, request):
        return DirectionParameters(rate, baud, carrier, request.get('trellis', 0),
                                   bool(request.get('shaping')), bool(request.get('nonlinear')),
                                   tuple((request.get('precoder') or [0] * 6)))
    def enabled(n, baud, request):
        return bool(mask & (1 << (n - 1))) and supported(n * 2400, baud, request.get('shaping', 0))
    tx_rates = [n for n in range(1, tx_cap + 1) if enabled(n, tx_baud, peer)]
    rx_rates = [n for n in range(1, rx_cap + 1) if enabled(n, rx_baud, local)]
    if not asym:
        common = set(tx_rates) & set(rx_rates)
        if not common:
            raise ValueError('no common symmetric V.34 rate')
        tx_rates = rx_rates = [max(common)]
    if not tx_rates or not rx_rates:
        raise ValueError('no common V.34 rate for selected symbol rates')
    return NegotiatedParameters(params(max(tx_rates)*2400, tx_baud, tx_carrier, peer),
                                params(max(rx_rates)*2400, rx_baud, rx_carrier, local), asym)


def final_mp(frames):
    """Select the last CRC-valid acknowledged MP, or last offer before acknowledgement."""
    messages = [f for f in frames if isinstance(f,dict)]
    acknowledged = [f for f in messages if f.get('ack')]
    return (acknowledged or messages)[-1] if messages else None
