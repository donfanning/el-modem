"""Independent V.34 (02/98) B1 construction, clauses 7, 8, 9 and 10.1.3.1.

No production modem imports. Shell unranking uses recursive polynomial counts;
the scrambler uses absolute bit indices rather than a shift register. Output
includes intermediate domains for diagnosing errors before shell decoding.
"""
from functools import lru_cache
import math


def scrambled_ones(count, role):
    if role not in ('call', 'answer'):
        raise ValueError('invalid role')
    tap = 18 if role == 'call' else 5
    bits = []
    for n in range(count):
        bits.append(1 ^ (bits[n-tap] if n >= tap else 0) ^
                    (bits[n-23] if n >= 23 else 0))
    return bits


@lru_cache(None)
def shell_counts(m, width):
    if width == 1:
        return (1,) * m
    half = shell_counts(m, width//2)
    return tuple(sum(half[k] * half[s-k] for k in range(len(half))
                     if 0 <= s-k < len(half))
                 for s in range(2*len(half)-1))


def shell_unrank(m, rank):
    counts = shell_counts(m, 8)
    total = 0
    while rank >= counts[total]:
        rank -= counts[total]
        total += 1
    def split(width, energy, ordinal):
        if width == 1:
            return [energy]
        half = shell_counts(m, width//2)
        for left in range(len(half)):
            right = energy-left
            if not 0 <= right < len(half):
                continue
            block = half[left]*half[right]
            if ordinal < block:
                return (split(width//2, left, ordinal % half[left]) +
                        split(width//2, right, ordinal // half[left]))
            ordinal -= block
        raise ValueError('shell rank outside constellation')
    return split(8, total, rank)


def quantize(value, step):
    def component(x):
        magnitude = abs(x)/step
        lower = math.floor(magnitude)
        return math.copysign((lower + (magnitude-lower > .5))*step, x)
    return complex(component(value.real), component(value.imag))


# Table 13: rows s(2m), columns s(2m+1), values Y4Y3Y2Y1.
_LABELS = ('00118899', '3223baab', '5544ddcc', '6776effe',
           '88990011', 'baab3223', 'ddcc5544', 'effe6776')


def subset(point):
    a = (int(point.real)-1)//2
    b = (int(point.imag)-1)//2
    return ((((a+1)//2 ^ (b+1)//2) & 1) << 2 |
            (a & 1) << 1 | ((a+b) & 1))


def build(rate, baud, role, shaping=False, precoder=(0,)*6,
          initial_state=(0,)*4, *, trace=False, initial_rotation=0):
    """Return independently constructed scrambled bits, mapper u, y and x.

    Nonzero initial_state is a diagnostic hypothesis, not normative B1 reset.
    Nonzero initial_rotation (differential-encoder Z0, quarter turns clockwise) is likewise a
    diagnostic hypothesis: an odd value is equivalent, up to a global quarter turn, to
    initial_state [0,1,0,0] (see lab/V34_B1_INITIAL_ROTATION.md).
    Nonlinear projection requires average linear power and is left to the caller.
    """
    return _build(rate,baud,role,shaping,precoder,initial_state,trace,initial_rotation=initial_rotation)


def diagnostic_build(rate,baud,role,shaping=False,precoder=(0,)*6,
                     initial_state=(0,)*4,*,flip_bits=(),inversion_offset_4d=0):
    """Nonstandard sequence hypotheses; never used by the transmitter.

    flip_bits indexes the scrambled bitstream. Offset shifts the inversion
    counter in 4D intervals, keeping the physical E/B1 boundary fixed.
    """
    if inversion_offset_4d not in (-2,-1,0,1,2):raise ValueError('invalid inversion offset')
    result=_build(rate,baud,role,shaping,precoder,initial_state,False,
                  tuple(flip_bits),inversion_offset_4d)
    result['standard_qualification']=False
    return result


def _build(rate,baud,role,shaping,precoder,initial_state,trace,
           flip_bits=(),inversion_offset_4d=0,initial_rotation=0):
    j, p = {2400:(7,12),2743:(8,12),2800:(7,14),3000:(7,15),
            3200:(7,16),3429:(8,15)}[round(baud)]
    count = round(rate*.28/j)
    b = math.ceil(count/p)
    r = count-(b-1)*p
    q = max(0, math.ceil((b-43)/8))
    k = max(0,b-12-8*q)
    m = math.floor(1.25*2**(k/8)+.5) if shaping else math.ceil(2**(k/8))
    quarter = sorted((complex(x,y) for x in range(-47,48,4)
                       for y in range(-47,48,4)), key=lambda z:(abs(z)**2,-z.imag))
    bits = scrambled_ones(count,role)
    for index in flip_bits:
        if not 0<=index<len(bits):raise ValueError('diagnostic bit index outside B1')
        bits[index]^=1
    coefficients = [complex(*precoder[t:t+2])/16384 for t in (0,2,4)]
    history = [0j]*3
    state = list(initial_state)
    inversion = '01110111111110' if j == 7 else '0111011111111010'
    if initial_rotation not in (0,1,2,3):raise ValueError('initial rotation must be 0..3 quarter turns')
    offset, accumulator, rotation, interval = 0,0,initial_rotation,0
    domains = dict(bits=bits,u=[],y=[],x=[],c=[],states=[])
    if trace:
        domains['alignment']=[]
        domains['specification']=dict(edition='V.34 (02/1998)',
            initialization='10.1.3.1',scrambler='clause 7',
            framing='8.1, Table 7',switching='8.2',shell='9.3',
            differential='9.5, equation 9-25',mapper='9.6.1, Table 11',
            precoder='9.6.2, equations 9-27 through 9-31',
            subset='9.6.3.1, Figure 9, Table 13',
            convolutional='9.6.3.2, Figure 10',modulo='9.6.3.3',
            inversion='9.6.3, equation 9-32, Table 12',
            nonlinear='9.7, equations 9-33 through 9-35')
        domains['parameters']=dict(J=j,P=p,N=count,b=b,r=r,K=k,q=q,M=m,
            initial_state=list(initial_state),initial_rotation=initial_rotation,inversion_start=2*j-2)
    def point(z):return [float(z.real),float(z.imag)]
    def take(n):
        nonlocal offset
        value=bits[offset:offset+n];offset+=n
        return value
    def little(seq):
        return sum(bit*2**index for index,bit in enumerate(seq))
    def prediction():
        pred=quantize(sum(a*h for a,h in zip(history,coefficients)),1/128)
        return pred,quantize(pred,2 if b<56 else 4)
    for frame in range(p):
        frame_bit_start=offset
        accumulator += r
        high = accumulator >= p
        if high: accumulator -= p
        shell_bit_start=offset
        shell_rank=little(take(k if high else k-1)) if k else 0
        rings = shell_unrank(m,shell_rank) if k else [0]*8
        shell_bit_end=offset
        for pair in range(4):
            pair_bit_start=offset
            rotation_before=rotation
            if k:
                i1,i2,i3=take(3)
                qa,qb=little(take(q)),little(take(q))
            else:
                group=take(3 if pair < (b if high else b-1)-8 else 2)
                i1,i2,i3=group+[0]*(3-len(group));qa=qb=0
            rotation=(rotation+i2+2*i3)%4
            pred0,c0=prediction()
            u0=quarter[rings[2*pair]*2**q+qa]*(-1j)**rotation
            y0=u0+c0;x0=y0-pred0
            history=[x0]+history[:2]
            pred1,c1=prediction()
            parity=(int(c0.real/2)+int(c0.imag/2)+int(c1.real/2)+int(c1.imag/2))&1
            counter=interval+inversion_offset_4d
            inv=int(inversion[(2*j-2+counter//(2*p))%len(inversion)]) if counter%(2*p)==0 else 0
            ubit=state[3]^parity^inv
            u1=quarter[rings[2*pair+1]*2**q+qb]*(-1j)**((rotation+2*i1+ubit)%4)
            y1=u1+c1;x1=y1-pred1
            history=[x1]+history[:2]
            domains['states'].append(tuple(state))
            label=int(_LABELS[subset(y0)][subset(y1)],16)
            y2,y1bit=(label>>1)&1,label&1
            d1,d2,d3,d4=state
            state=[d4,d1^y2^d4,d2^y2,d3^y1bit]
            if trace:
                domains['alignment'].append(dict(mapping_frame=frame,interval_4d=interval,
                    symbols_2d=[2*interval,2*interval+1],frame_high=bool(high),
                    frame_bit_start=frame_bit_start,pair_bit_span=[pair_bit_start,offset],
                    pair_bits=bits[pair_bit_start:offset],shell_bit_span=[shell_bit_start,shell_bit_end],
                    shell_rank=shell_rank,rings=rings[2*pair:2*pair+2],
                    I=[i1,i2,i3],Q=[qa,qb],rotation_before=rotation_before,Z=rotation,
                    predictor=[point(pred0),point(pred1)],c=[point(c0),point(c1)],
                    u=[point(u0),point(u1)],y=[point(y0),point(y1)],x=[point(x0),point(x1)],
                    subsets=[subset(y0),subset(y1)],Y4Y3Y2Y1=label,Y0=d4,C0=parity,V0=inv,U0=ubit,
                    inversion_index=(2*j-2+interval//(2*p)) if interval%(2*p)==0 else None,
                    state_before=[d1,d2,d3,d4],state_after=list(state)))
            for key,values in [('u',(u0,u1)),('y',(y0,y1)),('x',(x0,x1)),('c',(c0,c1))]:
                domains[key].extend(values)
            interval+=1
    assert offset==count
    return domains
