"""Forward model of the answerer's idle (scrambled ones) transmission after B1, for checking a receiver's data phase.

After B1 a transmitter that has no user data keeps sending scrambled ones (V.34 10.1.3.1, clause 7), so its waveform is
known exactly and can be compared with what a receiver recovers. This is the B1 construction of
``v34_b1_reference._build`` run for more mapping frames (same scrambler, shell mapper, differential encoder, trellis state and
inversion counter), for profiles without precoder or nonlinear encoding. ``build_idle(...)[:120]`` equals the B1 reference
over B1's own 120 symbols (tested), so the continuation uses the same arithmetic.
"""
import math

import numpy as np

import v34_b1_reference as B

def _trellis_step(trellis, state, label):
    """(y0, next_state) for the convolutional encoder of the given selection, written from Figures 10-12 of V.34 (02/98).

    ``state`` holds the outputs of the 2T delays D1..Dn in figure order and ``label`` is Y4Y3Y2Y1. Independent of
    ``v34_data.ConvEncoder*`` (it was read from the figures separately; the test compares the two).
    """
    y1, y2, y3, y4 = (label >> 0) & 1, (label >> 1) & 1, (label >> 2) & 1, (label >> 3) & 1
    if trellis == 0:                                           # Figure 10
        d1, d2, d3, d4 = state
        return d4, [d4, d1 ^ y2 ^ d4, d2 ^ y2, d3 ^ y1]
    if trellis == 1:                                           # Figure 11: Y2, Y1, Y4, Y2 into the four adders
        d1, d2, d3, d4, d5 = state
        return d5, [d5, d1 ^ y2, d2 ^ y1, d3 ^ y4, d4 ^ y2]
    if trellis == 2:                                           # Figure 12
        d1, d2, d3, d4, d5, d6 = state
        n1 = d1 ^ d2                                           # adder after D1, fed back from the output of D2
        t = d2 ^ y1                                            # adder after D2 and Y1
        and2 = d3 & t                                          # lower AND gate: D3 output and (D2 output + Y1)
        and1 = y2 & d3                                         # upper AND gate: Y2 and D3 output
        return d6, [y4 ^ n1 ^ and2,                            # Y4 + (n1 + and2) into D1
                    n1 ^ (y3 ^ and1) ^ d4,                     # n1 + (Y3 + and1) + D4 output into D2
                    t ^ d3,                                    # (D2 output + Y1) + D3 output into D3
                    d3,                                        # D3 output into D4
                    d6,                                        # Y0 (D6 output) into D5
                    d5 ^ y2 ^ d3]                              # D5 output + Y2 + D3 output into D6
    raise ValueError('trellis selection')


_PARAMS = {2400: (7, 12), 2743: (8, 12), 2800: (7, 14), 3000: (7, 15), 3200: (7, 16), 3429: (8, 15)}


def build_idle(rate, baud, role, frames, shaping=False, initial_state=None, trellis=0):
    """Return (x, parameters): ``frames`` mapping frames of idle waveform symbols starting at the B1 boundary (B1 is the first P frames)."""
    j, p = _PARAMS[round(baud)]
    count = round(rate * .28 / j)
    b = math.ceil(count / p)
    r = count - (b - 1) * p
    q = max(0, math.ceil((b - 43) / 8))
    k = max(0, b - 12 - 8 * q)
    m = math.floor(1.25 * 2 ** (k / 8) + .5) if shaping else math.ceil(2 ** (k / 8))
    quarter = sorted((complex(x, y) for x in range(-47, 48, 4) for y in range(-47, 48, 4)),
                     key=lambda z: (abs(z) ** 2, -z.imag))
    bits = B.scrambled_ones(frames * b + 64, role)
    state = list(initial_state) if initial_state is not None else [0] * (4, 5, 6)[trellis]
    inversion = '01110111111110' if j == 7 else '0111011111111010'
    offset, accumulator, rotation, interval = 0, 0, 0, 0
    x = []

    def take(n):
        nonlocal offset
        value = bits[offset:offset + n]
        offset += n
        return value

    def little(seq):
        return sum(bit * 2 ** i for i, bit in enumerate(seq))

    for _ in range(frames):
        accumulator += r
        high = accumulator >= p
        if high:
            accumulator -= p
        rank = little(take(k if high else k - 1)) if k else 0
        rings = B.shell_unrank(m, rank) if k else [0] * 8
        for pair in range(4):
            i1, i2, i3 = take(3)
            qa, qb = little(take(q)), little(take(q))
            rotation = (rotation + i2 + 2 * i3) % 4
            u0 = quarter[rings[2 * pair] * 2 ** q + qa] * (-1j) ** rotation
            inv = int(inversion[(2 * j - 2 + interval // (2 * p)) % len(inversion)]) if interval % (2 * p) == 0 else 0
            y0 = state[-1]                                  # Y0 is the output of the last delay
            ubit = y0 ^ 0 ^ inv                             # no precoder: both c are zero, parity 0
            u1 = quarter[rings[2 * pair + 1] * 2 ** q + qb] * (-1j) ** ((rotation + 2 * i1 + ubit) % 4)
            label = int(B._LABELS[B.subset(u0)][B.subset(u1)], 16)
            _, state = _trellis_step(trellis, state, label)
            x += [u0, u1]
            interval += 1
    return np.array(x), dict(J=j, P=p, count=count, b=b, r=r, q=q, K=k, M=m)


def residual_by_block(received, expected, block=160):
    """RMS of received - expected for each ``block`` symbols (20 mapping frames by default)."""
    n = min(len(received), len(expected))
    e = np.asarray(received[:n]) - np.asarray(expected[:n])
    return [float(np.sqrt(np.mean(abs(e[i:i + block]) ** 2))) for i in range(0, n - block + 1, block)]
