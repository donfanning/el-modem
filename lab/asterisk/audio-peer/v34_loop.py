"""In-process V.34 caller <-> answerer over a delay line (no V.8): used by the answerer tests."""
from __future__ import annotations

import numpy as np

from v34_ans import V34Answerer
from v34_call import V34Caller


def run(seconds: float = 20.0, delay_ms: float = 25.0, rate: int = 4800, syms=(2400,), noise: float = 0.0,
        send: bytes = b"HELLO FROM THE CALLER\r\n" * 3, reply: bytes = b"HELLO FROM THE ANSWERER\r\n" * 3,
        log=None, seed: int = 1):
    rng = np.random.default_rng(seed)
    call = V34Caller(symbol_rates=list(syms), data_rate=rate, log=(lambda m: log("C " + m)) if log else None)
    ans = V34Answerer(symbol_rates=list(syms), data_rate=rate, log=(lambda m: log("A " + m)) if log else None)
    d = int(delay_ms * 8)
    to_ans, to_call = [np.zeros(d)], [np.zeros(d)]
    sent = False
    t_data = None
    for tick in range(int(seconds * 50)):
        c, a = call.tx(160), ans.tx(160)
        to_ans.append(c)
        to_call.append(a)
        xa = np.concatenate(to_ans)[:160]
        xc = np.concatenate(to_call)[:160]
        to_ans = [np.concatenate(to_ans)[160:]]
        to_call = [np.concatenate(to_call)[160:]]
        if noise:
            xa = xa + rng.normal(0, noise, 160)
            xc = xc + rng.normal(0, noise, 160)
        call.rx(xc)
        ans.rx(xa)
        if call.stage == "data" and ans.stage == "data" and not sent:
            t_data = tick
            call.feed_bytes(send)
            ans.feed_bytes(reply)
            sent = True
    return call, ans
