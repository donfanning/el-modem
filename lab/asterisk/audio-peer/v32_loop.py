"""Our V32Modem (caller) <-> our V32Answerer, block by block over a simulated line.

    python3 v32_loop.py [rate] [delay_ms] [seconds] [echo_db]
"""
import struct, sys, time
from collections import deque

from v32_answer import V32Answerer
from v32_stream import V32Modem


def run(rate=9600, delay_ms=40, seconds=60, echo_db=None, caller_text=b"caller says hello\r\n",
        answerer_text=b"answerer says hello\r\n", fs=8000, v42=False, use_v8=True, verbose=False,
        ans_trn=None):
    blk = fs // 50
    d = int(delay_ms * fs / 1000)
    caller = V32Modem(fs=fs, my_rates=(rate,) if rate else (14400, 12000, 9600, 7200, 4800), use_v8=use_v8, debug=verbose)
    ans_kw = {"trn_length": ans_trn} if ans_trn else {}
    ans = V32Answerer(fs=fs, my_rates=(rate,) if rate else (14400, 12000, 9600, 7200, 4800), use_v8=use_v8, debug=verbose, **ans_kw)
    to_ans, to_caller = deque([0] * d), deque([0] * d)
    g = 10 ** (echo_db / 20) if echo_db is not None else 0.0
    c_hist, a_hist = deque([0] * (2 * d + 1), maxlen=2 * d + 1), deque([0] * (2 * d + 1), maxlen=2 * d + 1)
    got_c, got_a = b"", b""
    sent_c = sent_a = False
    t0 = time.time()
    for b in range(int(50 * seconds)):
        c_out = caller.tx_samples(blk)
        a_out = ans.tx_samples(blk)
        to_ans.extend(c_out)
        to_caller.extend(a_out)
        rx_a = [to_ans.popleft() for _ in range(blk)]
        rx_c = [to_caller.popleft() for _ in range(blk)]
        if g:                                           # each side hears a delayed copy of its own signal
            c_hist.extend(c_out)
            a_hist.extend(a_out)
            hc, ha = list(c_hist), list(a_hist)
            rx_c = [int(v + g * hc[len(hc) - 2 * d - blk + i]) if len(hc) > 2 * d + blk else v for i, v in enumerate(rx_c)]
            rx_a = [int(v + g * ha[len(ha) - 2 * d - blk + i]) if len(ha) > 2 * d + blk else v for i, v in enumerate(rx_a)]
        got_c += caller.feed_pcm(struct.pack(f"<{blk}h", *[max(-32768, min(32767, v)) for v in rx_c]))
        got_a += ans.feed_pcm(struct.pack(f"<{blk}h", *[max(-32768, min(32767, v)) for v in rx_a]))
        if caller.connected and ans.connected:
            if v42 and caller.v42_mode == "off":
                caller.start_v42(calling_party=True)
                ans.start_v42(calling_party=False)
            ready = (not v42) or (caller.v42_resolved and ans.v42_resolved)
            if ready and not sent_c:
                caller.feed_bytes(caller_text); sent_c = True
            if ready and not sent_a:
                ans.feed_bytes(answerer_text); sent_a = True
            if sent_c and sent_a and got_a.endswith(caller_text) and got_c.endswith(answerer_text):
                break
        if caller.state == "failed" or ans.state == "failed":
            break
    if verbose:
        for who, m in (("caller", caller), ("answerer", ans)):
            for t, e in m.events:
                print(f"  [{who:8s}] t={t:7.3f}s {e}")
    return caller, ans, got_c, got_a, time.time() - t0


if __name__ == "__main__":
    rate = int(sys.argv[1]) if len(sys.argv) > 1 else 9600
    delay = float(sys.argv[2]) if len(sys.argv) > 2 else 40
    secs = float(sys.argv[3]) if len(sys.argv) > 3 else 60
    echo = float(sys.argv[4]) if len(sys.argv) > 4 else None
    c, a, gc, ga, wall = run(rate, delay, secs, echo, verbose=True)
    print("caller:", c.state, c.connected, c.bit_rate, "| answerer:", a.state, a.connected, a.bit_rate)
    print("caller received:", gc); print("answerer received:", ga); print(f"wall {wall:.1f}s")
