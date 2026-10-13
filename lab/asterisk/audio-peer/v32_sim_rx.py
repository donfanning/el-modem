"""Offline simulation of the answerer's second transmission through a clock-offset channel into
V32Modem's receive state machine (no blob, seconds per run). Used to tune/test timing tracking.

    python3 v32_sim_rx.py [ppm] [rate] [noise_rms] [echo_ms]
"""
import math, random, struct, sys
from v32_caller import V32CallerTx, rate_word
from v32_stream import V32Modem


def sinc_resample(x, ratio, half=40, nph=256, cutoff=0.88):
    tbl = []
    for p in range(nph):
        fr = p / nph
        w = []
        for j in range(-half + 1, half + 1):
            d = j - fr
            a = cutoff * d
            h = cutoff if abs(a) < 1e-12 else cutoff * math.sin(math.pi * a) / (math.pi * a)
            c = math.cos(math.pi * d / half)
            w.append(h * (0.42 + 0.5 * c + 0.08 * math.cos(2 * math.pi * d / half)))
        s = sum(w)
        tbl.append([v / s for v in w])
    out, pos = [], float(half)
    x = [0] * half + list(x) + [0] * half
    while pos + half < len(x):
        i = int(pos)
        t = tbl[int((pos - i) * nph)]
        out.append(sum(a * b for a, b in zip(t, x[i - half + 1: i + half + 1])))
        pos += ratio
    return out


def make_stream(carrier=1800.0, rate=9600, text=b"Simulated 9600 baud text through a drifting clock.\r\n" * 4, fs=8000, idle_s=0.0):
    """Answerer-style second transmission: S, S-bar, TRN(8044), R3 words, E3, then data."""
    tx = V32CallerTx(role="answerer", sample_rate=fs, my_rates=(rate,), trn_length=8044, carrier=carrier)
    tx.start_training_now(gap_symbols=0)
    audio = [0] * int(0.6 * fs)
    audio += tx.tx_samples(int(fs * (256 + 16 + 8044) / 2400))
    audio += tx.tx_samples(int(fs * 0.5))                       # R3 words
    tx.request_e(rate)
    audio += tx.tx_samples(int(fs * 0.3))
    tx.feed_bytes(text)
    audio += tx.tx_samples(int(fs * (len(text) * 10 / rate + 1.0)))
    if idle_s:
        audio += tx.tx_samples(int(fs * idle_s))              # idle marks for a long call
        tx.feed_bytes(text)
        audio += tx.tx_samples(int(fs * (len(text) * 10 / rate + 1.0)))
    return audio, text


def run(ppm=0.0, rate=9600, noise=0.0, fs=8000, debug=True, quiet=False, idle_s=0.0, df=0.0):
    audio, text = make_stream(1800.0 + df, rate, fs=fs, idle_s=idle_s)
    if ppm:
        audio = sinc_resample(audio, 1.0 + ppm * 1e-6)
    rnd = random.Random(2)
    if noise:
        audio = [v + rnd.gauss(0, noise) for v in audio]
    m = V32Modem(fs=fs, my_rates=(rate,), debug=debug)
    # jump straight into the receive state machine for the second transmission
    m.state = "tx"
    m._tx_started = True
    m._tx_tm0 = 0
    m._rx_mode = "locate2"
    m._resume_at = 0.6
    m._echo_state = "done"
    got = bytearray()
    blk = fs // 50
    for i in range(0, len(audio) - blk, blk):
        m._txn += blk
        got += m.feed_pcm(struct.pack(f"<{blk}h", *[max(-32768, min(32767, int(round(v)))) for v in audio[i:i + blk]]))
    if not quiet:
        for t, e in m.events:
            print(f"  t={t:7.3f}s {e}")
    return m, bytes(got), text


if __name__ == "__main__":
    ppm = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
    rate = int(sys.argv[2]) if len(sys.argv) > 2 else 9600
    noise = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
    m, got, text = run(ppm, rate, noise)
    print("connected:", m.connected, "rate:", m.bit_rate, "| text ok:", got.count(text[:50]) >= 1)
    print(got[:120])
