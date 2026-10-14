#!/usr/bin/env python3
"""Replay the real ACTS capture through V32Modem and print the event log.

Offline reproduction of a real PSTN V.32 attempt: the .rx.raw capture recorded by
V32_CAPTURE is fed back in, so a handshake change can be evaluated without dialling.
"""
import math
import struct
import sys

from v32_stream import V32Modem

FS = 8000


def replay(path, secs=25.0, rate=9600, debug=True, use_v8=False, caller=False):
    """Feed a capture back into a V32Modem.

    ``caller=False`` keeps us silent, which models the answerer's leg: the engine's transmit timeline
    is nudged forward by hand and it only ever decodes. ``caller=True`` lets it generate its own
    transmit samples (discarded) so the whole handshake runs - our own TRN/rate words come out of
    the engine rather than the recording, and the far end in the capture answers those. That is
    what makes a real failing call reproducible offline.
    """
    with open(path, "rb") as fh:
        raw = fh.read()
    rx = struct.unpack(f"<{len(raw) // 2}h", raw)
    m = V32Modem(fs=FS, my_rates=(rate,), use_v8=use_v8, debug=debug)
    if not use_v8:
        m._v8 = None
    blk = FS // 50
    got = bytearray()
    n = min(len(rx), int(FS * secs))
    for i in range(0, n - blk, blk):
        if caller:
            m.tx_samples(blk)             # we transmit our own side; the audio is discarded
        else:
            m._txn += blk                  # we stay silent: this is the answerer's leg
        got += m.feed_pcm(struct.pack(f"<{blk}h", *rx[i:i + blk]))
        if m.state == "failed":
            break
    return m, bytes(got)


def envelope(path, secs=60.0, window=0.25):
    """Text envelope of a capture: RMS per window, as dB below the loudest window. Cheap way to
    see who was transmitting when, without needing the modem to have understood any of it."""
    with open(path, "rb") as fh:
        raw = fh.read()
    rx = struct.unpack(f"<{len(raw) // 2}h", raw[:int(FS * secs) * 2])
    step = int(FS * window)
    rows = []
    for i in range(0, len(rx) - step, step):
        acc = 0
        for v in rx[i:i + step]:
            acc += v * v
        rows.append((i / FS, (acc / step) ** 0.5))
    peak = max((r for _, r in rows), default=1.0) or 1.0
    bar = " .:-=+*#%@"
    for t, rms in rows:
        db = 20 * math.log10(rms / peak) if rms > 0 else -99.0
        n = max(0, min(9, int((db + 60) / 6)))
        print(f"  {t:6.2f}s {rms:8.0f} rms {db:6.1f} dB {bar[n] * 40}")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/opencode/v32acts/v32cap.rx.raw"
    secs = float(sys.argv[2]) if len(sys.argv) > 2 else 25.0
    if "--env" in sys.argv:
        envelope(path, secs)
    else:
        m, got = replay(path, secs, use_v8="--v8" in sys.argv, caller="--caller" in sys.argv)
        for t, e in m.events:
            print(f"  t={t:7.3f}s {e}")
        print(f"state={m.state} connected={m.connected} rate={m.bit_rate} got={len(got)}B")