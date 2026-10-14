#!/usr/bin/env python3
"""Decode a V.32 answerer recording (int16 LE) with v32_rx: decode_capture.py FILE FS TRN_START_S [TAP]"""
import struct, sys, time
from v32_rx import V32Rx, deframe_async

path, fs, trn_s = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
tap = int(sys.argv[4]) if len(sys.argv) > 4 else 5
raw = open(path, "rb").read()
x = struct.unpack(f"<{len(raw)//2}h", raw)
cut = int((trn_s - 0.6) * fs)
t0 = time.time()
bits, diag = V32Rx(fs, tap).run(x[cut:], 0.6, trn_len=int(sys.argv[5]) if len(sys.argv) > 5 else 7000)
print(f"decoded {len(bits)//2} symbols in {time.time()-t0:.1f}s; {diag}")
data = deframe_async(bits)
print("async bytes:", data[:200])
