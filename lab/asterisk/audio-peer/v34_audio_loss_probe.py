#!/usr/bin/env python3
"""Deterministic V.8/V.34/LAPM duplex probe with isolated PCM erasures.

This exercises the modem receiver, without SIP, codecs or RTP reordering.
Default: replace one 20 ms frame in each direction with its previous frame,
as the DCE currently does for short gaps, two/three seconds after queuing
data. --fill silence tests zero fill instead. Use --clean to check the same
transfer without erasures. A nonzero exit means negotiation or transfer failed.
"""

import argparse
import os
import random
import struct

from v34_answer_modem import V34AnswerModem
from v34_modem import V34Modem


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rate', type=int, default=9600)
    parser.add_argument('--bytes', type=int, default=8000)
    parser.add_argument('--seconds', type=float, default=90,
                        help='maximum simulated call duration')
    parser.add_argument('--clean', action='store_true')
    parser.add_argument('--fill', choices=('repeat', 'silence'), default='repeat')
    args = parser.parse_args()
    if args.bytes <= 0 or args.seconds <= 0:
        parser.error('--bytes and --seconds must be positive')
    os.environ['V34_RATE'] = str(args.rate)
    caller, answerer = V34Modem(data_rate=args.rate), V34AnswerModem(data_rate=args.rate)
    caller_payload = random.Random(31).randbytes(args.bytes)
    answer_payload = random.Random(32).randbytes(args.bytes)
    caller_rx, answer_rx = bytearray(), bytearray()
    started = set()
    data_start = None
    erasures = [0, 0]
    complete = False
    previous_caller = previous_answer = bytes(320)
    for frame in range(int(args.seconds * 50)):
        caller_pcm = struct.pack('<160h', *caller.tx_samples(160))
        answer_pcm = struct.pack('<160h', *answerer.tx_samples(160))
        if data_start is not None and not args.clean:
            if frame == data_start + 100:
                caller_pcm = previous_caller if args.fill == 'repeat' else bytes(320)
                erasures[0] += 1
            if frame == data_start + 150:
                answer_pcm = previous_answer if args.fill == 'repeat' else bytes(320)
                erasures[1] += 1
        answer_rx.extend(answerer.feed_pcm(caller_pcm))
        caller_rx.extend(caller.feed_pcm(answer_pcm))
        previous_caller, previous_answer = caller_pcm, answer_pcm
        for modem, calling in ((caller, True), (answerer, False)):
            if modem.connected and calling not in started:
                modem.start_v42(calling_party=calling, frame_size=32)
                started.add(calling)
        if data_start is None and caller.v42_mode == answerer.v42_mode == 'lapm':
            caller.feed_bytes(caller_payload)
            answerer.feed_bytes(answer_payload)
            data_start = frame
        complete = caller_rx == answer_payload and answer_rx == caller_payload
        if complete and (args.clean or erasures == [1, 1]):
            print(f'PASS: {args.rate}/LAPM, {args.bytes} bytes each way, '
                  f'{(frame + 1) / 50:.2f}s simulated, {args.fill} fill, PCM erasures {erasures}')
            return 0
        if complete:
            print('Payload finished before both erasures; increase --bytes.')
            return 2
        if caller.v42_lost or answerer.v42_lost or caller.failed_reason or answerer.failed_reason:
            break
    print(f'FAIL: {args.rate}/LAPM, caller received {len(caller_rx)}/{args.bytes}, '
          f'answerer received {len(answer_rx)}/{args.bytes}, PCM erasures {erasures}, '
          f'{args.fill} fill, V.42 modes {caller.v42_mode}/{answerer.v42_mode}')
    print('Caller:', caller.events[-3:])
    print('Answerer:', answerer.events[-3:])
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
