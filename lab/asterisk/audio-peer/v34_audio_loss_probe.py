#!/usr/bin/env python3
"""Deterministic V.8/V.34/LAPM duplex probe with isolated PCM erasures.

This pairs the production V.34 caller and answerer wrappers. It simulates a
lost RTP frame by applying DCE previous-frame concealment (or silence), and
notifies that endpoint through ``note_concealed_rx``. The default replaces one
20 ms frame in each direction, two/three seconds after queuing data.
``--erasure-ms 80`` tests a four-frame burst. This is a paired modem/media-path
test, not a socket, codec or Asterisk scheduling test.
"""

import argparse
import json
import os
import random
import struct
from pathlib import Path

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
    parser.add_argument('--erasure-ms', type=int, default=20,
                        help='contiguous concealed media per direction, in 20 ms RTP-frame multiples')
    parser.add_argument('--caller-erasure-offset-ms', type=int, default=2000,
                        help='caller-to-answer burst start after LAPM data begins (multiple of 20 ms)')
    parser.add_argument('--answerer-erasure-offset-ms', type=int, default=3000,
                        help='answer-to-caller burst start after LAPM data begins (multiple of 20 ms)')
    parser.add_argument('--seed', type=int, default=31,
                        help='deterministic payload seed; the reverse direction uses seed + 1')
    parser.add_argument('--output', type=Path, help='optional JSON result path')
    args = parser.parse_args()
    if (args.bytes <= 0 or args.seconds <= 0 or args.erasure_ms <= 0
            or args.erasure_ms % 20
            or args.caller_erasure_offset_ms < 0 or args.caller_erasure_offset_ms % 20
            or args.answerer_erasure_offset_ms < 0 or args.answerer_erasure_offset_ms % 20):
        parser.error('positive bytes/seconds, positive multiple-of-20 erasure duration, and nonnegative multiple-of-20 offsets required')
    os.environ['V34_RATE'] = str(args.rate)
    caller, answerer = V34Modem(data_rate=args.rate), V34AnswerModem(data_rate=args.rate)
    caller_payload = random.Random(args.seed).randbytes(args.bytes)
    answer_payload = random.Random(args.seed + 1).randbytes(args.bytes)
    caller_rx, answer_rx = bytearray(), bytearray()
    started = set()
    data_start = None
    erasures = [0, 0]
    complete = False
    status = 'FAIL'
    erasure_frames = args.erasure_ms // 20
    caller_erasure_start = args.caller_erasure_offset_ms // 20
    answerer_erasure_start = args.answerer_erasure_offset_ms // 20
    previous_caller = previous_answer = bytes(320)
    for frame in range(int(args.seconds * 50)):
        caller_pcm = struct.pack('<160h', *caller.tx_samples(160))
        answer_pcm = struct.pack('<160h', *answerer.tx_samples(160))
        if data_start is not None and not args.clean:
            if data_start + caller_erasure_start <= frame < data_start + caller_erasure_start + erasure_frames:
                caller_pcm = previous_caller if args.fill == 'repeat' else bytes(320)
                erasures[0] += 1
                answerer.note_concealed_rx(160)
            if data_start + answerer_erasure_start <= frame < data_start + answerer_erasure_start + erasure_frames:
                answer_pcm = previous_answer if args.fill == 'repeat' else bytes(320)
                erasures[1] += 1
                caller.note_concealed_rx(160)
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
        if complete and (args.clean or erasures == [erasure_frames, erasure_frames]):
            status = 'PASS'
            break
        if complete:
            print('Payload finished before both erasures; increase --bytes.')
            return 2
        if caller.v42_lost or answerer.v42_lost or caller.failed_reason or answerer.failed_reason:
            break
    def endpoint(modem):
        core = modem.caller if hasattr(modem, 'caller') else modem.answerer
        link = getattr(modem, '_v42', None)
        eq = getattr(getattr(core, '_p4rcv', None), 'eq', None)
        transport_guard = getattr(eq, '_transport_guard', None)
        return dict(v42_mode=modem.v42_mode,
                    lapm_state=getattr(link, 'state', None),
                    lapm_stats=dict(getattr(link, 'stats', {}) or {}),
                    lapm_events=list(getattr(link, 'events', ())[-20:]),
                    b1_errors=(getattr(core, 'rx_qualification', {}) or {}).get('bit_errors'),
                    timing_slips=getattr(eq, 'slips', None),
                    gap_feedback=(transport_guard.report() if transport_guard is not None else None),
                    failed_reason=getattr(modem, 'failed_reason', None),
                    received_bytes=len(caller_rx) if modem is caller else len(answer_rx),
                    events=list(modem.events[-8:]))
    result = dict(status=status, rate=args.rate, protocol='LAPM', payload_bytes_each_way=args.bytes,
                  exact_payload=dict(caller_to_answer=answer_rx == caller_payload,
                                    answer_to_caller=caller_rx == answer_payload),
                  simulated_seconds=(frame + 1) / 50,
                  media_gap=dict(fill=args.fill,
                                 concealment='previous 20 ms frame' if args.fill == 'repeat' else 'silence',
                                 duration_ms=args.erasure_ms,
                                 caller_to_answer_offset_ms=args.caller_erasure_offset_ms,
                                 answer_to_caller_offset_ms=args.answerer_erasure_offset_ms,
                                 caller_to_answer_frames=erasures[0],
                                 answer_to_caller_frames=erasures[1],
                                 receiver_notified_via='note_concealed_rx'),
                  endpoints=dict(caller=endpoint(caller), answerer=endpoint(answerer)),
                  scope='paired El-Modem V.34 wrappers; synthetic media concealment; no RTP socket, codec, Asterisk scheduling, or independent modem')
    encoded = json.dumps(result, indent=2) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    print(encoded, end='')
    return 0 if status == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
