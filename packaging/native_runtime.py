"""Render a shared native modem Compose override; never starts or dials it."""
import argparse
import json
from pathlib import PurePosixPath


NATIVE_ENV = {
    'OPENBLAS_NUM_THREADS': '1', 'V34_PHASE4_EQ_SPAN': '24',
    'V34_TRN_CLOCK': '1', 'V34_CLOCK_FIT_GATE': '1',
    'V34_SURVIVOR_STATE_SYNC': '1', 'V34_L2_GUARD': '1',
    'V34_RETRAIN_AFTER_E': '1', 'V34_INPUT_POWER_GUARD': 'fade',
    'V34_ANSWERER_RETRAIN': '1', 'V34_CALLER_RETRAIN': '1',
    'V34_NATIVE_FEEDBACK': '1', 'V34_SYMBOL_INTERP_NATIVE': '1',
    'V34_TRN_CLOCK_NATIVE': '1', 'MODEM_CAPTURE_ASYNC': '1',
    'V34_TRACKING_PACKET_POLL': '1', 'EL_MODEM_RTP_DEADLINE_DRAIN': '1',
    'EL_MODEM_DSP_TRAINING_STRIDE': '4', 'V34_TIMING_MODE': 'data-gradient',
    'V34_TRAJECTORY_GRID': '1', 'V34_CONNECTED_RATE': '1',
    'V34_QUALITY_MONITOR': '1', 'V34_AUTO_RATE': '1',
    'V42_DEFAULT_PARAMETERS': '0', 'V42_ASSUME_BIS': '0',
    'V34_PREIMPORT': '1', 'EL_MODEM_GC_PAUSE': '1', 'V34_NATIVE_TRELLIS': '1',
}


def override(image, capture_prefix):
    prefix = PurePosixPath(capture_prefix)
    if (not image or not prefix.is_absolute() or '..' in prefix.parts
            or not str(prefix).startswith('/tmp/cap/') or prefix.name in ('', '.')):
        raise ValueError('image and an absolute capture prefix below /tmp/cap are required')
    prefix = str(prefix)
    env = dict(NATIVE_ENV, MODEM_DEBUG='1',
               V34_CLOCK_FIT_DUMP=prefix,
               MODEM_CAPTURE_REQUIRED='1', MODEM_CAPTURE_TIMELINE='1')
    # Override the image's ENTRYPOINT as well as Compose's Python command:
    # otherwise Python/modem_dce.py would be appended twice.
    command = ['python3', '-u', '/app/modem_dce.py', '--line-mode', 'ms',
               '--link', '/tmp/modem0', '--tcp-listen', '0.0.0.0:9100',
               '--sip-host', '0.0.0.0', '--sip-port', '5062',
               '--media-host', '172.29.103.30', '--media-port', '16000',
               '--dial-host', '172.29.103.10', '--dial-port', '5060',
               '--timing-trace', '--capture', prefix+'.pcm',
               '--capture-seconds', '300', '--dsp-capture', prefix+'.dsp',
               '--metrics-file', prefix+'.metrics.json']
    return {'services': {'modem': {'image': image, 'pull_policy': 'never',
                                 'entrypoint': [], 'command': command,
                                 'environment': env}}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', required=True)
    parser.add_argument('--capture-prefix', required=True)
    args = parser.parse_args()
    print(json.dumps(override(args.image, args.capture_prefix), indent=2))


if __name__ == '__main__':
    main()
