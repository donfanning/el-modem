"""Reproduce saved live interpolation, training and single-pass B1 qualification."""
import argparse
import json
from pathlib import Path

import numpy as np

import v34_rx as rx
from v34_receive import Receiver
from v34_negotiation import DirectionParameters


def verify(prefix):
    prefix = str(prefix)
    metadata = json.loads(Path(prefix+'.json').read_text())
    failures = list(metadata.get('errors', []))
    if not metadata.get('closed') or metadata.get('truncated'):
        failures.append('receiver capture not finalized or truncated')
    report = dict(groups=[], training=[], b1=None, failures=failures)
    with np.load(prefix+'.npz', allow_pickle=False) as data:
        for group in metadata['groups']:
            key = 'eq'+str(group['id'])+'_'
            source, positions = data[key+'source'], data[key+'positions']
            rebuilt = np.asarray([rx.interp(s, p)*r for s,p,r in
                                  zip(source,positions,data[key+'rotation'])])
            window_error = float(np.max(abs(rebuilt-data[key+'window'])))
            observed = np.sum(data[key+'window']*data[key+'taps'], axis=1)
            symbol_error = float(np.max(abs(observed-data[key+'observed'])))
            report['groups'].append(dict(id=group['id'], rows=len(source),
                window_max_error=window_error, symbol_max_error=symbol_error,
                phase_counts={name:int(np.sum(data[key+'phase']==number))
                              for name,number in metadata['phases'].items()}))
            if window_error>1e-10 or symbol_error>1e-10:
                failures.append('live interpolation/equalizer reproduction failed for '+key)
        for i, fit in enumerate(metadata['training_fits']):
            key = 'fit'+str(i)+'_'
            known = data[key+'known']
            nmse = float(np.mean(abs(data[key+'matrix']@data[key+'taps']-known)**2)
                         /np.mean(abs(known)**2))
            report['training'].append(dict(id=i, nmse=nmse, recorded_nmse=fit['nmse']))
            if not np.isclose(nmse,fit['nmse'],rtol=1e-10,atol=1e-12):
                failures.append('training fit does not reproduce')
        if 'b1_windows' in data:
            error = float(np.max(abs(data['b1_windows']@data['b1_taps_before']-data['b1_observed_before'])))
            report['b1'] = dict(before_tap_max_error=error)
            if error>1e-10:
                failures.append('B1 prior observation does not reproduce')
            if 'b1_observed_after' in data:
                shift = metadata['b1']['chosen_shift']
                n = len(data['b1_observed_after'])
                error = float(np.max(abs(data['b1_windows'][shift:shift+n]@data['b1_taps_after']-data['b1_observed_after'])))
                report['b1']['after_tap_max_error'] = error
                if error>1e-10:
                    failures.append('B1 refined observation does not reproduce')
        q = metadata.get('qualification') or {}
        if 'qualified_observed' in data:
            profile = dict(q['profile'])
            profile['precoder'] = tuple(profile['precoder'])
            # Training polynomial selection and the qualified remote direction
            # are different identities. Older archives did not export the latter.
            role = metadata.get('qualification_remote_role')
            if role is None:
                role = metadata['groups'][-1]['role']
            receiver = Receiver(DirectionParameters(**profile), role, q['initial_trellis_state'])
            gain, reference = receiver.qualify(data['qualified_observed'])
            error = float(np.max(abs(reference-data['qualified_projected_reference'])))
            bits_equal = np.array_equal(receiver.bits,data['qualified_decoded_bits'])
            report['b1'] = (report['b1'] or {}) | dict(
                bit_errors=receiver.diagnostics['bit_errors'], recorded_bit_errors=q['bit_errors'],
                decoded_bits_identical=bool(bits_equal), reference_max_error=error,
                gain_error=float(abs(gain-data['qualified_gain'])),
                energy_error=float(abs(receiver.energy-data['qualified_energy'])))
            trace = json.loads(Path(prefix+'.detector.json').read_text())
            n = len(data['qualified_observed'])
            report['b1']['detector_intervals'] = len(trace['trace'])
            if (not bits_equal or receiver.diagnostics['bit_errors']!=q['bit_errors']
                    or error>1e-10 or len(trace['trace'])*2!=n
                    or len(trace['snapshots'])*2!=n):
                failures.append('single-pass B1 qualification does not reproduce')
    report['valid'] = not failures
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('prefix')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = verify(args.prefix)
    text = json.dumps(report,indent=2)+'\n'
    if args.output:
        args.output.write_text(text)
    print(text,end='')
    return int(not report['valid'])


if __name__=='__main__':
    raise SystemExit(main())
