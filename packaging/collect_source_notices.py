"""Retain component license files from verified compiled-dependency sources.

Archive-wide collection deliberately includes notices for potentially unused
parts. It is not a determination that every collected license applies to the ELF.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile


def collect(sources, output, packages):
    status = json.loads((sources / 'source-status.json').read_text())
    if status['errors']:
        raise ValueError('incomplete source collection')
    notices = []
    for package in packages:
        archives = [item for item in status['sources']
                    if item['package'] == package and '.orig.tar.' in item['filename']]
        if not archives:
            raise ValueError('no upstream source for ' + package)
        count = 0
        for archive in archives:
            path = sources / archive['file']
            if path.stat().st_size != archive['size'] or hashlib.sha256(path.read_bytes()).hexdigest() != archive['sha256']:
                raise ValueError('source archive checksum mismatch: ' + archive['file'])
            with tarfile.open(path) as bundle:
                for member in bundle:
                    name = PurePosixPath(member.name)
                    if not member.isfile() or name.name.lower() not in (
                            'license', 'license.txt', 'license.md', 'copying', 'copyright'):
                        continue
                    if name.is_absolute() or '..' in name.parts:
                        raise ValueError('unsafe notice member')
                    if member.size > 2 * 1024 * 1024:
                        raise ValueError('unexpectedly large notice')
                    content = bundle.extractfile(member).read()
                    target = output / package / Path(*name.parts)
                    if target.exists() and target.read_bytes() != content:
                        raise ValueError('notice output collision')
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(content)
                    notices.append(dict(package=package, archive=archive['file'],
                                        archive_sha256=archive['sha256'], member=member.name,
                                        file=str(target.relative_to(output)),
                                        sha256=hashlib.sha256(content).hexdigest()))
                    count += 1
        if not count:
            raise ValueError('no component notices found for ' + package)
    report = dict(artifact_sha256=status['artifact_sha256'], notices=notices,
                  scope='Source archive component notices; static linkage and license compatibility are not certified.')
    (output / 'inventory.json').write_text(json.dumps(report, indent=2) + '\n')
    (output / 'README.md').write_text(
        '# Compiled dependency source notices\n\n'
        'These notices are retained from the exact verified NumPy, LLVM and OpenBLAS '
        'source archives. See inventory.json for source and file hashes. Collection '
        'includes potentially unused archive components; it does not imply every '
        'listed license applies to this executable or certify complete static linkage '
        'coverage. Main installed-package notices and custom Python supplement remain '
        'part of the candidate notice sidecar.\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('sources', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--packages', nargs='+', default=['numpy', 'llvm-toolchain-14', 'openblas'])
    args = parser.parse_args()
    report = collect(args.sources, args.output, args.packages)
    print('Verified source notices:', len(report['notices']))
