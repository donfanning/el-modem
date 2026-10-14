"""Independently verify collected archives and Debian descriptor checksums."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def verify(root):
    report = json.loads((root / 'source-status.json').read_text())
    if report['errors']:
        raise ValueError('collection contains errors')
    descriptors = 0
    for record in report['sources']:
        path = (root / record['file']).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError('archive path escapes source directory')
        if path.stat().st_size != record['size'] or digest(path) != record['sha256']:
            raise ValueError('archive checksum mismatch: ' + record['file'])
        if path.suffix != '.dsc':
            continue
        descriptors += 1
        active = False
        entries = 0
        for line in path.read_text().splitlines():
            if line == 'Checksums-Sha256:':
                active = True
                continue
            if active and line.startswith(' '):
                expected, size, name = line.split()
                if Path(name).name != name:
                    raise ValueError('invalid descriptor filename')
                member = path.parent / name
                if member.stat().st_size != int(size) or digest(member) != expected:
                    raise ValueError('descriptor member checksum mismatch: ' + name)
                entries += 1
            elif active:
                break
        if not entries:
            raise ValueError('missing descriptor checksum stanza: ' + path.name)
    return dict(artifact_sha256=report['artifact_sha256'],
                archives_verified=len(report['sources']),
                debian_descriptors_verified=descriptors)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    print(json.dumps(verify(parser.parse_args().directory), indent=2))
