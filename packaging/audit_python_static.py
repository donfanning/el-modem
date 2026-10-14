"""Map custom Python's Setup.local inputs and collect static hash notices.

Uses an already verified Python source archive; never extracts archive paths or
executes source. The map covers Setup.local, not every CPython core object.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import tarfile


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(sources, output):
    status = json.loads((sources / 'source-status.json').read_text())
    if status['errors']:
        raise ValueError('source collection has unresolved errors')
    archives = [record for record in status['sources']
                if record['package'] == 'python3.11' and '.orig.tar.' in record['filename']]
    if len(archives) != 1:
        raise ValueError('expected one inventoried Python upstream archive')
    record = archives[0]
    archive = sources / record['file']
    if archive.stat().st_size != record['size'] or sha(archive) != record['sha256']:
        raise ValueError('Python source archive checksum mismatch')
    setup = sources / 'custom-python-build/Setup.local'
    setup_records = [item for item in status['build_material']
                     if item['file'] == 'custom-python-build/Setup.local']
    if len(setup_records) != 1 or sha(setup) != setup_records[0]['sha256']:
        raise ValueError('Setup.local differs from retained build record')
    modules = []
    shared = False
    with tarfile.open(archive) as bundle:
        members = {member.name: member for member in bundle.getmembers() if member.isfile()}
        roots = {name.split('/')[0] for name in members}
        if len(roots) != 1:
            raise ValueError('ambiguous Python source root')
        prefix = next(iter(roots)) + '/'
        for line in setup.read_text().splitlines():
            tokens = shlex.split(line, comments=True)
            if not tokens:
                continue
            if tokens[0] in ('*shared*', '*static*'):
                shared = tokens[0] == '*shared*'
                continue
            if tokens[0].startswith('*'):
                raise ValueError('unsupported Setup directive: ' + tokens[0])
            inputs = []
            for token in tokens[1:]:
                if token.endswith('.c'):
                    member_name = prefix + 'Modules/' + token
                    if member_name not in members:
                        raise ValueError('missing source member: ' + member_name)
                    content = bundle.extractfile(members[member_name]).read()
                    inputs.append(dict(member=member_name, sha256=hashlib.sha256(content).hexdigest()))
            if not inputs:
                raise ValueError('no source inputs: ' + tokens[0])
            modules.append(dict(module=tokens[0], linkage='shared' if shared else 'builtin',
                                inputs=inputs, link_flags=[token for token in tokens if token.startswith('-l')]))

        # Full files retain inline permission notices; source is already public
        # upstream material. They are deliberately not trimmed into guesses.
        notice_paths = ['LICENSE', 'Doc/license.rst', 'Modules/_sha3/LICENSE',
                        'Modules/_sha3/sha3.c', 'Modules/_sha3/sha3module.c',
                        'Modules/md5module.c', 'Modules/sha1module.c',
                        'Modules/sha256module.c', 'Modules/sha512module.c',
                        'Modules/_blake2/blake2module.c',
                        'Modules/_blake2/blake2b_impl.c', 'Modules/_blake2/blake2s_impl.c',
                        'Modules/_blake2/blake2module.h']
        notice_paths.extend(name[len(prefix):] for name in sorted(members)
                            if name.startswith(prefix + 'Modules/_blake2/impl/')
                            and name.endswith(('.c', '.h')))
        absent = [name for name in notice_paths if prefix + name not in members]
        if absent:
            raise ValueError('missing notice sources: ' + ', '.join(absent))
        output.mkdir(parents=True, exist_ok=True)
        notices = []
        for name in notice_paths:
            content = bundle.extractfile(members[prefix + name]).read()
            destination = output / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content)
            notices.append(dict(file=name, source_member=prefix + name,
                                sha256=hashlib.sha256(content).hexdigest()))
    result = dict(artifact_sha256=status['artifact_sha256'], archive=record['file'],
                  archive_sha256=record['sha256'], setup_sha256=sha(setup),
                  modules=modules, notices=notices,
                  limits=['Setup.local inputs and hash notices only; not a complete static-code or license audit.',
                          'Input hashes describe upstream files; retained Debian patches may change built sources.',
                          'A repeated pyexpat entry is retained as recorded, not silently simplified.'])
    (output / 'inventory.json').write_text(json.dumps(result, indent=2) + '\n')
    (output / 'README.md').write_text(
        '# Custom Python static-component notices\n\n'
        'Collected from the checksum-verified Python source archive for the qualified '
        'executable. Modules/_sha3/LICENSE is the MIT notice of Markku-Juhani O. Saarinen. '
        'The retained hash source files carry additional inline copyright/permission notices. '
        'LICENSE and Doc/license.rst retain Python licensing and incorporated-source terms. '
        'See inventory.json for source paths and hashes. This supplement does not certify '
        'complete license compatibility or enumerate every statically compiled core object.\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('sources', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    report = audit(args.sources, args.output)
    print(json.dumps(dict(setup_entries=len(report['modules']), notices=len(report['notices'])), indent=2))
