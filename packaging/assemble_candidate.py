"""Assemble a local review bundle from exact inventoried inputs and public source.

Does not rebuild, execute, publish, or approve redistribution of the candidate.
Only explicitly selected files are copied; private handoffs/results are excluded.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def assemble(executable, notices, sources, output):
    if output.exists():
        raise ValueError('output must be a new directory')
    measured = json.loads((notices / 'inventory.json').read_text())
    status = json.loads((sources / 'source-status.json').read_text())
    verification = json.loads((sources / 'source-verification.json').read_text())
    artifact = sha(executable)
    if any(item['artifact_sha256'] != artifact for item in (measured, status, verification)):
        raise ValueError('inputs belong to different executables')
    if measured['blockers'] or status['errors']:
        raise ValueError('inventory/source collection has blockers')
    catalog = json.loads((notices / 'debian-source-catalog.json').read_text())
    covered = {(item['package'], item['version']) for item in status['sources']}
    required = {(item['package'], item['version']) for item in catalog}
    required.update((item['name'], item['version']) for item in measured['python_distributions']
                    if not item.get('debian_packages'))
    required.add((measured['packaging_tool']['name'], measured['packaging_tool']['version']))
    if required - covered:
        raise ValueError('missing exact component sources: ' + repr(sorted(required - covered)))
    application = sources / 'application/el-modem-public-source.tar.gz'
    if sha(application) != verification['public_application_archive_sha256']:
        raise ValueError('public application archive changed')
    with tarfile.open(application) as bundle:
        names = bundle.getnames()
        if any('.git' in Path(name).parts for name in names):
            raise ValueError('application archive contains Git history')
        for name, expected in measured['build_manifest']['local_modules'].items():
            content = bundle.extractfile('el-modem/' + name).read()
            if hashlib.sha256(content).hexdigest() != expected:
                raise ValueError('application module differs from executable: ' + name)
        spec = bundle.extractfile('el-modem/packaging/el_modem.spec').read()
        if hashlib.sha256(spec).hexdigest() != measured['build_manifest']['specification_sha256']:
            raise ValueError('application build specification differs')
        public_files = {name: bundle.extractfile('el-modem/' + name).read() for name in (
            'LICENSE', 'NOTICE', 'CREDITS.md', 'docs/DESKTOP.md', 'docs/AT_COMMANDS.md',
            'docs/SERIAL.md', 'packaging/serial/README.md',
            'packaging/serial/build-driver.sh', 'packaging/serial/serialsim-flow.patch')}
        build_files = {name: bundle.extractfile('el-modem/packaging/' + name).read()
                       for name in ('build.sh', 'el_modem.spec', 'Dockerfile.build', 'Dockerfile.lto-libpython')}

    output.mkdir(parents=True)

    def copy(origin, destination, expected=None):
        if origin.is_symlink() or not origin.is_file():
            raise ValueError('input must be a regular file: ' + str(origin))
        if expected is not None and sha(origin) != expected:
            raise ValueError('input checksum mismatch: ' + str(origin))
        target = output / destination
        if not target.resolve().is_relative_to(output.resolve()):
            raise ValueError('destination escapes candidate')
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(origin, target)

    copy(executable, 'bin/el-modem', artifact)
    (output / 'bin/el-modem').chmod(0o755)
    for name, content in public_files.items():
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        if name.endswith('.sh'):
            target.chmod(0o755)
    for name, content in build_files.items():
        target = output / 'sources/application-build' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        if name.endswith('.sh'):
            target.chmod(0o755)
    for item in measured['notices']:
        copy(notices / item['file'], 'third-party-notices/' + item['file'], item['sha256'])
    for name in ('inventory.json', 'debian-source-catalog.json', 'THIRD_PARTY_NOTICES.md'):
        copy(notices / name, 'third-party-notices/' + name)
    supplement_count = 0
    for component in ('python-static', 'compiled-source'):
        directory = notices / 'licenses' / component
        supplement = json.loads((directory / 'inventory.json').read_text())
        if supplement['artifact_sha256'] != artifact:
            raise ValueError('notice supplement belongs to another executable')
        for item in supplement['notices']:
            copy(directory / item['file'], 'third-party-notices/licenses/' + component + '/' + item['file'], item['sha256'])
            supplement_count += 1
        for name in ('inventory.json', 'README.md'):
            copy(directory / name, 'third-party-notices/licenses/' + component + '/' + name)
    for item in status['sources']:
        origin = sources / item['file']
        if origin.stat().st_size != item['size']:
            raise ValueError('source archive size changed')
        copy(origin, 'sources/' + item['file'], item['sha256'])
    for item in status['build_material']:
        copy(sources / item['file'], 'sources/' + item['file'], item['sha256'])
    copy(application, 'sources/application/el-modem-public-source.tar.gz', verification['public_application_archive_sha256'])
    for name in ('source-status.json', 'source-plan.json'):
        copy(sources / name, 'sources/' + name)
    copy(sources / 'application-build/pip-versions.txt', 'sources/application-build/pip-versions.txt')
    (output / 'README.md').write_text(
        '# El-Modem Linux x86-64 review candidate\n\n'
        'This is a local review candidate, not a cleared public binary release. '
        'Original source-expression provenance and complete compiled dependency '
        'license/source coverage remain under review.\n\n'
        'The executable is the qualified frozen artifact, unchanged. Its SHA-256 is `' + artifact + '`. '
        'Use `bin/el-modem --self-test`, then `bin/el-modem` for the TUI. '
        'One modem starts by default; `--modems N` selects more. '
        'See docs/DESKTOP.md and docs/SERIAL.md. Full Linux modem signals require '
        'the separately installed serial prerequisite; its pinned build helper and patch '
        'are in packaging/serial. PTY/TCP are explicit alternatives with fewer signals.\n\n'
        'This artifact declares GLIBC versions through 2.36. The bootloader requires '
        'host libc, libdl, libpthread and libz before extracting bundled libraries. '
        'The bundled Pillow extension additionally declares host libxcb.so.1; '
        'not every optional path is tested without it. Debian Bookworm self-test passes.\n\n'
        'The binary has local independent-peer functional/TUI/sustained qualification; '
        'loaded hosts, larger drift, RTP loss and recovery repeatability have remaining '
        'limits. No universal interoperability claim is made.\n\n'
        'Original community terms are in LICENSE/NOTICE. Third-party notices include '
        'installed-package records and custom Python/compiled-source supplements. '
        'Some archive-wide notices describe unused components. Third-party terms '
        'are not relicensed by any commercial grant for original code.\n\n'
        'sources contains exact dependency archives and public application source '
        '(no private Git history), custom Python build records and public build recipes. '
        'The application source commit is `' + verification['public_application_source_commit'] + '`. '
        'Source collection and checksums do not certify full corresponding-source '
        'compliance or bit-for-bit reproducibility. Do not publish this candidate '
        'while the remaining review is open.\n')
    files = [dict(file=str(path.relative_to(output)), size=path.stat().st_size, sha256=sha(path))
             for path in sorted(output.rglob('*')) if path.is_file()]
    report = dict(schema=1, status='LOCAL_REVIEW_CANDIDATE', artifact_sha256=artifact,
                  public_application_commit=verification['public_application_source_commit'],
                  source_archives=len(status['sources']), notice_files=len(measured['notices']) + supplement_count,
                  files=files, publication_authorized=False)
    (output / 'MANIFEST.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('executable', 'notices', 'sources', 'output'):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    result = assemble(args.executable, args.notices, args.sources, args.output)
    print(json.dumps(dict(status=result['status'], files=len(result['files']),
                          source_archives=result['source_archives'], notice_files=result['notice_files']), indent=2))
