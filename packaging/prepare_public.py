"""Export a committed, allowlisted source preview without private Git history.

Never pushes, initializes Git, copies untracked files or changes the source tree.
Findings report locations only, never matched credential values. This targeted
review is not a substitute for a full secret scanner or provenance review.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = {
    'README.md': 'packaging/public/README.md',
    'docs/REDISTRIBUTION.md': 'packaging/public/REDISTRIBUTION.md',
    '.github/workflows/build.yml': 'packaging/public/build.yml',
}


def selected(path: str) -> bool:
    p = PurePosixPath(path)
    if path in {'lab/asterisk/audio-peer/analyze_second_leg.py','packaging/Dockerfile.native-rust'}:
        return False  # Private scratch helper, not reached by the application.
    if path == 'packaging/serial/COPYING':return True
    if path in {'CREDITS.md', 'LICENSE', 'NOTICE', 'CONTRIBUTING.md', '.gitattributes', 'docs/AT_COMMANDS.md',
                'docs/DESKTOP.md', 'docs/SERIAL.md'}:
        return True
    if p.parent in {PurePosixPath('lab/asterisk'),
                    PurePosixPath('lab/asterisk/audio-peer')}:
        return p.suffix == '.py'
    if p.parent == PurePosixPath('packaging'):
        return (p.suffix in {'.py', '.sh', '.spec'} or p.name.startswith('Dockerfile'))
    return (str(p.parent) == 'packaging/serial' and
            p.suffix in {'.md', '.sh', '.patch'})


def findings(path: str, data: bytes) -> list[dict]:
    text = data.decode('utf-8')
    rules = {
        'private-key': r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',
        'github-token': r'\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})\b',
        'aws-access-key': r'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b',
        'credential-url': r'\b[a-z][a-z0-9+.-]*://[^\s/]+:[^\s/@]+@',
        'private-workspace-path': r'/home/jfalcon\b',
        'phone-number-review': r'(?<!\w)(?:\+[1-9][0-9]{9,14}|[0-9]{3}[- ][0-9]{3}[- ][0-9]{4})(?!\w)',
        'literal-credential-review': r'''(?i)\b(?:password|passwd|secret|auth_pass|access_token)\s*=\s*['"][^'"\r\n]+['"]''',
    }
    return [dict(path=path, line=text.count('\n', 0, m.start()) + 1, rule=rule)
            for rule, pattern in rules.items() for m in re.finditer(pattern, text)]


def git(*args: str) -> bytes:
    return subprocess.check_output(['git', '-C', str(ROOT), *args])


def export(destination: Path, revision: str = 'HEAD') -> dict:
    destination = destination.resolve()
    if destination == ROOT or ROOT in destination.parents:
        raise ValueError('export must be outside the private repository')
    if destination.exists():
        raise ValueError('destination must not exist; never overwrite an earlier review')
    commit = git('rev-parse', '--verify', revision + '^{commit}').decode().strip()
    entries = {}
    excluded = []
    for record in git('ls-tree', '-rz', '--full-tree', commit).split(b'\0'):
        if not record:
            continue
        metadata, raw_path = record.split(b'\t', 1)
        mode, kind, digest = metadata.decode().split()
        path = raw_path.decode()
        if selected(path) and path not in {'packaging/prepare_public.py', 'packaging/test_prepare_public.py'}:
            if kind != 'blob' or mode not in {'100644', '100755'}:
                raise ValueError(f'refusing non-regular source file: {path}')
            entries[path] = (git('cat-file', 'blob', digest), mode)
        else:
            excluded.append(path)
    for target, template in TEMPLATES.items():
        entries[target] = (git('show', f'{commit}:{template}'), '100644')
    entries['.gitignore'] = (b'__pycache__/\n*.py[cod]\n.venv/\n.env\n.env.*\n!*.example\ndist/\nbuild/\npackaging/dist/\n*.raw\n*.wav\n*.pcap*\n*.npz\n', '100644')
    # Public docs must not send readers to private handoffs or absent lab notes.
    docs, mode = entries['docs/DESKTOP.md']
    text = docs.decode().replace('See the current handoff for which packaged/live gates have actually run.',
                                'See the README for the source preview qualification limits.')
    text = text.replace('[the runtime qualification note](../lab/PYTHON_LTO_LIBPYTHON_2026_10_13.md)',
                        'the private development qualification record (not distributed)')
    entries['docs/DESKTOP.md'] = (text.encode(), mode)
    credits, mode = entries['CREDITS.md']
    entries['CREDITS.md'] = (credits + b'\n## Public source preview\n\nReference submodules, lab peers, provider configurations and all recordings\nare excluded from this export. The community license is GPL-3.0-or-later;\nprovenance review remains open. See NOTICE and docs/REDISTRIBUTION.md.\n', mode)
    issues = []
    hashes = {}
    for path, (data, mode) in entries.items():
        issues.extend(findings(path, data))
        if path.endswith('.py'):
            ast.parse(data, filename=path)
        hashes[path] = dict(sha256=hashlib.sha256(data).hexdigest(), mode=mode)
    report = dict(schema=1, source_commit=commit, files=hashes,
                  excluded_count=len(excluded), findings=issues,
                  status='REVIEW_REQUIRED',
                  blockers=['Original-code provenance review and commercial/contributor agreements pending',
                            'Targeted scan requires human review and independent secret scanning',
                            'Binary dependency notice inventory and sustained timing qualification pending'],
                  history_included=False, external_calls_placed=False)
    # Refuse recognized credential formats; review flags never silently disappear.
    hard = {'private-key', 'github-token', 'aws-access-key', 'credential-url'}
    if any(i['rule'] in hard for i in issues):
        raise ValueError('recognized credential pattern found; export aborted (values suppressed)')
    destination.mkdir(parents=True)
    for path, (data, mode) in entries.items():
        target = destination / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        target.chmod(0o755 if mode == '100755' else 0o644)
    (destination / 'PUBLIC_EXPORT_MANIFEST.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination', type=Path)
    parser.add_argument('--revision', default='HEAD')
    args = parser.parse_args()
    report = export(args.destination, args.revision)
    print(json.dumps({key: report[key] for key in ('source_commit', 'status', 'excluded_count', 'findings', 'blockers')}, indent=2))


if __name__ == '__main__':
    main()
