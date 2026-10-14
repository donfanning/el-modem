"""Refresh or verify the public repository's tracked source snapshot manifest.

This hashes regular tracked files only. It does not export a private repository,
add files to Git, push, or claim source/license clearance. The manifest excludes
itself to avoid a recursive hash. Run after explicitly staging reviewed additions.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

NAME = 'PUBLIC_EXPORT_MANIFEST.json'


def snapshot(root):
    entries = subprocess.check_output(['git', '-C', str(root), 'ls-files', '--stage', '-z'])
    files = {}
    for entry in entries.split(b'\0'):
        if not entry:
            continue
        metadata, raw_name = entry.split(b'\t', 1)
        mode, _, stage = metadata.decode().split()
        name = raw_name.decode()
        if name == NAME:
            continue
        path = root / name
        if stage != '0' or mode not in ('100644', '100755') or path.is_symlink() or not path.is_file():
            raise ValueError('regular, unconflicted tracked source required: ' + name)
        files[name] = dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(), mode=mode)
    return files


def check(root):
    recorded = json.loads((root / NAME).read_text())['files']
    actual = snapshot(root)
    missing = sorted(actual.keys() - recorded.keys())
    extra = sorted(recorded.keys() - actual.keys())
    changed = sorted(name for name in actual.keys() & recorded.keys() if actual[name] != recorded[name])
    return dict(valid=not (missing or extra or changed), tracked_files=len(actual),
                missing=missing, extra=extra, changed=changed)


def refresh(root):
    old = json.loads((root / NAME).read_text())
    baseline = old.get('baseline_private_source_commit', old.get('source_commit'))
    report = dict(schema=2, baseline_private_source_commit=baseline,
                  public_base_revision=subprocess.check_output(
                      ['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip(),
                  scope='Current regular tracked public files; manifest itself excluded. Base revisions are historical anchors, not claims that changed files were built at those revisions.',
                  files=snapshot(root),
                  historical_public_overlays=old.get('historical_public_overlays', old.get('public_overlays', [])))
    (root / NAME).write_text(json.dumps(report, indent=2) + '\n')
    return check(root)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('.'))
    parser.add_argument('--refresh', action='store_true', help='write hashes for explicitly tracked files')
    args = parser.parse_args()
    result = refresh(args.root) if args.refresh else check(args.root)
    print(json.dumps(result, indent=2))
    if not result['valid']:
        raise SystemExit(1)
