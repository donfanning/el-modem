"""Retain licenses for exact Rust crates linked into the native IAX extension.

Run after the locked Cargo build, while registry sources are still available.
The output records Cargo's resolved versions and license expressions and copies
all root license/notice files shipped by each resolved crate. It is evidence
collection, not a license-compatibility or provenance determination.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess


NOTICE_NAME = re.compile(r'^(?:LICEN[CS]E|COPYING|NOTICE|COPYRIGHT|PATENTS?)(?:[._ -].*)?$', re.I)


def crate_notices(crate_dir: Path) -> list[Path]:
    return sorted(p for p in crate_dir.iterdir()
                  if p.is_file() and NOTICE_NAME.match(p.name))


def inventory(manifest: Path, output: Path) -> dict:
    manifest = manifest.resolve()
    output.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        ['cargo', 'metadata', '--locked', '--format-version', '1',
         '--manifest-path', str(manifest)],
        check=True, capture_output=True, text=True)
    metadata = json.loads(completed.stdout)
    workspace_root = Path(metadata['workspace_root']).resolve()
    crates_dir = output / 'licenses'
    packages = []
    missing = []
    for package in sorted(metadata['packages'], key=lambda row: (row['name'].lower(), row['version'])):
        if Path(package['manifest_path']).resolve() == manifest:
            continue
        crate_dir = Path(package['manifest_path']).resolve().parent
        files = crate_notices(crate_dir)
        name = re.sub(r'[^A-Za-z0-9_.-]', '_', package['name'])
        version = re.sub(r'[^A-Za-z0-9_.-]', '_', package['version'])
        retained = []
        for source in files:
            relative = Path(name + '-' + version) / source.name
            destination = crates_dir / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            data = source.read_bytes()
            destination.write_bytes(data)
            retained.append({
                'path': str(Path('licenses') / relative),
                'sha256': hashlib.sha256(data).hexdigest(),
                'source_name': source.name,
            })
        row = {
            'name': package['name'],
            'version': package['version'],
            'license_expression': package.get('license'),
            'repository': package.get('repository'),
            'notice_files': retained,
        }
        packages.append(row)
        if not files:
            missing.append(f"{package['name']} {package['version']}: no root license/notice file found")

    report = {
        'schema': 1,
        'status': 'COLLECTED_REVIEW_REQUIRED' if not missing else 'INCOMPLETE',
        'cargo_lock_sha256': hashlib.sha256((workspace_root / 'Cargo.lock').read_bytes()).hexdigest(),
        'manifest': str(manifest.relative_to(workspace_root)),
        'packages': packages,
        'missing_notice_files': missing,
        'limits': [
            'Cargo license expressions and retained crate notices do not establish source authorship, license compatibility, or permission.',
            'The resolved package graph includes build and target-specific dependencies; some listed packages may not contribute code to a particular target binary.',
            'Static native-code composition and corresponding-source obligations require separate review.',
        ],
    }
    (output / 'inventory.json').write_text(json.dumps(report, indent=2) + '\n')
    lines = ['# Rust dependency notices', '',
             'Crate versions and license expressions are from the locked Cargo metadata. '
             'The retained license files are the files shipped with those exact crate sources.', '',]
    for package in packages:
        lines.append(f"- **{package['name']} {package['version']}** — "
                     f"{package['license_expression'] or 'license expression not declared'}")
        for notice in package['notice_files']:
            lines.append(f"  - [{Path(notice['path']).name}]({notice['path']})")
    if missing:
        lines.extend(['', 'Missing notices:', *[f'- {item}' for item in missing]])
    lines.extend(['', 'This inventory is not a license-compatibility or provenance determination.'])
    (output / 'THIRD_PARTY_NOTICES.md').write_text('\n'.join(lines) + '\n')
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    report = inventory(args.manifest, args.output)
    print(json.dumps({
        'status': report['status'],
        'packages': len(report['packages']),
        'missing_notice_files': report['missing_notice_files'],
    }))
    if report['missing_notice_files']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
