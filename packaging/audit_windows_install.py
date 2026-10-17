"""Compare an installed Windows payload to its exact reviewed ZIP and driver.

No installer execution or publication. Reject untracked files, changed bytes,
symlinks, and missing package contents before a build is retained. Inno's two
generated uninstall files are the only automatic exclusions.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile

DRIVER_FILES = ('virtualserial2um.inf', 'VirtualSerial2um.dll', 'wudf.cat',
                'ElModemTestSigning.cer')
DRIVER_SCRIPTS = ('install_driver.ps1', 'uninstall_driver.ps1')


def audit(archive, installed, driver=None, scripts=None):
    expected = {}
    with zipfile.ZipFile(archive) as package:
        roots = set()
        for item in package.infolist():
            if item.is_dir():
                continue
            name = PurePosixPath(item.filename)
            if name.is_absolute() or len(name.parts) < 2 or '..' in name.parts:
                raise ValueError('Unsafe package member')
            roots.add(name.parts[0])
            key = '/'.join(name.parts[1:]).casefold()
            if key in expected:
                raise ValueError('Duplicate package member')
            expected[key] = hashlib.sha256(package.read(item)).hexdigest()
        if len(roots) != 1:
            raise ValueError('Expected exactly one package root')
    if driver is not None:
        for name in DRIVER_FILES:
            expected['driver/'+name.casefold()] = hashlib.sha256((driver/name).read_bytes()).hexdigest()
        for name in DRIVER_SCRIPTS:
            expected['driver/'+name] = hashlib.sha256((scripts/name).read_bytes()).hexdigest()
    actual = {}
    for path in installed.rglob('*'):
        if path.is_symlink():
            raise ValueError('Symlink in installed payload')
        if not path.is_file():
            continue
        key = path.relative_to(installed).as_posix().casefold()
        if key in ('unins000.exe', 'unins000.dat'):
            continue
        if key in actual:
            raise ValueError('Duplicate installed member')
        actual[key] = hashlib.sha256(path.read_bytes()).hexdigest()
    missing = sorted(expected.keys()-actual.keys())
    extra = sorted(actual.keys()-expected.keys())
    changed = sorted(k for k in expected.keys() & actual.keys() if expected[k] != actual[k])
    return dict(valid=not (missing or extra or changed), expected_files=len(expected),
                actual_files=len(actual), missing=missing, extra=extra, changed=changed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive',type=Path)
    parser.add_argument('installed',type=Path)
    parser.add_argument('--driver-package',type=Path)
    parser.add_argument('--scripts',type=Path,default=Path(__file__).parent/'windows')
    args = parser.parse_args()
    report = audit(args.archive,args.installed,args.driver_package,args.scripts)
    print(json.dumps(report,indent=2))
    if not report['valid']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
