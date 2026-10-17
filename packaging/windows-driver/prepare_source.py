# SPDX-License-Identifier: MS-PL
# Copyright (c) 2026 Don Fanning
"""Reconstruct the reviewed MS-PL UMDF2 overlay on pinned Microsoft source.

Fetches only the source files in the manifest, never a vendor tree, certificate,
binary or signing key. Creates a new destination and verifies every result.
This separate driver component and reconstruction helper are distributed under
the Microsoft Public License; see LICENSE.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import urllib.request


def normalized(data):
    encoding='utf-16' if data.startswith((b'\xff\xfe',b'\xfe\xff')) else 'utf-8-sig'
    text=data.decode(encoding).replace('\r\n','\n')
    return (text if text.endswith('\n') else text+'\n').encode()


def prepare(destination, upstream=None):
    here=Path(__file__).resolve().parent
    manifest=json.loads((here/'source-manifest.json').read_text())
    if destination.exists():
        raise ValueError('Fresh destination required; existing work is never replaced')
    destination.mkdir(parents=True)
    for name,expected in manifest['upstream_files'].items():
        if upstream is None:
            url=('https://raw.githubusercontent.com/microsoft/Windows-driver-samples/'
                 +manifest['revision']+'/'+manifest['subdirectory']+'/'+name)
            with urllib.request.urlopen(url,timeout=30) as response:data=response.read()
        else:
            data=(upstream/name).read_bytes()
        content=normalized(data)
        if hashlib.sha256(content).hexdigest()!=expected:
            raise ValueError('Upstream source mismatch: '+name)
        target=destination/name;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(content)
    patch=here/'overlay.patch'
    subprocess.run(['git','apply','--check',str(patch)],cwd=destination,check=True)
    subprocess.run(['git','apply',str(patch)],cwd=destination,check=True)
    for name,expected in manifest['result_files'].items():
        if hashlib.sha256((destination/name).read_bytes()).hexdigest()!=expected:
            raise ValueError('Overlay result mismatch: '+name)
    license=here/'LICENSE'
    if hashlib.sha256(license.read_bytes()).hexdigest()!=manifest['license_sha256']:
        raise ValueError('License identity mismatch')
    shutil.copyfile(license,destination/'LICENSE')
    return dict(status='PASS',upstream_revision=manifest['revision'],
                verified_source_files=len(manifest['result_files']))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination',type=Path)
    parser.add_argument('--upstream',type=Path,help='Offline source witness for verification')
    args=parser.parse_args()
    print(json.dumps(prepare(args.destination.resolve(),args.upstream),indent=2))
