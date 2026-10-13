"""Record ELF link requirements for exact inventoried builder files.

Run in the matched builder with readelf. Never load or execute inventoried ELFs.
DT_NEEDED closure does not enumerate static code or runtime dlopen dependencies.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess


def inspect(path):
    result = subprocess.run(['readelf', '-dW', str(path)], check=True,
                            capture_output=True, text=True)
    versions = subprocess.run(['readelf', '-VW', str(path)], check=True,
                              capture_output=True, text=True)
    return dict(needed=re.findall(r'\(NEEDED\).*?\[(.*?)\]', result.stdout),
                soname=re.findall(r'\(SONAME\).*?\[(.*?)\]', result.stdout),
                glibc_versions=sorted(set(re.findall(r'\bGLIBC_(\d+(?:\.\d+)+)\b', versions.stdout)),
                                      key=lambda value: tuple(map(int, value.split('.')))))


def audit(inventory, executable):
    measured = json.loads(inventory.read_text())
    libraries = []
    for record in measured['native_libraries']:
        path = Path(record['exact_builder_matches'][0])
        if hashlib.sha256(path.read_bytes()).hexdigest() != record['sha256']:
            raise ValueError('builder file changed: ' + str(path))
        libraries.append(dict(name=record['name'], sha256=record['sha256'],
                              builder_file=str(path), **inspect(path)))
    if hashlib.sha256(executable.read_bytes()).hexdigest() != measured['artifact_sha256']:
        raise ValueError('executable differs from inventory')
    bootloader = inspect(executable)
    bundled = {Path(item['name']).name for item in libraries}
    bundled.update(name for item in libraries for name in item['soname'])
    required = {name for item in libraries for name in item['needed']}
    required.update(bootloader['needed'])
    external = sorted(required - bundled)
    glibc_versions = {version for item in libraries for version in item['glibc_versions']}
    glibc_versions.update(bootloader['glibc_versions'])
    return dict(artifact_sha256=measured['artifact_sha256'], libraries=libraries,
                bootloader=bootloader, bundled_sonames=sorted(bundled),
                unbundled_dt_needed=external,
                bootloader_host_needed=bootloader['needed'],
                highest_declared_glibc_version=max(glibc_versions, key=lambda value: tuple(map(int, value.split('.')))),
                limits=['ELF names establish declared dynamic dependencies only.',
                        'Bootloader dependencies must exist on the host before bundled files are extracted.',
                        'Static code and dlopen dependencies require source/build review.',
                        'Host dependency licensing and System Library status are not certified.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inventory', type=Path)
    parser.add_argument('executable', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    result = audit(args.inventory, args.executable)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(libraries=len(result['libraries']),
                          unbundled_dt_needed=result['unbundled_dt_needed'],
                          bootloader_host_needed=result['bootloader_host_needed'],
                          highest_declared_glibc_version=result['highest_declared_glibc_version']), indent=2))
