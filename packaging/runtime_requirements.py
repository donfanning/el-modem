"""Read native requirements from a frozen executable; never execute its code.

Deployment/symbol floors are necessary constraints, not oldest-OS qualification.
Requires PyInstaller and GNU readelf/objdump on the inspection host.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import subprocess
import tempfile

TARGET_CPU = {'macos-x86_64': 0x1000007, 'macos-arm64': 0x100000C}


def version(value):
    return ((value >> 16) & 65535, (value >> 8) & 255, value & 255)


def version_string(value):
    return '.'.join(map(str, value))


def native_format(data):
    if data.startswith(b'\x7fELF'):
        return 'ELF'
    if data[:4] in (b'\xcf\xfa\xed\xfe', b'\xfe\xed\xfa\xcf',
                    b'\xce\xfa\xed\xfe', b'\xfe\xed\xfa\xce',
                    b'\xca\xfe\xba\xbe', b'\xca\xfe\xba\xbf',
                    b'\xbe\xba\xfe\xca', b'\xbf\xba\xfe\xca'):
        return 'Mach-O'
    if len(data) >= 64 and data[:2] == b'MZ':
        offset = struct.unpack_from('<I', data, 60)[0]
        if data[offset:offset + 4] == b'PE\0\0':
            return 'PE'
    return None


def macho_slices(data):
    magic = data[:4]
    fat = {b'\xca\xfe\xba\xbe': ('>', False), b'\xca\xfe\xba\xbf': ('>', True),
           b'\xbe\xba\xfe\xca': ('<', False), b'\xbf\xba\xfe\xca': ('<', True)}
    if magic in fat:
        endian, wide = fat[magic]
        count = struct.unpack_from(endian + 'I', data, 4)[0]
        if not 0 < count <= 128:
            raise ValueError('invalid fat architecture count')
        rows = []
        for index in range(count):
            fmt, width = ('IIQQII', 32) if wide else ('IIIII', 20)
            fields = struct.unpack_from(endian + fmt, data, 8 + index * width)
            cpu, _, offset, size = fields[:4]
            if size < 28 or offset + size > len(data):
                raise ValueError('fat slice outside file')
            slices = macho_slices(data[offset:offset + size])
            if len(slices) != 1 or slices[0]['cpu'] != cpu:
                raise ValueError('fat architecture differs from thin header')
            rows.extend(slices)
        return rows
    endian = '<' if magic in (b'\xcf\xfa\xed\xfe', b'\xce\xfa\xed\xfe') else '>'
    if magic not in (b'\xcf\xfa\xed\xfe', b'\xfe\xed\xfa\xcf',
                     b'\xce\xfa\xed\xfe', b'\xfe\xed\xfa\xce'):
        raise ValueError('not a Mach-O header')
    header = 32 if magic in (b'\xcf\xfa\xed\xfe', b'\xfe\xed\xfa\xcf') else 28
    cpu = struct.unpack_from(endian + 'I', data, 4)[0]
    count, length = struct.unpack_from(endian + 'II', data, 16)
    limit = header + length
    if limit > len(data):
        raise ValueError('load commands outside file')
    row = dict(cpu=cpu, minimums=[], dependencies=[], code_signature_command=False)
    position = header
    for _ in range(count):
        command, size = struct.unpack_from(endian + 'II', data, position)
        if size < 8 or position + size > limit:
            raise ValueError('invalid load command extent')
        if command == 0x32:
            if size < 24:
                raise ValueError('short LC_BUILD_VERSION')
            platform, minimum, sdk, _ = struct.unpack_from(endian + 'IIII', data, position + 8)
            row['minimums'].append(dict(command='LC_BUILD_VERSION', platform=platform,
                                        minimum=version(minimum), sdk=version(sdk)))
        elif command == 0x24:
            if size < 16:
                raise ValueError('short LC_VERSION_MIN_MACOSX')
            minimum, sdk = struct.unpack_from(endian + 'II', data, position + 8)
            row['minimums'].append(dict(command='LC_VERSION_MIN_MACOSX', platform=1,
                                        minimum=version(minimum), sdk=version(sdk)))
        elif command in (0xC, 0x80000018, 0x8000001F, 0x80000023):
            offset = struct.unpack_from(endian + 'I', data, position + 8)[0]
            if not 8 <= offset < size:
                raise ValueError('invalid dylib name offset')
            row['dependencies'].append(data[position + offset:position + size].split(b'\0')[0].decode())
        elif command == 0x1D:
            row['code_signature_command'] = True  # Not a Developer ID/notarization verdict.
        position += size
    if position != limit:
        raise ValueError('load command sizes disagree')
    return [row]


def needed_versions(text):
    """Ignore exported version definitions and symbol-table annotations."""
    active, names = False, set()
    for line in text.splitlines():
        if line.startswith('Version '):
            active = line.startswith('Version needs section')
        if active:
            names.update(re.findall(r'Name: (\S+)', line))
    return sorted(names)


def floor(records, target):
    candidates = []
    for row in records:
        if target == 'linux-x86_64':
            candidates.extend((tuple(map(int, name[6:].split('.'))), row['name'])
                              for name in row.get('version_requirements', [])
                              if re.fullmatch(r'GLIBC_\d+(?:\.\d+)+', name))
        elif target in TARGET_CPU:
            candidates.extend((tuple(item['minimum']), row['name'])
                              for section in row.get('macho', []) if section['cpu'] == TARGET_CPU[target]
                              for item in section['minimums'] if item['platform'] == 1)
    if not candidates:
        return None
    maximum = max(value for value, _ in candidates)
    return dict(version=version_string(maximum),
                components=sorted({name for value, name in candidates if value == maximum}),
                kind='GLIBC symbol requirement' if target.startswith('linux') else 'Mach-O deployment target',
                sufficient_for_all_OS_features=False)


def inspect(executable, target):
    from PyInstaller.archive.readers import CArchiveReader
    executable = Path(executable)
    bootloader = executable.read_bytes()
    archive = CArchiveReader(str(executable))
    blobs = [('bootloader', bootloader)]
    for name, entry in archive.toc.items():
        # Symbolic aliases are strings, not native bytes; never overwrite their targets.
        if entry[-1] != 'b':
            continue
        data = archive.extract(name)
        if native_format(data):
            blobs.append((name, data))
    records = []
    with tempfile.TemporaryDirectory() as temporary:
        for index, (name, data) in enumerate(blobs):
            kind = native_format(data)
            row = dict(name=name, format=kind, sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))
            path = Path(temporary) / str(index)  # Never trust archive names as filesystem paths.
            path.write_bytes(data)
            if kind == 'ELF':
                row['version_requirements'] = needed_versions(subprocess.check_output(
                    ['readelf', '--version-info', str(path)], text=True, env=dict(os.environ, LC_ALL='C')))
                text = subprocess.check_output(['readelf', '--dynamic', str(path)], text=True,
                                               env=dict(os.environ, LC_ALL='C'))
                row['needed'] = re.findall(r'\(NEEDED\).*?\[(.*?)\]', text)
                row['machine'] = struct.unpack_from(('<' if data[5] == 1 else '>') + 'H', data, 18)[0]
            elif kind == 'PE':
                offset = struct.unpack_from('<I', data, 60)[0]
                row['machine'] = struct.unpack_from('<H', data, offset + 4)[0]
                text = subprocess.check_output(['objdump', '-p', str(path)], text=True,
                                               env=dict(os.environ, LC_ALL='C'))
                row['needed'] = re.findall(r'DLL Name: (.*)', text)
            elif kind == 'Mach-O':
                row['macho'] = macho_slices(data)
            else:
                raise ValueError('unknown bootloader format')
            records.append(row)
    first = records[0]
    correct = (first['format'] == 'ELF' and first.get('machine') == 62 if target == 'linux-x86_64'
               else first['format'] == 'PE' and first.get('machine') == 0x8664 if target == 'windows-x86_64'
               else first['format'] == 'Mach-O' and
                    any(section['cpu'] == TARGET_CPU[target] for section in first.get('macho', [])))
    if not correct:
        raise ValueError('bootloader architecture/format differs from requested target')
    expected_format = 'ELF' if target.startswith('linux') else 'PE' if target.startswith('windows') else 'Mach-O'
    architecture_mismatches = []
    for row in records:
        matches = (row['format'] == expected_format and
                   (row.get('machine') == 62 if target.startswith('linux') else
                    row.get('machine') == 0x8664 if target.startswith('windows') else
                    any(section['cpu'] == TARGET_CPU[target] for section in row.get('macho', []))))
        if not matches:
            architecture_mismatches.append(row['name'])
    special = sorted({name for row in records for name in row.get('version_requirements', [])
                      if name.startswith('GLIBC_') and not re.fullmatch(r'GLIBC_\d+(?:\.\d+)+', name)})
    return dict(schema=1, target=target, executable_sha256=hashlib.sha256(bootloader).hexdigest(),
                native_count=len(records), floor=floor(records, target),
                architecture_mismatches=architecture_mismatches,
                special_glibc_requirements=special,
                external_msvc_dlls=sorted({name for row in records for name in row.get('needed', [])
                                          if re.search(r'(?i)^(?:msvcp|vcruntime|vcomp)', name)}),
                native=records, limits='Static necessary constraints; execution/terminal/driver qualification is separate.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('executable', type=Path)
    parser.add_argument('--target', required=True, choices=('linux-x86_64', 'windows-x86_64', *TARGET_CPU))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = inspect(args.executable, args.target)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: result[key] for key in ('target', 'executable_sha256', 'native_count', 'floor',
                                                 'special_glibc_requirements', 'external_msvc_dlls')}, indent=2))


if __name__ == '__main__':
    main()
