"""Inventory a frozen executable and gather notices from its matched builder.

Run using the qualified builder's Python/PyInstaller environment, offline.
This records evidence and missing coverage; it does not certify redistribution.
The executable is read-only and is never rebuilt or modified.
"""
import argparse
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import types

from PyInstaller.archive.readers import CArchiveReader


def digest(data):
    return hashlib.sha256(data).hexdigest()


def normalized(code):
    return code.replace(co_filename='', co_consts=tuple(
        normalized(value) if isinstance(value, types.CodeType) else value
        for value in code.co_consts))


def inventory(executable, output):
    output.mkdir(parents=True, exist_ok=False)
    notices = output / 'licenses'
    notices.mkdir()
    archive = CArchiveReader(str(executable))
    pyz = archive.open_embedded_archive('PYZ.pyz')
    manifest = json.loads(archive.extract('el-modem-build.json'))
    original = {Path(name).stem for name in manifest['local_modules']}
    package_map = metadata.packages_distributions()
    roots = sorted({name.split('.')[0] for name in pyz.toc})
    report = dict(schema=1, artifact_sha256=digest(executable.read_bytes()),
                  python=sys.version, build_manifest=manifest, archive_files=[],
                  python_distributions=[], native_libraries=[], blockers=[])
    notices_record = []
    debian_packages = set()

    def copy_notice(path, destination, component):
        if not path.is_file():
            return False
        target = notices / destination
        target.parent.mkdir(parents=True, exist_ok=True)
        data = path.read_bytes()
        target.write_bytes(data)
        notices_record.append(dict(component=component, file=str(target.relative_to(output)),
                                   source=str(path), sha256=digest(data)))
        return True

    def owners(path):
        query = subprocess.run(['dpkg-query', '-S', str(path)], capture_output=True, text=True)
        found = set()
        for line in query.stdout.splitlines():
            if ': /' in line:
                for item in line.split(': /', 1)[0].split(', '):
                    found.add(item)
        debian_packages.update(found)
        return sorted(found)

    # Exact builder files are preferred. PyInstaller may rewrite ELF search
    # paths; in that case retain the mismatch rather than claiming exact bytes.
    index = {}
    for root in ['/opt/el-modem-libpython', '/usr/lib', '/lib', '/opt/el-modem-build']:
        for directory, _, files in os.walk(root):
            for name in files:
                if '.so' in name:
                    index.setdefault(name, []).append(Path(directory) / name)

    for name, entry in archive.toc.items():
        if entry[-1] == 'o':
            continue
        data = archive.extract(name)
        report['archive_files'].append(dict(name=name, type=entry[-1], bytes=len(data), sha256=digest(data)))
        if name == 'licenses/LICENSE.python':
            target = notices / 'python/LICENSE.python'
            target.parent.mkdir();target.write_bytes(data)
            notices_record.append(dict(component='CPython',file=str(target.relative_to(output)),
                                       source='embedded executable',sha256=digest(data)))
        if '.so' not in name:
            continue
        candidates = list(dict.fromkeys(index.get(Path(name).name, [])))
        matches = [p for p in candidates if digest(p.read_bytes()) == digest(data)]
        record = dict(name=name, sha256=digest(data), exact_builder_matches=[str(p) for p in matches], packages=[])
        for path in matches:
            record['packages'].extend(owners(path))
        record['packages'] = sorted(set(record['packages']))
        if name == 'libpython3.11.so.1.0' and matches:
            record['custom_build'] = manifest.get('python_library')
            # Includes Debian changes and their notices in addition to CPython.
            owners(Path('/usr/lib/x86_64-linux-gnu/libpython3.11.so.1.0'))
        if not matches:
            report['blockers'].append('native library lacks exact builder match: '+name)
        elif not record['packages'] and not record.get('custom_build'):
            report['blockers'].append('native library needs wheel/source attribution: '+name)
        report['native_libraries'].append(record)

    # Metadata-only dependencies also carry copyright-bearing material, even
    # when no code from that distribution is in PYZ (for example wheel).
    metadata_names = {}
    for dist in metadata.distributions():
        prefix = Path(dist._path).name + '/'
        if any(n.startswith(prefix) for n in archive.toc):
            name = dist.metadata['Name']
            metadata_names['@metadata:'+name] = name
    visited = set()
    for root in roots + sorted(metadata_names):
        if root in original:
            continue
        names = [metadata_names[root]] if root in metadata_names else sorted(set(package_map.get(root, [])))
        if not names:
            if root not in sys.stdlib_module_names and root not in ('sitecustomize','_sysconfigdata__x86_64-linux-gnu','_aix_support','_bootsubprocess','_osx_support'):
                report['blockers'].append('Python root lacks distribution/stdlib attribution: '+root)
            continue
        for name in names:
            if name in visited:
                continue
            visited.add(name)
            dist = metadata.distribution(name)
            record = dict(name=dist.metadata['Name'], version=dist.version,
                          declared_license=dist.metadata.get('License'), license_expression=dist.metadata.get('License-Expression'),
                          roots=[r for r in roots if name in package_map.get(r, [])],
                          bundled_modules=0, source_matched_modules=0, source_unmatched_modules=[], source_examples=[], notices=[])
            for module in pyz.toc:
                if module.split('.')[0] not in record['roots']:
                    continue
                code = pyz.extract(module)
                if not isinstance(code, types.CodeType):
                    continue
                record['bundled_modules'] += 1
                relative = module.replace('.', '/')
                candidates = [Path(base)/(relative+'.py') for base in sys.path if base]
                candidates += [Path(base)/relative/'__init__.py' for base in sys.path if base]
                match = False
                for path in candidates:
                    if not path.is_file():
                        continue
                    for opt in (0,1,2):
                        try:
                            other=compile(path.read_bytes(),str(path),'exec',optimize=opt)
                        except (SyntaxError,ValueError):
                            continue
                        if normalized(other)==normalized(code):
                            match=True;break
                    if match:
                        break
                if match:
                    record['source_matched_modules'] += 1
                    if not record['source_examples']:
                        record['source_examples'].append(dict(module=module,path=str(path),sha256=digest(path.read_bytes())))
                else:record['source_unmatched_modules'].append(module)
            record['bundled_metadata'] = []
            prefix=Path(dist._path).name+'/'
            for archive_name in archive.toc:
                if archive_name.startswith(prefix):
                    path=Path(dist._path)/archive_name[len(prefix):]
                    identical=path.is_file() and digest(path.read_bytes())==digest(archive.extract(archive_name))
                    record['bundled_metadata'].append(dict(name=archive_name,exact_builder_match=identical))
                    if not identical:report['blockers'].append('metadata does not match builder: '+archive_name)
            for file in dist.files or []:
                path = Path(dist.locate_file(file))
                if re.search(r'(^|[/_.-])(licen[cs]es?|copying|notices?|copyrights?)([/_.-]|$)', str(file), re.I) and path.is_file() and path.suffix not in ('.py','.pyc'):
                    dest = Path('python-packages') / re.sub(r'[^a-zA-Z0-9_.-]', '_', name) / path.name
                    if copy_notice(path,dest,name):record['notices'].append(str(Path('licenses')/dest))
            if record['source_examples']:
                record['debian_packages']=owners(Path(record['source_examples'][0]['path']))
                for package in record['debian_packages']:
                    record['notices'].append(str(Path('licenses/debian')/package.split(':')[0]/'copyright'))
            if not record['source_examples'] and record['bundled_metadata']:
                record['debian_packages']=owners(Path(dist._path))
                for package in record['debian_packages']:
                    record['notices'].append(str(Path('licenses/debian')/package.split(':')[0]/'copyright'))
            if not record['notices'] and not record.get('debian_packages'):
                report['blockers'].append('no installed license notice found: '+name)
            if record['source_unmatched_modules']:
                report['blockers'].append('Python source matching incomplete: '+name)
            report['python_distributions'].append(record)

    # Tool bootloader/loader notice, distinct from dependencies of the program.
    dist = metadata.distribution('pyinstaller')
    report['packaging_tool'] = dict(name='PyInstaller',version=dist.version)
    for file in dist.files or []:
        if re.search(r'(copying|license|notice)',str(file),re.I):
            path=Path(dist.locate_file(file))
            if path.is_file():copy_notice(path,Path('pyinstaller')/path.name,'PyInstaller')
    for package in sorted(debian_packages):
        short=package.split(':')[0]
        path=Path('/usr/share/doc')/short/'copyright'
        if not copy_notice(path,Path('debian')/short/'copyright',package):
            report['blockers'].append('Debian copyright unavailable: '+package)
    for path in Path('/usr/share/common-licenses').glob('*'):
        copy_notice(path,Path('debian/common-licenses')/path.name,'Debian common license texts')
    report['debian_packages']=[]
    for package in sorted(debian_packages):
        fields=subprocess.check_output(['dpkg-query','-W','-f=${Version}\t${source:Package}\t${source:Version}',package],text=True).split('\t')
        report['debian_packages'].append(dict(name=package,version=fields[0],
            source_package=fields[1],source_version=fields[2]))
    report['notices']=notices_record
    for notice in notices_record:
        if digest((output/notice['file']).read_bytes())!=notice['sha256']:
            report['blockers'].append('notice destination collision: '+notice['file'])
    for distribution in report['python_distributions']:
        for name in distribution['notices']:
            if not (output/name).is_file():
                report['blockers'].append('distribution notice missing: '+name)
    source_catalog=sorted({(p['source_package'],p['source_version']) for p in report['debian_packages']})
    (output/'debian-source-catalog.json').write_text(json.dumps([
        dict(package=p,version=v,status='exact installed source version; source archive not collected')
        for p,v in source_catalog],indent=2)+'\n')
    header=['# Third-party notice inventory', '', 'Artifact SHA-256: `'+report['artifact_sha256']+'`.', '',
        'Review required. This lists actual bundled components; it does not grant',
        'commercial redistribution rights or certify corresponding-source compliance.', '',
        '## Python distributions', '', '| Distribution | Version | Source modules matched | Notices |',
        '| --- | --- | --- | --- |']
    for d in report['python_distributions']:
        links=', '.join('['+Path(n).name+']('+n+')' for n in d['notices'])
        header.append(f"| {d['name']} | {d['version']} | {d['source_matched_modules']}/{d['bundled_modules']} | {links} |")
    header += ['', '## Debian native and Python components', '',
        '| Installed package | Version | Source package/version | Notice |', '| --- | --- | --- | --- |']
    for d in report['debian_packages']:
        n='licenses/debian/'+d['name'].split(':')[0]+'/copyright'
        header.append(f"| {d['name']} | {d['version']} | {d['source_package']} {d['source_version']} | [copyright]({n}) |")
    header += ['', 'Common license texts referenced by Debian notices are in',
        '[licenses/debian/common-licenses](licenses/debian/common-licenses).', '',
        'Readline and libimagequant include GPL terms; retain and review those',
        'obligations separately from original El-Modem commercial licensing.', '',
        'Before release: review static/transitive code, host libraries, original',
        'code provenance, and corresponding-source obligations. The source catalog',
        'records exact Debian source versions but is not the corresponding source.', '']
    (output/'THIRD_PARTY_NOTICES.md').write_text('\n'.join(header))
    report['status']='REVIEW_REQUIRED'
    report['blockers']=sorted(set(report['blockers']))
    report['limits']=[
        'Notice collection is not a complete license compatibility or corresponding-source audit.',
        'Native matching uses exact extracted bytes; unmatched entries remain blockers.',
        'Only code present in the frozen archive selects Python distributions; builder extras are excluded.',
        'Host-provided runtime libraries and embedded static/transitive code need separate review.',
        'No executable modifications or public upload occurred.']
    (output/'inventory.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('executable',type=Path)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    report=inventory(args.executable,args.output)
    print(json.dumps({k:len(report[k]) for k in ('archive_files','python_distributions','native_libraries','notices')},indent=2))
    print(json.dumps(report['blockers'],indent=2))
    if report['blockers']:
        raise SystemExit(1)


if __name__=='__main__':main()
