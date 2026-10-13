"""Collect actual frozen package notices on native builders; preserve review gaps.

No Debian assumptions, execution of extracted code or claim of license clearance.
Native bytes are matched to installed package files / interpreter libraries;
patched or unknown libraries remain explicitly unresolved.
"""
import argparse
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import re
import sys
import sysconfig
from PyInstaller.archive.readers import CArchiveReader


def sha(data):return hashlib.sha256(data).hexdigest()


def inventory(executable,output):
    output.mkdir(parents=True,exist_ok=False)
    archive=CArchiveReader(str(executable));pyz=archive.open_embedded_archive('PYZ.pyz')
    manifest=json.loads(archive.extract('el-modem-build.json'))
    original={Path(name).stem for name in manifest['local_modules']}
    roots={name.split('.')[0] for name in pyz.toc}-original
    packages=metadata.packages_distributions();selected=set();unattributed=[]
    for root in sorted(roots):
        names=packages.get(root,[])
        selected.update(names)
        if not names and root not in sys.stdlib_module_names and root not in ('sitecustomize','_sysconfigdata__linux_x86_64-linux-gnu','_sysconfigdata__x86_64-linux-gnu','_aix_support','_bootsubprocess','_osx_support'):
            unattributed.append(root)
    distributions=list(metadata.distributions())
    for dist in distributions:
        prefix=Path(dist._path).name+'/'
        if any(name.replace('\\','/').startswith(prefix) for name in archive.toc):selected.add(dist.metadata['Name'])
    records=[];notices=[];native_index={}
    def notice(path,component,relative):
        if not path.is_file():return
        destination=output/'licenses'/relative
        destination.parent.mkdir(parents=True,exist_ok=True);data=path.read_bytes()
        if destination.exists() and destination.read_bytes()!=data:raise ValueError('conflicting notice destination')
        destination.write_bytes(data);notices.append(dict(component=component,path=str(destination.relative_to(output)),sha256=sha(data)))
    for dist in distributions:
        name=dist.metadata['Name']
        if name not in selected and name.lower()!='pyinstaller':continue
        component=re.sub(r'[^A-Za-z0-9_.-]','_',name)
        paths=[]
        for file in dist.files or []:
            path=Path(dist.locate_file(file))
            if not path.is_file():continue
            if path.suffix.lower() in ('.so','.dll','.pyd','.dylib') or '.so.' in path.name:
                native_index.setdefault(path.name,[]).append((path,name))
            if re.search(r'(^|[/_.-])(licen[cs]e|copying|notice|copyright)([/_.-]|$)',str(file),re.I) and path.suffix not in ('.py','.pyc'):
                relative=Path(component)/Path(*[part for part in Path(file).parts if part not in ('..','.')])
                notice(path,name,relative);paths.append(str(Path('licenses')/relative))
        records.append(dict(name=name,version=dist.version,license_expression=dist.metadata.get('License-Expression'),declared_license=dist.metadata.get('License'),notices=paths))
    # Interpreter extension provenance is separate from statically linked
    # OpenSSL/libffi/zlib etc. Finding a CPython file does not clear its internals.
    interpreter=Path(sys.base_prefix)
    for directory in (interpreter/'DLLs',Path(sysconfig.get_path('stdlib'))/'lib-dynload',interpreter/'lib'):
        if directory.is_dir():
            for path in directory.glob('*'):
                if path.is_file():native_index.setdefault(path.name,[]).append((path,'CPython builder'))
    for path in (interpreter/'python311.dll',interpreter/'Python',interpreter/'Python.framework/Versions/3.11/Python'):
        if path.is_file():native_index.setdefault(path.name,[]).append((path,'CPython builder'))
    for index,path in enumerate((interpreter/'LICENSE.txt',interpreter/'LICENSE',Path(sysconfig.get_path('stdlib'))/'LICENSE.txt',interpreter/'share/doc/python3.11/LICENSE')):
        if path.is_file():notice(path,'CPython',Path('CPython')/('LICENSE-'+str(index)+'.txt'))
    libraries=[]
    for name,entry in archive.toc.items():
        if entry[-1]!='b':continue
        data=archive.extract(name);digest=sha(data);matches=[]
        for path,component in native_index.get(Path(name).name,[]):
            if sha(path.read_bytes())==digest:matches.append(dict(component=component,builder_file=str(path)))
        libraries.append(dict(name=name,sha256=digest,bytes=len(data),exact_builder_matches=matches))
    gaps=['Original source-expression provenance and static/transitive native-code review remain open.']
    gaps+=['unattributed Python root: '+root for root in unattributed]
    gaps+=['native attribution unresolved: '+row['name'] for row in libraries if not row['exact_builder_matches']]
    gaps+=['installed notice missing: '+row['name'] for row in records if not row['notices']]
    if not any(row['component']=='CPython' for row in notices):gaps.append('CPython license not located in builder')
    report=dict(schema=1,status='REVIEW_REQUIRED',artifact_sha256=sha(executable.read_bytes()),system=platform.system(),machine=platform.machine(),python=sys.version,
        build_manifest=manifest,python_distributions=records,native_libraries=libraries,notices=notices,review_gaps=gaps,
        limits=['Installed versions and notice collection do not prove license compatibility or corresponding-source completeness.',
                'Unmatched native bytes may have loader/signature modifications; they are not silently attributed.',
                'Wheel/compiler static dependencies and host libraries require separate review.'])
    (output/'inventory.json').write_text(json.dumps(report,indent=2)+'\n')
    (output/'THIRD_PARTY_NOTICES.md').write_text('# Candidate third-party notices\n\nReview required; this is not redistribution clearance.\n\n'+''.join(f"- {row['name']} {row['version']}: "+', '.join('['+Path(path).name+']('+path+')' for path in row['notices'])+'\n' for row in records))
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('executable',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();r=inventory(a.executable,a.output);print(json.dumps(dict(status=r['status'],packages=len(r['python_distributions']),native_libraries=len(r['native_libraries']),review_gaps=len(r['review_gaps']))))

if __name__=='__main__':main()
