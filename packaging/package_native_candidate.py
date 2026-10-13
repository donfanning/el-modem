"""Assemble an explicitly qualified native review archive; no publication."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import re
import zipfile


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def assemble(executable,root,reports,notices,output,target,version,revision):
    if output.exists():raise ValueError('fresh archive path required')
    if not re.fullmatch(r'[A-Za-z0-9.-]+',version) or not re.fullmatch(r'[0-9a-f]{40}',revision):raise ValueError('explicit safe version and source revision required')
    actual={'Linux':'linux','Windows':'windows','Darwin':'macos'}[platform.system()]+'-'+('x86_64' if platform.machine().lower() in ('amd64','x86_64') else platform.machine().lower())
    if target!=actual:raise ValueError('archive target differs from actual native builder '+actual)
    fingerprint=digest(executable)
    for filename in ('reference.json','native.json'):
        report=json.loads((reports/filename).read_text())
        if report.get('status')!='PASS' or report.get('artifact_sha256')!=fingerprint or report.get('serial_at_smoke')!='PASS' or not report.get('payload_requested'):
            raise ValueError('artifact qualification missing/mismatched: '+filename)
        manifest=report['build_manifest']
        for path,expected in manifest['local_modules'].items():
            if Path(path).is_absolute() or '..' in Path(path).parts or digest(root/path)!=expected:raise ValueError('source mismatch: '+path)
        if digest(root/'packaging/el_modem.spec')!=manifest['specification_sha256']:raise ValueError('packaging specification mismatch')
    if json.loads((reports/'desktop.json').read_text()).get('status')!='PASS':raise ValueError('desktop worker qualification missing')
    inventory=json.loads((notices/'inventory.json').read_text())
    if inventory['artifact_sha256']!=fingerprint:raise ValueError('notices belong to different executable')
    base=f'el-modem-{version}-{target}';files={}
    def add(name,path):
        if path.is_symlink() or not path.is_file():raise ValueError('regular explicit package file required: '+str(path))
        files[name]=path.read_bytes()
    add('el-modem.exe' if actual.startswith('windows') else 'el-modem',executable)
    for name in ('LICENSE','NOTICE','CREDITS.md','docs/AT_COMMANDS.md','docs/DESKTOP.md','docs/SERIAL.md','docs/REDISTRIBUTION.md','packaging/serial/README.md','packaging/serial/build-driver.sh','packaging/serial/serialsim-flow.patch','packaging/serial/COPYING'):add(name,root/name)
    for path in sorted(reports.glob('*')):
        if path.is_file():add('qualification/'+path.name,path)
    for path in sorted(notices.rglob('*')):
        if path.is_file():add('third-party/'+str(path.relative_to(notices)).replace('\\','/'),path)
    files['README.txt']=(f'El-Modem {version} for {target}\n\n'
        'Local review candidate; not cleared for public binary redistribution.\n'
        'Run el-modem (el-modem.exe on Windows) in a terminal for the dashboard.\n'
        'Linux defaults to the separately installed serialsim driver; use\n'
        '--serial-backend pty or tcp if that driver is unavailable. macOS\n'
        'defaults to PTY; Windows defaults to TCP and needs a separate virtual\n'
        'COM/TCP bridge for COM applications. No virtual COM driver is bundled.\n'
        'Configure SIP in the dashboard/Hayes settings; no accounts supplied.\n'
        'LAPM is enabled, compression/echo cancellation disabled. IAX calling\n'
        'is not implemented. Windows/macOS binaries are not developer-signed\n'
        'or notarized. Tests cover offline engines and local worker startup,\n'
        'not native-OS PSTN interoperability or sustained real-time limits.\n'
        'Read docs/ and third-party/inventory.json for scope and review gaps.\n').encode()
    review=dict(schema=1,status='REVIEW_REQUIRED',version=version,target=target,source_revision=revision,artifact_sha256=fingerprint,
        signature_status='developer signing/notarization not supplied; macOS may be ad-hoc signed',review_gaps=inventory['review_gaps'],
        files={name:dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest()) for name,data in files.items()})
    files['MANIFEST.json']=(json.dumps(review,indent=2)+'\n').encode()
    output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(output,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for name,data in sorted(files.items()):
            entry=zipfile.ZipInfo(base+'/'+name,date_time=(1980,1,1,0,0,0));entry.create_system=3
            entry.external_attr=((0o100755 if name in ('el-modem','el-modem.exe','packaging/serial/build-driver.sh') else 0o100644)<<16)
            entry.compress_type=zipfile.ZIP_DEFLATED;archive.writestr(entry,data)
    with zipfile.ZipFile(output) as archive:
        for name,record in review['files'].items():
            if hashlib.sha256(archive.read(base+'/'+name)).hexdigest()!=record['sha256']:raise ValueError('archive verification failed')
    return dict(target=target,source_revision=revision,artifact_sha256=fingerprint,archive_sha256=digest(output),archive_bytes=output.stat().st_size,status='REVIEW_REQUIRED')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('executable',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--root',type=Path,default=Path('.'));p.add_argument('--reports',type=Path,default=Path('candidate-reports'))
    p.add_argument('--notices',type=Path,default=Path('candidate-notices'));p.add_argument('--target',required=True)
    p.add_argument('--version',default='0.1.0-rc1');p.add_argument('--revision',required=True)
    a=p.parse_args();r=assemble(a.executable,a.root,a.reports,a.notices,a.output,a.target,a.version,a.revision)
    a.output.with_suffix('.receipt.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))

if __name__=='__main__':main()
