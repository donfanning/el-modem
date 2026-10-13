"""Acquire exact dependency sources for an inventoried frozen Linux artifact.

Use the matched builder's retained, previously verified APT source indexes.
Archives are checksum-verified, never executed or extracted. Existing verified
files are reusable after interruption; missing versions remain explicit errors.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from importlib import metadata
import json
from pathlib import Path
import re
import shutil
import subprocess
from urllib.request import urlopen


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def collect(inventory, output):
    measured=json.loads(inventory.read_text())
    catalog=json.loads(inventory.with_name('debian-source-catalog.json').read_text())
    output.mkdir(parents=True,exist_ok=True)
    state=output/'source-status.json'
    if state.exists() and json.loads(state.read_text())['artifact_sha256']!=measured['artifact_sha256']:
        raise ValueError('output belongs to another executable')
    report=dict(schema=1,artifact_sha256=measured['artifact_sha256'],status='RUNNING',sources=[],errors=[],build_material=[])
    jobs=[]
    for source in catalog:
        package,version=source['package'],source['version']
        component='debian/'+package
        try:
            result=subprocess.run(['apt-get','--print-uris','source','--download-only',package+'='+version],
                                  capture_output=True,text=True,timeout=90)
            lines=[]
            for line in result.stdout.splitlines():
                match=re.fullmatch(r"'([^']+)' (\S+) (\d+) SHA256:([0-9a-f]{64})",line)
                if match:
                    url,name,size,digest=match.groups()
                    if Path(name).name!=name:raise ValueError('invalid source filename')
                    if url.startswith('http://'):url='https://'+url[7:]
                    lines.append(dict(component=component,package=package,version=version,
                                      url=url,filename=name,size=int(size),sha256=digest,
                                      verification='retained signed APT source index'))
            if result.returncode or not lines:
                raise RuntimeError('exact source not resolved from retained APT indexes: '+result.stderr.strip())
            jobs.extend(lines)
        except Exception as exc:
            report['errors'].append(dict(component=component,version=version,error=str(exc)))
    pip_packages={d['name']:d['version'] for d in measured['python_distributions'] if not d.get('debian_packages')}
    pip_packages['PyInstaller']=metadata.version('pyinstaller')
    pip_packages['pyinstaller-hooks-contrib']=metadata.version('pyinstaller-hooks-contrib')
    for package,version in sorted(pip_packages.items()):
        component='pypi/'+package
        try:
            with urlopen('https://pypi.org/pypi/'+package+'/'+version+'/json',timeout=45) as response:
                release=json.load(response)
            archives=[f for f in release['urls'] if f['packagetype']=='sdist']
            if len(archives)!=1:raise ValueError('expected exactly one source distribution')
            item=archives[0];name=item['filename']
            if Path(name).name!=name:raise ValueError('invalid sdist filename')
            jobs.append(dict(component=component,package=package,version=version,url=item['url'],
                             filename=name,size=item['size'],sha256=item['digests']['sha256'],
                             verification='PyPI exact-version release SHA-256 over HTTPS'))
        except Exception as exc:
            report['errors'].append(dict(component=component,version=version,error=str(exc)))
    (output/'source-plan.json').write_text(json.dumps(jobs,indent=2)+'\n')
    print('Source acquisition plan:',len(jobs),'archives,',sum(j['size'] for j in jobs),'bytes',flush=True)
    state.write_text(json.dumps(report,indent=2)+'\n')

    def download(job):
        target=output/job['component']/job['filename'];target.parent.mkdir(parents=True,exist_ok=True)
        record=dict(job,file=str(target.relative_to(output)))
        if target.is_file() and target.stat().st_size==job['size'] and sha(target)==job['sha256']:
            return dict(record,status='VERIFIED_REUSED')
        partial=target.with_name(target.name+'.partial')
        cached=Path('/opt/python-source')/job['filename']
        if cached.is_file() and cached.stat().st_size==job['size'] and sha(cached)==job['sha256']:
            shutil.copyfile(cached,partial)
        else:
            with urlopen(job['url'],timeout=90) as response,partial.open('wb') as stream:
                shutil.copyfileobj(response,stream,1024*1024)
        if partial.stat().st_size!=job['size'] or sha(partial)!=job['sha256']:
            raise ValueError('download size/SHA-256 mismatch: '+job['filename'])
        partial.replace(target)
        return dict(record,status='VERIFIED')

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures={pool.submit(download,j):j for j in jobs}
        for future in as_completed(futures):
            job=futures[future]
            try:
                report['sources'].append(future.result())
                print('Verified',job['component'],job['filename'],flush=True)
            except Exception as exc:
                report['errors'].append(dict(component=job['component'],filename=job['filename'],error=str(exc)))
                print('FAILED',job['component'],job['filename'],str(exc),flush=True)
            state.write_text(json.dumps(report,indent=2)+'\n')
    paths=[Path('/opt/el-modem-libpython')/name for name in ('build.json','config.log','source-sha256.txt')]
    paths += list(Path('/opt/python-source').glob('python3.11-*/Modules/Setup.local'))
    for path in paths:
        if path.is_file():
            target=output/'custom-python-build'/path.name;target.parent.mkdir(exist_ok=True);shutil.copyfile(path,target)
            report['build_material'].append(dict(file=str(target.relative_to(output)),sha256=sha(target)))
    report['sources'].sort(key=lambda x:(x['component'],x['filename']))
    report['status']='SOURCE_ARCHIVES_COLLECTED_REVIEW_REQUIRED' if not report['errors'] else 'INCOMPLETE'
    report['limits']=['Archives and notices do not independently prove license compatibility or complete corresponding-source compliance.',
                     'Custom application code, build scripts, static/transitive code and installation requirements still need review.',
                     'No dependency source archive is executed or extracted. Qualified executable unchanged.']
    state.write_text(json.dumps(report,indent=2)+'\n')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inventory',type=Path);parser.add_argument('output',type=Path)
    args=parser.parse_args();report=collect(args.inventory,args.output)
    print(report['status'],len(report['sources']),'verified archives,',len(report['errors']),'errors',flush=True)
    if report['errors']:raise SystemExit(1)


if __name__=='__main__':main()
