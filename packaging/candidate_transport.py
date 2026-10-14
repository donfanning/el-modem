"""Retain native review candidates with age encryption; private identities stay local."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import tarfile
import urllib.request
import zipfile

VERSION='1.3.2'
HASHES={
 'linux-amd64':'cbe24006683f8eb669266162894b9a522a1af52f2665fbc63a4bb032ed26ac10',
 'darwin-amd64':'1d1e4bc66e1427edad7739ae7616157de0e79db8b6d2a1497d7d9925fb06a539',
 'darwin-arm64':'e2020b073c44f692685a24d6abc378817eb81ffaaf49fd0531ef8565f767f2f5',
 'windows-amd64':'f48d8f8f9ebe903ab5027ed067652f2cc1db94bc206976430133b905dcd8e8c7',
}


def install_age(destination):
    system={'Linux':'linux','Darwin':'darwin','Windows':'windows'}[platform.system()]
    machine={'x86_64':'amd64','AMD64':'amd64','arm64':'arm64','aarch64':'arm64'}[platform.machine()]
    target=system+'-'+machine
    suffix='zip' if system=='windows' else 'tar.gz'
    filename=f'age-v{VERSION}-{target}.{suffix}'
    with urllib.request.urlopen('https://github.com/FiloSottile/age/releases/download/v'+VERSION+'/'+filename,timeout=60) as response:
        data=response.read(32*1024*1024+1)
    if hashlib.sha256(data).hexdigest()!=HASHES[target]:raise ValueError('age archive digest differs from pinned upstream release')
    member='age/age.exe' if system=='windows' else 'age/age'
    if system=='windows':
        with zipfile.ZipFile(io.BytesIO(data)) as archive:binary=archive.read(member)
    else:
        with tarfile.open(fileobj=io.BytesIO(data)) as archive:
            item=archive.getmember(member)
            if not item.isfile():raise ValueError('age executable is not a regular member')
            binary=archive.extractfile(item).read()
    destination.mkdir(parents=True,exist_ok=True);exe=destination/Path(member).name
    exe.write_bytes(binary);exe.chmod(0o755)
    return exe


def encrypt_candidate(source,output,recipient,age):
    if not recipient.startswith('age1') or not recipient.isalnum():raise ValueError('X25519 age public recipient required')
    if not source.is_file() or output.exists():raise ValueError('existing input and fresh encrypted output required')
    output.parent.mkdir(parents=True,exist_ok=True)
    subprocess.run([str(age),'-r',recipient,'-o',str(output),str(source)],check=True)
    return dict(plaintext_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                ciphertext_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                encrypted_bytes=output.stat().st_size,recipient=recipient,age_version=VERSION)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('source',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--recipient',default=os.environ.get('CANDIDATE_RECIPIENT'))
    p.add_argument('--age',type=Path);a=p.parse_args()
    if not a.recipient:p.error('public encryption recipient required; no plaintext artifact fallback')
    age=a.age or install_age(Path('build/candidate-age'))
    report=encrypt_candidate(a.source,a.output,a.recipient,age)
    a.output.with_suffix(a.output.suffix+'.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))

if __name__=='__main__':main()
