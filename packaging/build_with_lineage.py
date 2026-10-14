"""Run the normal PyInstaller build while journaling native-file processing.

Original processing arguments/results are unchanged. This records input and
actual processed bytes, not an ownership/license conclusion. No DSP code runs.
"""
import argparse
import hashlib
import inspect
import json
from pathlib import Path


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def record_processing(original,journal):
    signature=inspect.signature(original)
    def process(*args,**kwargs):
        arguments=signature.bind(*args,**kwargs);arguments.apply_defaults();values=arguments.arguments
        source=Path(values['src_name']);before=digest(source)
        result=original(*args,**kwargs)
        output=Path(result)
        record=dict(destination=str(values['dest_name']).replace('\\','/'),source_file=str(source.resolve()),
            source_sha256=before,processed_file=str(output.resolve()),processed_sha256=digest(output),
            source_after_sha256=digest(source),strip=values.get('use_strip'),upx=values.get('use_upx'),
            target_arch=values.get('target_arch'),signing_requested=bool(values.get('codesign_identity')))
        with journal.open('a',encoding='utf-8') as stream:stream.write(json.dumps(record)+'\n')
        return result
    return process


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--journal',type=Path,required=True)
    a,arguments=p.parse_known_args()
    if a.journal.exists():p.error('fresh lineage journal required')
    a.journal.parent.mkdir(parents=True,exist_ok=True)
    import PyInstaller.building.api as api
    import PyInstaller.__main__
    api.process_collected_binary=record_processing(api.process_collected_binary,a.journal)
    PyInstaller.__main__.run(arguments)
    if not a.journal.is_file():raise RuntimeError('build produced no native-processing evidence')

if __name__=='__main__':main()
