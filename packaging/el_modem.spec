# Single executable; the same dependency discovery is used on every platform.
import ast
from PyInstaller.utils.hooks import collect_data_files, collect_submodules
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

root = Path(SPECPATH).parent
source = root / 'lab' / 'asterisk'
modules = {p.stem: p for p in list(source.glob('*.py')) + list((source / 'audio-peer').glob('*.py')) + list((root / 'packaging').glob('*.py'))}
reachable = set()
pending = ['el_modem']
while pending:
    name = pending.pop()
    if name in reachable:
        continue
    reachable.add(name)
    for node in ast.walk(ast.parse(modules[name].read_text(encoding='utf-8'))):
        if isinstance(node, ast.Import):
            imports = [alias.name.split('.')[0] for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports = [node.module.split('.')[0]]
        elif (isinstance(node, ast.Call) and node.args and
              isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str) and
              ((isinstance(node.func, ast.Name) and node.func.id == '__import__') or
               (isinstance(node.func, ast.Attribute) and node.func.attr == 'import_module'))):
            imports = [node.args[0].value.split('.')[0]]
        else:
            imports = []
        pending.extend(n for n in imports if n in modules and n not in reachable)

manifest_directory = tempfile.TemporaryDirectory(prefix='el-modem-build-')
manifest = Path(manifest_directory.name) / 'el-modem-build.json'
library_override = os.environ.get('EL_MODEM_LIBPYTHON')
library_metadata = None
if library_override:
    library_override = Path(library_override).resolve(strict=True)
    if sys.platform != 'linux' or library_override.name != 'libpython3.11.so.1.0' or sys.version_info[:2] != (3, 11):
        raise RuntimeError('optimized library override requires Linux CPython 3.11')
    metadata = library_override.parent / 'build.json'
    if not metadata.is_file():
        raise RuntimeError('optimized library override needs its build provenance')
    library_metadata = json.loads(metadata.read_text())
    library_metadata['sha256'] = hashlib.sha256(library_override.read_bytes()).hexdigest()
manifest.write_text(json.dumps(dict(schema=1, local_modules={
    str(modules[name].relative_to(root)): hashlib.sha256(modules[name].read_bytes()).hexdigest()
    for name in sorted(reachable)}, specification_sha256=hashlib.sha256(
        (root / 'packaging' / 'el_modem.spec').read_bytes()).hexdigest(),
    python_library=library_metadata), sort_keys=True), encoding='utf-8')

a = Analysis([str(source / 'el_modem.py')], pathex=[str(source), str(source / 'audio-peer'), str(root / 'packaging')],
             binaries=[], datas=[(str(manifest), '.')] + collect_data_files('textual') +
                 ([(str(library_override.parent / 'LICENSE.python'), 'licenses')] if library_override else []),
             hiddenimports=sorted(reachable - {'el_modem'}) + collect_submodules('textual'),
             hookspath=[], hooksconfig={}, runtime_hooks=[], excludes=[], noarchive=False)
if library_override:
    matches = [entry for entry in a.binaries if entry[0] == library_override.name]
    if len(matches) != 1:
        raise RuntimeError('expected exactly one bundled Python shared library')
    a.binaries = [(name, str(library_override), kind) if name == library_override.name else (name, path, kind)
                  for name, path, kind in a.binaries]
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, a.binaries, a.datas, [], name='el-modem',
          debug=False, bootloader_ignore_signals=False, strip=False, upx=True, console=True)
