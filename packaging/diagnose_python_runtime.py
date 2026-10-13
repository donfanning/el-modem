"""Compare the ordinary interpreter with shared libpython, without a modem call.

The supplied builder image must have GCC and CPython 3.11 development headers.
An optimized-library path is inside that image, not on the host. Results are
diagnostic microbenchmarks; they do not qualify DSP deadlines or interoperability.
"""
import argparse
import json
from pathlib import Path
import subprocess

EMBED = """#include <Python.h>
int main(int argc, char **argv) { return Py_BytesMain(argc, argv); }
"""
BENCH = """import hashlib,json,os,platform,sys,sysconfig,timeit
from pathlib import Path
def work():
    x=0.0
    for i in range(400000):
        x += i*.1
    return x
library=Path(os.environ.get('AUDIT_LIBRARY',str(Path(sysconfig.get_config_var('LIBDIR'))/sysconfig.get_config_var('INSTSONAME'))))
loaded=sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines() if 'libpython3.11.so' in line})
print(json.dumps(dict(arm=os.environ['AUDIT_ARM'],executable=sys.executable,
    python=sys.version,machine=platform.machine(),python_home=os.environ.get('PYTHONHOME'),
    installed_library_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
    loaded_python_libraries=[dict(path=p,sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest()) for p in loaded],
    seconds=timeit.repeat(work,repeat=5,number=10))))
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--builder', default='el-modem-builder-bookworm')
    parser.add_argument('--optimized-library')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output/'runtime_embed.c').write_text(EMBED)
    (output/'runtime_bench.py').write_text(BENCH)
    command = ['docker', 'run', '--rm', '--network', 'none',
               '-v', f'{output}:/audit', args.builder, 'sh', '-ec',
               'gcc -O2 -I/usr/include/python3.11 /audit/runtime_embed.c -lpython3.11 -o /audit/python-shared']
    subprocess.run(command, check=True, capture_output=True)
    arms = [('source', 'python3', []), ('shared', '/audit/python-shared', [])]
    if args.optimized_library:
        library = Path(args.optimized_library)
        if not library.is_absolute() or library.name != 'libpython3.11.so.1.0':
            parser.error('optimized library must be an absolute CPython 3.11 library path')
        arms.append(('optimized', '/audit/python-shared',
                     ['-e', f'LD_LIBRARY_PATH={library.parent}', '-e', f'AUDIT_LIBRARY={library}']))
    results = []
    commands = []
    # Alternate arm order. Keep the entire sequence; do not select fastest runs.
    for index in range(2):
        for arm, executable, environment in arms[::1 if index == 0 else -1]:
            command = ['docker', 'run', '--rm', '--network', 'none',
                       '-v', f'{output}:/audit', '-e', 'PYTHONHOME=/usr',
                       '-e', f'AUDIT_ARM={arm}', *environment,
                       args.builder, executable, '/audit/runtime_bench.py']
            commands.append(command)
            run = subprocess.run(command, check=True, capture_output=True, text=True)
            results.append(json.loads(run.stdout))
    (output/'commands.json').write_text(json.dumps(commands, indent=2)+'\n')
    (output/'results.json').write_text(json.dumps(results, indent=2)+'\n')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
