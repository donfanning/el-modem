#!/bin/sh
# One executable and one specification for Linux, macOS and Windows.
# packaging/build.sh docker-bookworm -> experimental matched-runtime build
# packaging/build.sh docker-bookworm-lto -> PGO+LTO shared Python build
# packaging/build.sh docker-bookworm-optimized -> experimental PGO-only candidate
# packaging/build.sh docker -> packaging/dist/el-modem
# packaging/build.sh        -> dist/el-modem[.exe]
set -eu
cd "$(dirname "$0")/.."
if [ "${1:-}" = docker ]; then
  mkdir -p packaging/dist
  docker run --rm -v "$PWD/lab/asterisk":/work/lab/asterisk:ro \
    -v "$PWD/packaging":/work/packaging:ro \
    -v "$PWD/packaging/dist":/out -w /work python:3.12-slim sh -ec '
      apt-get update -qq >/dev/null
      apt-get install -y -qq binutils >/dev/null 2>&1
      pip install -q numpy numba pyinstaller textual==0.89.1
      pyinstaller --noconfirm --distpath /out --workpath /tmp/el-modem-build packaging/el_modem.spec
    '
elif [ "${1:-}" = docker-bookworm ] || [ "${1:-}" = docker-bookworm-optimized ] || [ "${1:-}" = docker-bookworm-lto ]; then
  mkdir -p packaging/dist
  docker build -f packaging/Dockerfile.build -t el-modem-builder-bookworm .
  task_builder=el-modem-builder-bookworm
  if [ "${1:-}" = docker-bookworm-optimized ]; then
    docker build -f packaging/Dockerfile.optimized-libpython -t el-modem-builder-bookworm-optimized .
    task_builder=el-modem-builder-bookworm-optimized
  fi
  if [ "${1:-}" = docker-bookworm-lto ]; then
    docker build -f packaging/Dockerfile.lto-libpython -t el-modem-builder-bookworm-lto .
    task_builder=el-modem-builder-bookworm-lto
  fi
  docker run --rm -v "$PWD/lab/asterisk":/work/lab/asterisk:ro \
    -v "$PWD/packaging":/work/packaging:ro \
    -v "$PWD/packaging/dist":/out -w /work "$task_builder" sh -ec '
      /opt/el-modem-build/bin/python -c "import sys,numpy,numba,llvmlite; print(dict(python=sys.version,numpy=numpy.__version__,numba=numba.__version__,llvmlite=llvmlite.__version__))"
      /opt/el-modem-build/bin/python -m PyInstaller --noconfirm --distpath /out --workpath /tmp/el-modem-build packaging/el_modem.spec
    '
else
  python -m pip install numpy numba pyinstaller textual==0.89.1
  python -m PyInstaller --noconfirm --distpath dist --workpath build/el-modem packaging/el_modem.spec
fi
