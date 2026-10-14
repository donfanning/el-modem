#!/bin/sh
# Build the separately installed prerequisite. Does not load/install it.
set -eu
if [ "$#" -ne 1 ]; then echo "Usage: $0 /absolute/new-build-directory" >&2; exit 2; fi
case "$1" in /*) ;; *) echo 'Use an absolute build directory' >&2; exit 2;; esac
revision=60c0f40d884d92b80442857ed1b659f60294d9fb
patch_dir=$(CDPATH= cd -- "$(dirname "$0")" && pwd)
git clone https://github.com/cminyard/serialsim.git "$1"
git -C "$1" checkout --detach "$revision"
git -C "$1" apply "$patch_dir/serialsim-flow.patch"
(cd "$1" && make)
echo "Built $1/serialsim.ko; module installation/signing is a separate administrator step."
