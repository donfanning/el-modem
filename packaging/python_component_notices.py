"""Collect the reviewed CPython 3.11 SHA3 component notice for native packages.

Read the official version-tagged notice without executing source. The reviewed
SHA-256 is shared by the release's CPython 3.11.9 and 3.11.17 sources. A changed
notice or another Python minor version requires a fresh component review.
"""
import hashlib
from pathlib import Path
import sys
from urllib.request import urlopen

SHA3_NOTICE_SHA256 = '4633fc5859344c6f244939c27700fb941d12c4636215a8eccde17d9a6d1dcb0d'


def retain(output, version=None):
    version = version or sys.version_info[:3]
    if tuple(version[:2]) != (3, 11):
        raise ValueError('CPython minor version needs its own component-notice review')
    url = 'https://raw.githubusercontent.com/python/cpython/v' + '.'.join(map(str, version)) + '/Modules/_sha3/LICENSE'
    with urlopen(url, timeout=45) as response:
        data = response.read(64 * 1024)
    if hashlib.sha256(data).hexdigest() != SHA3_NOTICE_SHA256:
        raise ValueError('CPython SHA3 notice differs from reviewed source')
    target = output / 'licenses' / 'CPython-components' / 'SHA3-LICENSE.txt'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    return dict(component='CPython SHA3', path=target.relative_to(output).as_posix(),
                sha256=SHA3_NOTICE_SHA256, source=url)
