"""Build identity embedded by packaging, never inferred from file timestamps."""
import json
import sys
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path


def compile_timestamp():
    """Called by the build specification, not when the modem starts."""
    return datetime.now(timezone.utc).isoformat(timespec='seconds').replace('+00:00', 'Z')


@lru_cache(maxsize=1)
def build_timestamp():
    if not getattr(sys, 'frozen', False):
        return 'uncompiled source'
    try:
        data = json.loads((Path(sys._MEIPASS) / 'el-modem-build.json').read_text(encoding='utf-8'))
        value = data['compiled_at_utc']
        if not isinstance(value, str) or not value.endswith('Z'):
            raise ValueError('missing UTC build timestamp')
        datetime.strptime(value, '%Y-%m-%dT%H:%M:%SZ')
        return value
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return 'unknown (build timestamp unavailable)'
