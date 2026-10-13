"""Private atomic persistence shared by supervisor and Hayes profiles."""
import json
import os
import tempfile
from pathlib import Path

def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, temporary = tempfile.mkstemp(prefix='.'+path.name, dir=path.parent)
    try:
        if hasattr(os,'fchmod'):os.fchmod(fd, 0o600)
        else:os.chmod(temporary,0o600) # Windows uses the user directory ACL.
        with os.fdopen(fd,'w',encoding='utf-8') as output:
            json.dump(data, output, indent=2)
            output.write('\n'); output.flush(); os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)
