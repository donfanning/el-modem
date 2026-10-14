"""Exact DTE bytes with timestamps and byte offsets; no terminal decoding loss."""
import json
from pathlib import Path
import time


class TranscriptSocket:
    def __init__(self, connection, prefix):
        self.connection = connection
        self.started = time.monotonic()
        self.offsets = dict(rx=0,tx=0)
        self.files = {name:Path(str(prefix)+'.'+name+'.bin').open('wb') for name in self.offsets}
        self.timeline = Path(str(prefix)+'.jsonl').open('w')
        self.closed = False

    def __getattr__(self, name):
        return getattr(self.connection,name)

    def _record(self, direction, data):
        data = bytes(data)
        self.files[direction].write(data)
        self.timeline.write(json.dumps(dict(direction=direction,
            byte_offset=self.offsets[direction], bytes=len(data),
            monotonic=time.monotonic(), elapsed=time.monotonic()-self.started,
            wall_time=time.time(), display=repr(data)))+'\n')
        self.offsets[direction] += len(data)

    def sendall(self, data):
        self.connection.sendall(data)
        self._record('tx',data)

    def send(self, data, flags=0):
        count = self.connection.send(data, flags)
        self._record('tx',data[:count])
        return count

    def recv(self, count):
        data = self.connection.recv(count)
        self._record('rx',data)
        return data

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.connection.close()
        for file in self.files.values():
            file.close()
        self.timeline.close()
