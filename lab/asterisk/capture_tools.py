"""Whole-call packet/PBX collectors and conservative capture completeness checks."""
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import struct
import subprocess
import time


def cli(command, container='asterisk-asterisk-1'):
    return subprocess.check_output(['docker','exec',container,'asterisk','-rx',command],text=True)


class PacketCapture:
    def __init__(self, directory, container='asterisk-asterisk-1'):
        self.directory = Path(directory)
        self.process = self.output = self.log = None
        self.container = container

    def start(self):
        pid = subprocess.check_output(['docker','inspect','--format','{{.State.Pid}}',self.container],text=True).strip()
        if not pid.isdigit() or int(pid)<=0:
            raise RuntimeError('PBX network namespace unavailable')
        self.output = (self.directory/'wire.pcap').open('wb')
        self.log = (self.directory/'tcpdump.log').open('w')
        # stdout is opened by this user, so tcpdump's elevated namespace access
        # does not leave root-owned artifacts or silently shorten snaplen.
        command = ['sudo','-n','nsenter','-t',pid,'-n','tcpdump','-Z','root',
                   '-i','any','-nn','-s','0','-U','-w','-','udp or tcp or icmp']
        self.process = subprocess.Popen(command,stdout=self.output,stderr=self.log,start_new_session=True)
        deadline = time.monotonic()+5
        while time.monotonic()<deadline:
            if self.process.poll() is not None:
                raise RuntimeError('packet capture failed: '+(self.directory/'tcpdump.log').read_text())
            if 'listening on' in (self.directory/'tcpdump.log').read_text():
                return dict(command=command,started_monotonic=time.monotonic())
            time.sleep(.05)
        raise RuntimeError('packet capture did not become ready before dialing')

    def stop(self):
        if self.process is not None and self.process.poll() is None:
            subprocess.run(['sudo','-n','kill','-INT','--','-'+str(self.process.pid)],check=True)
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                subprocess.run(['sudo','-n','kill','-TERM','--','-'+str(self.process.pid)],check=True)
                self.process.wait(timeout=3)
        for file in (self.output,self.log):
            if file is not None:
                file.close()


def _udp(frame, linktype):
    if linktype==276:  # Linux cooked v2, as used by tcpdump -i any
        protocol = struct.unpack_from('!H',frame)[0]; frame=frame[20:]
    elif linktype==113:
        protocol = struct.unpack_from('!H',frame,14)[0]; frame=frame[16:]
    elif linktype==1:
        protocol = struct.unpack_from('!H',frame,12)[0]; frame=frame[14:]
        if protocol in (0x8100,0x88a8):
            protocol=struct.unpack_from('!H',frame,2)[0];frame=frame[4:]
    else:
        return None
    if protocol!=0x800 or len(frame)<20 or frame[0]>>4!=4 or frame[9]!=17:
        return None
    if struct.unpack_from('!H',frame,6)[0]&0x3fff:
        return None  # Original fragments remain in the PCAP, not silently reassembled.
    size=(frame[0]&15)*4
    if len(frame)<size+8:
        return None
    source='.'.join(str(v) for v in frame[12:16]); destination='.'.join(str(v) for v in frame[16:20])
    sport,dport,length,_=struct.unpack_from('!HHHH',frame,size)
    return source,sport,destination,dport,frame[size+8:size+length]


def inspect_pcap(path, timeline=None):
    failures=[]; streams={}; call_ids=set(); messages=[]; count=sip=truncated=0
    out=Path(timeline).open('w') if timeline else None
    try:
        with Path(path).open('rb') as file:
            header=file.read(24)
            if len(header)!=24:
                return dict(valid=False,failures=['missing PCAP global header'])
            variants={b'\xd4\xc3\xb2\xa1':('<',1e6),b'\xa1\xb2\xc3\xd4':('>',1e6),
                      b'\x4d\x3c\xb2\xa1':('<',1e9),b'\xa1\xb2\x3c\x4d':('>',1e9)}
            if header[:4] not in variants:
                return dict(valid=False,failures=['unsupported PCAP format'])
            endian,units=variants[header[:4]]
            linktype=struct.unpack_from(endian+'I',header,20)[0]
            if linktype not in (1,113,276):
                failures.append('unsupported network link format for media inventory')
            while True:
                offset=file.tell(); record=file.read(16)
                if not record:
                    break
                if len(record)!=16:
                    failures.append('partial PCAP packet header');break
                sec,fraction,kept,original=struct.unpack(endian+'IIII',record)
                frame=file.read(kept)
                if len(frame)!=kept:
                    failures.append('partial PCAP packet data');break
                count+=1
                if kept!=original:
                    truncated+=1
                try:
                    udp=_udp(frame,linktype)
                except (IndexError,struct.error):
                    udp=None
                if udp is None:
                    continue
                source,sport,destination,dport,payload=udp
                if payload.startswith((b'INVITE ',b'ACK ',b'BYE ',b'CANCEL ',b'SIP/2.0 ')):
                    sip+=1
                    match=re.search(rb'(?:\r\n)(?:Call-ID|i):\s*([^\r\n]+)',payload,re.I)
                    if match:
                        call_ids.add(match[1].decode('ascii','replace'))
                    first=payload.split(b'\r\n',1)[0].decode('ascii','replace')
                    cseq=re.search(rb'\r\nCSeq:\s*(\d+)\s+(\w+)',payload,re.I)
                    messages.append(dict(first_line=first,call_id=match[1].decode('ascii','replace') if match else None,
                        cseq_method=cseq[2].decode('ascii','replace') if cseq else None,
                        arrival_wall_time=sec+fraction/units,source=source,destination=destination))
                if len(payload)<12 or payload[0]>>6!=2 or payload[1]&127!=0:
                    continue
                sequence,timestamp,ssrc=struct.unpack_from('!HII',payload,2)
                hlen=12+4*(payload[0]&15)
                if payload[0]&16:
                    if len(payload)<hlen+4:
                        continue
                    hlen+=4+4*struct.unpack_from('!H',payload,hlen+2)[0]
                plen=len(payload)-hlen-(payload[-1] if payload[0]&32 else 0)
                if plen<=0:
                    continue
                key=f'{source}:{sport}>{destination}:{dport}/{ssrc}'
                stream=streams.setdefault(key,dict(source=source,source_port=sport,
                    destination=destination,destination_port=dport,ssrc=ssrc,packets=0,
                    samples=0,first_arrival=sec+fraction/units,last_arrival=None,
                    timestamp_discontinuities=0,last_timestamp=None,last_samples=None))
                if stream['last_timestamp'] is not None and ((timestamp-stream['last_timestamp'])&0xffffffff)!=stream['last_samples']:
                    stream['timestamp_discontinuities']+=1
                stream.update(last_timestamp=timestamp,last_samples=plen,last_arrival=sec+fraction/units)
                stream['packets']+=1;stream['samples']+=plen
                if out:
                    out.write(json.dumps(dict(stream=key,arrival_wall_time=sec+fraction/units,
                        sequence=sequence,timestamp=timestamp,ssrc=ssrc,pcmu_samples=plen,
                        pcap_record_offset=offset,pcmu_payload_hex=payload[hlen:hlen+plen].hex()))+'\n')
    finally:
        if out:
            out.close()
    if not count:
        failures.append('no captured packets')
    if truncated:
        failures.append('packet snaplen truncated wire data')
    return dict(valid=not failures,failures=failures,packets=count,
        truncated_packets=truncated,sip_messages=sip,call_ids=sorted(call_ids),
        streams=streams,linktype=linktype,sip_summary=messages,
        timestamp_discontinuities_mean='arrival-order observations; may include reordering, not a loss estimate')


def pcm_inventory(directory):
    captures=[]; failures=[]
    for meta_path in Path(directory).rglob('*.pcm.*.txt'):
        values={}
        for line in meta_path.read_text().splitlines():
            parts=line.split(maxsplit=1)
            if len(parts)==2:
                values[parts[0]]=parts[1]
        stem=str(meta_path)[:-4]
        paths={direction:Path(stem+'.'+direction+'.raw') for direction in ('rx','tx')}
        sizes={direction:path.stat().st_size for direction,path in paths.items()}
        rows=[]; timeline=Path(stem+'.timeline.jsonl')
        if timeline.exists():
            rows=[json.loads(line) for line in timeline.read_text().splitlines()]
        for direction in sizes:
            expected=0
            for row in rows:
                if row['direction']==direction:
                    if row['sample_offset']!=expected:
                        failures.append('non-contiguous PCM index: '+stem)
                    expected+=row['samples']
            if expected*2!=sizes[direction]:
                failures.append('PCM timeline does not cover file: '+stem+'.'+direction)
        if values.get('truncated')!='False':
            failures.append('PCM capture truncated or unclosed: '+stem)
        captures.append(dict(stem=stem,bytes=sizes,metadata=values,
            timeline_rows=len(rows),concealed_samples=sum(r['samples'] for r in rows
                if r.get('context',{}).get('pcm_source')=='concealment'),
            skipped_samples=sum(r.get('context',{}).get('skipped_samples',0) for r in rows)))
    return dict(valid=bool(captures) and not failures,captures=captures,failures=failures)


def hash_manifest(directory):
    directory=Path(directory)
    return {str(path.relative_to(directory)):dict(bytes=path.stat().st_size,
        sha256=hashlib.file_digest(path.open('rb'),'sha256').hexdigest())
        for path in sorted(directory.rglob('*')) if path.is_file()
        and path.name!='manifest.json'}
