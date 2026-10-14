#!/usr/bin/env python3
"""Sustained local-extension V.34 gates. Extension 110 routes to modem2 only.

Recreates the two Compose lab DCE containers; does not dial a trunk.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

p=argparse.ArgumentParser()
p.add_argument('--output',type=Path,required=True)
p.add_argument('--rates',type=int,nargs='+',default=[9600,33600])
p.add_argument('--protocols',nargs='+',choices=['raw','lapm'],default=['raw','lapm'])
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
lab=Path(__file__).resolve().parent
results=[]
for rate in a.rates:
 env={**os.environ,'LINE_MODE':'v34','V34_RATE':str(rate)}
 subprocess.run(['docker','compose','up','-d','--no-deps','--force-recreate','modem','modem2'],cwd=lab,env=env,check=True)
 for protocol in a.protocols:
  name=f'{rate}-{protocol}'; size=60000 if rate==9600 else 200000
  common=['python3','/app/two_modem_duplex_throughput.py','--bytes',str(size),
          '--expected-rate',str(rate),'--error-control',protocol,'--transfer-timeout','150']
  processes=[]; logs=[]; start=time.monotonic()
  try:
   for service,pty,role in [('modem2','/tmp/modem1','answer'),('modem','/tmp/modem0','originate')]:
    log=(a.output/f'{name}-{role}.log').open('w');logs.append(log)
    processes.append(subprocess.Popen(['docker','compose','exec','-T',service,*common,pty,'--role',role,'--dial','110'],cwd=lab,stdout=log,stderr=subprocess.STDOUT))
    if role=='answer':time.sleep(2)
   codes=[proc.wait(timeout=220) for proc in processes]
   result=dict(rate=rate,protocol=protocol,bytes_per_direction=size,status='PASS' if codes==[0,0] else 'FAIL',exit_codes=codes)
  except subprocess.TimeoutExpired:
   for proc in processes:proc.kill();proc.wait()
   result=dict(rate=rate,protocol=protocol,status='FAIL',reason='timeout')
  finally:
   for log in logs:log.close()
   for service in ('modem','modem2'):
    with (a.output/f'{name}-{service}-dce.log').open('w') as log:
     subprocess.run(['docker','compose','logs','--no-color',service],cwd=lab,stdout=log,stderr=subprocess.STDOUT)
    capture_dir=a.output/f'{name}-{service}-audio'
    subprocess.run(['docker','compose','cp',f'{service}:/tmp/cap',str(capture_dir)],cwd=lab,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
  result['wall_seconds']=round(time.monotonic()-start,2);results.append(result)
  print(name,result['status'],flush=True)
  (a.output/'results.json').write_text(json.dumps(results,indent=2)+'\n')
  if result['status']!='PASS':
   subprocess.run(['docker','compose','restart','modem','modem2'],cwd=lab,check=True)
raise SystemExit(0 if all(r['status']=='PASS' for r in results) else 1)
