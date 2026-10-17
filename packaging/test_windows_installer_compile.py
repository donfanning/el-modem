"""Compile real installer layouts on Windows without installing any driver."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


@unittest.skipUnless(os.name=='nt','Inno compiler runs on Windows')
class InstallerCompileTests(unittest.TestCase):
    def test_absent_empty_and_explicit_driver_payloads(self):
        compiler=shutil.which('ISCC.exe') or r'C:\Program Files (x86)\Inno Setup 6\ISCC.exe'
        self.assertTrue(Path(compiler).is_file(),'Install Inno Setup before this gate')
        scripts=Path(__file__).parent/'windows'
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);package=root/'package';package.mkdir()
            for name in ('el-modem.exe','LICENSE','NOTICE','CREDITS.md','README.txt','MANIFEST.json',
                         'docs/readme.txt','third-party/notice.txt','packaging/serial/COPYING'):
                p=package/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'test payload')
            driver=root/'driver';driver.mkdir()
            for name in ('virtualserial2um.inf','VirtualSerial2um.dll','wudf.cat','ElModemTestSigning.cer',
                         'MUST-NOT-SHIP.pfx','runner.log'):
                (driver/name).write_bytes(b'test only')
            for name in ('install_driver.ps1','uninstall_driver.ps1'):
                shutil.copyfile(scripts/name,root/name)
            for variant,define,expected in (('absent',[],0),('empty',['/DDriverDir='],0),
                                            ('included',['/DDriverDir='+str(driver)],1)):
                with self.subTest(variant=variant):
                    expanded=root/(variant+'.expanded.iss');script=root/(variant+'.iss')
                    script.write_text((scripts/'el-modem.iss').read_text()+
                        '\n#if DriverIncluded != '+str(expected)+'\n#error Wrong driver branch\n#endif\n'+
                        '#expr SaveToFile("'+str(expanded)+'")\n')
                    result=subprocess.run([compiler,'/Q','/DSourceDir='+str(package),'/DAppVersion=0.1.0',
                                           '/DOutputDir='+str(root/variant),*define,str(script)],
                                          capture_output=True,text=True,timeout=120)
                    self.assertEqual(result.returncode,0,result.stdout+result.stderr)
                    text=expanded.read_text(encoding='utf-8-sig')
                    sources=[line for line in text.splitlines() if line.startswith('Source:')]
                    driver_sources=[line for line in sources if 'DestDir: "{app}\\driver"' in line]
                    self.assertEqual(len(driver_sources),6 if expected else 0)
                    self.assertTrue(all('\\*"' not in line for line in driver_sources))
                    self.assertNotIn('MUST-NOT-SHIP',text)
                    self.assertNotIn('runner.log',text)


if __name__=='__main__':unittest.main()
