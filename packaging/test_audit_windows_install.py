import tempfile
import unittest
import zipfile
from pathlib import Path
from audit_windows_install import audit, DRIVER_FILES, DRIVER_SCRIPTS


class WindowsInstallAuditTests(unittest.TestCase):
    def test_exact_install_and_uninstaller_exceptions(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);installed=root/'installed';installed.mkdir()
            archive=root/'package.zip'
            with zipfile.ZipFile(archive,'w') as data:
                data.writestr('modem/el-modem.exe',b'exe')
                data.writestr('modem/packaging/serial/COPYING',b'license')
            (installed/'el-modem.exe').write_bytes(b'exe')
            (installed/'packaging/serial').mkdir(parents=True)
            (installed/'packaging/serial/COPYING').write_bytes(b'license')
            for name in ('unins000.exe','unins000.dat'):(installed/name).write_bytes(b'generated')
            self.assertTrue(audit(archive,installed)['valid'])
            (installed/'packaging/serial/COPYING').unlink()
            self.assertEqual(audit(archive,installed)['missing'],['packaging/serial/copying'])
            (installed/'el-modem.exe').write_bytes(b'wrong')
            self.assertEqual(audit(archive,installed)['changed'],['el-modem.exe'])
            (installed/'driver').mkdir();(installed/'driver/runner.log').write_bytes(b'private')
            self.assertEqual(audit(archive,installed)['extra'],['driver/runner.log'])

    def test_explicit_driver_contents_only(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);installed=root/'installed';installed.mkdir()
            archive=root/'package.zip'
            with zipfile.ZipFile(archive,'w') as data:data.writestr('modem/app',b'app')
            (installed/'app').write_bytes(b'app')
            driver=root/'driver';driver.mkdir();scripts=root/'scripts';scripts.mkdir()
            (installed/'driver').mkdir()
            for name in DRIVER_FILES+DRIVER_SCRIPTS:
                payload=name.encode();source=driver if name in DRIVER_FILES else scripts
                (source/name).write_bytes(payload);(installed/'driver'/name).write_bytes(payload)
            self.assertTrue(audit(archive,installed,driver,scripts)['valid'])
            (installed/'driver/other.pfx').write_bytes(b'key')
            self.assertEqual(audit(archive,installed,driver,scripts)['extra'],['driver/other.pfx'])

    def test_unsafe_or_duplicate_package_paths_fail(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);archive=root/'package.zip'
            for names in (('modem/../bad',),('modem/APP','modem/app'),('one/app','two/app')):
                with zipfile.ZipFile(archive,'w') as data:
                    for name in names:data.writestr(name,b'file')
                with self.assertRaises(ValueError):audit(archive,root)


if __name__=='__main__':unittest.main()
