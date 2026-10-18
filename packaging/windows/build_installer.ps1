# Build and exercise the Windows installer on a CI runner. Usage:
#   build_installer.ps1 -Archive build\el-modem-0.1.0-rc1-windows-x86_64.zip -Version 0.1.0-rc1 -Out installers
#
# -DriverPackage stages an already-signed virtual COM driver into the installer
# (the .inf, .dll, wudf.cat and the public .cer). Produce it with
# build_driver_cert.ps1 and sign_driver.ps1; the signing key never reaches this
# script or the installer.
param([Parameter(Mandatory)][string]$Archive, [Parameter(Mandatory)][string]$Version, [Parameter(Mandatory)][string]$Out,
      [string]$DriverPackage)
$ErrorActionPreference = 'Stop'
$iscc = 'C:\Program Files (x86)\Inno Setup 6\ISCC.exe'
if (-not (Test-Path $iscc)) {
    choco install innosetup -y --no-progress
    if (-not (Test-Path $iscc)) { throw 'Inno Setup 6 not found after install' }
}
$work = Join-Path $env:RUNNER_TEMP 'installer-src'
Remove-Item -Recurse -Force $work -ErrorAction SilentlyContinue
Expand-Archive -LiteralPath $Archive -DestinationPath $work
$source = (Get-ChildItem $work -Directory | Select-Object -First 1).FullName
New-Item -ItemType Directory -Force $Out | Out-Null
# A signed driver is optional; without it the installer simply omits the driver.
$driverArgs = @()
if ($DriverPackage) {
    foreach ($name in @('virtualserial2um.inf', 'VirtualSerial2um.dll', 'wudf.cat', 'ElModemTestSigning.cer')) {
        if (-not (Test-Path -LiteralPath (Join-Path $DriverPackage $name) -PathType Leaf)) {
            throw "-DriverPackage is missing $name"
        }
    }
    if (Get-ChildItem -LiteralPath $DriverPackage -Recurse -File |
            Where-Object { $_.Extension -in '.pfx', '.p12', '.key', '.pem' }) {
        throw 'Private-key material is present in -DriverPackage; it must never be shipped'
    }
    $driverArgs = @("/DDriverDir=$((Resolve-Path -LiteralPath $DriverPackage).Path)")
}
# Resolve our own paths from this script: CI runs from the checkout root, which
# hid a dependency on the caller's working directory.
& $iscc "/DSourceDir=$source" "/DAppVersion=$Version" "/DOutputDir=$((Resolve-Path $Out).Path)" @driverArgs (Join-Path $PSScriptRoot 'el-modem.iss')
if ($LASTEXITCODE -ne 0) { throw "ISCC failed: $LASTEXITCODE" }
$setup = (Get-ChildItem $Out -Filter *-setup.exe | Select-Object -First 1).FullName

# Exercise it: silent install forcing the Microsoft download path, run the installed program, uninstall.
$target = Join-Path $env:RUNNER_TEMP 'El-Modem-installed'
Remove-Item -Recurse -Force $target -ErrorAction SilentlyContinue
$log = Join-Path $Out 'install.log'
$p = Start-Process $setup -ArgumentList "/VERYSILENT /SUPPRESSMSGBOXES /NORESTART /DIR=`"$target`" /LOG=`"$log`" /ForceVCRedist=1" -Wait -PassThru
Write-Host "installer exit code: $($p.ExitCode)"
Get-Content $log | Select-String -Pattern "redist|Download|signature|Exception|Failed" | ForEach-Object { Write-Host $_.Line }
if ($p.ExitCode -notin 0, 3010) { throw "installer failed: $($p.ExitCode)" }
$auditArgs = @((Join-Path (Split-Path $PSScriptRoot -Parent) 'audit_windows_install.py'), $Archive, $target)
if ($DriverPackage) { $auditArgs += @('--driver-package', $DriverPackage) }
python @auditArgs
if ($LASTEXITCODE -ne 0) { throw 'Installed payload differs from the reviewed archive/explicit driver files' }
$exe = Join-Path $target 'el-modem.exe'
if (-not (Test-Path $exe)) { throw 'installed program missing' }
& $exe --self-test | Out-File (Join-Path $Out 'installed-self-test.json') -Encoding utf8
if ($LASTEXITCODE -ne 0) { throw "installed self-test failed: $LASTEXITCODE" }
$un = Join-Path $target (Get-ChildItem $target -Filter 'unins*.exe' | Select-Object -First 1).Name
$q = Start-Process $un -ArgumentList '/VERYSILENT /SUPPRESSMSGBOXES /NORESTART' -Wait -PassThru
if ($q.ExitCode -ne 0) { throw "uninstall failed: $($q.ExitCode)" }
Start-Sleep -Seconds 2
if (Test-Path $exe) { throw 'program still present after uninstall' }
Write-Host "installer verified: $setup"
