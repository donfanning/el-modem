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
$driverArg = '/DDriverDir='
if ($DriverPackage) {
    if (-not (Test-Path (Join-Path $DriverPackage 'wudf.cat'))) {
        throw "-DriverPackage has no wudf.cat; it is not a signed driver package: $DriverPackage"
    }
    $key = Join-Path $DriverPackage 'ElModemTestSigning.pfx'
    if (Test-Path $key) {
        # The signing key must never end up in an installer.
        throw 'the signing key .pfx is present in -DriverPackage; it must never be shipped'
    }
    $driverArg = "/DDriverDir=$DriverPackage"
}
& $iscc "/DSourceDir=$source" "/DAppVersion=$Version" "/DOutputDir=$((Resolve-Path $Out).Path)" $driverArg packaging\windows\el-modem.iss
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
