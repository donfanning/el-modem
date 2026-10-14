# Build and exercise the Windows installer on a CI runner. Usage:
#   build_installer.ps1 -Archive build\el-modem-0.1.0-rc1-windows-x86_64.zip -Version 0.1.0-rc1 -Out installers
param([Parameter(Mandatory)][string]$Archive, [Parameter(Mandatory)][string]$Version, [Parameter(Mandatory)][string]$Out)
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
& $iscc "/DSourceDir=$source" "/DAppVersion=$Version" "/DOutputDir=$((Resolve-Path $Out).Path)" packaging\windows\el-modem.iss
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
