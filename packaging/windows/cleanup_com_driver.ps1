# Remove only the pair and publisher consent owned by this disposable test.
#
# com0com-specific: this cleans up the pair created by probe_com_driver.ps1, not
# the bundled El-Modem virtual-COM driver (see docs/WINDOWS_DRIVER.md). The
# COM41/COM42 identity check below is deliberate, so it removes only the pair it
# created and cannot touch an unrelated port.
$ErrorActionPreference = 'Stop'
$report = Join-Path $PWD 'com-reports'
New-Item -ItemType Directory -Force $report | Out-Null
$install = Join-Path $env:RUNNER_TEMP 'com0com-installed'
$setup = Get-ChildItem $install -Recurse -Filter setupc.exe -ErrorAction SilentlyContinue | Select-Object -First 1
if ($setup) {
    $listing = (& $setup.FullName list 2>&1 | Out-String)
    if ($listing -match 'CNCA(\d+)\s+[^\r\n]*PortName=COM41') {
        $pairNumber = $Matches[1]
        if ($listing -notmatch "CNCB$pairNumber\s+[^\r\n]*PortName=COM42") { throw 'Owned pair identity mismatch' }
        & $setup.FullName --silent remove $pairNumber 2>&1 | Out-File "$report/remove-pair.txt"
        if ($LASTEXITCODE -ne 0) { throw 'Pair removal failed' }
    }
}
$left = @(Get-PnpDevice -PresentOnly | Where-Object {$_.FriendlyName -match '\(COM41\)|\(COM42\)'})
if ($left.Count) { throw 'Owned COM devices remain present' }
$consent = Join-Path $report 'publisher-consent.json'
if (Test-Path $consent) {
    $c = Get-Content $consent -Raw | ConvertFrom-Json
    if ($c.newly_added) { Remove-Item "Cert:\LocalMachine\TrustedPublisher\$($c.thumbprint)" }
}
@{passed=$true;present_owned_devices=$left.Count;scope='Owned pair removed; temporary publisher consent removed if added. Remaining driver store disappears with disposable VM.'} | ConvertTo-Json | Set-Content "$report/driver-cleanup.json"
