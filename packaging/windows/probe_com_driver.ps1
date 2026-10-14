# Disposable runner probe only: never enable TESTSIGNING or change code integrity.
$ErrorActionPreference = 'Stop'
$report = Join-Path $PWD 'com-reports'
New-Item -ItemType Directory -Force $report | Out-Null
Start-Transcript -Path "$report/probe.txt"
try {
    & bcdedit /enum '{current}' 2>&1 | Out-File "$report/boot-policy.txt"
    Get-CimInstance Win32_OperatingSystem | Select-Object Caption,Version,BuildNumber | ConvertTo-Json | Set-Content "$report/os.json"
    $zip = Join-Path $env:RUNNER_TEMP 'com0com.zip'
    $downloads = @()
    $valid = $false
    foreach ($hostName in @('downloads.sourceforge.net', 'master.dl.sourceforge.net', 'phoenixnap.dl.sourceforge.net')) {
        try {
            $url = "https://$hostName/project/com0com/com0com/3.0.0.0/com0com-3.0.0.0-i386-and-x64-signed.zip"
            Invoke-WebRequest $url -TimeoutSec 30 -UserAgent 'Python-urllib/3.15' -OutFile $zip
            $hash = (Get-FileHash $zip -Algorithm SHA256).Hash.ToLower()
            $downloads += @{url=$url;sha256=$hash;bytes=(Get-Item $zip).Length}
            if ($hash -eq '6e5d4359865277430d4ae88c73fb7e648a0ed8e81aea5002478179cfcb0bb0e1') { $valid=$true; break }
        } catch { $downloads += @{url=$url;error=$_.Exception.Message} }
    }
    $downloads | ConvertTo-Json | Set-Content "$report/downloads.json"
    if (-not $valid) { throw 'No official mirror returned the pinned driver archive' }
    $extract = Join-Path $env:RUNNER_TEMP 'com0com-download'
    Expand-Archive $zip $extract -Force
    $installer = Join-Path $extract 'Setup_com0com_v3.0.0.0_W7_x64_signed.exe'
    Get-AuthenticodeSignature $installer | Select-Object Status,StatusMessage,@{n='Signer';e={$_.SignerCertificate.Subject}} | ConvertTo-Json | Set-Content "$report/installer-signature.json"
    $install = Join-Path $env:RUNNER_TEMP 'com0com-installed'
    # Extract the unchanged signed NSIS payload; avoid legacy GUI installer hangs.
    & 7z x $installer "-o$install" -y | Out-File "$report/extract.txt"
    if ($LASTEXITCODE -ne 0) { throw 'Installer payload extraction failed' }
    $inf = Get-ChildItem $install -Recurse -Filter com0com.inf | Select-Object -First 1
    if (-not $inf) { throw 'Signed payload has no com0com.inf' }
    & pnputil /add-driver $inf.FullName /install 2>&1 | Out-File "$report/install.txt"
    "pnputil exit: $LASTEXITCODE" | Add-Content "$report/install.txt"
    Get-ChildItem $install -Recurse | Select-Object FullName,Length | ConvertTo-Json | Set-Content "$report/files.json"
    Get-ChildItem $install -Recurse -Include *.cat,*.sys | ForEach-Object {
        $s = Get-AuthenticodeSignature $_.FullName
        [pscustomobject]@{File=$_.Name;SHA256=(Get-FileHash $_.FullName).Hash;Status=[string]$s.Status;Signer=$s.SignerCertificate.Subject}
    } | ConvertTo-Json | Set-Content "$report/driver-signatures.json"
    $setup = (Get-ChildItem $install -Recurse -Filter setupc.exe | Select-Object -First 1).FullName
    $pair = Start-Process $setup -ArgumentList @('--silent', 'install', 'PortName=COM41', 'PortName=COM42') -WorkingDirectory $inf.DirectoryName -PassThru -RedirectStandardOutput "$report/create-pair.txt" -RedirectStandardError "$report/create-pair-error.txt"
    if (-not $pair.WaitForExit(90000)) { $pair.Kill(); throw 'Pair setup timeout' }
    "setupc exit: $($pair.ExitCode)" | Add-Content "$report/create-pair.txt"
    & $setup list 2>&1 | Out-File "$report/pairs.txt"
    Start-Sleep -Seconds 5
    Get-CimInstance Win32_PnPEntity | Where-Object {$_.Name -match 'com0com|COM41|COM42'} | Select-Object Name,PNPDeviceID,ConfigManagerErrorCode,Status | ConvertTo-Json | Set-Content "$report/devices.json"
    Get-CimInstance Win32_SystemDriver | Where-Object {$_.Name -match 'com0com'} | Select-Object Name,State,StartMode,PathName | ConvertTo-Json | Set-Content "$report/service.json"
    Add-Type -AssemblyName System.IO.Ports
    [System.IO.Ports.SerialPort]::GetPortNames() | Set-Content "$report/ports.txt"
    $a = New-Object System.IO.Ports.SerialPort 'COM41',115200
    $b = New-Object System.IO.Ports.SerialPort 'COM42',115200
    try {
        $a.ReadTimeout = 5000; $b.ReadTimeout = 5000
        $a.Open(); $b.Open()
        $a.Write('driver-probe'); Start-Sleep -Milliseconds 200
        if ($b.ReadExisting() -ne 'driver-probe') { throw 'COM pair byte transfer mismatch' }
        @{passed=$true;scope='Actual serial driver load and COM pair transfer; no modem or legacy terminal yet'} | ConvertTo-Json | Set-Content "$report/result.json"
    } finally { $a.Dispose(); $b.Dispose() }
} catch {
    @{passed=$false;error=$_.Exception.Message} | ConvertTo-Json | Set-Content "$report/result.json"
    throw
} finally { Stop-Transcript }
