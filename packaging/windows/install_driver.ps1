<#
.SYNOPSIS
Install the El-Modem virtual COM driver and, unless told otherwise, trust its certificate.

.DESCRIPTION
Run by the El-Modem installer. Does four things in order:

  1. imports the signing certificate into the machine's Trusted Root and Trusted
     Publishers stores (skip with -SkipCertificate when it is already trusted, or
     when the driver carries a public signature);
  2. stages the signed driver package with pnputil;
  3. creates the device node, so a COM port appears;
  4. optionally pins the port to a specific COM number.

Step 3 needs devcon.exe, which pnputil cannot do: pnputil /add-driver only
copies the package into the driver store, and creating the device node for a
root-enumerated device is a SetupAPI operation. devcon ships with the Windows
Driver Kit, so it is located rather than shipped - redistributing a Microsoft
binary is not something this project should take on for a test-signed interim.

If devcon is missing the script does NOT fail the whole El-Modem install. It
stages the package, prints the exact command to finish by hand, and exits 2 so
the installer can report it.

.SECURITY
Importing a certificate into LocalMachine\Root grants this machine trust to
everything that certificate signs, without kernel code integrity review. Only
the .cer is ever read here; the signing key never ships.

.PARAMETER DriverPackage
Directory holding the signed package: the .inf, the .dll, wudf.cat and the .cer.

.PARAMETER Certificate
The public .cer to trust. Defaults to ElModemTestSigning.cer inside -DriverPackage.

.PARAMETER PortName
Pin the port to this COM name, e.g. COM3. Leave empty to accept whatever COM
number Windows assigns.

.PARAMETER SkipCertificate
Do not touch the certificate stores.

.NOTES
Exit codes: 0 success, 1 failed, 2 staged but the device node was not created.
#>
[CmdletBinding()]
param(
  [string]$DriverPackage = (Join-Path $PSScriptRoot 'driver'),
  [string]$Certificate,
  [string]$PortName = '',
  [switch]$SkipCertificate
)

$ErrorActionPreference = 'Stop'

function Write-Step { param([string]$Message) Write-Host "[driver] $Message" }
function Write-Warn { param([string]$Message) Write-Host "[driver] WARNING: $Message" }

# Find our device nodes.
#
# These are ROOT-enumerated devices: the instance id is ROOT\PORTS\0001 and does
# NOT contain the hardware id, so matching on PNPDeviceID alone finds nothing.
# The friendly name the INF assigns does contain a stable marker, so match on
# either. Getting this wrong means silently creating a second node and leaving
# two COM ports behind.
function Get-ModemNodes {
  @(Get-CimInstance Win32_PnPEntity |
    Where-Object { $_.Name -like '*El-Modem*' -or $_.PNPDeviceID -like '*VirtualSerial2um*' })
}

$inf = Get-ChildItem $DriverPackage -Filter *.inf -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not $inf) { Write-Step "no driver package at $DriverPackage"; exit 1 }
$cat = Join-Path $DriverPackage 'wudf.cat'
if (-not (Test-Path $cat)) { Write-Step "package has no wudf.cat; it is not signed"; exit 1 }
if (-not $Certificate) { $Certificate = Join-Path $DriverPackage 'ElModemTestSigning.cer' }

# 1. Trust the certificate.
if ($SkipCertificate) {
  Write-Step 'skipping certificate trust (-SkipCertificate)'
} elseif (-not (Test-Path $Certificate)) {
  Write-Step "certificate not found: $Certificate"
  exit 1
} else {
  # Deliberately not named $certificate: PowerShell variables are
  # case-insensitive, so that would collide with the $Certificate parameter.
  $signingCertificate = New-Object System.Security.Cryptography.X509Certificates.X509Certificate2 $Certificate
  foreach ($storeName in @('Root', 'TrustedPublisher')) {
    $store = New-Object System.Security.Cryptography.X509Certificates.X509Store $storeName, 'LocalMachine'
    $store.Open('ReadWrite')
    try {
      $already = @($store.Certificates | Where-Object { $_.Thumbprint -eq $signingCertificate.Thumbprint })
      if ($already) {
        Write-Step "$storeName already trusts $($signingCertificate.Thumbprint)"
      } else {
        # X509Store.Add is the call that persists. Mutating store.Certificates
        # looks like it works and reports no error, but nothing is written.
        $store.Add($signingCertificate)
        Write-Step "trusted $($signingCertificate.Subject) in $storeName"
      }
    } finally { $store.Close() }
  }
}

# Confirm the trust actually landed. Writing to a certificate store can appear
# to succeed and persist nothing, and a false positive here would only surface
# later as a confusing "untrusted root" from pnputil.
if (-not $SkipCertificate) {
  foreach ($storeName in @('Root', 'TrustedPublisher')) {
    $store = New-Object System.Security.Cryptography.X509Certificates.X509Store $storeName, 'LocalMachine'
    $store.Open('ReadOnly')
    try {
      $present = @($store.Certificates | Where-Object { $_.Thumbprint -eq $signingCertificate.Thumbprint })
      if (-not $present) {
        Write-Step "$storeName does NOT trust $($signingCertificate.Thumbprint) after the import"
        exit 1
      }
    } finally { $store.Close() }
  }
  Write-Step 'certificate trust confirmed in both stores'
}

# 2. Stage the package.
Write-Step "staging $($inf.Name)"
$previous = $ErrorActionPreference
$ErrorActionPreference = 'Continue'
try {
  $stageOutput = (pnputil /add-driver $inf.FullName 2>&1 | Out-String)
  $stageCode = $LASTEXITCODE
} finally { $ErrorActionPreference = $previous }
if ($stageCode -ne 0) {
  Write-Step "pnputil /add-driver failed:"
  Write-Host $stageOutput
  exit 1
}
Write-Step 'package staged'

# 3. Create the device node.
# Read the hardware ID from the INF's Models section rather than hardcoding it,
# so a renamed device class cannot silently break installation. (Using -and to
# pick between two values would yield a Boolean, not the string.)
$hardwareId = $null
$infText = Get-Content -LiteralPath $inf.FullName -Raw
if ($infText -match '(?m)^[^;\r\n]*=\s*[^;\r\n,]+,\s*(UMDF\\[A-Za-z0-9_]+)') {
  $hardwareId = $Matches[1]
}
if (-not $hardwareId) { $hardwareId = 'UMDF\VirtualSerial2um' }
Write-Step "hardware id: $hardwareId"

$devcon = Get-ChildItem 'C:\Program Files (x86)\Windows Kits\10\Tools' -Recurse -Filter devcon.exe `
           -ErrorAction SilentlyContinue |
           Where-Object { $_.FullName -match '\\x64\\' } |
           Sort-Object FullName -Descending | Select-Object -First 1

if (-not $devcon) {
  Write-Warn 'devcon.exe not found; the driver package is staged but no COM port was created.'
  Write-Warn 'Install the Windows Driver Kit, then run this command from an elevated prompt:'
  Write-Warn ("    devcon.exe install `"$($inf.FullName)`" `"$hardwareId`"")
  Write-Warn 'Or use --serial-backend com with a physical modem instead.'
  exit 2
}

# A node that is present but reporting a PnP problem is worse than no node at
# all: it blocks creation of a good one while presenting no COM port. Remove
# those before deciding whether anything is installed.
$existing = @(Get-ModemNodes)
$broken = @($existing | Where-Object { $_.ConfigManagerErrorCode -ne 0 })
foreach ($node in $broken) {
  Write-Step "removing unusable device node $($node.PNPDeviceID) (problem code $($node.ConfigManagerErrorCode))"
  $ErrorActionPreference = 'Continue'
  pnputil /remove-device $node.PNPDeviceID 2>&1 | Out-Null
  $ErrorActionPreference = $previous
  Start-Sleep -Seconds 2
}
$existing = @(Get-ModemNodes)

if ($existing.Count -gt 0) {
  Write-Step "device node already present ($($existing.Count)); leaving it in place"
} else {
  Write-Step "creating device node with $(Split-Path $devcon.FullName -Leaf)"
  $ErrorActionPreference = 'Continue'
  $devconOutput = (& $devcon.FullName install $inf.FullName $hardwareId 2>&1 | Out-String)
  $devconCode = $LASTEXITCODE
  $ErrorActionPreference = $previous
  if ($devconCode -ne 0) {
    Write-Step 'devcon failed:'
    Write-Host $devconOutput
    exit 1
  }
  Write-Step 'device node created'
  Start-Sleep -Seconds 3
}

# The INF's Models section can produce more than one node; keep a single port.
$nodes = @(Get-ModemNodes | Sort-Object PNPDeviceID)
if ($nodes.Count -gt 1) {
  Write-Step "removing $($nodes.Count - 1) extra device node(s)"
  $nodes | Select-Object -Skip 1 | ForEach-Object {
    $ErrorActionPreference = 'Continue'
    pnputil /remove-device $_.PNPDeviceID 2>&1 | Out-Null
    $ErrorActionPreference = $previous
  }
  Start-Sleep -Seconds 3
}

# 4. Optionally pin the port number.
if ($PortName) {
  $node = @(Get-ModemNodes | Select-Object -First 1)
  if ($node) {
    $key = "HKLM:\SYSTEM\CurrentControlSet\Enum\$($node.PNPDeviceID)\Device Parameters"
    New-Item -Path $key -Force | Out-Null
    # PortName must be the full string COMn, not the bare number.
    Set-ItemProperty -Path $key -Name PortName -Value $PortName -Type String
    $ErrorActionPreference = 'Continue'
    pnputil /restart-device "$($node.PNPDeviceID)" 2>&1 | Out-Null
    $ErrorActionPreference = $previous
    Write-Step "pinned to $PortName"
    Start-Sleep -Seconds 3
  }
}

$ports = @()
# The port takes a moment to appear after the device restarts, and the friendly
# name can lag the registry change by longer than that.
foreach ($attempt in 1..10) {
  $ports = [System.IO.Ports.SerialPort]::GetPortNames()
  if ($ports) { break }
  Start-Sleep -Seconds 2
}
Write-Step ("COM ports now: " + $(if ($ports) { $ports -join ', ' } else { '(none)' }))
if (-not $ports) {
  Write-Warn 'no COM port appeared; a reboot is usually needed after the port name is pinned'
}
exit 0