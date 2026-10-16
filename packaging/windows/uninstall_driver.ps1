<#
.SYNOPSIS
Remove the El-Modem virtual COM driver and its test certificate.

.DESCRIPTION
Counterpart to install_driver.ps1, run when El-Modem is uninstalled. Removes the
device node, the staged package, and - unless told otherwise - the signing
certificate from Trusted Root and Trusted Publishers.

Removing the certificate is the part that matters: while it is trusted, this
machine accepts anything signed by it. Uninstalling El-Modem should not leave
that behind.

.PARAMETER DriverPackage
Directory holding the driver package, used to find the .inf for published-name
detection and the .cer to remove.

.PARAMETER KeepCertificate
Leave the certificate trusted, for example when another tool depends on it.

.PARAMETER RemoveCertificate
Force removal of the certificate even if the .cer is missing.

.NOTES
Exit codes: 0 success, 1 failed.
#>
[CmdletBinding()]
param(
  [string]$DriverPackage = (Join-Path $PSScriptRoot 'driver'),
  [switch]$KeepCertificate,
  [switch]$RemoveCertificate
)

$ErrorActionPreference = 'Stop'

function Write-Step { param([string]$Message) Write-Host "[driver] $Message" }

# See install_driver.ps1: these are root-enumerated devices whose instance id is
# ROOT\PORTS\NNNN, so the friendly name is what identifies them.
function Get-ModemNodes {
  @(Get-CimInstance Win32_PnPEntity |
    Where-Object { $_.Name -like '*El-Modem*' -or $_.PNPDeviceID -like '*VirtualSerial2um*' })
}

# 1. Device nodes.
$nodes = Get-ModemNodes
foreach ($node in $nodes) {
  $ErrorActionPreference = 'Continue'
  $output = (pnputil /remove-device $node.PNPDeviceID 2>&1 | Out-String)
  $ErrorActionPreference = 'Stop'
  Write-Step ("removed device node " + $node.PNPDeviceID)
}

# 2. The staged package.
$listing = (pnputil /enum-drivers 2>&1 | Out-String)
$published = $null
if ($listing -match '(?s)Published Name\s*:\s*(oem\d+\.inf)(?=.{0,400}?virtualserial2um)') {
  $published = $Matches[1]
}
if (-not $published) {
  foreach ($block in ($listing -split '(?=Published Name)')) {
    if ($block -match 'virtualserial2um' -and $block -match '(oem\d+\.inf)') { $published = $Matches[1]; break }
  }
}
if ($published) {
  $ErrorActionPreference = 'Continue'
  pnputil /delete-driver $published /uninstall 2>&1 | Out-Null
  $ErrorActionPreference = 'Stop'
  Write-Step "deleted driver package $published"
} else {
  Write-Step 'no staged driver package found'
}

# 3. The certificate.
$cer = Join-Path $DriverPackage 'ElModemTestSigning.cer'
if ($KeepCertificate) {
  Write-Step 'keeping the signing certificate trusted (-KeepCertificate)'
} elseif ((Test-Path $cer) -or $RemoveCertificate) {
  if (-not (Test-Path $cer)) {
    Write-Step 'no certificate file present; pass -RemoveCertificate to remove by thumbprint instead'
    exit 0
  }
  $signingCertificate = New-Object System.Security.Cryptography.X509Certificates.X509Certificate2 $cer
  foreach ($storeName in @('Root', 'TrustedPublisher')) {
    $store = New-Object System.Security.Cryptography.X509Certificates.X509Store $storeName, 'LocalMachine'
    $store.Open('ReadWrite')
    try {
      $found = @($store.Certificates | Where-Object { $_.Thumbprint -eq $signingCertificate.Thumbprint })
      foreach ($one in $found) {
        # X509Store.Remove is the call that persists. Mutating
        # store.Certificates reports no error and writes nothing.
        $store.Remove($one)
        Write-Step "removed $($signingCertificate.Thumbprint) from $storeName"
      }
      if (-not $found) { Write-Step "$storeName did not trust $($signingCertificate.Thumbprint)" }
    } finally { $store.Close() }
  }
} else {
  Write-Step 'no certificate found in the driver package; nothing to remove'
}

$remaining = Get-ModemNodes
Write-Step ("remaining device nodes: " + $remaining.Count)

# Confirm the certificate really left the stores, rather than trusting a call
# that reports success without writing.
foreach ($storeName in @('Root', 'TrustedPublisher')) {
  $store = New-Object System.Security.Cryptography.X509Certificates.X509Store $storeName, 'LocalMachine'
  $store.Open('ReadOnly')
  try {
    $still = @($store.Certificates | Where-Object { $_.Thumbprint -eq $signingCertificate.Thumbprint })
    if ($still) { Write-Step "WARNING: $storeName still trusts the certificate"; exit 1 }
  } finally { $store.Close() }
}
Write-Step 'certificate no longer trusted in either store'
Write-Step 'a reboot clears any stale COM port name left behind'
exit 0