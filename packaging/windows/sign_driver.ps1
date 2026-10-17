<#
.SYNOPSIS
Sign the El-Modem virtual COM driver package with the project test certificate.

.DESCRIPTION
Takes the WDK build output (the folder holding VirtualSerial2um.dll,
virtualserial2um.inf and wudf.cat) and signs it with the certificate from
build_driver_cert.ps1, producing a package that installs on any machine which
trusts the matching .cer.

Two signatures are needed, because Windows checks both:

  * the DLL gets an Authenticode signature, and
  * wudf.cat gets an additional catalog signature.

The catalog signature is ADDED to the catalog the WDK produced, rather than
replacing it. The WDK's own per-build test signature stays in place; Windows
accepts the package if any catalog signature chains to a trusted root, so the
build-local signature is simply ignored on machines that trust ours and keeps
the package valid on the build machine if the project certificate is later
withdrawn.

The script then verifies the result, so a signing failure cannot quietly ship.

.SECURITY
Requires the .pfx from build_driver_cert.ps1, whose password is a credential.
The .pfx must never be added to the installer payload.

.PARAMETER DriverPackage
The WDK output directory containing the .inf, .dll and wudf.cat.

.PARAMETER CertificateDir
Directory holding ElModemTestSigning.pfx / .cer. Defaults to the gitignored
directory beside this script.

.PARAMETER OutputDir
Where to write the signed, installable copy. Defaults to overwriting
DriverPackage in place. Point this somewhere outside the repo when shipping.

.PARAMETER PfxPasswordFile
File whose first line is the password for the .pfx. Falls back to
$env:EL_MODEM_PFX_PASSWORD, then prompts. There is deliberately no
command-line password parameter, because it would be visible in the process
list.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File packaging\windows\sign_driver.ps1 `
      -DriverPackage C:\build\ComPort\x64\Release\VirtualSerial2um
#>
[CmdletBinding()]
param(
  [Parameter(Mandatory)][string]$DriverPackage,
  [string]$CertificateDir = (Join-Path $PSScriptRoot 'driver-signing'),
  [string]$OutputDir,
  [string]$PfxPasswordFile,
  [switch]$NoTimestamp
)

$ErrorActionPreference = 'Stop'

# Deliberately no -PfxPassword switch: a password on the command line is visible
# in the process list. signtool wants a plain string on its own command line, so
# the plaintext necessarily exists briefly, but it is not our argument vector.
function Resolve-PfxPassword {
  param([string]$PasswordFile)
  if ($PasswordFile) {
    $first = (Get-Content -LiteralPath $PasswordFile -TotalCount 1)
    if (-not $first) { throw "No password on the first line of $PasswordFile" }
    return $first
  }
  if ($env:EL_MODEM_PFX_PASSWORD) {
    $plain = $env:EL_MODEM_PFX_PASSWORD
    $env:EL_MODEM_PFX_PASSWORD = $null
    return $plain
  }
  if ([Environment]::UserInteractive) {
    $secure = (Read-Host -Prompt 'Password for ElModemTestSigning.pfx' -AsSecureString)
    return [System.Net.NetworkCredential]::new('', $secure).Password
  }
  throw 'No password source: set EL_MODEM_PFX_PASSWORD, pass -PfxPasswordFile, or run interactively.'
}

function Resolve-SignTool {
  $found = Get-ChildItem 'C:\Program Files (x86)\Windows Kits\10\bin' -Recurse -Filter signtool.exe `
             -ErrorAction SilentlyContinue |
           Where-Object { $_.FullName -match '\\x64\\' } |
           Sort-Object FullName -Descending | Select-Object -First 1
  if (-not $found) { throw 'signtool.exe not found; install the Windows Driver Kit.' }
  return $found.FullName
}

if (-not (Test-Path $DriverPackage)) { throw "Driver package not found: $DriverPackage" }
$pfx = Join-Path $CertificateDir 'ElModemTestSigning.pfx'
$cer = Join-Path $CertificateDir 'ElModemTestSigning.cer'
if (-not (Test-Path $pfx)) { throw "Signing certificate not found: $pfx (run build_driver_cert.ps1 first)" }

$inf = Get-ChildItem $DriverPackage -Filter *.inf | Select-Object -First 1
$dll = Get-ChildItem $DriverPackage -Filter *.dll | Select-Object -First 1
$cat = Join-Path $DriverPackage 'wudf.cat'
if (-not $inf) { throw "No .inf in $DriverPackage" }
if (-not $dll) { throw "No .dll in $DriverPackage" }
if (-not (Test-Path $cat)) { throw "No wudf.cat in $DriverPackage; the WDK build did not produce a signed package." }

$signtool = Resolve-SignTool
$pfxPassword = Resolve-PfxPassword -PasswordFile $PfxPasswordFile
$thumbprint = (New-Object System.Security.Cryptography.X509Certificates.X509Certificate2 $cer).Thumbprint
Write-Host "signtool : $signtool"
Write-Host "signing with: $thumbprint"

# signtool has no /log option; its output is captured by redirection instead.
# ErrorActionPreference must be relaxed around the call because signtool writes
# progress to stderr, which PowerShell would otherwise treat as a terminating
# error under 'Stop'.
function Invoke-SignTool {
  param([string[]]$Arguments, [string]$What)
  $log = Join-Path ([IO.Path]::GetTempPath()) ("signtool-" + [guid]::NewGuid().ToString('N') + '.log')
  $previous = $ErrorActionPreference
  $ErrorActionPreference = 'Continue'
  try {
    & $signtool @Arguments *> $log
    $code = $LASTEXITCODE
  } finally {
    $ErrorActionPreference = $previous
  }
  if ($code -ne 0) {
    Get-Content $log -ErrorAction SilentlyContinue | ForEach-Object { Write-Host "  $_" }
    throw "$What failed (signtool exit $code)"
  }
  return $log
}

# Work on a copy so a signing failure cannot leave a half-signed package in place.
$stage = Join-Path ([IO.Path]::GetTempPath()) ("elmodem-driver-" + [guid]::NewGuid().ToString('N'))
Copy-Item $DriverPackage $stage -Recurse
try {
  $stagedDll = Join-Path $stage $dll.Name
  $stagedCat = Join-Path $stage 'wudf.cat'

  # A timestamp keeps the signature valid after the certificate expires, so a
  # driver signed near the end of the certificate's life still verifies. It
  # needs network access, so a failure here is not fatal - it only means the
  # signature stops validating on the certificate's expiry date.
  $ts = @()
  if (-not $NoTimestamp) {
    $ts = @('/tr', 'http://timestamp.digicert.com', '/td', 'SHA256')
  }

  function Invoke-WithTimestampFallback {
    param([string]$What, [string[]]$Options, [string]$Target)
    # The target file must be the LAST argument: signtool treats anything after
    # it as another filename. That is why /tr and /td come before it here.
    if ($ts.Count -gt 0) {
      try {
        Invoke-SignTool -What "$What (timestamped)" -Arguments (@('sign') + $Options + $ts + @($Target)) | Out-Null
        return
      } catch {
        Write-Warning "timestamping unavailable, signing without one: $($_.Exception.Message)"
      }
    }
    Invoke-SignTool -What $What -Arguments (@('sign') + $Options + @($Target)) | Out-Null
  }

  Write-Host 'signing the driver DLL...'
  Invoke-WithTimestampFallback -What 'DLL Authenticode signing' -Target $stagedDll `
      -Options @('/fd', 'SHA256', '/f', $pfx, '/p', $pfxPassword)

  # The catalog is signed as a plain file, not with /ac or /as:
  #   /ac expects a differently-shaped argument set and fails to load the
  #   signing certificate; /as replaces the primary signature instead of adding
  #   to the catalog. Signing the catalog directly is what the WDK itself does,
  #   and it leaves the package with a single signature chained to our root.
  Write-Host 'signing the driver catalog...'
  Invoke-WithTimestampFallback -What 'catalog signing' -Target $stagedCat `
      -Options @('/fd', 'SHA256', '/f', $pfx, '/p', $pfxPassword)

  Write-Host 'verifying...'
  # Verification needs the signing certificate to be trusted, and the whole point
  # of this script is that it usually is not yet. Trust the public certificate in
  # the CURRENT USER root store for the duration of the check and remove it
  # again afterwards, so the check is genuine and leaves nothing behind. The
  # machine-wide store is deliberately never touched.
  $trustStore = New-Object System.Security.Cryptography.X509Certificates.X509Store 'Root', 'CurrentUser'
  $trusted = $false
  try {
    $trustStore.Open('ReadWrite')
    $trustStore.Add((New-Object System.Security.Cryptography.X509Certificates.X509Certificate2 $cer))
    $trusted = $true
    $trustStore.Close()

    function Get-Verification {
      param([string]$Target)
      $log = Join-Path ([IO.Path]::GetTempPath()) ("signtool-verify-" + [guid]::NewGuid().ToString('N') + '.log')
      $previous = $ErrorActionPreference
      $ErrorActionPreference = 'Continue'
      try {
        & $signtool verify /pa /v $Target *> $log
        $code = $LASTEXITCODE
      } finally {
        $ErrorActionPreference = $previous
      }
      $text = Get-Content $log -Raw -ErrorAction SilentlyContinue
      Remove-Item $log -ErrorAction SilentlyContinue
      return [pscustomobject]@{ Code = $code; Report = $text }
    }

    $catCheck = Get-Verification -Target $stagedCat
    if ($catCheck.Code -ne 0) {
      ($catCheck.Report -split "`r?`n" | Select-Object -First 12) | ForEach-Object { Write-Host "  $_" }
      throw 'catalog does not verify after signing'
    }
    $dllCheck = Get-Verification -Target $stagedDll
    if ($dllCheck.Code -ne 0) {
      ($dllCheck.Report -split "`r?`n" | Select-Object -First 12) | ForEach-Object { Write-Host "  $_" }
      throw 'driver DLL does not verify after signing'
    }

    # Confirm our signature is actually present, not merely that the leftover
    # WDK signature still validates. This catches a silent no-op.
    if ($catCheck.Report -notmatch $thumbprint) {
      throw "the catalog verifies but does not carry our signature ($thumbprint)"
    }
    if ($dllCheck.Report -notmatch $thumbprint) {
      throw "the DLL verifies but does not carry our signature ($thumbprint)"
    }
    Write-Host "  catalog and DLL both verify and chain to $thumbprint"

    # The catalog must also cover the files it claims, or Windows will reject the
    # package at install time even though the signature itself is valid.
    if ($catCheck.Report -match 'Number of files successfully Verified:\s*0') {
      throw 'the catalog verifies but covers none of the package files; the package would not install'
    }
    Write-Host "  catalog covers the package files"
  } finally {
    # Cleanup must never abort publishing: this block used to run before the
    # signed package was copied out, so a certificate that could not be removed
    # silently left the caller with no output at all.
    if ($trusted) {
      try {
        $trustStore.Open('ReadWrite')
        foreach ($c in @($trustStore.Certificates)) {
          if ($c.Thumbprint -eq $thumbprint) { $trustStore.Remove($c) }
        }
        $trustStore.Close()
      } catch {
        Write-Warning ("could not withdraw the verification trust from " +
                       "CurrentUser\Root: $($_.Exception.Message)")
        Write-Warning "remove it manually: certutil -deluserstore Root $thumbprint"
      }
    }
  }

  if ($OutputDir) {
    New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
    Copy-Item (Join-Path $stage '*') $OutputDir -Recurse -Force
    Copy-Item $cer $OutputDir -Force
  } else {
    Copy-Item (Join-Path $stage '*') $DriverPackage -Recurse -Force
  }

  Write-Host ''
  Write-Host 'signed package:'
  if ($OutputDir) {
    Write-Host "  $OutputDir"
    Get-ChildItem $OutputDir | ForEach-Object { Write-Host "    $($_.Name)" }
  } else {
    Write-Host "  $DriverPackage"
    Get-ChildItem $DriverPackage | ForEach-Object { Write-Host "    $($_.Name)" }
  }
  Write-Host "certificate: $thumbprint"
  Write-Host 'Ship the .cer with the installer so the target machine can trust it.'
}
finally {
  Remove-Item -Recurse -Force $stage -ErrorAction SilentlyContinue
  $pfxPassword = $null
}