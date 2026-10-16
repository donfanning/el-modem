<#
.SYNOPSIS
Create the El-Modem project test-signing certificate for the virtual COM driver.

.DESCRIPTION
The WDK's own test signing generates a throwaway certificate per build, kept in
the builder's CurrentUser\My store. That is fine for a developer machine but
useless for distribution: every build produces a different certificate, so a
prebuilt driver package cannot be installed by anyone else.

This script creates one long-lived, self-signed code-signing certificate for the
project, exports its public half for shipping, and keeps the private half in a
gitignored directory for signing with.

  Public half  driver-signing\ElModemTestSigning.cer  -> shipped in the installer
  Private half driver-signing\ElModemTestSigning.pfx  -> NEVER shipped

.SECURITY
The private key is the whole trust anchor. Anyone holding it can sign a driver
or catalog that every machine which trusts ElModemTestSigning.cer will accept,
without going through kernel code integrity review. Therefore:

  * driver-signing\ (this script's output) must stay out of git and out of the
    installer. .gitignore covers the first; the installer build must not add the
    .pfx to its [Files].
  * Treat the .pfx like any other credential: a secret manager or an offline
    machine, never a build artefact.
  * This is a TEST certificate. It exists only until the driver carries a public
    signature. Remove it then, and remove it from machines when uninstalling.

.PARAMETER OutputDir
Where to write the certificate. Defaults to a gitignored directory beside this
script.

.PARAMETER SubjectName
Certificate subject. Left distinctive so that anyone inspecting their trusted
root stores can tell what it is.

.PARAMETER Years
Validity period. The certificate cannot outlive its usefulness: once the driver
is publicly signed this is deleted.

.PARAMETER PfxPasswordFile
File whose first line is the password for the exported .pfx. Falls back to
$env:EL_MODEM_PFX_PASSWORD, then prompts interactively. There is deliberately
no command-line password parameter, because it would be visible in the process
list.

.PARAMETER Force
Replace an existing certificate for the same subject rather than reusing it.
Replacing changes the thumbprint, so any driver already signed with the old one
stops validating and every machine that trusted it needs the new .cer.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File packaging\windows\build_driver_cert.ps1 -PfxPassword (Read-Host -AsSecureString)
#>
[CmdletBinding()]
param(
  [string]$OutputDir = (Join-Path $PSScriptRoot 'driver-signing'),
  [string]$SubjectName = 'CN=El-Modem Test Signing, O=El-Modem Project',
  [int]$Years = 3,
  [string]$PfxPasswordFile,
  [switch]$Force
)

$ErrorActionPreference = 'Stop'

# Built with .NET rather than ConvertTo-SecureString: that cmdlet lives in
# Microsoft.PowerShell.Security, which does not autoload in every
# non-interactive or constrained host, and the PKI cmdlets themselves need the
# same module, so relying on it here would make the script fail for an
# avoidable reason.
function ConvertTo-SecureStringLocal {
  param([string]$Plain)
  $secure = New-Object System.Security.SecureString
  foreach ($character in $Plain.ToCharArray()) { $secure.AppendChar($character) }
  $secure.MakeReadOnly()
  return $secure
}

# Deliberately no -PfxPassword switch: a password on the command line is visible
# in the process list to any other process on the machine. Use the environment
# variable, a file, or the interactive prompt.
function Resolve-PfxPassword {
  param([string]$PasswordFile)
  if ($PasswordFile) {
    $first = (Get-Content -LiteralPath $PasswordFile -TotalCount 1)
    if (-not $first) { throw "No password on the first line of $PasswordFile" }
    return ConvertTo-SecureStringLocal -Plain $first
  }
  if ($env:EL_MODEM_PFX_PASSWORD) {
    $plain = $env:EL_MODEM_PFX_PASSWORD
    $env:EL_MODEM_PFX_PASSWORD = $null   # do not leave it in the environment
    return ConvertTo-SecureStringLocal -Plain $plain
  }
  if ([Environment]::UserInteractive) {
    return (Read-Host -Prompt 'Password for the exported .pfx' -AsSecureString)
  }
  throw 'No password source: set EL_MODEM_PFX_PASSWORD, pass -PfxPasswordFile, or run interactively.'
}

if (-not (Get-Command New-SelfSignedCertificate -ErrorAction SilentlyContinue)) {
  throw 'New-SelfSignedCertificate is unavailable; the PKI module is required.'
}

# The Cert: PSDrive is absent in some constrained hosts and non-interactive
# sessions, so the store is reached through X509Store directly rather than
# through the provider path.
function Get-CertificateStore {
  param([string]$Name, [string]$Location = 'CurrentUser')
  $store = New-Object System.Security.Cryptography.X509Certificates.X509Store $Name, $Location
  $store.Open('ReadOnly')
  return $store
}

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
$cerPath = Join-Path $OutputDir 'ElModemTestSigning.cer'
$pfxPath = Join-Path $OutputDir 'ElModemTestSigning.pfx'
$infoPath = Join-Path $OutputDir 'signing-info.json'

$store = Get-CertificateStore -Name 'My'
try {
  $existing = @($store.Certificates | Where-Object {
    $_.Subject -eq $_.Issuer -and $_.Subject -eq $SubjectName -and $_.NotAfter -gt (Get-Date)
  })
} finally {
  $store.Close()
}

if ($existing -and -not $Force) {
  $cert = $existing[0]
  Write-Host "Reusing existing certificate: $($cert.Thumbprint) (expires $($cert.NotAfter))"
  Write-Host 'Pass -Force to replace it. Replacing invalidates anything already signed with it.'
} else {
  if ($existing) { Write-Host 'Replacing the existing certificate (-Force).' }
  # RSA-3072/SHA-256: the WDK's default test cert still uses SHA-1, which older
  # policy paths reject. Exportable so sign_driver.ps1 can use the .pfx.
  # No -CertStoreLocation: the default is already the current user's My store.
  $cert = New-SelfSignedCertificate `
    -Type CodeSigningCert `
    -Subject $SubjectName `
    -FriendlyName 'El-Modem Test Signing' `
    -KeyAlgorithm RSA `
    -KeyLength 3072 `
    -HashAlgorithm SHA256 `
    -KeyExportPolicy Exportable `
    -NotAfter (Get-Date).AddYears($Years)
  Write-Host "Created certificate: $($cert.Thumbprint) (expires $($cert.NotAfter))"
}

$password = Resolve-PfxPassword -PasswordFile $PfxPasswordFile
Export-Certificate -Cert $cert -FilePath $cerPath -Type CERT -Force | Out-Null
Export-PfxCertificate -Cert $cert -FilePath $pfxPath -Password $password | Out-Null

[pscustomobject]@{
  thumbprint    = $cert.Thumbprint
  subject       = $cert.Subject
  not_before    = $cert.NotBefore.ToString('o')
  not_after     = $cert.NotAfter.ToString('o')
  public_cer    = $cerPath
  private_pfx   = $pfxPath
  signing_info  = $infoPath
} | ConvertTo-Json | Set-Content -LiteralPath $infoPath -Encoding ascii

Write-Host ''
Write-Host 'Public certificate (ship this):' $cerPath
Write-Host 'Private key (NEVER ship this):' $pfxPath
Write-Host 'Signing details (no secrets):' $infoPath
Write-Host ''
Write-Host 'Next: packaging\windows\sign_driver.ps1 -DriverPackage <build output> -PfxPassword ...'
Write-Host 'Anyone holding the .pfx can sign drivers this certificate makes trusted. Protect it accordingly.'