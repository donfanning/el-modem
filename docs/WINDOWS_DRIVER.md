# Installing the El-Modem virtual COM driver on Windows

## Availability

The public `v0.1.0-rc1-windows` ZIP contains no virtual-COM driver. This
repository currently contains the Python pipe backend and driver packaging
tools, but **not** the modified UMDF2 solution, INF or C++ source. The build
instructions below require that separate modified driver project; cloning this
repository alone cannot build it. Microsoft's original VirtualSerial2 sample
does not by itself establish compatibility with our named-pipe protocol.

Until the modified source is supplied and its complete path is qualified, use
TCP or the separately installed com0com route in [SERIAL.md](SERIAL.md).
`pipe` is an opt-in integration for an already available compatible driver.
Its byte protocol does not carry real DTR/DCD/RI/CTS transitions, and one driver
using `ElModemSerial` serves one modem only. Additional modems require separate
driver routing and distinct pipe names; that arrangement is not qualified.

The El-Modem virtual COM driver presents a real COM port to terminal programs
(PuTTY, ZOC, SyncTerm, TTerm, Terminate) and brokers bytes between that port and
El-Modem over `\\.\pipe\ElModemSerial`. It replaces the older arrangement that
needed com0com plus a TCP redirector: there is no peer port and no second piece
of third-party software. `el-modem --serial-backend pipe` then talks to it.

The driver is derived from Microsoft's `serial/VirtualSerial2` UMDF2 sample and
runs in user mode inside `WUDFHost.exe`.

> **Signing status.** The driver is currently **test-signed**, not publicly
> signed. A test-signed package only installs on a machine that trusts the
> signing certificate, so it will not install on a stock Windows install with
> Secure Boot until a public signature exists. See
> [Signing](#signing-and-removing-the-test-certificate) for how to trust it
> safely, and remove it when you are finished.

## Requirements

- Windows 10 or 11, x64.
- Visual Studio 2022 Build Tools with the **Desktop development with C++** workload.
- The **Windows Driver Kit**, matching the Windows SDK version (currently WDK
  10.0.26100.6584). The driver does not build under Visual Studio 2019; the WDK
  no longer supports it.
- Install the WDK **after** Visual Studio 2022, so it integrates with the
  instance the driver will be built from. See
  [If the build fails with MSB8020](#if-the-build-fails-with-msb8020).

## Build

From the separate modified driver's source root (not included here; the
directory holding `VirtualSerial.sln`):

```powershell
msbuild VirtualSerial.sln /t:Rebuild /p:Configuration=Release /p:Platform=x64
```

The WDK test-signs the output automatically. You get an installable package and
the certificate you will need next:

| Path (under `ComPort\x64\Release\`) | Contents |
| --- | --- |
| `VirtualSerial2um\` | `VirtualSerial2um.dll`, `virtualserial2um.inf`, `wudf.cat` — the installable package |
| `VirtualSerial2um.cer` | the certificate that signed `wudf.cat` |

### If the build fails with MSB8020

```
error MSB8020: The build tools for WindowsUserModeDriver10.0
  (Platform Toolset = 'WindowsUserModeDriver10.0') cannot be found.
```

This is the common failure, and it is **not** a problem with the project. The
driver targets the `WindowsUserModeDriver10.0` platform toolset, which Visual
Studio does not ship. It arrives as the **Windows Driver Kit VS extension** — a
VSIX installed into the Visual Studio instance — so having VS2022 Build Tools
and the WDK both installed is not by itself enough.

The staged extension is normally at:

```
C:\Program Files (x86)\Windows Kits\10\Vsix\VS2022\<kit version>\WDK.vsix
```

Check its manifest before trying to install it, because the folder name is not
the authority. A VS2019-era extension declares:

```xml
<InstallationTarget Version="[16.0,17.0)" Id="Microsoft.VisualStudio.Community" />
```

`[16.0,17.0)` excludes Visual Studio 2022 (v17) outright, and such a VSIX cannot
be installed into VS2022 at all. A VS2022-compatible one declares `[17.0,18.0)`.

To check what is registered:

```powershell
Get-ChildItem "$env:ProgramFiles(x86)\Microsoft Visual Studio\2022\BuildTools\MSBuild\Microsoft\VC\v170\Platforms\x64\PlatformToolsets"
```

`WindowsUserModeDriver10.0` must appear alongside `v143`.

The documented remedy when the extension is missing or too old is to remove both
the WDK and the "Windows Driver Kit" extension, reboot, and reinstall the WDK, so
that it stages and installs the extension matching your Visual Studio. An
extension left over from an older WDK can otherwise satisfy the installer's
detection while providing nothing usable. The **Enterprise WDK (EWDK)** avoids
this entirely: it is a standalone, self-contained build environment with the
build tools, SDK and WDK together, and needs no Visual Studio integration.

`wdf.h` lives under `Windows Kits\10\Include\wdf\umdf\<version>`, not under
`Include\<version>\km`. If the driver cannot find it, the WDK's WDF header
payload is not installed; that ships as a separate MSI from the SDK headers.

## Trusting the test certificate

**Do this before installing**, or `pnputil /add-driver` will refuse the package.

The certificate is a self-signed certificate that the WDK generates for you at
build time. Its **private key stays on the build machine**, in
`CurrentUser\My` of the account that built the driver. Only the public half is
exported to the `.cer`, and only that public half goes into the trusted stores
below.

> ### Read this before you trust it
>
> Trusting this certificate means *any* driver or catalog signed by that one
> private key is accepted by this machine — including something malicious, and
> including test-signed code, which does **not** get the normal kernel
> code-integrity review.
>
> * Do this on a **test machine**, not on a workstation holding real work.
> * Do not build on one machine and trust on another unless you accept that
>   link. Build and test on the same machine, or ship the `.cer` deliberately.
> * Never import a `WDKTestCert` you found on the internet. Some certificates
>   with that naming pattern ship with public private keys, which would let
>   anyone sign code your machine trusts. Use only the `.cer` from **your own**
>   build output.
> * Remove the certificate when you are done — see
>   [cleanup](#uninstall-and-remove-the-test-certificate).

From an **elevated** PowerShell:

```powershell
$pkg = 'C:\path\to\driver\ComPort\x64\Release'
Import-Certificate -FilePath "$pkg\VirtualSerial2um.cer" -CertStoreLocation Cert:\LocalMachine\Root
Import-Certificate -FilePath "$pkg\VirtualSerial2um.cer" -CertStoreLocation Cert:\LocalMachine\TrustedPublisher
```

`certutil` works if the `Cert:` drive is unavailable:

```powershell
certutil -addstore -f Root "$pkg\VirtualSerial2um.cer"
certutil -addstore -f TrustedPublisher "$pkg\VirtualSerial2um.cer"
```

Importing a certificate into `LocalMachine\Root` pops a confirmation prompt
warning that you are adding a root of trust — that prompt is expected.

`TrustedPublisher` is what Microsoft's test-signing guidance lists. Testing on
this project found that `LocalMachine\Root` alone is what actually gates
installation, so importing both is belt-and-braces rather than a requirement.

## Installing the driver

Elevated PowerShell, with `$pkg` set as above:

```powershell
$inf = "$pkg\VirtualSerial2um\virtualserial2um.inf"

# Stage the signed package into the driver store.
pnputil /add-driver $inf

# Create the device node.
$devcon = 'C:\Program Files (x86)\Windows Kits\10\Tools\10.0.26100.0\x64\devcon.exe'
& $devcon install "$inf" 'UMDF\VirtualSerial2um'
```

The port gets the next free COM number. Do not pin it by writing `PortName`
into the registry: doing that after the device has started, followed by
`pnputil /restart-device`, makes the next add fail with `CM_PROB_FAILED_ADD`
and `Problem Status 0xC0000017` (`STATUS_NO_SUCH_DEVICE`), and no COM port
appears at all. Set a specific port number through **Device Manager** instead,
which goes through the Ports class installer rather than editing the hardware
key directly.

The manual procedure is therefore:

```powershell
$id = (Get-CimInstance Win32_PnPEntity |
       Where-Object { $_.Name -like '*El-Modem*' }).PNPDeviceID
pnputil /restart-device "$id"
[System.IO.Ports.SerialPort]::GetPortNames()
```

`install_driver.ps1` does exactly this: create the node, then report whichever
port appeared. It deliberately exposes no way to choose the number.

Installing the INF creates two device nodes; remove the extra one so exactly one
COM port is presented:

```powershell
$nodes = @(Get-CimInstance Win32_PnPEntity |
           Where-Object { $_.Name -like '*El-Modem*' } | Sort-Object PNPDeviceID)
$nodes | Select-Object -Skip 1 | ForEach-Object { pnputil /remove-device $_.PNPDeviceID }
```

## Verifying

```powershell
[System.IO.Ports.SerialPort]::GetPortNames()   # expect COM3
Get-PnpDevice | Where-Object FriendlyName -like '*El-Modem*'
```

A quick end-to-end check with Python:

```python
import serial
port = serial.Serial('COM3', 115200, timeout=1)
print(port.read(64))      # the driver's AT responder replies to input
port.close()
```

Then start the modem with `--serial-backend pipe` and point a terminal program
at COM3.

## Updating after a rebuild

`pnputil /restart-device` **does not reload the driver DLL**. The running copy
stays loaded from the staged package, so a rebuilt driver appears to have no
effect. Delete the package and stage it again:

```powershell
# Find the published package name for our driver.
$drv = & pnputil.exe /enum-drivers 2>&1 | Out-String
$pub = $null
if ($drv -match '(?s)Published Name\s*:\s*(oem\d+\.inf)(?=.{0,400}?virtualserial2um)') {
    $pub = $Matches[1]
}
if (-not $pub) {
    foreach ($b in ($drv -split '(?=Published Name)')) {
        if ($b -match 'virtualserial2um' -and $b -match '(oem\d+\.inf)') { $pub = $Matches[1]; break }
    }
}
if ($pub) { pnputil /delete-driver $pub /uninstall }

pnputil /add-driver $inf
```

Then recreate the device node with `devcon` as above. A reboot is the
simplest way to be certain nothing stale is still loaded; it also clears a stale
friendly name left over from a previous port number.

## Uninstalling and removing the test certificate

Elevated PowerShell:

```powershell
Get-CimInstance Win32_PnPEntity | Where-Object { $_.Name -like '*El-Modem*' } |
    ForEach-Object { pnputil /remove-device $_.PNPDeviceID }

# $pub is the published package name found above.
if ($pub) { pnputil /delete-driver $pub /uninstall }
```

Then remove the certificate, which is the part that actually restores your
machine's security posture:

```powershell
$thumb = (New-Object System.Security.Cryptography.X509Certificates.X509Certificate2 `
            "$pkg\VirtualSerial2um.cer").Thumbprint
foreach ($store in 'Root','TrustedPublisher') {
    $s = New-Object System.Security.Cryptography.X509Certificates.X509Store $store,'LocalMachine'
    $s.Open('ReadWrite')
    @($s.Certificates) | Where-Object { $_.Thumbprint -eq $thumb } | ForEach-Object { $s.Remove($_) }
    $s.Close()
}
```

Confirm it is gone:

```powershell
Get-ChildItem Cert:\LocalMachine\Root, Cert:\LocalMachine\TrustedPublisher |
    Where-Object Subject -like 'WDKTestCert*'
```

Reboot to clear any remaining device node.

## Signing

A public signature is what makes this installable without touching certificate
stores. The plan is a Microsoft-issued signature for an OSI-approved open
source project, which costs nothing to obtain; the fallback is a paid EV
certificate. Until one of those exists, test signing is the only route, and the
trust-the-certificate step above is unavoidable.

The driver source is not yet published in its own repository, so these
instructions currently refer to a local build tree.

## Project test certificate and installer integration

The per-build WDK certificate described above cannot be used for distribution:
every build produces a different certificate, so a prebuilt package is
installable only on the machine that built it. For shipping, the project keeps
**one** certificate instead.

```powershell
# 1. Create the project certificate once. Output goes to a gitignored directory.
$env:EL_MODEM_PFX_PASSWORD = 'a password of your choosing'
packaging\windows\build_driver_cert.ps1

# 2. Build the driver with the WDK, then sign the output with that certificate.
packaging\windows\sign_driver.ps1 -DriverPackage <WDK output>\VirtualSerial2um -OutputDir <payload>

# 3. Build the installer with the signed payload.
packaging\windows\build_installer.ps1 -Archive <candidate.zip> -Version <v> -Out installers `
    -DriverPackage <payload>
```

`build_driver_cert.ps1` writes three files to `packaging/windows/driver-signing/`:

| File | Purpose |
| --- | --- |
| `ElModemTestSigning.cer` | public certificate; **this is the only one that ships** |
| `ElModemTestSigning.pfx` | private key; **never commit and never ship** |
| `signing-info.json` | thumbprint and validity dates; no secrets |

The `.pfx` is in `.gitignore`. `build_installer.ps1` refuses to build if a
`.pfx` is found in the payload, so a key cannot be shipped by accident.

`sign_driver.ps1` signs both the driver DLL and `wudf.cat`, then **verifies the
result** before publishing it — it re-checks that the signature chains to the
project certificate, that it covers the package files, and that the trust it
just assumed is genuinely removed afterwards. A signing step that silently did
nothing would otherwise ship a package that no user can install.

### What the installer does

The installer copies the signed package to `{app}\driver`, and after the rest of
El-Modem is in place runs `{app}\driver\install_driver.ps1`, which trusts the
certificate, stages the package with `pnputil`, and creates the device node. On
uninstall it runs `uninstall_driver.ps1`, which removes the node, the package,
**and the certificate**. Removing the certificate is the point: while it is
trusted, the machine accepts anything signed with it.

Because that is a real change to the machine's trust, the installer asks first
and explains exactly what it is about to do. Declining leaves El-Modem fully
installed, just without the COM port. `/VERYSILENT` installs skip the prompt;
pass `/NoDriver`-style decisions through the log instead — check the install log
for `driver` lines.

If `devcon.exe` is not found (no Windows Driver Kit installed), the driver
package is still staged and the installer prints the single command needed to
finish by hand. El-Modem itself installs normally either way.

### What this costs

The signing key is held by the project. Anyone who obtains that `.pfx` can sign a
driver that every machine which installed El-Modem will trust, with no kernel
code integrity review. That is inherent to distributing a test-signed driver and
is the reason it is only acceptable as an interim measure:

* keep the `.pfx` offline or in a secret manager; never in the repository, CI
  logs, or an artefact store;
* the installer removes the certificate on uninstall, so trust does not persist;
* when the driver carries a public signature, delete this certificate,
  `build_driver_cert.ps1`, `sign_driver.ps1` and the installer's driver step.

Until then, treat "the driver is test-signed" as a known limitation rather than a
finished state.
