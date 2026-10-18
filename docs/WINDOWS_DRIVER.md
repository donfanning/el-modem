# Installing the El-Modem virtual COM driver on Windows

## Availability

The public `v0.1.0-rc1-windows` ZIP contains no virtual-COM driver. This
repository includes a reviewed [MS-PL source overlay](../packaging/windows-driver/README.md)
and pinned, hash-checked reconstruction helper. It fetches 14 Microsoft baseline
files and reconstructs 16 source/project files; no full vendor tree, signing key,
certificate or driver binary is distributed.

The whole chain is qualified on one host (Windows 11 build 26340, x64):
reconstruct, build with the EWDK, sign, install and exchange bytes both ways,
with the installed binary byte-identical to the signed one. That is one host, one
certificate and one Windows build, and it says nothing about other Windows builds
or about a publicly signed package. Two ports on separate pipe names were then
measured routing independently and byte-exact, with no bytes crossing between
them; that is what [Running more than one
port](#running-more-than-one-port) relies on, and three or more ports are
untested rather than disproven. The port's
own signal surface is measured on that host -
CTS/DSR/RLSD asserted, RI idle, DTR/RTS reported through
`IOCTL_SERIAL_GET_DTRRTS` - but the pipe protocol carries no signal lines, so
nothing propagates between the port and the modem.
TCP and the separately installed com0com route in [SERIAL.md](SERIAL.md) remain
available without building this driver.
`pipe` is an opt-in integration for an already available compatible driver.
Its byte protocol does not carry real DTR/DCD/RI/CTS transitions.
One port owns one pipe: a port with no name of its own uses
`\\.\pipe\ElModemSerial`, so a second port needs its own name or it will not
reach a modem at all - see
[Running more than one port](#running-more-than-one-port) for the procedure and
for what an unconfigured second port does instead.

The El-Modem virtual COM driver presents a real COM port and brokers bytes
between that port and El-Modem over `\\.\pipe\ElModemSerial`. It replaces the
older arrangement that needed com0com plus a TCP redirector: there is no peer
port and no second piece of third-party software.
`el-modem --serial-backend pipe` then talks to it.

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

With Python 3 and Git installed, reconstruct a fresh source directory first:

```powershell
python packaging/windows-driver/prepare_source.py C:\dev\el-modem-driver
cd C:\dev\el-modem-driver
```

Then, from a Visual Studio developer shell with the matching SDK/WDK installed:

```powershell
msbuild VirtualSerial.sln /t:Rebuild /p:Configuration=Release /p:Platform=x64
```

The reconstruction helper overrides `core.autocrlf` for its patch application,
so you do not need to change your global Git configuration. If reconstruction
still stops at `Overlay result mismatch`, verify that you are using the current
helper and a fresh destination, then report the mismatching path:

```powershell
python packaging/windows-driver/prepare_source.py C:\dev\el-modem-driver
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

## Running more than one port

One port owns one pipe, and the pipe name is how the port and El-Modem find each
other. A port with no name of its own uses `\\.\pipe\ElModemSerial`; the first
port on a machine should keep that, because every existing configuration does.
Each additional port needs a name of its own, and El-Modem pointed at the same
name.

The name lives in the port's own `Device Parameters` key, beside the `PortName`
Windows assigned, so it can differ per port. A bare name (`ElModemSerial2`) or a
full path (`\\.\pipe\ElModemSerial2`) is accepted, matching `--serial-device`.

The driver reads it while the device is being added, so the port has to be
restarted afterwards for the value to take effect. `Disable-PnpDevice` followed
by `Enable-PnpDevice` does that, and the port came back clean (problem 0) on the
qualification host; re-creating the node is not required. This is a value of the
driver's own rather than `PortName`, which is what the `CM_PROB_FAILED_ADD`
warning above is about. If a port does not come back, delete the value and
restart it again.

Elevated PowerShell, for a second port that came up as `ROOT\PORTS\0001`:

```powershell
$key = 'HKLM:\SYSTEM\CurrentControlSet\Enum\ROOT\PORTS\0001\Device Parameters'
Set-ItemProperty -Path $key -Name 'ElModemPipeName' -Value 'ElModemSerial24' -Type String
Disable-PnpDevice -InstanceId 'ROOT\PORTS\0001' -Confirm:$false
Enable-PnpDevice  -InstanceId 'ROOT\PORTS\0001' -Confirm:$false
```

Then run a second modem against that name:

```powershell
el-modem --serial-backend pipe --serial-device ElModemSerial24
```

Confirm the pairing in the driver's log in `C:\Users\Public`, which names the
pipe and the port on every line:

```text
61621687 pipe=\\.\pipe\ElModemSerial port=COM3 connected error=0
61621843 pipe=\\.\pipe\ElModemSerial24 port=COM24 connected error=0
```

**A second port left unconfigured does not fail visibly.** It cannot attach to a
pipe another port already owns, and the sample's AT responder stays armed, so a
terminal on it dials, is told `CONNECT`, and is talking to the driver rather than
to a modem. That case is the one that logs `CreateFile failed error=231`
(`ERROR_PIPE_BUSY`) against the shared name. Two ports on two names were measured
byte-exact in both directions with no bytes crossing between them; three or more
are untested, and the limits under [Availability](#availability) still apply.

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

This source release supplies no production-signed or test-signed driver package.
Build locally and use your own development test certificate. Source availability
does not change Windows signing, trust or boot-policy requirements. No certificate
private key is included in this repository.

## Project test certificate and installer integration

The following optional helpers create a local development certificate and sign
a local package. Retain its private key outside source control; this is not a
production-signing workflow or permission to redistribute a project signing key.

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
