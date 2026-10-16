# Windows COM qualification

> **Historical record.** This documents the `v0.1.0-rc1` test of the com0com
> arrangement and is kept as evidence of what was actually verified. It is
> superseded for day-to-day use by the bundled virtual-COM driver; see
> [WINDOWS_DRIVER.md](WINDOWS_DRIVER.md) and
> [SERIAL.md](SERIAL.md#windows-bundled-virtual-com-driver). The COM41/COM42
> port numbers below refer to that specific run and are not a current
> arrangement. The captured machine output in
> [windows-com-evidence/](windows-com-evidence/) is likewise unmodified.

The published `v0.1.0-rc1` Windows executable passed a real serial-terminal
command test on Windows Server 2022 Standard Evaluation (`10.0.20348`) in a
local KVM VM. Kernel code integrity was enabled and test-signing was disabled
(effective options `1`). The original executable was not rebuilt or modified.

Path:

```text
Tera Term 5.7.1 → COM41 ↔ COM42 (com0com) → com2tcp → TCP DTE → packaged modem worker
```

The signed com0com 3.0.0.0 driver loaded, all three devices reported error code
zero, and a COM-pair transfer passed. Tera Term then exercised `ATZ`, `ATE0`,
`ATI3`, `ATS0=9`, readback (`9`), reset, and readback (`0`). The modem exited
cooperatively; the redirector was stopped before temporary files were removed.
The owned COM pair and temporary verified-publisher consent were removed.

See the [report and exact identities](WINDOWS_COM_QUALIFICATION.json),
[terminal transcript](windows-com-evidence/terminal.log),
[kernel policy](windows-com-evidence/code-integrity.json),
[device status](windows-com-evidence/devices.json), and
[cleanup](windows-com-evidence/driver-cleanup.json).
Microsoft VC x64 runtime `14.44.35211` was installed for the test.

## Reproducing the command path

Use the official [com0com signed package](https://sourceforge.net/projects/com0com/files/com0com/3.0.0.0/),
[com2tcp](https://sourceforge.net/projects/com0com/files/com2tcp/1.3.0.0/), and
[Tera Term](https://github.com/TeraTermProject/teraterm/releases/tag/v5.7.1).
They are separate dependencies, not bundled with the modem.

The source checkout's [driver probe](../packaging/windows/probe_com_driver.ps1)
provides the tested installation route: 7-Zip extracts the unchanged signed
package; Windows validates its catalog; verified vendor publisher consent
allows unattended installation through pnputil/setupc. It requires an elevated
PowerShell, 7-Zip in PATH, an existing writable `RUNNER_TEMP` directory, and
`EL_MODEM_REQUIRE_NORMAL_CI=1` to reject a test-mode fixture. It creates COM41/42
and leaves them available for the terminal check. Review it before running.
It does not enable test-signing, change trusted roots or disable enforcement.

Start a modem with a loopback TCP DTE endpoint (for example port 9100), then:

```text
com2tcp.exe \\.\COM42 127.0.0.1 9100
```

Open COM41 in Tera Term at 115200 baud, with flow control set to none. Use raw
TCP on the bridge: the modem endpoint is not Telnet. Send the commands above
and confirm identity and register readback. The automated
[terminal harness](../packaging/windows/qualify_com_terminal.py) exercises this
path against the digest-pinned published executable. The
[cleanup script](../packaging/windows/cleanup_com_driver.ps1) removes only the
pair and publisher consent owned by the probe.

## Limits

This qualifies the serial command path, not a connected DSP transfer or a PSTN
call. Raw TCP does not convey full DCD/RI/CTS signaling; TCP disconnect supplies
the modem's DTR-drop event. No Windows kernel driver is shipped by this project.
Windows 10/11, Secure Boot/HVCI configurations, and the original interactive
Windows installer remain separate qualification work.

Hosted Server 2022 and 2025 tests also passed, but their preconfigured kernels
allowed test-signing. They do not establish normal-mode installation. The local
normal-mode result closes that gap only for the stated stock BIOS VM profile.

A report receiver initially rejected an empty log, causing unintended fixture
retries. The final reports were recovered directly after fixing zero-byte
uploads. Only one retained final normal-mode evidence set is claimed; those
retries are not a documented repeated-run stability campaign.
