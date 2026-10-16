# Connecting a terminal program to el-modem

`el-modem` presents a Hayes-style AT modem. The Linux desktop defaults to native
virtual serial with the separately installed [driver prerequisite](../packaging/serial/README.md).
See [desktop configuration](DESKTOP.md). PTY and TCP are explicit alternatives;
their signal behavior differs from native virtual serial.

## Linux and macOS: explicit legacy pseudo-terminal interface
    el-modem --link ~/.el-modem0 --line-mode ms ...
    minicom -D ~/.el-modem0
The program prints the PTY name and, with `--link`, creates the symlink. Closing the terminal drops DTR; the modem hangs
up according to `AT&D` (default: hang up). Run minicom with a hang-up string of `~~+++~~ATH`.

The exact published macOS rc1 passed real minicom command sessions, close/reopen
on the same PTY and clean teardown on both Intel and ARM macOS15.7.9. See
[macOS qualification and usage](MACOS_PTY_QUALIFICATION.md). Disable minicom's
hardware/software flow control for this PTY. This does not qualify physical
modem-control signals or connected-call hangup behavior.

## Windows: bundled virtual-COM driver

`--serial-backend pipe` is the recommended Windows path. Ownership is inverted
relative to the `com` backend: the driver *is* the COM port, so terminal
programs (PuTTY, ZOC, SyncTerm, TTerm, Terminate) open the presented COM port
directly and El-Modem never opens it. The driver brokers between that port and
a named pipe, creating the pipe is El-Modem's job:

```text
el-modem --serial-backend pipe --modems 1
```

Terminals then connect to `COM3` (or whichever port the driver presents). No
`--serial-device` is needed; the default pipe is `\\.\pipe\ElModemSerial`.

This is the only backend on Windows where the modem needs no virtual-COM
package, no TCP redirector and no peer port. The driver must be installed
separately — see
[installing the El-Modem virtual COM driver](WINDOWS_DRIVER.md) for build,
certificate and install steps. The modem itself runs unprivileged and never
installs or configures the driver.

Because the pipe carries bytes only, `&K1`/`&K3` have no effect here and
`update_lines` is a no-op — RTS/CTS report as unknown rather than as a
fabricated `False` the DCE would misread. Terminals see a modem that is always
present (CTS, DSR and DCD asserted, RI idle). The dashboard ASCII terminal
cannot attach to a pipe; open a terminal program on the COM port instead.

**Current distribution status:** the driver is UMDF2 and currently test-signed,
so it installs only where the WDK test certificate is trusted. Full
installation instructions, including how to trust that certificate and how to
remove it again afterwards, are in
[WINDOWS_DRIVER.md](WINDOWS_DRIVER.md). A public signature is what removes the
need for that step.

## Windows: native COM backend

`--serial-backend com` owns one end of an already-installed virtual COM
pair and drives real modem-control lines (CTS/RTS hardware flow
control, DTR presence/drop, DCD carrier). It is the Windows counterpart
to the Linux driver backend. The modem runs unprivileged and never
installs or configures the driver.

This backend needs a virtual COM pair supplied by something else: a
physical modem, or a third-party pair such as com0com. It is the legacy
route; prefer the bundled driver above unless you specifically need real
RTS/CTS gating between two COM ports. See the
[com0com prerequisite](../packaging/com0com/README.md) for the pairing
tool.

With a virtual pair, create it once as administrator, wiring the terminal
side so DSR is independent of carrier. Using COM3 ↔ COM4 as the example:

```powershell
setupc.exe install PortName=COM3,dsr=on PortName=COM4
```

Then point the instance at the modem side and pick a baud:

```text
AT+CFG="serial.device","COM4"
AT+CFG="serial.baud","115200"
```

or `el-modem --serial-backend com --serial-device COM4 --serial-baud 115200`.
Each modem needs its own pair; the supervisor rejects duplicate devices.
`&K1`/`&K3` enforce real RTS/CTS on this backend.

RI has no independent pin in a statically wired virtual pair: two DCE
outputs (DTR, RTS) cannot drive three independent peer inputs (DCD, CTS,
RI). Incoming-ring detection still reaches terminals as the Hayes `RING`
result code. The dashboard ASCII terminal cannot take over a COM port
whose peer is unknown; attach a terminal program to the paired port.

## Every OS: TCP serial
    el-modem --serial-tcp 127.0.0.1:9100 ...
The modem listens on that port; one client at a time is the terminal. Disconnecting is a DTR drop. On Windows this is the
legacy single-DCE default (127.0.0.1:9100). Use a raw TCP client (PuTTY "Raw", `ncat 127.0.0.1 9100`);
this is a byte stream, not a Telnet protocol endpoint.

### Windows programs that insist on a COM port
Windows COM applications can use an external virtual COM pair and raw TCP
redirector. The published rc1 passed identity, S-register readback and reset
through **com0com 3.0.0.0 → com2tcp 1.3.0.0 → Tera Term 5.7.1**, with kernel
test-signing disabled on the documented Server 2022 VM. See the
[tested installation procedure and limits](WINDOWS_COM_QUALIFICATION.md).
That record is historical and the COM41/COM42 port numbers in it refer to
that run, not to a current arrangement.

1. Install the official signed com0com package and create a pair, such as COM3 ↔ COM4.
2. Start the modem's TCP DTE listener, for example `127.0.0.1:9100`.
3. Run `com2tcp.exe \\.\COM4 127.0.0.1 9100` and open COM3 in the terminal.

Use raw TCP, not Telnet. These tools are third-party dependencies and are not
bundled. The bridge qualifies byte/AT communication; full RS232 modem-control
signals and Windows 10/11 Secure Boot/HVCI deployment are not established by this
result.
