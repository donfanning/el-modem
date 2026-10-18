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

## Linux and macOS: real serial device

Choose the `device` backend for a physical or USB serial adapter. The modem
process opens and exclusively owns the tty while it runs; the dashboard
terminal is multiplexed through the worker, so it does not compete for the
device. Set the backend, absolute device path, and DTE baud in the per-modem
instance dialog, or at launch:

```sh
el-modem --modems 1 --serial-backend device \
  --serial-device /dev/ttyUSB0 --serial-baud 115200
```

Typical paths are `/dev/ttyUSB0` or `/dev/ttyACM0` on Linux and
`/dev/cu.usbserial-*` or `/dev/cu.usbmodem*` on macOS. Select `pty` instead for
the built-in virtual port. The backend is switchable per modem; stop the modem
before changing the selection or device path. Device paths must be absolute,
baud rates must be supported by the host OS, and one physical port can belong
to only one modem instance. The backend configures raw 8-bit serial and
supports DTR/DCD/RTS/CTS signals when the adapter and driver expose them. Some
USB adapters omit modem-control lines; those signals then remain unknown.

The native implementation is POSIX-based and has unit coverage using a PTY.
That verifies its byte path and ownership behavior; it is not a qualification
of every physical adapter or platform-specific control-line behavior. Ensure
the user running El-Modem has permission to open the selected device.

## Windows: optional virtual-COM driver and pipe backend

`--serial-backend pipe` requires a separately supplied compatible driver.
No driver binary is distributed in this checkout or the current Windows ZIP.
The [MS-PL source overlay](../packaging/windows-driver/README.md) reconstructs
the modified project from pinned Microsoft sources. A fresh driver build and
installation remain unqualified. TCP and the com0com route below remain available.
Ownership is inverted
relative to the `com` backend: the driver *is* the COM port, so the program that
wants the port opens the presented COM port directly and El-Modem never opens it.
The driver brokers between that port and a named pipe, creating the pipe is
El-Modem's job:

```text
el-modem --serial-backend pipe --modems 1
```

Terminals then connect to `COM3` (or whichever port the driver presents). No
`--serial-device` is needed; the default pipe is `\\.\pipe\ElModemSerial`. A
second port names its own pipe in the registry and is then passed to El-Modem as
`--serial-device <name>`; see
[running more than one port](WINDOWS_DRIVER.md#running-more-than-one-port).

Once the compatible driver is installed, this backend needs no TCP redirector
or peer COM port. See
[installing the El-Modem virtual COM driver](WINDOWS_DRIVER.md) for build,
certificate and install steps. The modem itself runs unprivileged and never
installs or configures the driver.

Because the pipe carries bytes only, `&K1`/`&K3` have no effect here and
`update_lines` is a no-op — RTS/CTS report as unknown rather than as a
fabricated `False` the DCE would misread. Terminals see a modem that is always
present (CTS, DSR and DCD asserted, RI idle). The dashboard ASCII terminal
cannot attach to a pipe; open a terminal program on the COM port instead.

**Current distribution status:** the separate driver is UMDF2 and test-signed,
so it installs only where the WDK test certificate is trusted. Full
installation instructions, including how to trust that certificate and how to
remove it again afterwards, are in
[WINDOWS_DRIVER.md](WINDOWS_DRIVER.md). A public signature is what removes the
need for that step.

Each modem needs an exclusive pipe endpoint. The supervisor rejects duplicate
names after normalization (Windows pipe names are case insensitive). Changing
the Python endpoint with `--serial-device OtherModem` also requires a driver
configured for that same pipe. Multiple driver/modem pairs are not qualified.

## Windows: native COM backend

`--serial-backend com` owns one end of an already-installed virtual COM
pair and drives real modem-control lines (CTS/RTS hardware flow
control, DTR presence/drop, DCD carrier). It is the Windows counterpart
to the Linux driver backend. The modem runs unprivileged and never
installs or configures the driver.

This backend needs a virtual COM pair supplied by something else: a
physical modem, or a third-party pair such as com0com. It is the legacy
route for real RTS/CTS gating between two COM ports. See the
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
