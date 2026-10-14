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

## Every OS, and the only mode on Windows: TCP serial
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

1. Install the official signed com0com package and create a pair, such as COM41 ↔ COM42.
2. Start the modem's TCP DTE listener, for example `127.0.0.1:9100`.
3. Run `com2tcp.exe \\.\COM42 127.0.0.1 9100` and open COM41 in the terminal.

Use raw TCP, not Telnet. These tools are third-party dependencies and are not
bundled. The bridge qualifies byte/AT communication; full RS232 modem-control
signals and Windows 10/11 Secure Boot/HVCI deployment are not established by this
result.
