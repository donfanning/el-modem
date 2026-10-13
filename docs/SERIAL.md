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
Windows has no PTYs and we ship no kernel driver. Pair the TCP port with a free virtual-COM tool:
1. Install **com0com** (creates a COM port pair, e.g. COM5 <-> COM6).
2. Bridge COM6 to `127.0.0.1:9100` with a TCP redirector (e.g. **com2tcp** or **hub4com** `--route=All:All
   --create-filter=tcp ...`), then point the terminal program at COM5.
(These tools are third-party and not bundled.)
