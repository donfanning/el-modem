# EL-MODEM desktop / headless runtime

The new launcher starts one modem and opens a terminal dashboard by default:

```sh
el-modem
el-modem --modems 4
el-modem --headless --modems 4
```

Linux full serial support needs the [serial driver prerequisite](../packaging/serial/README.md).
For explicit alternatives, use `--serial-backend pty` or `--serial-backend tcp`
when creating instances. These do not provide full hardware modem signals.
Driver failure is reported rather than silently changing the serial interface.

Configuration defaults to `$XDG_CONFIG_HOME/el-modem/config.json` (or
`~/.config/el-modem/config.json`). `--config /path/config.json` selects a separate
installation. One supervisor owns a configuration at a time. Modems have
independent process IDs, profiles, SIP/RTP ports, serial devices, logs, and
capture directories. No CPU-based modem limit is imposed. Monitor DSP timing
and sender lateness when adding simultaneous calls; available serial devices,
ports, memory, and CPU are finite.

New modem instances use automatic V.34 up to 33600, LAPM requested, and
compression disabled. Existing saved Hayes profiles retain their settings.
The existing explicit single-modem flags and offline self-tests remain usable;
for example, `el-modem --serial-tcp 127.0.0.1:9100 --line-mode ms` uses the legacy
single-DCE CLI. It retains its original factory defaults.

## Dashboard and settings

Each modem has live status and lights, settings, instance configuration, and
terminal access. Stop an instance before changing its listening endpoints or
serial adapter. Per-modem modem/SIP/IAX/advanced settings use the same validators
as the Hayes configuration interface. IAX settings are stored only: this release
has no native IAX calling client.

SIP listens on loopback initially. Set the SIP/media bind addresses to reachable
local IPv4 addresses for a remote PBX/provider, and configure SIP server,
registrar, credentials and advertised contact as appropriate for the network.
There is no bundled PBX or trunk account.

Settings identify immediate/idle/restart application. Mutations during an active
call are rejected. Save profile persists settings; DSP/runtime changes require
Save profile followed by restart. Active and pending values are shown separately.
Passwords are masked on reads and private profile files are mode 0600.

The existing Hayes commands still work, including ATS100–106 for SIP and
ATS110–115 for IAX. Named settings cover advanced configuration:

```text
AT+CFG=?
AT+CFG?
AT+CFG="sip.server","pbx.example"
AT+CFG="dsp.eq_span","24"
AT&W
```

The first query lists supported names; the second reports values/application
rules. `ATZ` reloads the saved profile and `AT&F` restores the factory profile.
Each modem has its own settings; worker startup isolates modem-specific inherited
environment variables and uses the declared native recipe plus saved overrides.

## Built-in ASCII terminal

Terminal access is exclusive. Release an external application's port first.
The settings/control channel remains usable while that application owns DTE data.

H on the dashboard or F9 in the terminal hangs up without stopping the worker.
Enter sends CR; Backspace sends BS; Tab and Ctrl-A through Ctrl-Z send their
ASCII control bytes. F10 releases the port using normal AT&D behavior. Incoming
CR/LF/backspace/tab are displayed; other nonprinting bytes are shown as `<XX>`.
The terminal does not execute remote ANSI escape sequences. Scrollback is bounded.
ANSI/CP437 and file-transfer protocols are later work.

Closing the application hangs up and stops its own workers. Active-call stop,
restart, remove and TUI exit request confirmation. A crashed modem is marked
failed; other workers continue. Removing an instance retains its profile and
recordings on disk and does not reuse that retained instance ID.

## Build and qualification

`packaging/build.sh docker-bookworm` builds against the qualified numerical
runtime, adding pinned Textual 0.89.1. The same executable starts its worker
processes. The build embeds source digests and includes Textual resources.
The UI stays outside DSP workers; status is coalesced at 10 Hz over bounded local
IPC. Full PCM/DSP recording is enabled by default, bounded to 600 seconds per
call; each worker run has its own capture directory. Recording duration and
enabling are configurable named settings applied after restart.

See the README for the source preview qualification limits. An
implemented feature is not a claim of qualification on every modem, kernel, or
host. Windows/macOS native serial qualification and native IAX remain future work.

`packaging/build.sh docker-bookworm-lto` builds with a PGO+LTO shared Python
library from the matching patched Debian source, including Debian's built-in
module set. This is the recipe used for the four passing short PPP cases and
ten passing family-reset cases documented in
the private development qualification record (not distributed).
The manifest records the library digest, optimization settings and source
checksums. The first build takes longer; Docker caches subsequent builds.
Rebuilding produces a new artifact that still needs its own qualification.

`docker-bookworm` remains the stock-library comparison. The experimental
`docker-bookworm-optimized` PGO-only recipe lacks Debian's built-in module set,
failed the frozen self-test, and did not improve the microbenchmark. Use the
LTO recipe for the desktop candidate. Sustained, drift and recovery checks with
the TUI attached remain open; these short gates do not qualify a loaded host.
