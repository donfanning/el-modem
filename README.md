# El-Modem

A software DSP data modem with a Hayes command interface, SIP/G.711 transport,
and a Linux terminal dashboard. One application manages one modem by default,
or multiple independent modem processes with `--modems N`.

Implemented modem families include V.21/Bell 103, V.22/V.22bis, V.32/V.32bis and
V.34 up to 33,600 bit/s, with V.8 selection and V.42 LAPM error control.
The desktop factory profile requests LAPM and disables compression and echo
cancellation. IAX2 calling is implemented over UDP with G.711 µ-law and
automatic registration; configure a reachable IAX2 endpoint before selecting
S120=1. SIP remains the default.

This is an experimental Linux-first source preview, not a stable release.
Real PSTN calls have reached V.34/LAPM and PPP, including login and communication
with the far-end PPP gateway. Short independent-peer tests and local exact-data
tests pass. The exact published Linux rc1 executable has passed ten-minute
transfers, drift and recovery gates on the documented performance CPU profile;
the retained dynamic-policy baseline passed data checks but failed timing gates
on six calls. See [Linux qualification and limits](docs/LINUX_QUALIFICATION.md).
New native build candidates need their own qualification; these Linux results
do not establish Windows/macOS real-time performance.
Multiple modem processes work, but their real-time capacity depends on the host.
Windows/macOS builds are not a claim of qualified native virtual COM support.

I’m **Don Fanning (jfalcon)** — [gh:donfanning](https://github.com/donfanning).
I created El-Modem, drawing on years of telephony and VoIP research, including
Telephreak and my collaboration with Champ Clark III (dabeave).
See [Credits](CREDITS.md) for development assistance, reference implementations
and research acknowledgments.

## Build and run

The directory layout retains `lab/asterisk` for the application sources;
this source preview does not include private lab infrastructure or recordings.

```sh
# Linux matched-runtime candidate; requires Docker, no call is placed.
packaging/build.sh docker-bookworm-lto
packaging/dist/el-modem --self-test
packaging/dist/el-modem --self-test-payload
packaging/dist/el-modem --serial-backend pty
```

Linux defaults to the separately installed native serial driver, macOS to PTY,
and Windows to TCP. Windows applications that require COM need an external
virtual-port/TCP bridge; a signed virtual COM driver is not bundled. The Linux
default serial backend needs the separately installed
[serialsim prerequisite](packaging/serial/README.md). PTY and TCP alternatives
have different modem-signal capabilities. The dashboard includes per-modem
settings, status lights and an exclusive ASCII terminal. Configure SIP for your
PBX or provider; no service account or trunk credentials are supplied.

[Desktop and Hayes settings](docs/DESKTOP.md) ·
[AT command reference](docs/AT_COMMANDS.md) · [Serial interfaces](docs/SERIAL.md) ·
[Release OS/runtime requirements](docs/RUNTIME_REQUIREMENTS.md)

For source development, install NumPy, Numba, `textual==0.89.1`, Rust/Cargo and
Maturin. Build and install the IAX2 extension into the active Python environment
with `maturin develop --release --locked --manifest-path native/iax/Cargo.toml`,
then use:

```sh
PYTHONPATH=lab/asterisk:lab/asterisk/audio-peer:packaging python lab/asterisk/el_modem.py --serial-backend pty
PYTHONPATH=lab/asterisk:lab/asterisk/audio-peer:packaging python -m unittest test_modem_desktop test_serial_port test_iax_transport test_sip_auth test_sip_uac test_sip_uas
```

Those tests do not require private recordings or dial an external number.
Other included development tests may require excluded recordings or independent
lab peers; do not interpret an unavailable fixture as a passing interop test.

## Source preview and licensing

The export contains no original Git history, third-party modem recordings,
derived recording fixtures, proprietary Smart Link blob, provider configuration,
or private call notes. The export manifest identifies the source revision and
each exported file's hash. This does not replace a credentials review.

The community source is licensed under [GPL-3.0-or-later](LICENSE).
Separate commercial licensing is planned for proprietary incorporation; contact
me for terms. See [NOTICE](NOTICE), [contribution policy](CONTRIBUTING.md),
[credits](CREDITS.md) and [redistribution review](docs/REDISTRIBUTION.md).
Provenance review remains open. Binary redistribution requires the complete
notices and licenses of the actual bundled dependencies.
