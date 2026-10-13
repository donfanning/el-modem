# El-Modem

A software DSP data modem with a Hayes command interface, SIP/G.711 transport,
and a Linux terminal dashboard. One application manages one modem by default,
or multiple independent modem processes with `--modems N`.

Implemented modem families include V.21/Bell 103, V.22/V.22bis, V.32/V.32bis and
V.34 up to 33,600 bit/s, with V.8 selection and V.42 LAPM error control.
The desktop factory profile requests LAPM and disables compression and echo
cancellation. Native IAX calling is not implemented; IAX settings are stored.

This is an experimental Linux-first source preview, not a stable release.
Real PSTN calls have reached V.34/LAPM and PPP, including login and communication
with the far-end PPP gateway. Short independent-peer tests and local exact-data
tests pass. Sustained timing qualification remains open: previous ten-minute
desktop tests transferred exact data but exceeded the receive-queue gate.
Multiple modem processes work, but their real-time capacity depends on the host.
Windows/macOS builds are not a claim of qualified native virtual COM support.

Created by **Don Fanning (JFalcon)**, drawing on years of telephony and VoIP
research, including Telephreak and collaboration with Champ Clark III (dabeave).
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

The normal default serial backend needs the separately installed Linux
[serialsim prerequisite](packaging/serial/README.md). PTY and TCP alternatives
have different modem-signal capabilities. The dashboard includes per-modem
settings, status lights and an exclusive ASCII terminal. Configure SIP for your
PBX or provider; no service account or trunk credentials are supplied.

[Desktop and Hayes settings](docs/DESKTOP.md) ·
[AT command reference](docs/AT_COMMANDS.md) · [Serial interfaces](docs/SERIAL.md)

For source development, install NumPy, Numba and `textual==0.89.1`, then use:

```sh
PYTHONPATH=lab/asterisk:lab/asterisk/audio-peer:packaging python lab/asterisk/el_modem.py --serial-backend pty
PYTHONPATH=lab/asterisk:lab/asterisk/audio-peer:packaging python -m unittest test_modem_desktop test_serial_port test_sip_auth test_sip_uac test_sip_uas
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
the maintainer for terms. See [NOTICE](NOTICE), [contribution policy](CONTRIBUTING.md),
[credits](CREDITS.md) and [redistribution review](docs/REDISTRIBUTION.md).
Provenance review remains open. Binary redistribution requires the complete
notices and licenses of the actual bundled dependencies.
