# El-Modem AT command reference

`AT$`, `AT&$`, `ATS$`, `AT%$`, `ATD$`, `AT\$` and `AT+$` print help screens listing what is implemented.
Human-readable ATI and help pages fit 80 columns with unlimited rows. Long
values wrap rather than being truncated; non-ASCII text and terminal control
characters are escaped. ATI9 is a machine-readable packet and is not reformatted.

## Information screens (ATI)
| Command | Shows |
|---|---|
| `ATI0` | four-digit product code (`0040` = V0.4.0) |
| `ATI1` | CRC-16 of the program image (the executable, or the source file) |
| `ATI2` | self test (CRC known answer, RAM pattern, FFT): `OK`/`ERROR`, then the command's own `OK` |
| `ATI3` | banner, embedded UTC compilation timestamp, revision, project URL |
| `ATI4` | active settings: S-register grid (7 per row), `&C &D &K E Q V X %C \N`, `+MS`, network registers |
| `ATI5` | stored profiles (`AT&Wn`) and the four stored phone numbers |
| `ATI6` | link diagnostics: local DTE/line byte counts, known local DTE RX queue drops, LAPM blocks/errors, rates, retrain lifecycle, retained post-connect fallback events, modulation, protocol, compression and disconnect reason; end-to-end lost characters remain N/A |
| `ATI7` | configuration: options, Python/NumPy, platform, program size, NVRAM file, build date, transport and SIP settings |
| `ATI8` | stored numbers (`AT&Zn=number`, `ATDSn`), redial (`ATDL`), black list |
| `ATI9` | Serial PnP packet: project-local `ELM0001`, class `MODEM`, compatible ID `PNPC103`, description and checksum. The vendor ID is not registered. |
| `ATI10` | explicit status: modem-level dial security is not implemented; inbound access is not password-protected by the modem |

| `ATI11` | live or last-call DSP report: directional rates, DTE octets/drop count and duration, decoded PCMU input RMS in dBFS (not analog dBm0), separately labeled startup probe estimate, V.34 B1 held-out residual, retrain lifecycle, echo observations, detector quality, RTP and processing timing; live SNR/BER remain N/A |
| `ATI16` | error-control counters and observed byte ratios |
| `ATI17` | PCM / V.90 / V.92 support status (not implemented) |

| `ATI12` | SIP and IAX settings |
| `ATI15` | identity of the last incoming SIP/IAX call (`DATE=MMDD TIME=HHMM NMBR= NAME=`); restricted SIP identity is masked |
| `ATI18` | software audio-loop and local `&T3` DTE-loop status; explicitly distinguishes them from physical V.54 loops |

Build timestamps are stamped by the PyInstaller specification and read from the
embedded `el-modem-build.json`. They do not change when an executable is copied
or launched. Source runs say `uncompiled source`; older packages without a stamp
say `unknown` rather than guessing from file modification times.

DSP figures are reported only when available from the selected engine. The V.34
startup probe SNR is labelled a startup estimate, not live data SNR. V.32 decision
residual MSE and V.34 survivor path cost are not converted into SNR or BER.
The V.34 B1 held-out residual is the RMS prediction error on the second half of
the known B1 sequence after gain/energy calibration on the first half; ATI reports
it as a percent of reference RMS. It includes noise and model/channel residual,
so it is not line SNR or BER.
Echo-canceller input/output power includes the remote signal and is not an
echo-only loss measurement. Unmeasured character loss, line conditioning and
fallback counts say `N/A`, not zero or "good". Last-call statistics are copied at
disconnect so a later call cannot mutate the record.

`AT+CLIP=1` enables a `+CLIP` indication on the first `RING` of an incoming
call; `AT+CLIP=0` disables it, `AT+CLIP?` reads the setting, and `AT+CLIP=?`
lists its values. The answerer currently obtains identity from SIP `From` and
honors `Privacy: id`; withheld identities are not exposed through `+CLIP` or
ATI15. IAX caller number is displayed when supplied by the IAX peer. This is
VoIP signaling identity, not analog Bell 202/V.23 Caller ID FSK decoding.

`AT&T3` starts a local DTE byte loopback and returns `CONNECT`; use `+++` then
`AT&T0` to stop it. `ATI18` reports the byte count. This tests the DTE byte
path only. It is not V.54 remote digital loopback, analog line loopback, or a
claim of V.54 conformance. `AT&T1` remains the profile-specific paired DSP
software audio loop.

For Windows, the inbox **Standard 33600 bps Modem** profile is `MDMGEN336`, section
`Gen336` in `mdmgen.inf`. This is not a seven-character serial PnP ID. The draft
`packaging/windows/el-modem-modem.inf` maps the El-Modem identity to that profile;
it is not signed, installed or qualified. `PNPC103` is only a generic fallback.
ATI9 alone does not implement Serenum's DTR/RTS discovery exchange, and the
current byte-pipe driver does not forward those signal transitions to Python.
Automatic Windows modem discovery is therefore not yet qualified.

## Network settings (string S-registers)
`ATSn=text` sets, `ATSn?` reads (passwords read back as `********`), `AT&W` stores them (the NVRAM file is created mode 0600),
`ATZ`/`AT&F` reload/reset. Everything after the `=` is the value (so it must be the last thing on the line); quotes are optional.

| Reg | Setting | | Reg | Setting |
|---|---|---|---|---|
| S100 | SIP server (dial target) | | S110 | IAX server |
| S101 | SIP port (5060) | | S111 | IAX port (4569) |
| S102 | SIP user | | S112 | IAX user |
| S103 | SIP password | | S113 | IAX password |
| S104 | SIP registrar `host[:port]`, empty = none | | S114 | IAX caller number |
| S105 | caller number | | S115 | IAX registration (automatic when configured) |
| S106 | contact host for incoming calls | | S120 | trunk for ATD: 0 = SIP, 1 = IAX |

Changing S102/S103/S104/S106 restarts the SIP registration at once; S100-S103/S105 are read at each dial.
IAX2 calling uses the configured IAX host, port, username and password, with G.711 µ-law media; the modem registers automatically when an IAX endpoint is configured. `S115` is retained for compatibility but does not currently disable that automatic registration. Select it for dialing with `ATS120=1`; SIP remains the default.

## Other
`ATX0-4` (X2/X4 report NO DIALTONE, X3/X4 BUSY), `AT&G0|1|2` (V.22 guard tone: off/550/1800 Hz), `AT&V`, `AT&Wn`, `ATZn`, `AT&Zn=number` (slots 0-3), `ATDSn`, `ATDL`,
`AT+MS=mod,auto,min,max` (auto=1 runs V.8), `AT+GMI/GMM/GMR/GCAP`, `S2` (escape character) and `S12` (escape guard time, 1/50 s).

When `\N3` requests V.42 and the selected V.8 menu includes V.22, V.32 or
V.34, El-Modem advertises LAPM in CM/JM. If both peers agree, it skips V.42
ODP/ADP detection and proceeds directly to LAPM/XID after modulation training.
If V.8 is absent or LAPM is not common, the normal V.42 detection path remains
available. Raw, MNP-only and V.21-only calls do not advertise V.8 LAPM.

Bell and ITU modulation families are selected explicitly with `AT+MS` (for
example, `B103` versus `V21`, or `B212` versus `V22B`). There is no global
country/standards selector; a country code is not a substitute for choosing
the modem waveform family. V.8 `auto=1` applies to the V-series modes that
support this V.8 chain; `B103` and `B212` remain explicit selections. The
separate caller-side `auto` listener detects an incoming V.22, V.21 or Bell
103 signal; it is not a two-way Bell-versus-ITU negotiation and does not
select Bell 212A.

`AT+MS` is the supported modulation and connection-rate control. `S37` is
currently retained in profiles but has no effect. Its rate table and interaction
with `N` vary across modem command sets; Courier `&N`/`&U` are vendor-specific
commands and are not implemented. `ATI10` reports that modem-level dial
security is unavailable. It does not protect inbound calls; use SIP/IAX
authentication and deployment network controls for access policy. Courier
account passwords, DTMF security and dialback are not implemented.

`AT\\K0`-`AT\\K5` select the BREAK handling policy (default `AT\\K5`). In
online command mode, `AT\\B` or `AT\\Bn` requests a V.42/LAPM BREAK; the
reliable-link duration is fixed at 300 ms, so the optional digit does not alter
it. `\\B` uses the current `\\K` setting: K0/K1 are destructive/expedited,
K2/K3 non-destructive/expedited, and K4/K5 non-destructive/in-sequence. It
shares a bounded serialized queue with native serial BREAK events and returns
`ERROR` if the call is not online or the queue is full. A later LAPM rejection
after transmission/retry is reported in modem diagnostics.
Raw-mode BREAK duration and transports without native BREAK signaling are not
implemented.

The preliminary switched V.23 profile is selected with
`AT+MS=V23,0,300,1200`; setting the maximum to 600 selects its 600-baud forward
mode. Switched calls use the V.25 caller/answer setup before V.23 data. The
calling station sends on the 600/1200-baud forward channel while the answering
station uses the 75-baud return channel. This profile does not support V.8
automode, V.42 or MNP. Local production-path exact-payload tests cover both
forward rates; independent switched-call qualification remains blocked because
the available SmartLink peer enters V.23 but sends no modem signal.

`AT+IFC?`, `AT+IFC=?` and `AT+IFC=dce-output,dte-input` configure DTE flow
control. Software XON/XOFF modes are available on byte-stream endpoints; RTS/
CTS modes are listed only when the selected serial backend exposes modem-control
lines. Byte-stream endpoints default to `+IFC=0,0`. While V.42/LAPM is active,
receive-side DTE backpressure is propagated with RNR/RR supervisory frames.
`AT+ICF?`, `AT+ICF=?` and `AT+ICF=<format>[,<parity>]` control command/result
framing on POSIX PTYs and native serial devices. Supported format codes reflect
what the active tty accepts; auto-detect (`0`) is not supported. The default is
`3,3` (8N1). TCP and pipe endpoints support only that byte-stream default and
reject other formats.
V.250 format codes are `1`=8N2, `2`=8 parity 1 stop, `3`=8N1, `4`=7N2,
`5`=7 parity 1 stop and `6`=7N1; parity codes are `0`=odd, `1`=even,
`2`=mark and `3`=space. Parity is relevant only to formats 2 and 5.
The V.250 definition applies to command and result framing; online data framing
is for further study. A PTY only carries bytes and does not create physical
parity/stop bits; a native serial device applies its UART framing where the
driver supports it. `AT+IPR?`, `AT+IPR=?` and `AT+IPR=<rate>` report, list and
change rates supported by the active POSIX tty. The setting changes after `OK`
is queued at the old rate. TCP and pipe endpoints return ERROR because their
byte streams have no baud setting. On Linux native serial, `+IPR` also exposes
the Bell 212A 1170–1212 bit/s adjustment range through `termios2`/`BOTHER` when
available; each requested rate is accepted by the device only after exact
readback. These limits do not imply full V.250 conformance.

`AT&G0` disables the V.22/V.22bis guard tone, `AT&G1` selects 550 Hz, and
`AT&G2` selects 1800 Hz. The setting is stored by `AT&W`. The transmitter
applies it only to the high-channel answer modem. For 1800 Hz it uses the
V.22 §2.2 level (6 dB below data power) and compensates data level to keep
total transmit power constant. V.22 leaves the 550-Hz level for further study;
El-Modem uses the same -6 dB engineering policy for that option. Bell 212A and
Bell 103/V.21 do not use this V.22 setting.

`AT+BDF?`, `AT+BDF=?` and `AT+BDF=9|10|11|12` are El-Modem extensions for the
preliminary Bell 212A profile (not V.250 commands). Nine bits means start + 7
data + stop; ten means start + 8 data + stop; eleven/twelve means start + 8
data + parity + one/two stop bits. `AT+BPAR?`, `AT+BPAR=?` and `AT+BPAR=E|O|M|S`
select even, odd, mark or space parity for 11/12-bit formats. Nine-bit mode
masks the DTE byte stream to seven data bits; parity is stripped from received
bytes. These settings are stored with the modem profile. The current evidence
is internal; independent peer framing, physical serial timing, and BREAK
qualification remain incomplete.

`AT+XID?`, `AT+XID=?` and `AT+XID=0|1` are an El-Modem extension for V.42
XID response-loss policy. `0` requires a completed XID exchange: if XID was
attempted and retries are exhausted, the call ends with `NO CARRIER` instead
of connecting in raw mode with uncertain LAPM parameters. `1` allows the
legacy-peer compatibility path after unanswered XID requests and is the
default. A line on which V.42 is not detected still follows the normal
auto-reliable raw fallback. The setting is stored with `AT&W` profiles and
shown by `AT&V`. The legacy `V42_XID_FALLBACK` environment variable sets the
factory default for new/reset DCE profiles; an AT command or stored profile can
override it per modem.
