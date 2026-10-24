# El-Modem AT command reference

`AT$`, `AT&$`, `ATS$`, `AT%$`, `ATD$`, `AT\$` and `AT+$` print help screens listing what is implemented.

## Information screens (ATI)
| Command | Shows |
|---|---|
| `ATI0` | four-digit product code (`0030` = V0.3.0) |
| `ATI1` | CRC-16 of the program image (the executable, or the source file) |
| `ATI2` | self test (CRC known answer, RAM pattern, FFT): `OK`/`ERROR`, then the command's own `OK` |
| `ATI3` | banner, build date, revision, project URL |
| `ATI4` | active settings: S-register grid (7 per row), `&C &D &K E Q V X %C \N`, `+MS`, network registers |
| `ATI5` | stored profiles (`AT&Wn`) and the four stored phone numbers |
| `ATI6` | link diagnostics of the last call: characters, octets, blocks (sent/received/resent), FCS errors, rates, modulation, protocol, compression, disconnect reason |
| `ATI7` | configuration: options, Python/NumPy, platform, program size, NVRAM file, build date, transport and SIP settings |
| `ATI8` | stored numbers (`AT&Zn=number`, `ATDSn`), redial (`ATDL`), black list |
| `ATI9` | PnP-style id (`ELM0001\El-Modem Software Modem`, not registered with any authority) |
| `ATI10` | dial security status (static: no dial security) |
| `ATI11` `ATI16` `ATI17` | technical support report sections 1-3. Unmeasured figures (SNR, echo, trellis errors, PCM) print `N/A`; ATI16 reports V.34 RX equalizer timing slips when available |
| `ATI12` | SIP and IAX settings |
| `ATI15` | caller ID of the last incoming call (`DATE=MMDD TIME=HHMM NMBR= NAME=`) |

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

The preliminary V.23 raw profile is selected with `AT+MS=V23,0,300,1200`;
setting the maximum to 600 selects its 600-baud forward mode. The calling
station sends on the 600/1200-baud forward channel while the answering station
uses the 75-baud return channel. This profile does not support V.8 automode,
V.42 or MNP and is not yet qualified against an independent V.23 modem.

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
