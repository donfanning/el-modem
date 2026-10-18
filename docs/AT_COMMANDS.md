# El-Modem AT command reference

`AT$`, `AT&$`, `ATS$`, `AT%$`, `ATD$`, `AT\$` and `AT+$` print help screens listing what is implemented.

## Information screens (ATI)
| Command | Shows |
|---|---|
| `ATI0` | four-digit product code (`0020` = V0.2.0) |
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
| `ATI11` `ATI16` `ATI17` | technical support report sections 1-3. Figures we do not measure (SNR, echo, trellis errors, frame slips, PCM) print `N/A` |
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
`ATX0-4` (X2/X4 report NO DIALTONE, X3/X4 BUSY), `AT&V`, `AT&Wn`, `ATZn`, `AT&Zn=number` (slots 0-3), `ATDSn`, `ATDL`,
`AT+MS=mod,auto,min,max` (auto=1 runs V.8), `AT+GMI/GMM/GMR/GCAP`, `S2` (escape character) and `S12` (escape guard time, 1/50 s).
