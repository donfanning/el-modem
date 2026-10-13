# Credits and acknowledgments


Created and maintain El-Modem by **Don Fanning (jfalcon)** — [gh:donfanning](https://github.com/donfanning)
and I direct its development, interoperability testing and intended behavior.

I have used **OpenCode**, **Anthropic Claude**,
**GitHub Copilot**, and **OpenAI ChatGPT/Codex** for implementation, research,
review and troubleshooting.

These credits acknowledge contributions and influences; they do not establish
copyright ownership or complete provenance clearance. Original community code
is offered under GPL-3.0-or-later; third-party material retains its respective
terms. See [NOTICE](NOTICE).

## Telephreak and earlier work

Special thanks to **[Telephreak](https://www.telephreak.org/)** and
**Champ Clark III (dabeave)**, creator of
[iWar](https://sourceforge.net/projects/iwar/), for years of shared telephony
and VoIP research and collaboration with me.

Our earlier 2600/HOPE and DEF CON presentations include:

- [Hacking International Networks and System(s) Using VoIP](https://infocondb.org/con/hope/the-last-hope/hacking-international-networks-and-systems-using-voip),
  The Last HOPE, July 19, 2008 — me (jfalcon) and Da Beave.
- [AAPL — Automated Analog Telephone Logging](https://infocondb.org/con/def-con/def-con-17/aapl-automated-analog-telephone-logging),
  DEF CON 17, August 2, 2009 — me (jfalcon) and Da Beave.

This collaboration is part of the project's background; it does not imply
that iWar code is incorporated or that Champ authored the current DSP.

## Standards and reference software

**ITU-T** modem recommendations and **IETF** protocols provide the project's
interoperability foundations. Implemented modem families include V.21,
V.22/V.22bis, V.32/V.32bis and V.34, with V.8 selection and V.42 error control.
The code also includes V.42bis compression and MNP compatibility. V.90/V.92
and V.150.1 were research references, not implemented features.

- **Dr. Tony Fisher and Randy Rossi** — Fisher's modem work and Rossi's
  [FisherSoftModem rewrite](https://github.com/randyrossi/fisher-modem).
  Our V.32 source comments reference constellation ordering, scramblers,
  rate words, trellis tables and timing-slip techniques. Rossi's rewrite
  carries an MIT license; original Fisher material and any incorporated
  material require their own provenance and notice review.
- **Steve Underwood and SpanDSP contributors** —
  [SpanDSP](https://github.com/freeswitch/spandsp) supplied independent
  comparisons and informed V.8/V.42 interoperability details. Early versions
  used SpanDSP at runtime; the native modem path now uses El-Modem engines.
  Reference wrappers and patches retain their applicable upstream licenses.
- **Smart Link and D-Modem contributors** — the proprietary slmodemd DSP,
  integrated through [D-Modem](https://github.com/strozfriedberg/D-Modem),
  served as an independent black-box test peer. The proprietary blob is not
  distributed with this source preview.
- **CasualArclamp's [BinModem](https://github.com/CasualArclamp/BinModem)** —
  V.32-family implementation and diagnostic references.
- **Fabrice Bellard's [linmodem](https://bellard.org/linmodem/)** — historical
  software-modem and receiver-architecture references.
- **Kamal Mostafa and [minimodem](https://github.com/kamalmostafa/minimodem)
  contributors** — independent FSK testing and reference tooling.
- **gnomsa's [sip-softmodem](https://github.com/gnomsa/sip-softmodem)**,
  **[Shadytel's shadysoftmodem](https://github.com/Shadytel/shadysoftmodem)**,
  and **[Osmocom](https://osmocom.org/projects/retronetworking/wiki/Software_Modems)** —
  software-modem and transport research references.

Reference use is not a claim that these projects supplied a verified fix or
that their code is incorporated into El-Modem.

## Recordings and DSP research

**Gough Lui's [V.34 modem recordings](https://goughlui.com/2017/05/30/project-the-large-collection-of-v-34-modem-sounds/)**
provided synchronized modem conversations for offline startup, training,
B1 and data-decoding investigations. Recordings and derived fixtures are
excluded from the public preview; attribution does not grant redistribution
permission.

Receiver research also drew on **Lindebaum's University of Cape Town work**
on trellis-coded modulation and **Christopher J. Buehler, Jonas Keating,
Hayden Metz, Carl J. Nuzman and Hemanth Sampath**, with advisor
**Dr. Steven A. Tretter**, for the University of Maryland report published
by Texas Instruments as
[SPRA159](https://www.ti.com/lit/an/spra159/spra159.pdf).

## Runtime and development infrastructure

Thanks to the contributors and maintainers of:

- **CPython/Python Software Foundation** and **NumPy** for the numerical runtime.
- **Numba, llvmlite and LLVM** for compiled trellis acceleration.
- **Textual and Rich** for the terminal dashboard and dialogs.
- **PyInstaller**, **Debian** and **GCC** for packaging and runtime builds.
- **Corey Minyard and [serialsim](https://github.com/cminyard/serialsim)**,
  and Linux serial-core contributors, for the separate Linux virtual-serial
  prerequisite. Our driver patch does not replace credit for the underlying driver.
- **Asterisk**, **pppd**, **FFmpeg**, **Docker**, **Git**, **GitHub Actions**
  and **Gitleaks** for lab infrastructure, protocol testing, codec verification,
  development and publication checks.

Binary releases require a version-specific dependency license and notice
inventory; this list is not a substitute for it.

## Interoperability environments

Special thanks to **[dialup.world](https://dialup.world/)** for its dial-up
modem service, which provided a real-world target for modem, LAPM and PPP
interoperability testing.

Services operated by **Astound** and **NIST ACTS**, together
with **voip.ms** and other SIP routes, provided real-world interoperability
conditions. This acknowledgment does not imply authorship, endorsement or
certification. Private PBX configuration, credentials and call records remain
outside the public source preview.

Corrections and additional attributable contributions are welcome.
