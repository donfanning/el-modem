# Credits and third-party material

This project aims to be self-contained: every modem signal path, protocol and tool in the shipped program is our own
code, written from public specifications. Other projects were used to **test against** and to **check interoperability
details**; where that informed code it is said here. Nothing below is copied into the shipped program unless stated.

## Specifications (primary sources)
ITU-T Recommendations V.8 (11/2000), V.21, V.22, V.22bis, V.32 and V.32bis (the constellation figures), V.34 (02/98),
V.42 (03/2002 and, for Annex A, the 1993/1996 editions), V.42bis (01/90), V.90, V.92, V.25. Obtained from itu.int
(free to read); the PDFs are not stored in this repository (`lab/tools/fetch_itu.sh`).

## Code we wrote ourselves (this repository)
V.21/Bell 103 FSK, V.32/V.32bis (`v32_*.py`), V.34 caller and answerer (`v34_*.py`), V.8 (`v8.py`), V.42 detection and LAPM
(`v42_lapm.py`), V.42bis (`v42bis.py`), MNP 2-4 class link from V.42 Annex A (`mnp_link.py`), the Hayes AT command
interpreter and DCE (`modem_dce.py`), the SIP/RTP legs, the serial-over-TCP bridge, tests and tooling.

## Projects used as references, test peers or sources of hints
* **SpanDSP** (Steve Underwood, LGPL-2.1; submodule `spandsp/`). It is **no longer used at run time**: V.8, V.42, V.42bis and
  V.22/V.22bis are our own code (`v8.py`, `v42_lapm.py`, `v42bis.py`, `v22.py`). It is kept only as the *independent reference*
  they are tested against (`test_v8.py`, `test_v42_lapm.py`, `test_v42bis.py`, `test_v22.py`; `spandsp_v22.py` is the test wrapper). Reading its `v42.c` and
  `v8.c` informed three interoperability details (XID parameter encoding, originator/answerer address octets, V.8 timing margins); the
  code itself was not copied. Our patches to it (bit-rate-scaled timers, V.42bis negotiation hooks, XID length fix) live in
  `lab/asterisk/audio-peer/*.patch` and are LGPL like the library.
* **Smart Link slmodemd** (binary-only DSP, from the D-Modem build). Used only as a black-box test peer in the lab
  (`lab/slmodem-peer/`); never distributed or linked, no code or tables used.
* **BinModem** (GPL-3, Rust): its V.32bis implementation was used as a hint source while diagnosing real-line behaviour; nothing copied.
* **philpem's slmodem reconstruction**: not used.
* **minimodem** (submodule, a test tool for audio FSK) and **Asterisk** (GPL-2, the lab PBX): lab infrastructure only, not part of the program.
* **Gough Lui's collection of V.32/V.34 modem sounds** (goughlui.com): real recordings used to check our signals and decoders.
* **University of Cape Town thesis on V.32 9600 bit/s modems** (D. Lindebaum): the 9600 bit/s constellation labelling cross-check.
* **voip.ms**: the VoIP provider used for real-line validation.
* Runtime libraries: Python (PSF), NumPy (BSD-3).

Corrections and further attributions are welcome: if you recognise your work in something here, open an issue.

## Public source preview

Reference submodules, lab peers, provider configurations and all recordings
are excluded from this export. The community license is GPL-3.0-or-later;
provenance review remains open. See NOTICE and docs/REDISTRIBUTION.md.
