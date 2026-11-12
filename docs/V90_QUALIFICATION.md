# Native V.90 qualification and limits

The native V.90 caller and digital-answerer path is experimental and remains
disabled unless a V.90 PCM-role profile is explicitly configured. The caller's
adaptive downstream selector uses the peer's CRC-valid Jd rate mask and the
local cap or pin. It uses the standard one-level DIL exchange and can make a
bounded startup walkdown after a failed B1d check. Invalid or empty rate
selections fail closed.

The caller was exercised on a live external SIP modem route in six attempts.
Two completed at 56,000 bit/s downstream and 28,800 bit/s upstream with LAPM;
the other four had recorded RTP timestamp discontinuities of 1, 3, 6 and 1
packets. One of those four briefly negotiated 56,000 bit/s before the link
dropped. This is a reproduced result, not a general success-rate claim.

On the two clean completed calls, the receiver reported a 54-byte server
banner/login prompt (two inbound I-frames), and an outbound one-byte I-frame
was acknowledged. Both had zero LAPM FCS errors and zero RTP timestamp gaps.
The measured mean receive processing times were 6.07 ms and 6.34 ms. These
observations show a small control exchange over the completed link; they do
not establish bulk payload throughput or generic payload verification.

The same release includes a PCMU ANSam detector change and repeated-INFO0a
recovery while the V.90 answerer is receiving a Phase-2 probe. These are
V.8/startup behavior changes. The ANSam detector change is global and applies
to all calls, whether or not a V.90 profile is configured. The V.90
digital-answerer protocol path is integrated, but independent high-rate
answerer interoperability has not been qualified. V.92 is not included.
