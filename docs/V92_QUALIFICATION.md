# Native V.92 qualification and limits

The native V.92 caller is experimental and remains disabled unless a V.90
PCM-role caller profile also sets `v92_pcm_upstream`. It builds on the V.90
caller path: when the far modem's INFO0d offers V.92, the caller runs the V.92
Phase 3 and Phase 4 exchanges with a PCM upstream; otherwise it continues as a
V.90 caller. The requested downstream rate is set with `v92_downstream_drn`
(1 to 22). Rate renegotiation responses, TRN2u dither (`v92_trn2u_noise`) and a
pad-aware upstream are separate profile options, all off by default.
`lab/asterisk/profiles/v92-pcm-upstream-analogue-caller.json` holds the settings
of the five-call 56,000 bit/s batch below (dither 0.015, pad-aware upstream and
renegotiation responses on); the other batches changed the downstream request
or the dither.

The caller was exercised on live calls to one commercial V.92 dial-up server
over a SIP route with a wide-area relay. Results by batch:

| Downstream request | Calls | Reached V.92 data with LAPM | Server login prompt received |
| --- | --- | --- | --- |
| 29,333 bit/s | 3 | 3 | 3 |
| 40,000 bit/s | 3 | 3 | 3 |
| 56,000 bit/s | 5 | 5 | 4 |
| 56,000 bit/s, lighter dither | 3 | 3 | 1 |

With TRN2u dither the server selected upstream rates of 26,667 to 30,667 bit/s.
Without dither it offered 38,667 to 44,000 bit/s, and those calls did not reach
a working data phase; higher upstream rates are not qualified. At 56,000 bit/s
downstream, B1d checks were error-free on every call that reached Phase 4, and
several calls rode out a server-initiated rate renegotiation. These are
reproduced results on one route and one server, not a general success-rate
claim.

At 56,000 bit/s downstream this server supplies fewer data bits than the line
rate and fills the difference with runs of marks inside its HDLC frames, which
aborts the frames they land in. That explains the calls that reached data
without a login prompt. The default-off `v92_downstream_mark_fill` option
removes those runs; it was verified offline against recordings of those calls,
where every frame then decoded with a good FCS, but it has not yet been used on
a live call.

Training tolerates RTP gaps of up to 100 ms in Phase 2 and longer ones in
Phases 3 and 4, by concealing them as silence or erasures. Paired
El-Modem-to-El-Modem calls over SIP also connect with V.92. Independent V.92 answerer
interoperability, quick connect, modem-on-hold, V.44 and bulk payload
throughput have not been qualified. The control exchange observed on live calls
was a short server login prompt; it does not establish sustained data transfer.
