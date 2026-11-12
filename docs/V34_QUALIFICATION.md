# Native V.34 caller qualification and limits

The 2026-11-11 live SIP call exercised the native V.34 caller at
`CONNECT 33600/LAPM`. B1 qualified with 0 errors across 1,176 bits. The caller
received a 54-byte server banner and sent one byte in an acknowledged LAPM
I-frame; LAPM reported zero FCS errors.

This was one clean control exchange, not a bulk-payload test. The call's
`payload_verified` result was false, so it does not establish sustained
throughput or exact bulk payload delivery. The answerer role is not qualified
by this call.

Combined data DSP processing averaged 10.24 ms per media interval. The
conservative p99 upper budget was 24.77 ms, and the call recorded one RTP
timestamp gap (160 concealed samples). These measurements are reported as
observed; they do not establish a broad real-time margin under load.
