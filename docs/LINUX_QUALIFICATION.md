# Linux rc1 qualification

Verified on 2026-10-14: **12/12 local calls pass the declared functional and
real-time gates on the documented host profile.** The retained baseline also
completed 12/12 functional transfers, with 6/12 timing passes.

I tested the exact Linux `v0.1.0-rc1` executable:
`836a8bf8a9959aefec5a227a31185f7ecfe43578188271cd7d36aa830a158afc`.
Its 78 embedded application modules and packaging specification match
[the immutable release source](https://github.com/donfanning/el-modem/tree/9fd8af82f497b472869007c5623bb24063846ce9).
These results concern these bytes; the separate Linux LTO executable has a
different hash and its own evidence.

## Test conditions

One packaged TUI/supervisor worker, Ubuntu 24.04 container, AMD Ryzen 5 3600
(six cores, twelve hardware threads), local SIP/G.711 through Asterisk to an
independent Smart Link software-modem peer, with TCP DTE. The proprietary peer
is a black-box test dependency and is not part of the public product.

Initial V.34 maximum 33,600 bit/s, V.42 LAPM, compression disabled, default
family-dependent RTP hold, and the packaged DSP/native runtime settings. PCM and DSP recording
remain enabled. Recovery can negotiate lower/asymmetric TX/RX rates; the
[numeric record](LINUX_QUALIFICATION.json) shows those rates, including
intentionally failed rollback trials. Initial B1 has zero errors on all twelve
calls; failed rollback trials are accounted for by the recovery validators.
Transfers are checked against independent expected payloads byte for byte in both directions. B1, profile exchanges, recovery events,
recording completeness and bounded runtime storage are checked separately.
No external/PSTN calls are part of this matrix.

## Host frequency policy matters

The baseline `schedutil`/`acpi-cpufreq` run completed all twelve calls with exact
payload and functional checks passing, but only six passed the timing gates.
The failures remain in the record; this is not an all-gates baseline pass.

Two subsequent 100 ppm drift pilot calls with the host policies set to
`performance` passed at 5.731 ms and 5.643 ms combined mean DSP time. The full
eight-gate/twelve-call confirmation passed on that explicit host profile. No
executable, numerical DSP code, modem setting or threshold was changed.
All twelve saved host policies were restored and checked after the bounded test.
The PBX was idle, temporary DUT containers were removed, and the original
executable and source bindings were rechecked.

The [kernel CPUFreq documentation](https://docs.kernel.org/admin-guide/pm/cpufreq.html)
describes the distinct frequency-selection policies. These measurements qualify
a tested host configuration; they do not isolate frequency as the sole cause
of every variation on a shared host. The application does not change the OS
power policy automatically.

## Confirmation results

| Probe | Role | Seed | Data seconds | Exact bytes each way | Mean DSP ms | Tail budget ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| sustained-call | call | 3942 | 605.909 | 1,081,344 | 5.622 | 9.504 |
| sustained-answer | answer | 3941 | 600.448 | 1,097,728 | 5.568 | 9.431 |
| drift | call | 3701 | 122.885 | 212,992 | 6.018 | 10.387 |
| drift | answer | 3701 | 127.642 | 229,376 | 5.986 | 10.065 |
| peer-retrain | call | 3711 | 27.038 | 32,768 | 5.405 | 13.755 |
| peer-retrain | answer | 3711 | 29.480 | 32,768 | 5.051 | 11.580 |
| online-retrain | call | 3712 | 55.562 | 65,536 | 5.724 | 13.374 |
| online-retrain | answer | 3712 | 59.347 | 65,536 | 5.484 | 11.565 |
| connected-rate | call | 3713 | 61.757 | 81,920 | 6.079 | 11.430 |
| connected-rate | answer | 3713 | 62.124 | 81,920 | 5.938 | 11.647 |
| auto-rollback | answer | 3717 | 318.276 | 540,672 | 5.888 | 10.272 |
| auto-rate | answer | 3718 | 308.792 | 540,672 | 5.648 | 9.748 |

Every confirmation row passes both checks, has exact payloads in both directions,
compression disabled, complete required captures and validated source/profile
evidence. The raw recordings, proprietary peer and private lab infrastructure
remain local. The numeric report is an explicit allowlist of measured facts.

## Scope and limits

The declared scope is ten-minute transfers in each role, 100 ppm peer-clock
drift for 120 seconds in each role, peer and online retrains in both roles,
connected down/up rate changes in both roles, and automatic rollback and upward
rate recovery in the answerer role. Timing gates remain combined TX/RX mean at
most 10 ms, sum of marginal TX/RX p99.5 budgets at most 20 ms, and the original
startup/queue/continuity checks. The tail budget is not a directly measured joint
percentile. Actual durations, payload sizes, per-epoch rates and outcomes
accompany the [numeric record](LINUX_QUALIFICATION.json).

One seed per probe/role is a bounded qualification, not a failure-rate estimate.
Larger drift, RTP loss, loaded-host/multiworker capacity, caller automatic-rate
recovery, independent V.32bis 14,400, native virtual COM/PTY applications, other
OS binaries and arbitrary PSTN peers are not established by this matrix. This machine is a tested configuration,
not a proved minimum-hardware requirement.

The release remains experimental. Independent rights review is required before
stable/commercial claims; see [redistribution and provenance limits](REDISTRIBUTION.md).
