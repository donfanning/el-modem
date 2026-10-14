# Windows installer qualification

The exact `v0.1.0-rc1` Windows setup was exercised on 2026-10-14 in a fresh,
isolated Windows Server 2022 Standard Evaluation KVM guest. The guest started
without a registered Visual C++ v14 x64 runtime or its three globally installed
DLLs. The VM ran with code integrity enabled and test signing disabled.

The interactive, non-silent setup executable was the published artifact with
SHA-256 `bab1cb831fbf230e45532a30c4a9608a9d5f3218ff5cd706e5092f06e802fa40`.
Its normal runtime path checked the downloaded Microsoft signature and installed
VC14 `v14.44.35211.00`. The installed executable matched published SHA-256
`f98372c76cb648e1faaf92c925659ab8e6ba065b7e5fd249230ccf3fc82cd7a7`.

The installed executable's `--self-test` then exited 0 with 22/22 gates and a
`PASS` report. The actual Start Menu shortcut launched the visible one-modem TUI,
which was closed with its normal `q` command. The actual uninstaller's confirm
and success dialogs were completed; it removed the payload, shortcut, and
uninstall registration while retaining the shared Microsoft runtime.

The first immediate post-install self-test wrapper emitted a valid `PASS` report
but did not retain its numerical process exit when it reported failure. The
same installed executable was then retested, with exit 0 recorded. This anomaly
is retained in the private qualification record rather than hidden.

The machine-readable evidence in `windows-installer-evidence/` is deliberately
limited to non-sensitive result summaries. Full VM images, screenshots, raw
installer logs, and fixture bootstrap material are not distributed.

This is one administrator-install profile. It does not qualify Windows 10/11,
standard-user/UAC installation, SmartScreen or installer signing behavior,
Windows ARM64, Secure Boot/HVCI, connected DSP operation, or the exact oldest
supported Windows/VC++ runtime. See [runtime requirements](RUNTIME_REQUIREMENTS.md).
