# Windows setup withdrawal — 2026-10-17

The setup EXE published with `v0.1.0-rc1-windows` was removed after its file
inventory was audited. It is identified by SHA-256
`419b1f8ad5c203080271941e3f19a781927b44f2625902d9f44e21a06c714bd7`.
Do not use or redistribute that setup. Its separate Windows ZIP was verified
independently and remains available, SHA-256
`609d8814414072a070f96e9e21312032f51c53b71a2a9f19d9a8f7912dbacf77`.

An empty `DriverDir` compiler definition was treated as a driver-enabled build.
That turned a recursive driver wildcard into a drive-root wildcard and included
CI runner files. Extracting the published setup found 1,063 unexpected files,
including action dependencies, workspace copies, temporary files and guest-agent
logs. The program executable and the other 41 shared package files matched the
ZIP. Five packaging/license-helper files from the ZIP were absent from the
setup. These observations supersede the earlier statement that this setup
contained only the application and no driver payload; no functioning virtual-COM
driver was found, but unrelated files occupied that directory.

The initial credential scan produced 13 generic-key matches: nine source-file
SHA-256 values and four copies of the standard Azure storage-emulator
development connection string in Actions dependencies. These are not evidence
of live credentials. This classification and the removal do not assert that
every potentially sensitive item in runner logs has been excluded. Extracted
runner files and detailed scan reports are not redistributed. An additional
scan with archive inspection found the same categories in nested copies and
333 firewall-rule GUIDs labeled `Key` in guest-agent logs, all false positives
for credentials. No live credential was identified by either scan.

The corrected build treats absent and empty driver paths alike, enumerates
only the four required driver files and two installation scripts when supplied,
and checks the installed tree against the reviewed ZIP and explicit driver
inputs before retaining a candidate. Windows CI compiles all three layouts:
driver absent, empty path, and explicit test payload. It does not install or
trust the test driver. New installer publication requires a fresh contents
audit and qualification; none is implied by the historical rc1 tests.
