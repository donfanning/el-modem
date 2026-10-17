# Windows virtual COM driver source — MS-PL

This reviewed overlay reconstructs the modified Windows UMDF2 ComPort project.
No binaries, build outputs, certificates, signing keys or machine credentials
are included. This component (overlay, reconstruction helper and documentation)
is distributed under the Microsoft Public License in [LICENSE](LICENSE),
separately from the GPL modem runtime. Original Microsoft notices are retained.

Microsoft's baseline is Windows-driver-samples commit
`2dc3fd3a0cc84a2933f2194e7ec0871584979071`, `serial/VirtualSerial2`.
The actual modified project uses ComPort, despite its older README's FakeModem
plan. Source additions/changes are in `overlay.patch`. It reconstructs 16
source/project files after normalizing BOM/encoding, CRLF and final newlines;
source statements are preserved. Baseline and result hashes are explicit in
`source-manifest.json`. This is source reconstruction, not a driver-build or
signature equivalence claim.

```text
python packaging/windows-driver/prepare_source.py /fresh/directory
```

Only 14 pinned source files are fetched; the complete Microsoft repository is
not copied. The resulting standalone project uses the existing WDK solution.
The full MS-PL license is retained in `LICENSE` and copied to output.
This component must not be described as part of the GPL-owned DSP core or
covered by proprietary incorporation rights over Microsoft's code. It remains
a separate driver communicating with the modem over a byte pipe.

The source hardcodes the `ElModemSerial` pipe. Multiple modems need distinct
driver routing, not just additional Python workers. The driver also has local
AT fallback when disconnected: AT responses alone cannot prove the Python
modem was reached. A live probe on the existing COM3 driver exchanged exact
530-byte arbitrary binary vectors both ways through the real Python pipe
backend; no driver installation, trust change or modem call was performed.

Further driver review is needed before a binary distribution: synchronous pipe
writes and close paths, ring-buffer concurrency, shutdown after the two-second
thread wait, and constant modem-signal behavior are not qualified by that byte
probe. The diagnostic log in Users/Public contains event codes/byte counts;
it is not a substitute for bounded structured diagnostics.

## Build and signing

Run the reconstruction command from the repository root with Python 3 and Git
installed. Network access fetches only the pinned baseline files. The destination
must not already exist. Every baseline and resulting source file is hash-checked;
a partial or failed reconstruction does not report PASS.

On Windows, install Visual Studio 2022 and the matching Windows SDK/WDK, then
run from a developer shell in the reconstructed directory:

```powershell
msbuild VirtualSerial.sln /t:Rebuild /p:Configuration=Release /p:Platform=x64
```

CI verifies source reconstruction, not a WDK compile or driver installation.
A fresh compiled package and its binding to these sources remain unqualified.
Use your own local development test certificate; no project private key or
certificate is published here. Test-signed packages require explicit machine
trust and applicable Windows test policy. See [Windows driver instructions](../../docs/WINDOWS_DRIVER.md)
for installation helpers and the remaining qualification limits.
