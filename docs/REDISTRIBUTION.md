# Redistribution review

The community license is GPL-3.0-or-later, with a separate commercial licensing
route I selected. Commercial terms and the contributor agreement
remain to be finalized. Attribution is not a license grant.
The credits and [source provenance record](SOURCE_PROVENANCE.md) preserve the
reference history, including Fisher's attribution request and the subsequent
spec-derived V.32 re-expression. I do not assert that Rossi's MIT notice licenses
the cached original Fisher files or that the re-expression was clean-room.

| Component | Role | Publication handling |
| --- | --- | --- |
| Original modem/TUI/SIP code | Application | GPL-3.0-or-later; provenance review remains open |
| CPython | Bundled interpreter | PSF license and included notices; inspect actual build |
| NumPy | DSP dependency | BSD notices and licenses of included numerical libraries |
| Numba / llvmlite / LLVM | Optional compiled DSP | Review pinned package license files and bundled LLVM notices |
| Textual / Rich and dependencies | TUI | MIT and each actual transitive dependency's notices |
| PyInstaller | Executable builder/bootloader | GPL exception permits applications under other licenses; retain required notices |
| serialsim patch | Separate Linux driver prerequisite | GPL-2.0-or-later; retain driver provenance and source |
| SpanDSP / minimodem | Optional independent lab references | Excluded from this preview; separate upstream repositories |
| Smart Link modem blob | Independent lab peer | Excluded; do not redistribute |
| Third-party audio and derived fixtures | Private regression corpus | Excluded pending explicit redistribution rights |
| ITU standards PDFs | Specifications | Excluded; reference standards rather than redistributing PDFs |

Primary references:

- [PyInstaller license and application exception](https://pyinstaller.org/en/stable/license.html)
- [Numba license](https://github.com/numba/numba/blob/main/LICENSE)
- [llvmlite licenses](https://github.com/numba/llvmlite)
- [Textual license](https://github.com/Textualize/textual/blob/main/LICENSE)
- [Python license](https://docs.python.org/3/license.html)
- [NumPy license](https://github.com/numpy/numpy/blob/main/LICENSE.txt)

Experimental `v0.1.0-rc1` binaries are already published. Each archive includes
the actual executable's package/native-library inventory, collected notices,
qualification reports and member hashes. This collection identifies build
providers; it does not by itself close static/transitive-code or corresponding
source review. Each archive currently records `REVIEW_REQUIRED`.

The release now also supplies a [notice supplement](https://github.com/donfanning/el-modem/releases/download/v0.1.0-rc1/el-modem-0.1.0-rc1-notice-supplement.zip)
and [source supplement](https://github.com/donfanning/el-modem/releases/download/v0.1.0-rc1/el-modem-0.1.0-rc1-source-supplement.tar.gz).
They correct the omitted CPython SHA3 and Numba vendored-code notices and provide
61 upstream/application source archives plus build recipes. All 78 embedded
application modules and the specification match the release source; Windows'
exact CRLF builder form accompanies the normal Git/LF form. The supplement's
README and source index state the GNU runtime mappings and review limits.
Existing executables and their initial manifest statuses are unchanged.

The current CI artifacts use CPython 3.11.17 on Linux and 3.11.9 on Windows/macOS.
The older qualified Linux LTO executable and its Debian source bundle are
different builds. Do not substitute their dependency versions or qualification
results for the released executables. Windows deliberately excludes Microsoft
runtime DLLs; the x64 Visual C++ Redistributable is a prerequisite downloaded
from Microsoft by the optional installer.

The application source and build scripts for the existing release are at
[the immutable release source revision](https://github.com/donfanning/el-modem/tree/9fd8af82f497b472869007c5623bb24063846ce9).
Corrections on the main branch do not change the source embedded in those bytes.
Supply exact required dependency sources and retained notices alongside the
binary downloads; local-only source archives are not public source delivery.
The [GPLv3 network-distribution conditions](https://www.gnu.org/licenses/gpl-3.0.html#section6)
are the primary reference for that delivery review.

Private historical notes, credentials, service configuration and recordings
remain excluded. Public updates come from reviewed explicit source paths;
private Git branches/tags and sibling reference projects are not published.
