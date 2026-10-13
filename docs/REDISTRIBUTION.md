# Redistribution review

The community license is GPL-3.0-or-later, with a separate commercial licensing
route selected by the owner. Commercial terms and the contributor agreement
remain to be finalized. Attribution is not a license grant.
The existing credits describe the project's claimed provenance; this review
does not independently prove that every algorithm/table was written from scratch.
In particular, V.32 source comments refer to Fisher's implementation and tables:
review those references before declaring the original-code provenance settled.

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

Before publishing binaries, inventory the exact frozen artifact's bundled
packages and shared libraries, include their license texts, and supply any
corresponding source required by their licenses. The existing build specification
only explicitly adds the optimized interpreter license; it is not a complete
binary-notice inventory. Source-preview readiness and binary-release readiness
are separate gates.

The private repository contains historical notes and recordings. Publish a
new repository from the reviewed export. Do not change the private repository's
visibility or push its branches/tags into the public repository.
