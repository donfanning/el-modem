# rc1 runtime requirements

I inspected the actual `v0.1.0-rc1` executables and bundled native libraries,
not just the build recipes or launcher headers. The
[machine-readable record](RUNTIME_REQUIREMENTS.json) binds these results to
the archive/executable SHA256 values. The binaries are unchanged.

## Versions and test coverage

| Download | Required version or declared policy | Runtime prerequisites | What I verified |
| --- | --- | --- | --- |
| Linux x86-64 ZIP | **GLIBC 2.38 or newer**, including the system libc/libm and GNU ELF loader | x86-64 Linux using GLIBC; writable temporary storage that permits loading executable shared libraries | GLIBC 2.36 rejects the executable; GLIBC 2.38 passes 33 native offline gates. Full local DSP qualification was on Ubuntu 24.04 with the documented CPU profile. |
| Windows x86-64 ZIP | Windows 10/11 native x64 is the target policy; older Windows is not supported by the installer/runtime policy | Current compatible **Microsoft Visual C++ v14 x64 Redistributable**, at least the installer's 14.40 guard; Windows' UCRT; an ANSI/VT-capable interactive console for the TUI | Native CI offline/worker smoke on Windows Server 2022. The exact oldest working VC runtime and older Windows builds are not qualified. |
| Windows x86-64 installer | **Windows NT 10.0 or newer**, enforced by `MinVersion=10.0`; `x64compatible` architecture rule | Administrator installation; downloads the Microsoft runtime when absent/old, or when explicitly forced for testing; internet needed for that download | Installer tested on a runtime-present Windows Server 2022 CI host. A clean runtime-less install remains open. Windows ARM64 emulation/native COM remain unqualified. |
| macOS Intel ZIP | **macOS 10.15 deployment floor**, x86-64 | Interactive terminal for TUI; writable temporary storage; explicit OS approval of the downloaded executable as required | All 90 native objects inspected; LLVM agrees with their deployment metadata. Native CI smoke was on macOS 15, not Catalina. |
| macOS Apple Silicon ZIP | **macOS 11.1 deployment floor**, ARM64 | Same terminal/temp prerequisites; use the native ARM64 download | All 90 native objects inspected; LLVM agrees with their deployment metadata. Native CI smoke was on macOS 15, not Big Sur 11.1. |

The Mac deployment floors are **necessary metadata constraints, not proof that
every feature works on those oldest OS versions**. SDK versions, generic wheel
tags and the launcher's lower target do not establish the whole application's
minimum. In particular, the Intel launcher targets 10.13 and the ARM launcher
11.0, but the required Numba/llvmlite libraries raise those floors to 10.15/11.1.
The current downloads are not universal macOS binaries; Rosetta/emulation is
not covered by this qualification.

The Windows launcher's PE subsystem version is also not evidence of support for
older Windows. The published installer source requires NT 10.0 and checks the
runtime's registry version against 14.40. That check is a declared lower bound,
not a clean-machine qualification of exactly 14.40. Use a current compatible
[Microsoft x64 v14 runtime](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170).
The imported runtime families are `MSVCP140.dll`, `VCRUNTIME140.dll`,
`VCRUNTIME140_1.dll` and `VCOMP140.DLL`; they are not bundled in the ZIP.
The installer is unsigned; the Mac downloads have no Developer ID signing or
notarization qualification. Developer signing remains separate release work.

## Included software and operating conditions

**No separate Python, NumPy, Numba or Textual installation is needed to run the
downloads.** Linux bundles CPython 3.11.17; Windows and both Macs bundle 3.11.9.
Developer/source builds have different setup requirements and produce different
bytes. Optional CUDA/TBB extensions in the collection do not make a GPU, CUDA
SDK or TBB installation a prerequisite of the shipped CPU modem profile.
The Linux boundary test passed without TBB/CUDA system libraries.

PyInstaller extracts one-file runtime contents into temporary storage. A usable
temporary directory and enough disk space for extraction and enabled recordings
are required. On Linux a `noexec` temporary filesystem can prevent shared-library
loading; choose an appropriate per-user temporary location. This application is
not a native musl/Alpine binary. The
[PyInstaller 6.16 platform guidance](https://pyinstaller.org/en/v6.16.0/usage.html#making-gnu-linux-apps-forward-compatible)
explains why the system GLIBC requirement is not removed by bundling Python.

The Linux release's real-time evidence is for one modem on an AMD Ryzen 5 3600
with `acpi-cpufreq` policies set to `performance`. All 12 functional/timing probes
passed on that profile; the retained `schedutil` baseline passed data checks but
only 6 timing checks. The application does not change OS power policies.
See [exact Linux qualification](LINUX_QUALIFICATION.md). This is a measured host
configuration, not a proved minimum CPU, RAM amount or multiworker capacity.

The GLIBC 2.38 witness used an isolated Ubuntu 23.10 container with networking
disabled. [That release is end-of-life](https://lists.ubuntu.com/archives/ubuntu-announce/2024-July/000303.html);
it was a compatibility boundary test, not a deployment recommendation. Use a
maintained OS satisfying the requirements. The older private Linux LTO build's
GLIBC 2.36 result does not apply to this rc1 Linux ZIP.

## Terminal, serial and network prerequisites

- The TUI needs an interactive ANSI/VT-capable terminal. On Windows,
  [Windows Terminal](https://learn.microsoft.com/en-us/windows/terminal/install)
  is an available option; its own OS requirements also apply. Headless modem
  operation does not require a graphical desktop or interactive TUI.
- Linux defaults to the separately installed
  [serialsim driver](../packaging/serial/README.md), built for the running kernel,
  with read/write access to `/dev/ttyPipe`. PTY/TCP are explicit alternatives.
  The release's wire tests used TCP DTE, not a legacy application through that
  kernel driver.
- macOS defaults to PTY. Windows defaults to TCP; a legacy Windows COM application
  needs an external virtual-COM/TCP bridge. No signed native Windows COM driver
  is bundled. PTY and native serial/modem-control signals are not interchangeable.
  See [serial interfaces](SERIAL.md) and [desktop settings](DESKTOP.md).
- Calls need configured SIP credentials/routing and reachable SIP/RTP endpoints;
  no provider configuration is supplied. Native IAX calling is not implemented.

Separate command-path evidence now covers [Windows COM](WINDOWS_COM_QUALIFICATION.md)
and [macOS PTY](MACOS_PTY_QUALIFICATION.md) with actual legacy applications on the
stated profiles. It does not close older-OS, original installer, connected-call
signals, packet-loss, loaded-host or multiworker qualification. The release
remains experimental, with the existing provenance and commercial-review limits.

## Reproduce the static inspection

On a developer inspection host with PyInstaller and GNU `readelf`/`objdump`:

```sh
python packaging/runtime_requirements.py /path/to/el-modem \
  --target linux-x86_64 --output runtime.json
```

Other targets are `windows-x86_64`, `macos-x86_64` and `macos-arm64`; point at the
corresponding executable. This tool reads foreign binaries without running them,
ignores symbolic aliases/data files and uses only required ELF symbol versions
or the selected Mach-O architecture's deployment commands. Its report is not a
substitute for running the program on the target OS.
