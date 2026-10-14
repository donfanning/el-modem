# macOS PTY qualification

The exact published `v0.1.0-rc1` executables passed a real legacy-terminal test
on **macOS 15.7.9, Intel and ARM64**, using Homebrew minicom 2.11.1.
No executable was rebuilt or replaced, and no calls were placed.

```text
minicom → actual macOS PTY → packaged modem supervisor/worker
```

For each architecture, minicom sent `ATZ`, `ATE0`, `ATI3`, `ATS0=9`, readback
(`9`), reset and readback (`0`). It exited without resetting the modem, then
reopened the same PTY and repeated the checks. Both minicom sessions exited
zero. The supervisor exited cooperatively with zero status, removed its serial
link and left no worker traceback.

See the [qualification report and exact hashes](MACOS_PTY_QUALIFICATION.json),
[Intel transcript](macos-pty-evidence/macos-x86_64/first-capture.txt),
[ARM transcript](macos-pty-evidence/macos-arm64/first-capture.txt), and
[successful run](https://github.com/donfanning/el-modem/actions/runs/38643125397).
The [harness](../packaging/qualify_pty_terminal.py) types into minicom's UI only;
Python never opens or writes the modem's serial endpoint.

## Using a legacy terminal

Choose PTY for the modem instance and start it. Open the reported serial link
in minicom, with hardware and software flow control disabled:

```sh
brew install minicom
minicom -o -D /path/to/instances/modem-1/serial -b 115200
```

Configure flow control in minicom's serial-port settings. `-o` skips automatic
modem initialization; send your own AT commands. Ctrl-A Q exits without a modem
reset. The built-in el-modem terminal and minicom are separate applications.

## Limits and retained failure

This is command/PTY compatibility, not connected DSP payload, PSTN, physical
RS232 signaling, performance under load or multiple simultaneous modems.
It does not qualify the older declared macOS deployment floors, nor the GUI
Gatekeeper/notarization path; these binaries remain ad-hoc signed.

The first [run](https://github.com/donfanning/el-modem/actions/runs/38642695979)
received correct AT responses but timed out closing minicom; forced cleanup
also interrupted the supervisor. The harness was corrected to read the quit
confirmation and drain its UI while exiting, and to signal the supervisor
cooperatively. The same frozen binaries then passed. The first run remains a
failed gate. There is one passing run per architecture, not a stability campaign.
