# Windows virtual serial prerequisite

The modem executable runs unprivileged. Full virtual-serial support on
Windows requires com0com 3.0.0.0 (pinned archive SHA-256
`6e5d4359865277430d4ae88c73fb7e648a0ed8e81aea5002478179cfcb0bb0e1`,
driver and catalog signed by `CN=CyberCircuits`). The modem does not
install or configure the driver automatically.

An administrator installs the signed package once using com0com's own
installer, then creates each modem's pair with the terminal side wired
so DSR is independent of carrier (the modem's DTR output carries
carrier only; with default wiring DSR and DCD would be tied together):

```powershell
setupc.exe install PortName=COM41,dsr=on PortName=COM42
```

The modem side (COM42 here) keeps com0com defaults so peer presence
(DSR) and RTS gating (CTS) read correctly. Give each modem instance its
own pair; the supervisor rejects duplicate devices. Do not grant the
modem process administrator rights to work around port ACLs; fix the
pair wiring and device permissions instead.

Qualification (no hardware beyond the installed pair):

```sh
PYTHONPATH=lab/asterisk python -m unittest test_modem_com -v
```

plus a live AT/ATI/ATZ round trip through the peer port, a CTS
backpressure check under `AT&K3`, and a DTR-drop hangup check. RI has
no pin action on com0com (two DCE outputs cannot drive DCD, CTS and RI
independently with static wiring); incoming-ring detection reaches
terminals as the Hayes `RING` result code. A future DCE-capable
presenting driver would light the RI pin with no DCE changes; until
then com0com pairs are consumed, not presented, and bridging them to
TCP is not a substitute for this backend.
