# Asterisk / PJSIP one-way audio (external media)

If El-Modem registers and places a call but the far side cannot hear you (or
the call dies shortly after connect), Asterisk may be advertising its
internal container/lan address in SDP instead of the address you actually
dialled.  El-Modem now also latches send-to to the address it receives RTP
from after three consecutive PCMU packets from one address and SSRC
(`modem_dce.py` `OnlineAudioSession.note_rtp_source`). Those initial packets
are buffered and delivered once qualified. The address and SSRC then stay
locked for the call; a different peer or NAT rebinding requires a new SIP
session. Source probation is not media authentication: restrict accessible RTP
ports to the intended peers. The correct
fix is at the SIP transport on the Asterisk side:

```ini
[udp-transport]
type = transport
protocol = udp
bind = 0.0.0.0:5060
external_signaling_address = <public/external Asterisk address>
external_media_address    = <public/external Asterisk address>
local_net                = <Asterisk-side subnet, e.g. 192.0.2.0/24; documentation-only>
```

Place the `external_*` settings on the **PJSIP transport** (`pjsip.conf`),
not only `rtp.conf`.  `rtp.conf` / `res_rtp_asterisk` `external_media_address`
is read by the RTP stack but is frequently ignored by PJSIP SDP rewriting.

Only list networks reachable **by Asterisk itself** in `local_net`.  If you also
list the El-Modem host's subnet, Asterisk assumes that host is "local" and
skips the external-address rewrite, reintroducing the one-way audio symptom.

Verify with server-side tcpdump after a call:

```
sudo tcpdump -i any -n 'udp and host <windows-host-ip> and port 10000'
```

A correct flow shows bidirectional UDP on 192.168.x.x ports (not the
private container address in the SDP).
