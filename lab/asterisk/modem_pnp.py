"""Serial PnP identity. Reporting it is separate from DTR/RTS enumeration."""
HARDWARE_ID = 'ELM0001'  # Project-local ID, not a registered vendor allocation.
COMPATIBLE_ID = 'PNPC103'  # Inbox generic external modem, not a 33.6-specific ID.
WINDOWS_PROFILE = 'MDMGEN336'
MODEL = 'El-Modem 33.6 Data Modem'


def identification():
    """External COM PnP packet, including additive checksum and delimiters.

    Revision 1.00 is two six-bit bytes (100 = 1*64 + 36). No serial number
    is claimed. The companion INF maps SERENUM\\ELM0001 to mdmgen.inf's
    Gen336 profile. This does not perform the electrical discovery sequence.
    """
    packet = '(\x01$' + HARDWARE_ID + '\\\\MODEM\\' + COMPATIBLE_ID + '\\' + MODEL
    checksum = (sum(packet.encode('ascii')) + ord(')')) & 0xff
    return packet + f'{checksum:02X})'
