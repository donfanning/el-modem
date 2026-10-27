"""ASCII command-reference pages, 80 columns and no fixed page length."""
import textwrap


PAGES = {
    '': ('BASIC COMMANDS', [
        ('A', 'Answer an incoming call.'),
        ('Dn', 'Dial n. Use ATDSn for a stored number or ATDL to redial.'),
        ('H', 'Hang up.'),
        ('O', 'Return to the current online data connection.'),
        ('En', 'Command echo: 0 off, 1 on.'),
        ('Qn', 'Result codes: 0 on, 1 quiet.'),
        ('Vn', 'Result format: 0 numeric, 1 words.'),
        ('Xn', 'Call-progress level: 0-4.'),
        ('Sr=n / Sr?', 'Write / read S-register r.'),
        ('In', 'Information screens: ATI0-ATI12 and ATI15-ATI17.'),
        ('Zn', 'Reload stored profile n (0-1).'),
    ], 'ATE0V1; ATI3'),
    '&': ('AMPERSAND COMMANDS', [
        ('&Gn', 'V.22 guard tone: 0 off, 1 550 Hz, 2 1800 Hz; high-channel answer only.'),
        ('&Cn', 'Carrier detect: 0 forced on, 1 follows carrier.'),
        ('&Dn', 'DTR drop: 0 ignore, 1 command mode, 2 hang up, 3 reset.'),
        ('&T0 / &T1', 'Stop a test / start the profile-specific software audio-loop test.'),
        ('&T3', 'Local DTE byte loopback test; not a remote or analog V.54 loop.'),
        ('&Kn', 'Flow control; 2,4-6 use XON/XOFF. Hardware lines depend on the serial backend.'),
        ('&F', 'Load factory defaults.'),
        ('&V', 'Display active and stored settings.'),
        ('&Wn', 'Save profile n (0-1) to NVRAM.'),
        ('&Zn=number', 'Store phone number n (0-3).'),
        ('&M / &Q', 'Accepted for compatibility only; do not enable synchronous DTE operation.'),
    ], 'AT&W0; AT&V'),
    'S': ('S-REGISTERS', [
        ('S0 / S1', 'Auto-answer ring count / current ring counter.'),
        ('S2', 'Escape character (default 43, plus sign).'),
        ('S3 / S4 / S5', 'Carriage return / line feed / backspace characters.'),
        ('S7 / S8', 'Carrier-wait timeout / dial-comma pause, in seconds.'),
        ('S10', 'Lost-carrier delay, in tenths of a second.'),
        ('S12', 'Escape guard time, in units of 1/50 second.'),
        ('Stored only', 'S11, S19, S25, S32, S33, S36, S37.'),
        ('S100-S106 SIP', 'Server, port, user, password, registrar, caller number, contact host.'),
        ('S110-S115 IAX', 'Server, port, user, password, caller number, register flag (retained; registration is automatic when configured).'),
        ('S120', 'Outbound trunk: 0 SIP, 1 IAX.'),
        ('Text values', 'ATSn=text sets a network register; ATSn? reads it. Passwords are masked. A text assignment must end the command line.'),
    ], 'ATS0=1; ATS7?; ATS120=1'),
    '%': ('PERCENT COMMANDS', [
        ('%C0 / %C1', 'V.42bis data compression disabled / offered. Offered does not mean negotiated.'),
    ], 'AT%C1'),
    '\\': ('BACKSLASH COMMANDS', [
        ('\\N0', 'Raw data, no error control.'),
        ('\\N1', 'Lab ARQ protocol (not V.42).'),
        ('\\N2', 'MNP 2-4 error control.'),
        ('\\N3', 'V.42 LAPM auto-reliable; falls back to raw.'),
    ], 'AT\\N3%C1'),
    'D': ('DIAL MODIFIERS', [
        ('0-9 * #', 'Dial digits.'),
        ('T / P', 'Tone / pulse selectors: accepted, ignored on a software line.'),
        (',', 'Pause for S8 seconds.'),
        (';', 'Return to command mode.'),
        ('Sn', 'Dial stored number n (0-3).'),
        ('L', 'Redial the last number.'),
    ], 'ATDT201; ATDS0; ATDL'),
    '+': ('EXTENDED COMMANDS', [
        ('+MS=mod,auto,min,max', 'Select modulation, negotiation and rate limits.'),
        ('Modulations', 'B103 B212 V21 V22 V22B V32 V32B V34. auto=1 runs V.8 negotiation.'),
        ('+MS? / +MS=?', 'Read the selection / list supported selections.'),
        ('+GMI / +GMM', 'Manufacturer / model identification.'),
        ('+GMR / +GCAP', 'Firmware revision / capability identification.'),
        ('+CLIP=n / +CLIP?', 'Enable/disable or read unsolicited incoming SIP caller identity.'),
        ('+IPR / +IFC / +ICF', 'DTE compatibility settings; do not create physical line clocks or synchronous operation.'),
    ], 'AT+MS?; AT+GMR'),
}


def help_screen(stem):
    page = PAGES.get(stem)
    if page is None:
        return None
    title, entries, examples = page
    lines = ['El-Modem Command Reference', '=' * 78, f'HELP - {title}:', '',
             f"  {'Command':<20}  Description", '  ' + '-' * 20 + '  ' + '-' * 54]
    for command, description in entries:
        prefix = f'  {command:<20}  '
        lines.extend(textwrap.wrap(description, width=80, initial_indent=prefix,
                                   subsequent_indent=' ' * 24, break_on_hyphens=False))
    lines += ['', '  Examples (separate commands):']
    lines += textwrap.wrap(examples, width=80, initial_indent='    ', subsequent_indent='    ')
    lines += ['', '  More help: AT$  AT&$  ATS$  AT%$  ATD$  AT\\$  AT+$', '']
    return lines
