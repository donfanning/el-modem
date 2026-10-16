"""Offline gates through the DCE's real AT profile and audio-session factories.

No SIP user agent, socket, or sender thread is started. These checks qualify
packaged dependencies and local DSP; they do not qualify independent peers,
SIP transport, or real-time operation.
"""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sys
import os
import struct
import time


class NoNetwork:
    def sendto(self, *args, **kwargs):
        raise AssertionError('offline self-test attempted network transmission')


@contextmanager
def controlled_environment():
    names = ('MODEM_CAPTURE', 'MODEM_CAPTURE_REQUIRED', 'MODEM_CAPTURE_TIMELINE',
             'V32_CAPTURE', 'EL_MODEM_DSP_CAPTURE', 'EL_MODEM_METRICS_FILE',
             'MODEM_DEBUG', 'V34_RATE', 'V34_SYMS')
    saved = {name: os.environ.pop(name) for name in names if name in os.environ}
    try:
        yield
    finally:
        for name in names:
            os.environ.pop(name, None)
        os.environ.update(saved)


def profile_options(dce_type, dispatch, profile):
    dce = dce_type()
    reply = dce.feed(('ATZ\rATE0\rAT+MS=' + profile + '\r').encode())
    if b'ERROR' in reply or not reply.endswith(b'OK\r\n'):
        raise AssertionError('AT profile rejected: ' + profile)
    return dispatch(dce.ms)


def make_session(session_type, options, calling):
    return session_type(NoNetwork(), ('127.0.0.1', 9),
                        'originate' if calling else 'answer',
                        'answer' if calling else 'originate', **options)


def probe_session(session):
    samples = session.transmitter.samples(160)
    if len(samples) != 160:
        raise AssertionError('transmitter returned incorrect block length')
    session.receiver.feed_pcm(bytes(320))
    if session._thread.is_alive():
        raise AssertionError('offline self-test started sender thread')


def factories(session_type, dispatch, dce_type):
    rows = []
    profiles = ('B103,0,300,300', 'V21,0,300,300', 'B212,0,1200,1200',
                'V22,0,1200,1200', 'V22B,0,1200,2400', 'V32,0,4800,9600',
                'V32B,0,4800,14400', 'V34,0,2400,33600')
    for profile in profiles:
        for calling in (True, False):
            session = None
            try:
                options = profile_options(dce_type, dispatch, profile)
                session = make_session(session_type, options, calling)
                probe_session(session)
                engine = session._v22
                rows.append(dict(gate='forced_factory', profile=profile,
                                 role='caller' if calling else 'answerer',
                                 engine=type(engine).__name__ if engine else options['line_mode'], passed=True))
            finally:
                if session is not None:
                    session.stop()
    # Exercise each deferred factory inside the actual all-in-one chain.
    # The dispatch is explicit here, not claimed as V.8 negotiation evidence.
    for calling in (True, False):
        for kind in ('v22', 'v32', 'v34'):
            session = None
            try:
                options = profile_options(dce_type, dispatch, 'V34,1,300,33600')
                session = make_session(session_type, options, calling)
                session._v22._choose(kind)
                probe_session(session)
                engine = session._v22.engine
                if hasattr(engine, 'start_v42'):
                    for protocol in ('lapm', 'mnp'):
                        engine.start_v42(calling, rate=9600, detect=False,
                                        compression=(3, 1024, 8), protocol=protocol)
                        engine.v42_feed_bytes(b'\x00\xffdependency check')
                        engine.v42_tx_bits(160)
                        engine.v42_rx_bits([1] * 160)
                rows.append(dict(gate='deferred_factory_and_link', profile='V34,1,300,33600',
                                 role='caller' if calling else 'answerer', selected=kind,
                                 engine=type(engine).__name__, passed=True))
            finally:
                if session is not None:
                    session.stop()
    return rows


def transmitter_idle(transmitter) -> bool:
    """True once the transmitter has nothing queued, prologue included.

    pending() reports queued bits only, so it cannot see an undrained
    Bell 103 prologue on its own.
    """
    prologue = getattr(transmitter, '_prologue', None)
    if prologue and transmitter._prologue_index < len(prologue):
        return False
    return not transmitter.pending()


def payload_exchange(session_type, options_a, options_b, expected, limit_seconds=60):
    """Exact byte gate on the production session's PCM adapters, no UDP."""
    sessions = []
    try:
        a = make_session(session_type, options_a, True); sessions.append(a)
        b = make_session(session_type, options_b, False); sessions.append(b)
        da = bytes(range(32)) + b'caller\r\n\x7f\x80\xfe\xff'
        db = bytes(range(31, -1, -1)) + b'answerer\r\n\xff\xfe\x80\x7f'
        ga, gb = bytearray(), bytearray()
        to_a, to_b = [0] * 40, [0] * 40
        sent = False
        for tick in range(int(limit_seconds * 50)):
            sa, sb = a.transmitter.samples(160), b.transmitter.samples(160)
            to_a.extend(sb); to_b.extend(sa)
            pa, pb = to_a[:160], to_b[:160]
            del to_a[:160]; del to_b[:160]
            pcm_a = struct.pack('<160h', *[int(max(-32768, min(32767, v))) for v in pa])
            pcm_b = struct.pack('<160h', *[int(max(-32768, min(32767, v))) for v in pb])
            ga.extend(a._receive_pcm(pcm_a)); gb.extend(b._receive_pcm(pcm_b))
            a.poll(a._created + tick / 50); b.poll(b._created + tick / 50)
            if not sent and tick >= 50 and a.negotiation_complete and b.negotiation_complete:
                # Only feed once the handshake has finished going out. The
                # answerer's Bell 103 prologue is still being transmitted for a
                # while after negotiation completes, and those bytes would
                # otherwise be decoded ahead of the payload and fail this gate.
                if transmitter_idle(a.transmitter) and transmitter_idle(b.transmitter):
                    del ga[:]; del gb[:]
                    a.transmitter.feed_bytes(da); b.transmitter.feed_bytes(db)
                    sent = True
            if sent and len(ga) >= len(db) and len(gb) >= len(da):
                break
        if bytes(ga) != db or bytes(gb) != da:
            raise AssertionError('payload mismatch: caller received %r, answerer received %r' % (bytes(ga), bytes(gb)))
        kinds = [getattr(s._v22, 'kind', s._v22.__class__.__name__ if s._v22 else s.mode) for s in sessions]
        if expected is not None and kinds != [expected, expected]:
            raise AssertionError('wrong selected engines: ' + repr(kinds))
        return dict(gate='pcm_payload', selected=kinds, rates=[s._v22.bit_rate if s._v22 else 300 for s in sessions],
                    link_modes=[getattr(s._v22, 'v42_mode', 'off') for s in sessions],
                    link_classes=[type(getattr(s._v22, '_v42', None)).__name__ for s in sessions],
                    compression=[getattr(s._v22, 'v42_compression', None) for s in sessions],
                    passed=True, sample_seconds=(tick + 1) / 50,
                    caller_rx_bytes=len(ga), answerer_rx_bytes=len(gb),
                    caller_rx_sha256=hashlib.sha256(ga).hexdigest(),
                    answerer_rx_sha256=hashlib.sha256(gb).hexdigest())
    finally:
        for session in sessions:
            session.stop()


def payloads(session_type, dispatch, dce_type):
    from v8 import V8_MOD_V22, V8_MOD_V32, V8_MOD_V34
    rows = []
    for profile in ('B103,0,300,300', 'V21,0,300,300', 'V22,0,1200,1200', 'V22B,0,1200,2400'):
        options = profile_options(dce_type, dispatch, profile)
        row = payload_exchange(session_type, options, options, None)
        row['profile'] = profile
        if row['rates'] != [int(profile.split(',')[-1])] * 2:
            raise AssertionError('forced profile negotiated wrong rates: ' + repr(row['rates']))
        rows.append(row)
    # The caller always uses the complete product profile. The answerer's
    # allowed menu selects a lower family; higher options remain offered.
    for kind, allow in (('v22', V8_MOD_V22), ('v32', V8_MOD_V22 | V8_MOD_V32),
                        ('v34', V8_MOD_V22 | V8_MOD_V32 | V8_MOD_V34)):
        options_a = profile_options(dce_type, dispatch, 'V34,1,300,33600')
        options_b = profile_options(dce_type, dispatch, 'V34,1,300,33600')
        options_b['chain']['allow'] = allow
        row = payload_exchange(session_type, options_a, options_b, kind)
        row['profile'] = 'V34,1,300,33600'; row['peer_menu'] = kind
        if row['rates'] != [{'v22': 2400, 'v32': 14400, 'v34': 33600}[kind]] * 2:
            raise AssertionError('automatic profile negotiated wrong rates: ' + repr(row['rates']))
        rows.append(row)
    # DCE starts error control via poll(), exactly as in an established call.
    for link in ('lapm', 'v42bis', 'mnp'):
        a = profile_options(dce_type, dispatch, 'V34,1,300,33600')
        b = profile_options(dce_type, dispatch, 'V34,1,300,33600')
        for options in (a, b):
            options.update(v42=True, v42bis=link == 'v42bis', mnp=link == 'mnp')
        row = payload_exchange(session_type, a, b, 'v34')
        if row['link_modes'] != ['lapm', 'lapm']:
            raise AssertionError('link failed to establish: ' + repr(row['link_modes']))
        expected_link = 'MnpLink' if link == 'mnp' else 'V42Link'
        if row['link_classes'] != [expected_link] * 2:
            raise AssertionError('wrong error-control implementation: ' + repr(row['link_classes']))
        if link == 'v42bis' and not all(row['compression']):
            raise AssertionError('compression did not negotiate')
        row['protocol'] = link; rows.append(row)
    # Legacy V22 answering signal, with no V.8 menu.
    a = profile_options(dce_type, dispatch, 'V34,1,300,33600')
    b = profile_options(dce_type, dispatch, 'V22B,0,1200,2400')
    row = payload_exchange(session_type, a, b, None)
    if row['selected'][0] != 'v22':
        raise AssertionError('legacy answerer did not select V22')
    row['peer_menu'] = 'legacy_v22'; rows.append(row)
    return rows


def run(session_type, dispatch, dce_type, *, with_payload=False):
    started = time.monotonic()
    manifest = Path(getattr(sys, '_MEIPASS', Path(__file__).parent)) / 'el-modem-build.json'
    report = dict(build_manifest=json.loads(manifest.read_text()) if manifest.exists() else None, schema=1, scope='offline DCE factory and local PCM gates; no network or independent-peer qualification',
                  status='FAIL', gates=[], payload_requested=with_payload)
    with controlled_environment():
        try:
            report['gates'].extend(factories(session_type, dispatch, dce_type))
            if with_payload:
                report['gates'].extend(payloads(session_type, dispatch, dce_type))
            report['status'] = 'PASS'
        except Exception as error:
            report['error'] = type(error).__name__ + ': ' + str(error)
    report['elapsed_seconds'] = time.monotonic() - started
    return report
