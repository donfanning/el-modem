"""Reactive V.92 Phase 4 controller (clause 9.6), both modems, detection in the loop, with the hand-over
to data mode.

Each side sends a stream of bits through its coder, reads the peer's stream back with the matching
decoder, and decides what to send next from the frames it has detected, with the rules of
``v92_final_training`` but driven by units of time (samples upstream, codewords downstream) rather
than slots:

* TRN2 first (digital at least ``digital_trn`` units; analogue at least ``analogue_trn``, or as soon
  as an SUV is seen), then SUV frames;
* the first SUV or CP seen from the peer makes it send a single CP, then SUV again;
* the acknowledge bit means "I have received your CP", so it is set from the first CP seen onward;
* a CP with no acknowledge seen within ``ack_wait`` units of being sent is repeated;
* once it has sent an acknowledging frame and has seen an acknowledge or the peer's E, it sends E
  and then goes to data mode; connected when it has sent E and seen the peer's E.

Hand-over: frames are padded with zero fill bits to a whole mapper frame (downstream: S + K bits, six
codewords) or data frame (upstream: 12 symbols), as the text's fill-bit rule asks, so E and everything
after it start on a frame boundary. The receiver finds the unit at which the peer's E ends, feeds the
rest of that chunk to the data-mode receiver and every later chunk too. Data mode is then
B1 (48 frames of scrambled ones) and a payload:

* downstream: the V.90 mapper, parameters from the analogue modem's CPu (``v92_data_mode``);
* upstream: ``PcmUpstreamTransmitter`` with the digital modem's CPd, received by
  ``PcmUpstreamStreamReceiver``. The channel/precoder model targets unit gain;
  equalized paths validate a residual gain correction on separate B1 segments.

Assumptions as in ``v92_phase4_upstream`` (symbol-label bit order; the upstream fill to 12 symbols,
which the text does not state) and ``v92_phase4_downstream``. One unit of time is one symbol upstream and
one codeword downstream, so both directions run at 8000 units/s.
"""

from __future__ import annotations

import dataclasses
import os
import threading
import time
from collections import deque

import numpy as np

import v34_g711
import v34_tx
import v90_digital_phase4 as dp4
import v90_digital_phase4_rx as rx4
import v92_cp
from v90_dil import g711_ucode_codewords
import v92_data_mode as dm
import v92_pcm_upstream as pu
import v92_pcm_upstream_rx as purx
import v92_phase3
import v92_phase4_upstream as up

B1_FRAMES = 48


def _e_boundary_cut(decoder, scanner_bit_origin: int, e_end_bit: int, base: int,
                    rx_filter_delay: int = 0) -> tuple[int, int, int]:
    """Map the scanner's exclusive end-of-E bit to a raw sample and chunk offset."""
    bit_index = int(scanner_bit_origin) + int(e_end_bit)
    raw_sample = int(decoder.raw_of_bit(bit_index))
    return bit_index, raw_sample, raw_sample - int(base) - int(rx_filter_delay)


def _units_from_raw_cut(units, base: int, cut: int, history_window=None):
    """Return units beginning at the absolute raw cut, using history for cross-chunk cuts.

    ``history_window`` is ``(combined_samples, absolute_origin)`` and must cover
    the requested cut. If it does not, retain the legacy clipped behavior.
    """
    base, cut = int(base), int(cut)
    raw_cut = base + cut
    if cut < 0 and history_window is not None:
        combined, origin = history_window
        offset = raw_cut - int(origin)
        if 0 <= offset <= len(combined):
            return combined[offset:], raw_cut, True
    start = max(cut, 0)
    start = min(start, len(units))
    return units[start:], base + start, False


class _Scanner:
    """Incremental frame / E detector on a decoded bit stream; stops at the end of E."""

    MAX_FRAME_BITS = 9000
    FB1_PROBE_BITS = 128            # E by its zero run: this many bits after it,
    FB1_PROBE_ONES = 100            # at least this many ones (FB1u, past a 23-bit descrambler transient)
    E_RUN_MAX_FRAMES = 8            # e_to_run_end: a zero run this many alignment frames long ends E where it stands

    def __init__(self, parsers, e_bits: int, align_bits: int, e_before_fb1: bool = False,
                 e_to_run_end: bool = False):
        self.parsers, self.e_bits, self.align = parsers, e_bits, align_bits
        # Ed's length differs between peers: ours is 2 TRN2d frames, Astound's 5 (2 data frames of zeros, 44 bits,
        # in whole 10-bit TRN2d frames; live calls 2 and 5, 2026-11-19). With e_to_run_end, a run of at least
        # e_bits zeros ends E where it stops (a one, or the decoder meeting B1d's codewords: end_of_stream),
        # rounded down to the alignment, since the next frame can decode as a few more zeros.
        self.e_to_run_end = bool(e_to_run_end)
        self._tail_start = 0
        # FPE: E2u is a data frame of zeros and FB1u follows; take E's end where a run of at least e_bits zeros
        # gives way to FB1u's ones, whatever the frame before it. SmartLink damages a bit at each sequence switch
        # in FPE (a 16-one sync, a CRC's last bit), so its SUVu' before E2u may not parse (rig, 26,667, 2026-11-18).
        # A CP frame cannot match: its words are separated by start bits. A late find stays in E's frame.
        self.e_before_fb1 = bool(e_before_fb1)
        self._fb1_zeros = 0
        self._fb1_end: int | None = None
        self._fb1_seen = self._fb1_ones = 0
        self.bits: list[int] = []
        # Absolute offset of bits[0] in the stream. Long training runs are
        # deliberately trimmed below; E's end index still has to map back to
        # the decoder's absolute bit coordinate.
        self._bit_base = 0
        self.pos = 0
        self.mode = "sync"
        self.zeros = 0
        self.e_end_bit: int | None = None
        self.failed = False
        self._next_try = 0                  # do not re-parse until this many bits are past ``pos``
        self._align_origin: int | None = None
        # Search for fallback sync candidates incrementally.  The old code
        # rescanned the entire retained stream after every failed parse, and
        # ran every parser at every candidate.  During V.92 training this
        # could turn a harmless false sync into unbounded RX work under the
        # shared media lock.
        self._later_scan_pos = 0
        self._later_candidates = deque()  # [bit offset, next available-bit threshold]

    def _trim_prefix(self, count: int) -> None:
        count = max(0, min(int(count), len(self.bits)))
        if not count:
            return
        del self.bits[:count]
        self._bit_base += count
        self.pos = max(0, self.pos - count)
        self._later_scan_pos = max(0, self._later_scan_pos - count)
        self._later_candidates = deque(
            [offset - count, retry]
            for offset, retry in self._later_candidates if offset >= count)

    @property
    def e_seen(self) -> bool:
        return self.e_end_bit is not None

    def end_of_stream(self) -> None:
        """e_to_run_end: the decoded stream stops (or a one arrives) after a run of at least e_bits zeros; E ends at
        the run's last whole alignment frame."""
        if (self.e_to_run_end and not self.e_seen and self.mode == "tail"
                and self.zeros >= self.e_bits):
            self.e_end_bit = self._tail_start + self.zeros // self.align * self.align

    def _e_before_fb1(self, new_bits, first: int) -> int | None:
        """The absolute end of an E2u zero run that FB1u's ones follow, or None (``first``: the first bit's index)."""
        for k, bit in enumerate(new_bits):
            if self._fb1_end is not None:
                self._fb1_seen += 1
                self._fb1_ones += bit
                if self._fb1_seen == self.FB1_PROBE_BITS:
                    if self._fb1_ones >= self.FB1_PROBE_ONES:
                        return self._fb1_end
                    self._fb1_end = None
                continue
            if bit:
                if self._fb1_zeros >= self.e_bits:
                    self._fb1_end, self._fb1_seen, self._fb1_ones = first + k, 1, 1
                self._fb1_zeros = 0
            else:
                self._fb1_zeros += 1
        return None

    def feed(self, new_bits) -> list[dict]:
        out: list[dict] = []
        if self.e_seen or self.failed:
            return out
        if self.e_before_fb1:
            end = self._e_before_fb1(new_bits, self._bit_base + len(self.bits))
            if end is not None:
                self.bits.extend(new_bits)
                self.e_end_bit = end
                return out
        self.bits.extend(new_bits)
        while not (self.failed or self.e_seen):
            if self.mode == "sync":
                # A reset can happen while the stream is in the middle of a
                # frame (notably after V.92 R-t).  The first zero is not enough
                # to identify a frame boundary: payload and fill bits contain
                # zeros too.  Require the complete 17-one sync word and its
                # zero start bit, retaining a suffix so a word split across
                # feed() calls is still found. The sync starts at ``pos`` or later: ones before ``pos`` would
                # find the candidate just abandoned again, and the scanner looped there for good (a long run of
                # ones, then more than MAX_FRAME_BITS that do not parse; a 60 ms RTP gap in TRN2d, 2026-11-19).
                z = self.pos + 17
                found = None
                while True:
                    try:
                        z = self.bits.index(0, z)
                    except ValueError:
                        break
                    if z >= 17 and all(self.bits[z - 17:z]):
                        found = z - 17
                        break
                    z += 1
                if found is None:
                    # Keep only the sync-sized suffix. The old scanner kept
                    # the entire (potentially tens of thousands of bits) TRN
                    # buffer and searched it again on every RTP feed. A run
                    # of descrambled ones therefore made the receive path
                    # quadratic while TX waited on the shared media lock.
                    drop = max(0, len(self.bits) - 17)
                    if drop:
                        self._trim_prefix(drop)
                    self.pos = 0
                    self._later_scan_pos = 0
                    break
                self.pos = found
                self.mode = "frames"
            if self.mode == "frames":
                if self.pos >= len(self.bits):
                    break
                if self.bits[self.pos] == 0:
                    self.mode, self.zeros = "tail", 0
                    self._tail_start = self._bit_base + self.pos
                    continue
                avail = len(self.bits) - self.pos
                if avail < self._next_try:
                    break                   # a failed parse is not repeated until enough new bits have come
                chunk = self.bits[self.pos:self.pos + self.MAX_FRAME_BITS]
                parsed = next((r for r in (p(chunk) for p in self.parsers) if r is not None), None)
                if parsed is None:
                    # TRN2u may decode to a very long run of ones. Once a
                    # false sync's start bit has failed, a suffix containing
                    # only ones cannot contain another sync *start* bit.
                    # Retain its 17-bit suffix and return to incremental sync
                    # search instead of repeatedly scanning/copying the whole
                    # training run while holding the media lock.
                    suffix = self.pos + 18
                    if suffix < len(self.bits) and all(self.bits[suffix:]):
                        drop = max(0, len(self.bits) - 17)
                        if drop:
                            self._trim_prefix(drop)
                        self.pos = 0
                        self.mode = "sync"
                        self._next_try = 0
                        self._later_scan_pos = 0
                        self._later_candidates.clear()
                        continue
                    # Incrementally discover later syncs.  Bound both the
                    # number of candidates scanned and parser calls per feed;
                    # retry thresholds let incomplete long CP frames mature
                    # without reparsing them for every 20 ms packet.
                    scan = max(self.pos + 1, self._later_scan_pos)
                    scanned = 0
                    while scanned < 12:
                        try:
                            z = self.bits.index(0, scan + 17)
                        except ValueError:
                            scan = max(scan, len(self.bits) - 17)
                            break
                        candidate = z - 17
                        scan = z + 1
                        if candidate > self.pos and all(self.bits[candidate:z]):
                            # _later_scan_pos advances monotonically, so a
                            # candidate can only be discovered once.
                            self._later_candidates.append([candidate, 0])
                            scanned += 1
                    self._later_scan_pos = scan

                    # Candidates whose maximum frame window has already
                    # passed can never parse. Drop them from the head (the
                    # queue is in stream order) so a dense false-sync stream
                    # cannot grow the fallback queue for the whole call.
                    oldest_live = len(self.bits) - self.MAX_FRAME_BITS
                    while (self._later_candidates
                           and self._later_candidates[0][0] < oldest_live):
                        self._later_candidates.popleft()

                    parser_calls = 0
                    later = None
                    # Round-robin a small number of pending candidates. A
                    # stream containing many false syncs must not turn queue
                    # maintenance itself into a quadratic scan.
                    for _ in range(min(4, len(self._later_candidates))):
                        candidate, retry_at = self._later_candidates.popleft()
                        available = len(self.bits) - candidate
                        if available > self.MAX_FRAME_BITS:
                            continue
                        if available < retry_at:
                            self._later_candidates.append([candidate, retry_at])
                            continue
                        if parser_calls >= 4:
                            self._later_candidates.appendleft([candidate, retry_at])
                            continue
                        chunk = self.bits[candidate:candidate + self.MAX_FRAME_BITS]
                        candidate_frame = None
                        for parser in self.parsers:
                            parser_calls += 1
                            candidate_frame = parser(chunk)
                            if candidate_frame is not None or parser_calls >= 4:
                                break
                        if candidate_frame is not None:
                            later = candidate
                            break
                        retry_at = available + max(12, available // 256)
                        self._later_candidates.append([candidate, retry_at])
                    if later is not None:
                        self._later_candidates = deque(
                            item for item in self._later_candidates if item[0] > later)
                        self.pos = later
                        self._next_try = 0
                        continue
                    if avail > self.MAX_FRAME_BITS:
                        # This candidate cannot be a complete frame anymore;
                        # slide past it and resume hunting instead of failing
                        # the whole control exchange.
                        self.pos += 1
                        self.mode = "sync"
                        self._next_try = 0
                        self._later_scan_pos = self.pos
                        continue
                    # Coarse retries avoid re-parsing an incomplete long CPd
                    # on every small media chunk.
                    self._next_try = avail + max(12, avail // 256)
                    break
                self._next_try = 0
                if self._align_origin is None:
                    # Anchor fill to a parsed frame, never to a false sync in
                    # training. This is important after a fresh Phase 4 start.
                    self._align_origin = (self._bit_base + self.pos) % self.align
                frame_end = self.pos + int(parsed["frame_bits"])
                end_absolute = self._bit_base + frame_end
                origin = 0 if self._align_origin is None else self._align_origin
                end_absolute += -(end_absolute - origin) % self.align  # mapper-aligned fill
                end = end_absolute - self._bit_base
                if end > len(self.bits):
                    break                                 # wait for the fill bits
                if any(self.bits[frame_end:end]):
                    # The peer's fill stopped short of this alignment: SmartLink pads FPE control to 72 bits in a
                    # 40-bit (26,667) data mode. The next frame starts inside it; carry on from the frame's end.
                    end = frame_end
                out.append(parsed)
                self.pos = end
                self._later_candidates = deque(
                    item for item in self._later_candidates if item[0] >= end)
                self._later_scan_pos = max(self._later_scan_pos, end)
                continue
            if self.mode == "tail":
                while self.pos < len(self.bits) and not self.e_seen:
                    bit = self.bits[self.pos]
                    self.pos += 1
                    if bit:
                        if self.e_to_run_end and self.zeros >= self.e_bits:
                            self.end_of_stream()
                            break
                        # Ones before a whole E: fill shorter than E, then another frame (or not our stream, which
                        # the sync hunt never leaves). Hunt for its sync from this bit.
                        self.pos -= 1
                        self.mode, self.zeros = "sync", 0
                        break
                    self.zeros += 1
                    if not self.e_to_run_end:
                        if self.zeros >= self.e_bits:
                            self.e_end_bit = self._bit_base + self.pos
                    elif self.zeros >= max(self.e_bits, self.E_RUN_MAX_FRAMES * self.align):
                        self.end_of_stream()
                if self.mode == "sync":
                    continue
                break
        return out


class _UpstreamDecoder:
    """Acquire TRN2u, fit its known sequence, then decode the peer's upstream bits."""

    START_RUN = 3
    LU_PROBE_SAMPLES = 1200
    LU_PROBE_MAX_SAMPLES = 4096
    # The paired training peer can terminate TRN2u early when it sees our SUV.
    # A 1008-symbol prefix is long enough for a stable 48-tap fit and stays
    # ahead of that exchange; real peers normally provide substantially more.
    TRN_SEARCH_SYMBOLS = 1008
    TRN_FIT_SYMBOLS = 1008
    TRN_TAPS = 48
    TRN_VITERBI_TAPS = 96          # trn2u-fit-viterbi data receiver: halves the arm-10c residual sigma vs 48 taps
    TRN_MAX_DELAY = 40
    TRN_MIN_CORRELATION = 0.28
    MAX_ACQUISITION_ATTEMPTS = 2

    def __init__(self, points: int, lu: float | None, *, initial_sign: int = 0,
                 acquire_trn2u: bool = False, async_fit: bool = True,
                 fit_output_bias: bool = False):
        self.coder, self.lu = up.SymbolCoder(points, initial_sign=initial_sign), lu
        self._decision_lu = lu
        self.points = points
        self._initial_sign = int(initial_sign) & 1
        self._acquire_trn2u = bool(acquire_trn2u)
        self._async_fit = bool(async_fit)
        self._fit_output_bias = bool(fit_output_bias)
        raw_fit_taps = os.environ.get("EL_MODEM_V92_TRN2U_FIT_TAPS")
        try:
            default_taps = self.TRN_VITERBI_TAPS if self._fit_output_bias else self.TRN_TAPS
            self.trn_fit_taps = default_taps if raw_fit_taps is None else int(raw_fit_taps)
        except ValueError as exc:
            raise ValueError("EL_MODEM_V92_TRN2U_FIT_TAPS must be an integer") from exc
        if not 1 <= self.trn_fit_taps <= 128:
            raise ValueError("EL_MODEM_V92_TRN2U_FIT_TAPS must be in 1..128")
        self.trn_fit_max_delay = self.TRN_MAX_DELAY
        self.fit_output_bias = 0.0
        self.acquisition_state = "search" if acquire_trn2u else "legacy"
        self.trn2u_at: int | None = None
        self.eq_delay: int | None = None
        self.eq_residual: float | None = None
        self.fit_weights: np.ndarray | None = None
        self.fit_target_rms: float | None = None
        self._search_tail = np.empty(0, dtype=float)
        self._search_template = self._make_trn_template() if acquire_trn2u else np.empty(0)
        self._candidate: np.ndarray | None = None
        self._candidate_at: int | None = None
        self._fit_result: list = []
        self._fit_thread: threading.Thread | None = None
        self.fit_wall_seconds: float | None = None
        self.fit_samples = 0
        self.acquisition_attempts = 0
        self.acquisition_retries = 0
        self.acquisition_failures: list[str] = []
        self._stream_filter = None
        self.first_non_one_bit: int | None = None
        self.acquisition_error: str | None = None
        self.zero_bits = 0
        self.decoded_codewords = 0
        self._auto_lu = lu is None
        self._lu_probe = np.empty(0, dtype=float)
        self._lu_probe_base: int | None = None
        self.start_raw: int | None = None
        self.raw_in = 0
        self._held = np.empty(0)
        self.nb = self.coder.bits
        self.bits_out = 0

    def set_initial_sign(self, initial_sign: int) -> None:
        """Carry the differential sign decoded from E1u into TRN2u reception."""
        if self.raw_in or self.start_raw is not None or self.bits_out:
            raise RuntimeError("cannot change the Phase-4 sign after upstream decoding starts")
        self._initial_sign = int(initial_sign) & 1
        self.coder._rx_sign = self._initial_sign
        if self._acquire_trn2u:
            self._search_template = self._make_trn_template()

    def _make_trn_template(self) -> np.ndarray:
        # The peer resets the TRN2u scrambler at the start and carries the E1u
        # differential sign into the first symbol. Correlation locates that
        # exact known sequence before the control-frame slicer is enabled.
        modulator = v92_phase3.Trn2uModulator(self.points, initial_sign=self._initial_sign)
        return np.asarray(modulator.symbols(self.TRN_SEARCH_SYMBOLS), dtype=float)

    def _find_trn2u(self, x: np.ndarray, base: int) -> tuple[np.ndarray, int] | None:
        template = self._search_template
        keep = len(template) - 1
        prefix = self._search_tail
        joined = np.concatenate((prefix, x)) if len(prefix) else x
        if len(joined) < len(template):
            self._search_tail = joined[-keep:].copy()
            return None
        windows = np.lib.stride_tricks.sliding_window_view(joined, len(template))
        if len(prefix):
            windows = windows[-len(x):]
            first_at = base - len(prefix)
        else:
            first_at = base
        if not len(windows):
            self._search_tail = joined[-keep:].copy()
            return None
        # A normalized matched filter is insensitive to line gain. Use RMS to
        # reject silence; an absolute energy threshold would reject short,
        # otherwise valid windows on quieter lines.
        energy = np.einsum("ij,ij->i", windows, windows)
        rms = np.sqrt(energy / len(template))
        denom = np.sqrt(energy * float(np.dot(template, template)))
        score = np.abs(windows @ template) / np.maximum(denom, 1e-12)
        score[rms < 20.0] = 0.0
        best = int(np.argmax(score))
        self._search_tail = joined[-keep:].copy()
        if float(score[best]) < self.TRN_MIN_CORRELATION:
            return None
        at = first_at + best
        offset = len(prefix) - (base - at)
        return joined[offset:].copy(), at

    def _begin_trn_fit(self) -> None:
        import v92_equaliser

        received = self._candidate[:self.TRN_FIT_SYMBOLS].copy()
        sent = v92_phase3.Trn2uModulator(
            self.points, initial_sign=self._initial_sign).symbols(self.TRN_FIT_SYMBOLS)
        sent = np.asarray(sent, dtype=float)
        self.fit_target_rms = float(np.sqrt(np.mean(sent ** 2)))

        def train():
            started = time.perf_counter()
            try:
                fit = (v92_equaliser.train_affine if self._fit_output_bias
                       else v92_equaliser.train)
                self._fit_result.append(fit(
                    received, sent, taps=self.trn_fit_taps,
                    max_delay=self.trn_fit_max_delay))
            except Exception as exc:  # retained for diagnostics; feed() fails closed below
                self._fit_result.append(exc)
            finally:
                self.fit_wall_seconds = time.perf_counter() - started

        self.acquisition_state = "training"
        self.acquisition_attempts += 1
        self.fit_samples = len(received)
        self.fit_wall_seconds = None
        self._fit_result.clear()
        if self._async_fit:
            self._fit_thread = threading.Thread(target=train, name="v92-trn2u-equalizer", daemon=True)
            self._fit_thread.start()
        else:
            train()

    def _retry_acquisition(self, reason: str) -> None:
        """Boundedly return to search after a false TRN2u candidate or failed fit."""
        self.acquisition_error = reason
        self.acquisition_failures.append(reason)
        self._candidate = None
        self._candidate_at = None
        self._fit_result.clear()
        self._fit_thread = None
        self._stream_filter = None
        if self.acquisition_attempts >= self.MAX_ACQUISITION_ATTEMPTS:
            self.acquisition_state = "failed"
            return
        self.acquisition_retries += 1
        self.acquisition_state = "search"
        # Do not immediately lock again on an overlapping view of the same bad
        # candidate. A subsequent training attempt is acquired from fresh input.
        self._search_tail = np.empty(0, dtype=float)

    def _record_bits(self, bits: list[int]) -> list[int]:
        if bits:
            zeros = sum(1 for bit in bits if not bit)
            if zeros and self.first_non_one_bit is None:
                self.first_non_one_bit = self.bits_out + bits.index(0)
            self.zero_bits += zeros
            self.bits_out += len(bits)
        return bits

    def _feed_acquiring(self, x: np.ndarray, base: int) -> list[int]:
        if self.acquisition_state == "search":
            found = self._find_trn2u(x, base)
            if found is None:
                return []
            self._candidate, self._candidate_at = found
            self.trn2u_at = self._candidate_at
            self.acquisition_state = "collecting"
        elif self._candidate is not None and len(x):
            self._candidate = np.concatenate((self._candidate, x))

        if self.acquisition_state == "collecting" and len(self._candidate) >= self.TRN_FIT_SYMBOLS:
            self._begin_trn_fit()
        if self.acquisition_state == "training":
            if self._fit_thread is not None and self._fit_thread.is_alive():
                return []
            if not self._fit_result:
                self._retry_acquisition("equalizer worker returned no result")
                return []
            result = self._fit_result[0]
            if isinstance(result, Exception):
                self._retry_acquisition(f"equalizer training failed: {result}")
                return []
            if not np.isfinite(result[2]) or float(result[2]) > 0.5:
                self._retry_acquisition(f"TRN2u equalizer residual is too large: {result[2]}")
                return []
            weights, delay, residual = result[:3]
            fit_bias = float(result[3]) if len(result) > 3 else 0.0
            # The fitted stream has a delay before its output corresponds to
            # TRN2u symbol zero. Wait for that support, then replay the saved
            # training and any frames that arrived while the fit ran.
            if len(self._candidate) < self.TRN_FIT_SYMBOLS + int(delay):
                return []
            from v92_equaliser import StreamFilter
            self._stream_filter = StreamFilter(weights)
            aligned = self._stream_filter.apply(self._candidate)[int(delay):]
            if self._fit_output_bias:
                aligned = aligned + fit_bias
            decoded = self.coder.decode(aligned)
            training_bits = self.TRN_FIT_SYMBOLS * self.coder.bits
            self.eq_delay, self.eq_residual = int(delay), float(residual)
            self.fit_weights = np.asarray(weights, dtype=float).copy()
            self.fit_output_bias = fit_bias
            self.lu = float(np.sqrt(np.mean(self._candidate[:self.TRN_FIT_SYMBOLS] ** 2)))
            self._decision_lu, self._auto_lu = 1.0, False
            self.start_raw = int(self._candidate_at) + self.TRN_FIT_SYMBOLS
            self.acquisition_state = "locked"
            self.decoded_codewords = max(0, len(aligned) - self.TRN_FIT_SYMBOLS)
            self._candidate = None
            self._search_tail = np.empty(0, dtype=float)
            return self._record_bits(decoded[training_bits:])
        return []

    def feed(self, units) -> list[int]:
        x = np.asarray(units, dtype=float)
        base = self.raw_in
        self.raw_in += len(x)
        if self._acquire_trn2u and self.acquisition_state != "locked":
            return self._feed_acquiring(x, base)
        if self._acquire_trn2u and self._stream_filter is not None:
            aligned = self._stream_filter.apply(x)
            if self._fit_output_bias:
                aligned = aligned + self.fit_output_bias
            bits = self.coder.decode(aligned / self._decision_lu)
            self.decoded_codewords += len(x)
            return self._record_bits(bits)
        if self._auto_lu:
            # Table 28/29 TRN2u has RMS exactly LU. Calibrate in the actual
            # decoder domain (after Side.receive's optional Phase-4 filter),
            # rather than assuming the peer's nominal transmit level or a
            # Phase-3 measurement survives line gain and equalization.
            if self._lu_probe_base is None:
                self._lu_probe_base = base
            self._lu_probe = np.concatenate((self._lu_probe, x))
            if len(self._lu_probe) > self.LU_PROBE_MAX_SAMPLES:
                drop = len(self._lu_probe) - self.LU_PROBE_MAX_SAMPLES
                self._lu_probe = self._lu_probe[drop:]
                self._lu_probe_base += drop
            if len(self._lu_probe) < self.LU_PROBE_SAMPLES:
                return []
            magnitude = np.abs(self._lu_probe)
            peak = float(np.max(magnitude)) if len(magnitude) else 0.0
            if not np.isfinite(peak) or peak < 20.0:
                # The new Phase-4 stream may begin with silence or filter
                # settling. Keep collecting a bounded probe until TRN2u is
                # visible, then calibrate before making symbol decisions.
                return []
            active = self._lu_probe[magnitude > max(5.0, peak * 0.08)]
            if len(active) < 64:
                return []
            measured = float(np.sqrt(np.mean(active * active)))
            if not np.isfinite(measured) or measured <= 0.0:
                return []
            self.lu = measured
            self._decision_lu = measured
            self._auto_lu = False
            x = self._lu_probe
            base = int(self._lu_probe_base)
            self._lu_probe = np.empty(0, dtype=float)
            self._lu_probe_base = None
        if self.start_raw is None:
            # Every real symbol is at least 0.44 LU, so a start is a run of loud ones. A lone sample over the bar is
            # the receive filter's start-up transient and would put the symbol clock (and the scanner) off.
            x = np.concatenate((self._held, x))
            base -= len(self._held)
            if len(x) < self.START_RUN:
                self._held = x
                return []
            loud = np.abs(x) > 0.3 * self.lu
            run = loud[:-(self.START_RUN - 1)].copy()
            for k in range(1, self.START_RUN):
                run &= loud[k:len(loud) - self.START_RUN + 1 + k]
            hit = np.nonzero(run)[0]
            if not len(hit):
                self._held = x[-(self.START_RUN - 1):]
                return []
            self.start_raw = base + int(hit[0])
            x = x[hit[0]:]
        bits = self.coder.decode(x / self.lu)
        self.decoded_codewords += len(x)
        return self._record_bits(bits)

    def raw_of_bit(self, bit_index: int) -> int:
        return self.start_raw + bit_index // self.nb


class _FastUpstreamDecoder:
    """Decode preceding-data-mode PCM-upstream control during V.92 FPE.

    FPE starts this decoder at the first SUVu data-frame boundary after RM/RM′.
    The PCM data receiver already implements the modem's descrambler and
    trellis; putting it in DATA state skips normal B1u qualification because
    FPE's FB1u/B1u sequence occurs after the control exchange.

    ``front`` (a ``_Trn2uFront``): the old data mode used the TRN2u-fit receiver, so its line samples are G x LU x U
    and only that mode's fitted FIR puts them back in the CPd point domain. The fit is exposed as ``fit_weights`` etc.
    so the new mode's B1u receiver starts from it at E (FPE sends no TRN2u)."""

    # Behind a ``front`` the old mode used the TRN2u-fit receiver, whose adaptive constellation is spaced for the soft
    # Viterbi (gap 4 sigma): hard decisions there err on ~5 % of symbols, enough to lose every CPu (rig, 26,667). Its
    # lag is kept short, and E is not rolled back: FB1u is checked by a fresh receiver from E (``fb1_receiver``).
    CONTROL_TRACEBACK_GROUPS = 16

    def __init__(self, profile: pu.PcmUpstreamProfile, law: str, front=None):
        self.profile, self.law = profile, law
        self.receiver = (self._soft_receiver() if front is not None
                         else purx.PcmUpstreamStreamReceiver(profile, law, start_skip=0))
        self.receiver.state = self.receiver.DATA
        self.raw_in = 0
        self.start_raw = 0
        self.bits_out = 0
        self.front = None
        if front is not None:
            self.reseed(front)
        # EL_MODEM_V92_FPE_BIT_TRACE=1: keep the first BIT_TRACE decoded control bits (diagnostics "bit_trace").
        self.bit_trace = [] if os.environ.get("EL_MODEM_V92_FPE_BIT_TRACE") == "1" else None

    def reseed(self, front) -> None:
        """Use ``front`` (before any input): the old mode's FIR as it stands when the new mode's input starts."""
        if self.raw_in:
            raise RuntimeError("cannot change the fast upstream FIR after decoding starts")
        self.front = front
        self.fit_weights = front.filter.w.copy()
        self.eq_delay = front.delay
        self.fit_target_rms = front.fit_target_rms
        self.eq_residual = front.fit_residual
        self.fit_output_bias = front.bias

    def set_initial_sign(self, initial_sign: int) -> None:
        """Seed the differential decoder before any upstream samples arrive."""
        if self.raw_in or self.bits_out:
            raise RuntimeError("cannot change the fast upstream sign after decoding starts")
        self.receiver._rx._sign = int(initial_sign) & 1

    def _soft_receiver(self):
        """A soft-decision receiver in DATA state from an unknown trellis state (all metrics equal)."""
        rx = purx.PcmUpstreamStreamReceiver(self.profile, self.law, start_skip=0, decoder_mode="viterbi32",
                                            viterbi_traceback_groups=self.CONTROL_TRACEBACK_GROUPS)
        rx.state = rx.DATA
        rx._soft.metrics[:] = 0.0
        rx.frame_states = []
        return rx

    def mark_e(self, e_end_bit: int) -> None:
        """E ends at decoder bit ``e_end_bit`` (a frame boundary): keep the differential sign and descrambler there."""
        frame = int(e_end_bit) // self.profile.bits_per_frame
        self.e_state = next(((sign, reg) for frames, sign, reg in getattr(self.receiver, "frame_states", None) or ()
                             if frames == frame), None)

    def fb1_receiver(self):
        """A soft receiver for FB1u from E's end, with the control decoder's sign and descrambler there. A wrong
        sign inverts the first frame, which the descrambler then spreads 23 bits on."""
        rx = self._soft_receiver()
        state = getattr(self, "e_state", None)
        if state is not None:
            rx._rx._sign, rx._reg = state[0], list(state[1])
        return rx

    def checkpoint(self) -> tuple | None:
        """Small rollback point so the Phase 4 scanner can cut exactly at E (none behind a front: see above)."""
        if self.front is not None:
            return None
        rx = self.receiver
        frame_rx = rx._rx
        return (self.raw_in, self.bits_out, rx.state, rx.frames, list(rx._pending),
                list(rx._reg), rx.b1_errors, rx.lost_frames, rx.consecutive_lost_frames,
                rx.max_consecutive_lost_frames, list(rx._data_bits), rx.failure,
                frame_rx._sign, frame_rx.conv.d.copy(), frame_rx.trellis_violations,
                len(rx.loss_trace))

    def restore(self, checkpoint: tuple) -> None:
        (self.raw_in, self.bits_out, state, frames, pending, reg, b1_errors, lost_frames,
         consecutive_lost_frames, max_consecutive_lost_frames, data_bits, failure,
         sign, conv_state, trellis_violations, loss_trace_len) = checkpoint
        rx = self.receiver
        rx.state, rx.frames, rx._pending = state, frames, pending
        rx._reg, rx.b1_errors = reg, b1_errors
        rx.lost_frames = lost_frames
        rx.consecutive_lost_frames = consecutive_lost_frames
        rx.max_consecutive_lost_frames = max_consecutive_lost_frames
        rx._data_bits, rx.failure = data_bits, failure
        rx._rx._sign = sign
        rx._rx.conv.d = conv_state
        rx._rx.trellis_violations = trellis_violations
        del rx.loss_trace[loss_trace_len:]

    def feed(self, units) -> list[int]:
        x = np.asarray(units, dtype=float).reshape(-1)
        if self.front is not None:
            self.raw_in += len(x)
            self.receiver.feed_samples(self.front.points(x))
        else:
            self.raw_in += len(x)
            codes = (v34_g711.encode(np.rint(x).astype(np.int32), self.law).astype(np.uint8) & 0xFF)
            self.receiver.feed(codes.tobytes())
        bits = self.receiver.take_bits()
        self.bits_out += len(bits)
        if self.bit_trace is not None and len(self.bit_trace) < self.BIT_TRACE:
            self.bit_trace.extend(bits[:self.BIT_TRACE - len(self.bit_trace)])
        return bits

    BIT_TRACE = 48000

    def raw_of_bit(self, bit_index: int) -> int:
        return self.start_raw + (int(bit_index) // self.profile.bits_per_frame) * pu.INTERVALS

class _Trn2uFront:
    """A TRN2u-fit data mode's receive FIR, continued for FPE: raw line samples to that mode's CPd point domain.

    Output n of the FIR is the symbol that arrived at raw n - ``delay``; the first ``drop`` outputs are discarded so a
    stream fed from raw sample s starts at the symbol of s. ``follow``: copy the weights of that live (NLMS) filter
    before each chunk; the passive RM detector uses it, since the data receiver keeps adapting while it listens."""

    def __init__(self, weights, history, *, delay: int, bias: float, gain: float,
                 fit_target_rms: float, fit_residual: float, drop: int | None = None, follow=None):
        from v92_equaliser import StreamFilter
        self.filter = StreamFilter(np.array(weights, dtype=float))
        history = np.asarray(history, dtype=float).reshape(-1)[-len(self.filter._history):]
        if len(history):
            self.filter._history[-len(history):] = history
        self.delay, self.bias, self.gain = int(delay), float(bias), float(gain)
        self.fit_target_rms, self.fit_residual = float(fit_target_rms), float(fit_residual)
        self.drop = self.delay if drop is None else int(drop)
        self.follow = follow

    def points(self, x) -> np.ndarray:
        if self.follow is not None:
            self.filter.w = np.array(self.follow.w, dtype=float)
        y = self.filter.apply(x)
        drop = min(self.drop, len(y))
        self.drop -= drop
        return (y[drop:] + self.bias) * self.gain

    def checkpoint(self) -> tuple:
        return self.filter._history.copy(), self.drop

    def restore(self, checkpoint: tuple) -> None:
        self.filter._history, self.drop = checkpoint[0].copy(), checkpoint[1]


def trn2u_front(data_rx, history=None, *, drop: int | None = None, follow: bool = False) -> _Trn2uFront | None:
    """The ``_Trn2uFront`` of a data receiver's installed TRN2u FIR, or None. ``history``: the raw samples just before
    the first one it will be fed (default: the live filter's own history)."""
    rx = getattr(data_rx, "new_receiver", data_rx)
    live = getattr(rx, "_trn2u_rx_filter", None)
    if live is None:
        return None
    diagnostic = rx.rx_filter_diagnostic
    return _Trn2uFront(live.w, live._history if history is None else history, delay=rx._trn2u_rx_delay,
                       bias=rx._trn2u_rx_bias, gain=rx._trn2u_rx_gain,
                       fit_target_rms=diagnostic["training_target_rms"],
                       fit_residual=diagnostic["trn2u_fit_residual_rms"], drop=drop,
                       follow=live if follow else None)


class _FpeUpstreamDataRx:
    """Consume old-mode FB1u, then qualify new-mode B1u after fast exchange."""

    def __init__(self, old_decoder: _FastUpstreamDecoder, new_receiver: _UpstreamDataRx):
        self.old_decoder = old_decoder
        self.new_receiver = new_receiver
        self.old_frames_remaining = B1_FRAMES
        self.fb1_errors = 0
        self.fb1_bits = 0
        self.fb1_scrambler_transient_errors = 0
        self._fb1_rx = None                          # TRN2u-fit FPE: the soft receiver that checks FB1u
        self._fb1_samples = 0
        self.fb1_error_bits: list[int] = []         # FB1u bit indices in error (the first 64), for diagnostics

    def __getattr__(self, name):
        # Everything else (TRN2u filter setup, diagnostics) is the new mode's receiver.
        if name == "new_receiver":
            raise AttributeError(name)
        return getattr(self.new_receiver, name)

    @property
    def rx(self):
        return self.new_receiver.rx

    @property
    def profile(self):
        return self.new_receiver.profile

    @property
    def cpd_gain(self):
        return self.new_receiver.cpd_gain

    @cpd_gain.setter
    def cpd_gain(self, value):
        self.new_receiver.cpd_gain = value

    # V.92 9.9.2 has the analogue modem zero its scrambler before SUVu, and 8.7.1 before B1u. SmartLink zeroes it at
    # FB1u instead of at B1u: the self-synchronising descrambler then errs in the first 23 bits of FB1u and of B1u and
    # nowhere else (rig, 2026-11-18: 12 errors at FB1u bits 1..22; 13 at B1u). Those are counted apart (FB1u here,
    # B1u in the receiver); both must be clean after them.
    SCRAMBLER_SPAN = 23

    def _count_fb1(self, bits) -> None:
        if bits and not self.fb1_bits:
            self.new_receiver.rx.b1_scrambler_transient_bits = self.SCRAMBLER_SPAN
        if bits:
            errors = [self.fb1_bits + i for i, bit in enumerate(bits) if bit != 1]
            self.fb1_error_bits.extend(errors[:64 - len(self.fb1_error_bits)])
            self.fb1_bits += len(bits)
            transient = sum(index < self.SCRAMBLER_SPAN for index in errors)
            self.fb1_scrambler_transient_errors += transient
            self.fb1_errors += len(errors) - transient

    def _feed_trn2u(self, x) -> None:
        """One FIR from E on: the old mode's FB1u at its CPd gain, then the new mode's B1u and data at the new one."""
        new = self.new_receiver
        if new._trn2u_rx_filter is None:
            new.set_filter_setup_error("filter was not installed at the E boundary")
            return
        filtered, first = new._trn2u_filtered(x)
        if self.old_frames_remaining:
            # FB1u: the first 48 frames from E through this FIR, at the old mode's CPd gain, to a fresh soft receiver
            # flushed at FB1u's end; the rest is the new mode's B1u.
            if self._fb1_rx is None:
                self._fb1_rx = self.old_decoder.fb1_receiver()
                self._fb1_samples = B1_FRAMES * pu.INTERVALS
            front = self.old_decoder.front
            old, filtered = filtered[:self._fb1_samples], filtered[self._fb1_samples:]
            first += len(old)
            self._fb1_samples -= len(old)
            self._fb1_rx.feed_samples((old + front.bias) * front.gain)
            if not self._fb1_samples:
                self._fb1_rx.flush_soft()
            self._count_fb1(self._fb1_rx.take_bits())
            if self._fb1_samples:
                return
            self.old_frames_remaining = 0
            if self.fb1_errors or self.fb1_bits != B1_FRAMES * self.old_decoder.profile.bits_per_frame:
                new.rx._fail(f"FB1u failed: {self.fb1_errors} bit errors at {self.fb1_error_bits}, "
                             f"{self.fb1_bits} bits")
                return
        new._trn2u_decode(filtered, first)

    def feed(self, units) -> None:
        x = np.asarray(units, dtype=float).reshape(-1)
        if self.new_receiver.rx_filter_mode == "trn2u-fit":
            self._feed_trn2u(x)
            return
        if self.old_frames_remaining:
            # The preceding-mode decoder may have a partial frame buffered at
            # the E boundary. Stop on the exact 48-frame FB1u boundary; do not
            # let any new-mode B1u samples enter the old trellis/descrambler.
            need = max(0, self.old_frames_remaining * pu.INTERVALS
                       - len(self.old_decoder.receiver._pending))
            old_part = x[:need]
            bits = self.old_decoder.feed(old_part)
            consumed = len(old_part)
            if bits:
                self._count_fb1(bits)
                self.old_frames_remaining -= len(bits) // self.old_decoder.profile.bits_per_frame
                x = x[consumed:]
            if self.old_frames_remaining:
                return
            if self.fb1_errors:
                self.new_receiver.rx._fail(f"FB1u failed: {self.fb1_errors} bit errors at {self.fb1_error_bits}")
                return
        if len(x):
            self.new_receiver.feed(x)

    @property
    def alignment(self):
        return self.new_receiver.alignment

    @property
    def gain(self):
        return self.new_receiver.gain

    @property
    def gain_calibration(self):
        return self.new_receiver.gain_calibration

    def take_bits(self) -> list[int]:
        return self.new_receiver.take_bits()

    @property
    def qualified(self) -> bool:
        return self.old_frames_remaining == 0 and self.new_receiver.qualified

    @property
    def b1_errors(self) -> int:
        return self.new_receiver.b1_errors

    def payload(self) -> bytes:
        return self.new_receiver.payload()


class _DownstreamDecoder:
    """Peer's downstream codewords to bits through the V.90 receiver, after the first non-idle one."""

    def __init__(self, profile: dict, law: str):
        self.profile, self.law = profile, law
        self.receiver = rx4.V90DigitalPhase4Receiver(profile, law)
        enc = dp4.V90DigitalPhase4Encoder(profile, law)
        self.fb = enc.s + enc.k
        self.silence = int(v34_g711.encode(np.zeros(1), law)[0]) & 0xFF
        self._silence_codes = {self.silence, self.silence ^ 0x80}
        self.erased_frames = 0
        self.start_raw: int | None = None
        self.raw_in = 0
        self.pending = bytearray()
        self.broken = False
        self.broken_reason: str | None = None
        self.broken_at_raw: int | None = None
        self.decoded_codewords = 0
        self.first_non_one_bit: int | None = None
        self.zero_bits = 0
        self.bits_out = 0

    def restart(self, raw_origin: int) -> None:
        """Reacquire the mapper stream after raw R-t and its preceding silence."""
        self.receiver = rx4.V90DigitalPhase4Receiver(self.profile, self.law)
        self.start_raw = None
        self.raw_in = int(raw_origin)
        self.pending.clear()
        self.broken = False
        self.broken_reason = None
        self.broken_at_raw = None
        self.decoded_codewords = 0
        self.first_non_one_bit = None
        self.zero_bits = 0
        self.bits_out = 0

    def feed(self, units) -> list[int]:
        data = bytes(units)
        base = self.raw_in
        self.raw_in += len(data)
        if self.start_raw is None:
            skip = next((i for i, b in enumerate(data) if b != self.silence), None)
            if skip is None:
                return []
            self.start_raw = base + skip
            data = data[skip:]
        self.pending.extend(data)
        bits: list[int] = []
        while len(self.pending) >= 6 and not self.broken:
            frame = bytes(self.pending[:6])
            try:
                decoded = self.receiver.decode_codewords(frame)
            except ValueError as exc:
                if any(code in self._silence_codes for code in frame):
                    # An RTP gap concealed as silence: Ucode 0 is in no CPt or CP constellation. Keep the bit
                    # cadence with a frame of ones (which ends an E zero run, as the gap does) and keep decoding;
                    # the descrambler resynchronises within 23 bits. Marking the decoder broken here lost the
                    # rest of Phase 4 to a 60 ms dropout.
                    self.receiver.discard_frame()
                    decoded = [1] * self.fb
                    self.erased_frames += 1
                    bits.extend(decoded)
                    self.decoded_codewords += 6
                    del self.pending[:6]
                    continue
                # Units that are not in the training mapper's alphabet. Normally these are the
                # first frames of the peer's data mode, in the chunk that holds the end of its E;
                # the caller re-routes them. Stop decoding rather than guess.
                self.broken = True
                self.broken_reason = str(exc)
                self.broken_at_raw = int(self.start_raw or 0) + self.decoded_codewords
                break
            if decoded:
                if self.first_non_one_bit is None and 0 in decoded:
                    self.first_non_one_bit = self.bits_out + decoded.index(0)
                self.zero_bits += decoded.count(0)
                bits.extend(decoded)
            self.decoded_codewords += 6
            del self.pending[:6]
        self.bits_out += len(bits)
        return bits

    def raw_of_bit(self, bit_index: int) -> int:
        return self.start_raw + bit_index // self.fb * 6


class _DownstreamDataRx:
    """B1d then payload from the V.90 receiver with a fresh mapper (analogue modem's side).

    A frame that is not codewords of the data mapper is an erasure (an RTP packet concealed as silence) until
    ``MAX_ERASED_FRAMES`` of them run together; then the peer has stopped sending data. Erased frames become ones,
    HDLC idle, once the stream decodes again, which keeps the bit cadence; LAPM retransmits the damaged frame.

    Astound's V.92 server sends one more frame of zeros in the data constellations after Ed's TRN2d zeros, then a
    spec B1d (V.90 8.6.1: scrambler and differential encoder zeroed) that decodes cleanly only once the receiver has
    passed through that frame (live calls 2 and 5, 2026-11-19). So a first frame that is not ones is held, and
    dropped as ``lead_frames`` when the frame after it is all ones; B1d is then the 48 frames from there.
    """

    MAX_ERASED_FRAMES = 1867           # 6-symbol frames: 1.4 s, S10's default carrier-loss time

    def __init__(self, profile: dict, law: str):
        self.rx = rx4.V90DigitalPhase4Receiver(profile, law)
        enc = dp4.V90DigitalPhase4Encoder(profile, law)
        self.fb = enc.s + enc.k
        self.bits: list[int] = []
        self.pending = bytearray()
        self.ended = False
        self._taken = 0
        self._erased = 0
        self.erased_frames = 0
        self._lead = "first"               # first -> held (first frame not ones) -> done
        self._held: list[int] | None = None
        self.lead_frames = 0

    def _flush_held(self) -> None:
        if self._held is not None:
            self.bits.extend(self._held)
            self._held = None
        self._lead = "done"

    def _append(self, bits: list[int]) -> None:
        if self._lead == "first" and not all(bits):
            self._lead, self._held = "held", list(bits)
            return
        if self._lead == "held" and all(bits):
            self._held, self.lead_frames = None, 1
        self._flush_held()
        self.bits.extend(bits)

    def feed(self, units) -> None:
        if self.ended:
            return
        self.pending.extend(bytes(units))
        while len(self.pending) >= 6:
            try:
                bits = self.rx.decode_codewords(bytes(self.pending[:6]))
            except ValueError:
                self.rx.discard_frame()
                del self.pending[:6]
                self._erased += 1
                if self._erased > self.MAX_ERASED_FRAMES:
                    self.ended = True
                    return
                continue
            if self._erased:
                self._flush_held()
                self.bits.extend([1] * (self._erased * self.fb))
                self.erased_frames += self._erased
                self._erased = 0
            self._append(bits)
            del self.pending[:6]

    def take_bits(self) -> list[int]:
        """Data bits (after B1d) not yet taken."""
        start = B1_FRAMES * self.fb + self._taken
        new = self.bits[start:]
        self._taken += len(new)
        return list(new)

    @property
    def b1_errors(self) -> int:
        return sum(1 for b in self.bits[:B1_FRAMES * self.fb] if b != 1)

    @property
    def qualified(self) -> bool:
        return len(self.bits) >= B1_FRAMES * self.fb and self.b1_errors == 0

    def payload(self) -> bytes:
        data = self.bits[B1_FRAMES * self.fb:]
        return bytes(sum(data[8 * j + i] << i for i in range(8)) for j in range(len(data) // 8))


# Adaptive CPd constellation (cpd_constellation_mode="adaptive"). The rms cap is that of the expanded10 set the
# SmartLink peer was validated with at CPd gain4 0x0300; the slicer noise is modelled as the TRN2u fit's relative
# residual times that rms (the local rig's error grows with the signal: resampler ISI). A gap of 4 sigma plus the
# local G.711 step reproduces expanded10's density on that rig; the 32-state trellis and LAPM carry the rest.
ADAPTIVE_MAX_RMS = 1328.0
# 3 sigma: 20/20 SmartLink rig calls clean at 26,667-29,333 (2026-11-18); 2.5 lost the upstream once.
ADAPTIVE_GAP_SIGMAS = float(os.environ.get("EL_MODEM_V92_CPD_ADAPTIVE_GAP", "3.0"))   # lab override
ADAPTIVE_PROVISIONAL_RESIDUAL = 0.026


def _trn2u_nlms_mu() -> float:
    """``EL_MODEM_V92_TRN2U_NLMS_MU``: step size of the decision-directed TRN2u FIR tracking (default 0.1, the
    middle of the 0.03-0.3 range that held the arm-10c replay at zero errors); 0 keeps the FIR fixed."""
    raw = os.environ.get("EL_MODEM_V92_TRN2U_NLMS_MU")
    try:
        mu = 0.1 if raw is None else float(raw)
    except ValueError as exc:
        raise ValueError("EL_MODEM_V92_TRN2U_NLMS_MU must be a number") from exc
    if not 0.0 <= mu <= 1.0:
        raise ValueError("EL_MODEM_V92_TRN2U_NLMS_MU must be in 0..1")
    return mu


class _UpstreamDataRx:
    """B1u qualification and data bits from the PCM-upstream receiver. ``keep`` retains every bit so ``payload()``
    can return the bytes (tests); a live session sets it false and only drains ``take_bits()``.

    ``align_search`` > 0: the data-frame boundary is only known to within that many samples (equaliser delay, channel
    cursor), so the first frames are held and decoded at each candidate alignment; the one with the fewest B1u errors
    and trellis violations is kept. These paths also fit a scalar input gain on
    26 B1 frames and require a waveform-error improvement on the unused 22;
    normal zero-error B1 qualification remains required afterward."""

    SEARCH_FRAMES = 26

    def __init__(self, profile: pu.PcmUpstreamProfile, law: str, keep: bool = True,
                 align_search: int = 0, max_lost_frames: int = 3,
                 max_data_lost_frames: int = 32,
                 calibrate_gain: bool | None = None, rx_filter_mode: str = "off",
                 fitted_fractional_delay_samples: float | None = None,
                 decoder_mode: str = "hard", cpd_gain: float = 1.0, nlms_mu: float = 0.0):
        if rx_filter_mode not in ("off", "trn2u-fit"):
            raise ValueError("rx_filter_mode must be off or trn2u-fit")
        if decoder_mode not in ("hard", "viterbi32"):
            raise ValueError("decoder_mode must be hard or viterbi32")
        if not np.isfinite(cpd_gain) or cpd_gain <= 0:
            raise ValueError("cpd_gain must be finite and positive")
        if not np.isfinite(nlms_mu) or nlms_mu < 0:
            raise ValueError("nlms_mu must be finite and not negative")
        self.profile, self.law = profile, law
        self.decoder_mode = decoder_mode
        self.cpd_gain = float(cpd_gain)
        self.rx = purx.PcmUpstreamStreamReceiver(
            profile, law, max_lost_frames=max_lost_frames,
            max_data_lost_frames=max_data_lost_frames, decoder_mode=decoder_mode)
        self.rx_filter_mode = rx_filter_mode
        self._decoder_timing_enabled = os.environ.get("EL_MODEM_V92_RX_TIMING") == "1"
        self._decoder_chunk_ns = []
        self._decoder_thread_ids = set()
        self.fitted_fractional_delay_samples = fitted_fractional_delay_samples
        self._trn2u_rx_filter = None
        self._trn2u_rx_drop = 0
        self._trn2u_rx_delay = 0
        self._trn2u_rx_gain = 1.0
        self._trn2u_rx_bias = 0.0
        # Decision-directed tracking of the TRN2u FIR (0 = fixed): each input sample's decision is the nearest signed
        # level of its interval; samples further than NLMS_GATE of the local spacing from it are not used.
        self.nlms_mu = float(nlms_mu) if rx_filter_mode == "trn2u-fit" else 0.0
        self._nlms_symbols = 0
        self._nlms_levels = []
        self._nlms_stats = None
        self._nlms_window = [0.0, 0]             # squared error and samples since the last window_sigma
        if self.nlms_mu:
            for interval in range(pu.INTERVALS):
                count = len(profile.constellations[profile.interval_set[interval]])
                self._nlms_levels.append(np.sort(np.asarray(
                    [profile.level(interval, eta) for eta in range(-count, count)], dtype=float)))
        self.rx_filter_diagnostic = {
            "mode": rx_filter_mode,
            "state": "off" if rx_filter_mode == "off" else "awaiting-e-boundary",
            "fitted_fractional_delay_samples": fitted_fractional_delay_samples,
        }
        self.keep = keep
        self._max_lost_frames = max_lost_frames
        self._max_data_lost_frames = max_data_lost_frames
        self._search = int(align_search)
        self._calibrate_gain = calibrate_gain
        self._held = bytearray()
        self._scores: list[tuple[int, int]] = []
        self.alignment: int | None = 0 if not align_search else None
        self._qualified = False
        self._all_bits: list[int] = []
        self._new_bits: list[int] = []
        self.gain = 1.0
        self.gain_calibration = None
        self._gain_ready = not (bool(align_search) if calibrate_gain is None else calibrate_gain)
        self._gain_frames = 0
        self._gain_reference = None
        self._gain_classes = []
        self._fast_candidates = None
        self._fast_index = 0
        self._gain_tx = self._gain_scrambler = None
        if not self._gain_ready:
            # u(n) selection depends on the precoder, but not on the following
            # transmit prefilter. Omit that long convolution for this reference.
            reference_profile = dataclasses.replace(
                profile, prefilter_z2=(1.0,), prefilter_p2=())
            self._gain_tx = pu.PcmUpstreamTransmitter(reference_profile, record=True)
            self._gain_scrambler = v34_tx.Scrambler("answer")

    def install_trn2u_filter(self, stream_filter, *, delay: int, gain: float,
                              target_rms: float, training_target_rms: float,
                              fit_residual: float, fractional_delay_samples: float | None,
                              bias: float = 0.0) -> None:
        """Install the receive-only TRN2u FIR probe at the E/B1u boundary."""
        if self.rx_filter_mode != "trn2u-fit":
            raise RuntimeError("TRN2u receive filter is not enabled")
        if self.rx.frames or self._held:
            raise RuntimeError("cannot install the TRN2u filter after PCM upstream decoding began")
        if (delay < 0 or not np.isfinite(gain) or gain <= 0
                or not np.isfinite(bias)):
            raise ValueError("invalid TRN2u delay, gain or bias")
        if self.nlms_mu:
            from v92_equaliser import AdaptiveStreamFilter
            adaptive = AdaptiveStreamFilter(stream_filter.w, self.nlms_mu)
            adaptive._history = np.array(stream_filter._history, dtype=float)
            stream_filter = adaptive
            self._nlms_stats = {"mu": self.nlms_mu, "gate": self.NLMS_GATE, "updates": 0, "samples": 0, "gated": 0,
                                "window_sigma": None, "tap_change": 0.0}
        self._trn2u_rx_filter = stream_filter
        self._trn2u_rx_drop = self._trn2u_rx_delay = int(delay)
        self._trn2u_rx_gain = float(gain)
        self._trn2u_rx_bias = float(bias)
        self.rx_filter_diagnostic = {
            "mode": self.rx_filter_mode, "state": "active", "coefficients": stream_filter.w.tolist(),
            "delay_samples": int(delay), "gain_to_cpd_point_domain": float(gain),
            "bias_correction_le": float(bias),
            "target_point_rms": float(target_rms), "training_target_rms": float(training_target_rms),
            "trn2u_fit_residual_rms": float(fit_residual),
            "fitted_fractional_delay_samples": fractional_delay_samples,
            "fractional_delay_applied": False,
            "filter_history_samples": len(stream_filter._history),
            "nlms": self._nlms_stats,
        }

    NLMS_GATE = 0.35
    NLMS_WINDOW = 8000

    def set_profile(self, profile: pu.PcmUpstreamProfile) -> None:
        """Use another upstream constellation (the CPd actually sent); only before any PCM-upstream input."""
        if self.rx.frames or self._held or self._trn2u_rx_filter is not None:
            raise RuntimeError("cannot change the upstream profile after PCM upstream decoding began")
        self.profile = profile
        self.rx = purx.PcmUpstreamStreamReceiver(
            profile, self.law, max_lost_frames=self._max_lost_frames,
            max_data_lost_frames=self._max_data_lost_frames, decoder_mode=self.decoder_mode)
        if self.nlms_mu:
            self._nlms_levels = []
            for interval in range(pu.INTERVALS):
                count = len(profile.constellations[profile.interval_set[interval]])
                self._nlms_levels.append(np.sort(np.asarray(
                    [profile.level(interval, eta) for eta in range(-count, count)], dtype=float)))

    def _nlms_step(self, filtered, first: int) -> None:
        """One block NLMS step on the TRN2u FIR from this chunk's tentative decisions (see ``AdaptiveStreamFilter``)."""
        gain = self._trn2u_rx_gain
        points = (filtered + self._trn2u_rx_bias) * gain
        intervals = (self._nlms_symbols + np.arange(len(points))) % pu.INTERVALS
        self._nlms_symbols += len(points)
        decided = np.empty_like(points)
        spacing = np.empty_like(points)
        for interval in range(pu.INTERVALS):
            mask = intervals == interval
            if not mask.any():
                continue
            levels = self._nlms_levels[interval]
            upper = np.clip(np.searchsorted(levels, points[mask]), 1, len(levels) - 1)
            low, high = levels[upper - 1], levels[upper]
            decided[mask] = np.where(high - points[mask] < points[mask] - low, high, low)
            spacing[mask] = high - low
        error = decided - points
        usable = np.abs(error) < self.NLMS_GATE * spacing
        stats = self._nlms_stats
        if self._trn2u_rx_filter.adapt(first, error, usable, gain):
            stats["updates"] += 1
            stats["tap_change"] = round(float(np.linalg.norm(self._trn2u_rx_filter.w - self._trn2u_rx_filter.initial)
                                              / np.linalg.norm(self._trn2u_rx_filter.initial)), 4)
        stats["samples"] += len(points)
        stats["gated"] += int(len(points) - np.count_nonzero(usable))
        acc = self._nlms_window
        acc[0] += float(np.sum(error * error))
        acc[1] += len(error)
        if acc[1] >= self.NLMS_WINDOW:
            stats["window_sigma"] = round(float(np.sqrt(acc[0] / acc[1])), 2)
            self._nlms_window = [0.0, 0]

    def set_filter_setup_error(self, reason: str) -> None:
        self.rx_filter_diagnostic = {
            "mode": self.rx_filter_mode, "state": "failed", "error": str(reason),
            "fitted_fractional_delay_samples": self.fitted_fractional_delay_samples,
        }
        self.rx._fail("TRN2u receive filter setup failed: " + str(reason))

    def _collect_received_bits(self) -> None:
        if self.rx.state == self.rx.DATA:
            self._qualified = True
        if self._qualified:
            bits = self.rx.take_bits()
            if bits:
                self._new_bits.extend(bits)
                if self.keep:
                    self._all_bits.extend(bits)

    def _advance_gain_reference(self) -> None:
        if self._gain_ready:
            return
        # Keep reference generation out of the constructor/TX callback and
        # bound the work while the receive alignment search is in progress.
        for _ in range(min(4, B1_FRAMES - self._gain_frames)):
            bits = [self._gain_scrambler.step(1) for _ in range(self.profile.bits_per_frame)]
            classes, _ = pu.modulus_encode(bits, self.profile.moduli,
                                           self.profile.total_modulus, self._gain_tx._sign)
            self._gain_classes.extend(classes)
            self._gain_tx.encode_frame(bits)
            self._gain_frames += 1
        if self._gain_frames == B1_FRAMES and self._gain_reference is None:
            self._gain_reference = np.asarray(self._gain_tx.levels, dtype=float)
            self._gain_reference.setflags(write=False)

    def _gain_model(self, observed):
        n = B1_FRAMES * pu.INTERVALS
        target = self._gain_reference
        split = self.SEARCH_FRAMES * pu.INTERVALS
        gain = float(observed[:split] @ target[:split] / (target[:split] @ target[:split]))
        before = (observed - target) ** 2
        valid = np.isfinite(gain) and gain > 0.0
        after = (observed / gain - target) ** 2 if valid else np.full(n, np.inf)
        fit_before, fit_after = float(before[:split].mean()), float(after[:split].mean())
        val_before, val_after = float(before[split:].mean()), float(after[split:].mean())
        accepted = valid and fit_after < 0.99 * fit_before and val_after < 0.99 * val_before
        applied = gain if accepted else 1.0
        calibration = {"candidate_gain": gain if np.isfinite(gain) else None,
                       "candidate_valid": bool(valid), "accepted": bool(accepted),
                       "fit_samples": split, "validation_samples": n - split,
                       "mse_units": "linear_pcm_squared",
                       "fit_before": fit_before, "fit_after": fit_after if valid else None,
                       "validation_before": val_before,
                       "validation_after": val_after if valid else None}
        return applied, calibration

    def _candidate_gain(self, octets, skip):
        n = B1_FRAMES * pu.INTERVALS
        if self._gain_reference is None or len(octets) < skip + n:
            return 1.0
        observed = v34_g711.decode(np.frombuffer(
            octets[skip:skip + n], dtype=np.uint8), self.law).astype(float)
        return self._gain_model(observed)[0]

    def _zero_candidates(self, octets):
        """Necessary label test for zero-error B1, before full trellis checks.

        Zero descrambled errors imply the known scrambled bits and hence the
        known modulus labels. Test those labels in parallel for every offset;
        parity/trellis and actual decoded bits are still checked by _score.
        """
        count = self.SEARCH_FRAMES * pu.INTERVALS
        raw = v34_g711.decode(np.frombuffer(
            octets[:self._search + count - 1], dtype=np.uint8), self.law).astype(float)
        windows = np.lib.stride_tricks.sliding_window_view(raw, count)[:self._search]
        gains = np.asarray([self._candidate_gain(octets, skip) for skip in range(self._search)])
        values = windows / gains[:, None]
        expected = np.asarray(self._gain_classes[:count]).reshape(-1, pu.INTERVALS)
        possible = np.ones(self._search, dtype=bool)
        for i in range(pu.INTERVALS):
            levels = self.rx._rx._signed[self.profile.interval_set[i]]
            samples = values[:, i::pu.INTERVALS]
            index = np.searchsorted(levels, samples)
            right = np.clip(index, 0, len(levels) - 1)
            left = np.clip(index - 1, 0, len(levels) - 1)
            chosen = np.where(samples - levels[left] <= levels[right] - samples, left, right)
            eta = chosen - len(levels) // 2
            labels = (eta if i % 4 < 3 else (eta - (eta & 1)) // 2) % self.profile.moduli[i]
            possible &= np.all(labels == expected[:, i], axis=1)
        return np.flatnonzero(possible).tolist()

    def _fit_b1_gain(self) -> None:
        n = B1_FRAMES * pu.INTERVALS
        observed = v34_g711.decode(np.frombuffer(
            bytes(self._held[self.alignment:self.alignment + n]), dtype=np.uint8), self.law).astype(float)
        self.gain, self.gain_calibration = self._gain_model(observed)
        self.gain_calibration["scored_alignments"] = self._fast_index + len(self._scores)
        self.rx = purx.PcmUpstreamStreamReceiver(
            self.profile, self.law, start_skip=self.alignment,
            gain=self.gain, max_lost_frames=self._max_lost_frames,
            max_data_lost_frames=self._max_data_lost_frames)
        self._gain_ready = True
        self._gain_tx = self._gain_scrambler = None

    # Six 26-frame trials cost about 12 ms before the rest of receive work.
    # Spread the same candidate set over more callbacks to preserve RTP headroom.
    SCORES_PER_CALL = 3

    def _trn2u_filtered(self, units):
        """The TRN2u FIR output for ``units`` past the fit delay, and its index in this call's output."""
        filtered = self._trn2u_rx_filter.apply(np.asarray(units, dtype=float).reshape(-1))
        drop = min(self._trn2u_rx_drop, len(filtered))
        self._trn2u_rx_drop -= drop
        return filtered[drop:], drop

    def _trn2u_decode(self, filtered, first: int) -> None:
        """Track (NLMS) and decode FIR output ``filtered``, output ``first`` onwards of the latest FIR call."""
        if len(filtered) and self.nlms_mu and self.rx.state != self.rx.FAILED:
            self._nlms_step(filtered, first)
        if len(filtered):
            started_ns = time.perf_counter_ns() if self._decoder_timing_enabled else 0
            self.rx.feed_samples((filtered + self._trn2u_rx_bias) * self._trn2u_rx_gain)
            if self._decoder_timing_enabled:
                self._decoder_chunk_ns.append(time.perf_counter_ns() - started_ns)
                if len(self._decoder_chunk_ns) > 20000:
                    del self._decoder_chunk_ns[:10000]
                self._decoder_thread_ids.add(threading.get_ident())
        self._collect_received_bits()

    def _score(self, octets: bytes, skip: int) -> int:
        trial = purx.PcmUpstreamStreamReceiver(self.profile, self.law, start_skip=skip,
                                               gain=self._candidate_gain(octets, skip), b1_error_limit=10 ** 9,
                                               b1_violation_limit=10 ** 9, max_lost_frames=10 ** 9)
        trial.feed(octets[:skip + self.SEARCH_FRAMES * pu.INTERVALS])
        return trial.b1_errors + 8 * trial.trellis_violations + 50 * trial.lost_frames

    def feed(self, units) -> None:
        if self.rx_filter_mode == "trn2u-fit":
            if self._trn2u_rx_filter is None:
                self.set_filter_setup_error("filter was not installed at the E boundary")
                return
            self._trn2u_decode(*self._trn2u_filtered(units))
            return
        self._advance_gain_reference()
        octets = (v34_g711.encode(np.rint(np.asarray(units, dtype=float)).astype(np.int32), self.law)
                  .astype(np.uint8) & 0xFF).tobytes()
        self._held.extend(octets)
        if self.alignment is None:
            if len(self._held) < self._search + self.SEARCH_FRAMES * pu.INTERVALS:
                return
            if not self._gain_ready and (self._gain_reference is None
                    or len(self._held) < self._search + B1_FRAMES * pu.INTERVALS):
                return
            held = bytes(self._held)
            budget = self.SCORES_PER_CALL
            if self._gain_reference is not None:
                if self._fast_candidates is None:
                    self._fast_candidates = self._zero_candidates(held)
                # Candidates are ascending: the first verified zero is also
                # the lowest-offset global minimum of the full score.
                while self._fast_index < len(self._fast_candidates) and budget:
                    skip = self._fast_candidates[self._fast_index]
                    self._fast_index += 1
                    budget -= 1
                    if self._score(held, skip) == 0:
                        self.alignment = skip
                        break
                if self.alignment is None and self._fast_index < len(self._fast_candidates):
                    return
            if self.alignment is None:
                # A nonzero/bad B1 still receives the complete original search.
                for skip in range(len(self._scores), min(self._search, len(self._scores) + budget)):
                    self._scores.append((self._score(held, skip), skip))
                if len(self._scores) < self._search:
                    return
                self.alignment = min(self._scores)[1]
            self.rx = purx.PcmUpstreamStreamReceiver(
                self.profile, self.law, start_skip=self.alignment,
                max_lost_frames=self._max_lost_frames,
                max_data_lost_frames=self._max_data_lost_frames)
        if not self._gain_ready:
            if (self._gain_frames < B1_FRAMES
                    or len(self._held) < self.alignment + B1_FRAMES * pu.INTERVALS):
                return
            self._fit_b1_gain()
        # Searching held the incoming B1/data. Decode at most twice this input
        # chunk per call (at least 320 samples) until caught up, rather than
        # replacing scoring with an unbounded decode of the accumulated buffer.
        take = min(len(self._held), max(len(octets), 160) * 2)
        ready = bytes(self._held[:take])
        del self._held[:take]
        self.rx.feed(ready)
        # Idle after the payload makes the receiver fail; what it delivered stays delivered.
        self._collect_received_bits()

    def decoder_timing_report(self, tx_thread_ids=()) -> dict:
        if not self._decoder_timing_enabled:
            return {"enabled": False}
        values = np.asarray(self._decoder_chunk_ns, dtype=np.float64) / 1e6
        rx_threads = sorted(self._decoder_thread_ids)
        tx_threads = sorted(set(tx_thread_ids))
        return {"enabled": True, "chunks": len(values),
                "median_ms": float(np.percentile(values, 50)) if values.size else None,
                "p99_ms": float(np.percentile(values, 99)) if values.size else None,
                "max_ms": float(values.max()) if values.size else None,
                "rx_thread_ids": rx_threads, "tx_generation_thread_ids": tx_threads,
                "same_thread_as_tx_generation": bool(set(rx_threads) & set(tx_threads))}

    def take_bits(self) -> list[int]:
        new, self._new_bits = self._new_bits, []
        return new

    @property
    def b1_errors(self) -> int:
        return self.rx.b1_errors

    @property
    def qualified(self) -> bool:
        return self._qualified

    def payload(self) -> bytes:
        whole = len(self._all_bits) // 8 * 8
        bits = self._all_bits
        return bytes(sum(bits[8 * j + i] << i for i in range(8)) for j in range(whole // 8))


class Side:
    def __init__(self, role: str, *, encode, decoder, parsers, build_suv, build_cp, e_bits: int,
                 peer_e_bits: int, trn_units: int, early: bool, ack_wait: int, trn_msg_bits: int,
                 align_bits: int, peer_align_bits: int, data_rx, make_data_units, idle_unit,
                 trn_frame_units: int = 1, data_more=None, silent_rt: bytes = b"", rt_transition=None,
                 silent_early_rt_after: int | None = None, flush_after_e=None, e_before_fb1: bool = False,
                 e_to_run_end: bool = False):
        self.role = role
        self._encode, self._decoder = encode, decoder
        # Called once E has been encoded, before the next block's mapper takes over.
        # A look-ahead mapper still holds E's final frame(s) at that point.
        self._flush_after_e = flush_after_e
        self._build_suv, self._build_cp = build_suv, build_cp
        self._e_bits = e_bits
        self._trn_units, self._early, self._ack_wait = trn_units, early, ack_wait
        self._trn_msg_bits, self._align = trn_msg_bits, align_bits
        if (isinstance(trn_frame_units, bool) or not isinstance(trn_frame_units, int)
                or trn_frame_units < 1):
            raise ValueError("trn_frame_units must be a positive integer")
        self._trn_frame_units = trn_frame_units
        self._make_data_units, self._idle = make_data_units, idle_unit
        self._data_more = data_more
        self._tx_thread_ids: set[int] = set()
        self._silent_rt = bytes(silent_rt)
        self._rt_transition = rt_transition
        if (silent_early_rt_after is not None
                and (isinstance(silent_early_rt_after, bool)
                     or not isinstance(silent_early_rt_after, int)
                     or not 2400 <= silent_early_rt_after < 8004)):
            raise ValueError("silent_early_rt_after must be an integer in 2400..8003 or None")
        if silent_early_rt_after is not None and role != "digital":
            raise ValueError("silent_early_rt_after is only valid for the digital side")
        self._silent_early_rt_after = silent_early_rt_after
        self._silent_wait_units = 0
        self._silent_state: str | None = None
        self._silent_active = False
        self._silent_peer_e_seen = False
        self._silent_trn_generated = 0
        self._silent_rx_buffer = bytearray()
        self._silent_rx_base: int | None = None
        self.rx_filter = None                   # StreamFilter for the control phases (equaliser), or None
        self.rx_filter_delay = 0                # its delay D: filtered[n] ~ the symbol that arrived at raw n - D
        self.isi = None                         # (h, cursor): the channel, to take the E symbols' tail out of B1u's start
        self._coded_tail = np.zeros(0)
        self._raw_tail = np.zeros(0)
        self.bit_source = None                  # callable(n) -> n bits for live data (e.g. LAPM's read_tx_bits)
        self.data_rx = data_rx
        self._b1u_rx_raw_history = np.empty(0, dtype=float)
        self._b1u_cut_diagnostics_enabled = os.environ.get("EL_MODEM_V92_B1_CUT_DIAGNOSTICS") == "1"
        self._b1u_cut_diagnostic: dict | None = None
        self.scanner = _Scanner(parsers, peer_e_bits, peer_align_bits, e_before_fb1=e_before_fb1,
                                e_to_run_end=e_to_run_end)
        self._rx_unit_history = b""           # e_to_run_end: recent codewords, for an E end in an earlier chunk
        # Scanner bit offsets are local to each post-E/restart window, while
        # raw_of_bit() is measured from the decoder's current stream origin.
        self._scanner_bit_origin = int(getattr(decoder, "bits_out", 0))
        self._units_out: list = []
        self.units_sent = 0
        self.phase = "TRN"
        self.sent: list[str] = []
        self.saw_suv = self.saw_cp = self.saw_ack = False
        self.sent_ack = self.sent_e = False
        self.cp_sent_at: int | None = None
        self.first_cp_tx_at: int | None = None
        self.frames_seen: list[tuple[str, bool]] = []
        self.tx_trace: list[dict] = []
        self.rx_trace: list[dict] = []
        # The first unacknowledged and acknowledged CP frames can be several
        # thousand bits. Keep their exact logical wire input only in an
        # explicitly enabled lab capture, so parser/peer length failures can
        # be audited independently without growing normal call metrics.
        self._capture_cp_bits = os.environ.get("EL_MODEM_V92_CAPTURE_CP_BITS") == "1"
        self.cp_tx_captures: list[dict] = []
        self.tx_counts: dict[str, int] = {}
        self.silent_period_requested: bool | None = None
        self.silent_period_events: list[tuple[str, bool]] = []
        # Bit 32 in the locally transmitted SUV is independent from the peer's
        # request recorded above.  Rate-renegotiation FSMs can change this only
        # at an SUV boundary; keeping it on Side makes the wire bit testable
        # through the real mapper instead of only through the frame codec.
        self.silent_period_tx = False
        self.peer_cp: dict | None = None
        self.raw_in = 0
        self.data_mode_rx = False

    def diagnostics(self) -> dict:
        """A small snapshot for live debugging: how far the control exchange got and what the data receiver has seen."""
        rx = self.data_rx
        decoder = self._decoder
        decoder_diagnostics = {
            "type": type(decoder).__name__,
            "acquisition_state": getattr(decoder, "acquisition_state", None),
            "trn2u_at": getattr(decoder, "trn2u_at", None),
            "eq_delay": getattr(decoder, "eq_delay", None),
            "eq_residual": getattr(decoder, "eq_residual", None),
            "acquisition_error": getattr(decoder, "acquisition_error", None),
            "acquisition_failures": list(getattr(decoder, "acquisition_failures", ())),
            "acquisition_attempts": getattr(decoder, "acquisition_attempts", None),
            "acquisition_retries": getattr(decoder, "acquisition_retries", None),
            "fit_samples": getattr(decoder, "fit_samples", None),
            "trn_fit_taps": getattr(decoder, "trn_fit_taps", None),
            "trn_fit_max_delay": getattr(decoder, "trn_fit_max_delay", None),
            "fit_wall_seconds": getattr(decoder, "fit_wall_seconds", None),
            "start_raw": getattr(decoder, "start_raw", None),
            "raw_in": getattr(decoder, "raw_in", None),
            "bits_out": getattr(decoder, "bits_out", None),
            "broken": getattr(decoder, "broken", None),
            "broken_reason": getattr(decoder, "broken_reason", None),
            "broken_at_raw": getattr(decoder, "broken_at_raw", None),
            "decoded_codewords": getattr(decoder, "decoded_codewords", None),
            "first_non_one_bit": getattr(decoder, "first_non_one_bit", None),
            "zero_bits": getattr(decoder, "zero_bits", None),
            "pending_units": len(getattr(decoder, "pending", ())),
            "bit_trace": (None if getattr(decoder, "bit_trace", None) is None
                          else "".join(map(str, decoder.bit_trace))),
        }
        return {"phase": self.phase, "sent": list(self.sent), "saw_suv": self.saw_suv, "saw_cp": self.saw_cp,
                "saw_ack": self.saw_ack, "sent_ack": self.sent_ack, "sent_e": self.sent_e, "saw_e": self.scanner.e_seen,
                "frames_seen": list(self.frames_seen[-12:]), "raw_in": self.raw_in, "units_sent": self.units_sent,
                "decoder": decoder_diagnostics,
                "upstream_rx_filter": getattr(rx, "rx_filter_diagnostic", None),
                "upstream_decoder_timing": (rx.decoder_timing_report(self._tx_thread_ids)
                                             if hasattr(rx, "decoder_timing_report") else None),
                "diagnostic_metadata": getattr(self, "diagnostic_metadata", {}),
                "tx_counts": dict(self.tx_counts),
                # These counters use different domains: trn2u_at is a receive
                # symbol index, while first_cp_tx_at is an outbound unit index.
                # Keep both for diagnosis; do not subtract them as a latency.
                "first_cp_tx_at": self.first_cp_tx_at,
                "tx_trace": list(self.tx_trace), "rx_trace": list(self.rx_trace),
                "cp_tx_captures": list(self.cp_tx_captures),
                "silent_period_requested": self.silent_period_requested,
                "silent_period_tx": self.silent_period_tx,
                "silent_period_state": self._silent_state,
                "silent_period_events": list(self.silent_period_events[-8:]),
                "alignment": getattr(rx, "alignment", None),
                "gain": getattr(rx, "gain", None),
                "gain_calibration": getattr(rx, "gain_calibration", None),
                "qualified": bool(rx.qualified),
                "b1_errors": getattr(rx, "b1_errors", None) if not callable(getattr(rx, "b1_errors", None)) else rx.b1_errors(),
                "b1u_cut_diagnostic": (
                    dict(self._b1u_cut_diagnostic)
                    if self._b1u_cut_diagnostics_enabled
                    and self._b1u_cut_diagnostic is not None else None),
                "b1u_frame_trace": list(getattr(
                    getattr(getattr(rx, "new_receiver", rx), "rx", None), "b1_frame_trace", ()))
                                    if self._b1u_cut_diagnostics_enabled else None}

    @property
    def saw_e(self) -> bool:
        return self.scanner.e_seen

    @property
    def connected(self) -> bool:
        return self.sent_e and self.saw_e

    @property
    def data_ready(self) -> bool:
        return self.connected and self.data_rx.qualified

    # -- receive ------------------------------------------------------------------
    def receive(self, units) -> None:
        units = list(units) if not isinstance(units, (bytes, bytearray)) else bytes(units)
        base = self.raw_in
        if (getattr(self.data_rx, "rx_filter_mode", "off") == "trn2u-fit"
                and isinstance(units, bytes)):
            target = getattr(self.data_rx, "new_receiver", self.data_rx)
            target.set_filter_setup_error(
                "TRN2u receive FIR requires linear PCM samples; received G.711 bytes")
            if self._b1u_cut_diagnostics_enabled:
                self._b1u_cut_diagnostic = {
                    "event": "rejected-g711-byte-input",
                    "input_type": "bytes",
                    "expected_input": "linear_pcm_samples",
                    "raw_sample_base": int(base),
                }
            return
        self.raw_in += len(units)
        b1u_rx_window = None
        if getattr(self.data_rx, "rx_filter_mode", "off") == "trn2u-fit":
            raw_now = (np.frombuffer(units, dtype=np.uint8).astype(float) if isinstance(units, bytes)
                       else np.asarray(units, dtype=float).reshape(-1))
            prior = self._b1u_rx_raw_history
            origin = base - len(prior)
            combined = np.concatenate((prior, raw_now))
            self._b1u_rx_raw_history = combined[-512:].copy()
            b1u_rx_window = (combined, origin)
        run_end_window = None
        if self.scanner.e_to_run_end and isinstance(units, bytes):
            # E's end is known only when its zero run stops, up to a frame into a later chunk.
            prior = self._rx_unit_history
            run_end_window = (prior + units, base - len(prior))
            self._rx_unit_history = run_end_window[0][-128:]
        if self._silent_state in ("analogue-trn", "analogue-wait-rt"):
            # The incoming Ucode-0 silence and R-t are outside the old downstream
            # mapper alphabet. Keep a rolling raw-codeword window in both states:
            # during TRN2u an early R-t terminates training; after the maximum
            # wait it is the expected transition into the new Phase 4 control.
            if not self._silent_rx_buffer:
                self._silent_rx_base = base
            self._silent_rx_buffer.extend(bytes(units))
            at = self._silent_rx_buffer.find(self._silent_rt) if self._silent_rt else -1
            if at < 0:
                keep = max(len(self._silent_rt) - 1, 0)
                if len(self._silent_rx_buffer) > keep:
                    drop = len(self._silent_rx_buffer) - keep
                    del self._silent_rx_buffer[:drop]
                    self._silent_rx_base = int(self._silent_rx_base) + drop
                return
            after = at + len(self._silent_rt)
            base = int(self._silent_rx_base) + after
            units = bytes(self._silent_rx_buffer[after:])
            self._silent_rx_buffer.clear()
            self._silent_rx_base = None
            was_analogue_trn = self._silent_state == "analogue-trn"
            self._silent_state = None
            if was_analogue_trn:
                # Clause 9.8.2.1.6 permits TRN2u to end when R-t arrives.
                # Restart the downstream decoder exactly after the complete
                # R-t/R-bar-t pair, then resume with SUVu(bit 32 clear).
                self._reset_control_round()
                self.silent_period_tx = False
                self.sent.append("SILENT:trn2u-terminated-by-rt")
            self._decoder.restart(base)
            self.scanner = _Scanner(self.scanner.parsers, self.scanner.e_bits, self.scanner.align,
                                    e_before_fb1=self.scanner.e_before_fb1,
                                    e_to_run_end=self.scanner.e_to_run_end)
            self._scanner_bit_origin = 0
            if self._rt_transition is not None:
                self._rt_transition("received")
            self.sent.append("SILENT:rt-detected")
            if not units:
                return
        if self.data_mode_rx:
            self.data_rx.feed(units)
            return
        coded = units
        equalized_control = (self.rx_filter is not None
                             and not isinstance(self._decoder, _FastUpstreamDecoder))
        # The PCM-upstream mapper uses the negotiated CPd precoder/prefilter
        # to put the line output directly in its constellation domain. The
        # Phase 4 equalizer is only for the ordinary symbol-coded control
        # stream; applying it here double-corrects the PCM data waveform.
        if equalized_control:
            raw = np.asarray(units, dtype=float)
            coded = self.rx_filter.apply(raw)
            self._coded_tail = np.concatenate((self._coded_tail, coded))[-256:]
        checkpoint = (self._decoder.checkpoint()
                      if hasattr(self._decoder, "checkpoint") else None)
        decoded_bits = self._decoder.feed(coded)
        for frame in self.scanner.feed(decoded_bits):
            self.frames_seen.append((frame["kind"], bool(frame.get("ack", False))))
            frame_summary = {key: frame[key] for key in (
                "kind", "ack", "frame_bits", "wait_for_cpu", "level_db",
                "silent_period", "drn", "rate", "trellis", "gain4",
                "extend_e2u") if key in frame}
            self._append_phase4_trace(self.rx_trace, {
                "sample_at_or_before": self.raw_in,
                "decoded_bits": getattr(self._decoder, "bits_out", None),
                **frame_summary,
            })
            if frame["kind"] in ("suvu", "suvd", "cpu", "cpd"):
                self.saw_suv = True
            if "silent_period" in frame:
                requested = bool(frame["silent_period"])
                if requested != self.silent_period_requested:
                    self.silent_period_events.append((frame["kind"], requested))
                self.silent_period_requested = requested
                self._silent_active |= requested or self.silent_period_tx
                if (self.role == "digital" and self._silent_state == "digital-wait-clear"
                        and frame["kind"] == "suvu" and not requested):
                    self._silent_state = "digital-send-rt"
                    self.silent_period_tx = False
                    self._reset_control_round()
                    self.sent.append("SILENT:request-cleared")
            if frame["kind"] in ("cpu", "cpd"):
                self.saw_cp, self.peer_cp = True, frame
            if frame.get("ack"):
                self.saw_ack = True
        if equalized_control:
            self._raw_tail = np.concatenate((self._raw_tail, np.asarray(units, dtype=float)))[-256:]
        if getattr(self._decoder, "broken", False):
            self.scanner.end_of_stream()              # B1d's codewords are outside the training mapper
        if self.scanner.e_seen:
            if self._silent_active:
                self._enter_silent_after_peer_e()
                return
            self.data_mode_rx = True
            cut_bit, decoder_raw_cut, cut = _e_boundary_cut(
                self._decoder, self._scanner_bit_origin, self.scanner.e_end_bit, base)
            if hasattr(self._decoder, "mark_e"):
                self._decoder.mark_e(cut_bit)
            cut_before_rx_filter_delay = cut
            if (not equalized_control
                    and getattr(self.data_rx, "rx_filter_mode", "off") == "trn2u-fit"):
                weights = getattr(self._decoder, "fit_weights", None)
                delay = getattr(self._decoder, "eq_delay", None)
                target_rms = getattr(self._decoder, "fit_target_rms", None)
                residual = getattr(self._decoder, "eq_residual", None)
                if weights is None or delay is None or target_rms is None or residual is None or b1u_rx_window is None:
                    self.data_rx.set_filter_setup_error("TRN2u fit or E-boundary sample history was unavailable")
                else:
                    from v92_equaliser import StreamFilter
                    combined, origin = b1u_rx_window
                    end = base + int(cut) - origin
                    need = len(weights) - 1
                    if end < 0 or end > len(combined) or end - need < 0:
                        self.data_rx.set_filter_setup_error("saved PCM does not cover the TRN2u FIR prehistory at E")
                    else:
                        filt = StreamFilter(weights)
                        filt._history = combined[end - need:end].astype(float, copy=True)
                        levels = np.asarray([level for constellation in self.data_rx.profile.constellations
                                             for level in constellation], dtype=float)
                        target_point_rms = float(np.sqrt(np.mean(levels ** 2))) * self.data_rx.cpd_gain
                        gain = pu.cpd_receive_scale(self.data_rx.cpd_gain)
                        bias = float(getattr(self._decoder, "fit_output_bias", 0.0))
                        self.data_rx.install_trn2u_filter(
                            filt, delay=int(delay), gain=gain, target_rms=target_point_rms,
                            training_target_rms=float(target_rms), fit_residual=float(residual),
                            fractional_delay_samples=self.data_rx.fitted_fractional_delay_samples,
                            bias=bias)
            if equalized_control:
                cut_filtered = cut
                # ``cut`` is in the filtered stream, D samples later than the raw one; go back to raw samples.
                cut -= self.rx_filter_delay
                raw_all = self._raw_tail                       # holds this chunk and what came before it
                start = len(raw_all) - len(units) + cut
                tail = raw_all[max(start, 0):].copy()
                if self.isi is not None and start >= 0 and len(tail):
                    self._remove_e_tail(tail, cut_filtered, len(coded))
                if len(tail):
                    self.data_rx.feed(tail)
            elif cut < len(units):
                if checkpoint is not None:
                    # The fast upstream decoder is stateful in old-data mode.
                    # It saw the whole RTP chunk to let the scanner find E;
                    # rewind and replay only the control prefix before handing
                    # the tail to the FB1u/new-B1u data receiver.
                    prefix = coded[:max(0, min(int(cut), len(coded)))]
                    self._decoder.restore(checkpoint)
                    self._decoder.feed(prefix)
                history_window = (b1u_rx_window
                                  if getattr(self.data_rx, "rx_filter_mode", "off") == "trn2u-fit"
                                  else run_end_window)
                data_units, first_sample_handed, recovered_from_history = _units_from_raw_cut(
                    units, base, cut, history_window)
                if self._b1u_cut_diagnostics_enabled:
                    fitted_filter = getattr(self.data_rx, "_trn2u_rx_filter", None)
                    self._b1u_cut_diagnostic = {
                        "scanner_bit_origin": int(self._scanner_bit_origin),
                        "e_end_bit": int(self.scanner.e_end_bit),
                        "e_end_bit_is_exclusive": True,
                        "cut_bit_index": int(cut_bit),
                        "decoder_start_raw": getattr(self._decoder, "start_raw", None),
                        "decoder_raw_cut_sample": int(decoder_raw_cut),
                        "input_chunk_base": int(base),
                        "cut_relative_before_rx_filter_delay": int(cut_before_rx_filter_delay),
                        "control_rx_filter_delay": int(self.rx_filter_delay),
                        "control_rx_filter_delay_subtracted": False,
                        "cut_relative_after_rx_filter_delay": int(cut),
                        "requested_raw_cut_sample": int(base + cut),
                        "legacy_clipped_first_sample": int(base + max(cut, 0)),
                        "first_sample_handed_to_data_rx": int(first_sample_handed),
                        "samples_recovered_from_history": max(0, int(base - first_sample_handed)),
                        "history_window_used": bool(recovered_from_history),
                        "history_recovery_status": ("recovered" if recovered_from_history else
                                                     "unavailable_clipped" if cut < 0 else "not_needed"),
                        "history_recovery_warning": (
                            "negative E cut was not covered by retained raw history; input was clipped"
                            if cut < 0 and not recovered_from_history else None),
                        "fir_history_samples": len(getattr(fitted_filter, "_history", ())),
                        "trn2u_fit_delay_samples": getattr(self._decoder, "eq_delay", None),
                        "filter_drop_count_before_feed": getattr(self.data_rx, "_trn2u_rx_drop", None),
                        "fitted_fractional_delay_samples": getattr(
                            self.data_rx, "fitted_fractional_delay_samples", None),
                        "fractional_delay_applied": False,
                        "align_search": getattr(self.data_rx, "_search", None),
                        "calibrate_gain": getattr(self.data_rx, "_calibrate_gain", None),
                        "gain_ready_before_b1": getattr(self.data_rx, "_gain_ready", None),
                        "chosen_alignment": getattr(self.data_rx, "alignment", None),
                    }
                self.data_rx.feed(data_units)

    def _remove_e_tail(self, tail: np.ndarray, cut_filtered: int, chunk: int) -> None:
        """The last symbols of E (known: just decoded, equalised) reach the first B1u samples through the channel's
        trailing taps. Take that out so the first trellis group is not decided wrong, which would poison every later
        frame: line sample T + i holds h[c + i + j] * s[-j] for each earlier symbol j >= 1 (c the main cursor)."""
        h, c = self.isi
        symbols = {}
        base = len(self._coded_tail) - chunk + cut_filtered
        for j in range(1, len(h) - c):
            if base - j >= 0:
                symbols[j] = float(self._coded_tail[base - j])
        for i in range(min(len(tail), len(h) - c)):
            tail[i] -= sum(h[c + i + j] * symbols.get(j, 0.0) for j in range(1, len(h) - c - i))

    # -- transmit -----------------------------------------------------------------
    def _next_message(self) -> tuple[str, list[int]]:
        if self._silent_state == "wait-peer-e":
            return "SILENCE", []
        if self._silent_state in ("digital-wait-clear", "digital-early-wait-rt") and self.sent_e:
            return "SILENCE", []
        if self._silent_state == "analogue-wait-rt":
            return "SUV", self._fill(self._build_suv(self.saw_cp, False))
        if self._silent_state == "analogue-trn":
            if self._silent_trn_generated < 8004:
                return "SILENT_TRN", [1] * self._trn_msg_bits
            self._silent_state = "analogue-wait-rt"
            self.silent_period_tx = False
            self._reset_control_round()
            self.sent.append("SILENT:trn2u-complete")
            return "SUV", self._fill(self._build_suv(False, False))
        if self._silent_state == "digital-send-rt":
            return "RT", []
        if self.phase == "TRN":
            early = self._early and self.saw_suv
            projected = self.units_sent + len(self._units_out)
            # TRN is a minimum duration. If the configured symbol count is
            # not an exact frame multiple, emit the final complete frame so
            # the requested minimum is never undershot.
            if projected < self._trn_units and not early:
                return "TRN", [1] * self._trn_msg_bits
            self.phase = "SUV"
        if self.phase == "DATA":
            return "DATA", []
        ack = self.saw_cp
        if ack:
            self.sent_ack = True
        if (self.sent_ack and (self.saw_ack or self.saw_e or self._silent_peer_e_seen)
                and self.cp_sent_at is not None):
            if self._silent_active:
                self.phase, self.sent_e = "SILENT", True
                self._silent_state = ("analogue-trn" if self.role == "analogue" else
                                      (self._digital_silent_wait_state() if self._silent_peer_e_seen
                                       else "wait-peer-e"))
                self._silent_trn_generated = 0
            else:
                self.phase, self.sent_e = "DATA", True
            return "E", [0] * self._e_bits
        if self.saw_suv and self.cp_sent_at is None:
            self.cp_sent_at = self.units_sent
            return "CP", self._fill(self._build_cp(ack))
        if (self.cp_sent_at is not None and not self.saw_ack
                and self.units_sent - self.cp_sent_at >= self._ack_wait):
            self.cp_sent_at = self.units_sent
            return "CP", self._fill(self._build_cp(ack))
        return "SUV", self._fill(self._build_suv(ack, self.silent_period_tx))

    @staticmethod
    def _append_phase4_trace(target: list[dict], item: dict) -> None:
        # Preserve the beginning of an exchange and a rolling tail without
        # letting a long call grow diagnostics without bound.
        if len(target) < 32:
            target.append(item)
        else:
            del target[16]
            target.append(item)

    def set_silent_period_request(self, requested: bool) -> None:
        """Set the bit-32 value used by subsequently built SUV frames.

        The caller must invoke this at a protocol transition where an SUV is
        next to be generated. It does not itself perform E/E2, silence, TRN2,
        R-t, or the associated data-frame state transitions.
        """
        if not isinstance(requested, bool):
            raise TypeError("silent-period request must be bool")
        self.silent_period_tx = requested
        self._silent_active |= requested

    def _enter_silent_after_peer_e(self) -> None:
        self.data_mode_rx = False
        self.scanner = _Scanner(self.scanner.parsers, self.scanner.e_bits, self.scanner.align,
                                    e_before_fb1=self.scanner.e_before_fb1,
                                    e_to_run_end=self.scanner.e_to_run_end)
        self._scanner_bit_origin = int(getattr(self._decoder, "bits_out", 0))
        if self.role == "digital":
            self._silent_state = self._digital_silent_wait_state()
        else:
            self._silent_state = "analogue-trn" if self.sent_e else "analogue-peer-e-seen"
        self._silent_trn_generated = 0
        self._silent_peer_e_seen = True
        self._silent_rx_buffer.clear()
        self._silent_rx_base = None
        self.sent.append("SILENT:peer-e")

    def _digital_silent_wait_state(self) -> str:
        if self._silent_early_rt_after is not None and self.silent_period_requested is False:
            self._silent_wait_units = 0
            self.sent.append("SILENT:early-rt-wait")
            return "digital-early-wait-rt"
        return "digital-wait-clear"

    def _reset_control_round(self) -> None:
        self.phase = "SUV"
        self.saw_suv = self.saw_cp = self.saw_ack = False
        self.sent_ack = self.sent_e = False
        self.cp_sent_at = None
        self.peer_cp = None
        self.data_mode_rx = False
        self._silent_active = False
        self._silent_peer_e_seen = False
        self.scanner = _Scanner(self.scanner.parsers, self.scanner.e_bits, self.scanner.align,
                                    e_before_fb1=self.scanner.e_before_fb1,
                                    e_to_run_end=self.scanner.e_to_run_end)
        self._scanner_bit_origin = int(getattr(self._decoder, "bits_out", 0))

    def _fill(self, bits) -> list[int]:
        bits = [int(b) & 1 for b in bits]
        return bits + [0] * (-len(bits) % self._align)

    def transmit(self, count: int):
        while len(self._units_out) < count:
            if self._silent_state == "digital-send-rt":
                position = self.units_sent + len(self._units_out)
                pad = (-position) % 6
                if pad:
                    self._units_out.extend([self._idle] * min(pad, count - len(self._units_out)))
                    if (len(self._units_out) < count
                            and (self.units_sent + len(self._units_out)) % 6 == 0):
                        self._units_out.extend(self._silent_rt)
                        self._silent_state = None
                        self._reset_control_round()
                        if self._rt_transition is not None:
                            self._rt_transition("sent")
                        self.sent.append("SILENT:rt-sent")
                    continue
                self._units_out.extend(self._silent_rt)
                self._silent_state = None
                self._reset_control_round()
                if self._rt_transition is not None:
                    self._rt_transition("sent")
                self.sent.append("SILENT:rt-sent")
                continue
            kind, bits = self._next_message()
            if kind == "SILENCE":
                silence_count = count - len(self._units_out)
                self._units_out.extend([self._idle] * silence_count)
                if self._silent_state == "digital-early-wait-rt":
                    needed = self._silent_early_rt_after - self._silent_wait_units
                    self._silent_wait_units += min(silence_count, needed)
                    if self._silent_wait_units >= self._silent_early_rt_after:
                        # Bit 32 was already clear in the peer's SUVu, so the
                        # digital modem may send R-t without waiting for another
                        # SUVu (V.92 9.8.1.1.5). Keep this opt-in path on a
                        # configured sub-8004T silence interval.
                        self._reset_control_round()
                        self.silent_period_tx = False
                        self._silent_state = "digital-send-rt"
                        self.sent.append("SILENT:early-rt-send")
                break
            if kind == "RT":
                continue
            if kind == "DATA":
                if self.sent[-1:] != ["DATA"]:
                    self.sent.append("DATA")
                    self._units_out.extend(self._make_data_units(self.peer_cp))
                # Generate only what this read still needs.  Data mappers emit whole frames
                # (six downstream symbols or twelve upstream PCM symbols), so passing the
                # full RTP request on every loop iteration overproduces by up to one frame.
                # That surplus used to accumulate on every 20 ms read, creating growing
                # media latency and making the transmitter's frame boundary drift away from
                # the actual wire position.
                missing = count - len(self._units_out)
                if missing > 0:
                    if self._data_more is not None:
                        self._tx_thread_ids.add(threading.get_ident())
                    self._units_out.extend(self._data_more(self, missing) if self._data_more is not None
                                           else [self._idle] * missing)
                continue
            self.tx_counts[kind] = self.tx_counts.get(kind, 0) + 1
            if kind == "CP" and self._capture_cp_bits and len(self.cp_tx_captures) < 2:
                self.cp_tx_captures.append({
                    "ack": bool(self.saw_cp),
                    "unit_at_or_after": self.units_sent + len(self._units_out),
                    "frame_bits": len(bits),
                    "bits": "".join("1" if int(bit) else "0" for bit in bits),
                })
            if kind in ("TRN", "SUV", "CP", "E"):
                tx_event = {"kind": kind,
                            "unit_at_or_after": self.units_sent + len(self._units_out),
                            "frame_number": self.tx_counts[kind]}
                if kind == "CP" and self.first_cp_tx_at is None:
                    self.first_cp_tx_at = tx_event["unit_at_or_after"]
                if kind in ("SUV", "CP"):
                    tx_event["ack"] = bool(self.saw_cp)
                self._append_phase4_trace(self.tx_trace, tx_event)
            self.sent.append(kind)
            encoded = self._encode(bits)
            if kind == "E" and self._flush_after_e is not None:
                encoded = list(encoded) + list(self._flush_after_e())
            self._units_out.extend(encoded)
            if kind == "SILENT_TRN":
                self._silent_trn_generated += len(encoded)
        out, self._units_out = self._units_out[:count], self._units_out[count:]
        self.units_sent += len(out)
        return out

    def transmit_to_frame_boundary(self, count: int, frame_units: int):
        """Drain up to the next data-frame boundary without emitting a partial new frame.

        The frame clock is the stream's consumed-unit count, not the RTP packet size.
        ``transmit`` keeps only the unfinished mapper frame buffered, so reaching a
        boundary also leaves ``_units_out`` empty. This is the safe insertion point for
        in-band V.92 R signals.
        """
        if self.phase != "DATA":
            raise RuntimeError("data-frame alignment is only available in data mode")
        if not isinstance(frame_units, int) or frame_units <= 0:
            raise ValueError("frame_units must be a positive integer")
        remaining = (-self.units_sent) % frame_units
        count = min(max(int(count), 0), remaining)
        return self.transmit(count) if count else []


def payload_bits(payload: bytes, k: int) -> list[int]:
    data = [(b >> i) & 1 for b in payload for i in range(8)]
    return data + [0] * (-len(data) % k)


def _scrambled_ones_and_payload(k: int, payload: bytes) -> list[int]:
    scr = v34_tx.Scrambler("answer")
    data = [(b >> i) & 1 for b in payload for i in range(8)]
    data += [0] * (-len(data) % k)
    return [scr.step(b) for b in [1] * (B1_FRAMES * k) + data]


_DOWNSTREAM_MAX_DBM0 = -12.5        # Astound's INFO0d allows -12 dBm0 of digital transmit power (2026-11-20)


def downstream_cpu_constellations(drn: int, law: str) -> tuple[list[tuple[int, ...]], tuple[int, ...]]:
    """CPu constellations and interval selectors for a downstream DRN.

    DRN 10 and below keep the 16-point Ucode blocks every call so far has used. Above it the six intervals share
    one constellation of M = ceil(2 ** ((drn + 14) / 6)) points with the largest minimum distance d whose average
    power stays under ``_DOWNSTREAM_MAX_DBM0``: for a given d, the lowest-power set takes points upward from the
    first level at least d/2 (the distance to its own negative), each at least d above the last. Packing the
    lowest Ucodes instead (1..64 at DRN 22) left 8 between points and the signal 13 dB under the limit; SmartLink's
    54,667 design for Astound keeps 40 (2026-11-20)."""
    if not 1 <= drn <= 22:
        raise ValueError("drn must be 1..22")
    if drn <= 10:
        return [tuple(range(16 + 16 * i, 32 + 16 * i)) for i in range(6)], (0, 1, 2, 3, 4, 5)
    points = int(np.ceil(2 ** ((drn + 14) / 6) - 1e-9))
    levels = np.abs(v34_g711.decode(np.asarray(g711_ucode_codewords(law), dtype=np.uint8), law).astype(float))

    def spaced(distance: float) -> list[int] | None:
        ucodes: list[int] = []
        for u in range(1, 128):
            if levels[u] - (levels[ucodes[-1]] if ucodes else -levels[u]) >= distance:
                ucodes.append(u)
                if len(ucodes) == points:
                    return ucodes
        return None

    best = None
    for distance in np.unique(np.concatenate([np.diff(levels), 2 * levels[1:]])):
        ucodes = spaced(float(distance))
        if ucodes is None:
            break
        rms = float(np.sqrt(np.mean(levels[ucodes] ** 2)))
        if 20 * np.log10(rms / 32768) + 3.14 <= _DOWNSTREAM_MAX_DBM0:
            best = tuple(ucodes)
    if best is None:
        raise ValueError(f"no constellation for DRN {drn} within the downstream power limit")
    return [best], (0,) * 6


_PAD_TABLES: dict = {}


def pad_aware_samples(samples, gain: float, law: str) -> list[float]:
    """For each sample, the G.711 value that lands nearest it after a digital ``gain`` and re-encoding.

    Astound's path raises our codes about 1.5 dB and re-encodes them (its CPd prefilter main tap is 0.84 while our
    TRN2u leaves our relay at exactly LU), so an upstream sample on our grid moves a whole step on the far side's.
    Aiming at the far side's grid instead keeps the point (2026-11-19)."""
    key = (round(float(gain), 4), law)
    table = _PAD_TABLES.get(key)
    if table is None:
        values = v34_g711.decode(np.arange(256, dtype=np.uint8), law).astype(float)
        after = v34_g711.roundtrip(np.rint(values * gain), law).astype(float)
        order = np.argsort(after, kind="stable")
        table = _PAD_TABLES[key] = (after[order], values[order])
    after, values = table
    target = np.asarray(samples, dtype=float) * gain
    hi = np.clip(np.searchsorted(after, target), 0, len(after) - 1)
    lo = np.clip(hi - 1, 0, len(after) - 1)
    pick = np.where(np.abs(after[lo] - target) <= np.abs(after[hi] - target), lo, hi)
    return values[pick].tolist()


def cpd_path_gain(peer_cpd: dict) -> float | None:
    """The path gain implied by the CPd prefilter's main tap, or None when it is within 2% of unity."""
    z2 = (peer_cpd.get("precoder") or {}).get("z2") or ()
    if not z2:
        return None
    main = abs(max(z2, key=abs))
    if main <= 0 or abs(1.0 / main - 1.0) < 0.02:
        return None
    return 1.0 / main


def analogue_side(profile: dict, law: str, *, points: int = 4, lu: float = 2000.0,
                  analogue_trn: int = 12096, ack_wait: int = 1200, drn: int = 2,
                  payload: bytes = b"", fast_upstream_profile: pu.PcmUpstreamProfile | None = None,
                  initial_sign: int = 0, fast_upstream_scale: float = 1.0,
                  trn2u_noise: float = 0.0, pad_aware_upstream: bool = False) -> Side:
    """Upstream: TRN2u, SUVu/CPu, E, then B1u and ``payload`` in PCM-upstream mode. Reads the digital
    modem's codewords; its downstream data mode is set by the CPu it sends.

    PCM-upstream samples go out at G*LU times the mapper output (G from the CPd, LU the TRN2u level), as
    SmartLink does; ``fast_upstream_scale`` is that factor for the preceding data mode of an FPE.

    ``trn2u_noise`` adds Gaussian noise of that fraction of LU to TRN2u and the control sequences. Our TRN2u sits
    exactly on the G.711 grid, so a digital modem that rates the line from it sees no quantisation and picks an
    upstream rate our data cannot reach (Astound's 44,000, 2026-11-19). ``pad_aware_upstream`` aims B1u and data at
    the far side's grid when the CPd prefilter implies a path gain (see ``pad_aware_samples``)."""
    if not 0.0 <= float(trn2u_noise) < 0.5:
        raise ValueError("trn2u_noise must be a fraction of LU in 0..0.5")
    noise_rng = np.random.default_rng(0x92) if trn2u_noise else None
    coder = up.SymbolCoder(points, initial_sign=initial_sign)
    nb = coder.bits
    carry: list[int] = []
    fast_tx = pu.PcmUpstreamTransmitter(fast_upstream_profile) if fast_upstream_profile is not None else None
    fast_scrambler = v34_tx.Scrambler("answer") if fast_upstream_profile is not None else None
    fast_guard_pending = bool(fast_tx and len(fast_upstream_profile.prefilter_z2) > 1
                              and np.any(np.abs(fast_upstream_profile.prefilter_z2[1:]) > 1e-4))
    if fast_guard_pending:
        z2_support = np.flatnonzero(np.abs(fast_upstream_profile.prefilter_z2) > 1e-4)
        fast_guard_frames = (int(z2_support[-1]) + pu.INTERVALS - 1) // pu.INTERVALS
    else:
        fast_guard_frames = 0
    fast_side: dict = {}

    def encode(bits):
        nonlocal fast_guard_pending
        if fast_tx is not None:
            k_fast = fast_upstream_profile.bits_per_frame
            out: list[float] = []
            if fast_guard_pending:
                # A line-designed CPd prefilter can have tens of intervals of
                # group delay. Prime its state with valid preceding-mode data
                # so the first post-FPE SUV/CP frames are not a startup tail.
                for _ in range(fast_guard_frames):
                    guard = [fast_scrambler.step(1) for _ in range(k_fast)]
                    out.extend(fast_tx.encode_frame(guard))
                fast_guard_pending = False
            scrambled = [fast_scrambler.step(int(b) & 1) for b in bits]
            if len(scrambled) % k_fast:
                raise ValueError("FPE upstream control must end on a PCM data-frame boundary")
            for i in range(0, len(scrambled), k_fast):
                out.extend(fast_tx.encode_frame(scrambled[i:i + k_fast]))
            return [fast_upstream_scale * v for v in out]
        carry.extend(int(b) & 1 for b in bits)
        whole = len(carry) - len(carry) % nb
        out = [v * lu for v in coder.encode(carry[:whole])]
        del carry[:whole]
        if noise_rng is not None and out:
            out = list(np.asarray(out) + noise_rng.normal(0.0, trn2u_noise * lu, len(out)))
        return out

    masks, interval_sets = downstream_cpu_constellations(drn, law)

    def cp(ack):
        return v92_cp.build_cpu(drn=drn, ack=ack, alaw=(law == "pcma"), constellations=masks,
                                interval_sets=interval_sets)

    own_cpu = dm.downstream_profile_from_cpu(v92_cp.parse_cpu(cp(False)))

    live = {}

    def data_units(peer_cpd):
        prof = dataclasses.replace(pu.profile_from_cpd(peer_cpd), gain=1.0)
        scale = float(peer_cpd["gain"]) * lu
        live["upstream_profile"], live["scale"] = prof, scale
        live["pad_gain"] = cpd_path_gain(peer_cpd) if pad_aware_upstream else None
        if "side" in live:
            live["side"].fast_upstream_profile = prof
            live["side"].upstream_scale = scale
        fb1u: list[float] = []
        if fast_tx is not None:
            # V.92 8.7.7: FB1u is sent in the preceding data mode. Continue
            # the control-mode scrambler and PCM mapper state; ordinary B1u
            # below begins with fresh state under the newly exchanged CPd.
            k_fast = fast_upstream_profile.bits_per_frame
            for _ in range(B1_FRAMES):
                fb1_bits = [fast_scrambler.step(1) for _ in range(k_fast)]
                fb1u.extend(fast_upstream_scale * v for v in fast_tx.encode_frame(fb1_bits))
            if "side" in fast_side and not fast_side.get("fb1u_marked"):
                fast_side["side"].sent.append("FPE:FB1u")
                # Side.transmit uses the last marker to ensure make_data_units
                # is called once. Preserve that state after adding diagnostics.
                fast_side["side"].sent.append("DATA")
                fast_side["fb1u_marked"] = True
        tx = pu.PcmUpstreamTransmitter(prof)
        if fast_tx is not None and prof == fast_upstream_profile:
            # FPE's new data phase resets the mapper, trellis and scrambler,
            # but this unchanged CPd keeps using the same causal precoder and
            # prefilter across FB1u/B1u. Preserve their signal histories so
            # B1u begins with the negotiated channel state, not a zero-state
            # FIR transient that the peer's B1 reference cannot tolerate.
            tx._u = list(fast_tx._u)
            tx._x = list(fast_tx._x)
            tx._v = list(fast_tx._v)
        k = prof.bits_per_frame
        bits = _scrambled_ones_and_payload(k, payload)
        out: list[float] = []
        for i in range(0, len(bits), k):
            out.extend(tx.encode_frame(bits[i:i + k]))
        live["tx"], live["k"], live["scr"] = tx, k, None
        # The scrambler continues from where B1u and the payload left it: rebuild its state.
        scr = v34_tx.Scrambler("answer")
        for b in [1] * (B1_FRAMES * k) + [(payload_bits(payload, k))[i] for i in range(len(payload_bits(payload, k)))]:
            scr.step(b)
        live["scr"] = scr
        return fb1u + _upstream_out([scale * float(v) for v in out])

    def _upstream_out(samples):
        gain = live.get("pad_gain")
        return samples if gain is None else pad_aware_samples(samples, gain, law)

    def data_more(side, count):
        """Frames of live data (K bits each from the bit source, scrambled) once B1u and any payload are out."""
        tx, k, scr = live["tx"], live["k"], live["scr"]
        frames = max(1, -(-count // 12))
        out: list[float] = []
        for _ in range(frames):
            bits = list(side.bit_source(k)) if side.bit_source is not None else [1] * k
            out.extend(tx.encode_frame([scr.step(int(b) & 1) for b in bits]))
        return _upstream_out([live["scale"] * float(v) for v in out])

    side = Side("analogue", encode=encode, decoder=_DownstreamDecoder(profile, law),
                parsers=(v92_cp.parse_suvd, v92_cp.parse_cpd),
                build_suv=lambda ack, silent: v92_cp.build_suvu(ack=ack, silent_period=silent), build_cp=cp,
                e_bits=(fast_upstream_profile.bits_per_frame if fast_upstream_profile else 12 * nb),
                peer_e_bits=2 * rx4_frame_bits(profile, law),
                trn_units=analogue_trn, early=True, ack_wait=ack_wait,
                trn_msg_bits=(fast_upstream_profile.bits_per_frame if fast_upstream_profile else 12 * nb),
                align_bits=(fast_upstream_profile.bits_per_frame if fast_upstream_profile else 12 * nb),
                peer_align_bits=rx4_frame_bits(profile, law),
                data_rx=_DownstreamDataRx(own_cpu, law), make_data_units=data_units, idle_unit=0.0,
                trn_frame_units=12, data_more=data_more,
                silent_rt=rt_t_pair_codewords(profile, law), e_to_run_end=True)
    side.own_downstream_profile = own_cpu
    side.fast_upstream_profile = fast_upstream_profile
    side.upstream_scale = fast_upstream_scale
    live["side"] = side

    def make_fpe_rm():
        """Encode RM/RM′ in the continuing old PCM-upstream data-mode state."""
        tx = live.get("tx")
        upstream_profile = live.get("upstream_profile")
        if tx is None or upstream_profile is None:
            raise RuntimeError("FPE requires an established PCM-upstream data transmitter")
        rm = pu.rm_intervals(upstream_profile.moduli)
        rm_prime = pu.rm_intervals(upstream_profile.moduli, prime=True)
        samples: list[float] = []
        z2 = np.asarray(upstream_profile.prefilter_z2, dtype=float)
        support = np.flatnonzero(np.abs(z2) > 1e-4)
        flush_samples = int(support[-1]) if len(support) else 0
        flush_frames = (flush_samples + pu.INTERVALS - 1) // pu.INTERVALS
        # A delayed CPd FIR can smear the first frame across the transition
        # from data. Send one guard RM frame, then the protocol's 32 RM + 2 RM′.
        if flush_frames:
            samples.extend(tx.encode_intervals(rm))
        for _ in range(32):
            samples.extend(tx.encode_intervals(rm))
        for _ in range(2):
            samples.extend(tx.encode_intervals(rm_prime))
        # CPd's causal prefilter may have a symbol-scale delay. Keep feeding
        # valid mapper intervals long enough for the complete RM/RM′ response
        # to emerge from its finite impulse response before Phase 4 switches.
        # These are flush inputs; their leading output is the tail of RM/RM′.
        for _ in range(flush_frames):
            samples.extend(tx.encode_intervals(rm_prime))
        return live["scale"] * np.asarray(samples, dtype=float), upstream_profile

    side.make_fpe_rm = make_fpe_rm
    if fast_tx is not None:
        fast_side["side"] = side
    return side


LU_FALLBACK = 2000.0        # TRN2u level assumed for G = 1/LU when neither a measurement nor ``lu`` is available


def digital_side(profile: dict, law: str, *, points: int = 4, lu: float | None = 2000.0,
                 digital_trn: int = 2040, ack_wait: int = 1200, drn: int = 9,
                 payload: bytes = b"", keep_payload: bool = True, cpd_builder=None,
                 empty_precoder_policy: str = "omit",
                 align_search: int = 0, max_lost_frames: int = 3,
                 async_upstream_equalizer: bool = False,
                 silent_early_rt_after: int | None = None,
                 fast_upstream_profile: pu.PcmUpstreamProfile | None = None,
                 upstream_data_profile: pu.PcmUpstreamProfile | None = None,
                 initial_sign: int = 0, b1u_rx_mode: str = "off",
                 fitted_fractional_delay_samples: float | None = None,
                 cpd_constellation_mode: str = "default", cpd_gain4: int | None = None,
                 empty_precoder_z2: list[int] | None = None, acquire_trn2u: bool | None = None,
                 adaptive_max_drn: int | None = None, fast_upstream_front=None) -> Side:
    """Downstream: TRN2d, SUVd/CPd, E, then B1d and ``payload`` in V.90 data mode (parameters from the
    received CPu). Reads the analogue modem's symbols.

    ``cpd_gain4`` is the CPd 4G field; None sends G = 1/LU, LU this call's TRN2u level when the first CPd goes
    out (``lu`` until it is measured), so the analogue modem's G*LU*U line samples are the CPd points U.

    ``acquire_trn2u`` (default: when ``digital_trn`` > 0) locates and fits the analogue modem's TRN2u; a rate
    renegotiation side sets it, since TRN2u is resent. ``adaptive_max_drn`` caps the adaptive CPd's DRN.

    ``fast_upstream_front`` (FPE from a TRN2u-fit data mode): that mode's fitted FIR (``trn2u_front``); it decodes the
    control and FB1u, and the new mode's receiver, and an adaptive CPd's design, start from its fit."""
    if (isinstance(max_lost_frames, bool) or not isinstance(max_lost_frames, int)
            or not 0 <= max_lost_frames <= 1024):
        raise ValueError("max_lost_frames must be an integer in 0..1024")
    if empty_precoder_policy not in ("include", "omit", "identity"):
        raise ValueError("empty_precoder_policy must be include, omit, or identity")
    if empty_precoder_z2 is not None:
        if empty_precoder_policy != "identity":
            raise ValueError("empty_precoder_z2 requires empty_precoder_policy='identity'")
        if (not isinstance(empty_precoder_z2, list) or not 1 <= len(empty_precoder_z2) <= 0x1FF
                or any(isinstance(t, bool) or not isinstance(t, int) or not -32768 <= t <= 32767
                       for t in empty_precoder_z2)):
            raise ValueError("empty_precoder_z2 must contain 1..511 signed 16-bit integer taps")
    if cpd_constellation_mode not in ("default", "sparse", "expanded10", "adaptive"):
        raise ValueError("cpd_constellation_mode must be default, sparse, expanded10 or adaptive")
    if cpd_gain4 is not None and (isinstance(cpd_gain4, bool) or not isinstance(cpd_gain4, int)
                                  or not 0 < cpd_gain4 <= 0xFFFF):
        raise ValueError("cpd_gain4 must be None or a nonzero 16-bit unsigned integer")
    gain = {"gain4": cpd_gain4, "live": False}

    def cpd_gain4_now():
        """The CPd 4G field: fixed, or 1/LU decided at the first CPd sent."""
        if gain["gain4"] is None:
            measured = getattr(decoder, "lu", None) if gain["live"] else None
            reference = (float(measured) if measured is not None and np.isfinite(measured) and measured > 0
                         else float(lu) if lu else LU_FALLBACK)
            gain4 = min(0xFFFF, max(1, round(4 * 65536 / reference)))
            if not gain["live"]:
                return gain4                                  # the construction-time parse only
            gain["gain4"] = side.lu_gain4 = gain4
            data_rx.cpd_gain = gain4 / 65536.0 / 4.0
            side.diagnostic_metadata["cpd_gain"] = {
                "gain4": gain4, "policy": "1/LU", "lu": reference,
                "lu_source": "measured" if measured is not None and reference == float(measured) else "nominal"}
        return gain["gain4"]
    if adaptive_max_drn is not None and (isinstance(adaptive_max_drn, bool) or not isinstance(adaptive_max_drn, int)
                                         or not 1 <= adaptive_max_drn <= 19):
        raise ValueError("adaptive_max_drn must be None or an integer in 1..19")
    if cpd_constellation_mode in ("sparse", "expanded10", "adaptive") and cpd_builder is not None:
        raise ValueError("CPd constellation diagnostics require the direct CPd builder")
    encoder = dp4.V90DigitalPhase4Encoder(profile, law)
    fb = encoder.s + encoder.k
    encoder.reset()
    levels = sorted({int(v) for v in v34_g711.decode(np.arange(256), law) if v > 0})[2:50]
    moduli = [30, 30, 30, 15] * 3
    selected_drn = drn
    cpd_diagnostic = {"mode": cpd_constellation_mode}
    if cpd_constellation_mode == "expanded10":
        # One-run SmartLink index-bound diagnostic: retain the original five
        # μ-law levels and extend them on the same quantized grid. LC=10 makes
        # every populated signed representative fit the peer's candidate stride.
        selected_drn = 1
        levels = [96, 292, 492, 716, 988, 1244, 1500, 1756, 1980, 2236]
        moduli = [10, 10, 10, 5] * 3
        cpd_diagnostic = {
            "mode": cpd_constellation_mode,
            "gain4": cpd_gain4,
            "levels_positive": levels,
            "moduli": moduli,
            "drn": selected_drn,
            "level_derivation": "positive PCMU decoded grid; preserve [96,292,492,716,988], then choose the next decoded positive value at least 196 above the prior level",
            "level_gaps": [b - a for a, b in zip(levels, levels[1:])],
            "trellis": 1,
            "interval_sets": [0] * 6,
        }
    adaptive = None
    if cpd_constellation_mode == "adaptive":
        # Designed from this call's TRN2u fit when the first CPd is sent (see cpd()); until then, and if the fit
        # never arrives, the provisional plan below. Levels come from the negotiated law's codeword grid.
        import v92_modulation_design as modulation_design
        grid = sorted({int(v) for v in v34_g711.decode(np.arange(256), law) if v > 0})

        def adaptive_plan(relative_residual):
            noise = relative_residual * ADAPTIVE_MAX_RMS
            plan = modulation_design.design_for_peer(grid, noise, max_rms=ADAPTIVE_MAX_RMS,
                                                     min_gap_sigmas=ADAPTIVE_GAP_SIGMAS,
                                                     max_drn=19 if adaptive_max_drn is None else adaptive_max_drn)
            if plan is None:
                return None
            plan["constellation"] = [int(v) for v in plan["constellation"]]
            plan["relative_residual"] = float(relative_residual)
            return plan

        adaptive = {"plan": adaptive_plan(ADAPTIVE_PROVISIONAL_RESIDUAL), "final": False, "make_plan": adaptive_plan}
        if adaptive["plan"] is None:
            raise RuntimeError("adaptive CPd designer found no plan for the provisional residual")

        def adopt(plan, final):
            nonlocal selected_drn, levels, moduli, cpd_diagnostic
            selected_drn, levels, moduli = int(plan["drn"]), list(plan["constellation"]), list(plan["moduli"])
            cpd_diagnostic = {
                "mode": cpd_constellation_mode, "gain4": cpd_gain4, "final": final,
                "levels_positive": levels, "moduli": moduli, "drn": selected_drn,
                "rate_bps": float(plan["rate"]), "relative_residual": plan["relative_residual"],
                "noise_rms": float(plan["noise_rms"]), "gap_sigmas": float(plan["gap_sigmas"]),
                "fallback_gap_sigmas": plan.get("fallback_gap_sigmas"), "max_rms": ADAPTIVE_MAX_RMS,
                "rms": float(plan["rms"]), "trellis": 1, "interval_sets": [0] * 6,
                "designer": "v92_modulation_design.design_for_peer"}

        adopt(adaptive["plan"], False)
    if cpd_constellation_mode == "sparse":
        # Arm 3 measured this normalized TRN2u-fit residual on the same peer
        # (artifact b1u-rx-trn2u-fit-1). Scale it into the default CPd point
        # domain, then let the standards-aware designer choose the levels and
        # moduli. This is a one-run diagnostic reference, not adaptive policy.
        import v92_modulation_design as modulation_design
        fit_residual_normalized = 0.07064116973921739
        reference_rms = float(np.sqrt(np.mean(np.asarray(levels, dtype=float) ** 2)))
        noise_rms = fit_residual_normalized * reference_rms
        plan = modulation_design.design(
            levels, noise_rms, max_drn=1, trellis=1, min_gap_sigmas=5.0)
        if plan is None:
            raise RuntimeError("sparse CPd designer found no clause-valid constellation")
        selected_drn = int(plan["drn"])
        levels = [int(level) for level in plan["constellation"]]
        moduli = list(plan["moduli"])
        cpd_diagnostic = {
            "mode": cpd_constellation_mode,
            "gain4": cpd_gain4,
            "levels_positive": levels,
            "moduli": moduli,
            "drn": selected_drn,
            "bits_per_frame": int(plan["bits"]),
            "rate_bps": float(plan["rate"]),
            "noise_rms_cpd_domain": float(noise_rms),
            "noise_rms_normalized": fit_residual_normalized,
            "reference_point_rms": reference_rms,
            "min_gap_sigmas": 5.0,
            "trellis": int(plan["trellis"]),
            "designer": "v92_modulation_design.design",
        }

    def finalise_adaptive():
        """At the first CPd that is sent: redesign from the TRN2u fit and point the receiver at it."""
        adaptive["final"] = True
        residual = getattr(decoder, "eq_residual", None)
        target = getattr(decoder, "fit_target_rms", None)
        plan = None
        if residual is not None and target:
            plan = adaptive["make_plan"](float(residual) / float(target))
        if plan is not None:
            adopt(plan, True)
        else:
            cpd_diagnostic["final"] = True
            cpd_diagnostic["final_note"] = "no TRN2u fit or no plan at CPd time; provisional plan kept"
        side.diagnostic_metadata["cpd_constellation"] = cpd_diagnostic
        parsed = v92_cp.parse_cpd(v92_cp.build_cpd(
            drn=selected_drn, trellis=1, gain4=cpd_gain4_now(), ack=False, moduli=moduli, interval_sets=(0,) * 6,
            constellations=[levels], precoder={"z1": [], "p1": [], "z2": [], "p2": []}))
        data_rx.set_profile(dataclasses.replace(pu.profile_from_cpd(parsed), gain=1.0))

    def cpd(ack):
        if adaptive is not None and not adaptive["final"] and adaptive.get("live"):
            finalise_adaptive()
        gain4 = cpd_gain4_now()
        if cpd_builder is not None:
            bits = v92_cp.with_cpd_gain(cpd_builder(ack), gain4)
        else:
            bits = v92_cp.build_cpd(drn=selected_drn, trellis=1, gain4=gain4, ack=ack, moduli=moduli,
                                    interval_sets=(0,) * 6, constellations=[levels],
                                    precoder={"z1": [], "p1": [], "z2": [], "p2": []})
        # Some receivers mishandle a present filter section whose four tap
        # counts are all zero. In omit mode, leave that optional section out;
        # identity mode is a diagnostic probe for parsers that require at
        # least one filter coefficient.
        if empty_precoder_policy != "include":
            parsed = v92_cp.parse_cpd(bits)
            pre = None if parsed is None else parsed.get("precoder")
            if pre is not None and not any(len(pre[name]) for name in ("z1", "p1", "z2", "p2")):
                replacement = (None if empty_precoder_policy == "omit" else
                               {"z1": [], "p1": [],
                                "z2": ([32767] if empty_precoder_z2 is None else empty_precoder_z2),
                                "p2": []})
                sets = parsed.get("constellations")
                bits = v92_cp.build_cpd(
                    drn=parsed["drn"], trellis=parsed["trellis"],
                    extend_e2u=bool(parsed["extend_e2u"]), ack=bool(parsed["ack"]),
                    gain4=round(parsed["gain"] * 4 * 65536),
                    moduli=parsed.get("moduli"), precoder=replacement,
                    interval_sets=parsed.get("interval_sets"),
                    constellations=None if sets is None else [list(group) for group in sets])
        return bits

    own_cpd = v92_cp.parse_cpd(cpd(False))
    own_cpd_gain = float(own_cpd["gain"]) if own_cpd is not None else 1.0
    if upstream_data_profile is None:
        own_up = dataclasses.replace(pu.profile_from_cpd(own_cpd), gain=1.0)
    else:
        # Internal rate-change callers can reuse the validated immutable CPd
        # geometry instead of serializing and parsing a long filter in TX.
        # They must pair this snapshot with the matching on-wire CPd builder.
        if upstream_data_profile.bits_per_frame != 2 * (drn + 17):
            raise ValueError("upstream data profile does not match the requested DRN")
        own_cpd_gain = float(upstream_data_profile.gain)
        own_up = dataclasses.replace(upstream_data_profile, gain=1.0)

    def encode(bits):
        return list(encoder.encode_bits([int(b) & 1 for b in bits]))

    def flush_after_e():
        # With S_r=1 and look-ahead, encode_bits() holds the last `lookahead` frames
        # in the shaping pipeline. Ed is the training mapper's final block (B1d uses
        # a fresh data-mode mapper, R-t resets this one), so emit them now or the
        # peer receives only part of Ed's 2-frame zero run and never leaves CPt.
        if encoder._pending_bits:
            raise RuntimeError("V.92 Ed did not end on a downstream mapper frame boundary")
        return list(encoder.flush_lookahead())

    live = {}

    def data_units(peer_cpu):
        prof = dm.downstream_profile_from_cpu(peer_cpu)
        enc = dp4.V90DigitalPhase4Encoder(prof, law)
        enc.reset()
        frame_bits = enc.s + enc.k
        wire = enc.encode_bits([1] * (B1_FRAMES * frame_bits))
        wire += enc.flush_lookahead()
        bits = [(byte >> i) & 1 for byte in payload for i in range(8)]
        bits += [0] * (-len(bits) % frame_bits)
        live["enc"], live["fb"] = enc, frame_bits
        return list(wire + enc.encode_bits(bits))

    def data_more(side, count):
        enc, fb = live["enc"], live["fb"]
        frames = max(1, -(-count // 6))
        bits = list(side.bit_source(frames * fb)) if side.bit_source is not None else [1] * (frames * fb)
        return list(enc.encode_bits(bits))

    if b1u_rx_mode not in ("off", "trn2u-fit-viterbi"):
        raise ValueError("b1u_rx_mode must be off or trn2u-fit-viterbi")
    use_trn2u_viterbi = b1u_rx_mode == "trn2u-fit-viterbi"
    if fast_upstream_profile is not None and use_trn2u_viterbi and fast_upstream_front is None:
        raise ValueError("FPE with the TRN2u-fit receiver needs the old data mode's fitted filter")
    decoder = (_FastUpstreamDecoder(fast_upstream_profile, law, front=fast_upstream_front) if fast_upstream_profile
               else _UpstreamDecoder(points, lu, initial_sign=initial_sign,
                                     acquire_trn2u=digital_trn > 0 if acquire_trn2u is None else acquire_trn2u,
                                     async_fit=async_upstream_equalizer,
                                     fit_output_bias=use_trn2u_viterbi))
    data_rx = _UpstreamDataRx(own_up, law, keep_payload, align_search, max_lost_frames,
                              rx_filter_mode="trn2u-fit" if use_trn2u_viterbi else "off",
                              fitted_fractional_delay_samples=fitted_fractional_delay_samples,
                              decoder_mode="viterbi32" if use_trn2u_viterbi else "hard",
                              cpd_gain=own_cpd_gain, nlms_mu=_trn2u_nlms_mu() if use_trn2u_viterbi else 0.0)
    if fast_upstream_profile is not None:
        data_rx = _FpeUpstreamDataRx(decoder, data_rx)
    side = Side("digital", encode=encode, decoder=decoder,
                parsers=(v92_cp.parse_suvu, v92_cp.parse_cpu),
                build_suv=lambda ack, silent: v92_cp.build_suvd(ack=ack, silent_period=silent), build_cp=cpd,
                e_bits=2 * fb,
                peer_e_bits=(fast_upstream_profile.bits_per_frame if fast_upstream_profile
                             else 12 * points_bits(points)), trn_units=digital_trn,
                early=False, ack_wait=ack_wait, trn_msg_bits=fb, align_bits=fb,
                peer_align_bits=(fast_upstream_profile.bits_per_frame if fast_upstream_profile
                                 else 12 * points_bits(points)),
                data_rx=data_rx, make_data_units=data_units,
                idle_unit=int(v34_g711.encode(np.zeros(1), law)[0]) & 0xFF,
                trn_frame_units=6, data_more=data_more,
                silent_rt=rt_t_pair_codewords(profile, law),
                rt_transition=lambda event: encoder.reset() if event == "sent" else None,
                silent_early_rt_after=silent_early_rt_after,
                flush_after_e=flush_after_e, e_before_fb1=fast_upstream_profile is not None)
    side.diagnostic_metadata = {"b1u_rx_mode": b1u_rx_mode,
                                "fitted_fractional_delay_samples": fitted_fractional_delay_samples,
                                "cpd_constellation": cpd_diagnostic}
    if adaptive is not None:
        if (fast_upstream_profile is not None and fast_upstream_front is None) or upstream_data_profile is not None:
            raise ValueError("adaptive CPd constellation needs a TRN2u fit")
        adaptive["live"] = True
    side.lu_gain4 = None                # the 4G of G = 1/LU once the first CPd fixes it; renegotiation keeps it
    gain["live"] = True
    return side


def rx4_frame_bits(profile: dict, law: str) -> int:
    """Input bits per six-symbol mapper frame (S + K) for a Phase 4 profile."""
    encoder = dp4.V90DigitalPhase4Encoder(profile, law)
    return encoder.s + encoder.k


def points_bits(points: int) -> int:
    return 2 if points == 4 else 3


def rt_t_pair_codewords(profile: dict, law: str) -> bytes:
    """Return the V.92 R-t 384T followed by complementary R-bar-t 24T."""
    ucodes = dp4._highest_power_interval_ucodes(profile, law)
    return dp4._r_signal(ucodes, law) * 16 + dp4._r_signal(ucodes, law, r_bar=True)
