"""Section-8 wire boundaries and queue preservation, not live qualification."""
import unittest
from itertools import islice
from v32_caller import A,B,C,D,POINTS,NR_DIFF,GPC,GPA,Scrambler,RATE_WORD_BASE,RATE_BITS
from v32_rate_change import (RateChangeTx,RateWordReader,RateControlReader,
                            RatePreambleDetector,RateDataReader,CLEAR_RATE_WORD,parse_offer,offered_rates)


def bits(word):return [(word>>s)&1 for s in range(15,-1,-1)]


def decode(symbols,role,last):
    desc=Scrambler(GPC if role=='call' else GPA);out=[]
    for symbol in symbols:
        point=POINTS.index(symbol);d=NR_DIFF[(last<<2)|point];last=point
        out.extend((desc.rev(d>>1),desc.rev(d&1)))
    return out


class RateChangeTests(unittest.TestCase):
    def test_idle_models_choose_reset_or_prior_state_without_payload_selection(self):
        import random
        from v32_caller import TrellisEncoder
        for role,receiver in (('call','answer'),('answer','call')):
            for rate in (9600,12000,14400):
                for transmitted_state in ((0,0),(2,1),(4,0),(5,1)):
                    with self.subTest(role=role,rate=rate,state=transmitted_state):
                        tx=RateChangeTx(role,RATE_BITS,rate,peer_word=RATE_WORD_BASE|RATE_BITS[rate])
                        g=tx.symbols(lambda:1);list(islice(g,64))
                        control=RateControlReader(receiver,tx.word)
                        for z in islice(g,72):control.feed(z)
                        prior=[1e30]*8;prior[2]=0.
                        if transmitted_state!=(0,0):prior[2]=1e30;prior[transmitted_state[0]]=0.
                        reader=RateDataReader(control,prior_seed=(prior,transmitted_state[1]))
                        scr=Scrambler(control.descrambler.tap);scr.wd=control.descrambler.wd
                        encoder=TrellisEncoder(rate);encoder.t,encoder.y=transmitted_state
                        payload=[random.Random(i+77).getrandbits(1) for i in range(1024)]
                        bits_in=iter([1]*(24*reader.nbits)+payload+[1]*200)
                        received=[]
                        for i in range(24+(len(payload)+reader.nbits-1)//reader.nbits+20):
                            group=0
                            for _ in range(reader.nbits):group=2*group+scr.fwd(next(bits_in))
                            received.extend(reader.feed(encoder.encode(group)))
                            if i<23:self.assertFalse(reader.physical_ready)
                        self.assertEqual(received[:len(payload)],payload)
                        self.assertEqual(reader.idle_errors,0)
                        self.assertEqual(reader.convention,'reset' if transmitted_state==(0,0) else 'continue')
                        self.assertEqual(reader.convention_scores['symbols'],24)

    def test_idle_model_ambiguity_and_poor_fit_keep_canonical_reset(self):
        import copy
        tx=RateChangeTx('call',RATE_BITS,9600,peer_word=RATE_WORD_BASE|RATE_BITS[9600])
        g=tx.symbols(lambda:1);list(islice(g,64))
        control=RateControlReader('answer',tx.word)
        for z in islice(g,72):control.feed(z)
        idle=list(islice(g,24))
        zero=[0.]+[1e30]*7
        same=RateDataReader(copy.deepcopy(control),prior_seed=(zero,0))
        for z in idle:same.feed(z)
        self.assertEqual(same.convention,'reset')
        self.assertEqual(same.convention_scores['reset'],same.convention_scores['continued'])
        poor=RateDataReader(copy.deepcopy(control),prior_seed=([1e30,1e30,0.]+[1e30]*5,1))
        for _ in range(24):poor.feed(0j)
        self.assertEqual(poor.convention,'reset')
        with self.assertRaises(ValueError):poor.feed(complex(float('nan'),0))
        for seed in (([0.]*7,0),([float('nan')]+[0.]*7,0),([0.]*8,4)):
            with self.assertRaises(ValueError):RateDataReader(copy.deepcopy(control),prior_seed=seed)

    def test_data_reader_preserves_descrambler_and_removes_idle_after_traceback(self):
        import random
        from itertools import chain,repeat
        for role,receiver in (('call','answer'),('answer','call')):
            for rate in RATE_BITS:
                payload=[random.Random(77+i).getrandbits(1) for i in range(1024)]
                source=iter(chain(payload,repeat(1)))
                tx=RateChangeTx(role,RATE_BITS,rate,peer_word=RATE_WORD_BASE|RATE_BITS[rate])
                g=tx.symbols(lambda:next(source));list(islice(g,64))
                control=RateControlReader(receiver,tx.word)
                for symbol in islice(g,72):control.feed(symbol)
                decoder=RateDataReader(control);received=[]
                self.assertIs(decoder.descrambler,control.descrambler)
                for _ in range(24):self.assertEqual(decoder.feed(next(g)),[])
                self.assertTrue(decoder.physical_ready)
                if rate!=4800:self.assertGreater(decoder.remaining_idle_groups,0)
                for _ in range(600):
                    received.extend(decoder.feed(next(g)))
                    if len(received)>=len(payload):break
                self.assertEqual(received[:len(payload)],payload)
                self.assertEqual(decoder.idle_errors,0)
                self.assertEqual(decoder.remaining_idle_groups,0)
        control=RateControlReader('answer',RATE_WORD_BASE|RATE_BITS[9600])
        with self.assertRaises(ValueError):RateDataReader(control)

    def test_preamble_detector_finds_both_roles_without_payload_or_rate_oracle(self):
        import numpy as np
        rng=np.random.default_rng(33)
        for role,receiver in (('call','answer'),('answer','call')):
            pattern=[A]*56+[C]*8 if role=='call' else [A,C]*28+[C,A]*4
            data=list(rng.choice(POINTS,100))+pattern+list(rng.choice(POINTS,100))
            noise=rng.normal(0,.05,len(data))+1j*rng.normal(0,.05,len(data))
            detector=RatePreambleDetector(receiver);events=[]
            for k,symbol in enumerate(.9*np.asarray(data)+noise):
                event=detector.feed(symbol,k)
                if event:events.append(event)
            self.assertEqual(len(events),1);self.assertEqual(events[0]['start_symbol'],100)
            self.assertEqual(events[0]['end_symbol'],164)
            self.assertLessEqual(events[0]['reported_symbol']-164,8)

    def test_preamble_detector_rejects_extended_full_tones_and_bad_indices(self):
        for receiver,data in (('answer',[A]*256+[C]*256),('call',[A,C]*128+[C,A]*128)):
            detector=RatePreambleDetector(receiver)
            self.assertFalse(any(detector.feed(s,k) for k,s in enumerate(data)))
        detector=RatePreambleDetector('answer');detector.feed(A,17)
        with self.assertRaises(ValueError):detector.feed(A,19)
        with self.assertRaises(ValueError):detector.feed(complex(float('nan'),0),18)

    def test_duplex_exchange_obeys_delayed_response_and_resumes_both_sources(self):
        for role,other in (('call','answer'),('answer','call')):
            for desired,remote_desired in ((14400,9600),(7200,14400),(14400,14400)):
                source_calls=[0,0]
                def source(n):
                    source_calls[n]+=1
                    return 1
                init=RateChangeTx(role,RATE_BITS,desired)
                gi=init.symbols(lambda:source(0))
                receive_r4=RateControlReader(other,RATE_WORD_BASE|sum(
                    b for r,b in RATE_BITS.items() if r<=remote_desired))
                receive_r5=RateControlReader(role,init.word)
                response=gr=None;response_age=0;offers_at=[]
                for tick in range(400):
                    symbol=next(gi)
                    if tick>=64 and not receive_r4.seen_e:
                        for event in receive_r4.feed(symbol):
                            if event['type']=='offer':
                                offers_at.append(tick)
                                response=RateChangeTx(other,RATE_BITS,remote_desired,
                                                      initiator=False,peer_word=event['word'])
                                gr=response.symbols(lambda:source(1))
                    if gr is not None:
                        symbol=next(gr)
                        if response_age>=64 and not receive_r5.seen_e:
                            for event in receive_r5.feed(symbol):
                                if event['type']=='offer':init.accept_peer_offer(event['word'])
                        response_age+=1
                    if (response is not None and init.tx_completed and response.tx_completed
                            and receive_r4.seen_e and receive_r5.seen_e):break
                else:self.fail('symbol-domain exchange did not complete')
                self.assertEqual(offers_at,[79]) # Two R4 words, after the 64T preamble.
                self.assertGreaterEqual(init.rate_symbols,64)
                self.assertEqual(response.rate_symbols,64)
                self.assertEqual(init.selected_rate,min(desired,remote_desired))
                self.assertEqual(response.selected_rate,init.selected_rate)
                self.assertTrue(all(source_calls))

    def test_control_reader_decodes_offer_then_e_at_boundary_in_both_roles(self):
        for role,receiver in (('call','answer'),('answer','call')):
            for rate in RATE_BITS:
                tx=RateChangeTx(role,RATE_BITS,rate,peer_word=RATE_WORD_BASE|RATE_BITS[rate])
                g=tx.symbols(lambda:1);list(islice(g,64))
                rx=RateControlReader(receiver,RATE_WORD_BASE|RATE_BITS[rate])
                events=[]
                for symbol in islice(g,72):events.extend(rx.feed(symbol))
                self.assertEqual([e['type'] for e in events],['offer','e'])
                self.assertEqual(events[0]['word'],tx.word)
                self.assertEqual(events[1]['rate'],rate)
                self.assertEqual(rx.reader.count,144)
                with self.assertRaises(RuntimeError):rx.feed(A)

    def test_control_reader_does_not_accept_single_offer_or_inconsistent_e(self):
        tx=RateChangeTx('call',[4800,9600],9600,peer_word=RATE_WORD_BASE|RATE_BITS[9600])
        g=tx.symbols(lambda:1);list(islice(g,64))
        rx=RateControlReader('answer',RATE_WORD_BASE|RATE_BITS[4800])
        for symbol in islice(g,8):self.assertEqual(rx.feed(symbol),[])
        events=[]
        for symbol in islice(g,64):events.extend(rx.feed(symbol))
        self.assertEqual([e['type'] for e in events],['offer'])
        self.assertFalse(rx.seen_e)

    def test_e_reserved_bits_are_ignored_at_the_validated_boundary(self):
        tx=RateChangeTx('call',[4800],4800)
        g=tx.symbols(lambda:1);list(islice(g,64))
        rx=RateControlReader('answer',tx.word)
        for symbol in islice(g,64):rx.feed(symbol)
        # Continue the same scrambler/differential histories, varying only
        # reserved B13/B14 on the received E word (Table 5).
        events=[]
        for symbol in tx._control_word(0xf000|tx.word|6):events.extend(rx.feed(symbol))
        self.assertEqual(events,[dict(type='e',word=0xf000|tx.word,rate=4800)])

    def test_exact_preambles_and_first_word_golden_vector(self):
        for role,expected in [('call',[A]*56+[C]*8),('answer',[A,C]*28+[C,A]*4)]:
            tx=RateChangeTx(role,[4800],4800,peer_word=RATE_WORD_BASE|RATE_BITS[4800])
            source_calls=[];g=tx.symbols(lambda:source_calls.append(1) or 1)
            self.assertEqual(list(islice(g,64)),expected)
            r=list(islice(g,8))
            # 0x0d91 dibits 0,0,3,1,2,1,0,1, previous point C (3).
            # NR_DIFF transitions: 3->2->0->2->2->1->1->3->3.
            if role=='call':self.assertEqual(r,[D,A,D,D,B,B,C,C])
            self.assertEqual(decode(r,role,3 if role=='call' else 0),bits(tx.word))
            self.assertEqual(source_calls,[])

    def test_all_modes_delay_payload_until_after_24_symbols_and_zero_trellis_seed(self):
        from v32_caller import TrellisEncoder,Trellis7200Encoder,TCM_BITS
        for rate in RATE_BITS:
            for role in ('call','answer'):
                tx=RateChangeTx(role,RATE_BITS,rate,peer_word=RATE_WORD_BASE|RATE_BITS[rate])
                calls=[];g=tx.symbols(lambda:calls.append(0) or 0)
                prefix=list(islice(g,64+64+8));self.assertEqual(calls,[])
                decoded=decode(prefix[64:],role,3 if role=='call' else 0)
                self.assertEqual(decoded[:128],bits(tx.word)*8)
                e=0xf000|RATE_WORD_BASE|RATE_BITS[rate]
                self.assertEqual(decoded[128:],bits(e))
                ones=list(islice(g,24));self.assertEqual(calls,[])
                self.assertFalse(tx.tx_completed)
                # Independent stream starts the new trellis state at zero,
                # keeping the descrambler history accumulated through R/E.
                if rate!=4800:
                    encoder=TrellisEncoder(rate) if rate in TCM_BITS else Trellis7200Encoder(0)
                    n=TCM_BITS.get(rate,3);scr=Scrambler(GPC if role=='call' else GPA)
                    for bit in decoded:scr.fwd(bit)
                    expected=[]
                    for _ in range(24):
                        b=0
                        for _ in range(n):b=(b<<1)|scr.fwd(1)
                        expected.append(encoder.encode(b))
                    self.assertEqual(ones,expected)
                next(g);self.assertTrue(tx.tx_completed)
                self.assertEqual(len(calls),rate//2400)

    def test_early_offer_waits_64_symbols_and_late_offer_completes_word(self):
        tx=RateChangeTx('call',[4800,9600,14400],14400)
        g=tx.symbols(lambda:1);list(islice(g,64+19))
        tx.accept_peer_offer(RATE_WORD_BASE|RATE_BITS[9600]|RATE_BITS[4800])
        list(islice(g,45));self.assertEqual(tx.rate_symbols,64)
        next(g);self.assertEqual(tx.state,'e');self.assertEqual(tx.selected_rate,9600)
        tx=RateChangeTx('answer',[4800,9600],9600)
        g=tx.symbols(lambda:1);list(islice(g,64+67));tx.accept_peer_offer(RATE_WORD_BASE|RATE_BITS[4800])
        list(islice(g,5));self.assertEqual(tx.rate_symbols,72)
        next(g);self.assertEqual(tx.state,'e')

    def test_responder_offer_is_independent_of_r4_and_empty_intersection_clears(self):
        tx=RateChangeTx('answer',[4800,7200,9600],9600,initiator=False,
                        peer_word=RATE_WORD_BASE|RATE_BITS[4800])
        self.assertEqual(offered_rates(tx.word),(4800,7200,9600));self.assertEqual(tx.selected_rate,4800)
        with self.assertRaises(ValueError):RateChangeTx('answer',[4800],4800,initiator=False)
        tx=RateChangeTx('call',[9600],9600,peer_word=RATE_WORD_BASE|RATE_BITS[4800])
        calls=[];symbols=list(tx.symbols(lambda:calls.append(1) or 1))
        self.assertEqual(len(symbols),64+64+64);self.assertEqual(calls,[])
        self.assertTrue(tx.cleared);self.assertFalse(tx.tx_completed)
        self.assertEqual(decode(symbols[64:],'call',3)[128:],bits(0xf000|CLEAR_RATE_WORD)*8)

    def test_cleardown_is_valid_without_b4_capability_and_its_e_clears_b4(self):
        self.assertEqual(parse_offer(CLEAR_RATE_WORD),CLEAR_RATE_WORD)
        self.assertEqual(offered_rates(CLEAR_RATE_WORD),())
        tx=RateChangeTx('call',[9600],9600,peer_word=CLEAR_RATE_WORD)
        g=tx.symbols(lambda:1);list(islice(g,64))
        rx=RateControlReader('answer',CLEAR_RATE_WORD);events=[]
        for symbol in islice(g,72):events.extend(rx.feed(symbol))
        self.assertEqual(events[-1],dict(type='e',word=0xf000|CLEAR_RATE_WORD,rate=0))

    def test_validation_repetition_alignment_reserved_and_legacy(self):
        w=RATE_WORD_BASE|RATE_BITS[9600]
        self.assertEqual(parse_offer(w|6),w)
        for bad in (-1,65536,w&~0x800,w&~0x80,w|0xf000,w^1):
            with self.assertRaises(ValueError):parse_offer(bad)
        reader=RateWordReader();self.assertEqual(reader.feed([1,0,1]+bits(w)),[])
        self.assertEqual(reader.feed(bits(w|6)),[w])
        self.assertEqual(reader.feed(bits(w)*4),[])
        self.assertEqual(reader.feed(bits(w|0xf000)*2),[])

    def test_timeout_and_changed_offer_do_not_consume_data(self):
        tx=RateChangeTx('call',[4800],4800,timeout_symbols=64)
        calls=[];g=tx.symbols(lambda:calls.append(1) or 1);list(islice(g,128))
        with self.assertRaises(TimeoutError):next(g)
        self.assertEqual(calls,[]);self.assertEqual(tx.state,'failed')
        tx=RateChangeTx('call',[4800,9600],9600,peer_word=RATE_WORD_BASE|RATE_BITS[4800])
        with self.assertRaises(ValueError):tx.accept_peer_offer(RATE_WORD_BASE|RATE_BITS[9600])

    def test_transmitter_hook_preserves_pcm_clock_pulse_history_and_byte_queue(self):
        from v32_caller import V32CallerTx
        for role in ('caller','answerer'):
            tx=V32CallerTx(my_rates=(9600,4800),role=role)
            tx._symbol_source=tx._rate_signal();tx.request_e(9600)
            tx.tx_samples(1600)
            tx.feed_bytes(b'queued')
            tx._byte_bits.extend([0,1,0])
            n,k,history=tx._n,tx._k_next,dict(tx._sym)
            exchange=RateChangeTx('call' if role=='caller' else 'answer',[9600,4800],4800,
                                  peer_word=RATE_WORD_BASE|RATE_BITS[4800])
            self.assertEqual(tx.start_rate_change(exchange),k)
            self.assertEqual((tx._n,tx._k_next),(n,k));self.assertEqual(tx._sym,history)
            self.assertEqual(list(tx._tx_bytes),list(b'queued'))
            self.assertEqual(list(tx._byte_bits),[0,1,0])
            with self.assertRaises(ValueError):tx.start_rate_change(exchange)
            out=tx.tx_samples(640)
            self.assertEqual(len(out),640);self.assertEqual(tx._n,n+640)
            self.assertTrue(exchange.tx_completed);self.assertEqual(tx.data_rate,4800)
            self.assertEqual(tx.rate_change_data_symbol-k,64+64+8+24)
