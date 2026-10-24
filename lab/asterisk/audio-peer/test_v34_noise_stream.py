"""Synthetic continuous-data check for the experimental TRN noise predictor."""
import copy
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent/'tools'))
from b1_noise_kernel import available
import v34_data as data
import v34_datarx as datarx
import v34_b1_reference as independent
import v34_receive as receive
from v34_negotiation import DirectionParameters
from v34_receive_dsp import ReceiveDsp
from v34_noise_kernel import (_spatial_tables, _stream_step,
                              _stream_step_exhaustive, _tables)


@unittest.skipUnless(available(),'optional Numba backend not installed')
class PredictiveStreamingTests(unittest.TestCase):
    def test_certified_spatial_search_matches_exhaustive_over_stream(self):
        length=1664
        points,classes,counts,transitions=_tables(length)
        grid,labels,parities,min_re,min_im=_spatial_tables(length)
        rng=np.random.default_rng(23023)
        for nonlinear in (False,True):
            for radius in (4,6):
                metrics=np.zeros(16);metrics[1:]=np.inf
                histories=np.zeros((16,3),np.complex128)
                residuals=np.zeros((16,3),np.complex128)
                h=np.asarray([.11+.03j,-.02+.01j,.008-.003j])
                coefficients=np.asarray([.35+.01j,.31+.02j,-.01+.02j])
                for interval in range(12):
                    # Stay within the negotiated 1,664-point V.34 constellation,
                    # while varying noise and forcing one neutral erased interval.
                    y0=complex(rng.uniform(-40,40),rng.uniform(-40,40))
                    y1=complex(rng.uniform(-40,40),rng.uniform(-40,40))
                    erasure=interval==7
                    common=(points,h,coefficients,30.,nonlinear,4.,classes,counts,
                            transitions,interval,metrics,histories,residuals)
                    fast=_stream_step(y0,y1,*common,grid,labels,parities,
                                      min_re,min_im,radius,erasure)
                    reference=_stream_step_exhaustive(y0,y1,*common,erasure)
                    for actual,expected in zip(fast,reference):
                        np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-9)
                    metrics,histories,residuals=fast[:3]

    def test_colored_noise_state_crosses_b1_to_bounded_data_stream(self):
        profile=DirectionParameters(rate=33600,baud=24000/7,
            carrier=1959.1836734693877,trellis=0,shaping=True,nonlinear=False,
            precoder=(129,188,-297,-160,222,-69))
        receiver=receive.Receiver(profile,'answer')
        truth=independent.build(profile.rate,profile.baud,'answer',profile.shaping,
                                profile.precoder,trace=True)
        gain=.11+.002j
        coefficients=np.asarray([.6-.12j,.12+.06j,-.08+.03j])
        rng=np.random.default_rng(340611)
        innovations=.45*(rng.normal(size=2000)+1j*rng.normal(size=2000))
        noise=np.zeros(2000,complex)
        for i in range(len(noise)):
            noise[i]=innovations[i]+sum(coefficients[k-1]*noise[i-k]
                                        for k in (1,2,3) if i>=k)
        b1=np.asarray(truth['x'])*gain
        b1+=noise[:len(b1)]*gain
        receiver.qualify(b1)
        self.assertGreater(receiver.diagnostics['bit_errors'],0)
        dsp=ReceiveDsp(type('Engine',(),{})(),predictive=True)
        dsp.noise_model=dict(accepted=True)
        dsp.coefficients=coefficients
        dsp.qualify(receiver,b1,gain)
        self.assertEqual(receiver.diagnostics['bit_errors'],0)
        self.assertEqual(receiver.diagnostics['noise_predictive']['data_detector'],
                         'TRN-noise predictive streaming')
        final_state=receiver.diagnostics['noise_predictive']['terminal_state']
        drx=receiver.drx
        self.assertEqual(tuple(drx.conv.d),tuple((final_state>>s)&1 for s in (3,2,1,0)))
        np.testing.assert_array_equal(drx.x_hist,truth['x'][-3:][::-1])

        tx=data.DataTx(profile.rate,profile.baud,expanded=profile.shaping,
                       precoder=receiver.h,nonlinear=False)
        tx.conv.d=drx.conv.d.copy();tx.x_hist=drx.x_hist.copy()
        tx.z=drx.z;tx.m4=drx.m4;tx.frame_pos=drx.frame_pos
        payload=np.random.default_rng(343).integers(0,2,size=tx.bits_per_data_frame).tolist()
        sent=np.asarray(tx.data_frame(payload),complex)
        observed=sent+noise[len(b1):len(b1)+len(sent)]
        self.assertEqual(len(observed)%2,0)

        ordinary=copy.copy(receiver);ordinary.drx=copy.deepcopy(drx)
        ordinary_detector=ordinary.detector(start=True);ordinary_detector.emit_states=True
        for a,b in observed.reshape(-1,2):ordinary_detector.feed(a,b)
        ordinary_detector.flush()
        ordinary_points=np.asarray([p for pair in ordinary_detector.decided for p in pair])
        ordinary_rx=copy.deepcopy(drx);ordinary_bits=[]
        for offset in range(0,len(ordinary_points),8):
            frame=offset//8
            states=ordinary_detector.decided_states[4*frame:4*(frame+1)]
            ordinary_bits.extend(ordinary_rx.decode_frame(
                ordinary_points[offset:offset+8],trellis_states=states)[0])

        detector=receiver.vit
        detector.emit_states=True
        seed_residuals=detector._survivor_e[final_state].copy()
        max_pending=0
        for a,b in observed.reshape(-1,2):
            detector.feed(a,b);max_pending=max(max_pending,len(detector.hist))
        detector.flush()
        decoded_points=np.asarray([p for pair in detector.decided for p in pair])
        self.assertLessEqual(max_pending,25)
        self.assertEqual(len(detector.decided),len(observed)//2)
        self.assertEqual(len(detector.decided_states),len(detector.decided))

        output=datarx.DataRx(profile.rate,profile.baud,expanded=profile.shaping,
                             precoder=receiver.h)
        output.conv.d=drx.conv.d.copy();output.x_hist=drx.x_hist.copy()
        output.z=drx.z;output.m4=drx.m4;output.frame_pos=drx.frame_pos
        decoded=[]
        for offset in range(0,len(decoded_points),8):
            frame=offset//8
            states=detector.decided_states[4*frame:4*(frame+1)]
            decoded.extend(output.decode_frame(decoded_points[offset:offset+8],
                                               trellis_states=states)[0])
        self.assertGreater(sum(a!=b for a,b in zip(ordinary_bits,payload)),0)
        self.assertEqual(decoded,payload)

        erased_rx=copy.copy(receiver);erased_rx.drx=copy.deepcopy(drx)
        neutral=type(detector)(erased_rx,coefficients,seed_residuals,
                               final_state=final_state,depth=24)
        neutral.feed(0j,0j,erasure=True)
        self.assertEqual(neutral.erased_intervals,1)
        self.assertEqual(neutral.metric_offset,0.)
        self.assertEqual(len(neutral.hist),1)


if __name__=='__main__':unittest.main()
