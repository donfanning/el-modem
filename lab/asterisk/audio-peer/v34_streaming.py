"""Bounded data-phase PCM storage with unchanged absolute receive coordinates."""
import numpy as np

KEEP_SAMPLES=16000
TRIM_BLOCK=4096


def update_data_baseband(engine, receiver, source_offset):
    origin=getattr(engine,'_rx_origin',0)
    # Offline replay tools may explicitly replace the buffer with full PCM.
    if origin and len(engine._rx)==engine.rx_t:
        engine._rx=engine._rx[origin:]
    start=max(0,source_offset-origin)
    local_origin=max(0,origin-source_offset)
    receiver._pcm_source_offset=source_offset
    receiver._z=receiver.update_baseband(engine._rx[start:],origin=local_origin)
    guard=getattr(getattr(getattr(engine,'_p4rcv',None),'eq',None),'_transport_guard',None)
    if guard is not None:guard.prune(source_offset+local_origin)


def trim_data_pcm(engine, eq, source_offset, *, tracking=False):
    """Retain two seconds behind the consumed window, trimming in blocks.

    Handshake and B1 collection retain original buffers. Once qualified, sample
    positions/oscillator epochs remain absolute and no future context is lost.
    Connected MP tracking can explicitly trim already-consumed input; B1
    collection keeps its complete window by leaving tracking=False.
    """
    if not tracking and not getattr(engine,'_rx_state',None):return
    origin=getattr(engine,'_rx_origin',0)
    consumed=source_offset+eq.R.lag+(eq.R.LEAD+eq.k+eq.R.delay)*eq.R.spl+eq.dt
    first=max(source_offset,int(consumed)-KEEP_SAMPLES)
    drop=(first-origin)//TRIM_BLOCK*TRIM_BLOCK
    if drop<=0 or drop>=len(engine._rx):return
    engine._rx=engine._rx[drop:].copy()
    engine._rx_origin=origin+drop
    engine._rx_retained_peak=max(getattr(engine,'_rx_retained_peak',0),len(engine._rx))
