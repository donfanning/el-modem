"""Causal PCM impairments for interactive independent-modem validation."""
import numpy as np
from v34_channel_controls import bandlimited_sample


class ImpairmentBufferError(RuntimeError):
    """The test transport cannot provide the requested sampling positions."""


class StreamImpairment:
    """Clock warp, carrier offset, noise, wander, slip, phase jitter and gain hit.

    Fixed 1024-sample latency covers sinc lookahead and +/-500 ppm over 180s.
    Input/output share a sample index; no handshake or payload oracle is used.
    """
    def __init__(self,clock_ppm=0.,carrier_hz=0.,noise_rms=0.,wander_samples=0.,slip_sample=None,seed=20261004,active=True,
                 *,phase_jitter_deg=0.,phase_jitter_hz=0.,gain_hit_db=0.):
        if abs(clock_ppm)>500 or abs(carrier_hz)>10 or noise_rms<0 or abs(wander_samples)>2:
            raise ValueError('impairment outside validated bounds')
        if not all(np.isfinite(v) for v in (phase_jitter_deg,phase_jitter_hz,gain_hit_db)) or \
                not 0<=phase_jitter_deg<=10 or not 0<=phase_jitter_hz<=300 or abs(gain_hit_db)>3 or \
                (phase_jitter_deg and not phase_jitter_hz):
            raise ValueError('jitter/gain impairment outside declared bounds')
        self.clock_ppm=clock_ppm;self.carrier_hz=carrier_hz;self.noise_rms=noise_rms
        self.wander_samples=wander_samples;self.slip_sample=slip_sample;self.seed=seed
        self.rng=np.random.default_rng(seed);self.index=0;self.offset=0;self.history=np.array([],float)
        self.raw_tail=np.zeros(126);self.latency=1024
        self.active=active;self.activation_sample=0 if active else None
        self._delayed_activation=False
        self.phase_jitter_deg=phase_jitter_deg;self.phase_jitter_hz=phase_jitter_hz
        self.gain_hit_db=gain_hit_db;self.clipped_samples=0
        k=np.arange(-63,64);h=np.zeros(127);odd=k%2!=0
        h[odd]=2/(np.pi*k[odd]);self.hilbert=h*np.hamming(127)

    def activate(self):
        """Change slope at the next output sample without changing position."""
        if not self.active:
            self.active=True;self.activation_sample=self.index;self._delayed_activation=True

    def positions(self,n):
        n=np.asarray(n,float)
        if not self.active:return n-self.latency
        ratio=1/(1+self.clock_ppm*1e-6)
        if self._delayed_activation:
            elapsed=n-self.activation_sample
            pos=n-self.latency+elapsed*(ratio-1)
        else:
            elapsed=n;pos=(n-self.latency)*ratio
        pos-=self.wander_samples*np.sin(2*np.pi*elapsed/(8000*2))
        if self.slip_sample is not None:pos-=n>=self.slip_sample
        return pos

    def process(self,pcm):
        pcm=np.asarray(pcm,float);n=np.arange(self.index,self.index+len(pcm))
        if not len(pcm):return np.array([],np.int16)
        if self.carrier_hz or self.phase_jitter_hz:
            joined=np.r_[self.raw_tail,pcm]
            q=np.convolve(joined,self.hilbert,mode='valid')
            delayed=joined[63:63+len(pcm)]
            self.raw_tail=joined[-126:].copy()
            # Retain the Hilbert group's delay before activation so switching
            # carrier translation cannot introduce a sampling-position jump.
            elapsed=n-self.activation_sample if self._delayed_activation else n-63
            phase=2*np.pi*self.carrier_hz*elapsed/8000 if self.active else np.zeros(len(n))
            if self.active and self.phase_jitter_deg:
                phase+=np.deg2rad(self.phase_jitter_deg)*np.sin(2*np.pi*self.phase_jitter_hz*elapsed/8000)
            pcm=delayed*np.cos(phase)-q*np.sin(phase)
        self.history=np.r_[self.history,pcm]
        positions=self.positions(n)
        valid=positions>=0
        if np.any(positions[valid]-self.offset<32) and self.offset:
            raise ImpairmentBufferError('impairment history exhausted')
        if np.any(positions[valid]>self.index+len(pcm)-33):
            raise ImpairmentBufferError('clock warp exhausted causal lookahead')
        out=np.zeros(len(pcm))
        out[valid]=bandlimited_sample(self.history,positions[valid]-self.offset)
        if self.gain_hit_db and self.active:out*=10**(self.gain_hit_db/20)
        if self.noise_rms and self.active:out+=self.rng.normal(0,self.noise_rms,len(out))
        self.index+=len(pcm)
        retain=max(0,int(np.floor(positions[-1]))-96-self.offset)
        if retain:self.history=self.history[retain:].copy();self.offset+=retain
        rounded=np.rint(out)
        self.clipped_samples+=int(np.count_nonzero((rounded<-32768)|(rounded>32767)))
        return np.clip(rounded,-32768,32767).astype(np.int16)

    def provenance(self):
        return dict(clock_ppm=self.clock_ppm,carrier_hz=self.carrier_hz,noise_rms=self.noise_rms,
            wander_samples=self.wander_samples,slip_sample=self.slip_sample,seed=self.seed,
            latency_samples=self.latency+(63 if self.carrier_hz or self.phase_jitter_hz else 0),uses_expected_bits=False,
            active=self.active,activation_sample=self.activation_sample,
            continuous_delayed_activation=self._delayed_activation,
            phase_jitter_deg=self.phase_jitter_deg,phase_jitter_hz=self.phase_jitter_hz,
            gain_hit_db=self.gain_hit_db,clipped_samples=self.clipped_samples,
            phase_output_activation_sample=(self.activation_sample+self.latency
                if self.activation_sample is not None and self.phase_jitter_hz else None),
            gain_output_activation_sample=self.activation_sample if self.gain_hit_db else None)
