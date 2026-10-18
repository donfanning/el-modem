use iax2::{Command, Config, Event, JitterBuffer, PbxClient, Pull};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::time::Instant;

const IAX_AUDIO_FRAME_MS: i32 = 20;

/// IAX2 timestamps are wall-clock millisecond values, but our media is always
/// 160-sample PCMU frames.  Scheduler jitter can make deltas 19 or 21 ms;
/// snap those to the nearest media slot before the exact-key jitter queue.
fn media_slot(origin: u32, timestamp: u32) -> u32 {
    let delta = timestamp.wrapping_sub(origin) as i32;
    let slot = (delta as f64 / IAX_AUDIO_FRAME_MS as f64).round() as i32;
    origin.wrapping_add((slot * IAX_AUDIO_FRAME_MS) as u32)
}

#[pyclass]
struct Client {
    inner: PbxClient,
}

#[pyclass]
struct MediaJitter {
    inner: JitterBuffer,
    origin: Option<u32>,
}

#[pymethods]
impl MediaJitter {
    #[new]
    fn new() -> Self { Self { inner: JitterBuffer::new(), origin: None } }

    fn reset(&mut self) { self.inner.reset(); self.origin = None; }

    fn push(&mut self, timestamp_ms: u32, ulaw: &[u8]) {
        let origin = *self.origin.get_or_insert(timestamp_ms);
        self.inner.push(media_slot(origin, timestamp_ms), ulaw.to_vec(), Instant::now());
    }

    fn pull(&mut self) -> (String, Vec<u8>) {
        match self.inner.pull() {
            Pull::Play(frame) => ("play".into(), frame),
            Pull::Conceal => ("conceal".into(), Vec::new()),
            Pull::Silence => ("silence".into(), Vec::new()),
        }
    }

    fn target_ms(&self) -> u32 { self.inner.stats().target_ms }

    fn stats(&self) -> (u64, u64, u64, u64, u32, usize) {
        let s = self.inner.stats();
        (s.played, s.concealed, s.late, s.overflow, s.target_ms, s.buffered)
    }
}

#[pymethods]
impl Client {
    #[new]
    fn new(name: String, username: String, secret: String) -> Self {
        Self { inner: PbxClient::new(Config::new(name, username, secret)) }
    }

    fn input(&mut self, datagram: &[u8]) {
        self.inner.handle_input(datagram, Instant::now());
    }

    fn tick(&mut self) {
        self.inner.handle_timeout(Instant::now());
    }

    fn command(&mut self, action: &str, value: Option<String>) -> PyResult<()> {
        let now = Instant::now();
        let cmd = match action {
            "dial" => Command::Dial { number: value.unwrap_or_default() },
            "answer" => Command::Answer { call: parse_call(value)? },
            "hangup" => Command::Hangup { call: parse_call(value)? },
            "dtmf" => {
                let digit = value.and_then(|v| v.chars().next()).ok_or_else(|| pyo3::exceptions::PyValueError::new_err("DTMF digit required"))?;
                Command::Dtmf { call: self.inner.call_ids().first().copied().unwrap_or(0), digit }
            }
            _ => return Err(pyo3::exceptions::PyValueError::new_err("unsupported IAX command")),
        };
        self.inner.handle_command(cmd, now);
        Ok(())
    }

    fn voice(&mut self, call: u16, ulaw: &[u8]) {
        self.inner.handle_command(Command::SendVoice { call, ulaw: ulaw.to_vec() }, Instant::now());
    }

    fn drain_tx(&mut self) -> Vec<Vec<u8>> {
        let mut out = Vec::new();
        while let Some(frame) = self.inner.poll_transmit() { out.push(frame); }
        out
    }

    fn drain_events(&mut self, py: Python<'_>) -> PyResult<Vec<Py<PyDict>>> {
        let mut out = Vec::new();
        while let Some(event) = self.inner.poll_event() {
            let row = PyDict::new(py);
            match event {
                Event::Registered => { row.set_item("type", "registered")?; }
                Event::RegisterLost { reason } => { row.set_item("type", "register_lost")?; row.set_item("reason", reason)?; }
                Event::Incoming { call, from, to } => { row.set_item("type", "incoming")?; row.set_item("call", call)?; row.set_item("from", from)?; row.set_item("to", to)?; }
                Event::Dialing { call, to } => { row.set_item("type", "dialing")?; row.set_item("call", call)?; row.set_item("to", to)?; }
                Event::Ringing { call } => { row.set_item("type", "ringing")?; row.set_item("call", call)?; }
                Event::Answered { call } => { row.set_item("type", "answered")?; row.set_item("call", call)?; }
                Event::Voice { call, ts, ulaw } => { row.set_item("type", "voice")?; row.set_item("call", call)?; row.set_item("ts", ts)?; row.set_item("ulaw", ulaw)?; }
                Event::Ended { call, reason } => { row.set_item("type", "ended")?; row.set_item("call", call)?; row.set_item("reason", reason)?; }
                Event::Dtmf { call, digit } => { row.set_item("type", "dtmf")?; row.set_item("call", call)?; row.set_item("digit", digit)?; }
                Event::Log(message) => { row.set_item("type", "log")?; row.set_item("message", message)?; }
                _ => continue,
            }
            out.push(row.unbind());
        }
        Ok(out)
    }
}

fn parse_call(value: Option<String>) -> PyResult<u16> {
    value.ok_or_else(|| pyo3::exceptions::PyValueError::new_err("call id required"))?
        .parse().map_err(|_| pyo3::exceptions::PyValueError::new_err("invalid call id"))
}

#[pymodule]
fn el_modem_iax(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<Client>()?;
    module.add_class::<MediaJitter>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::media_slot;

    #[test]
    fn snaps_wall_clock_rounding_to_fixed_pcm_media_slots() {
        let origin = 10_007;
        let timestamps = [10_007, 10_026, 10_047, 10_066, 10_087, 10_106];
        let normalized: Vec<u32> = timestamps.iter().map(|ts| media_slot(origin, *ts)).collect();
        assert_eq!(normalized, vec![10_007, 10_027, 10_047, 10_067, 10_087, 10_107]);
    }

    #[test]
    fn snapping_preserves_wraparound_and_reordering() {
        let origin = u32::MAX - 24;
        assert_eq!(media_slot(origin, origin.wrapping_add(19)), origin.wrapping_add(20));
        assert_eq!(media_slot(origin, origin.wrapping_add(41)), origin.wrapping_add(40));
        assert_eq!(media_slot(origin, origin.wrapping_add(19)), origin.wrapping_add(20));
    }
}
