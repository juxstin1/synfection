//! Wav read/write shared by the CLI and the UI.

use anyhow::{bail, Context, Result};

use crate::dsp;
use crate::synth;

/// Load any wav, mono-ize, resample to the engine SR, pad/trim to N, peak 0.9.
pub fn load_target(path: &str) -> Result<Vec<f32>> {
    let mut reader = hound::WavReader::open(path).with_context(|| format!("open {path}"))?;
    let spec = reader.spec();
    let ch = spec.channels as usize;
    let raw: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().collect::<Result<_, _>>()?,
        hound::SampleFormat::Int => {
            let scale = 1.0 / (1i64 << (spec.bits_per_sample - 1)) as f32;
            reader.samples::<i32>().map(|s| s.map(|v| v as f32 * scale)).collect::<Result<_, _>>()?
        }
    };
    if raw.is_empty() {
        bail!("{path} is empty");
    }
    let mono: Vec<f32> = raw
        .chunks(ch)
        .map(|fr| fr.iter().sum::<f32>() / ch as f32)
        .collect();
    let mut x = dsp::resample(&mono, spec.sample_rate as f32, synth::SR);
    x.resize(synth::N, 0.0);
    let peak = x.iter().fold(0.0f32, |m, v| m.max(v.abs())) + 1e-9;
    for v in x.iter_mut() {
        *v = *v / peak * 0.9;
    }
    Ok(x)
}

pub fn write_wav(path: &str, audio: &[f32], sr: f32) -> Result<()> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: sr as u32,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut w = hound::WavWriter::create(path, spec).with_context(|| format!("create {path}"))?;
    for &v in audio {
        w.write_sample(v)?;
    }
    w.finalize()?;
    Ok(())
}
