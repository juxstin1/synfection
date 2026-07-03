//! STFT / mel features (torch-parity), multi-scale spectral loss, pitch detection.

use realfft::num_complex::Complex;
use realfft::RealFftPlanner;

/// Periodic hann window, same as torch.hann_window.
fn hann(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / n as f32).cos())
        .collect()
}

/// Reflect-pad index (torch 'reflect': edge not repeated).
#[inline]
fn reflect(i: isize, len: usize) -> usize {
    let len = len as isize;
    let mut i = i;
    if i < 0 {
        i = -i;
    }
    if i >= len {
        i = 2 * len - 2 - i;
    }
    i as usize
}

/// Magnitude STFT matching torch.stft(center=True, reflect pad, periodic hann).
/// Returns (mags, n_bins, n_frames), mags laid out [bin * n_frames + frame].
pub fn stft_mag(x: &[f32], nfft: usize, hop: usize) -> (Vec<f32>, usize, usize) {
    let t = x.len();
    let pad = nfft / 2;
    let n_frames = 1 + t / hop;
    let n_bins = nfft / 2 + 1;
    let win = hann(nfft);
    let mut planner = RealFftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(nfft);
    let mut frame = vec![0.0f32; nfft];
    let mut spec: Vec<Complex<f32>> = fft.make_output_vec();
    let mut mags = vec![0.0f32; n_bins * n_frames];
    for m in 0..n_frames {
        let start = m as isize * hop as isize - pad as isize;
        for j in 0..nfft {
            frame[j] = x[reflect(start + j as isize, t)] * win[j];
        }
        fft.process(&mut frame, &mut spec).unwrap();
        for (b, c) in spec.iter().enumerate() {
            mags[b * n_frames + m] = c.norm();
        }
    }
    (mags, n_bins, n_frames)
}

/// Log-mel features, exact port of synth.py `melspec` (n_fft 1024, hop 512).
/// `mel_fb` is the exported librosa filterbank (n_mels x 513).
/// Returns (mel, n_mels, n_frames), laid out [mel * n_frames + frame].
pub fn melspec(x: &[f32], mel_fb: &[f32], n_mels: usize) -> (Vec<f32>, usize, usize) {
    let (mags, n_bins, n_frames) = stft_mag(x, 1024, 512);
    let mut mel = vec![0.0f32; n_mels * n_frames];
    for mi in 0..n_mels {
        for fr in 0..n_frames {
            let mut acc = 0.0f32;
            for b in 0..n_bins {
                let w = mel_fb[mi * n_bins + b];
                if w != 0.0 {
                    let m = mags[b * n_frames + fr];
                    acc += w * m * m; // power
                }
            }
            mel[mi * n_frames + fr] = 10.0 * (acc + 1e-6).log10();
        }
    }
    let max = mel.iter().cloned().fold(f32::MIN, f32::max);
    for v in mel.iter_mut() {
        *v = ((*v - max + 80.0) / 80.0).clamp(0.0, 1.0);
    }
    (mel, n_mels, n_frames)
}

/// Multi-scale STFT loss (losses.py): mean |X|-|Y| + mean |log X - log Y|
/// over 5 FFT sizes. Same objective the net was trained on.
pub fn multiscale_stft(x: &[f32], y: &[f32]) -> f32 {
    const FFTS: [usize; 5] = [2048, 1024, 512, 256, 128];
    let mut total = 0.0f64;
    for nf in FFTS {
        let hop = nf / 4;
        let (mx, ..) = stft_mag(x, nf, hop);
        let (my, ..) = stft_mag(y, nf, hop);
        let mut lin = 0.0f64;
        let mut log = 0.0f64;
        for (a, b) in mx.iter().zip(&my) {
            lin += (a - b).abs() as f64;
            log += ((a + 1e-5).ln() - (b + 1e-5).ln()).abs() as f64;
        }
        total += (lin + log) / mx.len() as f64;
    }
    (total / FFTS.len() as f64) as f32
}

/// YIN-style pitch detection on the loudest window; returns a MIDI note.
/// Stand-in for librosa pyin — plenty for mono bass/lead hooks.
pub fn detect_midi(x: &[f32], sr: f32) -> i32 {
    let w = 4096.min(x.len());
    // loudest window
    let hop = 1024;
    let mut best = (0.0f32, 0usize);
    let mut s = 0;
    while s + w <= x.len() {
        let e: f32 = x[s..s + w].iter().map(|v| v * v).sum();
        if e > best.0 {
            best = (e, s);
        }
        s += hop;
    }
    let seg = &x[best.1..best.1 + w];
    let tau_min = (sr / 2000.0) as usize;
    let tau_max = ((sr / 40.0) as usize).min(w / 2);
    if tau_max <= tau_min + 2 {
        return 48;
    }
    let span = w - tau_max;
    let mut d = vec![0.0f32; tau_max + 1];
    for tau in 1..=tau_max {
        let mut acc = 0.0f32;
        for i in 0..span {
            let diff = seg[i] - seg[i + tau];
            acc += diff * diff;
        }
        d[tau] = acc;
    }
    // cumulative-mean-normalized difference
    let mut cmnd = vec![1.0f32; tau_max + 1];
    let mut cum = 0.0f32;
    for tau in 1..=tau_max {
        cum += d[tau];
        cmnd[tau] = if cum > 0.0 { d[tau] * tau as f32 / cum } else { 1.0 };
    }
    let mut tau_pick = 0;
    for tau in tau_min..=tau_max {
        if cmnd[tau] < 0.15 {
            let mut t = tau;
            while t + 1 <= tau_max && cmnd[t + 1] < cmnd[t] {
                t += 1;
            }
            tau_pick = t;
            break;
        }
    }
    if tau_pick == 0 {
        // no dip under threshold -> global min
        tau_pick = (tau_min..=tau_max)
            .min_by(|&a, &b| cmnd[a].partial_cmp(&cmnd[b]).unwrap())
            .unwrap_or(0);
        if tau_pick == 0 || cmnd[tau_pick] > 0.6 {
            return 48; // unvoiced-ish: match.py's fallback
        }
    }
    // parabolic refinement
    let t = tau_pick;
    let f = if t > 1 && t < tau_max {
        let (a, b, c) = (cmnd[t - 1], cmnd[t], cmnd[t + 1]);
        let denom = a - 2.0 * b + c;
        let delta = if denom.abs() > 1e-9 { 0.5 * (a - c) / denom } else { 0.0 };
        sr / (t as f32 + delta.clamp(-1.0, 1.0))
    } else {
        sr / t as f32
    };
    (69.0 + 12.0 * (f / 440.0).log2()).round() as i32
}

/// Unison thickener: 4 detuned voices (±depth cents) with small decorrelating
/// offsets, wrap-around so looped buffers stay seamless. `amount` 0..1.
pub fn thicken(x: &[f32], sr: f32, amount: f32) -> Vec<f32> {
    if amount < 0.01 || x.len() < 64 {
        return x.to_vec();
    }
    let depth = 5.0 + 25.0 * amount; // cents at full spread
    let gain = 0.18 + 0.45 * amount; // per-voice level
    let mut out = x.to_vec();
    for (k, c) in [-1.0f32, -0.45, 0.45, 1.0].iter().enumerate() {
        let ratio = 2.0f32.powf(c * depth / 1200.0);
        let offs = (sr * 0.004 * (k as f32 + 1.0)) as usize;
        for i in 0..x.len() {
            let pos = i as f32 * ratio;
            let j = pos as usize;
            if j + 1 >= x.len() {
                break;
            }
            let frac = pos - j as f32;
            let s = x[j] * (1.0 - frac) + x[j + 1] * frac;
            out[(i + offs) % x.len()] += s * gain;
        }
    }
    out
}

const CEILING: f32 = 0.9; // peak ceiling
const RMS_CAP: f32 = 0.25; // loudness guard (~-12 dBFS) — screech protection
const KNEE: f32 = 0.75; // soft-clip knee start

/// Built-in output safety: peak normalize -> loudness guard -> soft-knee
/// ceiling -> click-killing edge fades (one-shots only; loops stay seamless).
/// Guarantees nothing played or saved can clip or blast sustained loudness.
pub fn safety(x: &mut [f32], sr: f32, looped: bool) {
    if x.is_empty() {
        return;
    }
    // subsonic guard first: one-pole DC/rumble blocker at ~25 Hz so an 8 Hz
    // sub can't steal the normalization headroom. For loops, warm the filter
    // state on the tail so the seam stays continuous.
    let a = 1.0 - std::f32::consts::TAU * 25.0 / sr;
    let (mut px, mut py) = (0.0f32, 0.0f32);
    if looped {
        for &v in &x[x.len().saturating_sub(8192)..] {
            let y = v - px + a * py;
            px = v;
            py = y;
        }
    }
    for v in x.iter_mut() {
        let y = *v - px + a * py;
        px = *v;
        py = y;
        *v = y;
    }
    let peak = x.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    if peak > CEILING {
        let s = CEILING / peak;
        x.iter_mut().for_each(|v| *v *= s);
    }
    let rms = (x.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>() / x.len() as f64).sqrt() as f32;
    if rms > RMS_CAP {
        let s = RMS_CAP / rms;
        x.iter_mut().for_each(|v| *v *= s);
    }
    let span = CEILING - KNEE;
    for v in x.iter_mut() {
        let a = v.abs();
        if a > KNEE {
            *v = v.signum() * (KNEE + ((a - KNEE) / span).tanh() * span);
        }
    }
    if !looped {
        let n = (sr * 0.003) as usize;
        let n = n.min(x.len() / 2).max(1);
        for i in 0..n {
            let g = i as f32 / n as f32;
            x[i] *= g;
            let last = x.len() - 1 - i;
            x[last] *= g;
        }
    }
}

/// Linear resample (fine for feature extraction of mono hooks).
pub fn resample(x: &[f32], from: f32, to: f32) -> Vec<f32> {
    if (from - to).abs() < 1e-3 {
        return x.to_vec();
    }
    let out_len = ((x.len() as f64) * (to as f64) / (from as f64)) as usize;
    (0..out_len)
        .map(|i| {
            let pos = i as f64 * from as f64 / to as f64;
            let j = pos.floor() as usize;
            let frac = (pos - j as f64) as f32;
            let a = x[j.min(x.len() - 1)];
            let b = x[(j + 1).min(x.len() - 1)];
            a + (b - a) * frac
        })
        .collect()
}
