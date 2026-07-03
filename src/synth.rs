//! The differentiable synth engine, ported 1:1 from synth.py (v2, 16-param).
//! Two wavetable oscillators + sub + shaped noise -> time-varying resonant
//! lowpass (2-pole magnitude per harmonic) -> ADSR amp/filter envelopes -> drive.

use rand::rngs::SmallRng;
use rand::Rng;
use realfft::RealFftPlanner;

use crate::genome::{denorm, Genome};

pub const SR: f32 = 22050.0;
pub const DUR: f32 = 1.2;
pub const N: usize = (SR * DUR) as usize; // 26460
pub const NOTE_DUR: f32 = 0.85;
pub const MAX_HARM: usize = 72;
const N_WT_FRAMES: usize = 8;

pub fn midi_to_hz(m: f32) -> f32 {
    440.0 * 2.0f32.powf((m - 69.0) / 12.0)
}

/// 8-frame harmonic-amplitude wavetable (sine, tri, square, saw, 25% pulse,
/// narrow pulse, formant, rich), each frame L1-normalized. Same as synth.py.
fn wavetable() -> [[f32; MAX_HARM]; N_WT_FRAMES] {
    let mut wt = [[0.0f64; MAX_HARM]; N_WT_FRAMES];
    for h in 0..MAX_HARM {
        let k = (h + 1) as f64;
        let odd = (h + 1) % 2 == 1;
        wt[0][h] = if h == 0 { 1.0 } else { 0.0 };
        wt[1][h] = if odd { 1.0 / (k * k) } else { 0.0 };
        wt[2][h] = if odd { 1.0 / k } else { 0.0 };
        wt[3][h] = 1.0 / k;
        wt[4][h] = (std::f64::consts::PI * k * 0.25).sin().abs() * 2.0 / (std::f64::consts::PI * k);
        wt[5][h] = (std::f64::consts::PI * k * 0.10).sin().abs() * 2.0 / (std::f64::consts::PI * k);
        wt[6][h] = (-((k - 5.0) * (k - 5.0)) / 8.0).exp()
            + 0.3 * (-((k - 12.0) * (k - 12.0)) / 20.0).exp();
        wt[7][h] = 1.0 / k.sqrt();
    }
    let mut out = [[0.0f32; MAX_HARM]; N_WT_FRAMES];
    for f in 0..N_WT_FRAMES {
        let sum: f64 = wt[f].iter().map(|x| x.abs()).sum::<f64>() + 1e-9;
        for h in 0..MAX_HARM {
            out[f][h] = (wt[f][h] / sum) as f32;
        }
    }
    out
}

fn wt_profile(pos: f32) -> [f32; MAX_HARM] {
    // rebuilt per call; trivial cost next to the render loop
    let wt = wavetable();
    let f = pos.clamp(0.0, 1.0) * (N_WT_FRAMES - 1) as f32;
    let lo = (f.floor() as usize).min(N_WT_FRAMES - 2);
    let frac = f - lo as f32;
    let mut out = [0.0f32; MAX_HARM];
    for h in 0..MAX_HARM {
        out[h] = (1.0 - frac) * wt[lo][h] + frac * wt[lo + 1][h];
    }
    out
}

/// Vectorized ADSR, same shape as synth.py's `_adsr`.
fn adsr(a: f32, d: f32, s: f32, r: f32, t_len: usize, sr: f32, note_dur: f32) -> Vec<f32> {
    let a_n = (a * sr).max(1.0);
    let d_n = (d * sr).max(1.0);
    let r_n = (r * sr).max(1.0);
    let note_off = ((note_dur * sr) as usize).min(t_len - 1);
    let mut env = vec![0.0f32; t_len];
    let e_on = |t: f32| -> f32 {
        let v = if t < a_n {
            t / a_n
        } else if t < a_n + d_n {
            1.0 + (s - 1.0) * (t - a_n) / d_n
        } else {
            s
        };
        v.clamp(0.0, 1.0)
    };
    let level = e_on(note_off as f32);
    for (i, e) in env.iter_mut().enumerate() {
        let t = i as f32;
        *e = if i < note_off {
            e_on(t)
        } else {
            (level * (1.0 - (t - note_off as f32) / r_n)).max(0.0)
        }
        .clamp(0.0, 1.0);
    }
    env
}

/// Analog 2-pole lowpass magnitude at freq f, cutoff fc, resonance q.
#[inline]
fn filter_mag(f: f32, fc: f32, q: f32) -> f32 {
    let x = f / fc.max(1e-3);
    (1.0 / ((1.0 - x * x).powi(2) + (x / q) * (x / q) + 1e-9).sqrt()).min(12.0)
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Spectrally-shaped noise: white gaussian -> rFFT -> filter magnitude -> irFFT.
fn shaped_noise(t_len: usize, fc: f32, q: f32, sr: f32, rng: &mut SmallRng) -> Vec<f32> {
    let mut w: Vec<f32> = Vec::with_capacity(t_len);
    while w.len() < t_len {
        // Box-Muller
        let u1: f32 = rng.gen_range(f32::EPSILON..1.0);
        let u2: f32 = rng.gen_range(0.0..1.0);
        let r = (-2.0 * u1.ln()).sqrt();
        let th = 2.0 * std::f32::consts::PI * u2;
        w.push(r * th.cos());
        if w.len() < t_len {
            w.push(r * th.sin());
        }
    }
    let mut planner = RealFftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(t_len);
    let ifft = planner.plan_fft_inverse(t_len);
    let mut spec = fft.make_output_vec();
    fft.process(&mut w, &mut spec).unwrap();
    for (i, c) in spec.iter_mut().enumerate() {
        let f = i as f32 * sr / t_len as f32;
        *c *= filter_mag(f, fc, q);
    }
    let mut out = ifft.make_output_vec();
    ifft.process(&mut spec, &mut out).unwrap();
    // realfft's inverse is unnormalized (scale n); peak-normalize like synth.py
    let peak = out.iter().fold(0.0f32, |m, x| m.max(x.abs())) + 1e-6;
    for x in out.iter_mut() {
        *x /= peak;
    }
    out
}

/// Genome + MIDI note -> audio in [-1,1]. Port of synth.py `render`.
pub fn render(g: &Genome, midi: f32, sr: f32, n: usize, note_dur: f32, rng: &mut SmallRng) -> Vec<f32> {
    let p = denorm(g);
    let [osc1_wt, osc2_wt, osc2_detune, osc_mix, sub_level, noise_level, drive, cutoff, reso, filt_env, filt_a, filt_d, amp_a, amp_d, amp_s, amp_r, pitch_env, pitch_dec, lfo_rate, lfo_depth] =
        p;

    let f0 = midi_to_hz(midi);
    let nyq = sr * 0.49;

    let amp = adsr(amp_a, amp_d, amp_s, amp_r, n, sr, note_dur);
    let top = cutoff + filt_env * (nyq - cutoff);
    let fe = adsr(filt_a, filt_d, 0.25, amp_r, n, sr, note_dur);
    let mut cutoff_curve: Vec<f32> = fe.iter().map(|e| cutoff + (top - cutoff) * e).collect();
    if lfo_depth > 1e-6 {
        let om = 2.0 * std::f32::consts::PI * lfo_rate / sr;
        for (i, c) in cutoff_curve.iter_mut().enumerate() {
            *c = (*c * 2.0f32.powf(lfo_depth * (om * i as f32).sin())).clamp(20.0, nyq);
        }
    }

    // pitch envelope: start `pitch_env` semitones above the note, decay to it.
    // Phase is the exclusive prefix-sum of the swept fundamental, accumulated
    // in f64 (mirrors torch's cumsum) — degenerates to the old static phase
    // exactly when pitch_env is 0. Per-harmonic filter/anti-alias stay keyed
    // to the target frequency (the sweep is fast; this matches synth.py).
    let ratio: Vec<f32> = (0..n)
        .map(|i| 2.0f32.powf(pitch_env * (-(i as f32) / sr / pitch_dec).exp() / 12.0))
        .collect();
    let mut ph = vec![0.0f64; n];
    let mut acc = 0.0f64;
    for i in 0..n {
        ph[i] = acc;
        acc += f0 as f64 * ratio[i] as f64;
    }
    let pscale = 2.0 * std::f64::consts::PI / sr as f64;

    let mut sig = vec![0.0f32; n];
    let mut osc = |fund: f32, wt_pos: f32, weight: f32| {
        if weight == 0.0 {
            return;
        }
        let prof = wt_profile(wt_pos);
        let rel = (fund / f0) as f64;
        for k in 1..=MAX_HARM {
            let fk = fund * k as f32;
            let aa = sigmoid((nyq - fk) / (0.02 * nyq + 1.0));
            let pk = prof[k - 1] * aa;
            if pk.abs() < 1e-8 {
                continue;
            }
            for (i, s) in sig.iter_mut().enumerate() {
                let mag = filter_mag(fk, cutoff_curve[i], reso);
                let phase = (ph[i] * pscale * rel) as f32;
                *s += weight * pk * mag * (k as f32 * phase).sin();
            }
        }
    };
    osc(f0, osc1_wt, 1.0 - osc_mix);
    let f2 = f0 * 2.0f32.powf(osc2_detune / 1200.0);
    osc(f2, osc2_wt, osc_mix);

    let fsub = f0 / 2.0;
    for (i, s) in sig.iter_mut().enumerate() {
        let phase = (ph[i] * pscale * 0.5) as f32;
        let sub = phase.sin() * filter_mag(fsub, cutoff_curve[i], reso);
        *s += 0.6 * sub_level * sub;
    }

    if noise_level > 1e-6 {
        let nz = shaped_noise(n, top, reso, sr, rng);
        for (s, z) in sig.iter_mut().zip(nz) {
            *s += noise_level * z;
        }
    }

    for (s, a) in sig.iter_mut().zip(&amp) {
        *s *= a;
    }

    // drive / waveshaper: blend clean <-> tanh-saturated (clean at drive=0)
    let norm = sig.iter().fold(0.0f32, |m, x| m.max(x.abs())) + 1e-6;
    let tanh6 = 6.0f32.tanh();
    for s in sig.iter_mut() {
        let clean = *s / norm;
        let shaped = (clean * 6.0).tanh() / tanh6;
        *s = (1.0 - drive) * clean + drive * shaped;
    }
    let peak = sig.iter().fold(0.0f32, |m, x| m.max(x.abs())) + 1e-6;
    for s in sig.iter_mut() {
        *s = *s / peak * 0.9;
    }
    sig
}

/// Render at the engine's native SR/length (what the matcher and net expect).
pub fn render_default(g: &Genome, midi: f32, rng: &mut SmallRng) -> Vec<f32> {
    render(g, midi, SR, N, NOTE_DUR, rng)
}
