//! The garden: archetype priors + offspring growing, scored by the embedded
//! RLHF reward model. Ported from genpatches.py's ARCHETYPES (v2 genome).

use rand::rngs::SmallRng;
use rand::Rng;

use crate::genome::{Genome, N_PARAMS};
use crate::matcher::gaussian;
use crate::net::Net;
use crate::synth;

pub const ARCHETYPE_NAMES: [&str; 7] =
    ["bass", "reese", "lead", "pluck", "stab", "pad", "keys"];

/// (lo, hi) windows on the normalized genome, PARAMS order:
/// osc1_wt, osc2_wt, osc2_detune, osc_mix, sub_level, noise_level, drive,
/// cutoff, reso, filt_env, filt_a, filt_d, amp_a, amp_d, amp_s, amp_r
pub fn windows(arch: &str) -> [(f32, f32); N_PARAMS] {
    match arch {
        "bass" => [(0.3, 0.55), (0.0, 0.5), (0.46, 0.54), (0.2, 0.6), (0.5, 1.0), (0.0, 0.1),
                   (0.0, 0.35), (0.22, 0.5), (0.1, 0.5), (0.2, 0.6), (0.0, 0.1), (0.3, 0.7),
                   (0.0, 0.06), (0.3, 0.7), (0.5, 1.0), (0.05, 0.4),
                   (0.0, 0.06), (0.3, 0.7), (0.2, 0.6), (0.0, 0.06)],
        "reese" => [(0.3, 0.55), (0.3, 0.55), (0.7, 0.95), (0.4, 0.6), (0.3, 0.8), (0.0, 0.15),
                    (0.2, 0.7), (0.3, 0.6), (0.45, 0.85), (0.2, 0.6), (0.0, 0.15), (0.25, 0.7),
                    (0.0, 0.08), (0.4, 0.9), (0.5, 1.0), (0.1, 0.5),
                   (0.0, 0.06), (0.3, 0.7), (0.2, 0.6), (0.0, 0.06)],
        "lead" => [(0.4, 0.9), (0.3, 0.8), (0.5, 0.62), (0.3, 0.7), (0.0, 0.3), (0.0, 0.1),
                   (0.0, 0.4), (0.55, 0.9), (0.15, 0.55), (0.1, 0.5), (0.0, 0.2), (0.2, 0.7),
                   (0.02, 0.25), (0.3, 0.8), (0.6, 1.0), (0.1, 0.5),
                   (0.0, 0.06), (0.3, 0.7), (0.2, 0.6), (0.0, 0.06)],
        "pluck" => [(0.2, 0.6), (0.2, 0.6), (0.47, 0.6), (0.2, 0.7), (0.0, 0.4), (0.0, 0.12),
                    (0.0, 0.3), (0.4, 0.75), (0.3, 0.8), (0.5, 0.95), (0.0, 0.05), (0.05, 0.3),
                    (0.0, 0.04), (0.1, 0.35), (0.0, 0.25), (0.05, 0.3),
                   (0.0, 0.06), (0.3, 0.7), (0.2, 0.6), (0.0, 0.06)],
        "stab" => [(0.3, 0.75), (0.3, 0.75), (0.45, 0.62), (0.3, 0.7), (0.1, 0.5), (0.0, 0.12),
                   (0.1, 0.5), (0.45, 0.75), (0.25, 0.7), (0.35, 0.8), (0.0, 0.06), (0.12, 0.4),
                   (0.0, 0.05), (0.15, 0.45), (0.15, 0.5), (0.08, 0.4),
                   (0.0, 0.06), (0.3, 0.7), (0.2, 0.6), (0.0, 0.06)],
        "pad" => [(0.05, 0.45), (0.05, 0.45), (0.55, 0.75), (0.35, 0.65), (0.1, 0.5), (0.0, 0.15),
                  (0.0, 0.2), (0.4, 0.7), (0.1, 0.45), (0.1, 0.5), (0.3, 0.7), (0.3, 0.8),
                  (0.35, 0.75), (0.4, 0.9), (0.6, 1.0), (0.4, 0.8),
                   (0.0, 0.06), (0.3, 0.7), (0.2, 0.6), (0.0, 0.06)],
        _ => [(0.1, 0.6), (0.1, 0.6), (0.48, 0.58), (0.3, 0.7), (0.1, 0.5), (0.0, 0.08),
              (0.0, 0.25), (0.45, 0.8), (0.15, 0.5), (0.2, 0.6), (0.0, 0.15), (0.2, 0.6),
              (0.0, 0.1), (0.25, 0.7), (0.4, 0.85), (0.1, 0.5),
                   (0.0, 0.06), (0.3, 0.7), (0.2, 0.6), (0.0, 0.06)], // keys
    }
}

/// A note that suits the archetype (midpoint of genpatches' NOTE_RANGE).
pub fn home_note(arch: &str) -> i32 {
    match arch {
        "bass" => 38,
        "reese" => 39,
        "lead" => 62,
        "pluck" => 56,
        "stab" => 52,
        "pad" => 53,
        _ => 56, // keys
    }
}

pub fn sample_archetype(arch: &str, rng: &mut SmallRng) -> Genome {
    let w = windows(arch);
    let mut g = [0.0f32; N_PARAMS];
    for (i, v) in g.iter_mut().enumerate() {
        *v = w[i].0 + (w[i].1 - w[i].0) * rng.gen::<f32>();
    }
    g
}

/// Listenability flags for a rendered buffer. Renders are peak-normalized
/// upstream, so a bad patch is never *quiet* — it's a click, a fizz, a
/// resonant whistle, or subsonic rumble at full scale. This hears those.
#[derive(Default, Clone, Copy)]
pub struct Vet {
    pub dud: bool,
    pub tick: bool,    // all the energy in < 60 ms
    pub harsh: bool,   // 2-5 kHz fizz dominates the spectrum
    pub screech: bool, // one high bin carries most frames (resonance whistle)
    pub rumble: bool,  // subsonic energy share
}

impl Vet {
    pub fn bad(&self) -> bool {
        self.dud || self.tick || self.harsh || self.screech || self.rumble
    }
}

/// Cheap audio-feature check on an already-rendered buffer. Generated sounds
/// (random, grow) are rejected on any flag; manual edits are never blocked.
/// Thresholds are calibrated so all shipped presets pass (see test below).
pub fn vet(audio: &[f32], sr: f32) -> Vet {
    let mut v = Vet::default();
    if audio.is_empty() {
        v.dud = true;
        return v;
    }
    let n = audio.len() as f32;
    let rms = (audio.iter().map(|x| x * x).sum::<f32>() / n).sqrt();
    let active_n = audio.iter().filter(|x| x.abs() > 0.05).count();
    v.dud = rms < 0.03 || (active_n as f32 / n) < 0.03;
    v.tick = (active_n as f32 / sr) < 0.06;

    // subsonic rumble: energy share below ~20 Hz. Two cascaded one-poles
    // (12 dB/oct) so a legit 27-35 Hz DJ sub passes but 8 Hz garbage doesn't.
    let a = 1.0 - std::f32::consts::TAU * 20.0 / sr;
    let (mut lp1, mut lp2, mut e_lp, mut e_x) = (0.0f32, 0.0f32, 0.0f64, 0.0f64);
    for &x in audio {
        lp1 = a * lp1 + (1.0 - a) * x;
        lp2 = a * lp2 + (1.0 - a) * lp1;
        e_lp += (lp2 * lp2) as f64;
        e_x += (x * x) as f64;
    }
    v.rumble = e_x > 0.0 && e_lp / e_x > 0.25;

    // spectral: harshness ratio + resonant-whistle dominance
    let (mags, n_bins, n_frames) = crate::dsp::stft_mag(audio, 1024, 256);
    let hz_per_bin = sr / 1024.0;
    let b80 = (80.0 / hz_per_bin).round() as usize;
    let b2k = (2000.0 / hz_per_bin).round() as usize;
    let b5k = ((5000.0 / hz_per_bin).round() as usize).min(n_bins - 1);
    let powers: Vec<f64> = (0..n_frames)
        .map(|m| (0..n_bins).map(|b| { let x = mags[b * n_frames + m] as f64; x * x }).sum())
        .collect();
    let max_p = powers.iter().cloned().fold(0.0, f64::max);
    let act: Vec<usize> = (0..n_frames).filter(|&m| powers[m] > max_p * 1e-4).collect();
    if !act.is_empty() && max_p > 0.0 {
        let (mut harsh_e, mut full_e) = (0.0f64, 0.0f64);
        let mut whistle = 0usize;
        for &m in &act {
            let (mut best_b, mut best_p) = (0usize, 0.0f64);
            for b in 0..n_bins {
                let x = mags[b * n_frames + m] as f64;
                let p = x * x;
                if b >= b80 {
                    full_e += p;
                    if (b2k..=b5k).contains(&b) {
                        harsh_e += p;
                    }
                }
                if p > best_p {
                    best_p = p;
                    best_b = b;
                }
            }
            if best_p / powers[m] > 0.5 && best_b as f32 * hz_per_bin > 2500.0 {
                whistle += 1;
            }
        }
        let harsh_ratio = if full_e > 0.0 { harsh_e / full_e } else { 0.0 };
        v.harsh = harsh_ratio > 0.55 || (rms > 0.55 && harsh_ratio > 0.4);
        v.screech = whistle * 2 > act.len();
    }
    v
}

/// Post-generation hygiene: snap the detune dead-zone (±2¢ slow beating reads
/// as "out of tune") and floor the amp decay when sustain is near zero so a
/// generated pluck can't collapse into a tick. Generated genomes only — never
/// applied to live slider values.
pub fn tidy(g: &mut Genome) {
    if (g[2] - 0.5).abs() < 0.04 {
        g[2] = 0.5; // osc2_detune
    }
    if g[14] < 0.1 && g[13] < 0.26 {
        g[13] = 0.26; // amp_s ≈ 0 → amp_d ≥ ~50 ms
    }
}

/// Predicted quality in [0,1] from the embedded reward model (0.5 if absent).
pub fn score(net: &Net, g: &Genome) -> f32 {
    net.reward(g).unwrap_or(0.5)
}

pub struct Seedling {
    pub genome: Genome,
    pub score: f32,
}

/// Grow a generation: mutants of `seed`, or archetype samples if `arch` given.
/// Dud-filtered (one re-roll), reward-scored, sorted best-first.
pub fn grow(
    net: &Net,
    seed: &Genome,
    arch: Option<&str>,
    note: i32,
    n: usize,
    amount: f32,
    rng: &mut SmallRng,
) -> Vec<Seedling> {
    let mut out = Vec::with_capacity(n);
    let mut tries = 0;
    while out.len() < n && tries < n * 3 {
        tries += 1;
        let mut g = match arch {
            Some(a) => sample_archetype(a, rng),
            None => {
                let mut g = *seed;
                for v in g.iter_mut() {
                    // reflect at the walls: clamping makes repeated breeding
                    // saturate params at 0/1 (max reso/drive drift)
                    let mut x = *v + gaussian(rng) * amount;
                    while !(0.0..=1.0).contains(&x) {
                        x = if x < 0.0 { -x } else { 2.0 - x };
                    }
                    *v = x;
                }
                g
            }
        };
        tidy(&mut g);
        let audio = synth::render_default(&g, note as f32, rng);
        if vet(&audio, synth::SR).bad() {
            continue;
        }
        out.push(Seedling { genome: g, score: score(net, &g) });
    }
    out.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
    out
}

/// "Surprise me": mostly archetype-window samples (the DJ's own taste priors,
/// weighted toward his top families) plus two fully-random wildcards to keep
/// the discovery tail alive. Everything is vetted for listenability, then the
/// mixed pool is ranked by the reward model — the one regime where it's
/// reliable (archetype-level taste). If the whole pool scores poorly, roll one
/// more pool and take the overall best. Hard cap: never loop-until-good
/// (greedy reward pressure is the known reward-hack vector).
pub fn lucky_dip(net: &Net, note: i32, pool: usize, rng: &mut SmallRng) -> Option<Genome> {
    const DIE: [(&str, f32); 7] = [
        ("bass", 0.24), ("reese", 0.22), ("stab", 0.14), ("pluck", 0.12),
        ("lead", 0.10), ("keys", 0.10), ("pad", 0.08),
    ];
    let vet_note = note.clamp(28, 72) as f32; // vet in a sane band; play at the user's note
    let mut best: Option<(f32, Genome)> = None;
    for _round in 0..2 {
        for i in 0..pool {
            let mut g = if i + 2 < pool {
                let mut r = rng.gen::<f32>();
                let mut arch = DIE[0].0;
                for (a, w) in DIE {
                    if r < w {
                        arch = a;
                        break;
                    }
                    r -= w;
                }
                sample_archetype(arch, rng)
            } else {
                let mut g = [0.0f32; N_PARAMS];
                g.iter_mut().for_each(|v| *v = rng.gen());
                g
            };
            tidy(&mut g);
            let audio = synth::render_default(&g, vet_note, rng);
            if vet(&audio, synth::SR).bad() {
                continue;
            }
            let s = score(net, &g);
            if best.as_ref().map(|(bs, _)| s > *bs).unwrap_or(true) {
                best = Some((s, g));
            }
        }
        if best.as_ref().map(|(s, _)| *s >= 0.35).unwrap_or(false) {
            break;
        }
    }
    best.map(|(_, g)| g)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    /// Calibration anchor: every shipped preset must pass `vet` at its home
    /// note. If a preset trips a threshold, loosen the threshold — not the preset.
    #[test]
    fn all_presets_pass_vet() {
        for p in crate::presets::PRESETS.iter() {
            let mut rng = SmallRng::seed_from_u64(0);
            let a = synth::render_default(&p.genome, p.note as f32, &mut rng);
            let v = vet(&a, synth::SR);
            assert!(
                !v.bad(),
                "{}: dud={} tick={} harsh={} screech={} rumble={}",
                p.name, v.dud, v.tick, v.harsh, v.screech, v.rumble
            );
        }
    }

    /// The generators only emit vetted sounds.
    #[test]
    fn lucky_dip_and_grow_emit_listenable_patches() {
        let net = Net::load().expect("embedded weights");
        let mut rng = SmallRng::seed_from_u64(7);
        let g = lucky_dip(&net, 36, 12, &mut rng).expect("a pool of 24 should yield something");
        let a = synth::render_default(&g, 36.0, &mut rng);
        assert!(!vet(&a, synth::SR).bad());
        let kids = grow(&net, &g, None, 36, 8, 0.3, &mut rng);
        for k in &kids {
            let a = synth::render_default(&k.genome, 36.0, &mut rng);
            assert!(!vet(&a, synth::SR).bad());
        }
    }
}
