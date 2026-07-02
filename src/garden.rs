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
                   (0.0, 0.06), (0.3, 0.7), (0.5, 1.0), (0.05, 0.4)],
        "reese" => [(0.3, 0.55), (0.3, 0.55), (0.7, 0.95), (0.4, 0.6), (0.3, 0.8), (0.0, 0.15),
                    (0.2, 0.7), (0.3, 0.6), (0.45, 0.85), (0.2, 0.6), (0.0, 0.15), (0.25, 0.7),
                    (0.0, 0.08), (0.4, 0.9), (0.5, 1.0), (0.1, 0.5)],
        "lead" => [(0.4, 0.9), (0.3, 0.8), (0.5, 0.62), (0.3, 0.7), (0.0, 0.3), (0.0, 0.1),
                   (0.0, 0.4), (0.55, 0.9), (0.15, 0.55), (0.1, 0.5), (0.0, 0.2), (0.2, 0.7),
                   (0.02, 0.25), (0.3, 0.8), (0.6, 1.0), (0.1, 0.5)],
        "pluck" => [(0.2, 0.6), (0.2, 0.6), (0.47, 0.6), (0.2, 0.7), (0.0, 0.4), (0.0, 0.12),
                    (0.0, 0.3), (0.4, 0.75), (0.3, 0.8), (0.5, 0.95), (0.0, 0.05), (0.05, 0.3),
                    (0.0, 0.04), (0.1, 0.35), (0.0, 0.25), (0.05, 0.3)],
        "stab" => [(0.3, 0.75), (0.3, 0.75), (0.45, 0.62), (0.3, 0.7), (0.1, 0.5), (0.0, 0.12),
                   (0.1, 0.5), (0.45, 0.75), (0.25, 0.7), (0.35, 0.8), (0.0, 0.06), (0.12, 0.4),
                   (0.0, 0.05), (0.15, 0.45), (0.15, 0.5), (0.08, 0.4)],
        "pad" => [(0.05, 0.45), (0.05, 0.45), (0.55, 0.75), (0.35, 0.65), (0.1, 0.5), (0.0, 0.15),
                  (0.0, 0.2), (0.4, 0.7), (0.1, 0.45), (0.1, 0.5), (0.3, 0.7), (0.3, 0.8),
                  (0.35, 0.75), (0.4, 0.9), (0.6, 1.0), (0.4, 0.8)],
        _ => [(0.1, 0.6), (0.1, 0.6), (0.48, 0.58), (0.3, 0.7), (0.1, 0.5), (0.0, 0.08),
              (0.0, 0.25), (0.45, 0.8), (0.15, 0.5), (0.2, 0.6), (0.0, 0.15), (0.2, 0.6),
              (0.0, 0.1), (0.25, 0.7), (0.4, 0.85), (0.1, 0.5)], // keys
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

/// Obviously dead patch: near-silent or a bare click (same test as genpatches.py).
pub fn is_dud(audio: &[f32]) -> bool {
    let rms = (audio.iter().map(|v| v * v).sum::<f32>() / audio.len() as f32).sqrt();
    let active = audio.iter().filter(|v| v.abs() > 0.05).count() as f32 / audio.len() as f32;
    rms < 0.03 || active < 0.03
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
        let g = match arch {
            Some(a) => sample_archetype(a, rng),
            None => {
                let mut g = *seed;
                for v in g.iter_mut() {
                    *v = (*v + gaussian(rng) * amount).clamp(0.0, 1.0);
                }
                g
            }
        };
        let audio = synth::render_default(&g, note as f32, rng);
        if is_dud(&audio) {
            continue;
        }
        out.push(Seedling { genome: g, score: score(net, &g) });
    }
    out.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
    out
}

/// "Surprise me": best-scoring non-dud of `pool` fully random genomes.
pub fn lucky_dip(net: &Net, note: i32, pool: usize, rng: &mut SmallRng) -> Option<Genome> {
    let mut best: Option<(f32, Genome)> = None;
    for _ in 0..pool {
        let mut g = [0.0f32; N_PARAMS];
        g.iter_mut().for_each(|v| *v = rng.gen());
        let audio = synth::render_default(&g, note as f32, rng);
        if is_dud(&audio) {
            continue;
        }
        let s = score(net, &g);
        if best.as_ref().map(|(bs, _)| s > *bs).unwrap_or(true) {
            best = Some((s, g));
        }
    }
    best.map(|(_, g)| g)
}
