//! The match pipeline shared by CLI and UI: net guess + evolutionary refinement.

use anyhow::Result;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

use crate::dsp;
use crate::genome::Genome;
use crate::net::Net;
use crate::synth;

/// Net guess from a prepared target (engine SR, N samples, peak-normalized).
pub fn guess(net: &Net, target: &[f32]) -> Result<Genome> {
    let fb = net.mel_fb()?;
    let (mel, h, w) = dsp::melspec(target, &fb.data, fb.shape[0]);
    net.forward(&mel, h, w)
}

pub fn loss_of(g: &Genome, target: &[f32], midi: f32, seed: u64) -> f32 {
    let mut rng = SmallRng::seed_from_u64(seed);
    dsp::multiscale_stft(&synth::render_default(g, midi, &mut rng), target)
}

/// (1+λ) evolution on the genome against the multi-scale spectral loss —
/// the gradient-free stand-in for match.py's Adam refinement.
/// `progress` is called each generation with (gen, best_loss).
pub fn refine(
    guess: &Genome,
    target: &[f32],
    midi: f32,
    gens: usize,
    seed: u64,
    mut progress: impl FnMut(usize, f32),
) -> (Genome, f32) {
    const LAMBDA: usize = 16;
    let mut rng = SmallRng::seed_from_u64(seed.wrapping_add(1));
    let mut best = *guess;
    let mut best_loss = loss_of(guess, target, midi, seed);
    let mut sigma = 0.08f32;
    for gen in 0..gens {
        let cands: Vec<Genome> = (0..LAMBDA)
            .map(|_| {
                let mut c = best;
                for v in c.iter_mut() {
                    *v = (*v + gaussian(&mut rng) * sigma).clamp(0.0, 1.0);
                }
                c
            })
            .collect();
        let losses: Vec<f32> = cands
            .par_iter()
            .enumerate()
            .map(|(i, c)| loss_of(c, target, midi, seed ^ (gen as u64 * 131 + i as u64)))
            .collect();
        let (bi, bl) = losses
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();
        if *bl < best_loss {
            best_loss = *bl;
            best = cands[bi];
        } else {
            sigma = (sigma * 0.85).max(0.005);
        }
        sigma = (sigma * 0.995).max(0.005);
        progress(gen, best_loss);
    }
    (best, best_loss)
}

pub fn gaussian(rng: &mut SmallRng) -> f32 {
    let u1: f32 = rng.gen_range(f32::EPSILON..1.0);
    let u2: f32 = rng.gen_range(0.0..1.0);
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
}
