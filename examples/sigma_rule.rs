//! Offline study harness (not shipped): does a re-expanding step size beat the
//! shipped shrink-only schedule? matcher::refine multiplies sigma by 0.85 on a
//! failed generation and 0.995 every generation, but never grows it on success,
//! so sigma is monotonically non-increasing and pins at the 0.005 floor around
//! generation 30 of 60. This compares that against the classic 1/5th-success
//! rule (grow on improvement, shrink on failure) at identical budget.
//!
//!   cargo run --release --features study --example sigma_rule -- <targets> <gens>
//!
//! CSV: variant,trial,gens,l_init,l_final

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

#[path = "../src/dsp.rs"]
mod dsp;
#[path = "../src/garden.rs"]
mod garden;
#[path = "../src/genome.rs"]
mod genome;
#[path = "../src/matcher.rs"]
mod matcher;
#[path = "../src/net.rs"]
mod net;
#[path = "../src/presets.rs"]
mod presets;
#[path = "../src/synth.rs"]
mod synth;

use genome::{Genome, N_PARAMS};

/// `grow`: factor applied to sigma on an improving generation (1.0 = shipped
/// behaviour). `drift`: per-generation multiplier (0.995 shipped, 1.0 = off).
fn refine_rule(
    guess: &Genome,
    target: &[f32],
    midi: f32,
    gens: usize,
    seed: u64,
    grow: f32,
    drift: f32,
    sigma0: f32,
) -> f32 {
    const LAMBDA: usize = 16;
    let mut rng = SmallRng::seed_from_u64(seed.wrapping_add(1));
    let mut best = *guess;
    let mut best_loss = matcher::loss_of(guess, target, midi, seed);
    let mut sigma = sigma0;
    for gen in 0..gens {
        let cands: Vec<Genome> = (0..LAMBDA)
            .map(|_| {
                let mut c = best;
                for v in c.iter_mut() {
                    *v = (*v + matcher::gaussian(&mut rng) * sigma).clamp(0.0, 1.0);
                }
                c
            })
            .collect();
        let losses: Vec<f32> = cands
            .par_iter()
            .enumerate()
            .map(|(i, c)| matcher::loss_of(c, target, midi, seed ^ (gen as u64 * 131 + i as u64)))
            .collect();
        let (bi, bl) = losses
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();
        if *bl < best_loss {
            best_loss = *bl;
            best = cands[bi];
            sigma = (sigma * grow).min(0.5);
        } else {
            sigma = (sigma * 0.85).max(0.005);
        }
        sigma = (sigma * drift).max(0.005);
    }
    best_loss
}

fn target_for(t: usize) -> (i32, Vec<f32>) {
    let mut rng = SmallRng::seed_from_u64(900_000 + t as u64);
    let mut truth = [0.0f32; N_PARAMS];
    truth.iter_mut().for_each(|v| *v = rng.gen());
    let note = 33 + (t % 32) as i32;
    (note, synth::render_default(&truth, note as f32, &mut rng))
}

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let nt: usize = a.first().map(|s| s.parse().unwrap()).unwrap_or(30);
    let gens: usize = a.get(1).map(|s| s.parse().unwrap()).unwrap_or(60);
    let n = net::Net::load().expect("weights");

    // grow=1.0, drift=0.995, sigma0=0.08 must reproduce the shipped refine exactly
    for t in 0..2 {
        let (note, target) = target_for(t);
        let init = matcher::guess(&n, &target).expect("guess");
        let (_g, lref) = matcher::refine(&init, &target, note as f32, 20, t as u64, |_, _| {});
        let lb = refine_rule(&init, &target, note as f32, 20, t as u64, 1.0, 0.995, 0.08);
        assert_eq!(lref.to_bits(), lb.to_bits(), "variant diverges from shipped refine");
    }
    eprintln!("variant validated bit-exact against matcher::refine at grow=1.0");

    let all: [(&str, f32, f32, f32); 4] = [
        ("shipped", 1.0, 0.995, 0.08),
        ("rule_1.5", 1.5, 0.995, 0.08),
        ("rule_1.3", 1.3, 0.995, 0.08),
        ("rule_1.5_nodrift", 1.5, 1.0, 0.08),
    ];
    // optional third arg: comma-separated variant names to run (default all)
    let pick = a.get(2).cloned().unwrap_or_default();
    let variants: Vec<(&str, f32, f32, f32)> = if pick.is_empty() {
        all.to_vec()
    } else {
        all.iter().filter(|v| pick.split(',').any(|p| p == v.0)).copied().collect()
    };
    println!("variant,trial,gens,l_init,l_final");
    for (name, grow, drift, s0) in variants {
        for t in 0..nt {
            let (note, target) = target_for(t);
            let init = matcher::guess(&n, &target).expect("guess");
            let li = matcher::loss_of(&init, &target, note as f32, t as u64);
            let lf = refine_rule(&init, &target, note as f32, gens, t as u64, grow, drift, s0);
            println!("{name},{t},{gens},{li:.6},{lf:.6}");
        }
    }
}
