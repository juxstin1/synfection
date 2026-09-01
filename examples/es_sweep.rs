//! Offline study harness (not shipped): sweeps the (1+lambda) ES that does the
//! actual work in `matcher::refine` -- generation budget and lambda at matched
//! evaluation cost. The variant below is a faithful copy of matcher::refine
//! with lambda exposed; it self-validates bit-exact against the shipped
//! function at lambda=16 before any sweep runs.
//!
//!   cargo run --release --features study --example es_sweep -- <targets>
//!
//! CSV to stdout: sweep,targets,lambda,gens,evals,trial,l_init,l_final

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

/// matcher::refine with LAMBDA lifted to a parameter. Everything else --
/// sigma schedule, seeding, comparison order -- is identical.
fn refine_lambda(
    guess: &Genome,
    target: &[f32],
    midi: f32,
    gens: usize,
    seed: u64,
    lambda: usize,
) -> (Genome, f32) {
    let mut rng = SmallRng::seed_from_u64(seed.wrapping_add(1));
    let mut best = *guess;
    let mut best_loss = matcher::loss_of(guess, target, midi, seed);
    let mut sigma = 0.08f32;
    for gen in 0..gens {
        let cands: Vec<Genome> = (0..lambda)
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
        } else {
            sigma = (sigma * 0.85).max(0.005);
        }
        sigma = (sigma * 0.995).max(0.005);
    }
    (best, best_loss)
}

fn target_for(t: usize) -> (i32, Vec<f32>) {
    let mut rng = SmallRng::seed_from_u64(900_000 + t as u64);
    let mut truth = [0.0f32; N_PARAMS];
    truth.iter_mut().for_each(|v| *v = rng.gen());
    let note = 33 + (t % 32) as i32;
    let audio = synth::render_default(&truth, note as f32, &mut rng);
    (note, audio)
}

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let n_targets: usize = a.first().map(|s| s.parse().unwrap()).unwrap_or(30);
    let n = net::Net::load().expect("embedded weights");

    // --- self-validation: the copy must reproduce the shipped refine exactly
    for t in 0..3 {
        let (note, target) = target_for(t);
        let init = matcher::guess(&n, &target).expect("net guess");
        let (ga, la) = matcher::refine(&init, &target, note as f32, 20, t as u64, |_, _| {});
        let (gb, lb) = refine_lambda(&init, &target, note as f32, 20, t as u64, 16);
        assert_eq!(la.to_bits(), lb.to_bits(), "loss mismatch on target {t}");
        for i in 0..N_PARAMS {
            assert_eq!(ga[i].to_bits(), gb[i].to_bits(), "genome[{i}] mismatch on target {t}");
        }
    }
    eprintln!("copy validated bit-exact against matcher::refine (3 targets, 20 gens)");

    println!("sweep,lambda,gens,evals,trial,l_init,l_final");

    // --- sweep 1: generation budget at the shipped lambda
    for gens in [15usize, 30, 60, 120, 240] {
        for t in 0..n_targets {
            let (note, target) = target_for(t);
            let init = matcher::guess(&n, &target).expect("net guess");
            let li = matcher::loss_of(&init, &target, note as f32, t as u64);
            let (_g, lf) = refine_lambda(&init, &target, note as f32, gens, t as u64, 16);
            println!("gens,16,{gens},{},{t},{li:.6},{lf:.6}", 16 * gens);
        }
    }

    // --- sweep 2: lambda at matched evaluation budget (960 renders)
    for lambda in [4usize, 8, 16, 32] {
        let gens = 960 / lambda;
        for t in 0..n_targets {
            let (note, target) = target_for(t);
            let init = matcher::guess(&n, &target).expect("net guess");
            let li = matcher::loss_of(&init, &target, note as f32, t as u64);
            let (_g, lf) = refine_lambda(&init, &target, note as f32, gens, t as u64, lambda);
            println!("lambda,{lambda},{gens},{},{t},{li:.6},{lf:.6}", lambda * gens);
        }
    }
}
