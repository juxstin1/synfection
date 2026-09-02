//! Offline study harness (not shipped): instruments the sigma schedule inside
//! matcher::refine to see whether the search is still exploring at the end of
//! its budget or has collapsed to the floor. Same copy discipline as
//! es_sweep -- validated bit-exact against the shipped refine.
//!
//!   cargo run --release --features study --example sigma_trace -- <targets> <gens>

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

struct Trace {
    final_loss: f32,
    improved: usize,
    floor_gen: Option<usize>,
    last_improve: usize,
    sigma_end: f32,
}

fn refine_traced(guess: &Genome, target: &[f32], midi: f32, gens: usize, seed: u64, grow: f32) -> Trace {
    const LAMBDA: usize = 16;
    let mut rng = SmallRng::seed_from_u64(seed.wrapping_add(1));
    let mut best = *guess;
    let mut best_loss = matcher::loss_of(guess, target, midi, seed);
    let mut sigma = 0.08f32;
    let (mut improved, mut floor_gen, mut last_improve) = (0usize, None, 0usize);
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
            improved += 1;
            last_improve = gen;
            sigma = (sigma * grow).min(0.5);
        } else {
            sigma = (sigma * 0.85).max(0.005);
        }
        sigma = (sigma * 0.995).max(0.005);
        if floor_gen.is_none() && sigma <= 0.005 + 1e-9 {
            floor_gen = Some(gen);
        }
    }
    Trace { final_loss: best_loss, improved, floor_gen, last_improve, sigma_end: sigma }
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
    let nt: usize = a.first().map(|s| s.parse().unwrap()).unwrap_or(10);
    let gens: usize = a.get(1).map(|s| s.parse().unwrap()).unwrap_or(60);
    let grow: f32 = a.get(2).map(|s| s.parse().unwrap()).unwrap_or(1.0);
    let n = net::Net::load().expect("weights");

    // self-validation against the shipped refine
    for t in 0..2 {
        let (note, target) = target_for(t);
        let init = matcher::guess(&n, &target).expect("guess");
        let (_g, lref) = matcher::refine(&init, &target, note as f32, 20, t as u64, |_, _| {});
        let tr = refine_traced(&init, &target, note as f32, 20, t as u64, 1.0);
        assert_eq!(lref.to_bits(), tr.final_loss.to_bits(), "trace diverges from shipped refine");
    }
    eprintln!("trace validated bit-exact against matcher::refine");

    println!("grow,trial,gens,l_init,l_final,improved,floor_gen,last_improve,sigma_end");
    for t in 0..nt {
        let (note, target) = target_for(t);
        let init = matcher::guess(&n, &target).expect("guess");
        let li = matcher::loss_of(&init, &target, note as f32, t as u64);
        let tr = refine_traced(&init, &target, note as f32, gens, t as u64, grow);
        let fg = tr.floor_gen.map(|g| g as i64).unwrap_or(-1);
        println!("{grow},{t},{gens},{li:.6},{:.6},{},{fg},{},{:.6}",
                 tr.final_loss, tr.improved, tr.last_improve, tr.sigma_end);
    }
}
