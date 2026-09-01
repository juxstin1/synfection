//! Offline study harness (not shipped in the binary): distribution of
//! `matcher::refine` final loss over N seeded trials, and the value of the
//! net's initial guess vs. random / uniform-mid initialization.
//!
//!   cargo run --release --example refine_study -- <trials> <gens> <init>
//!   init: net | random | mid | all
//!
//! Prints CSV to stdout: mode,trial,note,l_init,l_final
//! Fixed trial budget per run; no adaptive restarts, no retry-until-good.

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

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

/// Held-out target for trial `t`: a fresh uniform-random ground-truth genome
/// rendered by the engine. Deterministic in `t`, disjoint from any seed used
/// by the shipped tests.
fn target_for(t: usize) -> (Genome, i32, Vec<f32>) {
    let mut rng = SmallRng::seed_from_u64(900_000 + t as u64);
    let mut truth = [0.0f32; N_PARAMS];
    truth.iter_mut().for_each(|v| *v = rng.gen());
    let note = 33 + (t % 32) as i32; // sweep 33..64, deterministic
    let audio = synth::render_default(&truth, note as f32, &mut rng);
    (truth, note, audio)
}

fn init_genome(mode: &str, n: &net::Net, target: &[f32], t: usize) -> Genome {
    match mode {
        "net" => matcher::guess(n, target).expect("net guess"),
        "mid" => [0.5f32; N_PARAMS],
        "random" => {
            let mut rng = SmallRng::seed_from_u64(700_000 + t as u64);
            let mut g = [0.0f32; N_PARAMS];
            g.iter_mut().for_each(|v| *v = rng.gen());
            g
        }
        other => panic!("unknown init mode {other}"),
    }
}

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let trials: usize = a.first().map(|s| s.parse().unwrap()).unwrap_or(200);
    let gens: usize = a.get(1).map(|s| s.parse().unwrap()).unwrap_or(60);
    let which = a.get(2).cloned().unwrap_or_else(|| "net".into());
    let modes: Vec<&str> = if which == "all" {
        vec!["net", "random", "mid"]
    } else {
        vec![Box::leak(which.into_boxed_str())]
    };

    let n = net::Net::load().expect("embedded weights");
    println!("mode,trial,note,l_init,l_final");
    for mode in modes {
        for t in 0..trials {
            let (_truth, note, target) = target_for(t);
            let init = init_genome(mode, &n, &target, t);
            let l_init = matcher::loss_of(&init, &target, note as f32, t as u64);
            let (_g, l_final) =
                matcher::refine(&init, &target, note as f32, gens, t as u64, |_, _| {});
            println!("{mode},{t},{note},{l_init:.6},{l_final:.6}");
        }
    }
}
