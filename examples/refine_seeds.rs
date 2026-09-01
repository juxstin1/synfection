//! Offline study harness (not shipped): isolates ES randomness from target
//! difficulty. Same target, same net init, many refine seeds. If `refine`
//! stalls in local minima, final losses within one target split into clusters.
//!
//!   cargo run --release --example refine_seeds -- <targets> <seeds> <gens>
//!
//! CSV to stdout: target,seed,note,l_init,l_final
//! Fixed budget; no restarts, no retry-until-good.

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

use genome::N_PARAMS;

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let n_targets: usize = a.first().map(|s| s.parse().unwrap()).unwrap_or(8);
    let n_seeds: usize = a.get(1).map(|s| s.parse().unwrap()).unwrap_or(12);
    let gens: usize = a.get(2).map(|s| s.parse().unwrap()).unwrap_or(60);

    let n = net::Net::load().expect("embedded weights");
    println!("target,seed,note,l_init,l_final");
    for t in 0..n_targets {
        // identical construction to refine_study::target_for
        let mut rng = SmallRng::seed_from_u64(900_000 + t as u64);
        let mut truth = [0.0f32; N_PARAMS];
        truth.iter_mut().for_each(|v| *v = rng.gen());
        let note = 33 + (t % 32) as i32;
        let target = synth::render_default(&truth, note as f32, &mut rng);

        let init = matcher::guess(&n, &target).expect("net guess");
        for s in 0..n_seeds {
            let seed = 5_000 + (t * 1_000 + s) as u64;
            let l_init = matcher::loss_of(&init, &target, note as f32, seed);
            let (_g, l_final) =
                matcher::refine(&init, &target, note as f32, gens, seed, |_, _| {});
            println!("{t},{s},{note},{l_init:.6},{l_final:.6}");
        }
    }
}
