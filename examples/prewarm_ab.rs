//! Offline study harness (not shipped): what does the loop prewarm length in
//! `dsp::safety` actually change? Faithful re-implementation with the warmup
//! length parameterized, self-validated against the shipped function.
//!
//!   cargo run --release --example prewarm_ab

use rand::rngs::SmallRng;
use rand::SeedableRng;

#[path = "../src/dsp.rs"]
mod dsp;
#[path = "../src/genome.rs"]
mod genome;
#[path = "../src/loops.rs"]
mod loops;
#[path = "../src/presets.rs"]
mod presets;
#[path = "../src/synth.rs"]
mod synth;

const CEILING: f32 = 0.9;
const RMS_CAP: f32 = 0.25;
const KNEE: f32 = 0.75;

/// Byte-for-byte copy of dsp::safety with the loop warmup length exposed.
fn safety_warm(x: &mut [f32], sr: f32, looped: bool, warm: usize) {
    if x.is_empty() {
        return;
    }
    let a = 1.0 - std::f32::consts::TAU * 25.0 / sr;
    let (mut px, mut py) = (0.0f32, 0.0f32);
    if looped {
        for &v in &x[x.len().saturating_sub(warm)..] {
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
    let rms =
        (x.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>() / x.len() as f64).sqrt() as f32;
    if rms > RMS_CAP {
        let s = RMS_CAP / rms;
        x.iter_mut().for_each(|v| *v *= s);
    }
    let span = CEILING - KNEE;
    for v in x.iter_mut() {
        let av = v.abs();
        if av > KNEE {
            *v = v.signum() * (KNEE + ((av - KNEE) / span).tanh() * span);
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

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).fold(0.0f32, |m, (x, y)| m.max((x - y).abs()))
}

/// Seam discontinuity: |first sample - what the filter would emit if the
/// buffer continued past the end into its own head|, approximated by the
/// step between the last and first sample relative to the local slope.
fn seam_step(x: &[f32]) -> f32 {
    let n = x.len();
    (x[0] - x[n - 1]).abs()
}

fn main() {
    let sr = loops::SR_OUT;
    let cases: [(&str, usize, f32); 4] = [
        ("Deep Sub / reese_hold", 4, 128.0),
        ("Reese Growl / wobble_hold", 5, 138.0),
        ("Garage Stab / garage_roll", 1, 140.0),
        ("Warm Pad / house_offbeat", 8, 124.0),
    ];
    let pats = ["reese_hold", "wobble_hold", "garage_roll", "house_offbeat"];

    println!("case,peak,shipped_vs_8192_maxdiff,d2048,d1024,d512,d0,seam8192,seam2048");
    for (i, (name, pi, bpm)) in cases.iter().enumerate() {
        let p = &presets::PRESETS[*pi];
        let pat = loops::pattern(pats[i]).unwrap();
        let mut rng = SmallRng::seed_from_u64(0);
        let raw = loops::render_loop(&p.genome, p.note, *bpm, &pat, 2, 0.0, &mut rng);
        let base = dsp::thicken(&raw, sr, 0.3); // UI default unison

        let mut shipped = base.clone();
        dsp::safety(&mut shipped, sr, true);

        let mut w8192 = base.clone();
        safety_warm(&mut w8192, sr, true, 8192);
        let copy_err = max_abs_diff(&shipped, &w8192);

        let mut out = Vec::new();
        for w in [2048usize, 1024, 512, 0] {
            let mut y = base.clone();
            safety_warm(&mut y, sr, true, w);
            out.push(max_abs_diff(&shipped, &y));
        }
        let mut w2048 = base.clone();
        safety_warm(&mut w2048, sr, true, 2048);
        let peak = shipped.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        println!(
            "{name},{peak:.6},{copy_err:.3e},{:.3e},{:.3e},{:.3e},{:.3e},{:.3e},{:.3e}",
            out[0], out[1], out[2], out[3], seam_step(&shipped), seam_step(&w2048)
        );
    }
}
