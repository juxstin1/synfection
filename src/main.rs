//! synfection — reverse-engineer a synth patch from a sound, then play with it.
//! Single binary: the trained net is embedded, no Python or model files needed.

mod dsp;
mod garden;
mod genome;
mod loops;
mod matcher;
mod net;
mod presets;
mod synth;
mod ui;
mod wavio;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

use genome::{Genome, N_PARAMS, PARAMS};
use wavio::write_wav;

#[derive(Parser)]
#[command(name = "synfection", version, about = "Clone a synth sound into a playable patch (Genopatch-style)")]
struct Cli {
    /// No subcommand opens the UI
    #[command(subcommand)]
    cmd: Option<Cmd>,
}

#[derive(Subcommand)]
enum Cmd {
    /// Reverse-engineer a patch from a wav and re-render it
    Match {
        /// input .wav (an isolated mono hook / one-shot works best)
        audio: String,
        #[arg(short, long, default_value = "remake.wav")]
        out: String,
        /// note like C2 / F1 / 48 (default: auto-detect pitch)
        #[arg(long)]
        note: Option<String>,
        /// refinement generations (0 disables, same as --no-refine)
        #[arg(long, default_value_t = 60)]
        iters: usize,
        /// keep the raw net guess, skip refinement
        #[arg(long)]
        no_refine: bool,
        #[arg(long, default_value_t = 0)]
        seed: u64,
    },
    /// Render a genome (file or inline "0.2,0.8,...") to a wav
    Render {
        genome: String,
        #[arg(short, long, default_value = "C3")]
        note: String,
        #[arg(short, long, default_value = "patch.wav")]
        out: String,
        #[arg(long, default_value_t = 0)]
        seed: u64,
    },
    /// Spawn mutated sibling sounds of a patch
    Vary {
        genome: String,
        #[arg(short, long, default_value_t = 6)]
        n: usize,
        /// mutation strength
        #[arg(long, default_value_t = 0.15)]
        amount: f32,
        #[arg(long, default_value = "C2")]
        note: String,
        /// comma-separated param names to freeze (e.g. cutoff,osc2_detune)
        #[arg(long, default_value = "")]
        lock: String,
        #[arg(long, default_value = "variations")]
        dir: String,
        #[arg(long, default_value_t = 0)]
        seed: u64,
    },
    /// Breed two patches (uniform crossover + light mutation)
    Breed {
        a: String,
        b: String,
        #[arg(short, long, default_value_t = 6)]
        n: usize,
        #[arg(long, default_value_t = 0.15)]
        amount: f32,
        #[arg(long, default_value = "C2")]
        note: String,
        #[arg(long, default_value = "variations")]
        dir: String,
        #[arg(long, default_value_t = 0)]
        seed: u64,
    },
    /// Turn a patch into a seamless, tempo-locked bass loop (44.1k)
    Loop {
        genome: String,
        /// root note, e.g. F1
        #[arg(long, default_value = "F1")]
        key: String,
        #[arg(long, default_value_t = 138.0)]
        bpm: f32,
        /// see `synfection patterns`
        #[arg(long, default_value = "garage_roll")]
        pattern: String,
        /// push odd 16ths late: 0 straight, ~0.12 garage shuffle
        #[arg(long, default_value_t = 0.0)]
        swing: f32,
        #[arg(long, default_value_t = 2)]
        bars: usize,
        #[arg(short, long, default_value = "loop.wav")]
        out: String,
        #[arg(long, default_value_t = 0)]
        seed: u64,
    },
    /// Render random patches to ear-check the engine
    Gallery {
        #[arg(short, long, default_value_t = 12)]
        n: usize,
        #[arg(long, default_value = "C3")]
        note: String,
        #[arg(long, default_value = "gallery")]
        dir: String,
        #[arg(long, default_value_t = 0)]
        seed: u64,
    },
    /// Prove the loop on a known random patch (render -> match -> compare)
    Selftest {
        #[arg(long, default_value_t = 60)]
        iters: usize,
    },
    /// List the loop patterns
    Patterns,
    /// Open the UI (also the default when run with no arguments)
    Ui {
        /// open with a patch loaded
        #[arg(long)]
        genome: Option<String>,
        /// save a screenshot and exit (for docs)
        #[arg(long, hide = true)]
        screenshot: Option<String>,
    },
}

fn main() -> Result<()> {
    let Some(cmd) = Cli::parse().cmd else {
        return ui::run(None, None);
    };
    match cmd {
        Cmd::Ui { genome, screenshot } => ui::run(genome, screenshot),
        Cmd::Match { audio, out, note, iters, no_refine, seed } => {
            cmd_match(&audio, &out, note.as_deref(), if no_refine { 0 } else { iters }, seed)
        }
        Cmd::Render { genome, note, out, seed } => {
            let g = genome::load(&genome)?;
            let midi = genome::note_to_midi(&note)?;
            let mut rng = SmallRng::seed_from_u64(seed);
            write_wav(&out, &synth::render_default(&g, midi as f32, &mut rng), synth::SR)?;
            println!("rendered genome at MIDI {midi} -> {out}");
            Ok(())
        }
        Cmd::Vary { genome, n, amount, note, lock, dir, seed } => {
            cmd_vary(&genome, n, amount, &note, &lock, &dir, seed)
        }
        Cmd::Breed { a, b, n, amount, note, dir, seed } => {
            cmd_breed(&a, &b, n, amount, &note, &dir, seed)
        }
        Cmd::Loop { genome, key, bpm, pattern, swing, bars, out, seed } => {
            let g = genome::load(&genome)?;
            let root = genome::note_to_midi(&key)?;
            let pat = loops::pattern(&pattern)
                .with_context(|| format!("unknown pattern {pattern:?} (see `synfection patterns`)"))?;
            let mut rng = SmallRng::seed_from_u64(seed);
            let audio = loops::render_loop(&g, root, bpm, &pat, bars, swing, &mut rng);
            write_wav(&out, &audio, loops::SR_OUT)?;
            println!("loop -> {out}  ({key} {bpm:.0}bpm {pattern} {bars}bar 44.1k)");
            Ok(())
        }
        Cmd::Gallery { n, note, dir, seed } => {
            let midi = genome::note_to_midi(&note)?;
            std::fs::create_dir_all(&dir)?;
            let mut rng = SmallRng::seed_from_u64(seed);
            for i in 0..n {
                let mut g = [0.0f32; N_PARAMS];
                g.iter_mut().for_each(|v| *v = rng.gen());
                let audio = synth::render_default(&g, midi as f32, &mut rng);
                write_wav(&format!("{dir}/patch_{i:02}.wav"), &audio, synth::SR)?;
            }
            println!("wrote {n} patches at {note} -> {dir}/");
            Ok(())
        }
        Cmd::Selftest { iters } => cmd_selftest(iters),
        Cmd::Patterns => {
            for p in loops::PATTERN_NAMES {
                println!("{p}");
            }
            Ok(())
        }
    }
}

// ---- matching ---------------------------------------------------------------

fn cmd_match(audio: &str, out: &str, note: Option<&str>, iters: usize, seed: u64) -> Result<()> {
    let target = wavio::load_target(audio)?;
    let midi = match note {
        Some(s) => genome::note_to_midi(s)?,
        None => dsp::detect_midi(&target, synth::SR),
    };
    println!(
        "target {audio}  match at MIDI {midi} ({})",
        if note.is_some() { "given" } else { "detected" }
    );

    let net = net::Net::load()?;
    let guess = matcher::guess(&net, &target)?;

    let mut rng = SmallRng::seed_from_u64(seed);
    let l0 = matcher::loss_of(&guess, &target, midi as f32, seed);
    let g = if iters > 0 {
        let (g, l1) = matcher::refine(&guess, &target, midi as f32, iters, seed, |_, _| {});
        println!("spec-loss  nn-only {l0:.3}  ->  refined {l1:.3}");
        g
    } else {
        println!("spec-loss (nn-only) {l0:.3}");
        guess
    };

    write_wav(out, &synth::render_default(&g, midi as f32, &mut rng), synth::SR)?;
    genome::save(&format!("{out}.genome.txt"), &g)?;
    genome::print_patch(&g);
    println!("remake -> {out}   genome -> {out}.genome.txt");
    Ok(())
}

fn cmd_selftest(iters: usize) -> Result<()> {
    let mut rng = SmallRng::seed_from_u64(7);
    let mut truth = [0.0f32; N_PARAMS];
    truth.iter_mut().for_each(|v| *v = rng.gen());
    let midi = 48.0;
    let target = synth::render_default(&truth, midi, &mut rng);

    let net = net::Net::load()?;
    let guess = matcher::guess(&net, &target)?;
    let l0 = matcher::loss_of(&guess, &target, midi, 7);
    let (g, l1) = matcher::refine(&guess, &target, midi, iters, 7, |_, _| {});
    let mae = |a: &Genome| -> f32 {
        a.iter().zip(&truth).map(|(x, y)| (x - y).abs()).sum::<f32>() / N_PARAMS as f32
    };
    println!("spec-loss  nn-only {l0:.3}  ->  refined {l1:.3}");
    println!("genome MAE nn-only {:.3}  ->  refined {:.3}", mae(&guess), mae(&g));
    write_wav("selftest_target.wav", &target, synth::SR)?;
    write_wav("selftest_remake.wav", &synth::render_default(&g, midi, &mut rng), synth::SR)?;
    genome::print_patch(&g);
    println!("wrote selftest_target.wav / selftest_remake.wav");
    Ok(())
}

// ---- vary / breed -----------------------------------------------------------

fn lock_indices(lock: &str) -> Result<Vec<usize>> {
    lock.split(',')
        .filter(|s| !s.trim().is_empty())
        .map(|s| {
            PARAMS
                .iter()
                .position(|&(n, ..)| n == s.trim())
                .with_context(|| format!("unknown param {s:?}"))
        })
        .collect()
}

fn mutate(g: &Genome, amount: f32, lock: &[usize], rng: &mut SmallRng) -> Genome {
    let mut out = *g;
    for (i, v) in out.iter_mut().enumerate() {
        if lock.contains(&i) {
            continue;
        }
        let u1: f32 = rng.gen_range(f32::EPSILON..1.0);
        let u2: f32 = rng.gen_range(0.0..1.0);
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
        *v = (*v + z * amount).clamp(0.0, 1.0);
    }
    out
}

fn cmd_vary(spec: &str, n: usize, amount: f32, note: &str, lock: &str, dir: &str, seed: u64) -> Result<()> {
    let base = genome::load(spec)?;
    let midi = genome::note_to_midi(note)? as f32;
    let lock = lock_indices(lock)?;
    std::fs::create_dir_all(dir)?;
    let mut rng = SmallRng::seed_from_u64(seed);
    write_wav(&format!("{dir}/original.wav"), &synth::render_default(&base, midi, &mut rng), synth::SR)?;
    for i in 0..n {
        let v = mutate(&base, amount, &lock, &mut rng);
        write_wav(&format!("{dir}/var_{i:02}.wav"), &synth::render_default(&v, midi, &mut rng), synth::SR)?;
        genome::save(&format!("{dir}/var_{i:02}.genome.txt"), &v)?;
    }
    println!("spawned {n} sound variations at {note}  amount={amount} -> {dir}/");
    Ok(())
}

fn cmd_breed(a: &str, b: &str, n: usize, amount: f32, note: &str, dir: &str, seed: u64) -> Result<()> {
    let pa = genome::load(a)?;
    let pb = genome::load(b)?;
    let midi = genome::note_to_midi(note)? as f32;
    std::fs::create_dir_all(dir)?;
    let mut rng = SmallRng::seed_from_u64(seed);
    write_wav(&format!("{dir}/parent_A.wav"), &synth::render_default(&pa, midi, &mut rng), synth::SR)?;
    write_wav(&format!("{dir}/parent_B.wav"), &synth::render_default(&pb, midi, &mut rng), synth::SR)?;
    for i in 0..n {
        let mut child = [0.0f32; N_PARAMS];
        for j in 0..N_PARAMS {
            child[j] = if rng.gen_bool(0.5) { pa[j] } else { pb[j] }; // uniform crossover
        }
        let child = mutate(&child, amount * 0.5, &[], &mut rng);
        write_wav(&format!("{dir}/child_{i:02}.wav"), &synth::render_default(&child, midi, &mut rng), synth::SR)?;
        genome::save(&format!("{dir}/child_{i:02}.genome.txt"), &child)?;
    }
    println!("bred {n} children (+2 parents) at {note} -> {dir}/");
    Ok(())
}

