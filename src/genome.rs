//! The genome: 16 normalized [0,1] knobs and their real-value mappings.
//! Mirrors synth.py's PARAMS table exactly.

use anyhow::{bail, Context, Result};

pub const N_PARAMS: usize = 16;
pub const DRIVE_IDX: usize = 6;

/// (name, lo, hi, log-mapped)
pub const PARAMS: [(&str, f32, f32, bool); N_PARAMS] = [
    ("osc1_wt", 0.0, 1.0, false),
    ("osc2_wt", 0.0, 1.0, false),
    ("osc2_detune", -50.0, 50.0, false),
    ("osc_mix", 0.0, 1.0, false),
    ("sub_level", 0.0, 1.0, false),
    ("noise_level", 0.0, 0.6, false),
    ("drive", 0.0, 1.0, false),
    ("cutoff", 60.0, 10000.0, true),
    ("reso", 0.6, 9.0, false),
    ("filt_env", 0.0, 1.0, false),
    ("filt_a", 0.001, 0.4, true),
    ("filt_d", 0.02, 0.7, true),
    ("amp_a", 0.001, 0.4, true),
    ("amp_d", 0.02, 0.7, true),
    ("amp_s", 0.0, 1.0, false),
    ("amp_r", 0.02, 0.7, true),
];

pub type Genome = [f32; N_PARAMS];

/// Normalized [0,1] -> real parameter values.
pub fn denorm(g: &Genome) -> Genome {
    let mut out = [0.0f32; N_PARAMS];
    for (i, &(_, lo, hi, log)) in PARAMS.iter().enumerate() {
        let x = g[i].clamp(0.0, 1.0);
        out[i] = if log { lo * (hi / lo).powf(x) } else { lo + (hi - lo) * x };
    }
    out
}

/// Legacy 15-param (v1 engine) genome -> current: insert neutral drive=0.
pub fn upgrade(vals: &[f32]) -> Result<Genome> {
    let mut g = [0.0f32; N_PARAMS];
    match vals.len() {
        N_PARAMS => g.copy_from_slice(vals),
        15 => {
            g[..DRIVE_IDX].copy_from_slice(&vals[..DRIVE_IDX]);
            g[DRIVE_IDX] = 0.0;
            g[DRIVE_IDX + 1..].copy_from_slice(&vals[DRIVE_IDX..]);
        }
        n => bail!("genome has {n} params, expected {N_PARAMS} (or 15 legacy)"),
    }
    Ok(g)
}

/// Genome from a file of whitespace-separated floats, or an inline "0.2,0.8,..." string.
/// Lines starting with '#' are comments (numpy-compatible).
pub fn load(spec: &str) -> Result<Genome> {
    Ok(load_with_note(spec)?.0)
}

/// Like `load`, but also returns the note stored in a `# note=NN` comment if present.
pub fn load_with_note(spec: &str) -> Result<(Genome, Option<i32>)> {
    let text = match std::fs::read_to_string(spec) {
        Ok(t) => t,
        Err(_) => spec.replace(',', " "),
    };
    let mut note = None;
    let mut vals = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix('#') {
            if let Some(n) = rest.split("note=").nth(1) {
                note = n.trim().parse::<i32>().ok();
            }
            continue;
        }
        for tok in line.split(|c: char| c.is_whitespace() || c == ',') {
            if !tok.is_empty() {
                vals.push(tok.parse::<f32>().with_context(|| format!("bad genome value {tok:?}"))?);
            }
        }
    }
    Ok((upgrade(&vals)?, note))
}

pub fn save(path: &str, g: &Genome) -> Result<()> {
    let text: String = g.iter().map(|x| format!("{x:.5}\n")).collect();
    std::fs::write(path, text)?;
    Ok(())
}

/// Save a patch with its note (readable by numpy, the CLI, and the app).
pub fn save_patch(path: &std::path::Path, g: &Genome, note: i32) -> Result<()> {
    let mut text = format!("# synfection patch  note={note}\n");
    text.extend(g.iter().map(|x| format!("{x:.5}\n")));
    std::fs::write(path, text)?;
    Ok(())
}

pub fn print_patch(g: &Genome) {
    let p = denorm(g);
    println!("  matched patch:");
    for (i, &(name, ..)) in PARAMS.iter().enumerate() {
        println!("    {name:12} {:.3}", p[i]);
    }
}

const NOTE_NAMES: [&str; 12] =
    ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];

/// "F1" / "C#3" / "48" -> MIDI note number.
pub fn note_to_midi(s: &str) -> Result<i32> {
    let s = s.trim();
    if let Ok(n) = s.parse::<i32>() {
        return Ok(n);
    }
    let (name, oct) = s.split_at(s.len() - 1);
    let name = name.to_uppercase().replace('S', "#");
    let oct: i32 = oct.parse().with_context(|| format!("bad note {s:?}"))?;
    let idx = NOTE_NAMES
        .iter()
        .position(|&n| n == name)
        .with_context(|| format!("bad note name {s:?}"))?;
    Ok(idx as i32 + (oct + 1) * 12)
}

pub fn midi_to_name(m: i32) -> String {
    format!("{}{}", NOTE_NAMES[m.rem_euclid(12) as usize], m / 12 - 1)
}
