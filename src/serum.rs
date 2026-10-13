//! Serum 2 knobs -> Synfection genome, so the app can play what the Serum
//! net predicts (lab/train_serum.py labels these knobs as raw 0..1 values).
//!
//! Curves read off Serum 2.1.5's own display text (DawDreamer
//! `get_parameter_text`, 2026-10-12): filter Hz = 8.1758 * (22050 / 8.1758)^v (MIDI 0 .. 22.05k), env times =
//! 32 s * v^5, levels/sustain = amplitude v^2 (-12 dB at 0.5), res/drive
//! linear 0..100%. WT Pos maps position-for-position, but Synfection's 8-frame
//! table is not Serum's wavetable, so that knob carries shape, not timbre.

use anyhow::{bail, Context, Result};

use crate::genome::{Genome, N_PARAMS, PARAMS};

/// Knobs this module understands, with the Serum init-patch value used when one is missing.
pub const KNOBS: [(&str, f32); 10] = [
    ("A WT Pos", 0.0),
    ("A Level", 0.75),
    ("B Level", 0.0),
    ("Filter 1 Freq", 1.0),
    ("Filter 1 Res", 0.1),
    ("Filter 1 Drive", 0.0),
    ("Env 1 Attack", 0.0),
    ("Env 1 Decay", 0.5),
    ("Env 1 Sustain", 1.0),
    ("Env 1 Release", 0.25),
];

/// Real value -> normalized [0,1] for genome slot `i` (inverse of `genome::denorm`).
fn norm(i: usize, real: f32) -> f32 {
    let (_, lo, hi, log) = PARAMS[i];
    let x = if log { (real.max(lo) / lo).ln() / (hi / lo).ln() } else { (real - lo) / (hi - lo) };
    x.clamp(0.0, 1.0)
}

fn slot(name: &str) -> usize {
    PARAMS.iter().position(|p| p.0 == name).expect("genome param")
}

fn env_secs(v: f32) -> f32 {
    32.0 * v.powi(5)
}

/// Build a genome from `(Serum knob name, raw 0..1 value)` pairs. Everything
/// Serum's knobs don't cover (sub, noise, pitch env, LFO, filter env) is neutral.
pub fn to_genome(knobs: &[(String, f32)]) -> Result<Genome> {
    let mut v: Vec<f32> = KNOBS.iter().map(|k| k.1).collect();
    for (name, x) in knobs {
        let i = KNOBS.iter().position(|k| k.0 == name)
            .with_context(|| format!("unknown Serum knob {name:?}; known: {:?}", KNOBS.map(|k| k.0)))?;
        if !(0.0..=1.0).contains(x) {
            bail!("{name} = {x}: Serum knobs are raw 0..1 values");
        }
        v[i] = *x;
    }
    let [wt, a_lvl, b_lvl, freq, res, drive, att, dec, sus, rel] = v[..] else { unreachable!() };

    let mut g: Genome = [0.0; N_PARAMS];
    for (name, real) in [
        ("osc2_detune", 0.0), ("filt_a", 0.01), ("filt_d", 0.2),
        ("pitch_dec", 0.05), ("lfo_rate", 1.0),
    ] {
        g[slot(name)] = norm(slot(name), real);
    }
    g[slot("osc1_wt")] = wt;
    g[slot("osc2_wt")] = wt;
    let (a, b) = (a_lvl * a_lvl, b_lvl * b_lvl);
    g[slot("osc_mix")] = if a + b > 0.0 { b / (a + b) } else { 0.0 };
    g[slot("drive")] = drive;
    g[slot("cutoff")] = norm(slot("cutoff"), 8.1758 * (22050.0f32 / 8.1758).powf(freq));
    g[slot("reso")] = res;
    g[slot("amp_a")] = norm(slot("amp_a"), env_secs(att));
    g[slot("amp_d")] = norm(slot("amp_d"), env_secs(dec));
    g[slot("amp_s")] = norm(slot("amp_s"), sus * sus);
    g[slot("amp_r")] = norm(slot("amp_r"), env_secs(rel));
    Ok(g)
}

/// Parse `"A WT Pos=0.3, Filter 1 Freq=0.7"` (or one pair per line) into pairs.
pub fn parse(spec: &str) -> Result<Vec<(String, f32)>> {
    let text = std::fs::read_to_string(spec).unwrap_or_else(|_| spec.to_string());
    text.split([',', '\n'])
        .map(str::trim)
        .filter(|s| !s.is_empty() && !s.starts_with('#'))
        .map(|pair| {
            let (k, v) = pair.split_once('=').with_context(|| format!("expected name=value, got {pair:?}"))?;
            Ok((k.trim().to_string(), v.trim().parse().with_context(|| format!("bad value in {pair:?}"))?))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::genome::denorm;

    fn real(knobs: &[(&str, f32)]) -> Genome {
        let pairs: Vec<_> = knobs.iter().map(|(k, v)| (k.to_string(), *v)).collect();
        denorm(&to_genome(&pairs).unwrap())
    }

    #[test]
    fn display_curves_land_on_serum_values() {
        let p = real(&[("Filter 1 Freq", 0.5), ("Env 1 Attack", 0.4), ("Env 1 Sustain", 0.5)]);
        assert!((p[slot("cutoff")] - 425.0).abs() < 1.0, "cutoff {}", p[slot("cutoff")]);
        assert!((p[slot("amp_a")] - 0.328).abs() < 0.01, "attack {}", p[slot("amp_a")]);
        assert!((p[slot("amp_s")] - 0.25).abs() < 0.01, "sustain {}", p[slot("amp_s")]);
    }

    #[test]
    fn out_of_range_values_clamp_to_synfection_limits() {
        let p = real(&[("Filter 1 Freq", 1.0), ("Env 1 Release", 1.0), ("Env 1 Attack", 0.0)]);
        assert_eq!(p[slot("cutoff")], 10000.0);
        assert!((p[slot("amp_r")] - 0.7).abs() < 1e-4);
        assert!((p[slot("amp_a")] - 0.001).abs() < 1e-6);
    }

    #[test]
    fn levels_set_the_osc_balance() {
        assert_eq!(real(&[("A Level", 1.0), ("B Level", 0.0)])[slot("osc_mix")], 0.0);
        assert!((real(&[("A Level", 0.5), ("B Level", 0.5)])[slot("osc_mix")] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn rejects_unknown_knobs_and_raw_values_outside_0_1() {
        assert!(to_genome(&[("Macro 1".into(), 0.5)]).is_err());
        assert!(to_genome(&[("A Level".into(), 1.5)]).is_err());
        assert_eq!(parse("A Level=0.5, Filter 1 Freq = 0.7").unwrap().len(), 2);
    }
}
