//! Tempo-locked, seamless bass loops from a patch — port of loops.py.

use rand::rngs::SmallRng;

use crate::genome::Genome;
use crate::synth::render;

pub const SR_OUT: f32 = 44100.0;

/// A pattern is 16 sixteenth-steps: None = rest, Some((semitones, gate_steps, gain)).
pub type Pattern = [Option<(i32, usize, f32)>; 16];

pub const PATTERN_NAMES: [&str; 5] = [
    "garage_roll",
    "house_offbeat",
    "reese_hold",
    "speed_walk",
    "four_pulse",
];

pub fn pattern(name: &str) -> Option<Pattern> {
    const G: f32 = 0.95;
    let p: Pattern = match name {
        // classic UK garage skippy bass — offbeat stabs with a little movement
        "garage_roll" => [
            Some((0, 1, G)), None, None, Some((0, 1, G)),
            None, Some((12, 1, G)), None, Some((0, 2, G)),
            None, None, Some((0, 1, G)), None,
            Some((10, 1, G)), None, Some((0, 1, G)), None,
        ],
        // 2-step / house offbeat — bass on the "&"s, fat and simple
        "house_offbeat" => [
            None, None, Some((0, 2, G)), None,
            None, None, Some((0, 2, G)), None,
            None, None, Some((0, 2, G)), None,
            None, None, Some((0, 2, G)), None,
        ],
        // reese / dub hold — long sustained notes, room to growl
        "reese_hold" => [
            Some((0, 8, G)), None, None, None,
            None, None, None, None,
            Some((0, 7, G)), None, None, None,
            None, None, None, Some((3, 1, G)),
        ],
        // rolling bassline with octave + fifth movement (speed garage flavour)
        "speed_walk" => [
            Some((0, 1, G)), None, Some((0, 1, G)), Some((12, 1, G)),
            None, Some((0, 1, G)), Some((7, 1, G)), None,
            Some((0, 1, G)), None, Some((0, 1, G)), Some((10, 1, G)),
            None, Some((0, 1, G)), Some((5, 1, G)), Some((7, 1, G)),
        ],
        // 4-to-floor root pulse — every beat, tight gate (bass house)
        "four_pulse" => [
            Some((0, 2, G)), None, None, None,
            Some((0, 2, G)), None, None, None,
            Some((0, 2, G)), None, None, None,
            Some((0, 2, G)), None, None, None,
        ],
        _ => return None,
    };
    Some(p)
}

/// Mono bass loop, seamless (release tails wrap to the head). 44.1k output.
pub fn render_loop(
    g: &Genome,
    root_midi: i32,
    bpm: f32,
    pat: &Pattern,
    bars: usize,
    rng: &mut SmallRng,
) -> Vec<f32> {
    let sr = SR_OUT;
    let step = 60.0 / bpm / 4.0;
    let step_n = (step * sr).round() as usize;
    let loop_n = step_n * 16 * bars;
    let mut buf = vec![0.0f32; loop_n];
    let tail = (0.30 * sr) as usize;
    for bar in 0..bars {
        for (i, slot) in pat.iter().enumerate() {
            let Some((offset, gate, gain)) = slot else { continue };
            let midi = root_midi + offset;
            let gate_n = gate * step_n;
            let n = gate_n + tail;
            let note_dur = gate_n as f32 / sr;
            let a = render(g, midi as f32, sr, n, note_dur, rng);
            let pos = (bar * 16 + i) * step_n;
            for (j, v) in a.iter().enumerate() {
                buf[(pos + j) % loop_n] += v * gain;
            }
        }
    }
    let peak = buf.iter().fold(0.0f32, |m, x| m.max(x.abs())) + 1e-9;
    for v in buf.iter_mut() {
        *v = *v / peak * 0.9;
    }
    buf
}
