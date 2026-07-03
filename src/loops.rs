//! Tempo-locked, seamless bass loops from a patch — port of loops.py.

use rand::rngs::SmallRng;

use crate::genome::Genome;
use crate::synth::render;

pub const SR_OUT: f32 = 44100.0;

/// A pattern is 16 sixteenth-steps: None = rest, Some((semitones, gate_steps, gain)).
pub type Pattern = [Option<(i32, usize, f32)>; 16];

pub const PATTERN_NAMES: [&str; 12] = [
    "garage_roll",
    "speed_walk",
    "four_pulse",
    "speed_run",
    "garage_bounce",
    "bassline_seesaw",
    "organ_hop",
    "skippy_ghost",
    "wobble_hold",
    "dnb_roller",
    "house_offbeat",
    "reese_hold",
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
        // relentless speed-garage roller — near-constant 8ths, octave/fifth flicks
        "speed_run" => [
            Some((0, 1, G)), None, Some((0, 1, G)), None,
            Some((12, 1, G)), None, Some((0, 1, G)), Some((7, 1, G)),
            Some((0, 1, G)), None, Some((10, 1, G)), None,
            Some((12, 1, G)), Some((0, 1, G)), Some((7, 1, G)), Some((5, 1, G)),
        ],
        // 2-step skip with octave pops on the off-16ths
        "garage_bounce" => [
            Some((0, 2, G)), None, None, Some((12, 1, 0.7)),
            None, None, Some((0, 1, G)), None,
            Some((0, 2, G)), None, Some((10, 1, 0.6)), None,
            Some((12, 1, G)), None, None, Some((0, 1, G)),
        ],
        // niche / bassline octave seesaw — root vs ghosted octave every 16th
        "bassline_seesaw" => [
            Some((0, 1, G)), Some((12, 1, 0.55)), Some((0, 1, G)), Some((12, 1, 0.55)),
            Some((0, 1, G)), Some((12, 1, 0.55)), Some((3, 1, G)), Some((12, 1, 0.55)),
            Some((0, 1, G)), Some((12, 1, 0.55)), Some((0, 1, G)), Some((12, 1, 0.55)),
            Some((5, 1, G)), Some((12, 1, 0.55)), Some((7, 1, G)), Some((10, 1, 0.6)),
        ],
        // speed-garage organ bass — offbeat octave hops with fifth/b7 turns
        "organ_hop" => [
            None, Some((12, 1, G)), None, Some((0, 1, G)),
            None, Some((12, 1, 0.8)), None, Some((7, 1, G)),
            None, Some((12, 1, G)), None, Some((0, 1, G)),
            Some((5, 1, G)), None, Some((7, 1, G)), Some((10, 1, 0.8)),
        ],
        // classic UKG skip with ghost notes for shuffle feel
        "skippy_ghost" => [
            Some((0, 2, G)), None, Some((0, 1, 0.45)), None,
            None, Some((0, 1, G)), None, Some((0, 1, 0.5)),
            None, Some((0, 2, G)), None, Some((0, 1, 0.45)),
            Some((10, 1, 0.7)), None, Some((0, 1, G)), None,
        ],
        // half-bar growl holds with a minor walk-up turnaround
        "wobble_hold" => [
            Some((0, 6, G)), None, None, None,
            None, None, Some((0, 2, 0.8)), None,
            Some((3, 4, G)), None, None, None,
            Some((5, 1, G)), None, Some((7, 1, G)), Some((10, 1, G)),
        ],
        // 174-friendly roller — syncopated, octave flick mid-bar
        "dnb_roller" => [
            Some((0, 2, G)), None, None, Some((0, 1, G)),
            None, None, Some((12, 1, 0.7)), None,
            None, Some((0, 2, G)), None, None,
            Some((0, 1, G)), None, Some((7, 1, 0.7)), Some((10, 1, 0.7)),
        ],
        _ => return None,
    };
    Some(p)
}

/// Mono bass loop, seamless (release tails wrap to the head). 44.1k output.
/// `swing` pushes every odd 16th late by that fraction of a step (0 = straight,
/// ~0.12 = garage shuffle, 0.3 = drunk).
pub fn render_loop(
    g: &Genome,
    root_midi: i32,
    bpm: f32,
    pat: &Pattern,
    bars: usize,
    swing: f32,
    rng: &mut SmallRng,
) -> Vec<f32> {
    let sr = SR_OUT;
    let step = 60.0 / bpm / 4.0;
    let step_n = (step * sr).round() as usize;
    let loop_n = step_n * 16 * bars;
    let mut buf = vec![0.0f32; loop_n];
    let tail = (0.30 * sr) as usize;
    let swing_n = (swing.clamp(0.0, 0.5) * step_n as f32) as usize;
    for bar in 0..bars {
        for (i, slot) in pat.iter().enumerate() {
            let Some((offset, gate, gain)) = slot else { continue };
            let midi = root_midi + offset;
            let gate_n = gate * step_n;
            let n = gate_n + tail;
            let note_dur = gate_n as f32 / sr;
            let a = render(g, midi as f32, sr, n, note_dur, rng);
            let pos = (bar * 16 + i) * step_n + if i % 2 == 1 { swing_n } else { 0 };
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

/// One note of an imported MIDI groove: onset and duration in beats,
/// semitone offset above the loop root, velocity gain 0..1.
#[derive(Clone, Copy)]
pub struct Ev {
    pub beat: f32,
    pub dur: f32,
    pub semi: i32,
    pub gain: f32,
}

/// Render a MIDI groove: same additive, seamless wrap-around as `render_loop`,
/// but free timing/length/polyphony. `beats` is the loop length (whole bars).
pub fn render_events(g: &Genome, root_midi: i32, bpm: f32, evs: &[Ev], beats: f32, rng: &mut SmallRng) -> Vec<f32> {
    let sr = SR_OUT;
    let beat_n = (60.0 / bpm * sr).round() as usize;
    let loop_n = ((beat_n as f32 * beats).round() as usize).max(1);
    let mut buf = vec![0.0f32; loop_n];
    let tail = (0.30 * sr) as usize;
    for e in evs {
        let midi = root_midi + e.semi;
        // floor at 30 ms: 1-tick DAW notes should render as notes, not clicks
        let gate_n = ((e.dur * 60.0 / bpm * sr).round() as usize).max((0.030 * sr) as usize);
        let n = gate_n + tail;
        let note_dur = gate_n as f32 / sr;
        let a = render(g, midi as f32, sr, n, note_dur, rng);
        let pos = ((e.beat * beat_n as f32).round() as usize) % loop_n;
        for (j, v) in a.iter().enumerate() {
            buf[(pos + j) % loop_n] += v * e.gain;
        }
    }
    let peak = buf.iter().fold(0.0f32, |m, x| m.max(x.abs())) + 1e-9;
    for v in buf.iter_mut() {
        *v = *v / peak * 0.9;
    }
    buf
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn groove_renders_full_length_audio() {
        let g = [0.5f32; crate::genome::N_PARAMS];
        let evs = [
            Ev { beat: 0.0, dur: 1.0, semi: 0, gain: 0.9 },
            Ev { beat: 2.0, dur: 0.5, semi: 12, gain: 0.7 },
        ];
        let mut rng = SmallRng::seed_from_u64(0);
        let buf = render_events(&g, 41, 138.0, &evs, 4.0, &mut rng);
        let beat_n = (60.0 / 138.0 * SR_OUT).round() as usize;
        assert_eq!(buf.len(), beat_n * 4);
        let rms = (buf.iter().map(|v| v * v).sum::<f32>() / buf.len() as f32).sqrt();
        assert!(rms > 0.01, "groove should not be silent (rms {rms})");
        assert!(buf.iter().all(|v| v.abs() <= 0.91), "peak-normalized to 0.9");
    }
}
