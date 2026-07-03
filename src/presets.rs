//! Factory presets. Values are normalized [0,1] genomes in PARAMS order:
//! osc1_wt, osc2_wt, osc2_detune, osc_mix, sub_level, noise_level, drive,
//! cutoff, reso, filt_env, filt_a, filt_d, amp_a, amp_d, amp_s, amp_r

use crate::genome::Genome;

pub struct Preset {
    pub name: &'static str,
    pub note: i32,
    pub genome: Genome,
}

pub const PRESETS: [Preset; 12] = [
    // warm init patch — a starting point, not a statement
    Preset { name: "First Light", note: 36, genome:
        [0.42, 0.50, 0.50, 0.25, 0.80, 0.05, 0.25, 0.45, 0.20, 0.35, 0.30, 0.50, 0.02, 0.50, 0.70, 0.30, 0.00, 0.50, 0.40, 0.00] },
    // tight UKG chord-stab bass: snappy, filtered, gone before the hat
    Preset { name: "Garage Stab", note: 45, genome:
        [0.50, 0.55, 0.56, 0.50, 0.30, 0.06, 0.30, 0.60, 0.50, 0.65, 0.02, 0.28, 0.01, 0.30, 0.10, 0.25, 0.00, 0.50, 0.40, 0.00] },
    // speed-garage organ bass stab: hollow, percussive, octave-hop fuel
    Preset { name: "Organ Stab", note: 45, genome:
        [0.05, 0.30, 0.52, 0.45, 0.55, 0.02, 0.15, 0.55, 0.35, 0.40, 0.01, 0.22, 0.005, 0.25, 0.20, 0.20, 0.00, 0.50, 0.40, 0.00] },
    // long evolving stab, RUFUS DU SOL Innerbloom lane: slow filter bloom, long tail
    Preset { name: "Innerbloom", note: 41, genome:
        [0.43, 0.43, 0.62, 0.50, 0.45, 0.08, 0.12, 0.42, 0.25, 0.50, 0.85, 0.85, 0.65, 0.90, 0.85, 0.92, 0.00, 0.50, 0.40, 0.00] },
    // pure weight: sine sub with a hint of drive to read on small speakers
    Preset { name: "Deep Sub", note: 33, genome:
        [0.05, 0.00, 0.50, 0.10, 1.00, 0.00, 0.20, 0.28, 0.15, 0.15, 0.05, 0.50, 0.02, 0.60, 0.80, 0.35, 0.00, 0.50, 0.40, 0.00] },
    // wide-detuned growler for wobble_hold / reese_hold patterns
    Preset { name: "Reese Growl", note: 36, genome:
        [0.45, 0.45, 0.88, 0.50, 0.55, 0.05, 0.55, 0.45, 0.60, 0.35, 0.10, 0.55, 0.03, 0.70, 0.85, 0.40, 0.00, 0.50, 0.40, 0.00] },
    // 303-ish squelch: high reso, hard filter env, short and rude
    Preset { name: "Acid Squelch", note: 45, genome:
        [0.43, 0.43, 0.50, 0.20, 0.15, 0.00, 0.45, 0.35, 0.90, 0.85, 0.02, 0.35, 0.01, 0.40, 0.30, 0.25, 0.00, 0.50, 0.40, 0.00] },
    // bright glassy pluck: fast filter snap, no sustain
    Preset { name: "Glass Pluck", note: 57, genome:
        [0.10, 0.55, 0.54, 0.40, 0.15, 0.04, 0.10, 0.70, 0.55, 0.75, 0.01, 0.18, 0.005, 0.20, 0.05, 0.30, 0.00, 0.50, 0.40, 0.00] },
    // soft sustained bed, slow attack both envelopes
    Preset { name: "Warm Pad", note: 50, genome:
        [0.25, 0.30, 0.60, 0.50, 0.30, 0.10, 0.05, 0.50, 0.20, 0.30, 0.60, 0.70, 0.60, 0.80, 0.90, 0.80, 0.00, 0.50, 0.40, 0.00] },
    // vowel-ish formant lead, resonant and talkative
    Preset { name: "Formant Talk", note: 48, genome:
        [0.86, 0.86, 0.55, 0.50, 0.25, 0.05, 0.30, 0.55, 0.70, 0.55, 0.15, 0.40, 0.02, 0.50, 0.60, 0.35, 0.00, 0.50, 0.40, 0.00] },
    // detuned pulse rave swell — hoover adjacent
    Preset { name: "Hoover Rush", note: 50, genome:
        [0.60, 0.65, 0.80, 0.50, 0.40, 0.12, 0.50, 0.60, 0.55, 0.60, 0.35, 0.60, 0.15, 0.70, 0.80, 0.45, 0.00, 0.50, 0.40, 0.00] },
    // dusty electric-piano-ish keys for 2-step chords
    Preset { name: "Dust Keys", note: 55, genome:
        [0.15, 0.20, 0.53, 0.40, 0.30, 0.08, 0.10, 0.55, 0.30, 0.35, 0.03, 0.45, 0.02, 0.55, 0.60, 0.40, 0.00, 0.50, 0.40, 0.00] },
];
