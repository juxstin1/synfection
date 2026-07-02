//! Numerical parity with the PyTorch reference (weights/parity.json, generated
//! by the snippet in export_net.py's docstring workflow). Guards the Rust port
//! of melspec, GenoNet inference, and the synth engine against drift.

use rand::rngs::SmallRng;
use rand::SeedableRng;
use serde_json::Value;

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

fn fixture() -> Value {
    let text = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/weights/parity.json"
    ))
    .expect("weights/parity.json");
    serde_json::from_str(&text).unwrap()
}

fn floats(v: &Value) -> Vec<f32> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect()
}

fn load_wav_target() -> Vec<f32> {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/weights/fixture.wav");
    let mut reader = hound::WavReader::open(path).expect("weights/fixture.wav");
    let spec = reader.spec();
    let raw: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().map(|s| s.unwrap()).collect(),
        hound::SampleFormat::Int => {
            let scale = 1.0 / (1i64 << (spec.bits_per_sample - 1)) as f32;
            reader.samples::<i32>().map(|s| s.unwrap() as f32 * scale).collect()
        }
    };
    let mut x = raw;
    x.resize(synth::N, 0.0);
    let peak = x.iter().fold(0.0f32, |m, v| m.max(v.abs())) + 1e-9;
    x.iter_mut().for_each(|v| *v = *v / peak * 0.9);
    x
}

#[test]
fn mel_and_net_match_pytorch() {
    let fix = fixture();
    let target = load_wav_target();
    let n = net::Net::load().unwrap();
    let fb = n.mel_fb().unwrap();
    let (mel, h, w) = dsp::melspec(&target, &fb.data, fb.shape[0]);

    assert_eq!(h, fix["mel_shape"][0].as_u64().unwrap() as usize);
    assert_eq!(w, fix["mel_shape"][1].as_u64().unwrap() as usize);

    let mel_sum: f32 = mel.iter().sum();
    let ref_sum = fix["mel_sum"].as_f64().unwrap() as f32;
    assert!(
        (mel_sum - ref_sum).abs() / ref_sum < 1e-3,
        "mel_sum {mel_sum} vs torch {ref_sum}"
    );
    for (i, r) in floats(&fix["mel_first"]).iter().enumerate() {
        let v = mel[i * w]; // column 0 of mel row i
        assert!((v - r).abs() < 2e-3, "mel[{i},0] {v} vs {r}");
    }

    let g = n.forward(&mel, h, w).unwrap();
    for (i, r) in floats(&fix["genome"]).iter().enumerate() {
        assert!((g[i] - r).abs() < 5e-3, "genome[{i}] {} vs {r}", g[i]);
    }
}

#[test]
fn render_matches_pytorch() {
    let fix = fixture();
    let gv = floats(&fix["render_genome"]);
    let g = genome::upgrade(&gv).unwrap();
    let mut rng = SmallRng::seed_from_u64(0);
    let audio = synth::render_default(&g, 45.0, &mut rng);

    let rms = (audio.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>()
        / audio.len() as f64)
        .sqrt() as f32;
    let ref_rms = fix["render_rms"].as_f64().unwrap() as f32;
    assert!((rms - ref_rms).abs() / ref_rms < 1e-2, "rms {rms} vs torch {ref_rms}");

    for (i, r) in floats(&fix["render_mid"]).iter().enumerate() {
        let v = audio[1000 + i];
        assert!((v - r).abs() < 5e-3, "sample[{}] {v} vs {r}", 1000 + i);
    }
}

#[test]
fn legacy_genome_upgrades() {
    let g = genome::upgrade(&[0.5f32; 15]).unwrap();
    assert_eq!(g[genome::DRIVE_IDX], 0.0);
    assert_eq!(g[5], 0.5);
    assert_eq!(g[7], 0.5);
    assert!(genome::upgrade(&[0.5f32; 10]).is_err());
}

#[test]
fn presets_are_alive_and_scored() {
    let net = net::Net::load().unwrap();
    let mut rng = SmallRng::seed_from_u64(1);
    for p in presets::PRESETS.iter() {
        let a = synth::render_default(&p.genome, p.note as f32, &mut rng);
        assert!(!garden::is_dud(&a), "preset {:?} renders as a dud", p.name);
        let s = net.reward(&p.genome).expect("reward model missing from weights bundle");
        assert!((0.0..=1.0).contains(&s), "preset {:?} score {s}", p.name);
    }
}

#[test]
fn archetype_seeds_are_mostly_alive() {
    let mut rng = SmallRng::seed_from_u64(2);
    for arch in garden::ARCHETYPE_NAMES {
        let mut alive = 0;
        for _ in 0..6 {
            let g = garden::sample_archetype(arch, &mut rng);
            let a = synth::render_default(&g, garden::home_note(arch) as f32, &mut rng);
            if !garden::is_dud(&a) {
                alive += 1;
            }
        }
        assert!(alive >= 4, "archetype {arch}: only {alive}/6 samples audible");
    }
}

#[test]
fn safety_limits_loudness_and_peaks() {
    // worst case: full-scale constant screech
    let sr = 22050.0;
    let mut x: Vec<f32> = (0..22050)
        .map(|i| (2.0 * std::f32::consts::PI * 3000.0 * i as f32 / sr).sin() * 1.5)
        .collect();
    dsp::safety(&mut x, sr, false);
    let peak = x.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let rms = (x.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>() / x.len() as f64).sqrt();
    assert!(peak <= 0.9 + 1e-3, "peak {peak} above ceiling");
    assert!(rms <= 0.25 + 1e-2, "rms {rms} above loudness cap");
    // click-killing fades: edges start/end at silence
    assert!(x[0].abs() < 1e-6 && x[x.len() - 1].abs() < 1e-6);
}

#[test]
fn thicken_preserves_length_and_survives_safety() {
    let sr = 22050.0;
    let x: Vec<f32> = (0..8192)
        .map(|i| (2.0 * std::f32::consts::PI * 110.0 * i as f32 / sr).sin() * 0.8)
        .collect();
    let mut y = dsp::thicken(&x, sr, 1.0);
    assert_eq!(y.len(), x.len());
    dsp::safety(&mut y, sr, true);
    let peak = y.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    assert!(peak <= 0.9 + 1e-3);
    assert!(peak > 0.1, "thickened audio should not be silent");
}

#[test]
fn patch_roundtrip_with_note() {
    let dir = std::env::temp_dir().join("synfection_test");
    std::fs::create_dir_all(&dir).unwrap();
    let p = dir.join("roundtrip.genome.txt");
    let mut g = [0.0f32; genome::N_PARAMS];
    g.iter_mut().enumerate().for_each(|(i, v)| *v = i as f32 / 16.0);
    genome::save_patch(&p, &g, 41).unwrap();
    let (g2, note) = genome::load_with_note(&p.to_string_lossy()).unwrap();
    assert_eq!(note, Some(41));
    for (a, b) in g.iter().zip(&g2) {
        assert!((a - b).abs() < 1e-4);
    }
}

#[test]
fn note_parsing() {
    assert_eq!(genome::note_to_midi("C3").unwrap(), 48);
    assert_eq!(genome::note_to_midi("F1").unwrap(), 29);
    assert_eq!(genome::note_to_midi("48").unwrap(), 48);
    assert_eq!(genome::note_to_midi("A4").unwrap(), 69);
}
