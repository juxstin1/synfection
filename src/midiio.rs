//! .mid import: any DAW clip becomes a loop-lab groove.
//! Tempo comes from the loop lab (the file's tempo map is ignored), and the
//! groove's lowest note is mapped onto the loop-lab key — so grooves transpose.

use std::collections::HashMap;

use anyhow::{bail, Context, Result};

use crate::loops::Ev;

/// Longest groove we'll loop: 16 bars of 4/4.
const MAX_BEATS: f32 = 64.0;
const MAX_NOTES: usize = 256;

pub struct Groove {
    pub name: String,
    pub events: Vec<Ev>,
    pub beats: f32,
}

impl Groove {
    pub fn bars(&self) -> usize {
        (self.beats / 4.0).round().max(1.0) as usize
    }
}

pub fn load_groove(path: &str) -> Result<Groove> {
    let bytes = std::fs::read(path).with_context(|| format!("couldn't read {path}"))?;
    let smf = midly::Smf::parse(&bytes).context("not a valid midi file")?;
    let tpb = match smf.header.timing {
        midly::Timing::Metrical(t) => t.as_int() as f32,
        midly::Timing::Timecode(..) => bail!("SMPTE-timed midi isn't supported — export with bar/beat timing"),
    };

    // (onset beat, duration beats, key, velocity) across all tracks
    let mut notes: Vec<(f32, f32, u8, u8)> = Vec::new();
    for track in &smf.tracks {
        let mut tick = 0u64;
        let mut open: HashMap<(u8, u8), (u64, u8)> = HashMap::new();
        for ev in track {
            tick += u64::from(ev.delta.as_int());
            let midly::TrackEventKind::Midi { channel, message } = ev.kind else { continue };
            match message {
                midly::MidiMessage::NoteOn { key, vel } if vel.as_int() > 0 => {
                    open.insert((channel.as_int(), key.as_int()), (tick, vel.as_int()));
                }
                midly::MidiMessage::NoteOn { key, .. } | midly::MidiMessage::NoteOff { key, .. } => {
                    if let Some((t0, vel)) = open.remove(&(channel.as_int(), key.as_int())) {
                        let beat = t0 as f32 / tpb;
                        let dur = ((tick - t0) as f32 / tpb).max(1.0 / 16.0);
                        notes.push((beat, dur, key.as_int(), vel));
                    }
                }
                _ => {}
            }
        }
    }
    if notes.is_empty() {
        bail!("no notes in the midi file");
    }

    notes.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    notes.retain(|n| n.0 < MAX_BEATS);
    notes.truncate(MAX_NOTES);

    let low = notes.iter().map(|n| n.2).min().unwrap() as i32;
    let end = notes.iter().map(|n| n.0 + n.1).fold(0.0f32, f32::max);
    let beats = ((end / 4.0).ceil() * 4.0).clamp(4.0, MAX_BEATS);
    let events = notes
        .iter()
        .map(|&(beat, dur, key, vel)| Ev {
            beat,
            dur,
            semi: key as i32 - low,
            gain: (0.95 * vel as f32 / 127.0).clamp(0.2, 0.95),
        })
        .collect();

    let name = std::path::Path::new(path)
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "groove".into());
    Ok(Groove { name, events, beats })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal SMF-0: 480 tpb, F2 on beat 1 (one beat), F3 on beat 3 (half beat).
    fn test_smf() -> Vec<u8> {
        let mut b = vec![];
        b.extend(b"MThd");
        b.extend(6u32.to_be_bytes());
        b.extend(0u16.to_be_bytes()); // format 0
        b.extend(1u16.to_be_bytes()); // one track
        b.extend(480u16.to_be_bytes()); // ticks per beat
        let mut t: Vec<u8> = vec![];
        t.extend([0x00, 0x90, 41, 100]); // t=0       F2 on, vel 100
        t.extend([0x83, 0x60, 0x80, 41, 0]); // +480     F2 off (1 beat)
        t.extend([0x83, 0x60, 0x90, 53, 64]); // +480    F3 on at beat 2...
        t.extend([0x81, 0x70, 0x80, 53, 0]); // +240     F3 off (half beat)
        t.extend([0x00, 0xFF, 0x2F, 0x00]); // end of track
        b.extend(b"MTrk");
        b.extend((t.len() as u32).to_be_bytes());
        b.extend(t);
        b
    }

    #[test]
    fn parses_groove() {
        let dir = std::env::temp_dir().join("synfection_test_groove.mid");
        std::fs::write(&dir, test_smf()).unwrap();
        let g = load_groove(&dir.to_string_lossy()).unwrap();
        assert_eq!(g.events.len(), 2);
        assert_eq!(g.beats, 4.0); // rounded up to a whole bar
        assert_eq!(g.bars(), 1);
        // first note: lowest → semi 0, one beat long, at beat 0
        assert_eq!(g.events[0].semi, 0);
        assert!((g.events[0].beat - 0.0).abs() < 1e-4);
        assert!((g.events[0].dur - 1.0).abs() < 1e-4);
        // second note: an octave up, at beat 2, half a beat
        assert_eq!(g.events[1].semi, 12);
        assert!((g.events[1].beat - 2.0).abs() < 1e-4);
        assert!((g.events[1].dur - 0.5).abs() < 1e-4);
        // velocity maps into the pattern gain range
        assert!(g.events[0].gain > g.events[1].gain);
        let _ = std::fs::remove_file(&dir);
    }
}
