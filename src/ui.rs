//! The synfection app: plant a sound, grow a patch.
//! Preset browser · A/B · undo/redo · clone-from-wav · radial plant editor ·
//! reward-scored garden · loop lab · gapless audio engine with live metering.
//! Hardware-inspired look: dark bio-metal panels, glow accents.

use std::sync::mpsc;
use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;
use eframe::egui::{self, Align2, Color32, FontId, Pos2, Rect, Sense, Stroke, Vec2};
use rand::rngs::SmallRng;
use rand::SeedableRng;
use rodio::Source;

use crate::garden::{self, Seedling};
use crate::genome::{self, Genome, N_PARAMS, PARAMS};
use crate::loops;
use crate::matcher;
use crate::midiio;
use crate::net::Net;
use crate::presets::PRESETS;
use crate::synth;
use crate::wavio;

const BG: Color32 = Color32::from_rgb(9, 12, 10);
const PANEL: Color32 = Color32::from_rgb(15, 21, 16);
const PANEL_EDGE: Color32 = Color32::from_rgb(42, 60, 46);
const DIM: Color32 = Color32::from_rgb(70, 95, 78);
const TEXT: Color32 = Color32::from_rgb(196, 220, 201);
const ACCENT: Color32 = Color32::from_rgb(92, 224, 138);
const ACCENT_HOT: Color32 = Color32::from_rgb(180, 255, 160);
const SEED: Color32 = Color32::from_rgb(34, 54, 38);
const CORE: Color32 = Color32::from_rgb(6, 9, 7);
const METAL: Color32 = Color32::from_rgb(30, 40, 33);
const METAL_HI: Color32 = Color32::from_rgb(62, 82, 66);

// ---- audio engine -----------------------------------------------------------

struct Playing {
    samples: Vec<f32>,
    sr: f32,
    start: Instant,
    looping: bool,
}

struct AudioEngine {
    _stream: Option<rodio::OutputStream>,
    handle: Option<rodio::OutputStreamHandle>,
    sink: Option<rodio::Sink>,
    playing: Option<Playing>,
    volume: f32,
}

impl AudioEngine {
    fn new() -> Self {
        let (stream, handle) = match rodio::OutputStream::try_default() {
            Ok((s, h)) => (Some(s), Some(h)),
            Err(_) => (None, None),
        };
        AudioEngine { _stream: stream, handle, sink: None, playing: None, volume: 0.9 }
    }

    fn play(&mut self, samples: &[f32], sr: f32, looped: bool) {
        self.stop();
        if let Some(h) = &self.handle {
            if let Ok(sink) = rodio::Sink::try_new(h) {
                sink.set_volume(self.volume);
                let src = rodio::buffer::SamplesBuffer::new(1, sr as u32, samples.to_vec());
                if looped {
                    sink.append(src.repeat_infinite());
                } else {
                    sink.append(src);
                }
                self.sink = Some(sink);
                self.playing = Some(Playing {
                    samples: samples.to_vec(),
                    sr,
                    start: Instant::now(),
                    looping: looped,
                });
            }
        }
    }

    fn stop(&mut self) {
        if let Some(s) = self.sink.take() {
            s.stop();
        }
        self.playing = None;
    }

    fn set_volume(&mut self, v: f32) {
        self.volume = v;
        if let Some(s) = &self.sink {
            s.set_volume(v);
        }
    }

    fn loop_playing(&self) -> bool {
        self.playing.as_ref().map(|p| p.looping).unwrap_or(false)
            && self.sink.as_ref().map(|s| !s.empty()).unwrap_or(false)
    }

    /// How far through the current loop we are, [0,1) — for phase-preserving swaps.
    fn loop_pos_frac(&self) -> f32 {
        let Some(p) = &self.playing else { return 0.0 };
        if !p.looping || p.samples.is_empty() || !self.is_playing() {
            return 0.0;
        }
        let idx = (p.start.elapsed().as_secs_f32() * p.sr) as usize % p.samples.len();
        idx as f32 / p.samples.len() as f32
    }

    fn is_playing(&self) -> bool {
        self.sink.as_ref().map(|s| !s.empty()).unwrap_or(false)
    }

    /// Current output level [0,1] for the meter, from playback position.
    fn level(&self) -> f32 {
        let Some(p) = &self.playing else { return 0.0 };
        if !self.is_playing() {
            return 0.0;
        }
        let len = p.samples.len();
        if len == 0 {
            return 0.0;
        }
        let mut idx = (p.start.elapsed().as_secs_f32() * p.sr) as usize;
        if p.looping {
            idx %= len;
        } else if idx >= len {
            return 0.0;
        }
        let end = (idx + 1200).min(len);
        p.samples[idx..end].iter().fold(0.0f32, |m, v| m.max(v.abs())) * self.volume
    }
}

// ---- app --------------------------------------------------------------------

enum MatchMsg {
    Progress(usize, f32),
    Done { genome: Genome, midi: i32, l0: f32, l1: f32 },
    Failed(String),
}

pub struct App {
    net: Arc<Net>,
    genome: Genome,
    note: i32,
    status: String,
    preset_idx: Option<usize>,
    undo: Vec<(Genome, i32)>,
    redo: Vec<(Genome, i32)>,
    ab_other: (Genome, i32),
    ab_is_b: bool,
    target: Option<Vec<f32>>,
    target_name: String,
    rx: Option<mpsc::Receiver<MatchMsg>>,
    progress: f32,
    seedlings: Vec<Seedling>,
    grow_arch: usize,
    grow_amount: f32,
    audio: AudioEngine,
    last_audio: Vec<f32>,
    last_sr: f32,
    bpm: f32,
    key: String,
    pattern_idx: usize,
    swing: f32,
    unison: f32,
    // user patch library
    patch_name: String,
    user_patches: Vec<(String, std::path::PathBuf)>,
    user_sel: Option<String>,
    show_help: bool,
    show_keys: bool,
    shot: Option<String>,
    frame: u64,
    // fit-to-screen: content height measured last frame (points), one-shot monitor clamp
    content_h: f32,
    fitted_to_monitor: bool,
    // render worker (cold path): jobs out, results in
    job_tx: mpsc::Sender<Job>,
    out_rx: mpsc::Receiver<Out>,
    busy: bool,
    cue: f32,
    /// What `last_audio` currently holds — lets play skip the render entirely.
    rendered_key: Option<PatchKey>,
    /// What the playing (or requested) loop was rendered from.
    last_loop_sig: Option<LoopSig>,
    // imported MIDI groove for the loop lab
    groove: Option<midiio::Groove>,
    use_groove: bool,
    groove_gen: u64,
}

/// Documents/synfection/patches (or ~/synfection/patches) — the user's library.
fn patches_dir() -> std::path::PathBuf {
    let home = std::env::var("USERPROFILE")
        .or_else(|_| std::env::var("HOME"))
        .unwrap_or_else(|_| ".".into());
    let docs = std::path::Path::new(&home).join("Documents");
    let base = if docs.is_dir() { docs } else { home.into() };
    base.join("synfection").join("patches")
}

fn list_patches() -> Vec<(String, std::path::PathBuf)> {
    let mut out = Vec::new();
    if let Ok(rd) = std::fs::read_dir(patches_dir()) {
        for e in rd.flatten() {
            let p = e.path();
            if p.extension().map(|x| x == "txt").unwrap_or(false) {
                if let Some(stem) = p.file_stem().and_then(|s| s.to_str()) {
                    out.push((stem.trim_end_matches(".genome").to_string(), p.clone()));
                }
            }
        }
    }
    out.sort();
    out
}

impl App {
    pub fn new(cc: &eframe::CreationContext<'_>, genome_path: Option<String>, shot: Option<String>) -> Self {
        theme(&cc.egui_ctx);
        let net = Arc::new(Net::load().expect("embedded weights"));
        let (genome, note, preset_idx) = match genome_path.and_then(|p| genome::load(&p).ok()) {
            Some(g) => (g, 36, None),
            None => (PRESETS[0].genome, PRESETS[0].note, Some(0)),
        };
        let cue = garden::score(&net, &genome);
        let (job_tx, job_rx) = mpsc::channel();
        let (out_tx, out_rx) = mpsc::channel();
        {
            let net = net.clone();
            let ctx = cc.egui_ctx.clone();
            std::thread::spawn(move || worker_thread(net, job_rx, out_tx, ctx));
        }
        let mut app = App {
            net,
            genome,
            note,
            status: "drop a .wav (or open one) to clone it — or garden the plant by hand".into(),
            preset_idx,
            undo: Vec::new(),
            redo: Vec::new(),
            ab_other: (genome, note),
            ab_is_b: false,
            target: None,
            target_name: String::new(),
            rx: None,
            progress: 0.0,
            seedlings: Vec::new(),
            grow_arch: 0,
            grow_amount: 0.15,
            audio: AudioEngine::new(),
            last_audio: Vec::new(),
            last_sr: synth::SR,
            bpm: 138.0,
            key: "F1".into(),
            pattern_idx: 0,
            swing: 0.12,
            unison: 0.3,
            patch_name: "my_patch".into(),
            user_patches: list_patches(),
            user_sel: None,
            show_help: false,
            show_keys: false,
            shot,
            frame: 0,
            content_h: 1010.0,
            fitted_to_monitor: false,
            job_tx,
            out_rx,
            busy: false,
            cue,
            rendered_key: None,
            last_loop_sig: None,
            groove: None,
            use_groove: false,
            groove_gen: 0,
        };
        app.request_patch(false);
        app
    }

    fn patch_key(&self) -> PatchKey {
        (self.genome, self.note, self.unison.to_bits())
    }

    fn send_job(&mut self, job: Job) {
        self.busy = true;
        let _ = self.job_tx.send(job);
    }

    /// Synchronous render — screenshot mode only (needs deterministic frames).
    fn render_now(&mut self) {
        let mut rng = SmallRng::seed_from_u64(0);
        let raw = synth::render_default(&self.genome, self.note as f32, &mut rng);
        self.last_audio = post_dsp(raw, synth::SR, self.unison, false);
        self.last_sr = synth::SR;
        self.rendered_key = Some(self.patch_key());
        self.cue = garden::score(&self.net, &self.genome);
    }

    /// Hot path: if the current patch is already rendered, play is instant.
    /// Otherwise hand the render to the worker (cold path).
    fn request_patch(&mut self, play: bool) {
        if self.shot.is_some() {
            self.render_now();
            if play {
                self.audio.play(&self.last_audio, self.last_sr, false);
            }
            return;
        }
        if play && self.rendered_key == Some(self.patch_key()) {
            self.audio.play(&self.last_audio, self.last_sr, false);
            return;
        }
        self.send_job(Job::Patch { genome: self.genome, note: self.note, unison: self.unison, play });
    }

    fn checkpoint(&mut self) {
        self.undo.push((self.genome, self.note));
        if self.undo.len() > 64 {
            self.undo.remove(0);
        }
        self.redo.clear();
    }

    fn set_patch(&mut self, g: Genome, note: i32, play: bool) {
        self.checkpoint();
        self.genome = g;
        self.note = note;
        self.request_patch(play);
    }

    fn play_patch(&mut self) {
        self.request_patch(true);
    }

    fn toggle_play(&mut self) {
        if self.audio.is_playing() {
            self.audio.stop();
        } else {
            self.play_patch();
        }
    }

    fn nudge_note(&mut self, d: i32) {
        self.note = (self.note + d).clamp(12, 96);
        self.request_patch(false);
    }

    fn step_preset(&mut self, dir: i32) {
        let n = PRESETS.len() as i32;
        let i = self.preset_idx.map(|i| (i as i32 + dir).rem_euclid(n) as usize).unwrap_or(0);
        self.load_preset(i);
    }

    fn swap_ab(&mut self) {
        std::mem::swap(&mut self.genome, &mut self.ab_other.0);
        std::mem::swap(&mut self.note, &mut self.ab_other.1);
        self.ab_is_b = !self.ab_is_b;
        self.request_patch(true);
    }

    fn do_undo(&mut self) {
        if let Some((g, n)) = self.undo.pop() {
            self.redo.push((self.genome, self.note));
            self.genome = g;
            self.note = n;
            self.request_patch(false);
        }
    }

    fn do_redo(&mut self) {
        if let Some((g, n)) = self.redo.pop() {
            self.undo.push((self.genome, self.note));
            self.genome = g;
            self.note = n;
            self.request_patch(false);
        }
    }

    fn random_patch(&mut self) {
        self.status = "rolling the dice...".into();
        let job = Job::LuckyDip { note: self.note, rng_seed: self.frame };
        self.send_job(job);
    }

    fn toggle_loop(&mut self) {
        if self.audio.loop_playing() {
            self.audio.stop();
        } else {
            self.request_loop(false);
        }
    }

    fn loop_sig(&self, root: i32) -> LoopSig {
        let gid = if self.use_groove && self.groove.is_some() { self.groove_gen } else { 0 };
        (self.genome, root, self.pattern_idx, self.bpm.to_bits(), self.swing.to_bits(), self.unison.to_bits(), gid)
    }

    fn request_loop(&mut self, save: bool) {
        let root = match genome::note_to_midi(&self.key) {
            Ok(r) => r,
            Err(_) => {
                self.status = format!("bad key {:?}", self.key);
                return;
            }
        };
        if !save {
            self.last_loop_sig = Some(self.loop_sig(root));
            if !self.audio.loop_playing() {
                self.status = "rendering loop...".into();
            }
        }
        let groove = match (&self.groove, self.use_groove) {
            (Some(g), true) => Some((g.events.clone(), g.beats)),
            _ => None,
        };
        let groove_name = self.groove.as_ref().filter(|_| self.use_groove).map(|g| g.name.clone());
        let save = save.then(|| {
            let what = groove_name.as_deref().unwrap_or(loops::PATTERN_NAMES[self.pattern_idx]);
            format!("loop_{}_{}bpm_{}.wav", what, self.bpm as u32, self.key)
        });
        let job = Job::Loop {
            genome: self.genome,
            root,
            pattern_idx: self.pattern_idx,
            bpm: self.bpm,
            swing: self.swing,
            unison: self.unison,
            groove,
            save,
        };
        self.send_job(job);
    }

    /// Import a .mid as a loop-lab groove and start it playing.
    fn load_midi_groove(&mut self, path: &str) {
        match midiio::load_groove(path) {
            Ok(g) => {
                self.groove_gen += 1;
                self.use_groove = true;
                self.status = format!("groove: {} — {} bars, key {} moves it", g.name, g.bars(), self.key);
                self.groove = Some(g);
                self.request_loop(false);
            }
            Err(e) => self.status = format!("midi import failed: {e}"),
        }
    }

    fn grow(&mut self) {
        if self.shot.is_some() {
            self.grow_now();
            return;
        }
        let note = if self.grow_arch == 0 {
            self.note
        } else {
            garden::home_note(garden::ARCHETYPE_NAMES[self.grow_arch - 1])
        };
        if self.grow_arch != 0 {
            self.note = note;
        }
        self.status = "growing seedlings...".into();
        let job = Job::Grow {
            seed: self.genome,
            arch: self.grow_arch,
            note,
            amount: self.grow_amount,
            rng_seed: self.frame,
        };
        self.send_job(job);
    }

    /// Synchronous grow — screenshot mode only.
    fn grow_now(&mut self) {
        let mut rng = SmallRng::seed_from_u64(self.frame);
        let (arch, note) = if self.grow_arch == 0 {
            (None, self.note)
        } else {
            let a = garden::ARCHETYPE_NAMES[self.grow_arch - 1];
            (Some(a), garden::home_note(a))
        };
        self.seedlings = garden::grow(&self.net, &self.genome, arch, note, 8, self.grow_amount, &mut rng);
        if arch.is_some() {
            self.note = note;
        }
        let kind = arch.unwrap_or("this patch");
        self.status = format!("grew {} seedlings from {kind} — click to hear, ✔ to adopt", self.seedlings.len());
    }

    /// Apply everything the worker finished since last frame.
    fn drain_worker(&mut self) {
        while let Ok(out) = self.out_rx.try_recv() {
            match out {
                Out::Patch { audio, sr, play, cue, key } => {
                    self.last_audio = audio;
                    self.last_sr = sr;
                    self.rendered_key = Some(key);
                    self.cue = cue;
                    if play {
                        self.audio.play(&self.last_audio, self.last_sr, false);
                    }
                }
                Out::LoopReady { audio, sr } => {
                    // hot swap: pick the new loop up at the bar position the old
                    // one was at. Rotating a seamless loop stays seamless.
                    let phase = self.audio.loop_pos_frac();
                    if phase > 0.0 && !audio.is_empty() {
                        let cut = ((phase * audio.len() as f32) as usize).min(audio.len() - 1);
                        let mut rot = Vec::with_capacity(audio.len());
                        rot.extend_from_slice(&audio[cut..]);
                        rot.extend_from_slice(&audio[..cut]);
                        self.audio.play(&rot, sr, true);
                    } else {
                        self.audio.play(&audio, sr, true);
                    }
                    self.last_audio = audio;
                    self.last_sr = sr;
                    self.rendered_key = None; // waveform now holds the loop
                    self.status = "looping — ⏹ to stop".into();
                }
                Out::Seedling { audio, sr } => {
                    self.audio.play(&audio, sr, false);
                }
                Out::Grown { seedlings, arch } => {
                    let kind = if arch == 0 { "this patch" } else { garden::ARCHETYPE_NAMES[arch - 1] };
                    self.status =
                        format!("grew {} seedlings from {kind} — click to hear, ✔ to adopt", seedlings.len());
                    self.seedlings = seedlings;
                }
                Out::Lucky { genome: Some(g) } => {
                    self.preset_idx = None;
                    let note = self.note;
                    self.set_patch(g, note, true);
                    self.status = "🎲 dealt — best of 12, taste-ranked".into();
                }
                Out::Lucky { genome: None } => {
                    self.status = "all duds — roll again".into();
                }
                Out::SavedPatch { name, msg } => {
                    if let Some(n) = name {
                        self.user_patches = list_patches();
                        self.user_sel = Some(n);
                    }
                    self.status = msg;
                }
                Out::Status(msg) => self.status = msg,
                Out::Idle => self.busy = false,
            }
        }
    }

    fn start_match(&mut self, path: String) {
        let target = match wavio::load_target(&path) {
            Ok(t) => t,
            Err(e) => {
                self.status = format!("couldn't read {path}: {e}");
                return;
            }
        };
        self.target_name = std::path::Path::new(&path)
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or(path.clone());
        self.target = Some(target.clone());
        self.status = format!("cloning {} ...", self.target_name);
        self.progress = 0.0;
        let (tx, rx) = mpsc::channel();
        self.rx = Some(rx);
        std::thread::spawn(move || match_thread(target, tx));
    }

    fn load_preset(&mut self, i: usize) {
        self.preset_idx = Some(i);
        self.user_sel = None;
        self.set_patch(PRESETS[i].genome, PRESETS[i].note, true);
        self.status = format!("preset: {}", PRESETS[i].name);
    }

    fn load_user_patch(&mut self, name: &str) {
        let Some((_, path)) = self.user_patches.iter().find(|(n, _)| n == name).cloned() else {
            return;
        };
        match genome::load_with_note(&path.to_string_lossy()) {
            Ok((g, note)) => {
                self.preset_idx = None;
                self.user_sel = Some(name.to_string());
                self.patch_name = name.to_string();
                let note = note.unwrap_or(self.note);
                self.set_patch(g, note, true);
                self.status = format!("patch: {name}");
            }
            Err(e) => self.status = format!("couldn't load {name}: {e}"),
        }
    }

    fn save_current_patch(&mut self) {
        let name: String = self
            .patch_name
            .trim()
            .chars()
            .map(|c| if c.is_alphanumeric() || c == '-' || c == '_' { c } else { '_' })
            .collect();
        let name = if name.is_empty() { "my_patch".to_string() } else { name };
        self.status = format!("saving {name}...");
        let job = Job::SavePatch { genome: self.genome, note: self.note, unison: self.unison, name };
        self.send_job(job);
    }
}

fn match_thread(target: Vec<f32>, tx: mpsc::Sender<MatchMsg>) {
    let fail = |tx: &mpsc::Sender<MatchMsg>, e: anyhow::Error| {
        let _ = tx.send(MatchMsg::Failed(e.to_string()));
    };
    let net = match Net::load() {
        Ok(n) => n,
        Err(e) => return fail(&tx, e),
    };
    let midi = crate::dsp::detect_midi(&target, synth::SR);
    let guess = match matcher::guess(&net, &target) {
        Ok(g) => g,
        Err(e) => return fail(&tx, e),
    };
    let l0 = matcher::loss_of(&guess, &target, midi as f32, 0);
    const GENS: usize = 60;
    let (g, l1) = matcher::refine(&guess, &target, midi as f32, GENS, 0, |gen, best| {
        let _ = tx.send(MatchMsg::Progress(gen * 100 / GENS, best));
    });
    let _ = tx.send(MatchMsg::Done { genome: g, midi, l0, l1 });
}

// ---- render worker (cold path) ------------------------------------------------
// All synth/DSP/NN work runs here so the UI thread only paints and plays cached
// audio. Rapid requests (plant/slider drags) coalesce to latest-wins.

/// Identity of a rendered patch preview: (genome, note, unison bits).
type PatchKey = (Genome, i32, u32);

/// Everything a loop render depends on — changes while looping trigger a hot swap.
/// Last field: 0 = built-in pattern, else the import id of the MIDI groove.
type LoopSig = (Genome, i32, usize, u32, u32, u32, u64);

enum Job {
    Patch { genome: Genome, note: i32, unison: f32, play: bool },
    Loop {
        genome: Genome,
        root: i32,
        pattern_idx: usize,
        bpm: f32,
        swing: f32,
        unison: f32,
        groove: Option<(Vec<loops::Ev>, f32)>,
        save: Option<String>,
    },
    Seedling { genome: Genome, note: i32, unison: f32 },
    Grow { seed: Genome, arch: usize, note: i32, amount: f32, rng_seed: u64 },
    LuckyDip { note: i32, rng_seed: u64 },
    SavePatch { genome: Genome, note: i32, unison: f32, name: String },
}

enum Out {
    Patch { audio: Vec<f32>, sr: f32, play: bool, cue: f32, key: PatchKey },
    LoopReady { audio: Vec<f32>, sr: f32 },
    Seedling { audio: Vec<f32>, sr: f32 },
    Grown { seedlings: Vec<Seedling>, arch: usize },
    Lucky { genome: Option<Genome> },
    SavedPatch { name: Option<String>, msg: String },
    Status(String),
    Idle,
}

fn post_dsp(audio: Vec<f32>, sr: f32, unison: f32, looped: bool) -> Vec<f32> {
    let mut a = crate::dsp::thicken(&audio, sr, unison);
    crate::dsp::safety(&mut a, sr, looped);
    a
}

/// Drop superseded preview jobs, keeping only the newest of each kind.
/// Saves are never dropped. A play request survives coalescing.
fn coalesce(q: &mut std::collections::VecDeque<Job>) {
    let (mut seen_patch, mut seen_loop, mut seen_seed, mut seen_grow, mut seen_lucky) =
        (false, false, false, false, false);
    let mut play_any = false;
    let mut keep: Vec<Job> = Vec::with_capacity(q.len());
    while let Some(j) = q.pop_back() {
        // newest-first walk: the first of a kind we meet is the one to keep
        let fresh = match &j {
            Job::Patch { play, .. } => {
                play_any |= *play;
                !std::mem::replace(&mut seen_patch, true)
            }
            Job::Loop { save: None, .. } => !std::mem::replace(&mut seen_loop, true),
            Job::Seedling { .. } => !std::mem::replace(&mut seen_seed, true),
            Job::Grow { .. } => !std::mem::replace(&mut seen_grow, true),
            Job::LuckyDip { .. } => !std::mem::replace(&mut seen_lucky, true),
            _ => true,
        };
        if fresh {
            keep.push(j);
        }
    }
    for j in keep.into_iter().rev() {
        q.push_back(j);
    }
    if play_any {
        for j in q.iter_mut() {
            if let Job::Patch { play, .. } = j {
                *play = true;
            }
        }
    }
}

fn run_job(net: &Net, job: Job) -> Out {
    match job {
        Job::Patch { genome, note, unison, play } => {
            let mut rng = SmallRng::seed_from_u64(0);
            let raw = synth::render_default(&genome, note as f32, &mut rng);
            let audio = post_dsp(raw, synth::SR, unison, false);
            let cue = garden::score(net, &genome);
            Out::Patch { audio, sr: synth::SR, play, cue, key: (genome, note, unison.to_bits()) }
        }
        Job::Loop { genome, root, pattern_idx, bpm, swing, unison, groove, save } => {
            let mut rng = SmallRng::seed_from_u64(0);
            let raw = match &groove {
                Some((evs, beats)) => loops::render_events(&genome, root, bpm, evs, *beats, &mut rng),
                None => {
                    let pat = loops::pattern(loops::PATTERN_NAMES[pattern_idx]).unwrap();
                    loops::render_loop(&genome, root, bpm, &pat, 2, swing, &mut rng)
                }
            };
            let audio = post_dsp(raw, loops::SR_OUT, unison, true);
            match save {
                Some(name) => match wavio::write_wav(&name, &audio, loops::SR_OUT) {
                    Ok(()) => Out::Status(format!("saved {name}")),
                    Err(e) => Out::Status(format!("save failed: {e}")),
                },
                None => Out::LoopReady { audio, sr: loops::SR_OUT },
            }
        }
        Job::Seedling { genome, note, unison } => {
            let mut rng = SmallRng::seed_from_u64(0);
            let raw = synth::render_default(&genome, note as f32, &mut rng);
            Out::Seedling { audio: post_dsp(raw, synth::SR, unison, false), sr: synth::SR }
        }
        Job::Grow { seed, arch, note, amount, rng_seed } => {
            let mut rng = SmallRng::seed_from_u64(rng_seed);
            let a = if arch == 0 { None } else { Some(garden::ARCHETYPE_NAMES[arch - 1]) };
            Out::Grown { seedlings: garden::grow(net, &seed, a, note, 8, amount, &mut rng), arch }
        }
        Job::LuckyDip { note, rng_seed } => {
            let mut rng = SmallRng::seed_from_u64(rng_seed);
            Out::Lucky { genome: garden::lucky_dip(net, note, 12, &mut rng) }
        }
        Job::SavePatch { genome, note, unison, name } => {
            let dir = patches_dir();
            if let Err(e) = std::fs::create_dir_all(&dir) {
                return Out::SavedPatch { name: None, msg: format!("couldn't create {}: {e}", dir.display()) };
            }
            let gpath = dir.join(format!("{name}.genome.txt"));
            match genome::save_patch(&gpath, &genome, note) {
                Ok(()) => {
                    let mut rng = SmallRng::seed_from_u64(0);
                    let raw = synth::render_default(&genome, note as f32, &mut rng);
                    let audio = post_dsp(raw, synth::SR, unison, false);
                    let _ = wavio::write_wav(&dir.join(format!("{name}.wav")).to_string_lossy(), &audio, synth::SR);
                    Out::SavedPatch { msg: format!("saved {name} -> {}", dir.display()), name: Some(name) }
                }
                Err(e) => Out::SavedPatch { name: None, msg: format!("save failed: {e}") },
            }
        }
    }
}

fn worker_thread(net: Arc<Net>, rx: mpsc::Receiver<Job>, tx: mpsc::Sender<Out>, ctx: egui::Context) {
    let mut queue = std::collections::VecDeque::new();
    loop {
        if queue.is_empty() {
            match rx.recv() {
                Ok(j) => queue.push_back(j),
                Err(_) => return, // app closed
            }
        }
        while let Ok(j) = rx.try_recv() {
            queue.push_back(j);
        }
        coalesce(&mut queue);
        let job = queue.pop_front().unwrap();
        if tx.send(run_job(&net, job)).is_err() {
            return;
        }
        if queue.is_empty() {
            match rx.try_recv() {
                Ok(j) => queue.push_back(j),
                Err(_) => {
                    let _ = tx.send(Out::Idle);
                }
            }
        }
        ctx.request_repaint();
    }
}

// ---- style helpers ------------------------------------------------------------

fn theme(ctx: &egui::Context) {
    let mut v = egui::Visuals::dark();
    v.panel_fill = BG;
    v.window_fill = PANEL;
    v.override_text_color = Some(TEXT);
    v.widgets.inactive.bg_fill = METAL;
    v.widgets.inactive.weak_bg_fill = METAL;
    v.widgets.inactive.bg_stroke = Stroke::new(1.0, PANEL_EDGE);
    v.widgets.hovered.bg_fill = SEED;
    v.widgets.hovered.weak_bg_fill = SEED;
    v.widgets.hovered.bg_stroke = Stroke::new(1.0, ACCENT);
    v.widgets.active.bg_fill = SEED;
    v.widgets.active.weak_bg_fill = SEED;
    v.selection.bg_fill = SEED;
    v.selection.stroke = Stroke::new(1.0, ACCENT);
    v.widgets.inactive.fg_stroke = Stroke::new(1.0, TEXT);
    v.widgets.hovered.fg_stroke = Stroke::new(1.2, ACCENT_HOT);
    v.widgets.active.fg_stroke = Stroke::new(1.2, ACCENT_HOT);
    v.widgets.noninteractive.fg_stroke = Stroke::new(1.0, TEXT);
    v.widgets.noninteractive.bg_stroke = Stroke::new(1.0, SEED);
    v.widgets.inactive.rounding = 6.0.into();
    v.widgets.hovered.rounding = 6.0.into();
    v.widgets.active.rounding = 6.0.into();
    ctx.set_visuals(v);
    let mut style = (*ctx.style()).clone();
    style.spacing.button_padding = Vec2::new(10.0, 5.0);
    ctx.set_style(style);
}

/// Layered translucent circles = cheap bloom.
fn glow(p: &egui::Painter, c: Pos2, r: f32, col: Color32, strength: f32) {
    for (m, a) in [(2.3, 0.08), (1.6, 0.16), (1.15, 0.28)] {
        p.circle_filled(c, r * m, col.gamma_multiply(a * strength));
    }
}

/// Framed hardware panel: shadowed plate, edge highlight, corner screws.
fn panel<R>(ui: &mut egui::Ui, title: &str, add: impl FnOnce(&mut egui::Ui) -> R) -> R {
    let frame = egui::Frame::none()
        .fill(PANEL)
        .stroke(Stroke::new(1.0, PANEL_EDGE))
        .rounding(10.0)
        .inner_margin(egui::Margin::symmetric(12.0, 10.0))
        .shadow(egui::epaint::Shadow {
            offset: Vec2::new(0.0, 3.0),
            blur: 10.0,
            spread: 0.0,
            color: Color32::from_black_alpha(140),
        });
    let ir = frame.show(ui, |ui| {
        ui.label(egui::RichText::new(title.to_uppercase()).color(DIM).size(10.5).strong());
        ui.add_space(2.0);
        add(ui)
    });
    let r = ir.response.rect;
    let p = ui.painter();
    // top edge light catch
    p.line_segment(
        [Pos2::new(r.left() + 10.0, r.top() + 1.0), Pos2::new(r.right() - 10.0, r.top() + 1.0)],
        Stroke::new(1.0, Color32::from_white_alpha(7)),
    );
    // corner screws
    for c in [
        Pos2::new(r.left() + 9.0, r.top() + 9.0),
        Pos2::new(r.right() - 9.0, r.top() + 9.0),
        Pos2::new(r.left() + 9.0, r.bottom() - 9.0),
        Pos2::new(r.right() - 9.0, r.bottom() - 9.0),
    ] {
        p.circle_filled(c, 2.6, METAL);
        p.circle_stroke(c, 2.6, Stroke::new(0.8, METAL_HI));
        p.line_segment([c + Vec2::new(-1.4, -1.4), c + Vec2::new(1.4, 1.4)], Stroke::new(0.8, CORE));
    }
    ir.inner
}

fn lerp_color(a: Color32, b: Color32, t: f32) -> Color32 {
    let t = t.clamp(0.0, 1.0);
    let l = |x: u8, y: u8| (x as f32 + (y as f32 - x as f32) * t) as u8;
    Color32::from_rgb(l(a.r(), b.r()), l(a.g(), b.g()), l(a.b(), b.b()))
}

/// Small labelled chip for the plant's param tags.
fn chip(p: &egui::Painter, pos: Pos2, text: &str, on: bool) {
    let galley = p.layout_no_wrap(
        text.to_string(),
        FontId::proportional(10.0),
        if on { TEXT } else { DIM },
    );
    let pad = Vec2::new(6.0, 2.5);
    let rect = Rect::from_center_size(pos, galley.size() + pad * 2.0);
    p.rect(rect, 4.0, Color32::from_rgb(12, 17, 13), Stroke::new(1.0, SEED));
    p.galley(rect.min + pad, galley, TEXT);
}

/// The plant: 16 branches around a glowing seed on a ticked dial.
fn plant(ui: &mut egui::Ui, g: &mut Genome, note: i32, size_hint: f32) -> bool {
    let side = ui.available_width().min(size_hint);
    let (resp, p) = ui.allocate_painter(Vec2::new(ui.available_width(), side), Sense::click_and_drag());
    let c = resp.rect.center();
    let r0 = side * 0.11;
    let r_max = side * 0.36;
    let mut changed = false;

    // dial plate
    p.circle_filled(c, r_max + 26.0, Color32::from_rgb(12, 17, 13));
    p.circle_stroke(c, r_max + 26.0, Stroke::new(1.0, PANEL_EDGE));
    p.circle_stroke(c, r_max + 25.0, Stroke::new(1.0, Color32::from_white_alpha(5)));
    // tick ring
    for i in 0..64 {
        let a = std::f32::consts::TAU * i as f32 / 64.0;
        let d = Vec2::new(a.cos(), a.sin());
        let (r1, r2) = if i % 4 == 0 { (r_max + 16.0, r_max + 22.0) } else { (r_max + 19.0, r_max + 22.0) };
        p.line_segment([c + d * r1, c + d * r2], Stroke::new(1.0, if i % 4 == 0 { DIM } else { SEED }));
    }
    for i in 1..=3 {
        p.circle_stroke(c, r0 + (r_max - r0) * i as f32 / 3.0, Stroke::new(1.0, SEED));
    }

    let angle_of = |i: usize| -> f32 {
        std::f32::consts::TAU * i as f32 / N_PARAMS as f32 - std::f32::consts::FRAC_PI_2
    };

    if resp.dragged() || resp.clicked() {
        if let Some(pos) = resp.interact_pointer_pos() {
            let d = pos - c;
            let dist = d.length();
            if dist > r0 * 0.6 && dist < r_max + 30.0 {
                let ang = d.y.atan2(d.x);
                let mut bi = 0usize;
                let mut bd = f32::MAX;
                for i in 0..N_PARAMS {
                    let mut da = (ang - angle_of(i)).abs();
                    da = da.min(std::f32::consts::TAU - da);
                    if da < bd {
                        bd = da;
                        bi = i;
                    }
                }
                g[bi] = ((dist - r0) / (r_max - r0)).clamp(0.0, 1.0);
                changed = true;
            }
        }
    }

    for i in 0..N_PARAMS {
        let a = angle_of(i);
        let dir = Vec2::new(a.cos(), a.sin());
        let v = g[i];
        let tip = c + dir * (r0 + v * (r_max - r0));
        let col = lerp_color(DIM, ACCENT, v);
        let bend = Vec2::new(-dir.y, dir.x) * 6.0 * (i as f32 * 2.399).sin();
        let mid = c + dir * (r0 + v * (r_max - r0) * 0.55) + bend;
        p.line_segment([c + dir * r0, mid], Stroke::new(2.0, col));
        p.line_segment([mid, tip], Stroke::new(2.0, col));
        if v > 0.05 {
            glow(&p, tip, 4.5 + 3.0 * v, ACCENT, v);
        }
        p.circle_filled(tip, 4.5 + 3.0 * v, col);
        p.circle_stroke(tip, 4.5 + 3.0 * v, Stroke::new(0.8, Color32::from_white_alpha(20)));
        p.circle_filled(tip, 1.8 + 1.2 * v, CORE);
        let lp = c + dir * (r_max + 38.0 + 6.0 * dir.x.abs());
        chip(&p, lp, PARAMS[i].0, v > 0.02);
    }

    // seed: glow + leaf + note
    glow(&p, c, r0 * 0.9, ACCENT, 0.8);
    p.circle_filled(c, r0 * 0.82, SEED);
    p.circle_stroke(c, r0 * 0.82, Stroke::new(1.5, ACCENT));
    p.circle_stroke(c, r0 * 0.70, Stroke::new(1.0, Color32::from_white_alpha(8)));
    // simple leaf silhouette behind the note (lens shape, rotated 45°)
    let leaf_pt = |t: f32, side: f32| -> Pos2 {
        let w = (std::f32::consts::PI * t).sin() * r0 * 0.26 * side;
        let along = (t - 0.5) * r0 * 1.15;
        // rotate (along, w) by 45°
        c + Vec2::new((along - w) * 0.707, (along + w) * 0.707)
    };
    let mut leaf: Vec<Pos2> = (0..=16).map(|i| leaf_pt(i as f32 / 16.0, 1.0)).collect();
    leaf.extend((0..=16).rev().map(|i| leaf_pt(i as f32 / 16.0, -1.0)));
    p.add(egui::Shape::convex_polygon(leaf, ACCENT.gamma_multiply(0.10), Stroke::NONE));
    p.text(c, Align2::CENTER_CENTER, genome::midi_to_name(note), FontId::monospace(16.0), ACCENT_HOT);
    changed
}

/// Hardware-style slider: recessed groove, glowing fill, ridged metal thumb.
fn param_slider_w(ui: &mut egui::Ui, v: &mut f32, width: f32) -> bool {
    let (resp, p) = ui.allocate_painter(Vec2::new(width, 18.0), Sense::click_and_drag());
    let rect = resp.rect;
    let pad = 7.0;
    let (x0, x1) = (rect.left() + pad, rect.right() - pad);
    let y = rect.center().y;
    let mut changed = false;
    if resp.dragged() || resp.clicked() {
        if let Some(pos) = resp.interact_pointer_pos() {
            *v = ((pos.x - x0) / (x1 - x0)).clamp(0.0, 1.0);
            changed = true;
        }
    }
    let hot = resp.hovered() || resp.dragged();
    let col = lerp_color(DIM, if hot { ACCENT_HOT } else { ACCENT }, *v);
    let hx = x0 + *v * (x1 - x0);
    // groove
    let groove = Rect::from_min_max(Pos2::new(x0, y - 2.5), Pos2::new(x1, y + 2.5));
    p.rect(groove, 2.5, CORE, Stroke::new(1.0, Color32::from_black_alpha(180)));
    p.line_segment(
        [Pos2::new(x0, y + 3.0), Pos2::new(x1, y + 3.0)],
        Stroke::new(1.0, Color32::from_white_alpha(6)),
    );
    // fill
    if *v > 0.01 {
        let fill = Rect::from_min_max(Pos2::new(x0, y - 2.0), Pos2::new(hx, y + 2.0));
        p.rect(fill, 2.0, col.gamma_multiply(0.8), Stroke::NONE);
        glow(&p, Pos2::new(hx, y), 3.0, ACCENT, *v * if hot { 1.0 } else { 0.6 });
    }
    // ridged thumb
    let th = Rect::from_center_size(Pos2::new(hx, y), Vec2::new(11.0, 15.0));
    p.rect(th, 3.0, METAL, Stroke::new(1.0, if hot { ACCENT } else { METAL_HI }));
    for dx in [-2.5f32, 0.0, 2.5] {
        p.line_segment(
            [Pos2::new(hx + dx, y - 4.0), Pos2::new(hx + dx, y + 4.0)],
            Stroke::new(1.0, Color32::from_black_alpha(120)),
        );
    }
    changed
}

fn param_slider(ui: &mut egui::Ui, v: &mut f32) -> bool {
    param_slider_w(ui, v, 132.0)
}

/// Bordered value readout box.
fn value_box(ui: &mut egui::Ui, text: &str, on: bool) {
    let (rect, _) = ui.allocate_exact_size(Vec2::new(48.0, 18.0), Sense::hover());
    let p = ui.painter();
    p.rect(rect, 4.0, CORE, Stroke::new(1.0, SEED));
    p.text(
        rect.right_center() - Vec2::new(6.0, 0.0),
        Align2::RIGHT_CENTER,
        text,
        FontId::monospace(10.0),
        if on { ACCENT } else { DIM },
    );
}

fn fmt_real(x: f32) -> String {
    if x.abs() >= 100.0 {
        format!("{x:.0}")
    } else if x.abs() >= 10.0 {
        format!("{x:.1}")
    } else {
        format!("{x:.2}")
    }
}

/// Rotary knob (drag vertically) for the master volume.
fn knob(ui: &mut egui::Ui, v: &mut f32, label: &str) -> bool {
    let size = 34.0;
    let (resp, p) = ui.allocate_painter(Vec2::new(size + 8.0, size + 14.0), Sense::click_and_drag());
    let c = Pos2::new(resp.rect.center().x, resp.rect.top() + size / 2.0 + 2.0);
    let mut changed = false;
    if resp.dragged() {
        *v = (*v - resp.drag_delta().y * 0.006).clamp(0.0, 1.0);
        changed = true;
    }
    let r = size / 2.0 - 2.0;
    p.circle_filled(c, r, METAL);
    p.circle_stroke(c, r, Stroke::new(1.0, METAL_HI));
    p.circle_stroke(c, r - 2.0, Stroke::new(1.0, Color32::from_black_alpha(100)));
    let a0 = 0.75 * std::f32::consts::PI;
    let sweep = 1.5 * std::f32::consts::PI;
    let arc_pts = |t0: f32, t1: f32, rr: f32| -> Vec<Pos2> {
        (0..=20)
            .map(|i| {
                let t = a0 + t0 * sweep + (t1 - t0) * sweep * i as f32 / 20.0;
                c + Vec2::new(t.cos(), t.sin()) * rr
            })
            .collect()
    };
    p.add(egui::Shape::line(arc_pts(0.0, 1.0, r + 3.5), Stroke::new(1.5, SEED)));
    if *v > 0.01 {
        p.add(egui::Shape::line(arc_pts(0.0, *v, r + 3.5), Stroke::new(2.0, ACCENT)));
    }
    let ang = a0 + *v * sweep;
    let d = Vec2::new(ang.cos(), ang.sin());
    p.line_segment([c + d * (r * 0.35), c + d * (r * 0.85)], Stroke::new(2.0, ACCENT_HOT));
    p.text(
        Pos2::new(c.x, resp.rect.bottom() - 5.0),
        Align2::CENTER_CENTER,
        label,
        FontId::proportional(9.0),
        DIM,
    );
    changed
}

fn waveform(ui: &mut egui::Ui, audio: &[f32], color: Color32, width: f32, height: f32) {
    let (resp, p) = ui.allocate_painter(Vec2::new(width, height), Sense::hover());
    let rect = resp.rect;
    p.rect(rect, 4.0, CORE, Stroke::new(1.0, SEED));
    if audio.is_empty() {
        return;
    }
    let cols = rect.width() as usize;
    let per = (audio.len() / cols.max(1)).max(1);
    let mid = rect.center().y;
    let half = rect.height() * 0.44;
    for x in 0..cols {
        let s = x * per;
        let e = (s + per).min(audio.len());
        if s >= e {
            break;
        }
        let (mut lo, mut hi) = (0.0f32, 0.0f32);
        for v in &audio[s..e] {
            lo = lo.min(*v);
            hi = hi.max(*v);
        }
        let px = rect.left() + x as f32;
        p.line_segment(
            [Pos2::new(px, mid - hi * half), Pos2::new(px, mid - lo * half)],
            Stroke::new(1.0, color),
        );
    }
}

/// Segmented level meter, lit from the live playback position.
fn meter(ui: &mut egui::Ui, level: f32, height: f32) {
    let (resp, p) = ui.allocate_painter(Vec2::new(24.0, height), Sense::hover());
    let rect = resp.rect;
    p.rect(rect, 4.0, CORE, Stroke::new(1.0, SEED));
    let n = (((height - 8.0) / 5.0) as usize).max(4);
    let lit = (level.clamp(0.0, 1.0) * n as f32).ceil() as usize;
    for i in 0..n {
        let frac = i as f32 / n as f32;
        let y1 = rect.bottom() - 4.0 - frac * (rect.height() - 8.0);
        let seg = Rect::from_min_max(
            Pos2::new(rect.left() + 5.0, y1 - 3.0),
            Pos2::new(rect.right() - 5.0, y1 - 1.0),
        );
        let base = if i + 2 >= n { Color32::from_rgb(230, 210, 90) } else { ACCENT };
        let col = if i < lit { base } else { base.gamma_multiply(0.12) };
        p.rect(seg, 1.0, col, Stroke::NONE);
    }
}

/// Seedling bud: glow scales with reward score.
fn bud(ui: &mut egui::Ui, score: f32) -> egui::Response {
    let (resp, mut p) = ui.allocate_painter(Vec2::new(34.0, 34.0), Sense::click());
    p.set_clip_rect(resp.rect.expand(18.0)); // glow blooms past the hit rect
    let c = resp.rect.center();
    let col = lerp_color(DIM, ACCENT, score);
    let r = 9.0 + 5.0 * score;
    glow(&p, c, r, ACCENT, score);
    if resp.hovered() {
        p.circle_stroke(c, r + 3.5, Stroke::new(1.5, ACCENT_HOT));
    }
    p.circle_filled(c, r, col);
    p.circle_stroke(c, r, Stroke::new(0.8, Color32::from_white_alpha(18)));
    p.circle_filled(c, r * 0.35, CORE);
    resp
}

// ---- update -------------------------------------------------------------------

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.frame += 1;
        if self.shot.is_some() && self.frame == 2 {
            self.grow_arch = 5; // "stab" — a fuller screenshot
            self.grow();
            if self.shot.as_deref().map(|s| s.contains("help")).unwrap_or(false) {
                self.show_help = true;
            }
        }

        let mut msgs = Vec::new();
        if let Some(rx) = &self.rx {
            while let Ok(m) = rx.try_recv() {
                msgs.push(m);
            }
            ctx.request_repaint();
        }
        for m in msgs {
            match m {
                MatchMsg::Progress(pct, best) => {
                    self.progress = pct as f32 / 100.0;
                    self.status = format!("growing... {pct}%  spec-loss {best:.3}");
                }
                MatchMsg::Done { genome, midi, l0, l1 } => {
                    self.rx = None;
                    self.preset_idx = None;
                    self.set_patch(genome, midi, true);
                    self.status = format!(
                        "cloned {} at {}  ·  spec-loss {l0:.2} → {l1:.2}  ·  cue score {:.0}%",
                        self.target_name,
                        genome::midi_to_name(midi),
                        garden::score(&self.net, &self.genome) * 100.0
                    );
                }
                MatchMsg::Failed(e) => {
                    self.status = format!("match failed: {e}");
                    self.rx = None;
                }
            }
        }

        let dropped: Vec<_> = ctx.input(|i| i.raw.dropped_files.clone());
        if let Some(f) = dropped.first() {
            if let Some(path) = &f.path {
                let p = path.to_string_lossy().into_owned();
                let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("").to_ascii_lowercase();
                if ext == "mid" || ext == "midi" {
                    self.load_midi_groove(&p);
                } else {
                    self.start_match(p);
                }
            }
        }

        self.drain_worker();

        // live loop lab: while a loop plays, any edit to what it's built from
        // (pattern, bpm, swing, key, the genome itself) re-renders and hot-swaps
        if self.shot.is_none() && self.audio.loop_playing() {
            if let Ok(root) = genome::note_to_midi(&self.key) {
                if self.last_loop_sig != Some(self.loop_sig(root)) {
                    self.request_loop(false);
                }
            }
        }

        // ---- keyboard shortcuts (see the ⌨ panel, top right) --------------
        if ctx.input_mut(|i| i.consume_key(egui::Modifiers::COMMAND, egui::Key::Z)) {
            self.do_undo();
        }
        if ctx.input_mut(|i| i.consume_key(egui::Modifiers::COMMAND, egui::Key::Y)) {
            self.do_redo();
        }
        let popup_open = ctx.memory(|m| m.any_popup_open()); // combo dropdowns own the arrows
        if !ctx.wants_keyboard_input() && !popup_open && self.shot.is_none() {
            let hit = |k: egui::Key, m: egui::Modifiers| ctx.input_mut(|i| i.consume_key(m, k));
            use egui::{Key, Modifiers as M};
            if hit(Key::Space, M::NONE) {
                self.toggle_play();
            }
            if hit(Key::ArrowLeft, M::NONE) {
                self.step_preset(-1);
            }
            if hit(Key::ArrowRight, M::NONE) {
                self.step_preset(1);
            }
            if hit(Key::ArrowUp, M::NONE) {
                self.nudge_note(1);
            }
            if hit(Key::ArrowDown, M::NONE) {
                self.nudge_note(-1);
            }
            if hit(Key::R, M::NONE) {
                self.random_patch();
            }
            if hit(Key::G, M::NONE) {
                self.grow();
            }
            if hit(Key::L, M::NONE) {
                self.toggle_loop();
            }
            if hit(Key::A, M::NONE) {
                self.swap_ab();
            }
            if hit(Key::S, M::COMMAND) {
                self.save_current_patch();
            }
            if hit(Key::K, M::NONE) {
                self.show_keys = !self.show_keys;
            }
            if hit(Key::H, M::NONE) || hit(Key::F1, M::NONE) {
                self.show_help = !self.show_help;
            }
            if hit(Key::Escape, M::NONE) {
                if self.show_help || self.show_keys {
                    self.show_help = false;
                    self.show_keys = false;
                } else {
                    self.audio.stop();
                }
            }
        }

        if self.audio.is_playing() {
            ctx.request_repaint_after(std::time::Duration::from_millis(50));
        }

        // ---- fit to screen ----------------------------------------------
        // Zoom the whole UI so the layout fits the window. The fit is computed
        // in physical pixels — invariant under zoom — so it cannot oscillate.
        // Skipped in screenshot mode to keep docs images at 1:1 scale.
        if self.shot.is_none() {
            if self.frame == 1 {
                ctx.options_mut(|o| o.zoom_with_keyboard = false); // zoom is driven below
            }
            if !self.fitted_to_monitor {
                // one-shot: never open larger than the monitor's usable area
                let vp = ctx.input(|i| (i.viewport().monitor_size, i.viewport().inner_rect));
                if let (Some(mon), Some(inner)) = vp {
                    self.fitted_to_monitor = true;
                    let usable = Vec2::new(mon.x * 0.94, mon.y * 0.90); // taskbar/border slack
                    if inner.width() > usable.x || inner.height() > usable.y {
                        ctx.send_viewport_cmd(egui::ViewportCommand::InnerSize(inner.size().min(usable)));
                        if let Some(cmd) = egui::ViewportCommand::center_on_screen(ctx) {
                            ctx.send_viewport_cmd(cmd);
                        }
                    }
                }
            }
            const DESIGN_W: f32 = 956.0; // 940 content column + margins + scrollbar
            const MIN_ZOOM: f32 = 0.65; // readability floor — below this we scroll instead
            const MAX_ZOOM: f32 = 1.6;
            let native = ctx.native_pixels_per_point().unwrap_or(1.0);
            let phys = ctx.screen_rect().size() * ctx.pixels_per_point();
            let design_h = self.content_h + 16.0; // + CentralPanel margins
            let fit = (phys.x / (native * DESIGN_W)).min(phys.y / (native * design_h));
            let zoom = fit.clamp(MIN_ZOOM, MAX_ZOOM);
            if (zoom - ctx.zoom_factor()).abs() > 0.005 {
                ctx.set_zoom_factor(zoom);
            }
        }

        egui::CentralPanel::default().show(ctx, |ui| {
            egui::ScrollArea::both().auto_shrink([false, false]).show(ui, |ui| {
                // center a capped-width content column when the window is wide
                let full = ui.available_width();
                let content_w = full.min(940.0);
                let pad = ((full - content_w) / 2.0).max(0.0);
                ui.horizontal_top(|ui| {
                ui.add_space(pad);
                ui.vertical(|ui| {
                ui.set_width(content_w);

                self.top_bar(ui);
                ui.add_space(2.0);
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new(&self.status).size(12.5).color(TEXT));
                    if self.busy {
                        ui.add(egui::Spinner::new().size(12.0).color(ACCENT));
                    }
                });
                if self.rx.is_some() {
                    ui.add(egui::ProgressBar::new(self.progress).desired_width(300.0));
                }
                ui.add_space(6.0);

                let avail = ui.available_width();
                let right_w = 312.0;
                let stack = avail < 700.0; // narrow window: stack panels vertically
                let left_w = if stack { avail - 4.0 } else { (avail - right_w - 24.0).clamp(360.0, 560.0) };

                let plant_panel = |app: &mut App, ui: &mut egui::Ui| {
                    panel(ui, "shape — drag the plant", |ui| {
                        ui.set_width(left_w - 24.0);
                        if plant(ui, &mut app.genome, app.note, left_w - 40.0) {
                            app.preset_idx = None;
                            app.request_patch(false);
                        }
                        ui.horizontal(|ui| {
                            if ui.button("▶ play").on_hover_text("space").clicked() {
                                app.play_patch();
                            }
                            if ui.button("−").clicked() {
                                app.nudge_note(-1);
                            }
                            ui.label(egui::RichText::new(genome::midi_to_name(app.note)).monospace());
                            if ui.button("+").clicked() {
                                app.nudge_note(1);
                            }
                            ui.add_space(6.0);
                            let cue = app.cue;
                            ui.label(
                                egui::RichText::new(format!("taste {:.0}%", cue * 100.0))
                                    .monospace()
                                    .size(11.0)
                                    .color(lerp_color(DIM, ACCENT_HOT, cue)),
                            )
                            .on_hover_text("the taste model's guess at how good this sounds\n(trained on juxxs's star ratings)");
                            ui.add_space(6.0);
                            ui.label(egui::RichText::new("unison").color(DIM).small());
                            if param_slider_w(ui, &mut app.unison, 120.0) {
                                app.request_patch(false);
                            }
                            value_box(ui, &format!("{:.0}%", app.unison * 100.0), app.unison > 0.01);
                        });
                        ui.add_space(2.0);
                        egui::CollapsingHeader::new(
                            egui::RichText::new("fine-tune").color(DIM).size(11.0),
                        )
                        .default_open(false)
                        .show(ui, |ui| {
                            let mut changed = false;
                            let reals = genome::denorm(&app.genome);
                            for (title, lo, hi) in
                                [("oscillators", 0usize, 6usize), ("filter & drive", 6, 12), ("envelope", 12, 16)]
                            {
                                ui.add_space(2.0);
                                ui.label(egui::RichText::new(title.to_uppercase()).color(DIM).size(9.5));
                                egui::Grid::new(title).num_columns(3).spacing([8.0, 3.0]).show(ui, |ui| {
                                    for i in lo..hi {
                                        ui.label(egui::RichText::new(PARAMS[i].0).monospace().size(11.0));
                                        changed |= param_slider_w(ui, &mut app.genome[i], 200.0);
                                        value_box(ui, &fmt_real(reals[i]), app.genome[i] > 0.02);
                                        ui.end_row();
                                    }
                                });
                            }
                            if changed {
                                app.preset_idx = None;
                                app.request_patch(false);
                            }
                        });
                    });
                };

                let right_panels = |app: &mut App, ui: &mut egui::Ui| {
                    panel(ui, "grow — breed it to taste (g)", |ui| {
                        ui.horizontal(|ui| {
                            ui.label(egui::RichText::new("seed").color(DIM).small());
                            egui::ComboBox::from_id_salt("arch")
                                .selected_text(if app.grow_arch == 0 {
                                    "this patch"
                                } else {
                                    garden::ARCHETYPE_NAMES[app.grow_arch - 1]
                                })
                                .width(110.0)
                                .show_ui(ui, |ui| {
                                    ui.selectable_value(&mut app.grow_arch, 0, "this patch");
                                    for (i, a) in garden::ARCHETYPE_NAMES.iter().enumerate() {
                                        ui.selectable_value(&mut app.grow_arch, i + 1, *a);
                                    }
                                });
                            if ui.button("🌱 grow").clicked() {
                                app.grow();
                            }
                        });
                        ui.horizontal(|ui| {
                            ui.label(egui::RichText::new("wildness").color(DIM).small());
                            param_slider(ui, &mut app.grow_amount);
                        });
                        if !app.seedlings.is_empty() {
                            ui.add_space(4.0);
                            ui.horizontal_wrapped(|ui| {
                                let mut adopt: Option<usize> = None;
                                for i in 0..app.seedlings.len() {
                                    ui.vertical(|ui| {
                                        let s = &app.seedlings[i];
                                        if bud(ui, s.score).clicked() {
                                            let job = Job::Seedling {
                                                genome: s.genome,
                                                note: app.note,
                                                unison: app.unison,
                                            };
                                            app.send_job(job);
                                        }
                                        ui.label(
                                            egui::RichText::new(format!("{:.0}%", app.seedlings[i].score * 100.0))
                                                .monospace()
                                                .size(10.0)
                                                .color(DIM),
                                        );
                                        if ui.small_button("✔").clicked() {
                                            adopt = Some(i);
                                        }
                                    });
                                }
                                if let Some(i) = adopt {
                                    let g = app.seedlings[i].genome;
                                    app.preset_idx = None;
                                    let note = app.note;
                                    app.set_patch(g, note, true);
                                    app.grow_arch = 0;
                                    app.grow();
                                    app.status = "adopted — growing the next generation from it".into();
                                }
                            });
                            ui.label(egui::RichText::new("click = hear · ✔ = keep").color(DIM).small());
                        }
                    });
                    ui.add_space(6.0);
                    panel(ui, "loop lab (l)", |ui| {
                        ui.horizontal(|ui| {
                            let sel = if app.use_groove {
                                app.groove.as_ref().map(|g| format!("🎹 {}", g.name)).unwrap_or_default()
                            } else {
                                loops::PATTERN_NAMES[app.pattern_idx].to_string()
                            };
                            egui::ComboBox::from_id_salt("pat")
                                .selected_text(sel)
                                .show_ui(ui, |ui| {
                                    for (i, name) in loops::PATTERN_NAMES.iter().enumerate() {
                                        if ui.selectable_label(!app.use_groove && app.pattern_idx == i, *name).clicked() {
                                            app.pattern_idx = i;
                                            app.use_groove = false;
                                        }
                                    }
                                    if let Some(g) = &app.groove {
                                        ui.separator();
                                        if ui.selectable_label(app.use_groove, format!("🎹 {}", g.name)).clicked() {
                                            app.use_groove = true;
                                        }
                                    }
                                });
                            ui.add(egui::DragValue::new(&mut app.bpm).range(80.0..=180.0).suffix(" bpm"));
                            ui.add(egui::TextEdit::singleline(&mut app.key).desired_width(30.0));
                        });
                        ui.horizontal(|ui| {
                            if app.use_groove {
                                let bars = app.groove.as_ref().map(|g| g.bars()).unwrap_or(1);
                                ui.label(
                                    egui::RichText::new(format!("your midi groove · {bars} bars · key moves it"))
                                        .color(DIM)
                                        .small(),
                                );
                            } else {
                                ui.label(egui::RichText::new("swing").color(DIM).small());
                                param_slider(ui, &mut app.swing);
                                value_box(ui, &format!("{:.0}%", app.swing * 100.0), app.swing > 0.01);
                            }
                        });
                        ui.horizontal(|ui| {
                            let playing = app.audio.loop_playing();
                            let label = if playing { "⏹ stop" } else { "▶ loop" };
                            if ui.button(label).on_hover_text("L").clicked() {
                                app.toggle_loop();
                            }
                            if ui.button("💾 save loop").clicked() {
                                app.request_loop(true);
                            }
                            if playing {
                                ui.label(egui::RichText::new("• looping — edits update it live").color(ACCENT).small());
                            }
                        });
                    });
                };

                if stack {
                    plant_panel(self, ui);
                    ui.add_space(6.0);
                    right_panels(self, ui);
                } else {
                    ui.horizontal_top(|ui| {
                        ui.vertical(|ui| {
                            ui.set_width(left_w);
                            plant_panel(self, ui);
                        });
                        ui.add_space(8.0);
                        ui.vertical(|ui| {
                            ui.set_width(right_w);
                            right_panels(self, ui);
                        });
                    });
                }

                // waveform strip — passive readout, no panel chrome
                ui.add_space(8.0);
                let level = self.audio.level();
                ui.horizontal(|ui| {
                    let w = ui.available_width() - 34.0;
                    waveform(ui, &self.last_audio, ACCENT, w, 30.0);
                    meter(ui, level, 30.0);
                });
                if self.target.is_some() {
                    ui.add_space(3.0);
                    let (mut play_target, mut reclone) = (false, false);
                    ui.horizontal(|ui| {
                        ui.label(egui::RichText::new(format!("target · {}", self.target_name)).color(DIM).small());
                        play_target = ui.small_button("▶ target").clicked();
                        reclone = ui.small_button("↻ re-clone").clicked();
                    });
                    let w = ui.available_width();
                    waveform(ui, self.target.as_deref().unwrap_or(&[]), DIM, w, 26.0);
                    if play_target {
                        let mut safe = self.target.clone().unwrap();
                        crate::dsp::safety(&mut safe, synth::SR, false);
                        self.audio.play(&safe, synth::SR, false);
                    }
                    if reclone {
                        self.status = "re-cloning...".into();
                        self.progress = 0.0;
                        let (tx, rx) = mpsc::channel();
                        self.rx = Some(rx);
                        let target = self.target.clone().unwrap();
                        std::thread::spawn(move || match_thread(target, tx));
                    }
                }

                ui.add_space(40.0); // keep the ? button clear of content
                self.content_h = ui.min_rect().height(); // drives next frame's zoom fit

                }); // content column
                }); // centering row
            });
        });

        // floating round buttons: ? bottom-right, ⌨ top-right
        let round_btn = |id: &str, anchor: Align2, offset: Vec2, glyph: &str, on: bool| -> bool {
            let mut clicked = false;
            egui::Area::new(egui::Id::new(id)).anchor(anchor, offset).show(ctx, |ui| {
                let (resp, mut p) = ui.allocate_painter(Vec2::splat(30.0), Sense::click());
                // let the glow bloom past the widget rect — clipping it to the
                // 30px square reads as a square plate behind the circle
                p.set_clip_rect(resp.rect.expand(28.0));
                let c = resp.rect.center();
                let hot = resp.hovered() || on;
                glow(&p, c, 13.0, ACCENT, if hot { 0.8 } else { 0.3 });
                p.circle_filled(c, 13.0, if hot { SEED } else { METAL });
                p.circle_stroke(c, 13.0, Stroke::new(1.2, if hot { ACCENT } else { METAL_HI }));
                p.text(c, Align2::CENTER_CENTER, glyph, FontId::proportional(15.0), if hot { ACCENT_HOT } else { TEXT });
                clicked = resp.clicked();
            });
            clicked
        };
        if round_btn("help_btn", Align2::RIGHT_BOTTOM, Vec2::new(-14.0, -14.0), "?", self.show_help) {
            self.show_help = !self.show_help;
        }
        if round_btn("keys_btn", Align2::RIGHT_TOP, Vec2::new(-14.0, 14.0), "⌨", self.show_keys) {
            self.show_keys = !self.show_keys;
        }

        if self.show_keys {
            egui::Area::new(egui::Id::new("keys_panel"))
                .anchor(Align2::RIGHT_TOP, Vec2::new(-14.0, 52.0))
                .show(ctx, |ui| {
                    panel(ui, "keyboard", |ui| {
                        let key_chip = |ui: &mut egui::Ui, keys: &str| {
                            let galley = ui.painter().layout_no_wrap(
                                keys.to_string(),
                                FontId::monospace(10.0),
                                ACCENT,
                            );
                            let (rect, _) = ui.allocate_exact_size(galley.size() + Vec2::new(12.0, 5.0), Sense::hover());
                            let p = ui.painter();
                            p.rect(rect, 4.0, CORE, Stroke::new(1.0, SEED));
                            p.galley(rect.min + Vec2::new(6.0, 2.5), galley, ACCENT);
                        };
                        egui::Grid::new("keys").num_columns(2).spacing([10.0, 4.0]).show(ui, |ui| {
                            for (keys, what) in [
                                ("space", "play / stop"),
                                ("← →", "browse presets"),
                                ("↑ ↓", "note up / down"),
                                ("r", "random patch"),
                                ("g", "grow seedlings"),
                                ("l", "loop start / stop"),
                                ("a", "a/b swap"),
                                ("ctrl z · y", "undo / redo"),
                                ("ctrl s", "save patch"),
                                ("h · f1", "help"),
                                ("k", "this panel"),
                                ("esc", "stop sound / close"),
                            ] {
                                key_chip(ui, keys);
                                ui.label(egui::RichText::new(what).size(11.5));
                                ui.end_row();
                            }
                        });
                    });
                });
        }
        if self.show_help {
            let mut open = true;
            egui::Window::new("how to use synfection")
                .open(&mut open)
                .collapsible(false)
                .default_width((ctx.screen_rect().width() - 60.0).min(430.0))
                .anchor(Align2::CENTER_CENTER, Vec2::ZERO)
                .show(ctx, |ui| {
                    let help_h = (ctx.screen_rect().height() - 140.0).clamp(120.0, 430.0);
                    egui::ScrollArea::vertical().max_height(help_h).show(ui, |ui| {
                        let section = |ui: &mut egui::Ui, title: &str, lines: &[&str]| {
                            ui.label(egui::RichText::new(title).color(ACCENT).strong());
                            for l in lines {
                                ui.label(egui::RichText::new(format!("•  {l}")).size(12.5));
                            }
                            ui.add_space(8.0);
                        };
                        section(ui, "GET A SOUND", &[
                            "Pick a preset from the dropdown up top and press ▶ play.",
                            "Or drop any .wav onto the window (or ⬆ open wav) — synfection listens and rebuilds it as a patch you can play at any note.",
                            "🎲 random deals you a fresh sound, ranked by the taste model.",
                        ]);
                        section(ui, "SHAPE IT", &[
                            "Drag the plant's branches — longer branch = more of that ingredient. fine-tune (under the plant) opens precision sliders.",
                            "unison makes it thicker and wider. − / + moves the note.",
                            "A/B flips between two versions. ↺ ↻ are undo and redo (ctrl+z / ctrl+y).",
                            "taste % is the model's guess at how good the patch sounds.",
                        ]);
                        section(ui, "GROW IT", &[
                            "In grow (top right), pick a seed — this patch, or a style (bass, reese, stab, pad...).",
                            "Press 🌱 grow: eight buds appear. Brighter glow = the taste model likes it more.",
                            "Click a bud to hear it. Press ✔ to keep it — the next generation grows from your pick. Repeat until it's yours.",
                            "wildness controls how different the children are.",
                        ]);
                        section(ui, "MAKE LOOPS", &[
                            "loop lab plays your patch as a bassline groove — pick a pattern, bpm and key.",
                            "swing adds shuffle. ▶ loop plays seamlessly until ⏹ stop.",
                            "While it loops, everything is live: change the pattern, bpm, swing, or the patch itself and the loop follows in place.",
                            "Got your own groove? Drop a .mid from your DAW onto the window — it becomes a pattern. Its lowest note lands on the loop key, so it transposes.",
                            "💾 save loop writes a 44.1 kHz .wav next to the app.",
                        ]);
                        section(ui, "KEEP IT", &[
                            "Type a name up top and press 💾 save — patches live in Documents/synfection/patches.",
                            "They appear under YOUR PATCHES in the dropdown, every time you open the app.",
                        ]);
                        section(ui, "SAFETY", &[
                            "Everything you hear or save runs through a built-in limiter and loudness guard.",
                            "Random experiments can't clip, blast, or hurt your ears — go wild.",
                        ]);
                        section(ui, "SHORTCUTS", &[
                            "space play · ←→ presets · ↑↓ note · r random · g grow · l loop",
                            "a a/b · ctrl+z/y undo/redo · ctrl+s save · esc stop",
                            "Press k (or the ⌨ button, top right) for the full list.",
                        ]);
                    });
                });
            if !open {
                self.show_help = false;
            }
        }

        if let Some(path) = self.shot.clone() {
            if self.frame == 8 {
                ctx.send_viewport_cmd(egui::ViewportCommand::Screenshot);
            }
            let img = ctx.input(|i| {
                i.raw.events.iter().find_map(|e| match e {
                    egui::Event::Screenshot { image, .. } => Some(image.clone()),
                    _ => None,
                })
            });
            if let Some(image) = img {
                let [w, h] = image.size;
                let _ = image::save_buffer(&path, image.as_raw(), w as u32, h as u32, image::ColorType::Rgba8);
                println!("screenshot -> {path}");
                std::process::exit(0);
            }
            ctx.request_repaint();
        }
    }
}

impl App {
    fn top_bar(&mut self, ui: &mut egui::Ui) {
        ui.horizontal_wrapped(|ui| {
            // two-tone wordmark
            ui.spacing_mut().item_spacing.x = 0.0;
            ui.label(egui::RichText::new("syn").size(24.0).color(Color32::WHITE).strong());
            ui.label(egui::RichText::new("fection").size(24.0).color(ACCENT).strong());
            ui.spacing_mut().item_spacing.x = 8.0;
            ui.add_space(10.0);

            let cur = self
                .preset_idx
                .map(|i| PRESETS[i].name.to_string())
                .or_else(|| self.user_sel.clone())
                .unwrap_or_else(|| "custom".into());
            if ui.button("◀").on_hover_text("previous preset (←)").clicked() {
                self.step_preset(-1);
            }
            let mut pick: Option<usize> = None;
            let mut pick_user: Option<String> = None;
            egui::ComboBox::from_id_salt("preset")
                .selected_text(egui::RichText::new(cur).monospace())
                .width(126.0)
                .show_ui(ui, |ui| {
                    for (i, p) in PRESETS.iter().enumerate() {
                        if ui.selectable_label(self.preset_idx == Some(i), p.name).clicked() {
                            pick = Some(i);
                        }
                    }
                    if !self.user_patches.is_empty() {
                        ui.separator();
                        ui.label(egui::RichText::new("YOUR PATCHES").color(DIM).size(9.5));
                        for (name, _) in self.user_patches.clone() {
                            let on = self.user_sel.as_deref() == Some(name.as_str());
                            if ui.selectable_label(on, &name).clicked() {
                                pick_user = Some(name);
                            }
                        }
                    }
                });
            if let Some(i) = pick {
                self.load_preset(i);
            }
            if let Some(name) = pick_user {
                self.load_user_patch(&name);
            }
            if ui.button("▶").on_hover_text("next preset (→)").clicked() {
                self.step_preset(1);
            }

            let ab = if self.ab_is_b { "B/A" } else { "A/B" };
            if ui.button(ab).on_hover_text("swap with the other slot (A)").clicked() {
                self.swap_ab();
            }
            if ui.button("↺").on_hover_text("undo (ctrl+z)").clicked() {
                self.do_undo();
            }
            if ui.button("↻").on_hover_text("redo (ctrl+y)").clicked() {
                self.do_redo();
            }

            if ui.button("🎲 random").on_hover_text("surprise me — best of 12, taste-ranked (R)").clicked() {
                self.random_patch();
            }
            if ui
                .button("⬆ open")
                .on_hover_text("wav → clone it as a patch\nmid → loop it as your own groove")
                .clicked()
            {
                if let Some(path) = rfd::FileDialog::new()
                    .add_filter("wav or midi", &["wav", "mid", "midi"])
                    .pick_file()
                {
                    let p = path.to_string_lossy().into_owned();
                    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("").to_ascii_lowercase();
                    if ext == "mid" || ext == "midi" {
                        self.load_midi_groove(&p);
                    } else {
                        self.start_match(p);
                    }
                }
            }

            ui.add(
                egui::TextEdit::singleline(&mut self.patch_name)
                    .desired_width(96.0)
                    .hint_text("patch name"),
            );
            if ui
                .button("💾 save")
                .on_hover_text("save to your patch library (ctrl+s)\nDocuments/synfection/patches")
                .clicked()
            {
                self.save_current_patch();
            }

            let mut vol = self.audio.volume;
            if knob(ui, &mut vol, "master") {
                self.audio.set_volume(vol);
            }
        });
    }
}

pub fn run(genome_path: Option<String>, shot: Option<String>) -> Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([956.0, 860.0]) // full design size; clamped to monitor below
            .with_min_inner_size([420.0, 320.0])
            .with_clamp_size_to_monitor_size(true)
            .with_title("synfection"),
        ..Default::default()
    };
    eframe::run_native(
        "synfection",
        options,
        Box::new(move |cc| Ok(Box::new(App::new(cc, genome_path, shot)))),
    )
    .map_err(|e| anyhow::anyhow!("ui: {e}"))
}
