//! The synfection app: plant a sound, grow a patch.
//! Preset browser · A/B · undo/redo · drag&drop or open-file cloning ·
//! radial plant editor · reward-scored garden · loop lab · gapless audio engine.

use std::sync::mpsc;
use std::sync::Arc;

use anyhow::Result;
use eframe::egui::{self, Align2, Color32, FontId, Pos2, Sense, Stroke, Vec2};
use rand::rngs::SmallRng;
use rand::SeedableRng;
use rodio::Source;

use crate::garden::{self, Seedling};
use crate::genome::{self, Genome, N_PARAMS, PARAMS};
use crate::loops;
use crate::matcher;
use crate::net::Net;
use crate::presets::PRESETS;
use crate::synth;
use crate::wavio;

const BG: Color32 = Color32::from_rgb(10, 14, 11);
const PANEL: Color32 = Color32::from_rgb(15, 21, 16);
const DIM: Color32 = Color32::from_rgb(70, 95, 78);
const TEXT: Color32 = Color32::from_rgb(196, 220, 201);
const ACCENT: Color32 = Color32::from_rgb(92, 224, 138);
const ACCENT_HOT: Color32 = Color32::from_rgb(180, 255, 160);
const SEED: Color32 = Color32::from_rgb(34, 54, 38);
const CORE: Color32 = Color32::from_rgb(8, 12, 9);

// ---- audio engine -----------------------------------------------------------

struct AudioEngine {
    _stream: Option<rodio::OutputStream>,
    handle: Option<rodio::OutputStreamHandle>,
    sink: Option<rodio::Sink>,
    volume: f32,
    looping: bool,
}

impl AudioEngine {
    fn new() -> Self {
        let (stream, handle) = match rodio::OutputStream::try_default() {
            Ok((s, h)) => (Some(s), Some(h)),
            Err(_) => (None, None),
        };
        AudioEngine { _stream: stream, handle, sink: None, volume: 0.9, looping: false }
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
                self.looping = looped;
            }
        }
    }

    fn stop(&mut self) {
        if let Some(s) = self.sink.take() {
            s.stop();
        }
        self.looping = false;
    }

    fn set_volume(&mut self, v: f32) {
        self.volume = v;
        if let Some(s) = &self.sink {
            s.set_volume(v);
        }
    }

    fn loop_playing(&self) -> bool {
        self.looping && self.sink.as_ref().map(|s| !s.empty()).unwrap_or(false)
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
    // history / compare
    undo: Vec<(Genome, i32)>,
    redo: Vec<(Genome, i32)>,
    ab_other: (Genome, i32),
    ab_is_b: bool,
    // matching
    target: Option<Vec<f32>>,
    target_name: String,
    rx: Option<mpsc::Receiver<MatchMsg>>,
    progress: f32,
    // garden
    seedlings: Vec<Seedling>,
    grow_arch: usize, // 0 = this patch, 1.. = archetypes
    grow_amount: f32,
    // playback
    audio: AudioEngine,
    last_audio: Vec<f32>,
    last_sr: f32,
    // loop lab
    bpm: f32,
    key: String,
    pattern_idx: usize,
    swing: f32,
    // screenshot mode
    shot: Option<String>,
    frame: u64,
}

impl App {
    pub fn new(cc: &eframe::CreationContext<'_>, genome_path: Option<String>, shot: Option<String>) -> Self {
        theme(&cc.egui_ctx);
        let net = Arc::new(Net::load().expect("embedded weights"));
        let (genome, note, preset_idx) = match genome_path.and_then(|p| genome::load(&p).ok()) {
            Some(g) => (g, 36, None),
            None => (PRESETS[0].genome, PRESETS[0].note, Some(0)),
        };
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
            shot,
            frame: 0,
        };
        app.render_current();
        app
    }

    fn render_current(&mut self) {
        let mut rng = SmallRng::seed_from_u64(0);
        self.last_audio = synth::render_default(&self.genome, self.note as f32, &mut rng);
        self.last_sr = synth::SR;
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
        self.render_current();
        if play {
            self.audio.play(&self.last_audio, self.last_sr, false);
        }
    }

    fn play_patch(&mut self) {
        self.render_current();
        self.audio.play(&self.last_audio, self.last_sr, false);
    }

    fn grow(&mut self) {
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
        self.status = format!("grew {} seedlings from {kind} — click to hear, ✓ to adopt", self.seedlings.len());
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

    fn render_loop_audio(&mut self) -> Option<Vec<f32>> {
        let root = match genome::note_to_midi(&self.key) {
            Ok(r) => r,
            Err(_) => {
                self.status = format!("bad key {:?}", self.key);
                return None;
            }
        };
        let pat = loops::pattern(loops::PATTERN_NAMES[self.pattern_idx]).unwrap();
        let mut rng = SmallRng::seed_from_u64(0);
        Some(loops::render_loop(&self.genome, root, self.bpm, &pat, 2, self.swing, &mut rng))
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

// ---- widgets ------------------------------------------------------------------

fn theme(ctx: &egui::Context) {
    let mut v = egui::Visuals::dark();
    v.panel_fill = BG;
    v.window_fill = PANEL;
    v.override_text_color = Some(TEXT);
    v.widgets.inactive.bg_fill = PANEL;
    v.widgets.inactive.weak_bg_fill = PANEL;
    v.widgets.hovered.bg_fill = SEED;
    v.widgets.hovered.weak_bg_fill = SEED;
    v.widgets.active.bg_fill = SEED;
    v.selection.bg_fill = SEED;
    v.selection.stroke = Stroke::new(1.0, ACCENT);
    v.widgets.inactive.fg_stroke = Stroke::new(1.0, TEXT);
    v.widgets.hovered.fg_stroke = Stroke::new(1.2, ACCENT_HOT);
    v.widgets.active.fg_stroke = Stroke::new(1.2, ACCENT_HOT);
    v.widgets.noninteractive.fg_stroke = Stroke::new(1.0, TEXT);
    v.widgets.noninteractive.bg_stroke = Stroke::new(1.0, SEED);
    ctx.set_visuals(v);
}

fn panel_frame() -> egui::Frame {
    egui::Frame::none()
        .fill(PANEL)
        .stroke(Stroke::new(1.0, SEED))
        .rounding(8.0)
        .inner_margin(10.0)
}

fn header(ui: &mut egui::Ui, title: &str) {
    ui.label(egui::RichText::new(title.to_uppercase()).color(DIM).size(10.5).strong());
    ui.add_space(2.0);
}

/// The plant: 16 branches around a seed; drag a branch tip to grow/prune it.
fn plant(ui: &mut egui::Ui, g: &mut Genome, note: i32) -> bool {
    let side = ui.available_width().min(420.0);
    let (resp, p) = ui.allocate_painter(Vec2::splat(side), Sense::click_and_drag());
    let rect = resp.rect;
    let c = rect.center();
    let r0 = side * 0.11;
    let r_max = side * 0.385;
    let mut changed = false;

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
            if dist > r0 * 0.6 {
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
        p.circle_filled(tip, 4.5 + 3.0 * v, col);
        p.circle_filled(tip, 1.8 + 1.2 * v, CORE);
        let lp = c + dir * (r_max + 14.0 + 8.0 * dir.x.abs());
        p.text(
            lp,
            Align2::CENTER_CENTER,
            PARAMS[i].0,
            FontId::proportional(10.0),
            if v > 0.02 { TEXT } else { DIM },
        );
    }

    p.circle_filled(c, r0 * 0.78, SEED);
    p.circle_stroke(c, r0 * 0.78, Stroke::new(1.5, ACCENT));
    p.text(c, Align2::CENTER_CENTER, genome::midi_to_name(note), FontId::monospace(16.0), ACCENT_HOT);
    changed
}

fn lerp_color(a: Color32, b: Color32, t: f32) -> Color32 {
    let t = t.clamp(0.0, 1.0);
    let l = |x: u8, y: u8| (x as f32 + (y as f32 - x as f32) * t) as u8;
    Color32::from_rgb(l(a.r(), b.r()), l(a.g(), b.g()), l(a.b(), b.b()))
}

/// A genome slider in the plant's visual language.
fn param_slider(ui: &mut egui::Ui, v: &mut f32) -> bool {
    let (resp, p) = ui.allocate_painter(Vec2::new(132.0, 16.0), Sense::click_and_drag());
    let rect = resp.rect;
    let pad = 6.0;
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
    p.line_segment([Pos2::new(x0, y), Pos2::new(x1, y)], Stroke::new(2.0, SEED));
    p.line_segment([Pos2::new(x0, y), Pos2::new(hx, y)], Stroke::new(2.0, col));
    let r = if hot { 5.5 } else { 4.5 };
    p.circle_filled(Pos2::new(hx, y), r, col);
    p.circle_filled(Pos2::new(hx, y), r * 0.4, CORE);
    changed
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

fn waveform(ui: &mut egui::Ui, audio: &[f32], color: Color32) {
    let w = ui.available_width();
    let (resp, p) = ui.allocate_painter(Vec2::new(w, 54.0), Sense::hover());
    let rect = resp.rect;
    p.rect_filled(rect, 4.0, CORE);
    if audio.is_empty() {
        return;
    }
    let cols = rect.width() as usize;
    let per = (audio.len() / cols.max(1)).max(1);
    let mid = rect.center().y;
    let half = rect.height() * 0.46;
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

/// A seedling bud: circle whose glow is the reward score.
fn bud(ui: &mut egui::Ui, score: f32, selected: bool) -> egui::Response {
    let (resp, p) = ui.allocate_painter(Vec2::new(34.0, 34.0), Sense::click());
    let c = resp.rect.center();
    let col = lerp_color(DIM, ACCENT, score);
    let r = 9.0 + 5.0 * score;
    if selected || resp.hovered() {
        p.circle_stroke(c, r + 3.5, Stroke::new(1.5, ACCENT_HOT));
    }
    p.circle_filled(c, r, col);
    p.circle_filled(c, r * 0.35, CORE);
    resp
}

// ---- update -------------------------------------------------------------------

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.frame += 1;
        // screenshot mode: pre-grow a stab generation so the shot shows the garden
        if self.shot.is_some() && self.frame == 2 {
            self.grow_arch = 5; // "stab"
            self.grow();
        }

        // background match progress
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

        // drag & drop
        let dropped: Vec<_> = ctx.input(|i| i.raw.dropped_files.clone());
        if let Some(f) = dropped.first() {
            if let Some(path) = &f.path {
                self.start_match(path.to_string_lossy().into_owned());
            }
        }

        // undo / redo keys
        if ctx.input_mut(|i| i.consume_key(egui::Modifiers::COMMAND, egui::Key::Z)) {
            if let Some((g, n)) = self.undo.pop() {
                self.redo.push((self.genome, self.note));
                self.genome = g;
                self.note = n;
                self.render_current();
            }
        }
        if ctx.input_mut(|i| i.consume_key(egui::Modifiers::COMMAND, egui::Key::Y)) {
            if let Some((g, n)) = self.redo.pop() {
                self.undo.push((self.genome, self.note));
                self.genome = g;
                self.note = n;
                self.render_current();
            }
        }

        egui::CentralPanel::default().show(ctx, |ui| {
            self.top_bar(ui);
            ui.add_space(2.0);
            ui.label(egui::RichText::new(&self.status).size(12.5).color(TEXT));
            if self.rx.is_some() {
                ui.add(egui::ProgressBar::new(self.progress).desired_width(300.0));
            }
            ui.add_space(6.0);

            ui.horizontal_top(|ui| {
                // left: the plant
                panel_frame().show(ui, |ui| {
                    ui.set_width(440.0);
                    header(ui, "plant modulation system");
                    if plant(ui, &mut self.genome, self.note) {
                        self.preset_idx = None;
                        self.render_current();
                    }
                    ui.horizontal(|ui| {
                        if ui.button("▶ play").clicked() {
                            self.play_patch();
                        }
                        if ui.button("−").clicked() {
                            self.note -= 1;
                            self.render_current();
                        }
                        ui.label(egui::RichText::new(genome::midi_to_name(self.note)).monospace());
                        if ui.button("+").clicked() {
                            self.note += 1;
                            self.render_current();
                        }
                        ui.add_space(8.0);
                        let cue = garden::score(&self.net, &self.genome);
                        ui.label(
                            egui::RichText::new(format!("cue {:.0}%", cue * 100.0))
                                .monospace()
                                .size(11.0)
                                .color(lerp_color(DIM, ACCENT_HOT, cue)),
                        )
                        .on_hover_text("predicted quality from the RLHF reward model\ntrained on juxxs's star ratings");
                        ui.add_space(8.0);
                        if ui.button("💾 save patch").clicked() {
                            let _ = wavio::write_wav("patch.wav", &self.last_audio, self.last_sr);
                            let _ = genome::save("patch.wav.genome.txt", &self.genome);
                            self.status = "saved patch.wav + patch.wav.genome.txt".into();
                        }
                    });
                });

                ui.add_space(8.0);

                // right: genome + loop lab
                ui.vertical(|ui| {
                    ui.set_width(310.0);
                    panel_frame().show(ui, |ui| {
                        header(ui, "genome");
                        let mut changed = false;
                        let reals = genome::denorm(&self.genome);
                        egui::Grid::new("params").num_columns(3).spacing([8.0, 3.0]).show(ui, |ui| {
                            for i in 0..N_PARAMS {
                                ui.label(egui::RichText::new(PARAMS[i].0).monospace().size(11.0));
                                changed |= param_slider(ui, &mut self.genome[i]);
                                ui.label(
                                    egui::RichText::new(fmt_real(reals[i]))
                                        .monospace()
                                        .size(10.0)
                                        .color(if self.genome[i] > 0.02 { TEXT } else { DIM }),
                                );
                                ui.end_row();
                            }
                        });
                        if changed {
                            self.preset_idx = None;
                            self.render_current();
                        }
                    });
                    ui.add_space(6.0);
                    panel_frame().show(ui, |ui| {
                        header(ui, "loop lab");
                        ui.horizontal(|ui| {
                            egui::ComboBox::from_id_salt("pat")
                                .selected_text(loops::PATTERN_NAMES[self.pattern_idx])
                                .show_ui(ui, |ui| {
                                    for (i, name) in loops::PATTERN_NAMES.iter().enumerate() {
                                        ui.selectable_value(&mut self.pattern_idx, i, *name);
                                    }
                                });
                            ui.add(egui::DragValue::new(&mut self.bpm).range(80.0..=180.0).suffix(" bpm"));
                            ui.add(egui::TextEdit::singleline(&mut self.key).desired_width(30.0));
                        });
                        ui.horizontal(|ui| {
                            ui.label(egui::RichText::new("swing").color(DIM).small());
                            param_slider(ui, &mut self.swing);
                            ui.label(
                                egui::RichText::new(format!("{:.0}%", self.swing * 100.0))
                                    .monospace()
                                    .size(10.0),
                            );
                        });
                        ui.horizontal(|ui| {
                            let playing = self.audio.loop_playing();
                            let label = if playing { "⏹ stop" } else { "▶ loop" };
                            if ui.button(label).clicked() {
                                if playing {
                                    self.audio.stop();
                                } else if let Some(a) = self.render_loop_audio() {
                                    self.audio.play(&a, loops::SR_OUT, true);
                                    self.last_audio = a;
                                    self.last_sr = loops::SR_OUT;
                                }
                            }
                            if ui.button("💾 save loop").clicked() {
                                if let Some(a) = self.render_loop_audio() {
                                    let name = format!(
                                        "loop_{}_{}bpm_{}.wav",
                                        loops::PATTERN_NAMES[self.pattern_idx], self.bpm as u32, self.key
                                    );
                                    let _ = wavio::write_wav(&name, &a, loops::SR_OUT);
                                    self.status = format!("saved {name}");
                                }
                            }
                            if self.audio.loop_playing() {
                                ui.label(egui::RichText::new("looping").color(ACCENT).small());
                                ctx.request_repaint_after(std::time::Duration::from_millis(250));
                            }
                        });
                    });
                });
            });

            ui.add_space(6.0);

            // garden
            panel_frame().show(ui, |ui| {
                header(ui, "garden — grow it to taste");
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new("seed").color(DIM).small());
                    egui::ComboBox::from_id_salt("arch")
                        .selected_text(if self.grow_arch == 0 {
                            "this patch"
                        } else {
                            garden::ARCHETYPE_NAMES[self.grow_arch - 1]
                        })
                        .show_ui(ui, |ui| {
                            ui.selectable_value(&mut self.grow_arch, 0, "this patch");
                            for (i, a) in garden::ARCHETYPE_NAMES.iter().enumerate() {
                                ui.selectable_value(&mut self.grow_arch, i + 1, *a);
                            }
                        });
                    ui.label(egui::RichText::new("wildness").color(DIM).small());
                    param_slider(ui, &mut self.grow_amount);
                    if ui.button("🌱 grow").clicked() {
                        self.grow();
                    }
                    if !self.seedlings.is_empty() {
                        ui.label(
                            egui::RichText::new("buds glow by cue score — click to hear, ✓ adopts")
                                .color(DIM)
                                .small(),
                        );
                    }
                });
                if !self.seedlings.is_empty() {
                    ui.add_space(4.0);
                    ui.horizontal(|ui| {
                        let mut adopt: Option<usize> = None;
                        for i in 0..self.seedlings.len() {
                            ui.vertical(|ui| {
                                let s = &self.seedlings[i];
                                if bud(ui, s.score, false).clicked() {
                                    let mut rng = SmallRng::seed_from_u64(0);
                                    let a = synth::render_default(&s.genome, self.note as f32, &mut rng);
                                    self.audio.play(&a, synth::SR, false);
                                }
                                ui.label(
                                    egui::RichText::new(format!("{:.0}%", self.seedlings[i].score * 100.0))
                                        .monospace()
                                        .size(10.0)
                                        .color(DIM),
                                );
                                if ui.small_button("✓").clicked() {
                                    adopt = Some(i);
                                }
                            });
                        }
                        if let Some(i) = adopt {
                            let g = self.seedlings[i].genome;
                            self.preset_idx = None;
                            let note = self.note;
                            self.set_patch(g, note, true);
                            self.grow_arch = 0; // next generation grows from the adopted patch
                            self.grow();
                            self.status = "adopted — next generation grown from it".into();
                        }
                    });
                }
            });

            ui.add_space(6.0);

            // waveforms
            panel_frame().show(ui, |ui| {
                header(ui, "patch waveform");
                let audio = self.last_audio.clone();
                waveform(ui, &audio, ACCENT);
                if let Some(t) = &self.target {
                    let t = t.clone();
                    ui.add_space(4.0);
                    ui.horizontal(|ui| {
                        ui.label(egui::RichText::new(format!("target · {}", self.target_name)).color(DIM).small());
                        if ui.small_button("▶ target").clicked() {
                            self.audio.play(&t, synth::SR, false);
                        }
                        if ui.small_button("↻ re-clone").clicked() {
                            self.status = "re-cloning...".into();
                            self.progress = 0.0;
                            let (tx, rx) = mpsc::channel();
                            self.rx = Some(rx);
                            let target = t.clone();
                            std::thread::spawn(move || match_thread(target, tx));
                        }
                    });
                    waveform(ui, &t, DIM);
                }
            });
        });

        // screenshot mode: capture once the UI has settled, then quit
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
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new("synfection").size(24.0).color(ACCENT_HOT).strong());
            ui.add_space(10.0);

            // preset browser
            let cur = self.preset_idx.map(|i| PRESETS[i].name).unwrap_or("custom");
            if ui.button("◀").clicked() {
                let i = self.preset_idx.map(|i| (i + PRESETS.len() - 1) % PRESETS.len()).unwrap_or(0);
                self.load_preset(i);
            }
            let mut pick: Option<usize> = None;
            egui::ComboBox::from_id_salt("preset")
                .selected_text(egui::RichText::new(cur).monospace())
                .width(130.0)
                .show_ui(ui, |ui| {
                    for (i, p) in PRESETS.iter().enumerate() {
                        if ui.selectable_label(self.preset_idx == Some(i), p.name).clicked() {
                            pick = Some(i);
                        }
                    }
                });
            if let Some(i) = pick {
                self.load_preset(i);
            }
            if ui.button("▶").clicked() {
                let i = self.preset_idx.map(|i| (i + 1) % PRESETS.len()).unwrap_or(0);
                self.load_preset(i);
            }
            ui.add_space(6.0);

            // A/B, undo, redo
            let ab = if self.ab_is_b { "B/A" } else { "A/B" };
            if ui.button(ab).on_hover_text("swap with the other slot").clicked() {
                std::mem::swap(&mut self.genome, &mut self.ab_other.0);
                std::mem::swap(&mut self.note, &mut self.ab_other.1);
                self.ab_is_b = !self.ab_is_b;
                self.render_current();
                self.audio.play(&self.last_audio, self.last_sr, false);
            }
            if ui.button("↶").on_hover_text("undo (ctrl+z)").clicked() {
                if let Some((g, n)) = self.undo.pop() {
                    self.redo.push((self.genome, self.note));
                    self.genome = g;
                    self.note = n;
                    self.render_current();
                }
            }
            if ui.button("↷").on_hover_text("redo (ctrl+y)").clicked() {
                if let Some((g, n)) = self.redo.pop() {
                    self.undo.push((self.genome, self.note));
                    self.genome = g;
                    self.note = n;
                    self.render_current();
                }
            }
            ui.add_space(6.0);

            if ui.button("🎲 random").on_hover_text("best of 12 random patches, reward-ranked").clicked() {
                let mut rng = SmallRng::seed_from_u64(self.frame);
                if let Some(g) = garden::lucky_dip(&self.net, self.note, 12, &mut rng) {
                    self.preset_idx = None;
                    let note = self.note;
                    self.set_patch(g, note, true);
                }
            }
            if ui.button("🎲 mutate").clicked() {
                let mut rng = SmallRng::seed_from_u64(self.frame);
                let mut g = self.genome;
                for v in g.iter_mut() {
                    *v = (*v + matcher::gaussian(&mut rng) * 0.12).clamp(0.0, 1.0);
                }
                self.preset_idx = None;
                let note = self.note;
                self.set_patch(g, note, true);
            }
            if ui.button("⬆ open wav").clicked() {
                if let Some(path) = rfd::FileDialog::new()
                    .add_filter("wav", &["wav"])
                    .pick_file()
                {
                    self.start_match(path.to_string_lossy().into_owned());
                }
            }

            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                let mut vol = self.audio.volume;
                if param_slider(ui, &mut vol) {
                    self.audio.set_volume(vol);
                }
                ui.label(egui::RichText::new("master").color(DIM).small());
            });
        });
    }

    fn load_preset(&mut self, i: usize) {
        self.preset_idx = Some(i);
        self.set_patch(PRESETS[i].genome, PRESETS[i].note, true);
        self.status = format!("preset: {}", PRESETS[i].name);
    }
}

pub fn run(genome_path: Option<String>, shot: Option<String>) -> Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([880.0, 810.0])
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
