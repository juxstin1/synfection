//! A little Synplant-flavoured UI: drop a wav on the window, the net grows a
//! patch, and the genome is a plant — 16 branches around a seed you can drag.

use std::sync::mpsc;

use anyhow::Result;
use eframe::egui::{self, Align2, Color32, FontId, Pos2, Sense, Stroke, Vec2};
use rand::rngs::SmallRng;
use rand::SeedableRng;

use crate::genome::{self, Genome, N_PARAMS, PARAMS};
use crate::loops;
use crate::matcher;
use crate::net::Net;
use crate::synth;
use crate::wavio;

const BG: Color32 = Color32::from_rgb(10, 14, 11);
const PANEL: Color32 = Color32::from_rgb(16, 22, 17);
const DIM: Color32 = Color32::from_rgb(70, 95, 78);
const TEXT: Color32 = Color32::from_rgb(196, 220, 201);
const ACCENT: Color32 = Color32::from_rgb(92, 224, 138);
const ACCENT_HOT: Color32 = Color32::from_rgb(180, 255, 160);
const SEED: Color32 = Color32::from_rgb(34, 54, 38);

enum MatchMsg {
    Progress(usize, f32),
    Done { genome: Genome, midi: i32, l0: f32, l1: f32 },
    Failed(String),
}

pub struct App {
    genome: Genome,
    note: i32,
    status: String,
    target: Option<Vec<f32>>,
    target_name: String,
    last_audio: Vec<f32>,
    last_sr: f32,
    rx: Option<mpsc::Receiver<MatchMsg>>,
    progress: f32,
    // audio out (stream must stay alive while playing)
    audio: Option<(rodio::OutputStream, rodio::OutputStreamHandle)>,
    // loop section
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
        let genome = genome_path
            .and_then(|p| genome::load(&p).ok())
            .unwrap_or_else(|| {
                // a pleasant default: warm saw bass with a bit of drive
                let mut g = [0.5f32; N_PARAMS];
                g[0] = 0.42; g[3] = 0.25; g[4] = 0.8; g[5] = 0.05; g[6] = 0.25;
                g[7] = 0.45; g[9] = 0.35; g[12] = 0.02; g[14] = 0.7;
                g
            });
        let mut app = App {
            genome,
            note: 36,
            status: "drop a .wav on the window to clone it — or garden the plant by hand".into(),
            target: None,
            target_name: String::new(),
            last_audio: Vec::new(),
            last_sr: synth::SR,
            rx: None,
            progress: 0.0,
            audio: rodio::OutputStream::try_default().ok(),
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

    fn play(&self, audio: &[f32], sr: f32) {
        if let Some((_, handle)) = &self.audio {
            if let Ok(sink) = rodio::Sink::try_new(handle) {
                sink.append(rodio::buffer::SamplesBuffer::new(1, sr as u32, audio.to_vec()));
                sink.detach();
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

/// The plant: 16 branches around a seed; drag a branch tip to grow/prune it.
fn plant(ui: &mut egui::Ui, g: &mut Genome, note: i32) -> bool {
    let side = ui.available_width().min(440.0);
    let (resp, p) = ui.allocate_painter(Vec2::splat(side), Sense::click_and_drag());
    let rect = resp.rect;
    let c = rect.center();
    let r0 = side * 0.11;
    let r_max = side * 0.385;
    let mut changed = false;

    // soil rings
    for i in 1..=3 {
        p.circle_stroke(c, r0 + (r_max - r0) * i as f32 / 3.0, Stroke::new(1.0, SEED));
    }

    let angle_of = |i: usize| -> f32 {
        std::f32::consts::TAU * i as f32 / N_PARAMS as f32 - std::f32::consts::FRAC_PI_2
    };

    // drag: nearest branch by angle
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

    // branches
    for i in 0..N_PARAMS {
        let a = angle_of(i);
        let dir = Vec2::new(a.cos(), a.sin());
        let v = g[i];
        let tip = c + dir * (r0 + v * (r_max - r0));
        let col = lerp_color(DIM, ACCENT, v);
        // curved-ish stem: two segments with a slight bend
        let bend = Vec2::new(-dir.y, dir.x) * 6.0 * (i as f32 * 2.399).sin();
        let mid = c + dir * (r0 + v * (r_max - r0) * 0.55) + bend;
        p.line_segment([c + dir * r0, mid], Stroke::new(2.0, col));
        p.line_segment([mid, tip], Stroke::new(2.0, col));
        p.circle_filled(tip, 4.5 + 3.0 * v, col);
        p.circle_filled(tip, 1.8 + 1.2 * v, Color32::from_rgb(8, 12, 9));
        // label just outside the ring
        let lp = c + dir * (r_max + 14.0 + 8.0 * dir.x.abs());
        p.text(
            lp,
            Align2::CENTER_CENTER,
            PARAMS[i].0,
            FontId::proportional(10.0),
            if v > 0.02 { TEXT } else { DIM },
        );
    }

    // seed
    p.circle_filled(c, r0 * 0.78, SEED);
    p.circle_stroke(c, r0 * 0.78, Stroke::new(1.5, ACCENT));
    p.text(
        c,
        Align2::CENTER_CENTER,
        genome::midi_to_name(note),
        FontId::monospace(16.0),
        ACCENT_HOT,
    );
    changed
}

fn lerp_color(a: Color32, b: Color32, t: f32) -> Color32 {
    let t = t.clamp(0.0, 1.0);
    let l = |x: u8, y: u8| (x as f32 + (y as f32 - x as f32) * t) as u8;
    Color32::from_rgb(l(a.r(), b.r()), l(a.g(), b.g()), l(a.b(), b.b()))
}

/// A genome slider in the plant's visual language: the track fills dim -> green
/// with the value, the handle is a branch tip that glows on hover.
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
    // rail + fill
    p.line_segment([Pos2::new(x0, y), Pos2::new(x1, y)], Stroke::new(2.0, SEED));
    p.line_segment([Pos2::new(x0, y), Pos2::new(hx, y)], Stroke::new(2.0, col));
    // branch-tip handle
    let r = if hot { 5.5 } else { 4.5 };
    p.circle_filled(Pos2::new(hx, y), r, col);
    p.circle_filled(Pos2::new(hx, y), r * 0.4, Color32::from_rgb(8, 12, 9));
    changed
}

/// Compact real-value readout for a param row (Hz, cents, seconds...).
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
    let (resp, p) = ui.allocate_painter(Vec2::new(w, 56.0), Sense::hover());
    let rect = resp.rect;
    p.rect_filled(rect, 4.0, PANEL);
    if audio.is_empty() {
        return;
    }
    let cols = rect.width() as usize;
    let per = (audio.len() / cols.max(1)).max(1);
    let mid = rect.center().y;
    let half = rect.height() * 0.48;
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

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.frame += 1;

        // background match progress
        let mut msgs = Vec::new();
        if let Some(rx) = &self.rx {
            while let Ok(m) = rx.try_recv() {
                msgs.push(m);
            }
            ctx.request_repaint();
        }
        let mut done_msg = None;
        for m in msgs {
            match m {
                MatchMsg::Progress(pct, best) => {
                    self.progress = pct as f32 / 100.0;
                    self.status = format!("growing... {pct}%  spec-loss {best:.3}");
                }
                MatchMsg::Done { genome, midi, l0, l1 } => {
                    done_msg = Some((genome, midi, l0, l1));
                }
                MatchMsg::Failed(e) => {
                    self.status = format!("match failed: {e}");
                    self.rx = None;
                }
            }
        }
        if let Some((g, midi, l0, l1)) = done_msg {
            self.genome = g;
            self.note = midi;
            self.status = format!(
                "cloned {} at {}  ·  spec-loss {l0:.2} → {l1:.2}",
                self.target_name,
                genome::midi_to_name(midi)
            );
            self.rx = None;
            self.render_current();
            self.play(&self.last_audio.clone(), self.last_sr);
        }

        // drag & drop
        let dropped: Vec<_> = ctx.input(|i| i.raw.dropped_files.clone());
        if let Some(f) = dropped.first() {
            if let Some(path) = &f.path {
                self.start_match(path.to_string_lossy().into_owned());
            }
        }

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new("synfection").size(26.0).color(ACCENT_HOT).strong());
                ui.label(egui::RichText::new("— plant a sound, grow a patch").size(13.0).color(DIM));
            });
            ui.add_space(2.0);
            ui.label(egui::RichText::new(&self.status).size(13.0).color(TEXT));
            if self.rx.is_some() {
                ui.add(egui::ProgressBar::new(self.progress).desired_width(300.0));
            }
            ui.add_space(6.0);

            ui.horizontal_top(|ui| {
                // left: the plant
                ui.vertical(|ui| {
                    if plant(ui, &mut self.genome, self.note) {
                        self.render_current();
                    }
                    ui.horizontal(|ui| {
                        if ui.button("▶ play").clicked() {
                            self.render_current();
                            self.play(&self.last_audio.clone(), self.last_sr);
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
                        if ui.button("🎲 mutate").clicked() {
                            let mut rng = SmallRng::seed_from_u64(self.frame);
                            for v in self.genome.iter_mut() {
                                *v = (*v + matcher::gaussian(&mut rng) * 0.12).clamp(0.0, 1.0);
                            }
                            self.render_current();
                            self.play(&self.last_audio.clone(), self.last_sr);
                        }
                        if ui.button("💾 save patch").clicked() {
                            let _ = wavio::write_wav("patch.wav", &self.last_audio, self.last_sr);
                            let _ = genome::save("patch.wav.genome.txt", &self.genome);
                            self.status = "saved patch.wav + patch.wav.genome.txt".into();
                        }
                    });
                });

                ui.add_space(12.0);

                // right: genome readout + loop lab
                ui.vertical(|ui| {
                    ui.set_width(300.0);
                    ui.label(egui::RichText::new("genome").color(DIM).small());
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
                        self.render_current();
                    }

                    ui.add_space(10.0);
                    ui.label(egui::RichText::new("loop lab").color(DIM).small());
                    ui.horizontal(|ui| {
                        egui::ComboBox::from_id_salt("pat")
                            .selected_text(loops::PATTERN_NAMES[self.pattern_idx])
                            .show_ui(ui, |ui| {
                                for (i, name) in loops::PATTERN_NAMES.iter().enumerate() {
                                    ui.selectable_value(&mut self.pattern_idx, i, *name);
                                }
                            });
                        ui.add(egui::DragValue::new(&mut self.bpm).range(80.0..=180.0).suffix(" bpm"));
                        ui.add(egui::TextEdit::singleline(&mut self.key).desired_width(34.0));
                    });
                    ui.horizontal(|ui| {
                        ui.label(egui::RichText::new("swing").color(DIM).small());
                        param_slider(ui, &mut self.swing);
                        ui.label(
                            egui::RichText::new(format!("{:.0}%", self.swing * 100.0))
                                .monospace()
                                .size(10.0)
                                .color(TEXT),
                        );
                    });
                    ui.horizontal(|ui| {
                        if ui.button("▶ loop").clicked() {
                            if let Ok(root) = genome::note_to_midi(&self.key) {
                                let pat = loops::pattern(loops::PATTERN_NAMES[self.pattern_idx]).unwrap();
                                let mut rng = SmallRng::seed_from_u64(0);
                                let audio = loops::render_loop(&self.genome, root, self.bpm, &pat, 2, self.swing, &mut rng);
                                self.play(&audio, loops::SR_OUT);
                                self.last_audio = audio;
                                self.last_sr = loops::SR_OUT;
                            } else {
                                self.status = format!("bad key {:?}", self.key);
                            }
                        }
                        if ui.button("💾 save loop").clicked() {
                            if let Ok(root) = genome::note_to_midi(&self.key) {
                                let pat = loops::pattern(loops::PATTERN_NAMES[self.pattern_idx]).unwrap();
                                let mut rng = SmallRng::seed_from_u64(0);
                                let audio = loops::render_loop(&self.genome, root, self.bpm, &pat, 2, self.swing, &mut rng);
                                let name = format!("loop_{}bpm_{}.wav", self.bpm as u32, self.key);
                                let _ = wavio::write_wav(&name, &audio, loops::SR_OUT);
                                self.status = format!("saved {name}");
                            }
                        }
                    });

                    if let Some(t) = &self.target {
                        ui.add_space(10.0);
                        ui.label(egui::RichText::new(format!("target · {}", self.target_name)).color(DIM).small());
                        let t = t.clone();
                        waveform(ui, &t, DIM);
                        ui.horizontal(|ui| {
                            if ui.button("▶ target").clicked() {
                                self.play(&t, synth::SR);
                            }
                            if ui.button("↻ re-clone").clicked() {
                                self.status = "re-cloning...".into();
                                self.progress = 0.0;
                                let (tx, rx) = mpsc::channel();
                                self.rx = Some(rx);
                                let target = t.clone();
                                std::thread::spawn(move || match_thread(target, tx));
                            }
                        });
                    }
                });
            });

            ui.add_space(6.0);
            ui.label(egui::RichText::new("patch").color(DIM).small());
            let audio = self.last_audio.clone();
            waveform(ui, &audio, ACCENT);
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
                let _ = image::save_buffer(
                    &path,
                    image.as_raw(),
                    w as u32,
                    h as u32,
                    image::ColorType::Rgba8,
                );
                println!("screenshot -> {path}");
                std::process::exit(0);
            }
            ctx.request_repaint();
        }
    }
}

pub fn run(genome_path: Option<String>, shot: Option<String>) -> Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([820.0, 640.0])
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
