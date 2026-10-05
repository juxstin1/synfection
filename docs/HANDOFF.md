# Handoff: Synfection → Serum 2 export, and the road after it

Status: spec only. Nothing here is implemented yet. Written from a full read of
`src/` and `lab/` at v0.6.0 so the numbers below are the engine's, not the README's.

The one-line diagnosis: the engine was built to keep PyTorch happy, not to be
played or to sound like hardware. Everything that feels "demo" about the app
flows from that. The neural matcher is the valuable part and it is
engine-agnostic. So the plan is: get the *results* into Serum 2 first (cheap,
useful tomorrow), then replace the engine with a real-time voice, then retrain.

---

## 1. What the engine actually is (read this before mapping anything)

| Fact | Where | Why it matters for export |
|---|---|---|
| Offline renderer: one 1.2 s clip per call, no voice, no callback | `synth.rs::render` | Serum is real-time; the mapping is a *recipe*, not a port |
| Additive: up to 72 harmonics per osc, per sample | `MAX_HARM = 72` | A wavetable frame built from those 72 partials is an exact reproduction of the osc. Also: at F1 the 72nd harmonic is 3.1 kHz, so bass is band-limited there. That's a lot of the "polite" |
| Filter is a 2-pole lowpass *magnitude curve* applied per harmonic, capped at 12× (21.6 dB) | `filter_mag` | No phase, no feedback, no drive inside. Serum's `Low 12` is the honest equivalent; `MG Low 12` is the one you'll actually want |
| Filter env is linear in Hz between `cutoff` and `top` | `cutoff + (top-cutoff)*e` | Serum mods cutoff in the knob (log) domain. Shape differs; peak matches. Calibrate by ear |
| `top = cutoff + filt_env * (nyq - cutoff)`, `nyq = 0.49 * sr` | `render` | **Quirk:** patch preview renders at 22.05k (nyq 10 804 Hz) but loops render at 44.1k (nyq 21 609 Hz), so the same genome sweeps twice as high in the loop lab. Use the 22.05k definition; it is what the net and the ratings were tuned on |
| Filter env sustain is fixed at 0.25, release = `amp_r` | `adsr(filt_a, filt_d, 0.25, amp_r)` | Env 2 in Serum: S = 25 % |
| Sub is a sine one octave down at `0.6 * sub_level`, routed *through* the filter | `render` | Serum sub: octave −1, sine, "direct out" **off** |
| Noise is white, spectrally shaped once with the filter curve at `top`, then × `noise_level`; it does *not* follow the sweep | `shaped_noise(n, top, reso)` | Serum noise through the filter will follow the sweep instead. Accept the difference |
| Signal order: (osc + sub + noise) → filter → amp env → drive → peak-normalise 0.9 | `render` | Maps cleanly: Serum voice (env 1 → filter) → FX distortion. Drive in the FX rack is post-voice-sum, which only matters for chords |
| Drive: `(1-d)*x + d*tanh(6x)/tanh(6)` on the peak-normalised signal | `render` | A soft-clip with a dry/wet mix, exactly what a Distortion FX with Mix = d is |
| Pitch env: `f(t) = f0 * 2^(pitch_env * exp(-t/pitch_dec) / 12)` on osc A, B and sub | `render` | Exponential decay with time constant τ; not a linear ramp |
| Cutoff LFO: `cutoff * 2^(lfo_depth * sin(2π lfo_rate t))`, phase 0 at note-on | `render` | Sine, bipolar, ±depth octaves, retriggered per note, free-running Hz |
| Wavetable frames are L1-normalised (sum of |a_k| = 1) | `wavetable()` | Frame peaks are ≤ 1 by construction. Do **not** let Serum normalise on import or osc_mix balance breaks |
| Anti-alias: sigmoid roll-off of harmonics near nyq | `render` | Serum handles its own. Ignore |
| Amp/filter ADSRs are linear segments | `adsr` | Serum env curve = 0 for parity; leave default for taste |
| Unison "thickener" is a post-render resample trick, not part of the genome | `dsp::thicken` | Optional section 4.10 |

---

## 2. Patch file format (input to the exporter)

```
# synfection patch  note=45
0.50000
0.55000
...            (20 lines, normalised 0..1, PARAMS order)
```

Legacy files with 15 or 16 values are upgraded by `genome::upgrade`
(v1 inserts `drive = 0` at index 6; v2 pads the four v3 params with
`[0.0, 0.5, 0.4, 0.0]`). The exporter should call the same function.

Denormalisation, per `genome::PARAMS` `(name, lo, hi, log)`:

- linear: `real = lo + (hi - lo) * x`
- log:    `real = lo * (hi / lo) ^ x`

| # | name | lo | hi | map | unit |
|---|---|---|---|---|---|
| 0 | osc1_wt | 0 | 1 | lin | table position |
| 1 | osc2_wt | 0 | 1 | lin | table position |
| 2 | osc2_detune | −50 | 50 | lin | cents |
| 3 | osc_mix | 0 | 1 | lin | B weight (A = 1 − mix) |
| 4 | sub_level | 0 | 1 | lin | × 0.6 in the engine |
| 5 | noise_level | 0 | 0.6 | lin | linear gain |
| 6 | drive | 0 | 1 | lin | dry/wet |
| 7 | cutoff | 60 | 10000 | log | Hz |
| 8 | reso | 0.6 | 9.0 | lin | Q |
| 9 | filt_env | 0 | 1 | lin | fraction of (nyq − cutoff) |
| 10 | filt_a | 0.001 | 0.4 | log | s |
| 11 | filt_d | 0.02 | 0.7 | log | s |
| 12 | amp_a | 0.001 | 0.4 | log | s |
| 13 | amp_d | 0.02 | 0.7 | log | s |
| 14 | amp_s | 0 | 1 | lin | level |
| 15 | amp_r | 0.02 | 0.7 | log | s |
| 16 | pitch_env | 0 | 48 | lin | semitones above note at t = 0 |
| 17 | pitch_dec | 0.005 | 0.4 | log | s (time constant τ) |
| 18 | lfo_rate | 0.05 | 12 | log | Hz |
| 19 | lfo_depth | 0 | 2 | lin | ± octaves of cutoff |

---

## 3. The wavetable (ship one file, reuse it for every patch)

The engine's 8 frames, in order: sine, triangle, square, saw, 25 % pulse,
10 % pulse, formant, "rich" (1/√k). Harmonic amplitudes for k = 1..72:

```
sine     a_k = 1 if k == 1 else 0
tri      a_k = 1/k²          (odd k only)
square   a_k = 1/k           (odd k only)
saw      a_k = 1/k
pulse25  a_k = |sin(π k 0.25)| · 2/(π k)
pulse10  a_k = |sin(π k 0.10)| · 2/(π k)
formant  a_k = exp(−(k−5)²/8) + 0.3 · exp(−(k−12)²/20)
rich     a_k = 1/√k
```

Each frame is then L1-normalised: `a_k /= Σ|a_k|`. All partials are sine phase
(the engine does `sin(k · phase)`), so a frame is `Σ a_k · sin(2π k n / 2048)`.

The engine morphs by **linearly interpolating harmonic amplitudes** between
adjacent frames, `pos ∈ [0,1]` mapped onto the 7 gaps. Serum morphs by
crossfading frames, which is the same thing when the frames are pre-interpolated.
So export **256 frames**, frame *i* at `pos = i/255`, each built from the
interpolated amplitudes. Then `WT Pos = osc_wt × 100 %` is exact.

File: mono, 32-bit float, 2048 samples per frame, 256 frames = 524 288 samples,
with the standard Serum `clm ` chunk (`<!>2048 20000000 wavetable`). Do not
normalise on import.

Two variants of the same table:

- `synfection_wt.wav` (default): harmonics 1..72 only. Reproduces the engine's
  band-limit per note exactly, including the 3 kHz ceiling at F1. Use this to
  A/B against the engine.
- `synfection_wt_bright.wav` (`--bright`): extend every formula to k = 1..1024
  before normalising. This is the sound the engine *couldn't* make. Cutoff
  mapping stays the same; expect to pull cutoff down by ear on bright patches.

Reference generator (numpy, ~20 lines) belongs in `lab/export_serum_wt.py`;
the Rust exporter can embed the same math and write via `hound`.

---

## 4. Genome → Serum 2 recipe

Values are given in real units. Serum 2 parameter names, ranges and mod-depth
units are **your side** (the automation). Where the engine and Serum disagree
in shape, the entry says so and names the calibration constant.

### 4.1 Osc A
- Wavetable: `synfection_wt.wav`
- WT Pos: `osc1_wt` (0..1 → 0..100 %)
- Level: `1 − osc_mix`
- Coarse / fine: 0
- Unison: 1 (see 4.10 for the optional thickener)
- Phase: any; the engine's is deterministic sine-phase but nobody can hear that

### 4.2 Osc B
- Same wavetable
- WT Pos: `osc2_wt`
- Level: `osc_mix`
- Fine: `osc2_detune` cents (±50 fits inside Serum's ±100)
- Unison: 1

Level note: the engine's weights multiply L1-normalised profiles, so a sine
frame's fundamental is ~5× a saw frame's at equal weight (1.0 vs 0.206). That
ratio is baked into the wavetable file, so Serum's linear osc levels reproduce
the balance for free. Global loudness is normalised at the end anyway.

### 4.3 Sub
- Shape: sine, Octave: −1
- Level: `0.6 · sub_level` relative to the osc levels above
- Direct out: **off** (must pass through the filter)
- Disable if `sub_level < 0.01`

### 4.4 Noise
- Type: white
- Level: `noise_level` (already 0..0.6 in real units)
- Route to filter: on. Difference accepted: the engine shapes noise statically
  at `top`; Serum's will ride the sweep. It reads as slightly more animated.
- Disable if `noise_level < 1e-3`

### 4.5 Filter
- Type: `Low 12` for parity. `MG Low 12` for the version you'll keep.
- Cutoff: `cutoff` Hz
- Resonance: from Q. The engine's peak gain at cutoff is ≈ `20·log10(Q)` dB,
  hard-capped at 21.6 dB:

  | Q | peak |
  |---|---|
  | 0.6 | −4.4 dB (no bump) |
  | 1 | 0 dB |
  | 2 | +6 dB |
  | 4 | +12 dB |
  | 9 | +19 dB |

  **Calibration A:** find the Serum resonance % that gives those peaks on
  `Low 12` (sweep a sine through cutoff, read the peak). Fit a 3-point curve
  and bake it in. Expect the map to be roughly linear in dB.
- Drive (Serum's filter drive): 0. Drive lives in FX, see 4.8.

### 4.6 Env 1 (amp)
- A `amp_a` s, D `amp_d` s, S `amp_s`, R `amp_r` s
- Curves: 0 / linear for parity

### 4.7 Env 2 (filter) → Cutoff
- A `filt_a` s, D `filt_d` s, **S 25 %**, R `amp_r` s (yes, the amp release)
- Target: filter cutoff, unipolar +
- Depth: the env peak must land cutoff at `top`:

  ```
  top          = cutoff + filt_env · (10804.5 − cutoff)     # 22.05k nyquist
  sweep_oct    = log2(top / cutoff)
  ```
  Express depth as `+sweep_oct` octaves (semitones × 12 if Serum wants semis).
  **Calibration B:** Serum's cutoff mod depth is in knob units, not octaves;
  measure knob-units-per-octave once at three cutoffs and fit.
- Shape difference, accepted: the engine sweeps linearly in Hz (spends more
  time near `top`), Serum sweeps log (spends more time near `cutoff`). Serum
  will sound snappier. If a patch loses its bloom, add a little env curve
  toward "slow start".
- Skip the routing entirely when `filt_env < 0.01`.

### 4.8 Drive → FX Distortion
- Mode: soft clip / tube (whichever is closest to `tanh`)
- Mix: `drive × 100 %` (the dry/wet blend *is* the engine's formula)
- Drive amount: **Calibration C.** The engine peak-normalises to 1.0, then
  applies `tanh(6x)`. Find the Serum drive that gives the same harmonic
  signature on a 0 dBFS sine. Measured on `tanh(6·sin)`, relative to the
  fundamental: 3rd −10.3 dB, 5th −16.3 dB, 7th −21.3 dB, 9th −25.9 dB.
  Set a pre-gain so the voice hits the distortion near 0 dBFS.
- Placement: after the voice sum. For mono bass this equals the engine. For
  chords the voices intermodulate; that's a feature.
- Bypass when `drive < 0.01`.

### 4.9 Pitch env and cutoff LFO (v3 params)
Pitch env → Osc A pitch, Osc B pitch, Sub pitch (three routings, same source):
- Only when `pitch_env > 0.05` semitones
- Env 3: A 0, D see below, S 0, R 0
- Depth: `+pitch_env` semitones on each of the three targets
- Decay: the engine is `exp(−t/τ)` with `τ = pitch_dec`. That reaches 5 % at
  3τ. Start with Serum D = `3 · pitch_dec` s and the decay curve pulled hard
  toward exponential. **Calibration D:** match the 808-style drop on the
  `Deep Sub` preset with `pitch_env` forced to 12 st, τ = 0.05 s.

LFO → Cutoff:
- Only when `lfo_depth > 0.01`
- LFO 1: shape sine, rate `lfo_rate` Hz, **not** tempo-synced, trigger mode
  per-note (phase 0 at note-on), **bipolar**
- Depth: `±lfo_depth` octaves of cutoff (reuse Calibration B's constant)

### 4.10 Optional: the unison thickener
Not in the genome. The UI knob `unison ∈ 0..1` runs `dsp::thicken`:
4 extra copies at `±c` and `±0.45c` cents, `c = 5 + 25·amount`, each at gain
`0.18 + 0.45·amount` added to the dry centre, with 4 ms staggers. In Serum:
- Osc A and B: Unison 5, Detune so the outer voices sit at `±c` cents,
  Blend so the four detuned voices sum to roughly `4·(0.18 + 0.45·amount)`
  relative to centre. Serum's inner voices will not be at exactly 0.45c;
  nobody will notice. Ship it as a flag, default off.

### 4.11 Optional: the original sound as a sample layer
Serum 2 can run an oscillator in sample mode. When the patch came from a
clone, carry the source audio alongside the synthesized recipe:
- File: the exact 1.2 s window the matcher used (loudest window, 22.05k mono,
  peak-normalised to 0.9), written next to the JSON as `<name>.source.wav`.
- Root key: the detected note, so keytracking plays it in tune.
- Level: 0 by default. The patch opens sounding like the clone; raise the
  sample for A/B, or layer the clone's sub under the original's top end.
- Not through the filter, no loop, one-shot.
- Only when the patch has a source clip. Presets and garden patches don't.

CLI shape: `synfection export --serum --from-audio hook.wav` runs the match
first, then exports with the sample layer attached.

---

## 5. Handoff JSON (what the exporter writes, what your automation reads)

One JSON per patch plus the shared wavetable. Real units only; no Serum knob
positions in here so the file stays correct if Serum changes. Your automation
applies the four calibration curves.

```json
{
  "schema": "synfection-serum-handoff/1",
  "source": {
    "name": "Garage Stab",
    "note_midi": 45,
    "genome_norm": [0.50, 0.55, 0.56, 0.50, 0.30, 0.06, 0.30, 0.60, 0.50, 0.65,
                    0.02, 0.28, 0.01, 0.30, 0.10, 0.25, 0.00, 0.50, 0.40, 0.00],
    "engine": "synfection v3 genome, 22050 Hz reference"
  },
  "wavetable": { "file": "synfection_wt.wav", "frames": 256,
                 "samples_per_frame": 2048, "harmonics": 72 },
  "osc_a": { "enabled": true, "wt_pos": 0.50, "level": 0.50, "fine_cents": 0.0 },
  "osc_b": { "enabled": true, "wt_pos": 0.55, "level": 0.50, "fine_cents": 6.0 },
  "sub":   { "enabled": true, "shape": "sine", "octave": -1, "level": 0.18,
             "through_filter": true },
  "noise": { "enabled": true, "level": 0.036, "through_filter": true },
  "filter": { "type_parity": "Low 12", "type_taste": "MG Low 12",
              "cutoff_hz": 1288.0, "q": 4.8, "peak_db": 13.6,
              "env_sweep_octaves": 2.34, "top_hz": 6524.0 },
  "env_amp":    { "a_s": 0.0011, "d_s": 0.058, "s": 0.10, "r_s": 0.049, "curve": "linear" },
  "env_filter": { "a_s": 0.0011, "d_s": 0.055, "s": 0.25, "r_s": 0.049,
                  "target": "filter.cutoff", "depth_octaves": 2.34 },
  "env_pitch":  { "enabled": false, "semitones": 0.0, "tau_s": 0.045,
                  "targets": ["osc_a.pitch", "osc_b.pitch", "sub.pitch"] },
  "lfo_cutoff": { "enabled": false, "shape": "sine", "rate_hz": 0.45,
                  "tempo_sync": false, "retrigger": true, "bipolar": true,
                  "depth_octaves": 0.0 },
  "drive": { "enabled": true, "curve": "tanh", "k": 6.0, "mix": 0.30,
             "placement": "post_voice_sum" },
  "unison_thicken": { "enabled": false, "amount": 0.0 },
  "sample_layer": { "enabled": false, "file": null, "root_midi": 45,
                    "level": 0.0, "keytrack": true, "loop": false,
                    "through_filter": false }
}
```

Derived fields (`top_hz`, `env_sweep_octaves`, `peak_db`, `sub.level`) are
written out so the consumer never re-implements engine math.

CLI shape when we build it:

```
synfection export --serum <patch.genome.txt> [--out dir/] [--bright] [--unison 0.4]
synfection export --serum --all-presets --out serum/     # 12 factory patches
```

Writes `dir/<name>.serum.json` and `dir/synfection_wt.wav` (once).

---

## 6. Calibration plan (one evening, then never again)

1. `synfection render` the 12 presets at their stored notes as reference wavs
   (22.05k, no unison).
2. Build each in Serum 2 through the automation, bounce at the same note.
3. Resample the bounce to 22.05k and score with `dsp::multiscale_stft` (or
   `lab/losses.py`) against the reference. The parity fixture path in
   `tests/parity.rs` shows how to load a wav to the engine's frame.
4. Tune the four constants (A resonance, B cutoff-mod units, C drive, D pitch
   decay) until the loss stops moving, then switch filter to `MG Low 12` and
   let the loss go up on purpose. That is the point.

Presets that stress each constant: `Acid Squelch` (A, B), `Reese Growl` (C),
`Innerbloom` (B, env shape), `Deep Sub` with a forced pitch env (D).

---

## 7. The road after export

Direction chosen: **instrument**, not matcher. Export first because it is
useful regardless and costs an afternoon.

1. **Serum 2 export** (this doc).
2. **Real-time voice in Rust.** Anti-aliased wavetable oscillators (mip-mapped
   frames from the same table), a TPT state-variable or ladder filter with
   `tanh` in the loop, linear ADSRs, 44.1/48k. Keep the 20-param genome so
   every preset and saved patch survives. Ear-check preset by preset against
   the old engine.
3. **cpal audio callback, MIDI in (midir), 8-voice poly** in the standalone.
4. **Python twin of the new engine and retrain GenoNet.** Turn augmentation on
   by default (`train.py --augment` currently defaults to 0; confirm the
   shipped `genonet.bin` was trained with it). Add the real-audio path from
   section 9 so your own stems train it. Swap the (1+16)-ES refiner for CMA-ES: fewer of the ~1000
   renders per clone, better convergence. Only the trainer needs gradients;
   the Rust refiner is already gradient-free, so a non-differentiable filter
   in Rust costs nothing there.
5. **nih-plug wrapper** for CLAP + VST3. `nih_plug_egui` keeps the plant UI,
   with an egui version wrangle (app is on eframe 0.29).
6. **Grow the genome** only once the engine is settled: FM operator, second
   filter, an LFO that can hit pitch and amp, stereo. Each is a retrain.
7. **Taste model, properly.** Rate a fresh round on the new engine, ship the
   mel reward (the binary currently only has the knob-space MLP, ~150 ratings
   mostly on v1 renders), and hand it to the refiner as a prior.

Stop investing in: the 12 built-in loop patterns (MIDI import covers it) and
the 22.05k render path (dies with the old engine).

---

## 8. Code quirks found on the way (fix when touching the area)

- `render` derives `nyq` from the sample rate passed in, so loops (44.1k) sweep
  filter envelopes to twice the height of the preview (22.05k). Pin `top` to
  the 22.05k definition, or accept that loops are brighter and document it.
- `wt_profile` rebuilds the whole wavetable on every call. Harmless today,
  wasteful in a real-time voice. Build once, `static`.
- `dsp::thicken` is linear-interp resampling; fine as a preview, aliasy as a
  product. Dies with the new engine's real unison.
- `dsp::resample` is linear too. Use it only for feature extraction, as the
  comment says.
- `train.py` `--augment` default 0.0 (see step 4).
- `lab/drumset.py` writes `drums/oneshots/manifest.jsonl` "for real-data
  training", and nothing in the repo reads it. The slicer exists, the loader
  does not. Section 9 is that loader.

---

## 9. Real audio into training (the missing loader)

Today every training example is a random genome the engine rendered for
itself. Real audio never reaches `train.py`. The fix is small because the
trainer already has the right loss: the spectral term compares re-rendered
audio to target audio and needs no genome label, only the audio and its pitch.

### 9.1 `lab/realset.py`: folder of sounds → training tensor

```
python realset.py --dir real/bass --out real/bass.npz [--label bass]
python realset.py --manifest drums/oneshots/manifest.jsonl --labels kick --out real/kicks.npz
```

Per file, recursively (`wav`, `aif`, `flac`, `mp3` via librosa):
1. Mono, resample to 22 050 Hz.
2. Loudest 1.2 s window (`demos.best_window`), zero-pad to `N = 26 460`.
3. Peak-normalise to 0.9. Matches engine output and the parity fixture.
4. Pitch via `match.detect_note`. Drop the clip if unvoiced or outside
   MIDI 36..72, the trainer's note range.
5. Drop if window RMS is under −40 dBFS.

Output `.npz`: `audio [n, N] float32`, `note [n] int`, `path [n] str`,
`label [n] str` (folder name or `--label`). Never committed; `real/` goes in
`.gitignore`.

### 9.2 `train.py` additions

```
--real real/bass.npz real/kicks.npz    # one or more sets
--real-frac 0.25                       # share of each batch drawn from them
```

Per step with batch size `B`: `k = round(real_frac · B)` rows sampled with
replacement from the real sets, `B − k` synthetic rows as now.

- Parameter MSE: synthetic rows only (mask the real rows; they have no label).
- Spectral loss: all rows.
- Augmentation: synthetic rows only. Real rows already carry real-world grime;
  that is the point of them.
- Validation: keep the synthetic val set as-is, hold out 10 % of each real set
  and report its spectral loss separately as `val-real`. That number is the
  one that tells you whether your stems are being learned.

Guardrails: cap `real_frac` at 0.5 so synthetic rows keep anchoring the
parameter head. Weight sets by size so a folder of 400 kicks does not drown
30 basslines, or pass `--real-weights`.

### 9.3 What to feed it, and what not to

In lane for the current engine: basses, reeses, leads, plucks, stabs, kicks
(the v3 pitch env exists for them). Out of lane: snares, hats, chords, pads
with movement, anything stereo-wide. Out-of-lane audio teaches a 20-param mono
engine to hedge and every match gets mushier. The restriction lifts with the
engine in roadmap step 2.

### 9.4 After that, taste on real sounds

`reward_mel.py` scores a rendered mel. Nothing stops it scoring a real clip's
mel. A rating round that mixes your own favourite one-shots with engine
renders teaches the reward what "good" is independent of what the engine can
make, which is the prior you want in the refiner once the engine can make
more. One extra flag on `serve.py` to list wavs from a folder; no new model.

### 9.5 Day-to-day once it exists

Drop wavs into `real/<label>/`, run `realset.py` once per folder, add the
`.npz` to the train command. No code. The rating loop was already code-free;
this makes the matching loop match it.
