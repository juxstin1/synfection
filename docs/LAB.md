# The lab — training, RLHF, and research notes

Everything here needs Python 3.10+ with PyTorch (`pip install -r requirements.txt`).
The shipped Rust binary only *runs* the model; this is where it's made.

## How Synplant 2 / Genopatch works (the research)

Sonic Charge's Synplant 2 ships **Genopatch**: feed it audio, a neural net emits
synth-engine parameters that recreate the sound. They trained it on millions of
sounds rendered from their own engine — it learns the inverse map **audio →
parameters**. In the literature this is **automatic synthesizer programming /
inverse synthesis / sound matching**: InverSynth (CNN spectrogram→params), Flow
Synthesizer (Esling, VAE/flows over patch space), Spiegelib (research lib), and
**DDSP** (Google — differentiable synths trained with a spectral loss).

The free lunch: *you own the synth*, so `random params → audio` gives infinite
labeled data. synfection keeps Synplant's genetic spirit too — the NN gives the
first guess, then analysis-by-synthesis polishes it.

## Architecture

The whole synth is **differentiable PyTorch on the GPU** — one engine used for
data, training, and refinement. We backprop a **multi-scale spectral loss** from
the rendered audio to the predicted parameters (the DDSP recipe), so matches
sound right despite synth params being many-to-one.

- **Engine** (`synth.py`, v2): 2 **wavetable** oscillators (8-frame harmonic
  morph: sine → tri → square → saw → pulses → formant → rich) + sub sine +
  spectrally-shaped noise + **drive/waveshaper** → time-varying **resonant
  lowpass** (analog 2-pole magnitude evaluated per-harmonic, fully
  differentiable) → ADSR amp + filter envelopes. **16-parameter genome.**
- **Net** (`model.py`): compact CNN, log-mel → 16 params (sigmoid).
- **Loss** (`losses.py`): multi-scale STFT (5 FFT sizes, lin + log magnitude).
- **Training** (`train.py`): **on-the-fly infinite data** — every step samples
  random genomes+notes, renders targets, and trains the net to invert them with
  a hybrid *param-MSE + spectral* loss. No dataset files needed.
- **Matching** (`match.py`): net guess → Adam refinement through the synth
  (the Rust binary uses a (1+λ) evolution strategy instead — no autograd).

Genome: `osc1_wt, osc2_wt, osc2_detune, osc_mix, sub_level, noise_level, drive,
cutoff, reso, filt_env, filt_a, filt_d, amp_a, amp_d, amp_s, amp_r`.

The shipped checkpoint was trained for **6000 steps × batch 32 = 192,000
freshly-rendered examples** (`train.log`): val spectral loss 3.66 → 1.51,
val param-MAE 0.249 → 0.183, ~50 min on a consumer GPU.

> `synth_v1.py` is the archived 15-param v1 engine (saw↔square morph, no drive).
> Legacy 15-value genomes auto-upgrade everywhere with a neutral `drive=0`.

## Train / match / export

```bash
python train.py --steps 6000 --bs 32 --out genonet.pt   # train
python match.py --selftest                              # prove the loop
python match.py --audio hook.wav --out remake.wav       # match with Adam refine
python export_net.py                                    # -> weights/genonet.bin for the Rust binary
```

After retraining, run `export_net.py`, regenerate `weights/parity.json` (see
`tests/parity.rs` for what it checks), and rebuild the binary.

## The RLHF loop (teach it taste)

Spectral loss says nothing about whether a patch sounds *good*. So: generate
rateable patches from musical **archetypes** (bass / reese / lead / pluck / stab
/ pad / keys), star-rate them in a browser, and train a reward model that steers
the next generation.

```bash
python genpatches.py --n 48 --dir rlhf/round01          # generate round 1
python serve.py --dir rlhf/round01                      # rate: 1-5 stars, 0 discard, saves live

# reward model on knob-space (fast, learns archetype-level taste)
python reward.py train --rounds rlhf/round01 --out reward.pt
python reward.py gen   --reward reward.pt --n 48 --pool 600 --dir rlhf/round02

# perceptual reward on the *rendered mel* (learns within-archetype quality)
python reward_mel.py train --rounds rlhf/round01 rlhf/round02 --out reward_mel.pt
python reward_mel.py gen   --reward reward_mel.pt --n 48 --dir rlhf/round03
```

`gen` mixes **exploit** (top predicted reward, diversity-guarded, per-archetype
capped) with **explore** (MC-dropout uncertainty — the patches the model would
learn most from). Both trainers report honest **out-of-fold** correlations
(group-aware for the mel model so augmented views never leak across folds).

This repo ships 3 rated rounds (~149 ratings) in `rlhf/` as jsonl+csv — wavs are
regenerable from the genomes. Rounds 01–03 were rated on v1-engine renders;
retrain reward models before using `gen`, and rate a fresh v2-native round when
possible.

## Python extras

```bash
python vary.py --genome remake.wav.genome.txt --n 6      # siblings (also in the binary)
python loops.py --pack --genomes a.txt --keys F1,G1 \
                --bpms 130,138 --patterns garage_roll,reese_hold --dir packs/demo
python demos.py bass="stems/track/bass.wav"              # batch-clone stems
python dataset.py --n 12 --note C3 --dir gallery         # ear-check the engine
```

## Limitations & next upgrades

- **Monophonic single-note** timbre matching — best on bass/lead/pluck hooks,
  not chords or drums. Feed isolated stems (demucs `bass`/`other`) or one-shots.
- 2-pole filter (smooth, analog-style) rather than a steep/zero-delay model.
- **Next:** bigger genome (FM operator, 2nd filter, LFO/mod-matrix); CMA-ES
  refiner; reward-model prior inside refinement; export to a real VST/Surge
  patch; MIDI input in the UI.
