# synfection 🧬🎛️

A small neural network that **reverse-engineers a synth patch from a sound** and
re-renders it — a from-scratch, fully **differentiable** take on **Synplant 2's
Genopatch** — plus an **RLHF loop** that learns which patches *sound good* from
star ratings, and a sequencer that turns matched patches into tempo-locked loops.

Drop in a synth hook → get back the synth parameters (the "genome") that recreate
it, plus a rendered `.wav`.

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
first guess, then **gradient analysis-by-synthesis** polishes it.

## The "real deal" architecture

The whole synth is **differentiable PyTorch on the GPU** — one engine used for
data, training, and refinement. That's the key upgrade: we backprop a **multi-
scale spectral loss** from the rendered audio to the predicted parameters (the
DDSP recipe), so matches sound right despite synth params being many-to-one
(different knobs, same timbre).

- **Engine** (`synth.py`, v2): 2 **wavetable** oscillators (8-frame harmonic
  morph: sine → tri → square → saw → pulses → formant → rich) + sub sine +
  spectrally-shaped noise + **drive/waveshaper** → time-varying **resonant
  lowpass** (analog 2-pole magnitude evaluated per-harmonic, fully
  differentiable) → ADSR amp + filter envelopes. **16-parameter genome.**
- **Net** (`model.py`): compact CNN, log-mel → 16 params (sigmoid).
- **Loss** (`losses.py`): multi-scale STFT (5 FFT sizes, lin + log magnitude).
- **Training** (`train.py`): **on-the-fly infinite data** — every step samples
  random genomes+notes, renders targets, and trains net to invert them with a
  hybrid *param-MSE + spectral* loss. No dataset files needed.
- **Matching** (`match.py`): net guess → Adam refinement through the synth.

Genome: `osc1_wt, osc2_wt, osc2_detune, osc_mix, sub_level, noise_level, drive,
cutoff, reso, filt_env, filt_a, filt_d, amp_a, amp_d, amp_s, amp_r`.

The shipped checkpoint (`genonet.pt`, gitignored — retrain in ~50 min on a
consumer GPU) was trained for **6000 steps × batch 32 = 192,000 freshly-rendered
examples** (see `train.log`): val spectral loss 3.66 → 1.51, val param-MAE
0.249 → 0.183.

> `synth_v1.py` is the archived 15-param v1 engine (saw↔square morph, no drive).
> Old 15-value genomes still load everywhere — they're auto-upgraded with a
> neutral `drive=0` (approximate under the v2 wavetable, close enough to reuse).

## Quick start

```bash
pip install -r requirements.txt

# train (uses the GPU automatically; no dataset prep)
python train.py --steps 6000 --bs 32 --out genonet.pt

# prove the loop on a known patch
python match.py --selftest

# remake a real hook
python match.py --audio hook.wav --out remake.wav
#   -> remake.wav + remake.wav.genome.txt + printed parameters

# ear-check the engine itself
python dataset.py --n 12 --note C3 --dir gallery

# render a genome by hand (16 comma-separated 0..1 values)
python match.py --genome 0.4,0.5,0.5,0.3,0.8,0.0,0.2,0.4,0.2,0.4,0.1,0.4,0.02,0.4,0.7,0.2 --note C3 --out patch.wav
```

`--audio` auto-detects pitch (librosa pyin) and renders the match at that note;
override with `--note C3` / `--note 48`. Add `--no-refine` for the raw NN guess,
or `--iters 600` for a longer polish.

## Beyond matching

```bash
# sibling sounds of a matched patch (mutate / breed the genome)
python vary.py --genome remake.wav.genome.txt --n 6 --amount 0.15 --note G1 --dir variations
python vary.py --breed a.genome.txt b.genome.txt --n 6 --note C2

# tempo-locked, seamless bass loops from a patch (UKG/speed-garage/house patterns)
python loops.py --genome remake.wav.genome.txt --key F1 --bpm 138 --pattern garage_roll
python loops.py --pack --genomes remake.wav.genome.txt --keys F1,G1 --bpms 130,138 \
                --patterns garage_roll,reese_hold --dir packs/demo

# batch-clone a folder of stems in one process
python demos.py bloom_bass="stems/Bloom/bass.wav" closer_lead="stems/Closer/other.wav"
```

## The RLHF loop (teach it taste)

Spectral loss says nothing about whether a patch sounds *good*. So: generate
rateable patches from musical **archetypes** (bass / reese / lead / pluck / stab
/ pad / keys), star-rate them in a browser, and train a reward model that steers
the next generation.

```bash
# round 1: generate 48 archetype patches + a browser rater
python genpatches.py --n 48 --dir rlhf/round01
python serve.py --dir rlhf/round01        # rate: 1-5 stars, 0 = discard, saves live

# reward model on knob-space (fast, learns archetype-level taste)
python reward.py train --rounds rlhf/round01 --out reward.pt
python reward.py gen   --reward reward.pt --n 48 --pool 600 --dir rlhf/round02

# perceptual reward model on the *rendered mel* (learns within-archetype quality)
python reward_mel.py train --rounds rlhf/round01 rlhf/round02 --out reward_mel.pt
python reward_mel.py gen   --reward reward_mel.pt --n 48 --dir rlhf/round03
```

`gen` mixes **exploit** (top predicted reward, diversity-guarded, per-archetype
capped) with **explore** (MC-dropout uncertainty — the patches the model would
learn most from). Both trainers report honest **out-of-fold** correlations
(group-aware for the mel model, so augmented views never leak across folds).

This repo ships 3 rated rounds (~149 ratings) in `rlhf/` as jsonl+csv — the wavs
are regenerable from the genomes.

## Limitations & next upgrades

- **Monophonic single-note** timbre matching — best on bass/lead/pluck hooks, not
  chords or drums. For a melodic hook, feed an isolated stem (demucs `bass`/
  `other`) or a clean one-shot.
- 2-pole filter (smooth, analog-style) rather than a steep/zero-delay model.
- RLHF rounds 01–03 were rated on v1 renders; the reward checkpoints predate the
  v2 engine. Retrain them (`reward.py train ...`) — legacy genomes auto-upgrade —
  and rate a fresh v2-native round when possible.
- **Next:** bigger genome (FM operator, 2nd filter, LFO/mod-matrix); CMA-ES as an
  alt refiner; reward-model prior inside `match.py` refinement; export to a real
  VST/Surge patch.

## Note for Cue / the crate

Point `match.py --audio` at an isolated hook (demucs `other`/`bass` stem or a
clean loop) to clone a lead into a reusable patch you can replay at any note in
your set — e.g. transpose a Sammy-Virji-style bass stab into your track's key.
