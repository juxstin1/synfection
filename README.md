# synfection 🧬🎛️

A small neural network that **reverse-engineers a synth patch from a sound** and
re-renders it — a from-scratch, fully **differentiable** take on **Synplant 2's
Genopatch**.

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

- **Engine** (`synth.py`): 2 band-limited harmonic oscillators (saw↔square morph)
  + sub sine + spectrally-shaped noise → time-varying **resonant lowpass**
  (analog 2-pole magnitude evaluated per-harmonic, fully differentiable) → ADSR
  amp + filter envelopes. **15-parameter genome.**
- **Net** (`model.py`): compact CNN, log-mel → 15 params (sigmoid).
- **Loss** (`losses.py`): multi-scale STFT (5 FFT sizes, lin + log magnitude).
- **Training** (`train.py`): **on-the-fly infinite data** — every step samples
  random genomes+notes, renders targets, and trains net to invert them with a
  hybrid *param-MSE + spectral* loss. No dataset files needed.
- **Matching** (`match.py`): net guess → Adam refinement through the synth.

Genome: `osc1_wave, osc2_wave, osc2_detune, osc_mix, sub_level, noise_level,
cutoff, reso, filt_env, filt_a, filt_d, amp_a, amp_d, amp_s, amp_r`.

## Quick start

```bash
# train (uses the ROCm GPU automatically; no dataset prep)
python train.py --steps 6000 --bs 32 --out genonet.pt

# prove the loop on a known patch
python match.py --selftest

# remake a real hook
python match.py --audio hook.wav --out remake.wav
#   -> remake.wav + remake.wav.genome.txt + printed parameters

# ear-check the engine itself
python dataset.py --n 12 --note C3 --dir gallery

# render a genome by hand (15 comma-separated 0..1 values)
python match.py --genome 0.2,0.8,0.5,0.3,0.1,0.0,0.6,0.2,0.4,0.1,0.2,0.05,0.2,0.4,0.2 --note C3 --out patch.wav
```

`--audio` auto-detects pitch (librosa pyin) and renders the match at that note;
override with `--note C3` / `--note 48`. Add `--no-refine` for the raw NN guess,
or `--iters 600` for a longer polish.

## Limitations & next upgrades

- **Monophonic single-note** timbre matching — best on bass/lead/pluck hooks, not
  chords or drums. For a melodic hook, feed an isolated stem (demucs `bass`/
  `other`) or a clean one-shot.
- 2-pole filter (smooth, analog-style) rather than a steep/zero-delay model.
- **Next:** bigger genome (FM operator, 2nd filter, LFO/mod-matrix); CMA-ES as an
  alt refiner; perceptual (mel/loudness) weighting in the loss; export to a real
  VST/Surge patch.

## Note for Cue / the crate

Point `match.py --audio` at an isolated hook (demucs `other`/`bass` stem or a
clean loop) to clone a lead into a reusable patch you can replay at any note in
your set — e.g. transpose a Sammy-Virji-style bass stab into your track's key.
