# synfection 🧬🎛️

**Drop in a sound → get back the synth patch that recreates it.**

A tiny neural network listens to a synth hook (a bass stab, a lead, a pluck),
figures out the 16 synth knobs that produce that sound, and re-renders it as a
patch you can retune, mutate, and turn into tempo-locked loops. Think
[Synplant 2's Genopatch](https://soniccharge.com/synplant), built from scratch.

One small binary. No install, no Python, no plugins — the trained network is
baked into the executable.

![synfection UI](docs/ui.png)

## Install

Grab a binary from [**Releases**](https://github.com/juxstin1/synfection/releases/latest):

| Platform | File |
|---|---|
| Windows | `synfection-windows-x86_64.exe` |
| macOS (Apple Silicon) | `synfection-macos-arm64` |
| macOS (Intel) | `synfection-macos-x86_64` |
| Linux | `synfection-linux-x86_64` |

macOS / Linux: `chmod +x synfection-*` first. On macOS the first launch needs
right-click → Open (unsigned binary), or `xattr -d com.apple.quarantine synfection-*`.

Or build from source (needs [Rust](https://rustup.rs)):

```bash
cargo install --git https://github.com/juxstin1/synfection
```

## Use it

**Double-click the binary** (or run `synfection` with no arguments) to open the
app:

- **Clone**: drop a `.wav` on the window (or hit *open wav*) and the net grows a
  patch that recreates it, with live progress.
- **Presets**: 12 factory patches — garage stabs, an Innerbloom-style long stab,
  organ stab, reese, acid, deep sub, plucks, pads — browse with ◀ ▶.
- **Garden**: grow 8 offspring from the current patch (or from bass / reese /
  stab / pad... archetype seeds), each **scored by a reward model trained on
  real star-ratings**. Click a bud to hear it, ✓ to adopt it as the new seed and
  grow the next generation — Synplant-style breed-to-taste.
- **Plant**: drag the 16 branches to garden the patch by hand; precise sliders
  with real values (Hz / cents / seconds) on the right.
- **Loop lab**: 12 UK-garage / bass-house / DnB patterns with swing — loops play
  gapless until you stop them, and save as 44.1k wavs.
- Plus A/B compare, undo/redo (ctrl+z/y), reward-ranked *random*, and a master
  volume.

Or from the terminal:

```bash
# clone a sound -> remake.wav + remake.wav.genome.txt (the 16 knob values)
synfection match hook.wav

# play a genome at any note
synfection render remake.wav.genome.txt --note F1 -o bass_f1.wav

# spawn 6 sibling sounds / breed two patches
synfection vary remake.wav.genome.txt --n 6 --amount 0.15
synfection breed a.genome.txt b.genome.txt

# tempo-locked seamless loop (patterns: synfection patterns)
synfection loop remake.wav.genome.txt --key F1 --bpm 138 --pattern garage_roll

# sanity-check the whole pipeline on a known patch
synfection selftest
```

`match` auto-detects the pitch; override with `--note C2`. Feed it an **isolated,
mono-ish hook** — a one-shot or a demucs `bass`/`other` stem works far better
than a full mix. It matches single-note timbre, not chords or drums.

## How it works

The synth engine (2 wavetable oscillators, sub, noise, drive, resonant filter,
ADSR envelopes — a 16-knob "genome") exists twice: once in differentiable
PyTorch where the net was trained on **192,000 self-rendered sounds** (the
engine labels its own data), and once in Rust for this binary. A compact CNN
maps a log-mel spectrogram to the genome, then an evolutionary polish refines
the match against a multi-scale spectral loss. The Rust port is verified
sample-level against PyTorch in CI (`tests/parity.rs`).

Training, the RLHF "does it actually sound good" loop, and the research notes
live in [docs/LAB.md](docs/LAB.md) — that side needs Python + PyTorch.

## License

MIT
