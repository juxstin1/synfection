"""
Genopatch: reverse-engineer a synth patch from a sound, then re-render it.

  1. GenoNet gives an instant genome from the mel-spectrogram.
  2. Gradient refinement (analysis-by-synthesis) polishes the genome by
     backprop through the differentiable synth against a multi-scale spectral
     loss — the same objective the net was trained on.

Modes:
  python match.py --audio hook.wav --out remake.wav
  python match.py --selftest
  python match.py --genome 0.2,0.8,...  --note C3 --out patch.wav
"""

import argparse
import numpy as np
import soundfile as sf
import torch

from synth import (render, melspec, denorm, to_wav, upgrade_genome,
                   N_PARAMS, PARAM_NAMES, SR, N)
from model import GenoNet, device
from losses import multiscale_stft

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def note_to_midi(s):
    s = s.strip()
    if s.lstrip("-").isdigit():
        return int(s)
    name = s[:-1].upper().replace("S", "#")
    return NOTE_NAMES.index(name) + (int(s[-1]) + 1) * 12


def detect_note(audio_np):
    import librosa
    try:
        f0, _, _ = librosa.pyin(audio_np, sr=SR, fmin=40, fmax=2000)
        f0 = f0[~np.isnan(f0)]
        if len(f0):
            return int(round(69 + 12 * np.log2(np.median(f0) / 440.0)))
    except Exception:
        pass
    return 48


def load_target(path, dev):
    import librosa
    a, _ = librosa.load(path, sr=SR, mono=True)
    note = detect_note(a)
    a = np.pad(a, (0, max(0, N - len(a))))[:N]
    a = a / (np.max(np.abs(a)) + 1e-9) * 0.9
    return torch.tensor(a, dtype=torch.float32, device=dev), note


def refine(genome, target, note, dev, iters=300, lr=0.05):
    """Gradient analysis-by-synthesis in logit space."""
    p = genome.clamp(1e-4, 1 - 1e-4)
    z = torch.log(p / (1 - p)).detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([z], lr=lr)
    tgt = target.unsqueeze(0)
    best_g, best_l = genome.detach().clone(), float("inf")
    for i in range(iters):
        g = torch.sigmoid(z)
        rec = render(g, note).unsqueeze(0)
        loss = multiscale_stft(rec, tgt)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if loss.item() < best_l:
            best_l, best_g = loss.item(), torch.sigmoid(z).detach().clone()
    return best_g, best_l


def genopatch(target, note, net, dev, iters=300):
    with torch.no_grad():
        mel = melspec(target).unsqueeze(0)
        guess = net(mel)[0]
        l0 = multiscale_stft(render(guess, note).unsqueeze(0),
                             target.unsqueeze(0)).item()
    g, l1 = refine(guess, target, note, dev, iters=iters)
    return g, l1, guess.detach(), l0


def print_patch(genome):
    p = denorm(genome).cpu().numpy()
    print("  matched patch:")
    for i, name in enumerate(PARAM_NAMES):
        print(f"    {name:12s} {p[i]:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="genonet.pt")
    ap.add_argument("--audio")
    ap.add_argument("--genome")
    ap.add_argument("--note", default=None)
    ap.add_argument("--out", default="remake.wav")
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--no-refine", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    dev = device()

    if a.genome:
        g = torch.tensor([float(x) for x in a.genome.split(",")],
                         dtype=torch.float32, device=dev)
        g = upgrade_genome(g)   # accepts legacy 15-param (v1) genomes too
        note = note_to_midi(a.note) if a.note else 48
        sf.write(a.out, to_wav(render(g, note)), SR)
        print(f"rendered genome at MIDI {note} -> {a.out}")
        return

    net = GenoNet().to(dev)
    net.load_state_dict(torch.load(a.model, map_location=dev))
    net.eval()

    if a.selftest:
        gen = torch.Generator(device=dev).manual_seed(7)
        true_g = torch.rand(N_PARAMS, device=dev, generator=gen)
        note = 48
        target = render(true_g, note, gen=gen)
        g, l1, g0, l0 = genopatch(target, note, net, dev, a.iters)
        print(f"spec-loss  nn-only {l0:.3f}  ->  refined {l1:.3f}")
        print(f"genome MAE nn-only {(g0-true_g).abs().mean():.3f}  ->  "
              f"refined {(g-true_g).abs().mean():.3f}")
        sf.write("selftest_target.wav", to_wav(target), SR)
        sf.write("selftest_remake.wav", to_wav(render(g, note)), SR)
        print_patch(g)
        print("wrote selftest_target.wav / selftest_remake.wav")
        return

    if not a.audio:
        ap.error("need --audio, --genome, or --selftest")

    target, det = load_target(a.audio, dev)
    note = note_to_midi(a.note) if a.note else det
    print(f"target {a.audio}  match at MIDI {note} "
          f"({'given' if a.note else 'detected'})")
    if a.no_refine:
        with torch.no_grad():
            g = net(melspec(target).unsqueeze(0))[0]
        l1 = multiscale_stft(render(g, note).unsqueeze(0),
                             target.unsqueeze(0)).item()
        print(f"spec-loss (nn-only) {l1:.3f}")
    else:
        g, l1, g0, l0 = genopatch(target, note, net, dev, a.iters)
        print(f"spec-loss  nn-only {l0:.3f}  ->  refined {l1:.3f}")
    sf.write(a.out, to_wav(render(g, note)), SR)
    np.savetxt(a.out + ".genome.txt", g.cpu().numpy(), fmt="%.5f")
    print_patch(g)
    print(f"remake -> {a.out}   genome -> {a.out}.genome.txt")


if __name__ == "__main__":
    import os
    import sys
    main()
    sys.stdout.flush(); sys.stderr.flush()   # os._exit skips buffer flush
    os._exit(0)   # dodge ROCm-on-Windows teardown deadlock (see train.py)
