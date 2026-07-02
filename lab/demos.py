"""Batch-clone real stems into synth patches (one process, one model load).

Each target gets the loudest 1.2s window picked out, pitch-detected, matched
(NN guess + gradient refinement), and written as a target/remake wav pair.

  python demos.py bloom_bass="stems/101 - Bloom/bass.wav" closer_lead="stems/44/other.wav"
  python demos.py --out demos --iters 450 lead=hook.wav
"""
import argparse
import os
import numpy as np
import soundfile as sf
import librosa
import torch

from synth import render, melspec, SR, N
from model import GenoNet, device
from losses import multiscale_stft
from match import clone_patch, detect_note


def best_window(y, sr):
    win = int(1.2 * sr); hop = int(0.1 * sr); best = (-1, 0)
    for s in range(0, max(1, len(y) - win), hop):
        e = float(np.sum(y[s:s + win] ** 2))
        if e > best[0]: best = (e, s)
    s = best[1]; c = y[s:s + win]
    if len(c) < win: c = np.pad(c, (0, win - len(c)))
    return c / (np.max(np.abs(c)) + 1e-9) * 0.9, s / sr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("targets", nargs="+", help="label=path/to/stem.wav pairs")
    ap.add_argument("--model", default="genonet.pt")
    ap.add_argument("--out", default="demos")
    ap.add_argument("--iters", type=int, default=450)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    dev = device()
    net = GenoNet().to(dev)
    net.load_state_dict(torch.load(a.model, map_location=dev))
    net.eval()

    for spec in a.targets:
        label, path = spec.split("=", 1)
        y, _ = librosa.load(path, sr=SR, mono=True)
        clip, t0 = best_window(y, SR)
        note = detect_note(clip)
        tgt = torch.tensor(clip, dtype=torch.float32, device=dev)
        g, l1, g0, l0 = clone_patch(tgt, note, net, dev, iters=a.iters)
        sf.write(f"{a.out}/{label}_target.wav", clip.astype(np.float32), SR)
        sf.write(f"{a.out}/{label}_remake.wav",
                 render(g, note).detach().cpu().numpy().astype(np.float32), SR)
        np.savetxt(f"{a.out}/{label}.genome.txt", g.cpu().numpy(), fmt="%.5f")
        print(f"{label:13s} note {librosa.midi_to_note(note):4s} @ {t0:5.1f}s   "
              f"spec {l0:.2f} -> {l1:.2f}")
    print(f"\nwrote pairs to {a.out}/")


if __name__ == "__main__":
    import sys
    main()
    sys.stdout.flush(); sys.stderr.flush()   # os._exit skips buffer flush
    os._exit(0)   # dodge ROCm-on-Windows teardown deadlock (see train.py)
