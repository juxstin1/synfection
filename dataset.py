"""
Render a gallery of random patches to .wav for ear-checking the synth engine.
(Training needs no precomputed dataset — train.py generates data on the fly.)

    python dataset.py --n 12 --note C3 --dir gallery
"""

import argparse
import os
import soundfile as sf
import torch

from synth import render, to_wav, N_PARAMS, SR
from match import note_to_midi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--note", default="C3")
    ap.add_argument("--dir", default="gallery")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.dir, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gen = torch.Generator(device=dev).manual_seed(a.seed)
    note = note_to_midi(a.note)
    g = torch.rand(a.n, N_PARAMS, device=dev, generator=gen)
    audio = render(g, note, gen=gen)
    for i in range(a.n):
        p = os.path.join(a.dir, f"patch_{i:02d}.wav")
        sf.write(p, to_wav(audio[i]), SR)
    print(f"wrote {a.n} patches at {a.note} -> {a.dir}/")


if __name__ == "__main__":
    import os
    main()
    os._exit(0)   # dodge ROCm-on-Windows teardown deadlock
