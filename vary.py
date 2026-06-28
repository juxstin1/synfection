"""
Axis 1 — SOUND variations (patch space). Same note, different character.

Takes a genome (from match.py's *.genome.txt, or inline) and spawns siblings by
mutating / breeding the 15 knobs, then renders each. This is the seed of the
breeding / grows-to-taste mode.

  # sibling sounds of a matched patch
  python vary.py --genome remake_bass.wav.genome.txt --n 6 --amount 0.15 --note G1 --dir variations

  # breed two patches together
  python vary.py --breed a.genome.txt b.genome.txt --n 6 --note C2 --dir variations

  # vary only the "character" knobs, lock pitch/filter base
  python vary.py --genome p.txt --lock cutoff,osc2_detune --amount 0.25
"""

import argparse
import os
import numpy as np
import soundfile as sf
import torch

from synth import render, to_wav, N_PARAMS, PARAM_NAMES, SR
from match import note_to_midi


def load_genome(spec, dev):
    if os.path.exists(spec):
        g = np.loadtxt(spec).astype(np.float32)
    else:
        g = np.array([float(x) for x in spec.split(",")], dtype=np.float32)
    return torch.tensor(g, device=dev)


def mutate(g, amount, lock_idx, gen):
    noise = torch.randn(g.shape, device=g.device, generator=gen) * amount
    if lock_idx:
        noise[lock_idx] = 0.0
    return (g + noise).clamp(0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--genome")
    ap.add_argument("--breed", nargs=2)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--amount", type=float, default=0.15, help="mutation strength")
    ap.add_argument("--note", default="C2")
    ap.add_argument("--lock", default="", help="comma param names to freeze")
    ap.add_argument("--dir", default="variations")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gen = torch.Generator(device=dev).manual_seed(a.seed)
    note = note_to_midi(a.note)
    os.makedirs(a.dir, exist_ok=True)
    lock_idx = [PARAM_NAMES.index(x.strip()) for x in a.lock.split(",") if x.strip()]

    if a.breed:
        pa, pb = load_genome(a.breed[0], dev), load_genome(a.breed[1], dev)
        sf.write(os.path.join(a.dir, "parent_A.wav"), to_wav(render(pa, note)), SR)
        sf.write(os.path.join(a.dir, "parent_B.wav"), to_wav(render(pb, note)), SR)
        for i in range(a.n):
            mask = (torch.rand(N_PARAMS, device=dev, generator=gen) < 0.5).float()
            child = mask * pa + (1 - mask) * pb          # uniform crossover
            child = mutate(child, a.amount * 0.5, lock_idx, gen)  # light mutation
            sf.write(os.path.join(a.dir, f"child_{i:02d}.wav"),
                     to_wav(render(child, note)), SR)
        print(f"bred {a.n} children (+2 parents) at {a.note} -> {a.dir}/")
        return

    if not a.genome:
        ap.error("need --genome or --breed")
    base = load_genome(a.genome, dev)
    sf.write(os.path.join(a.dir, "original.wav"), to_wav(render(base, note)), SR)
    for i in range(a.n):
        v = mutate(base, a.amount, lock_idx, gen)
        sf.write(os.path.join(a.dir, f"var_{i:02d}.wav"), to_wav(render(v, note)), SR)
    locked = f" (locked: {a.lock})" if a.lock else ""
    print(f"spawned {a.n} sound variations of {a.genome} at {a.note}"
          f"  amount={a.amount}{locked} -> {a.dir}/")


if __name__ == "__main__":
    import os as _os
    main()
    _os._exit(0)   # dodge ROCm-on-Windows teardown deadlock
