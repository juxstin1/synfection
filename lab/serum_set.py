"""Serum 2 -> labelled training set (HANDOFF.md section 10.3).
Random Serum params in, 1.2 s 22.05k mono clip out, labels = the raw 0..1 param values.

  lab/.venv/Scripts/python.exe lab/serum_set.py --n 200 --out real/serum_probe.npz
  lab/.venv/Scripts/python.exe lab/serum_set.py --check real/serum_probe.npz   # re-render + compare
"""
import argparse, io, contextlib, time
import numpy as np
from pedalboard import load_plugin

VST = "C:/Program Files/Common Files/VST3/Serum2.vst3/Contents/x86_64-win/Serum2.vst3"
SR, N, NOTE_DUR, TAIL = 22050, 26460, 0.85, 0.35
# (name, lo, hi) windows on the raw 0..1 value
PARAMS = [
    ("a_wt_pos", 0.0, 1.0),
    ("filter_1_freq_hz", 0.15, 0.95),
    ("filter_1_res", 0.0, 0.7),
    ("env_1_attack", 0.0, 0.45),
    ("env_1_decay", 0.0, 0.8),
    ("env_1_sustain", 0.0, 1.0),
    ("env_1_release", 0.0, 0.6),
]
NAMES = [p[0] for p in PARAMS]


def open_plugin():
    with contextlib.redirect_stdout(io.StringIO()):
        p = load_plugin(VST, plugin_name="Serum 2")
    p.parameters["filter_1_on"].raw_value = 1.0
    return p


def render(p, note, vals):
    for n, v in zip(NAMES, vals):
        p.parameters[n].raw_value = float(v)
    msgs = [(bytes([0x90, int(note), 100]), 0.0), (bytes([0x80, int(note), 0]), NOTE_DUR)]
    a = p(msgs, duration=NOTE_DUR + TAIL, sample_rate=SR, num_channels=2, reset=True)
    x = a.mean(axis=0).astype(np.float32)
    x = np.pad(x, (0, max(0, N - len(x))))[:N]
    peak = float(np.abs(x).max())
    return None if peak < 1e-4 else x / peak * 0.9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", default="real/serum_probe.npz")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--check", default=None)
    a = ap.parse_args()
    p = open_plugin()
    if a.check:
        d = np.load(a.check)
        errs = []
        for i in range(min(20, len(d["note"]))):
            x = render(p, d["note"][i], d["params"][i])
            errs.append(float(np.abs(x - d["audio"][i]).max()) if x is not None else 9.9)
        print(f"re-render max abs diff over {len(errs)} clips: worst {max(errs):.2e}")
        return
    rng = np.random.default_rng(a.seed)
    lo = np.array([w[1] for w in PARAMS]); hi = np.array([w[2] for w in PARAMS])
    audio, notes, vals = [], [], []
    t0, dropped = time.time(), 0
    while len(audio) < a.n:
        v = lo + (hi - lo) * rng.random(len(PARAMS))
        note = int(rng.integers(36, 61))
        x = render(p, note, v)
        if x is None:
            dropped += 1
            continue
        audio.append(x); notes.append(note); vals.append(v)
    import os
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    np.savez(a.out, audio=np.stack(audio), note=np.array(notes), params=np.stack(vals).astype(np.float32),
             names=np.array(NAMES))
    dt = time.time() - t0
    print(f"{len(audio)} clips ({dropped} silent dropped) in {dt:.0f}s -> {a.out}")


if __name__ == "__main__":
    main()
