"""Serum 2 -> labelled training set built on the user's own presets (HANDOFF.md 10.3).

Each clip: load one of your .SerumPreset files as the base patch, randomize the
7 labelled knobs (same names/windows as serum_set.py), render 1.2 s at 22.05k
mono. Labels are exact (readback verified), the timbre is your preset's.

Validation is split BY PRESET (stable hash), so val measures generalization to
presets the net never heard, not memorized timbres.

  .venv-serum/Scripts/python.exe lab/serum_preset_set.py --n 50000 --jobs 8

Shards land in real/serum_presets/shard_NNNN.npz; reruns skip finished shards.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
VST = "C:/Program Files/Common Files/VST3/Serum2.vst3"
SR, N, NOTE_DUR, TAIL = 22050, 26460, 0.85, 0.35
NOTE_LO, NOTE_HI = 36, 60
# (Serum display name, lo, hi) windows on the raw 0..1 value — mirrors serum_set.py
PARAMS = [
    ("A WT Pos", 0.0, 1.0),
    ("Filter 1 Freq", 0.15, 0.95),
    ("Filter 1 Res", 0.0, 0.7),
    ("Env 1 Attack", 0.0, 0.45),
    ("Env 1 Decay", 0.0, 0.8),
    ("Env 1 Sustain", 0.0, 1.0),
    ("Env 1 Release", 0.0, 0.6),
]
VAL_PCT = 15


def preset_dirs():
    home = Path.home()
    base = home / "Documents/Xfer/Serum 2 Presets/Presets"
    return [base / "User", base / "Splice", home / "Downloads"]


def gather(dirs):
    seen, out = set(), []
    for d in dirs:
        if d.is_dir():
            for p in sorted(d.rglob("*.SerumPreset")):
                if p.name not in seen:
                    seen.add(p.name)
                    out.append(p)
    return out


def is_val(path):
    return int(hashlib.sha256(Path(path).name.encode()).hexdigest(), 16) % 100 < VAL_PCT


def hard_exit(code=0):
    # Serum 2.1.5 + DawDreamer 0.9.0 access-violates during teardown; skip it.
    sys.stdout.flush()
    sys.stderr.flush()
    if os.name == "nt":
        import ctypes
        k = ctypes.windll.kernel32
        k.TerminateProcess(k.GetCurrentProcess(), code)
    os._exit(code)


def worker(job_path):
    import dawdreamer as daw  # must load before numpy
    import numpy as np
    import tempfile
    from serum2_preset_loader import convert_preset_file

    job = json.loads(Path(job_path).read_text(encoding="utf-8"))
    presets = job["presets"]
    eng = daw.RenderEngine(SR, 512)
    syn = eng.make_plugin_processor("serum", VST)
    eng.load_graph([(syn, [])])
    syn.add_midi_note(48, 127, 0.0, 0.05)
    eng.render(0.1)  # Serum 2 cold-render warmup
    idx = {d["name"]: d["index"] for d in syn.get_plugin_parameters_description()}
    knobs = [idx[name] for name, *_ in PARAMS]
    filt_on = idx["Filter 1 On"]
    lo = np.array([w[1] for w in PARAMS]); hi = np.array([w[2] for w in PARAMS])
    state = Path(tempfile.mkdtemp(prefix="serumset_")) / "state.bin"
    blobs = {}
    rng = np.random.default_rng(job["seed"])
    audio, notes, vals, pids = [], [], [], []
    dropped, broken = 0, set()
    while len(audio) < job["n"]:
        pid = int(rng.choice(job["pool"]))
        if pid in broken:
            continue
        try:
            if pid not in blobs:
                blobs[pid] = convert_preset_file(presets[pid])
            state.write_bytes(blobs[pid])
            syn.load_state(str(state))
        except Exception as exc:
            print(f"preset {pid} failed to load: {exc}", flush=True)
            broken.add(pid)
            if broken >= set(job["pool"]):
                raise RuntimeError("every preset failed to load")
            continue
        v = lo + (hi - lo) * rng.random(len(PARAMS))
        syn.set_parameter(filt_on, 1.0)
        for i, x in zip(knobs, v):
            syn.set_parameter(i, float(x))
        note = int(rng.integers(NOTE_LO, NOTE_HI + 1))
        syn.clear_midi()
        syn.add_midi_note(note, 100, 0.0, NOTE_DUR)
        eng.render(NOTE_DUR + TAIL)
        x = eng.get_audio().mean(axis=0).astype(np.float32)
        x = np.pad(x, (0, max(0, N - len(x))))[:N]
        peak = float(np.abs(x).max()) if np.isfinite(x).all() else 0.0
        if peak < 1e-4:
            dropped += 1
            continue
        audio.append(np.round(x / peak * 0.9 * 32767).astype(np.int16))
        notes.append(note); vals.append(v); pids.append(pid)
    out = Path(job["out"])
    tmp = out.with_name(out.stem + ".tmp.npz")
    np.savez(tmp, audio=np.stack(audio), note=np.array(notes, np.int16),
             params=np.stack(vals).astype(np.float32), preset=np.array(pids, np.int16))
    tmp.replace(out)  # the shard file is the completion marker
    print(f"wrote {out.name}: {len(audio)} clips, {dropped} silent dropped, "
          f"{len(broken)} presets unloadable", flush=True)
    hard_exit(0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=50000, help="total clips (train + val)")
    ap.add_argument("--shard", type=int, default=1000)
    ap.add_argument("--jobs", type=int, default=8, help="parallel Serum processes")
    ap.add_argument("--out", type=Path, default=ROOT / "real/serum_presets")
    ap.add_argument("--timeout", type=int, default=1800, help="seconds per shard")
    ap.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.worker:
        worker(a.worker)
        return 0

    presets = gather(preset_dirs())
    if not presets:
        ap.error("no .SerumPreset files found")
    val = [i for i, p in enumerate(presets) if is_val(p)]
    train = [i for i, p in enumerate(presets) if not is_val(p)]
    if not val or not train:
        ap.error("need presets on both sides of the train/val split")
    a.out.mkdir(parents=True, exist_ok=True)
    meta = {"params": [p[0] for p in PARAMS], "windows": [p[1:] for p in PARAMS],
            "sr": SR, "n": N, "note_range": [NOTE_LO, NOTE_HI],
            "presets": [str(p) for p in presets], "val_presets": val}
    (a.out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"{len(presets)} presets: {len(train)} train / {len(val)} val (held out)", flush=True)

    n_shards = -(-a.n // a.shard)
    n_val = max(1, round(n_shards * VAL_PCT / 100))
    pending = []
    for s in range(n_shards):
        out = a.out / f"shard_{s:04d}.npz"
        if out.exists():
            continue
        job = {"n": a.shard, "seed": 1000 + s, "out": str(out), "presets": meta["presets"],
               "pool": val if s < n_val else train}
        jp = a.out / f"shard_{s:04d}.job.json"
        jp.write_text(json.dumps(job), encoding="utf-8")
        pending.append((s, jp, out))
    print(f"{n_shards} shards ({n_val} val), {len(pending)} to render, {a.jobs} at a time", flush=True)

    t0, running, failed = time.time(), [], []
    # Below-normal priority: renders yield the CPU to whatever else is running.
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0)
    while pending or running:
        while pending and len(running) < a.jobs:
            s, jp, out = pending.pop(0)
            log = open(jp.with_suffix(".log"), "w", encoding="utf-8")
            proc = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()), "--worker", str(jp)],
                                    stdout=log, stderr=subprocess.STDOUT, cwd=ROOT, creationflags=flags)
            running.append((s, proc, out, log, time.time()))
        time.sleep(1)
        for item in list(running):
            s, proc, out, log, start = item
            if proc.poll() is None and time.time() - start < a.timeout:
                continue
            if proc.poll() is None:
                proc.kill()
            log.close()
            running.remove(item)
            done = len([p for p in a.out.glob("shard_*.npz") if ".tmp" not in p.name])
            if out.exists():
                print(f"shard {s:04d} ok  ({done}/{n_shards}, {time.time() - t0:.0f}s)", flush=True)
            else:
                failed.append(s)
                print(f"shard {s:04d} FAILED (exit {proc.returncode}); see {log.name}", flush=True)
    print(f"done in {time.time() - t0:.0f}s; {len(failed)} failed shards {failed}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
