"""Which Serum knobs can you actually hear, on which presets? CPU only.

Picks the knob set for serum_preset_set.py: a knob is only worth labelling if
moving it changes the sound more than Serum's own render-to-render noise.
For every preset x knob: render the preset untouched (twice -> noise floor),
then with the knob forced to 0 / 0.5 / 1. Audibility = max spectral change /
floor. Run twice per preset: as-is, and "derand" with Serum's global
randomizers (Env/Cutoff/Detune Rand, random phase, unison random start) zeroed.

    .venv-serum/Scripts/python.exe lab/serum_knob_probe.py --jobs 3

Writes real/serum_presets/knob_audibility.json:
  {mode: {knob: {preset index: ratio}}}
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "real/serum_presets"
KNOBS = [
    # current label set
    "A WT Pos", "Filter 1 Freq", "Filter 1 Res", "Env 1 Attack", "Env 1 Decay", "Env 1 Sustain", "Env 1 Release",
    # candidates
    "B WT Pos", "A Level", "B Level", "Sub Level", "Noise Level", "A Uni Detune", "A Warp",
    "Filter 1 Drive", "Filter 1 Wet", "Env 2 Decay", "LFO 1 Rate", "Macro 1", "Macro 2", "Macro 3", "Macro 4",
]
RANDOMIZERS = ["Env Rand", "Cutoff Rand", "Osc Detune Rnd", "Osc Pan Rand",
               "A Rand Phase", "B Rand Phase", "C Rand Phase", "Noise Rand Phase",
               "A Uni Rand Start", "B Uni Rand Start", "C Uni Rand Start"]
SWEEP = [0.0, 0.5, 1.0]


def worker(job_path):
    import dawdreamer as daw  # must load before numpy
    import numpy as np
    from serum2_preset_loader import convert_preset_file
    from serum_preset_set import NOTE_DUR, SR, TAIL, VST, hard_exit

    job = json.loads(Path(job_path).read_text(encoding="utf-8"))
    eng = daw.RenderEngine(SR, 512)
    syn = eng.make_plugin_processor("serum", VST)
    eng.load_graph([(syn, [])])
    syn.add_midi_note(48, 127, 0.0, 0.05)
    eng.render(0.1)
    idx = {d["name"]: d["index"] for d in syn.get_plugin_parameters_description()}
    rands = [idx[n] for n in RANDOMIZERS if n in idx]
    state = Path(tempfile.mkdtemp(prefix="knobprobe_")) / "state.bin"

    def logspec(x):
        frames = np.lib.stride_tricks.sliding_window_view(x, 1024)[::512] * np.hanning(1024)
        db = 10 * np.log10(np.abs(np.fft.rfft(frames, axis=1)) ** 2 + 1e-6)
        return np.clip((db - db.max() + 80) / 80, 0, 1)

    def render(blob, derand, knob=None, value=None):
        state.write_bytes(blob)
        syn.load_state(str(state))
        if derand:
            for i in rands:
                syn.set_parameter(i, 0.0)
        if knob:
            syn.set_parameter(idx[knob], value)
        syn.clear_midi()
        syn.add_midi_note(48, 100, 0.0, NOTE_DUR)
        eng.render(NOTE_DUR + TAIL)
        return logspec(eng.get_audio().mean(axis=0))

    out = {"raw": {k: {} for k in KNOBS}, "derand": {k: {} for k in KNOBS}, "floor": {"raw": {}, "derand": {}}}
    for i in job["presets"]:
        try:
            blob = convert_preset_file(job["paths"][i])
        except Exception as exc:
            print(f"{i} load failed: {exc}", flush=True)
            continue
        for mode in ("raw", "derand"):
            d = mode == "derand"
            base = render(blob, d)
            floor = float(np.abs(render(blob, d) - base).mean())
            out["floor"][mode][i] = floor
            for k in KNOBS:
                if k not in idx:
                    continue
                effect = max(float(np.abs(render(blob, d, k, v) - base).mean()) for v in SWEEP)
                out[mode][k][i] = round(effect / max(floor, 1e-3), 2)
        print(f"preset {i} done", flush=True)
    Path(job["out"]).write_text(json.dumps(out), encoding="utf-8")
    hard_exit(0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--min", type=float, default=2.0, help="audible threshold (x noise floor)")
    ap.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.worker:
        worker(a.worker)
        return
    meta = json.loads((DATA / "meta.json").read_text(encoding="utf-8"))
    n = len(meta["presets"])
    tmp = Path(tempfile.mkdtemp(prefix="knobprobe_jobs_"))
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0)
    procs = []
    for j in range(a.jobs):
        jp = tmp / f"job{j}.json"
        jp.write_text(json.dumps({"presets": list(range(j, n, a.jobs)), "paths": meta["presets"],
                                  "out": str(tmp / f"out{j}.json")}), encoding="utf-8")
        log = open(tmp / f"job{j}.log", "w", encoding="utf-8")
        procs.append((subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()), "--worker", str(jp)],
                                       stdout=log, stderr=subprocess.STDOUT, cwd=ROOT / "lab", creationflags=flags), log))
    for p, log in procs:
        p.wait(); log.close()
    merged = {"raw": {k: {} for k in KNOBS}, "derand": {k: {} for k in KNOBS}, "floor": {"raw": {}, "derand": {}}}
    for j in range(a.jobs):
        part = tmp / f"out{j}.json"
        if not part.exists():
            sys.exit(f"worker {j} produced nothing; see {tmp / f'job{j}.log'}")
        d = json.loads(part.read_text(encoding="utf-8"))
        for mode in merged:
            for k, v in d[mode].items():
                merged[mode][k].update(v)
    (DATA / "knob_audibility.json").write_text(json.dumps(merged, indent=1), encoding="utf-8")

    import statistics as st
    for mode in ("raw", "derand"):
        fl = list(merged["floor"][mode].values())
        print(f"\n[{mode}] noise floor median {st.median(fl):.4f}; presets with floor > 0.02: "
              f"{sum(f > 0.02 for f in fl)}/{len(fl)}")
    print(f"\n{'knob':16s} {'raw: audible':>13s} {'derand: audible':>16s}  (presets >= x{a.min}, of {n})")
    for k in sorted(KNOBS, key=lambda k: -sum(v >= a.min for v in merged["derand"][k].values())):
        r = sum(v >= a.min for v in merged["raw"][k].values())
        d = sum(v >= a.min for v in merged["derand"][k].values())
        print(f"{k:16s} {r:13d} {d:16d}")


if __name__ == "__main__":
    main()
