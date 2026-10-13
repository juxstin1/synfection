"""Hear what serum_net.pt learned: render its knob guesses back through Serum.

For clips from held-out presets (never seen in training) this writes three
renders of the same preset + note, each peak-normalized like the training set:

  target.wav  the real Serum clip from the val shard (true knob values)
  net.wav     same preset, the 7 knobs set to the net's prediction
  mean.wav    same preset, the 7 knobs set to the train-set mean (the baseline)

If net.wav isn't audibly closer to target.wav than mean.wav is, the net
hasn't learned anything you can hear, whatever the MAE says.

    python lab/serum_listen.py --per-preset 3

Prediction runs here (needs torch); rendering runs in .venv-serum (DawDreamer).
"""
from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
VENV = ROOT / ".venv-serum/Scripts/python.exe"


def predict(a):
    import numpy as np
    import soundfile as sf
    import torch
    sys.path.insert(0, str(ROOT / "lab"))
    from model import GenoNet, device
    from synth import melspec

    meta = json.loads((a.data / "meta.json").read_text(encoding="utf-8"))
    names, val = meta["params"], set(meta["val_presets"])
    lo = np.array([w[0] for w in meta["windows"]], np.float32)
    hi = np.array([w[1] for w in meta["windows"]], np.float32)
    shards = sorted(p for p in a.data.glob("shard_*.npz") if ".tmp" not in p.name)
    # Baseline = train-set mean of the raw knob values, same as train_serum.py's bar.
    sums, count, clips = np.zeros(len(names)), 0, []
    for p in shards:
        d = np.load(p)
        pre = d["preset"]
        tr = ~np.isin(pre, list(val))
        sums += d["params"][tr].sum(0); count += int(tr.sum())
        if (~tr).any():
            clips.append((d["audio"], d["note"], d["params"], pre))
    mean_raw = (sums / count).astype(np.float32)
    audio = np.concatenate([c[0] for c in clips]); note = np.concatenate([c[1] for c in clips])
    params = np.concatenate([c[2] for c in clips]); pre = np.concatenate([c[3] for c in clips])

    rng = np.random.default_rng(a.seed)
    pick = []
    for pid in sorted(val):
        idx = np.flatnonzero(pre == pid)
        pick += sorted(rng.choice(idx, min(a.per_preset, len(idx)), replace=False).tolist())

    dev = device()
    net = GenoNet(n_params=len(names)).to(dev)
    net.load_state_dict(torch.load(a.net, map_location=dev))
    net.eval()
    with torch.no_grad():
        x = torch.from_numpy(audio[pick].astype(np.float32) / 32767).to(dev)
        y = net(melspec(x)).cpu().numpy()
    net_raw = lo + y * (hi - lo)

    a.out.mkdir(parents=True, exist_ok=True)
    jobs = []
    for k, i in enumerate(pick):
        folder = a.out / f"{k:03d}"
        folder.mkdir(exist_ok=True)
        sf.write(folder / "target.wav", audio[i], meta["sr"], subtype="PCM_16")
        jobs.append({"dir": str(folder), "preset": meta["presets"][int(pre[i])], "note": int(note[i]),
                     "knobs": names, "true": params[i].tolist(), "net": net_raw[k].tolist(),
                     "mean": mean_raw.tolist(),
                     # v1 data was rendered with Serum's randomizers on; match whatever the shards used.
                     "derand": meta.get("derandomized", False)})
    (a.out / "jobs.json").write_text(json.dumps(jobs, indent=2), encoding="utf-8")
    print(f"predicted {len(jobs)} clips from {len(val)} held-out presets on {dev}", flush=True)
    return jobs, meta


def worker(jobs_path):
    import dawdreamer as daw  # must load before numpy
    import numpy as np
    import soundfile as sf
    import tempfile
    from serum_preset_set import NOTE_DUR, TAIL, N, SR, VST, derandomize, hard_exit
    from serum2_preset_loader import convert_preset_file

    jobs = json.loads(Path(jobs_path).read_text(encoding="utf-8"))
    eng = daw.RenderEngine(SR, 512)
    syn = eng.make_plugin_processor("serum", VST)
    eng.load_graph([(syn, [])])
    syn.add_midi_note(48, 127, 0.0, 0.05)
    eng.render(0.1)  # Serum 2 cold-render warmup
    idx = {d["name"]: d["index"] for d in syn.get_plugin_parameters_description()}
    state = Path(tempfile.mkdtemp(prefix="serumlisten_")) / "state.bin"
    blobs = {}
    for job in jobs:
        if job["preset"] not in blobs:
            blobs[job["preset"]] = convert_preset_file(job["preset"])
        for kind in ("net", "mean"):
            # Reload the preset each time so nothing carries over between renders.
            state.write_bytes(blobs[job["preset"]])
            syn.load_state(str(state))
            if job["derand"]:
                derandomize(syn, idx)
            syn.set_parameter(idx["Filter 1 On"], 1.0)
            for name, v in zip(job["knobs"], job[kind]):
                syn.set_parameter(idx[name], float(v))
            syn.clear_midi()
            syn.add_midi_note(job["note"], 100, 0.0, NOTE_DUR)
            eng.render(NOTE_DUR + TAIL)
            x = eng.get_audio().mean(axis=0).astype(np.float32)
            x = np.pad(x, (0, max(0, N - len(x))))[:N]
            peak = float(np.abs(x).max()) if np.isfinite(x).all() else 0.0
            if peak >= 1e-4:
                x = x / peak * 0.9
            sf.write(Path(job["dir"]) / f"{kind}.wav", x, SR, subtype="PCM_16")
        print(f"rendered {Path(job['dir']).name} {Path(job['preset']).stem}", flush=True)
    (Path(jobs_path).parent / "rendered.json").write_text("{}", encoding="utf-8")
    hard_exit(0)


def gallery(out, jobs):
    cards = []
    for job in jobs:
        rel = Path(job["dir"]).name
        rows = "".join(
            f"<tr><td>{html.escape(n)}</td><td>{t:.2f}</td><td>{p:.2f}</td><td>{m:.2f}</td>"
            f"<td class={'win' if abs(p - t) < abs(m - t) else 'lose'}>{abs(p - t) - abs(m - t):+.2f}</td></tr>"
            for n, t, p, m in zip(job["knobs"], job["true"], job["net"], job["mean"]))
        cards.append(f"""<article><p class="tag">MIDI {job['note']}</p><h2>{html.escape(Path(job['preset']).stem)}</h2>
<label>Target: real Serum, true knobs<audio controls preload="none" src="{rel}/target.wav"></audio></label>
<label>Net guess<audio controls preload="none" src="{rel}/net.wav"></audio></label>
<label>Baseline: train-mean knobs<audio controls preload="none" src="{rel}/mean.wav"></audio></label>
<details><summary>Knob values</summary><table><tr><th>knob</th><th>true</th><th>net</th><th>mean</th><th>net vs mean error</th></tr>{rows}</table></details></article>""")
    page = """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Serum net listening</title><style>
body{background:#111714;color:#e7f1e9;font:16px system-ui;margin:0;padding:16px 5vw;max-width:1200px}p{color:#b4c8bb;line-height:1.5}.tag{color:#a9d787;font-size:13px;text-transform:uppercase}
main{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:20px}article{border:1px solid #405548;padding:20px;border-radius:12px}h2{font-size:19px;word-break:break-word}
audio{display:block;width:100%;margin:6px 0 16px}table{font-size:13px;border-collapse:collapse;width:100%}td,th{padding:3px 6px;text-align:right}td:first-child,th:first-child{text-align:left}.win{color:#9be07a}.lose{color:#f08a7a}
</style><h1>Can you hear what the net learned?</h1>
<p>Each card uses a preset the net never trained on. Compare <b>Net guess</b> with <b>Baseline</b>: whichever sounds closer to <b>Target</b> wins. In the last column, negative (green) means the net was closer than the baseline on that knob. Start with low volume.</p><main>""" + "".join(cards) + """</main>
<script>document.addEventListener('play',e=>{document.querySelectorAll('audio').forEach(a=>{if(a!==e.target)a.pause()})},true)</script></html>"""
    (out / "index.html").write_text(page, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=ROOT / "real/serum_presets")
    ap.add_argument("--net", type=Path, default=ROOT / "lab/serum_net.pt")
    ap.add_argument("--out", type=Path, default=ROOT / "lab/serum-listen")
    ap.add_argument("--per-preset", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.worker:
        worker(a.worker)
        return 0
    jobs, _ = predict(a)
    jobs_path = a.out / "jobs.json"
    (a.out / "rendered.json").unlink(missing_ok=True)
    with (a.out / "render.log").open("w", encoding="utf-8") as log:
        r = subprocess.run([str(VENV), "-u", str(Path(__file__).resolve()), "--worker", str(jobs_path)],
                           stdout=log, stderr=subprocess.STDOUT, cwd=ROOT / "lab", timeout=a.timeout,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    # Serum's teardown crash gives a nonzero exit; the marker file is the real success signal.
    if not (a.out / "rendered.json").exists():
        sys.exit(f"render failed (exit {r.returncode}); see {a.out / 'render.log'}")
    gallery(a.out, jobs)
    print(f"gallery: {a.out / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0)  # dodge ROCm-on-Windows teardown deadlock
