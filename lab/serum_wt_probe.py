"""Can you hear A WT Pos on each preset? CPU-only (Serum render + numpy).

train_serum.py can't predict A WT Pos (-5% vs baseline). Suspect: on many
presets the knob does nothing (osc A off or quiet, single-frame wavetable),
so its label is noise. For each preset in meta.json this renders the training
setup with only A WT Pos swept, and compares the spread against a repeat
render at the same setting (Serum's own render-to-render noise).

    .venv-serum/Scripts/python.exe lab/serum_wt_probe.py

Writes real/serum_presets/wt_audibility.json: {preset index: effect / floor}.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile

import dawdreamer as daw  # must load before numpy
import numpy as np
from serum2_preset_loader import convert_preset_file

from serum_preset_set import NOTE_DUR, PARAMS, ROOT, SR, TAIL, VST, hard_exit

# The other six knobs held at mid-window, filter fairly open so WT changes pass through.
FIXED = {"Filter 1 Freq": 0.8, "Filter 1 Res": 0.1, "Env 1 Attack": 0.0,
         "Env 1 Decay": 0.5, "Env 1 Sustain": 0.8, "Env 1 Release": 0.2}
POSITIONS = [0.0, 0.25, 0.5, 0.75, 1.0]


def logspec(x):
    frames = np.lib.stride_tricks.sliding_window_view(x, 1024)[::512] * np.hanning(1024)
    db = 10 * np.log10(np.abs(np.fft.rfft(frames, axis=1)) ** 2 + 1e-6)
    return np.clip((db - db.max() + 80) / 80, 0, 1)


def main():
    data = ROOT / "real/serum_presets"
    meta = json.loads((data / "meta.json").read_text(encoding="utf-8"))
    eng = daw.RenderEngine(SR, 512)
    syn = eng.make_plugin_processor("serum", VST)
    eng.load_graph([(syn, [])])
    syn.add_midi_note(48, 127, 0.0, 0.05)
    eng.render(0.1)
    idx = {d["name"]: d["index"] for d in syn.get_plugin_parameters_description()}
    assert [p[0] for p in PARAMS] == meta["params"]
    state = Path(tempfile.mkdtemp(prefix="wtprobe_")) / "state.bin"

    def render(blob, wt):
        state.write_bytes(blob)
        syn.load_state(str(state))
        syn.set_parameter(idx["Filter 1 On"], 1.0)
        for name, v in FIXED.items():
            syn.set_parameter(idx[name], v)
        syn.set_parameter(idx["A WT Pos"], wt)
        syn.clear_midi()
        syn.add_midi_note(48, 100, 0.0, NOTE_DUR)
        eng.render(NOTE_DUR + TAIL)
        return logspec(eng.get_audio().mean(axis=0))

    out, rows = {}, []
    for i, path in enumerate(meta["presets"]):
        try:
            blob = convert_preset_file(path)
        except Exception as exc:
            print(f"{i:3d} load failed: {exc}", flush=True)
            continue
        specs = [render(blob, p) for p in POSITIONS]
        floor = np.abs(render(blob, 0.0) - specs[0]).mean()
        effect = max(np.abs(s - specs[0]).mean() for s in specs[1:])
        a_on, a_lvl = syn.get_parameter(idx["A Enable"]), syn.get_parameter(idx["A Level"])
        ratio = effect / max(floor, 1e-4)
        out[i] = round(float(ratio), 2)
        val = "val" if i in meta["val_presets"] else "   "
        rows.append((ratio, f"{i:3d} {val} A on={a_on:.0f} lvl={a_lvl:.2f}  effect {effect:.4f}  floor {floor:.4f}  x{ratio:7.1f}  {Path(path).stem}"))
        print(rows[-1][1], flush=True)
    (data / "wt_audibility.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("\nsorted by audibility (effect / repeat-render floor):")
    for _, line in sorted(rows):
        print(line)
    sys.stdout.flush()
    hard_exit(0)


if __name__ == "__main__":
    main()
