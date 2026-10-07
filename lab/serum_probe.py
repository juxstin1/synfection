"""Feasibility probe for HANDOFF.md section 10.2: can Serum 2 be driven headless?
Run: lab/.venv/Scripts/python.exe lab/serum_probe.py
"""
import sys, time
import numpy as np
from pedalboard import load_plugin
from pedalboard.io import AudioFile

VST = "C:/Program Files/Common Files/VST3/Serum2.vst3/Contents/x86_64-win/Serum2.vst3"
SR = 22050

t0 = time.time()
try:
    p = load_plugin(VST, plugin_name="Serum 2")
except Exception as e:
    print("1. LOAD FAILED:", repr(e)); sys.exit(1)
print(f"1. loaded in {time.time()-t0:.1f}s  instrument={p.is_instrument}")

params = p.parameters
print(f"   {len(params)} parameters")
for k in list(params)[:40]:
    print("  ", k, "=", params[k].raw_value)


def render(note=48, dur=0.85, tail=0.35):
    msgs = [(bytes([0x90, note, 100]), 0.0), (bytes([0x80, note, 0]), dur)]
    return p(msgs, duration=dur + tail, sample_rate=SR, num_channels=2, reset=True)

a = render()
print("2. render shape", a.shape, "peak", float(np.abs(a).max()))

import re
names = list(params)
for pat in [r"^a_position", r"filter.*(cutoff|freq)", r"filter.*(res|reso)", r"^env_?[1-4]_(attack|decay|sustain|release|hold)", r"^env.*(attack|decay|sustain|release)_"]:
    hits = [n for n in names if re.search(pat, n)]
    print(f"   /{pat}/ -> {hits[:12]}")

# 3. set + readback + audio actually changes
k = "a_position"
params[k].raw_value = 0.0; a0 = render()
params[k].raw_value = 0.6; a1 = render()
print("3. readback", params[k].raw_value, " audio differs:", float(np.abs(a0 - a1).mean()) > 1e-4)

# 4. throughput
t0 = time.time(); n = 30
for _ in range(n): render()
dt = time.time() - t0
print(f"4. {n} renders in {dt:.1f}s  -> {n/dt*60:.0f} clips/min")
