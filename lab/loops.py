"""
loops.py — turn a patch (genome) into seamless, tempo-locked bass LOOPS at 44.1k.

The engine plays one note; a sellable loop is a riff in a key at a BPM that loops
clean. This adds the sequencer on top of synth.render: a mono bassline pattern ->
overlap-add (with wrap-around tails so the loop is seamless) -> normalized 44.1k wav.

Mono is right for bass (you fold sub to mono anyway), so no polyphony needed.

  # one loop from a matched/curated patch
  python loops.py --genome remake_bass.wav.genome.txt --key F1 --bpm 138 --pattern garage_roll

  # SPAM a pack: every sound x pattern x key, organized + named
  python loops.py --pack --genomes remake_bass.wav.genome.txt --keys F1,G1,A1,C2 \\
                  --bpms 130,138 --patterns garage_roll,reese_hold,house_offbeat --dir packs/demo
"""

import argparse
import os
import numpy as np
import soundfile as sf
import torch

from synth import render, upgrade_genome
from match import note_to_midi

SR_OUT = 44100

# A pattern is 16 sixteenth-steps over one bar. Each slot is None (rest) or
# (semitone_offset_from_root, gate_in_steps[, gain]). Mono: tails wrap for seamless.
PATTERNS = {
    # classic UK garage skippy bass — offbeat stabs with a little movement
    "garage_roll": [(0, 1), None, None, (0, 1),  None, (12, 1), None, (0, 2),
                    None, None, (0, 1), None,     (10, 1), None, (0, 1), None],
    # 2-step / house offbeat — bass on the "&"s, fat and simple
    "house_offbeat": [None, None, (0, 2), None,   None, None, (0, 2), None,
                      None, None, (0, 2), None,    None, None, (0, 2), None],
    # reese / dub hold — long sustained notes, two per bar, room to growl
    "reese_hold": [(0, 8), None, None, None,       None, None, None, None,
                   (0, 7), None, None, None,        None, None, None, (3, 1)],
    # rolling bassline with octave + fifth movement (speed garage flavour)
    "speed_walk": [(0, 1), None, (0, 1), (12, 1),  None, (0, 1), (7, 1), None,
                   (0, 1), None, (0, 1), (10, 1),  None, (0, 1), (5, 1), (7, 1)],
    # 4-to-floor root pulse — every beat, tight gate (bass house)
    "four_pulse": [(0, 2), None, None, None,       (0, 2), None, None, None,
                   (0, 2), None, None, None,        (0, 2), None, None, None],
    # relentless speed-garage roller — near-constant 8ths, octave/fifth flicks
    "speed_run": [(0, 1), None, (0, 1), None,      (12, 1), None, (0, 1), (7, 1),
                  (0, 1), None, (10, 1), None,      (12, 1), (0, 1), (7, 1), (5, 1)],
    # 2-step skip with octave pops on the off-16ths
    "garage_bounce": [(0, 2), None, None, (12, 1, 0.7),  None, None, (0, 1), None,
                      (0, 2), None, (10, 1, 0.6), None,   (12, 1), None, None, (0, 1)],
    # niche / bassline octave seesaw — root vs ghosted octave every 16th
    "bassline_seesaw": [(0, 1), (12, 1, 0.55), (0, 1), (12, 1, 0.55),
                        (0, 1), (12, 1, 0.55), (3, 1), (12, 1, 0.55),
                        (0, 1), (12, 1, 0.55), (0, 1), (12, 1, 0.55),
                        (5, 1), (12, 1, 0.55), (7, 1), (10, 1, 0.6)],
    # speed-garage organ bass — offbeat octave hops with fifth/b7 turns
    "organ_hop": [None, (12, 1), None, (0, 1),     None, (12, 1, 0.8), None, (7, 1),
                  None, (12, 1), None, (0, 1),      (5, 1), None, (7, 1), (10, 1, 0.8)],
    # classic UKG skip with ghost notes for shuffle feel
    "skippy_ghost": [(0, 2), None, (0, 1, 0.45), None,   None, (0, 1), None, (0, 1, 0.5),
                     None, (0, 2), None, (0, 1, 0.45),    (10, 1, 0.7), None, (0, 1), None],
    # half-bar growl holds with a minor walk-up turnaround
    "wobble_hold": [(0, 6), None, None, None,      None, None, (0, 2, 0.8), None,
                    (3, 4), None, None, None,       (5, 1), None, (7, 1), (10, 1)],
    # 174-friendly roller — syncopated, octave flick mid-bar
    "dnb_roller": [(0, 2), None, None, (0, 1),     None, None, (12, 1, 0.7), None,
                   None, (0, 2), None, None,        (0, 1), None, (7, 1, 0.7), (10, 1, 0.7)],
}


def render_loop(genome, root_midi, bpm, pattern, bars=2, swing=0.0, sr=SR_OUT, dev="cpu"):
    """Mono bass loop -> float32 np audio, seamless (release tails wrap the loop).
    swing pushes every odd 16th late by that fraction of a step (~0.12 = garage)."""
    step = 60.0 / bpm / 4.0                      # seconds per 16th note
    step_n = int(round(step * sr))
    loop_n = step_n * 16 * bars
    buf = np.zeros(loop_n, dtype=np.float32)
    tail = int(0.30 * sr)                        # room for release/decay
    swing_n = int(max(0.0, min(swing, 0.5)) * step_n)
    for bar in range(bars):
        for i, slot in enumerate(pattern):
            if slot is None:
                continue
            offset, gate = slot[0], slot[1]
            gain = slot[2] if len(slot) > 2 else 0.95
            midi = root_midi + offset
            gate_n = gate * step_n
            n = gate_n + tail
            note_dur = gate_n / sr
            a = render(genome, midi, sr=sr, n=n, note_dur=note_dur).cpu().numpy() * gain
            pos = (bar * 16 + i) * step_n + (swing_n if i % 2 == 1 else 0)
            end = pos + len(a)
            if end <= loop_n:                    # fits
                buf[pos:end] += a
            else:                                # wrap the tail back to the head
                first = loop_n - pos
                buf[pos:] += a[:first]
                rem = a[first:]
                buf[:len(rem)] += rem[:loop_n]   # clamp if a note is absurdly long
    peak = np.max(np.abs(buf)) + 1e-9
    return (buf / peak * 0.9).astype(np.float32)


def load_genome(spec, dev):
    if os.path.exists(spec):
        g = np.loadtxt(spec).astype(np.float32)
    else:
        g = np.array([float(x) for x in spec.split(",")], dtype=np.float32)
    return torch.tensor(upgrade_genome(g), device=dev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--genome", help="single patch: genome.txt or inline csv")
    ap.add_argument("--key", default="F1", help="root note, e.g. F1 / 33")
    ap.add_argument("--bpm", type=float, default=138)
    ap.add_argument("--pattern", default="garage_roll", choices=list(PATTERNS))
    ap.add_argument("--swing", type=float, default=0.0,
                    help="push odd 16ths late: 0 straight, ~0.12 garage shuffle")
    ap.add_argument("--bars", type=int, default=2)
    ap.add_argument("--out", default="loop.wav")
    # pack mode
    ap.add_argument("--pack", action="store_true")
    ap.add_argument("--genomes", default=None, help="comma list of genome files (sounds)")
    ap.add_argument("--keys", default="F1,G1,A1,C2")
    ap.add_argument("--bpms", default="130,138")
    ap.add_argument("--patterns", default="garage_roll,reese_hold,house_offbeat")
    ap.add_argument("--dir", default="packs/demo")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not a.pack:
        g = load_genome(a.genome, dev)
        audio = render_loop(g, note_to_midi(a.key), a.bpm, PATTERNS[a.pattern],
                            a.bars, swing=a.swing, dev=dev)
        sf.write(a.out, audio, SR_OUT)
        print(f"loop -> {a.out}  ({a.key} {a.bpm:.0f}bpm {a.pattern} {a.bars}bar 44.1k)")
        return

    os.makedirs(a.dir, exist_ok=True)
    sounds = [s.strip() for s in a.genomes.split(",") if s.strip()]
    keys = [k.strip() for k in a.keys.split(",")]
    bpms = [float(b) for b in a.bpms.split(",")]
    pats = [p.strip() for p in a.patterns.split(",")]
    n = 0
    for si, spec in enumerate(sounds):
        g = load_genome(spec, dev)
        snd = os.path.splitext(os.path.basename(spec))[0].replace(".wav", "")[:12] or f"snd{si}"
        for bpm in bpms:
            for key in keys:
                for pat in pats:
                    audio = render_loop(g, note_to_midi(key), bpm, PATTERNS[pat],
                                        a.bars, swing=a.swing, dev=dev)
                    name = f"{snd}_{int(bpm)}bpm_{key}_{pat}.wav"
                    sf.write(os.path.join(a.dir, name), audio, SR_OUT)
                    n += 1
    print(f"spammed {n} loops -> {a.dir}/  "
          f"({len(sounds)} sounds x {len(bpms)} bpm x {len(keys)} keys x {len(pats)} patterns)")


if __name__ == "__main__":
    import os as _os
    main()
    import sys; sys.stdout.flush()
    _os._exit(0)   # dodge ROCm-on-Windows teardown deadlock (see train.py)
