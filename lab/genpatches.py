"""
genpatches.py — generate a diverse batch of synth patches for RLHF *quality* rating.

The point of RLHF here: spectral loss says nothing about whether a patch sounds
*good*. So we generate patches you can actually judge by ear, you star-rate them,
and a reward model learns "sounds-good" to steer breeding + the net's priors.

To make rating worth your time we don't sample uniform [0,1]^15 (that's mostly
mush — silent, all-noise, or clicks). We sample from musical **archetypes** with
per-parameter priors (bass / reese / lead / pluck / stab / pad / keys), plus
mutations of any known-good seed genome. Each is rendered at a note that suits the
archetype, obvious duds (silent / click-only) are dropped, and we write:

  <dir>/patch_NNN.wav     rendered audio
  <dir>/patches.jsonl     full record  {id,file,archetype,note,genome:[N_PARAMS]}
  <dir>/ratings.csv       id,file,archetype,note,rating,notes   (you fill rating)
  <dir>/index.html        in-browser rater: scrub, 1-5 stars, export ratings.csv

  python genpatches.py --n 48 --dir rlhf/round01
  python genpatches.py --n 64 --seed-genome remake_bass.wav.genome.txt --dir rlhf/round02

Then rate in the browser (open index.html), drop the exported ratings.csv back into
<dir>/, and feed it to the reward trainer.
"""

import argparse
import json
import os
import numpy as np
import soundfile as sf
import torch

from synth import render, to_wav, upgrade_genome, N_PARAMS, PARAM_NAMES, SR

# Each archetype is a set of (lo,hi) windows on the *normalized* [0,1] genome.
# Params not listed default to the full (0,1) range. Values reflect the synth.py
# mappings (e.g. cutoff is log 60Hz..10kHz, so 0.3 ~= 230Hz, 0.6 ~= 1.4kHz).
_FULL = {n: (0.0, 1.0) for n in PARAM_NAMES}

# wavetable morph reference (osc*_wt): 0=sine, ~0.14 tri, ~0.29 square,
# ~0.43 saw, then pulses/formant/rich toward 1.
ARCHETYPES = {
    # deep/sub bass: low cutoff, strong sub, saw-ish, punchy amp
    "bass":  dict(osc1_wt=(0.3, 0.55), osc2_wt=(0.0, 0.5), osc2_detune=(0.46, 0.54),
                  osc_mix=(0.2, 0.6), sub_level=(0.5, 1.0), noise_level=(0.0, 0.1),
                  drive=(0.0, 0.35),
                  cutoff=(0.22, 0.5), reso=(0.1, 0.5), filt_env=(0.2, 0.6),
                  filt_a=(0.0, 0.1), filt_d=(0.3, 0.7), amp_a=(0.0, 0.06),
                  amp_d=(0.3, 0.7), amp_s=(0.5, 1.0), amp_r=(0.05, 0.4),
                  pitch_env=(0.0, 0.06), pitch_dec=(0.3, 0.7), lfo_rate=(0.2, 0.6), lfo_depth=(0.0, 0.06)),
    # reese / dirty bass: wide detune, more resonance, growl
    "reese": dict(osc1_wt=(0.3, 0.55), osc2_wt=(0.3, 0.55), osc2_detune=(0.7, 0.95),
                  osc_mix=(0.4, 0.6), sub_level=(0.3, 0.8), noise_level=(0.0, 0.15),
                  drive=(0.2, 0.7),
                  cutoff=(0.3, 0.6), reso=(0.45, 0.85), filt_env=(0.2, 0.6),
                  filt_a=(0.0, 0.15), filt_d=(0.25, 0.7), amp_a=(0.0, 0.08),
                  amp_d=(0.4, 0.9), amp_s=(0.5, 1.0), amp_r=(0.1, 0.5),
                  pitch_env=(0.0, 0.06), pitch_dec=(0.3, 0.7), lfo_rate=(0.2, 0.6), lfo_depth=(0.0, 0.06)),
    # bright sustained lead
    "lead":  dict(osc1_wt=(0.4, 0.9), osc2_wt=(0.3, 0.8), osc2_detune=(0.5, 0.62),
                  osc_mix=(0.3, 0.7), sub_level=(0.0, 0.3), noise_level=(0.0, 0.1),
                  drive=(0.0, 0.4),
                  cutoff=(0.55, 0.9), reso=(0.15, 0.55), filt_env=(0.1, 0.5),
                  filt_a=(0.0, 0.2), filt_d=(0.2, 0.7), amp_a=(0.02, 0.25),
                  amp_d=(0.3, 0.8), amp_s=(0.6, 1.0), amp_r=(0.1, 0.5),
                  pitch_env=(0.0, 0.06), pitch_dec=(0.3, 0.7), lfo_rate=(0.2, 0.6), lfo_depth=(0.0, 0.06)),
    # pluck: snappy filter env, little/no sustain
    "pluck": dict(osc1_wt=(0.2, 0.6), osc2_wt=(0.2, 0.6), osc2_detune=(0.47, 0.6),
                  osc_mix=(0.2, 0.7), sub_level=(0.0, 0.4), noise_level=(0.0, 0.12),
                  drive=(0.0, 0.3),
                  cutoff=(0.4, 0.75), reso=(0.3, 0.8), filt_env=(0.5, 0.95),
                  filt_a=(0.0, 0.05), filt_d=(0.05, 0.3), amp_a=(0.0, 0.04),
                  amp_d=(0.1, 0.35), amp_s=(0.0, 0.25), amp_r=(0.05, 0.3),
                  pitch_env=(0.0, 0.06), pitch_dec=(0.3, 0.7), lfo_rate=(0.2, 0.6), lfo_depth=(0.0, 0.06)),
    # stab: pluck-ish but a touch fuller / mid cutoff
    "stab":  dict(osc1_wt=(0.3, 0.75), osc2_wt=(0.3, 0.75), osc2_detune=(0.45, 0.62),
                  osc_mix=(0.3, 0.7), sub_level=(0.1, 0.5), noise_level=(0.0, 0.12),
                  drive=(0.1, 0.5),
                  cutoff=(0.45, 0.75), reso=(0.25, 0.7), filt_env=(0.35, 0.8),
                  filt_a=(0.0, 0.06), filt_d=(0.12, 0.4), amp_a=(0.0, 0.05),
                  amp_d=(0.15, 0.45), amp_s=(0.15, 0.5), amp_r=(0.08, 0.4),
                  pitch_env=(0.0, 0.06), pitch_dec=(0.3, 0.7), lfo_rate=(0.2, 0.6), lfo_depth=(0.0, 0.06)),
    # pad: slow attack, soft, sustained
    "pad":   dict(osc1_wt=(0.05, 0.45), osc2_wt=(0.05, 0.45), osc2_detune=(0.55, 0.75),
                  osc_mix=(0.35, 0.65), sub_level=(0.1, 0.5), noise_level=(0.0, 0.15),
                  drive=(0.0, 0.2),
                  cutoff=(0.4, 0.7), reso=(0.1, 0.45), filt_env=(0.1, 0.5),
                  filt_a=(0.3, 0.7), filt_d=(0.3, 0.8), amp_a=(0.35, 0.75),
                  amp_d=(0.4, 0.9), amp_s=(0.6, 1.0), amp_r=(0.4, 0.8),
                  pitch_env=(0.0, 0.06), pitch_dec=(0.3, 0.7), lfo_rate=(0.2, 0.6), lfo_depth=(0.0, 0.06)),
    # keys / organ-ish: balanced, medium everything
    "keys":  dict(osc1_wt=(0.1, 0.6), osc2_wt=(0.1, 0.6), osc2_detune=(0.48, 0.58),
                  osc_mix=(0.3, 0.7), sub_level=(0.1, 0.5), noise_level=(0.0, 0.08),
                  drive=(0.0, 0.25),
                  cutoff=(0.45, 0.8), reso=(0.15, 0.5), filt_env=(0.2, 0.6),
                  filt_a=(0.0, 0.15), filt_d=(0.2, 0.6), amp_a=(0.0, 0.1),
                  amp_d=(0.25, 0.7), amp_s=(0.4, 0.85), amp_r=(0.1, 0.5),
                  pitch_env=(0.0, 0.06), pitch_dec=(0.3, 0.7), lfo_rate=(0.2, 0.6), lfo_depth=(0.0, 0.06)),
}

# an archetype key that isn't a real synth param would be silently ignored by
# _window's dict merge — the exact bug that hit the v1->v2 rename. Fail loud.
for _arch, _spec in ARCHETYPES.items():
    _bad = set(_spec) - set(PARAM_NAMES)
    assert not _bad, f"archetype {_arch!r} has unknown params: {sorted(_bad)}"

# representative MIDI note range per archetype (inclusive)
NOTE_RANGE = {
    "bass": (33, 43), "reese": (33, 45), "lead": (55, 69), "pluck": (48, 64),
    "stab": (45, 60), "pad": (45, 62), "keys": (48, 64),
}


def _window(arch):
    """Build (lo,hi) float arrays for an archetype, full-range for unspecified."""
    spec = {**_FULL, **ARCHETYPES[arch]}
    lo = np.array([spec[n][0] for n in PARAM_NAMES], dtype=np.float32)
    hi = np.array([spec[n][1] for n in PARAM_NAMES], dtype=np.float32)
    return lo, hi


def sample_archetype(arch, k, rng):
    lo, hi = _window(arch)
    g = lo + (hi - lo) * rng.random((k, N_PARAMS)).astype(np.float32)
    return np.clip(g, 0.0, 1.0)


def sample_seed_mutants(seed, k, rng, amount=0.12):
    """Mutations around a known-good genome — anchors the batch with real material."""
    g = seed[None, :] + rng.normal(0, amount, (k, N_PARAMS)).astype(np.float32)
    return np.clip(g, 0.0, 1.0)


def is_dud(audio):
    """True for obviously dead patches: near-silent or a bare click with no body.
    audio is post-normalized (peak ~0.9), so we judge by energy distribution."""
    x = np.abs(audio)
    rms = float(np.sqrt(np.mean(audio ** 2)))
    active = float(np.mean(x > 0.05))          # fraction of audibly-loud samples
    return rms < 0.03 or active < 0.03


def plan_counts(n, archs=None):
    """Spread n patches across archetypes (round-robin) + a slice of seed mutants."""
    archs = archs or list(ARCHETYPES.keys())
    counts = {a: 0 for a in archs}
    for i in range(n):
        counts[archs[i % len(archs)]] += 1
    return counts


def generate(n, dev, rng, gen, seed_genome=None, archs=None):
    """Return list of dicts {archetype,note,genome(np)} of n non-dud patches."""
    counts = plan_counts(n, archs)
    kept = []
    for arch, want in counts.items():
        lo_n, hi_n = NOTE_RANGE[arch]
        tries = 0
        got = 0
        while got < want and tries < want * 8 + 16:
            need = want - got
            cand = sample_archetype(arch, need * 2, rng)         # oversample for duds
            notes = rng.integers(lo_n, hi_n + 1, size=cand.shape[0])
            gt = torch.tensor(cand, device=dev)
            nt = torch.tensor(notes, dtype=torch.float32, device=dev)
            with torch.no_grad():
                audio = render(gt, nt, gen=gen).cpu().numpy()
            for j in range(cand.shape[0]):
                if got >= want:
                    break
                if not is_dud(audio[j]):
                    kept.append(dict(archetype=arch, note=int(notes[j]),
                                     genome=cand[j].copy()))
                    got += 1
            tries += need
    if seed_genome is not None:
        k = max(2, n // 8)
        muts = sample_seed_mutants(seed_genome, k, rng)
        notes = rng.integers(33, 45, size=k)
        gt = torch.tensor(muts, device=dev)
        nt = torch.tensor(notes, dtype=torch.float32, device=dev)
        with torch.no_grad():
            audio = render(gt, nt, gen=gen).cpu().numpy()
        for j in range(k):
            if not is_dud(audio[j]):
                kept.append(dict(archetype="seed", note=int(notes[j]),
                                 genome=muts[j].copy()))
    rng.shuffle(kept)
    return kept


HTML = """<!doctype html><html><head><meta charset=utf-8><title>synfection rater</title>
<style>
 body{{font:15px/1.4 system-ui,sans-serif;background:#111;color:#eee;margin:0;padding:24px}}
 h1{{font-size:18px;margin:0 0 4px}} .sub{{color:#888;margin:0 0 18px}}
 .card{{background:#1b1b1b;border:1px solid #2a2a2a;border-radius:10px;padding:14px 16px;
   margin:10px 0;display:flex;align-items:center;gap:16px}}
 .card.done{{border-color:#3a5}} .meta{{flex:1;min-width:0}}
 .name{{font-weight:600}} .tag{{color:#7bd;font-size:13px}} audio{{height:34px}}
 .stars{{font-size:26px;cursor:pointer;user-select:none;white-space:nowrap}}
 .stars span{{color:#444;padding:0 1px}} .stars span.on{{color:#fc4}}
 .x{{color:#a55;cursor:pointer;font-size:13px;padding-left:6px}}
 .x.on{{color:#f55;font-weight:700}}
 textarea{{background:#161616;color:#ccc;border:1px solid #333;border-radius:6px;
   width:160px;height:34px;resize:none;font:13px system-ui}}
 #bar{{position:sticky;top:0;background:#111;padding:10px 0;z-index:9;
   display:flex;gap:12px;align-items:center;border-bottom:1px solid #222}}
 button{{background:#2d7;color:#012;border:0;border-radius:6px;padding:8px 14px;
   font-weight:700;cursor:pointer}} #prog{{color:#9c9}}
 kbd{{background:#222;border:1px solid #333;border-radius:4px;padding:1px 5px;font-size:12px}}
</style></head><body>
<h1>synfection — patch quality rater</h1>
<p class=sub>{n} patches. Click stars (1–5), or ✕ to discard. Keys: focus a card, press
<kbd>1</kbd>–<kbd>5</kbd> to rate, <kbd>0</kbd> to discard, <kbd>Space</kbd> to play.
Then <b>Export ratings.csv</b> and drop it into this folder.</p>
<div id=bar><button onclick=expo()>⬇ Export ratings.csv</button>
 <span id=prog></span></div>
<div id=list></div>
<script>
const DATA = {data};
const R = {{}};   // id -> {{rating, notes}}
const list = document.getElementById('list');
function draw(){{
 list.innerHTML='';
 DATA.forEach((p,i)=>{{
  const r = R[p.id]||{{rating:'',notes:''}};
  const card = document.createElement('div');
  card.className='card'+(r.rating!==''?' done':''); card.tabIndex=0; card.dataset.i=i;
  let stars='';
  for(let s=1;s<=5;s++) stars+=`<span class="${{r.rating>=s&&r.rating!==0?'on':''}}" onclick="rate(${{i}},${{s}})">★</span>`;
  card.innerHTML=`<div class=meta><div class=name>${{p.file}}</div>
    <div class=tag>${{p.archetype}} · MIDI ${{p.note}}</div></div>
    <audio controls preload=none src="${{p.file}}"></audio>
    <div class=stars>${{stars}}<span class="x ${{r.rating===0?'on':''}}" onclick="rate(${{i}},0)">✕</span></div>
    <textarea placeholder=notes oninput="note(${{i}},this.value)">${{r.notes}}</textarea>`;
  list.appendChild(card);
 }});
 const done=Object.values(R).filter(x=>x.rating!=='').length;
 document.getElementById('prog').textContent=`${{done}} / ${{DATA.length}} rated`;
}}
function rate(i,s){{const p=DATA[i];R[p.id]={{rating:s,notes:(R[p.id]||{{}}).notes||''}};draw();}}
function note(i,v){{const p=DATA[i];R[p.id]={{rating:(R[p.id]||{{rating:''}}).rating,notes:v}};}}
document.addEventListener('keydown',e=>{{
 const c=document.activeElement.closest?document.activeElement.closest('.card'):null;
 if(!c)return; const i=+c.dataset.i;
 if(e.key>='0'&&e.key<='5'){{rate(i,+e.key);e.preventDefault();
   const cards=[...document.querySelectorAll('.card')];if(cards[i+1])cards[i+1].focus();}}
 if(e.key===' '){{const a=c.querySelector('audio');a.paused?a.play():a.pause();e.preventDefault();}}
}});
function expo(){{
 let csv='id,file,archetype,note,rating,notes\\n';
 DATA.forEach(p=>{{const r=R[p.id]||{{rating:'',notes:''}};
  const nt=(r.notes||'').replace(/"/g,'""');
  csv+=`${{p.id}},${{p.file}},${{p.archetype}},${{p.note}},${{r.rating}},"${{nt}}"\\n`;}});
 const b=new Blob([csv],{{type:'text/csv'}});const a=document.createElement('a');
 a.href=URL.createObjectURL(b);a.download='ratings.csv';a.click();
}}
draw();
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--dir", default="rlhf/round01")
    ap.add_argument("--seed-genome", default=None,
                    help="path to a known-good *.genome.txt to anchor with mutants")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--archetypes", default=None,
                    help="comma list to focus the round (e.g. bass,reese,stab); default all")
    a = ap.parse_args()

    archs = None
    if a.archetypes:
        archs = [s.strip() for s in a.archetypes.split(",") if s.strip()]
        unknown = [s for s in archs if s not in ARCHETYPES]
        if unknown:
            raise SystemExit(f"unknown archetypes {unknown}; known: {list(ARCHETYPES)}")

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(a.seed)
    gen = torch.Generator(device=dev).manual_seed(a.seed)
    os.makedirs(a.dir, exist_ok=True)

    seed_genome = None
    if a.seed_genome and os.path.exists(a.seed_genome):
        seed_genome = upgrade_genome(np.loadtxt(a.seed_genome).astype(np.float32))

    patches = generate(a.n, dev, rng, gen, seed_genome, archs)

    records = []
    for i, p in enumerate(patches):
        fname = f"patch_{i:03d}.wav"
        g = torch.tensor(p["genome"], device=dev)
        audio = render(g, p["note"], gen=gen)
        sf.write(os.path.join(a.dir, fname), to_wav(audio), SR)
        records.append(dict(id=i, file=fname, archetype=p["archetype"],
                            note=p["note"], genome=[round(float(x), 5) for x in p["genome"]]))

    with open(os.path.join(a.dir, "patches.jsonl"), "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    with open(os.path.join(a.dir, "ratings.csv"), "w") as f:
        f.write("id,file,archetype,note,rating,notes\n")
        for r in records:
            f.write(f"{r['id']},{r['file']},{r['archetype']},{r['note']},,\n")

    # web rater carries only what the page needs (id/file/archetype/note)
    light = [dict(id=r["id"], file=r["file"], archetype=r["archetype"], note=r["note"])
             for r in records]
    with open(os.path.join(a.dir, "index.html"), "w", encoding="utf-8") as f:
        f.write(HTML.format(n=len(records), data=json.dumps(light)))

    by_arch = {}
    for r in records:
        by_arch[r["archetype"]] = by_arch.get(r["archetype"], 0) + 1
    spread = "  ".join(f"{k}:{v}" for k, v in sorted(by_arch.items()))
    print(f"generated {len(records)} patches -> {a.dir}/")
    print(f"  spread: {spread}")
    print(f"  rate them: open {os.path.join(a.dir, 'index.html')} in a browser")
    print(f"  then drop the exported ratings.csv back into {a.dir}/")


if __name__ == "__main__":
    import sys
    main()
    sys.stdout.flush(); sys.stderr.flush()   # os._exit skips buffer flush
    os._exit(0)   # dodge ROCm-on-Windows teardown deadlock (see train.py)
