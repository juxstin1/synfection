"""
reward_mel.py — perceptual reward model. Judges the *sound*, not the knob values.

The genome reward model (reward.py) only learned archetype, not within-archetype
quality — because raw 15-d params don't linearize "sounds good". This one renders
the patch and scores its **log-mel spectrogram** through a small CNN, so it can (in
principle) tell a great reese from a mediocre one.

Tiny dataset (~102 ratings), so we:
  * augment each patch by rendering it at several notes (quality is ~note-invariant)
    -> ~3x the labels, and teaches note-invariance,
  * use GROUP-aware CV (all views of one patch stay in the same fold) for an honest
    within-archetype correlation — the whole point of the upgrade.

  python reward_mel.py train --rounds rlhf/round01 rlhf/round02 --out reward_mel.pt
  python reward_mel.py gen   --reward reward_mel.pt --n 48 --pool 400 --dir rlhf/round03
  #   -> rate it:  python serve.py --dir rlhf/round03
"""

import argparse
import numpy as np
import torch
import torch.nn as nn

from synth import render, melspec, N_MELS
from model import _block
import reward as RW          # reuse load_rounds, diverse_select, write_round
import genpatches as gp

AUG_NOTES = (-4, 0, 5)       # semitone offsets per patch view
NOTE_CLAMP = (28, 84)


class MelRewardNet(nn.Module):
    """log-mel -> quality[0,1]. Compact CNN; heavy dropout for the small dataset."""
    def __init__(self, p=0.3):
        super().__init__()
        self.features = nn.Sequential(
            _block(1, 16), _block(16, 32), _block(32, 64),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(), nn.Linear(64, 64), nn.GELU(), nn.Dropout(p),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        return torch.sigmoid(self.head(self.features(x))).squeeze(-1)


def featurize(rounds, dev, seed=0):
    """Render every rated patch at AUG_NOTES offsets -> log-mel views.
    Returns mel(M,1,n_mels,F) cpu tensor, y(M), group(M), arch(list M)."""
    X, y, meta = RW.load_rounds(rounds)
    genomes = torch.tensor(X, device=dev)
    notes0 = []
    # recover each patch's note from its round record (load_rounds drops it, so
    # re-read alongside). Simplest: re-pull notes via a parallel load.
    notes0 = _patch_notes(rounds)
    assert len(notes0) == len(X), f"note/label mismatch {len(notes0)} vs {len(X)}"

    mels, ys, groups, archs = [], [], [], []
    gen = torch.Generator(device=dev).manual_seed(seed)
    for gi, off in enumerate(AUG_NOTES):
        notes = torch.tensor(
            [min(max(n + off, NOTE_CLAMP[0]), NOTE_CLAMP[1]) for n in notes0],
            dtype=torch.float32, device=dev)
        # chunk to keep memory sane
        for s in range(0, len(X), 64):
            g = genomes[s:s + 64]
            nt = notes[s:s + 64]
            with torch.no_grad():
                audio = render(g, nt, gen=gen)
                m = melspec(audio).cpu()              # (b,n_mels,F)
            mels.append(m)
            ys.extend(y[s:s + 64])
            groups.extend(range(s, s + g.shape[0]))   # group = patch index
            archs.extend(meta[i]["archetype"] for i in range(s, s + g.shape[0]))
    mel = torch.cat(mels, 0).unsqueeze(1)             # (M,1,n_mels,F)
    return mel, np.array(ys, np.float32), np.array(groups), archs


def _patch_notes(rounds):
    import json, os
    notes = []
    for d in rounds:
        recs = {}
        with open(os.path.join(d, "patches.jsonl")) as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    recs[r["id"]] = r
        path = os.path.join(d, "ratings.csv")
        if not os.path.exists(path):
            continue
        import csv
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                if (row.get("rating") or "").strip() == "":
                    continue
                if int(row["id"]) in recs:
                    notes.append(int(recs[int(row["id"])]["note"]))
    return notes


def fit(mel, y, dev, epochs=250, lr=1e-3, wd=1e-3, p=0.3, seed=0):
    torch.manual_seed(seed)
    net = MelRewardNet(p=p).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    Xt = mel.to(dev)
    yt = torch.tensor(y, device=dev)
    wt = torch.tensor(RW._sample_weights(y), device=dev)
    net.train()
    for _ in range(epochs):
        idx = torch.randperm(len(yt), device=dev)
        for s in range(0, len(yt), 64):                # minibatch (BatchNorm-friendly)
            b = idx[s:s + 64]
            pred = net(Xt[b])
            loss = (wt[b] * (pred - yt[b]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    return net


def group_cv(mel, y, groups, archs, dev, k=5, seed=0):
    """Group k-fold: views of a patch never split across folds. Per-group OOF pred."""
    uniq = np.unique(groups)
    rng = np.random.default_rng(seed)
    folds = np.array_split(rng.permutation(uniq), k)
    oof = np.full(len(uniq), np.nan, np.float32)
    gpos = {gid: i for i, gid in enumerate(uniq)}
    for fold in folds:
        te = np.isin(groups, fold)
        tr = ~te
        net = fit(mel[torch.tensor(tr)], y[tr], dev, seed=seed)
        net.eval()
        with torch.no_grad():
            pr = net(mel[torch.tensor(te)].to(dev)).cpu().numpy()
        # average view-preds within each test group
        tg = groups[te]
        for gid in fold:
            m = tg == gid
            if m.any():
                oof[gpos[gid]] = pr[m].mean()
    # per-group labels + archetype (constant within group)
    gy = np.array([y[groups == gid][0] for gid in uniq])
    ga = [archs[np.where(groups == gid)[0][0]] for gid in uniq]
    return oof, gy, np.array(ga)


def cmd_train(a):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("featurizing (render + mel, augmented)...")
    mel, y, groups, archs = featurize(a.rounds, dev, seed=a.seed)
    print(f"  {len(np.unique(groups))} patches x {len(AUG_NOTES)} notes = {len(y)} views")
    oof, gy, ga = group_cv(mel, y, groups, archs, dev, seed=a.seed)
    ok = ~np.isnan(oof)
    pear = float(np.corrcoef(oof[ok], gy[ok])[0, 1])
    try:
        from scipy.stats import spearmanr
        spear = float(spearmanr(oof[ok], gy[ok]).correlation)
    except Exception:
        spear = float("nan")
    print(f"\nout-of-fold (per patch): Spearman {spear:.3f}  Pearson {pear:.3f}  "
          f"MAE {np.mean(np.abs(oof[ok]-gy[ok])):.3f}")
    print("within-archetype (the acceptance test):")
    for arch in ["reese", "bass", "stab", "pad"]:
        m = (ga == arch) & ok
        if m.sum() > 4 and np.std(gy[m]) > 1e-6 and np.std(oof[m]) > 1e-6:
            c = float(np.corrcoef(oof[m], gy[m])[0, 1])
            flag = "  <- positive!" if c > 0.15 else ("  (flat)" if abs(c) <= 0.15 else "  <- still negative")
            print(f"  {arch:6s} n={m.sum():2d}  Pearson {c:+.3f}{flag}")
    net = fit(mel, y, dev, seed=a.seed)
    torch.save(dict(state_dict=net.state_dict(), p=0.3, kind="mel"), a.out)
    print(f"saved perceptual reward -> {a.out}")


def score_mel(net, genomes, notes, dev):
    net.eval()
    out = []
    G = torch.tensor(np.asarray(genomes, np.float32), device=dev)
    nt = torch.tensor(np.asarray(notes, np.float32), device=dev)
    gen = torch.Generator(device=dev).manual_seed(0)
    with torch.no_grad():
        for s in range(0, len(genomes), 64):
            audio = render(G[s:s + 64], nt[s:s + 64], gen=gen)
            out.append(net(melspec(audio)).cpu().numpy())
    return np.concatenate(out)


def cmd_gen(a):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(a.reward, map_location=dev)
    net = MelRewardNet(p=ckpt.get("p", 0.3)).to(dev)
    net.load_state_dict(ckpt["state_dict"])

    rng = np.random.default_rng(a.seed)
    gen = torch.Generator(device=dev).manual_seed(a.seed)
    seedg = None
    if a.seed_genome:
        import os
        if os.path.exists(a.seed_genome):
            seedg = np.loadtxt(a.seed_genome).astype(np.float32)
    print(f"sampling pool of {a.pool}...")
    pool = gp.generate(a.pool, dev, rng, gen, seedg)
    if a.archetypes:
        allowed = {x.strip() for x in a.archetypes.split(",")}
        pool = [p for p in pool if p["archetype"] in allowed]
        print(f"  gated to {sorted(allowed)} -> {len(pool)} candidates")
    scores = score_mel(net, [p["genome"] for p in pool], [p["note"] for p in pool], dev)
    for p, s in zip(pool, scores):
        p["pred"] = float(s)
    picks = RW.diverse_select(pool, scores, a.n, min_dist=a.min_dist, cap=a.per_arch_cap)
    picked = [pool[i] for i in picks]
    rng.shuffle(picked)
    recs = RW.write_round(a.dir, picked, dev, gen)

    ps = np.array([r["pred_reward"] for r in recs])
    import collections
    by = collections.defaultdict(list)
    for r in recs:
        by[r["archetype"]].append(r["pred_reward"])
    print(f"\nwrote {len(recs)} patches -> {a.dir}/  pred {ps.min():.2f}/{np.median(ps):.2f}/{ps.max():.2f}")
    print("  spread: " + "  ".join(f"{k}:{len(v)}@{np.mean(v):.2f}"
          for k, v in sorted(by.items(), key=lambda x: -np.mean(x[1]))))
    print(f"\n  rate it:  python serve.py --dir {a.dir}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--rounds", nargs="+", default=["rlhf/round01", "rlhf/round02"])
    t.add_argument("--out", default="reward_mel.pt")
    t.add_argument("--seed", type=int, default=0)
    t.set_defaults(fn=cmd_train)
    g = sub.add_parser("gen")
    g.add_argument("--reward", default="reward_mel.pt")
    g.add_argument("--n", type=int, default=48)
    g.add_argument("--pool", type=int, default=400)
    g.add_argument("--dir", default="rlhf/round03")
    g.add_argument("--min-dist", type=float, default=0.45)
    g.add_argument("--per-arch-cap", type=int, default=10)
    g.add_argument("--archetypes", default=None,
                   help="comma list to gate the pool, e.g. reese,bass,seed (your good ones)")
    g.add_argument("--seed-genome", default=None)
    g.add_argument("--seed", type=int, default=2)
    g.set_defaults(fn=cmd_gen)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    import os, sys
    main()
    sys.stdout.flush(); sys.stderr.flush()   # os._exit skips buffer flush
    os._exit(0)   # dodge ROCm-on-Windows teardown deadlock (see train.py)
