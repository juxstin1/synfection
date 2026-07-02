"""
reward.py — the RLHF reward model + reward-guided patch generation.

Two modes:

  # 1. learn "sounds-good" from your star ratings (one or more rated rounds)
  python reward.py train --rounds rlhf/round01 --out reward.pt

  # 2. use it to breed the next round, pre-filtered toward what you like
  python reward.py gen --reward reward.pt --n 48 --pool 600 --dir rlhf/round02
  #   -> then rate it:  python serve.py --dir rlhf/round02

The reward model maps a genome (N_PARAMS) -> predicted quality in [0,1] (= stars/5,
discard=0). Legacy 15-param (v1 engine) rounds are auto-upgraded on load with a
neutral drive=0 — approximate under the v2 wavetable engine, good enough to keep
the ratings useful. Tiny dataset, so it's a small, heavily-regularized MLP and we report
honest out-of-fold correlation rather than a meaningless train score. `gen` samples
a big archetype pool, scores it, and keeps the top-N with a diversity guard so you
get good *and* varied patches to rate — which sharpens the next reward model.
"""

import argparse
import csv
import json
import os
import numpy as np
import torch
import torch.nn as nn

from synth import render, to_wav, upgrade_genome, N_PARAMS, SR
import genpatches as gp


# ----- data -----------------------------------------------------------------

def stars_to_reward(stars):
    """0 (discard) -> 0.0 ; 1..5 stars -> 0.2..1.0."""
    return 0.0 if stars == 0 else stars / 5.0


def load_rounds(round_dirs):
    """Join patches.jsonl + ratings.csv by id across rounds. -> X(n,15), y(n,), meta."""
    X, y, meta = [], [], []
    for d in round_dirs:
        genomes = {}
        with open(os.path.join(d, "patches.jsonl")) as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    genomes[r["id"]] = r
        path = os.path.join(d, "ratings.csv")
        if not os.path.exists(path):
            print(f"  ! {d} has no ratings.csv, skipping")
            continue
        n = 0
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                rt = (row.get("rating") or "").strip()
                if rt == "":
                    continue
                rec = genomes.get(int(row["id"]))
                if rec is None:
                    continue
                X.append(upgrade_genome(np.asarray(rec["genome"], dtype=np.float32)))
                y.append(stars_to_reward(int(rt)))
                meta.append(dict(round=d, archetype=rec["archetype"]))
                n += 1
        print(f"  {d}: {n} rated patches")
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), meta


# ----- model ----------------------------------------------------------------

class RewardNet(nn.Module):
    """genome(15) -> reward[0,1]. Small + dropout-heavy; the dataset is tiny."""
    def __init__(self, d=N_PARAMS, h=32, p=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, h), nn.GELU(), nn.Dropout(p),
            nn.Linear(h, h), nn.GELU(), nn.Dropout(p),
            nn.Linear(h, 1),
        )

    def forward(self, x):
        return torch.sigmoid(self.net(x)).squeeze(-1)


def _sample_weights(y):
    """Inverse-frequency weights (binned) so the rare 5★s aren't drowned by 1★s."""
    bins = np.clip((y * 5 + 0.5).astype(int), 0, 5)
    counts = np.bincount(bins, minlength=6).astype(np.float32)
    w = 1.0 / np.maximum(counts[bins], 1.0)
    return (w / w.mean()).astype(np.float32)


def fit(X, y, dev, epochs=600, lr=3e-3, wd=2e-3, h=32, p=0.2, seed=0):
    torch.manual_seed(seed)
    net = RewardNet(h=h, p=p).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    Xt = torch.tensor(X, device=dev)
    yt = torch.tensor(y, device=dev)
    wt = torch.tensor(_sample_weights(y), device=dev)
    net.train()
    for _ in range(epochs):
        pred = net(Xt)
        loss = (wt * (pred - yt) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return net


def cross_val(X, y, dev, k=5, seed=0):
    """Pooled out-of-fold predictions -> honest Spearman/Pearson + MAE."""
    n = len(X)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    folds = np.array_split(idx, min(k, n))
    oof = np.zeros(n, dtype=np.float32)
    for fi in folds:
        tr = np.setdiff1d(idx, fi)
        net = fit(X[tr], y[tr], dev, seed=seed)
        net.eval()
        with torch.no_grad():
            oof[fi] = net(torch.tensor(X[fi], device=dev)).cpu().numpy()
    pear = float(np.corrcoef(oof, y)[0, 1]) if np.std(oof) > 1e-6 else float("nan")
    try:
        from scipy.stats import spearmanr
        spear = float(spearmanr(oof, y).correlation)
    except Exception:
        spear = float("nan")
    mae = float(np.mean(np.abs(oof - y)))
    return dict(pearson=pear, spearman=spear, mae=mae, oof=oof)


# ----- reward-guided generation ---------------------------------------------

def score_genomes(net, genomes, dev):
    net.eval()
    with torch.no_grad():
        return net(torch.tensor(np.asarray(genomes, dtype=np.float32),
                                device=dev)).cpu().numpy()


def mc_uncertainty(net, genomes, dev, passes=30):
    """MC-dropout std per genome — how unsure the reward model is. High = informative
    to rate (active learning), because that's where it'll learn the most."""
    net.train()                                   # keep dropout ON
    X = torch.tensor(np.asarray(genomes, dtype=np.float32), device=dev)
    with torch.no_grad():
        outs = torch.stack([net(X) for _ in range(passes)])   # (passes, n)
    net.eval()
    return outs.std(0).cpu().numpy()


def diverse_select(cands, rank_val, k, min_dist=0.3, cap=None, exclude=None):
    """Greedy pick by rank_val (desc), skipping genomes within min_dist (L2) of a
    pick and capping per-archetype. exclude: set of already-taken indices.
    Falls back to filling without the diversity/cap constraints to reach k."""
    G = np.asarray([c["genome"] for c in cands], dtype=np.float32)
    order = np.argsort(-rank_val)
    exclude = set(exclude or [])
    picks, per_arch = [], {}
    def ok(i):
        a = cands[i]["archetype"]
        if cap and per_arch.get(a, 0) >= cap:
            return False
        return all(np.linalg.norm(G[i] - G[j]) >= min_dist for j in picks)
    for i in order:
        if len(picks) >= k:
            break
        if i in exclude or i in picks:
            continue
        if ok(i):
            picks.append(i)
            arch = cands[i]["archetype"]
            per_arch[arch] = per_arch.get(arch, 0) + 1
    for i in order:                               # relax to guarantee k
        if len(picks) >= k:
            break
        if i not in exclude and i not in picks:
            picks.append(i)
    return picks


def write_round(directory, records, dev, gen):
    os.makedirs(directory, exist_ok=True)
    out = []
    for i, p in enumerate(records):
        fname = f"patch_{i:03d}.wav"
        g = torch.tensor(np.asarray(p["genome"], dtype=np.float32), device=dev)
        import soundfile as sf
        sf.write(os.path.join(directory, fname), to_wav(render(g, p["note"], gen=gen)), SR)
        out.append(dict(id=i, file=fname, archetype=p["archetype"], note=p["note"],
                        genome=[round(float(x), 5) for x in p["genome"]],
                        pred_reward=round(float(p.get("pred", 0)), 4)))
    with open(os.path.join(directory, "patches.jsonl"), "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    with open(os.path.join(directory, "ratings.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "file", "archetype", "note", "rating", "notes"])
        for r in out:
            w.writerow([r["id"], r["file"], r["archetype"], r["note"], "", ""])
    return out


def cmd_train(a):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("loading ratings:")
    X, y, meta = load_rounds(a.rounds)
    if len(X) < 8:
        print(f"only {len(X)} rated patches — rate more first."); return
    cv = cross_val(X, y, dev, seed=a.seed)
    print(f"\n{len(X)} examples  | out-of-fold: "
          f"Spearman {cv['spearman']:.3f}  Pearson {cv['pearson']:.3f}  MAE {cv['mae']:.3f}")
    net = fit(X, y, dev, seed=a.seed)
    torch.save(dict(state_dict=net.state_dict(), h=32, p=0.2,
                    n_train=len(X)), a.out)
    print(f"saved reward model -> {a.out}")


def cmd_gen(a):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(a.reward, map_location=dev)
    d_ckpt = ckpt["state_dict"]["net.0.weight"].shape[1]
    if d_ckpt != N_PARAMS:
        raise SystemExit(
            f"{a.reward} expects a {d_ckpt}-param genome but the engine now has "
            f"{N_PARAMS} — retrain first:  python reward.py train --rounds rlhf/round01 ...")
    net = RewardNet(h=ckpt.get("h", 32), p=ckpt.get("p", 0.2)).to(dev)
    net.load_state_dict(ckpt["state_dict"])

    rng = np.random.default_rng(a.seed)
    gen = torch.Generator(device=dev).manual_seed(a.seed)
    seedg = None
    if a.seed_genome and os.path.exists(a.seed_genome):
        seedg = upgrade_genome(np.loadtxt(a.seed_genome).astype(np.float32))
    print(f"sampling pool of {a.pool} archetype patches...")
    pool = gp.generate(a.pool, dev, rng, gen, seedg)
    scores = score_genomes(net, [p["genome"] for p in pool], dev)
    for p, s in zip(pool, scores):
        p["pred"] = float(s)

    n_explore = round(a.explore * a.n)
    n_exploit = a.n - n_explore
    # exploit: your best patches, diverse + capped so it can't collapse to one archetype
    ex = diverse_select(pool, scores, n_exploit, min_dist=a.min_dist, cap=a.per_arch_cap)
    for i in ex:
        pool[i]["kind"] = "exploit"
    # explore: where the reward model is *unsure* — these teach it the most
    picks = list(ex)
    if n_explore:
        unc = mc_uncertainty(net, [p["genome"] for p in pool], dev)
        exp = diverse_select(pool, unc, n_explore, min_dist=a.min_dist, exclude=set(ex))
        for i in exp:
            pool[i]["kind"] = "explore"
        picks += exp
    picked = [pool[i] for i in picks]
    rng.shuffle(picked)
    recs = write_round(a.dir, picked, dev, gen)
    kind = {i: pool[i].get("kind", "?") for i in picks}

    ps = np.array([r["pred_reward"] for r in recs])
    by = {}
    for r in recs:
        by.setdefault(r["archetype"], []).append(r["pred_reward"])
    print(f"\nwrote {len(recs)} patches -> {a.dir}/  ({n_exploit} exploit + {n_explore} explore)")
    print(f"  exploit predicted reward: min {ps.min():.2f}  med {np.median(ps):.2f}  max {ps.max():.2f}")
    print(f"  (pool median was {np.median(scores):.2f})")
    print("  archetype spread: " +
          "  ".join(f"{k}:{len(v)}@{np.mean(v):.2f}" for k, v in sorted(by.items(), key=lambda x: -np.mean(x[1]))))
    print(f"\n  rate it:  python serve.py --dir {a.dir}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train")
    t.add_argument("--rounds", nargs="+", default=["rlhf/round01"])
    t.add_argument("--out", default="reward.pt")
    t.add_argument("--seed", type=int, default=0)
    t.set_defaults(fn=cmd_train)

    g = sub.add_parser("gen")
    g.add_argument("--reward", default="reward.pt")
    g.add_argument("--n", type=int, default=48)
    g.add_argument("--pool", type=int, default=600)
    g.add_argument("--dir", default="rlhf/round02")
    g.add_argument("--min-dist", type=float, default=0.45)
    g.add_argument("--per-arch-cap", type=int, default=10)
    g.add_argument("--explore", type=float, default=0.25,
                   help="fraction picked by reward-model uncertainty, not score")
    g.add_argument("--seed-genome", default=None)
    g.add_argument("--seed", type=int, default=1)
    g.set_defaults(fn=cmd_gen)

    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    import sys
    main()
    sys.stdout.flush(); sys.stderr.flush()   # os._exit skips buffer flush
    os._exit(0)   # dodge ROCm-on-Windows teardown deadlock (see train.py)
