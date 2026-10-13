"""Train GenoNet to predict Serum 2 knob values from audio (HANDOFF.md 10).

Serum isn't differentiable, so this is plain parameter regression on the
pre-rendered set from serum_preset_set.py — no spectral re-render loss.
Validation shards use presets never seen in training. The bar to beat is the
mean predictor (always guess the train-set mean); per-param MAE is reported
against it so a knob the net can't hear shows up as ~0% gain, not hidden.

    python lab/train_serum.py --data real/serum_presets --epochs 30
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from model import GenoNet, device
from synth import melspec


def load_mels(data, dev):
    """All shards -> (mel f16, params, preset, is_val); mels cached beside the shards."""
    cache = data / "mels_cache.npz"
    shards = sorted(p for p in data.glob("shard_*.npz") if ".tmp" not in p.name)
    meta = json.loads((data / "meta.json").read_text(encoding="utf-8"))
    names = [p.name for p in shards]
    if cache.exists():
        c = np.load(cache, allow_pickle=False)
        if list(c["shards"]) == names:
            return c["mel"], c["params"], c["preset"], meta
    mels, params, presets = [], [], []
    for p in shards:
        d = np.load(p)
        with torch.no_grad():
            for i in range(0, len(d["audio"]), 500):
                a = torch.from_numpy(d["audio"][i:i + 500].astype(np.float32) / 32767).to(dev)
                mels.append(melspec(a).half().cpu().numpy())
        params.append(d["params"]); presets.append(d["preset"])
        print(f"  mel {p.name}", flush=True)
    mel, par, pre = np.concatenate(mels), np.concatenate(params), np.concatenate(presets)
    np.savez(cache, mel=mel, params=par, preset=pre, shards=np.array(names))
    return mel, par, pre, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("real/serum_presets"))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", type=Path, default=Path("lab/serum_net.pt"))
    a = ap.parse_args()
    dev = device()
    mel, par, pre, meta = load_mels(a.data, dev)
    names = meta["params"]
    # normalize labels to 0..1 inside each sampling window so all knobs weigh equally
    lo = np.array([w[0] for w in meta["windows"]], np.float32)
    hi = np.array([w[1] for w in meta["windows"]], np.float32)
    y = (par - lo) / (hi - lo)
    val = np.isin(pre, meta["val_presets"])
    tr = ~val
    print(f"device={dev}  clips {len(y)}: train {tr.sum()} / val {val.sum()} "
          f"(val = {len(meta['val_presets'])} held-out presets)", flush=True)

    Xtr = torch.from_numpy(mel[tr]).to(dev); Ytr = torch.from_numpy(y[tr]).to(dev)
    Xva = torch.from_numpy(mel[val]).to(dev); Yva = torch.from_numpy(y[val]).to(dev)
    base = (Yva - Ytr.mean(0)).abs().mean(0)  # mean-predictor MAE per knob
    print("baseline (predict train mean) val MAE: " +
          "  ".join(f"{n}={v:.3f}" for n, v in zip(names, base.tolist())), flush=True)

    net = GenoNet(n_params=len(names)).to(dev)
    steps = a.epochs * (len(Xtr) // a.bs)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.1)
    mse = nn.MSELoss()
    gen = torch.Generator(device=dev).manual_seed(0)
    best, best_mae, t0 = float("inf"), None, time.time()
    for ep in range(1, a.epochs + 1):
        net.train()
        order = torch.randperm(len(Xtr), device=dev, generator=gen)
        for i in range(len(Xtr) // a.bs):
            b = order[i * a.bs:(i + 1) * a.bs]
            loss = mse(net(Xtr[b].float()), Ytr[b])
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        net.eval()
        with torch.no_grad():
            pv = torch.cat([net(Xva[i:i + 1024].float()) for i in range(0, len(Xva), 1024)])
            mae = (pv - Yva).abs().mean(0)
        m = mae.mean().item()
        mark = ""
        if m < best:
            best, best_mae = m, mae.tolist()
            torch.save(net.state_dict(), a.out)
            mark = "  *saved"
        print(f"epoch {ep:3d}  train-mse {loss.item():.4f}  val-MAE {m:.4f} "
              f"(baseline {base.mean().item():.4f})  {time.time() - t0:.0f}s{mark}", flush=True)

    report = {"val_presets": [meta["presets"][i] for i in meta["val_presets"]],
              "per_param": {n: {"net": round(v, 4), "baseline": round(b, 4),
                                "gain_pct": round(100 * (1 - v / b), 1)}
                            for n, v, b in zip(names, best_mae, base.tolist())},
              "val_mae": round(best, 4), "baseline_mae": round(base.mean().item(), 4)}
    a.out.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\nper-knob val MAE (held-out presets), net vs baseline:")
    for n, r in report["per_param"].items():
        print(f"  {n:15s} {r['net']:.3f} vs {r['baseline']:.3f}  ({r['gain_pct']:+.0f}%)")
    print(f"-> {a.out}  report -> {a.out.with_suffix('.json')}")


if __name__ == "__main__":
    import os
    import sys
    main()
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0)  # dodge ROCm-on-Windows teardown deadlock
