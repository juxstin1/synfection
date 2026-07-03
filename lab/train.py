"""
Train GenoNet DDSP-style on infinite on-the-fly data.

Each step: sample random genomes + notes -> render target audio (no grad) ->
net predicts a genome from the target's mel -> re-render (with grad) -> optimize
a hybrid loss: parameter MSE + multi-scale spectral loss on the re-rendered audio.
The spectral term is what makes matches sound right despite the many-to-one
nature of synth parameters.

    python train.py --steps 4000 --bs 32 --out genonet.pt
"""

import argparse
import time
import torch
import torch.nn as nn

from augment import augment
from synth import render, melspec, N_PARAMS, SR
from model import GenoNet, device
from losses import multiscale_stft

NOTE_LO, NOTE_HI = 36, 72


def sample_batch(B, dev, gen):
    g = torch.rand(B, N_PARAMS, device=dev, generator=gen)
    notes = torch.randint(NOTE_LO, NOTE_HI + 1, (B,), device=dev, generator=gen).float()
    return g, notes


def make_val(n, dev, seed=12345):
    gen = torch.Generator(device=dev).manual_seed(seed)
    g, notes = sample_batch(n, dev, gen)
    with torch.no_grad():
        audio = render(g, notes, gen=gen)
        mel = melspec(audio)
    return g, notes, audio, mel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--spec_w", type=float, default=1.0)
    ap.add_argument("--param_w", type=float, default=1.0)
    ap.add_argument("--val_every", type=int, default=200)
    ap.add_argument("--out", default="genonet.pt")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--augment", type=float, default=0.0,
                    help="prob of real-world coloration on the input mel (0 = off)")
    a = ap.parse_args()

    dev = device()
    print(f"device={dev}  steps={a.steps}  bs={a.bs}  params={N_PARAMS}", flush=True)

    net = GenoNet().to(dev)
    if a.resume:
        net.load_state_dict(torch.load(a.resume, map_location=dev))
        print(f"resumed from {a.resume}")
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.1)
    mse = nn.MSELoss()

    Vg, Vn, Va, Vmel = make_val(256, dev)
    gen = torch.Generator(device=dev).manual_seed(0)
    best = float("inf")
    t0 = time.time()

    for step in range(1, a.steps + 1):
        net.train()
        g, notes = sample_batch(a.bs, dev, gen)
        with torch.no_grad():
            target = render(g, notes, gen=gen)
            # color only what the net hears — the loss target stays clean,
            # so it learns to see through real-world grime to honest genomes
            heard = augment(target, gen, p=a.augment) if a.augment > 0 else target
            tmel = melspec(heard)
        pred = net(tmel)
        recon = render(pred, notes)
        loss = a.param_w * mse(pred, g) + a.spec_w * multiscale_stft(recon, target)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        opt.step()
        sched.step()

        if step % a.val_every == 0 or step == 1:
            net.eval()
            with torch.no_grad():
                vp = net(Vmel)
                vrec = render(vp, Vn)
                vspec = multiscale_stft(vrec, Va).item()
                vmae = (vp - Vg).abs().mean().item()
            mark = ""
            if vspec < best:
                best = vspec
                torch.save(net.state_dict(), a.out)
                mark = "  *saved"
            rate = step / (time.time() - t0)
            print(f"step {step:5d}/{a.steps}  loss {loss.item():.3f}  "
                  f"val-spec {vspec:.3f}  val-pMAE {vmae:.3f}  "
                  f"({rate:.1f} it/s){mark}", flush=True)

    print(f"best val-spec {best:.3f} -> {a.out}")


if __name__ == "__main__":
    import os
    import sys
    main()
    # ROCm-on-Windows deadlocks during interpreter teardown (driver cleanup),
    # leaving unkillable GPU-hogging zombies. Skip cleanup entirely.
    sys.stdout.flush(); sys.stderr.flush()   # os._exit skips buffer flush
    os._exit(0)
