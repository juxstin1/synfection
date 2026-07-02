"""
Export the trained GenoNet to weights/genonet.bin for the Rust CLI.

Folds each BatchNorm into the preceding conv (inference-only), and bundles the
librosa mel filterbank so the Rust side needs no audio deps to reproduce the
net's input features exactly.

Format (little-endian): magic "SYNF", u32 version, u32 n_tensors, then per
tensor: u16 name_len, name utf8, u8 ndim, u32 dims..., f32 data.

    python export_net.py --model genonet.pt --out weights/genonet.bin
"""

import argparse
import os
import struct

import numpy as np
import torch

from synth import SR, N_MELS
from model import GenoNet


def fold_bn(conv_w, conv_b, bn):
    g = bn["weight"]; b = bn["bias"]; m = bn["running_mean"]; v = bn["running_var"]
    s = g / torch.sqrt(v + 1e-5)
    return conv_w * s[:, None, None, None], (conv_b - m) * s + b


def write_tensor(f, name, t):
    a = np.ascontiguousarray(t.detach().cpu().numpy().astype(np.float32))
    f.write(struct.pack("<H", len(name)))
    f.write(name.encode())
    f.write(struct.pack("<B", a.ndim))
    for d in a.shape:
        f.write(struct.pack("<I", d))
    f.write(a.tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="genonet.pt")
    ap.add_argument("--out", default="weights/genonet.bin")
    a = ap.parse_args()

    net = GenoNet()
    net.load_state_dict(torch.load(a.model, map_location="cpu"))
    net.eval()
    sd = net.state_dict()

    def bn(prefix):
        return {k: sd[f"{prefix}.{k}"] for k in
                ("weight", "bias", "running_mean", "running_var")}

    tensors = []
    for blk in range(4):
        w1, b1 = fold_bn(sd[f"features.{blk}.0.weight"], sd[f"features.{blk}.0.bias"],
                         bn(f"features.{blk}.1"))
        w2, b2 = fold_bn(sd[f"features.{blk}.3.weight"], sd[f"features.{blk}.3.bias"],
                         bn(f"features.{blk}.4"))
        tensors += [(f"b{blk}c1w", w1), (f"b{blk}c1b", b1),
                    (f"b{blk}c2w", w2), (f"b{blk}c2b", b2)]
    tensors += [("fc1w", sd["head.1.weight"]), ("fc1b", sd["head.1.bias"]),
                ("fc2w", sd["head.4.weight"]), ("fc2b", sd["head.4.bias"])]

    import librosa
    fb = librosa.filters.mel(sr=SR, n_fft=1024, n_mels=N_MELS)
    tensors.append(("melfb", torch.tensor(fb)))

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "wb") as f:
        f.write(b"SYNF")
        f.write(struct.pack("<II", 1, len(tensors)))
        for name, t in tensors:
            write_tensor(f, name, t)
    print(f"exported {len(tensors)} tensors -> {a.out} "
          f"({os.path.getsize(a.out)/1e6:.2f} MB)")


if __name__ == "__main__":
    import sys
    main()
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0)   # dodge ROCm-on-Windows teardown deadlock (see train.py)
