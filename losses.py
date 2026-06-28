"""Multi-scale STFT loss (DDSP-style) — the perceptual objective."""

import torch

_W = {}

def _win(n, dev):
    k = (n, dev)
    if k not in _W:
        _W[k] = torch.hann_window(n, device=dev)
    return _W[k]


def multiscale_stft(x, y, ffts=(2048, 1024, 512, 256, 128), log_w=1.0):
    """Sum of L1 magnitude + L1 log-magnitude over several FFT sizes.
    x, y: (B,T) audio. Symmetric, differentiable."""
    total = 0.0
    for nf in ffts:
        hop = nf // 4
        w = _win(nf, x.device)
        X = torch.stft(x, nf, hop, window=w, return_complex=True, center=True).abs()
        Y = torch.stft(y, nf, hop, window=w, return_complex=True, center=True).abs()
        lin = (X - Y).abs().mean()
        log = (torch.log(X + 1e-5) - torch.log(Y + 1e-5)).abs().mean()
        total = total + lin + log_w * log
    return total / len(ffts)
