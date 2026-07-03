"""
augment.py — real-world coloration for synth renders (domain-gap closure).

GenoNet trains on its own pristine renders, so real wavs — with their EQ tilt,
noise floor, limiter glue, and a little room — are out-of-distribution. This
colors the audio the net *hears* (the mel input). The spectral-loss target
stays clean, so the net still learns honest genomes: hear through the grime,
predict the underlying patch.

Effects, each applied per-item with probability p:
  tilt    spectral tilt ±3 dB/oct around 500 Hz, plus low/high shelves ±4 dB
  noise   broadband floor at -50..-30 dB relative to the render
  glue    gentle tanh saturation (limiter/codec flavor)
  room    short exponential-decay noise IR (15-60 ms), mixed low

Usage (train.py):
    tmel = melspec(augment(target, gen, p=a.augment))   # input side only
    loss = ... multiscale_stft(recon, target)           # target stays clean
"""

import torch

SR = 22050.0


def _rand(B, lo, hi, dev, gen):
    return lo + (hi - lo) * torch.rand(B, device=dev, generator=gen)


def _mask(B, p, dev, gen):
    return (torch.rand(B, device=dev, generator=gen) < p).float()


def augment(x, gen, p=0.7):
    """x: [B, N] rendered audio -> colored copy, same shape. No grad needed."""
    B, N = x.shape
    dev = x.device
    with torch.no_grad():
        peak0 = x.abs().amax(dim=1, keepdim=True) + 1e-9

        # --- frequency shaping: one FFT round trip -----------------------
        X = torch.fft.rfft(x)
        f = torch.fft.rfftfreq(N, 1.0 / SR, device=dev).clamp(min=20.0)
        oct_from_mid = torch.log2(f / 500.0)  # [F]
        tilt_db = (_rand(B, -3.0, 3.0, dev, gen) * _mask(B, p, dev, gen))[:, None] * oct_from_mid
        lo_shelf = (_rand(B, -4.0, 4.0, dev, gen) * _mask(B, p, dev, gen))[:, None] \
            * torch.sigmoid((120.0 - f) / 30.0)
        hi_shelf = (_rand(B, -4.0, 4.0, dev, gen) * _mask(B, p, dev, gen))[:, None] \
            * torch.sigmoid((f - 4000.0) / 800.0)
        gain = 10.0 ** ((tilt_db + lo_shelf + hi_shelf) / 20.0)
        y = torch.fft.irfft(X * gain, n=N)

        # --- room: convolve with a short exponential-decay noise IR ------
        ir_len = int(0.06 * SR)
        t = torch.arange(ir_len, device=dev, dtype=torch.float32)
        decay = torch.exp(-t[None, :] / (_rand(B, 0.015, 0.06, dev, gen)[:, None] * SR))
        ir = torch.randn(B, ir_len, device=dev, generator=gen) * decay
        ir = ir / (ir.abs().amax(dim=1, keepdim=True) + 1e-9)
        wet = torch.fft.irfft(
            torch.fft.rfft(y, n=N + ir_len) * torch.fft.rfft(ir, n=N + ir_len), n=N + ir_len
        )[:, :N]
        mix = (_rand(B, 0.05, 0.20, dev, gen) * _mask(B, p * 0.6, dev, gen))[:, None]
        y = (1.0 - mix) * y + mix * wet

        # --- glue: gentle saturation --------------------------------------
        drive = 1.0 + (_rand(B, 0.2, 1.5, dev, gen) * _mask(B, p * 0.5, dev, gen))[:, None]
        y = torch.tanh(y * drive) / torch.tanh(drive)

        # --- noise floor ---------------------------------------------------
        nz_db = _rand(B, -50.0, -30.0, dev, gen)
        nz = torch.randn(B, N, device=dev, generator=gen) \
            * (10.0 ** (nz_db / 20.0) * _mask(B, p, dev, gen))[:, None]
        y = y + nz * peak0

        # restore the original per-item peak so level itself isn't a tell
        y = y / (y.abs().amax(dim=1, keepdim=True) + 1e-9) * peak0
    return y
