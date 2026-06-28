"""
ARCHIVE — synfection engine v1 (15-param subtractive/additive, saw<->square morph).

Superseded by synth.py (v2: wavetable oscillator + drive). Kept for reference /
reproducibility. NOT imported by the current pipeline. To run v1 again, point the
training/match scripts at this module (it exposes the same API: render, melspec,
denorm, to_wav, N_PARAMS, PARAM_NAMES). v1's trained weights were overwritten by
the v2 retrain; regenerate by training against this engine.

Difference vs v2:
- oscillator timbre = saw<->square blend (osc1_wave/osc2_wave) instead of an
  8-frame wavetable morph.
- no drive/waveshaper knob.
- 15 params total (v2 has 16).
"""

import numpy as np
import torch

SR = 22050
DUR = 1.2
N = int(SR * DUR)
NOTE_DUR = 0.85
N_MELS = 64
MAX_HARM = 72

PARAMS = [
    ("osc1_wave",   0.0,    1.0,    False),  # 0 saw -> 1 square
    ("osc2_wave",   0.0,    1.0,    False),
    ("osc2_detune", -50.0,  50.0,   False),  # cents
    ("osc_mix",     0.0,    1.0,    False),
    ("sub_level",   0.0,    1.0,    False),
    ("noise_level", 0.0,    0.6,    False),
    ("cutoff",      60.0,   10000., True),
    ("reso",        0.6,    9.0,    False),
    ("filt_env",    0.0,    1.0,    False),
    ("filt_a",      0.001,  0.4,    True),
    ("filt_d",      0.02,   0.7,    True),
    ("amp_a",       0.001,  0.4,    True),
    ("amp_d",       0.02,   0.7,    True),
    ("amp_s",       0.0,    1.0,    False),
    ("amp_r",       0.02,   0.7,    True),
]
PARAM_NAMES = [p[0] for p in PARAMS]
N_PARAMS = len(PARAMS)

_LO  = torch.tensor([p[1] for p in PARAMS], dtype=torch.float32)
_HI  = torch.tensor([p[2] for p in PARAMS], dtype=torch.float32)
_LOG = torch.tensor([p[3] for p in PARAMS], dtype=torch.bool)


def denorm(g):
    lo, hi, lg = _LO.to(g.device), _HI.to(g.device), _LOG.to(g.device)
    g = g.clamp(0, 1)
    lin = lo + (hi - lo) * g
    safe_lo = torch.where(lg, lo, torch.ones_like(lo))
    safe_hi = torch.where(lg, hi, torch.ones_like(hi))
    log = safe_lo * (safe_hi / safe_lo) ** g
    return torch.where(lg, log, lin)


def midi_to_hz(m):
    return 440.0 * 2.0 ** ((m - 69.0) / 12.0)


def _adsr(a, d, s, r, T, sr, note_dur):
    dev = a.device
    t = torch.arange(T, device=dev).float().unsqueeze(0)
    a_n = (a * sr).clamp(min=1).unsqueeze(1)
    d_n = (d * sr).clamp(min=1).unsqueeze(1)
    r_n = (r * sr).clamp(min=1).unsqueeze(1)
    s = s.unsqueeze(1)
    note_off = min(int(note_dur * sr), T - 1)
    att = t / a_n
    dec = 1.0 + (s - 1.0) * (t - a_n) / d_n
    e_on = torch.where(t < a_n, att, torch.where(t < a_n + d_n, dec, s)).clamp(0, 1)
    level = e_on[:, note_off:note_off + 1]
    rt = t - note_off
    rel = (level * (1.0 - rt / r_n)).clamp(min=0.0)
    return torch.where(t < note_off, e_on, rel).clamp(0, 1)


def _filter_mag(f, fc, Q):
    x = f / fc.clamp(min=1e-3)
    mag = 1.0 / torch.sqrt((1.0 - x * x) ** 2 + (x / Q) ** 2 + 1e-9)
    return mag.clamp(max=12.0)


def _noise(B, T, cutoff, Q, sr, dev, gen=None):
    w = torch.randn(B, T, device=dev, generator=gen)
    W = torch.fft.rfft(w, dim=1)
    freqs = torch.fft.rfftfreq(T, d=1.0 / sr).to(dev).unsqueeze(0)
    fc = cutoff.unsqueeze(1)
    mag = _filter_mag(freqs, fc, Q.unsqueeze(1))
    out = torch.fft.irfft(W * mag, n=T, dim=1)
    return out / (out.abs().amax(dim=1, keepdim=True) + 1e-6)


def render(genome, midi_note, sr=SR, n=N, note_dur=NOTE_DUR, gen=None):
    single = genome.dim() == 1
    if single:
        genome = genome.unsqueeze(0)
    B = genome.shape[0]
    dev = genome.device
    p = denorm(genome)
    g = {name: p[:, i] for i, name in enumerate(PARAM_NAMES)}

    if not torch.is_tensor(midi_note):
        midi_note = torch.full((B,), float(midi_note), device=dev)
    midi_note = midi_note.to(dev).float()
    f0 = midi_to_hz(midi_note)
    nyq = sr * 0.49

    t = torch.arange(n, device=dev).float() / sr
    amp = _adsr(g["amp_a"], g["amp_d"], g["amp_s"], g["amp_r"], n, sr, note_dur)

    base = g["cutoff"]
    top = base + g["filt_env"] * (nyq - base)
    fe = _adsr(g["filt_a"], g["filt_d"], torch.full_like(base, 0.25),
               g["amp_r"], n, sr, note_dur)
    cutoff_curve = base.unsqueeze(1) + (top - base).unsqueeze(1) * fe
    Q = g["reso"]

    def osc(fund, wave):
        out = torch.zeros(B, n, device=dev)
        phase = 2.0 * np.pi * fund.unsqueeze(1) * t.unsqueeze(0)
        for k in range(1, MAX_HARM + 1):
            fk = fund * k
            saw = 1.0 / k
            sq = (1.0 / k) if (k % 2 == 1) else 0.0
            prof = (1.0 - wave) * saw + wave * sq
            aa = torch.sigmoid((nyq - fk) / (0.02 * nyq + 1.0))
            mag = _filter_mag(fk.unsqueeze(1), cutoff_curve, Q.unsqueeze(1))
            amp_k = (prof * aa).unsqueeze(1) * mag
            out = out + amp_k * torch.sin(k * phase)
        return out

    o1 = osc(f0, g["osc1_wave"])
    f2 = f0 * 2.0 ** (g["osc2_detune"] / 1200.0)
    o2 = osc(f2, g["osc2_wave"])
    sig = (1.0 - g["osc_mix"]).unsqueeze(1) * o1 + g["osc_mix"].unsqueeze(1) * o2

    fsub = f0 / 2.0
    sub = torch.sin(2.0 * np.pi * fsub.unsqueeze(1) * t.unsqueeze(0))
    sub = sub * _filter_mag(fsub.unsqueeze(1), cutoff_curve, Q.unsqueeze(1))
    sig = sig + 0.6 * g["sub_level"].unsqueeze(1) * sub

    nz = _noise(B, n, top, Q, sr, dev, gen)
    sig = sig + g["noise_level"].unsqueeze(1) * nz

    sig = sig * amp
    sig = sig / (sig.abs().amax(dim=1, keepdim=True) + 1e-6) * 0.9
    return sig[0] if single else sig


_MEL_FB = None
_HANN = {}

def _mel_fb(dev):
    global _MEL_FB
    if _MEL_FB is None or _MEL_FB.device != dev:
        import librosa
        fb = librosa.filters.mel(sr=SR, n_fft=1024, n_mels=N_MELS)
        _MEL_FB = torch.tensor(fb, dtype=torch.float32, device=dev)
    return _MEL_FB

def _hann(nfft, dev):
    key = (nfft, dev)
    if key not in _HANN:
        _HANN[key] = torch.hann_window(nfft, device=dev)
    return _HANN[key]

def melspec(audio):
    single = audio.dim() == 1
    if single:
        audio = audio.unsqueeze(0)
    dev = audio.device
    S = torch.stft(audio, 1024, 512, window=_hann(1024, dev),
                   return_complex=True, center=True).abs() ** 2
    mel = torch.matmul(_mel_fb(dev), S)
    mel = 10.0 * torch.log10(mel + 1e-6)
    mel = (mel - mel.amax(dim=(1, 2), keepdim=True) + 80.0) / 80.0
    return mel.clamp(0, 1)[0] if single else mel.clamp(0, 1)


def to_wav(audio):
    return audio.detach().cpu().numpy().astype(np.float32)
