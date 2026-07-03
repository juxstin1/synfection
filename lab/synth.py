"""
synfection — differentiable synth engine (the "genome"), pure PyTorch.

One GPU source of truth used everywhere: dataset rendering, on-the-fly training
targets, AND gradient-based patch refinement (analysis-by-synthesis). Because the
whole engine is differentiable, we can backprop a multi-scale spectral loss from
the rendered audio all the way to the predicted parameters — the DDSP recipe, and
the recipe that makes matches perceptually right
rather than merely number-matched ones.

Engine: two band-limited harmonic oscillators (saw<->square morph) + sub sine +
spectrally-shaped noise, through a time-varying resonant lowpass (analog 2-pole
magnitude, evaluated per-harmonic so it's fully differentiable and vectorized),
with independent ADSR amp + filter envelopes.

A genome is a vector in [0,1]^N_PARAMS. See PARAMS for the layout.
"""

import numpy as np
import torch

SR = 22050
DUR = 1.2
N = int(SR * DUR)
NOTE_DUR = 0.85
N_MELS = 64
MAX_HARM = 72            # harmonics per oscillator

# (name, min, max, log?)  — genome stored normalized [0,1], mapped to these.
PARAMS = [
    ("osc1_wt",     0.0,    1.0,    False),  # wavetable morph position (Serum-style)
    ("osc2_wt",     0.0,    1.0,    False),
    ("osc2_detune", -50.0,  50.0,   False),  # cents
    ("osc_mix",     0.0,    1.0,    False),  # 0 osc1 -> 1 osc2
    ("sub_level",   0.0,    1.0,    False),  # sine one octave down
    ("noise_level", 0.0,    0.6,    False),  # filtered noise
    ("drive",       0.0,    1.0,    False),  # waveshaper / saturation amount
    ("cutoff",      60.0,   10000., True),   # lowpass cutoff (Hz)
    ("reso",        0.6,    9.0,    False),  # filter Q (resonance)
    ("filt_env",    0.0,    1.0,    False),  # filter-env amount
    ("filt_a",      0.001,  0.4,    True),   # filter-env attack
    ("filt_d",      0.02,   0.7,    True),   # filter-env decay
    ("amp_a",       0.001,  0.4,    True),   # amp attack
    ("amp_d",       0.02,   0.7,    True),   # amp decay
    ("amp_s",       0.0,    1.0,    False),  # amp sustain
    ("amp_r",       0.02,   0.7,    True),   # amp release
    ("pitch_env",   0.0,    48.0,   False),  # start offset above the note (semitones)
    ("pitch_dec",   0.005,  0.4,    True),   # pitch-env decay (s)
    ("lfo_rate",    0.05,   12.0,   True),   # cutoff LFO (Hz)
    ("lfo_depth",   0.0,    2.0,    False),  # cutoff LFO depth (± octaves)
]
PARAM_NAMES = [p[0] for p in PARAMS]
N_PARAMS = len(PARAMS)

_LO  = torch.tensor([p[1] for p in PARAMS], dtype=torch.float32)
_HI  = torch.tensor([p[2] for p in PARAMS], dtype=torch.float32)
_LOG = torch.tensor([p[3] for p in PARAMS], dtype=torch.bool)


def _build_wavetable(H=MAX_HARM):
    """Harmonic-amplitude wavetable bank: F frames x H harmonics, each frame an
    L1-normalized magnitude spectrum. A morph position in [0,1] interpolates
    across frames (sine -> triangle -> saw -> square -> pulses -> formant ->
    rich), giving Serum-style wavetable timbres while staying band-limited and
    differentiable. Phase is ignored (the matcher's loss is magnitude-STFT)."""
    k = np.arange(1, H + 1, dtype=np.float64)
    frames = [
        (k == 1).astype(float),                              # sine
        np.where(k % 2 == 1, 1.0 / k**2, 0.0),               # triangle
        np.where(k % 2 == 1, 1.0 / k, 0.0),                  # square
        1.0 / k,                                             # saw
        np.abs(np.sin(np.pi * k * 0.25)) * 2.0 / (np.pi * k),  # 25% pulse
        np.abs(np.sin(np.pi * k * 0.10)) * 2.0 / (np.pi * k),  # narrow pulse
        np.exp(-((k - 5.0) ** 2) / 8.0) + 0.3 * np.exp(-((k - 12.0) ** 2) / 20.0),  # formant
        1.0 / np.sqrt(k),                                    # rich / bright
    ]
    wt = np.stack(frames, 0)
    wt /= (np.abs(wt).sum(1, keepdims=True) + 1e-9)          # L1-normalize each frame
    return torch.tensor(wt, dtype=torch.float32)

_WT = _build_wavetable()         # (F, H)
N_WT_FRAMES = _WT.shape[0]


def _wt_profile(pos, dev):
    """pos: (B,) in [0,1] -> harmonic-amplitude profile (B, H), morphed across frames."""
    wt = _WT.to(dev)
    f = pos.clamp(0, 1) * (N_WT_FRAMES - 1)
    lo = f.floor().long().clamp(0, N_WT_FRAMES - 2)          # (B,)
    frac = (f - lo.float()).unsqueeze(1)                     # (B,1)
    return (1 - frac) * wt[lo] + frac * wt[lo + 1]           # (B, H)


def denorm(g):
    """Genome (...,N_PARAMS) in [0,1] -> real values (same shape), differentiable.

    The log branch is evaluated for every column, so its base must stay positive
    even for linear params (lo can be 0 or negative there) — otherwise inf/NaN in
    the unused branch poisons gradients via where (0 * NaN = NaN). We feed the log
    branch safe 1.0s on linear columns; where() then discards them cleanly."""
    lo, hi, lg = _LO.to(g.device), _HI.to(g.device), _LOG.to(g.device)
    g = g.clamp(0, 1)
    lin = lo + (hi - lo) * g
    safe_lo = torch.where(lg, lo, torch.ones_like(lo))
    safe_hi = torch.where(lg, hi, torch.ones_like(hi))
    log = safe_lo * (safe_hi / safe_lo) ** g
    return torch.where(lg, log, lin)


DRIVE_IDX = PARAM_NAMES.index("drive")

V3_NEUTRAL = [0.0, 0.5, 0.4, 0.0]  # pitch_env, pitch_dec, lfo_rate, lfo_depth


def upgrade_genome(g):
    """Migrate any schema to current: v1 (15) inserts neutral drive=0, v2 (16)
    pads neutral pitch-env/LFO, v3 (20) passes through. Accepts np array or
    1-d torch tensor; returns the same type."""
    if len(g) == N_PARAMS:
        return g
    if len(g) == 15:  # v1 -> v2: insert drive
        if torch.is_tensor(g):
            zero = torch.zeros(1, dtype=g.dtype, device=g.device)
            g = torch.cat([g[:DRIVE_IDX], zero, g[DRIVE_IDX:]])
        else:
            g = np.insert(np.asarray(g), DRIVE_IDX, 0.0)
    if len(g) == 16:  # v2 -> v3: pad neutral pitch-env/LFO
        if torch.is_tensor(g):
            pad = torch.tensor(V3_NEUTRAL, dtype=g.dtype, device=g.device)
            return torch.cat([g, pad])
        return np.concatenate([np.asarray(g), np.array(V3_NEUTRAL, dtype=np.float32)])
    raise ValueError(f"genome has {len(g)} params, expected {N_PARAMS} (or 15/16 legacy)")


def midi_to_hz(m):
    return 440.0 * 2.0 ** ((m - 69.0) / 12.0)


def _adsr(a, d, s, r, T, sr, note_dur):
    """Vectorized ADSR. a,d,s,r,: (B,) tensors. -> (B,T)."""
    dev = a.device
    t = torch.arange(T, device=dev).float().unsqueeze(0)          # (1,T)
    a_n = (a * sr).clamp(min=1).unsqueeze(1)                       # (B,1)
    d_n = (d * sr).clamp(min=1).unsqueeze(1)
    r_n = (r * sr).clamp(min=1).unsqueeze(1)
    s = s.unsqueeze(1)
    note_off = min(int(note_dur * sr), T - 1)

    att = t / a_n
    dec = 1.0 + (s - 1.0) * (t - a_n) / d_n
    e_on = torch.where(t < a_n, att, torch.where(t < a_n + d_n, dec, s))
    e_on = e_on.clamp(0, 1)
    level = e_on[:, note_off:note_off + 1]                        # (B,1)
    rt = t - note_off
    rel = (level * (1.0 - rt / r_n)).clamp(min=0.0)
    env = torch.where(t < note_off, e_on, rel)
    return env.clamp(0, 1)


def _filter_mag(f, fc, Q):
    """Analog 2-pole lowpass magnitude at signal freq f, cutoff fc, resonance Q.
    Differentiable. Broadcasts. Peaks ~Q near f==fc, rolls off above."""
    x = f / fc.clamp(min=1e-3)
    mag = 1.0 / torch.sqrt((1.0 - x * x) ** 2 + (x / Q) ** 2 + 1e-9)
    return mag.clamp(max=12.0)


def _noise(B, T, cutoff, Q, sr, dev, gen=None):
    """Spectrally-shaped noise via rFFT (differentiable wrt cutoff/Q)."""
    w = torch.randn(B, T, device=dev, generator=gen)
    W = torch.fft.rfft(w, dim=1)
    freqs = torch.fft.rfftfreq(T, d=1.0 / sr).to(dev).unsqueeze(0)   # (1,F)
    # use the *peak* (end-of-attack) cutoff as a static shape for noise
    fc = cutoff.unsqueeze(1)                                          # (B,1)
    mag = _filter_mag(freqs, fc, Q.unsqueeze(1))
    out = torch.fft.irfft(W * mag, n=T, dim=1)
    return out / (out.abs().amax(dim=1, keepdim=True) + 1e-6)


def render(genome, midi_note, sr=SR, n=N, note_dur=NOTE_DUR, gen=None):
    """Genome(s) + MIDI note(s) -> audio (B,T) in [-1,1]. Differentiable.

    genome: (N_PARAMS,) or (B,N_PARAMS) tensor in [0,1].
    midi_note: int / float / (B,) tensor.
    """
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
    f0 = midi_to_hz(midi_note)                                       # (B,)
    nyq = sr * 0.49

    t = torch.arange(n, device=dev).float() / sr                    # (T,)
    amp = _adsr(g["amp_a"], g["amp_d"], g["amp_s"], g["amp_r"], n, sr, note_dur)

    base = g["cutoff"]
    top = base + g["filt_env"] * (nyq - base)
    fe = _adsr(g["filt_a"], g["filt_d"], torch.full_like(base, 0.25),
               g["amp_r"], n, sr, note_dur)
    cutoff_curve = base.unsqueeze(1) + (top - base).unsqueeze(1) * fe  # (B,T)
    lfo = torch.sin(2.0 * np.pi * g["lfo_rate"].unsqueeze(1) * t.unsqueeze(0))
    cutoff_curve = (cutoff_curve * 2.0 ** (g["lfo_depth"].unsqueeze(1) * lfo)).clamp(20.0, nyq)
    Q = g["reso"]

    # pitch envelope: start `pitch_env` semitones above the note, decay to it.
    # Phase = exclusive prefix-sum of the swept fundamental (f64 cumsum) —
    # exactly the old static phase when pitch_env is 0. Per-harmonic filter/
    # anti-alias stay keyed to the target frequency (the sweep is fast).
    ratio = 2.0 ** (
        g["pitch_env"].unsqueeze(1)
        * torch.exp(-t.unsqueeze(0) / g["pitch_dec"].unsqueeze(1)) / 12.0
    )                                                                # (B,T)
    inst = f0.unsqueeze(1).double() * ratio.double()                 # (B,T) Hz
    ph = (torch.cumsum(inst, dim=1) - inst) * (2.0 * np.pi / sr)     # (B,T) f64

    def osc(fund, wt_pos):
        out = torch.zeros(B, n, device=dev)
        phase = (ph * (fund / f0).unsqueeze(1).double()).float()     # (B,T)
        prof_all = _wt_profile(wt_pos, dev)                         # (B,H) wavetable
        for k in range(1, MAX_HARM + 1):
            fk = fund * k                                            # (B,)
            prof = prof_all[:, k - 1]                                # (B,)
            aa = torch.sigmoid((nyq - fk) / (0.02 * nyq + 1.0))      # (B,)
            mag = _filter_mag(fk.unsqueeze(1), cutoff_curve, Q.unsqueeze(1))
            amp_k = (prof * aa).unsqueeze(1) * mag                   # (B,T)
            out = out + amp_k * torch.sin(k * phase)
        return out

    o1 = osc(f0, g["osc1_wt"])
    f2 = f0 * 2.0 ** (g["osc2_detune"] / 1200.0)
    o2 = osc(f2, g["osc2_wt"])
    sig = (1.0 - g["osc_mix"]).unsqueeze(1) * o1 + g["osc_mix"].unsqueeze(1) * o2

    fsub = f0 / 2.0
    sub = torch.sin((ph * 0.5).float())
    sub = sub * _filter_mag(fsub.unsqueeze(1), cutoff_curve, Q.unsqueeze(1))
    sig = sig + 0.6 * g["sub_level"].unsqueeze(1) * sub

    nz = _noise(B, n, top, Q, sr, dev, gen)
    sig = sig + g["noise_level"].unsqueeze(1) * nz

    sig = sig * amp

    # drive / waveshaper: blend clean <-> tanh-saturated (clean at drive=0)
    drv = g["drive"].unsqueeze(1)
    norm = sig.abs().amax(dim=1, keepdim=True) + 1e-6
    clean = sig / norm
    shaped = torch.tanh(clean * 6.0) / np.tanh(6.0)
    sig = (1.0 - drv) * clean + drv * shaped

    sig = sig / (sig.abs().amax(dim=1, keepdim=True) + 1e-6) * 0.9
    return sig[0] if single else sig


# ---- mel features (torch, differentiable; net input) -----------------------

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
    """audio (T,) or (B,T) -> log-mel (B,N_MELS,frames) normalized ~[0,1]."""
    single = audio.dim() == 1
    if single:
        audio = audio.unsqueeze(0)
    dev = audio.device
    S = torch.stft(audio, 1024, 512, window=_hann(1024, dev),
                   return_complex=True, center=True).abs() ** 2     # (B,F,frames)
    mel = torch.matmul(_mel_fb(dev), S)                             # (B,n_mels,frames)
    mel = 10.0 * torch.log10(mel + 1e-6)
    mel = (mel - mel.amax(dim=(1, 2), keepdim=True) + 80.0) / 80.0
    mel = mel.clamp(0, 1)
    return mel[0] if single else mel


def to_wav(audio):
    """torch audio -> float32 numpy for soundfile."""
    return audio.detach().cpu().numpy().astype(np.float32)
