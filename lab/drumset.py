"""
drumset.py — slice a DJ library into labeled drum one-shots for real-data training.

Full mixes are full of bleed, so this is ruthless about quality: a hit is kept
only if it has a clean attack (quiet run-up), decays fast (percussive), and
doesn't collide with the next onset. Survivors are classified kick / snare /
hat by band-energy shares, deduped per track (the same kick repeats hundreds of
times per tune), peak-normalized, faded, and written with a manifest.

    python drumset.py --library "C:/Users/juxxs/music/tracks" --out drums/oneshots --limit 20
    python drumset.py --library "C:/Users/juxxs/music/tracks" --out drums/oneshots

Output: <out>/<label>_NNNN.wav (22050 Hz mono, <=0.6 s) + <out>/manifest.jsonl
with {file, label, track, t, sub, mid, high, centroid}. Nothing here is ever
committed — it's training-local (see .gitignore).
"""

import argparse
import json
import os
import sys

import numpy as np
import librosa
import soundfile as sf

SR = 22050
HIT_LEN = int(0.6 * SR)
HEAD = int(0.120 * SR)  # classification window: the attack + early body


def band_shares(hit):
    """Energy shares of the first 120 ms in sub / mid / high bands + centroid."""
    head = hit[:HEAD]
    spec = np.abs(np.fft.rfft(head * np.hanning(len(head)))) ** 2
    freqs = np.fft.rfftfreq(len(head), 1 / SR)
    total = spec.sum() + 1e-12
    sub = spec[freqs < 120].sum() / total
    mid = spec[(freqs >= 400) & (freqs < 3000)].sum() / total
    high = spec[freqs >= 4000].sum() / total
    centroid = float((spec * freqs).sum() / total)
    return float(sub), float(mid), float(high), centroid


def classify(sub, mid, high, centroid):
    if sub > 0.45 and centroid < 600:
        return "kick"
    if high > 0.45 and centroid > 3500:
        return "hat"
    if mid > 0.30 and 500 < centroid < 3500:
        return "snare"
    return None  # ambiguous — not worth training on


def clean_hits(y, onsets_s):
    """Yield (t, hit) for onsets that are isolated, sharp, and fast-decaying."""
    n = len(y)
    for i, t in enumerate(onsets_s):
        s = int(t * SR)
        if s < int(0.05 * SR) or s + HIT_LEN >= n:
            continue
        gap = (onsets_s[i + 1] - t) if i + 1 < len(onsets_s) else 1.0
        if gap < 0.11:
            continue
        hit = y[s : s + HIT_LEN].copy()
        peak = np.abs(hit).max()
        if peak < 0.05:
            continue
        # clean attack: the 50 ms before the onset must be quiet
        pre = y[s - int(0.05 * SR) : s]
        atk = hit[: int(0.03 * SR)]
        if np.sqrt((pre**2).mean()) > 0.18 * (np.sqrt((atk**2).mean()) + 1e-9):
            continue
        # percussive: body must decay hard by 200-300 ms
        early = np.sqrt((hit[: int(0.05 * SR)] ** 2).mean())
        late = np.sqrt((hit[int(0.20 * SR) : int(0.30 * SR)] ** 2).mean())
        if late > 0.25 * early:
            continue
        yield t, hit


def mel_print(hit):
    """Small fingerprint for near-duplicate detection within a track."""
    m = librosa.feature.melspectrogram(y=hit[:HEAD], sr=SR, n_mels=24, n_fft=512, hop_length=256)
    v = np.log(m + 1e-6).flatten()
    return v / (np.linalg.norm(v) + 1e-9)


def process_track(path, per_class):
    try:
        y, _ = librosa.load(path, sr=SR, mono=True)
    except Exception as e:
        print(f"  ! {os.path.basename(path)}: {e}")
        return []
    if len(y) < SR * 10:
        return []
    onsets = librosa.onset.onset_detect(y=y, sr=SR, units="time", backtrack=True)
    picked = {"kick": [], "snare": [], "hat": []}
    for t, hit in clean_hits(y, onsets):
        sub, mid, high, centroid = band_shares(hit)
        label = classify(sub, mid, high, centroid)
        if label is None:
            continue
        fp = mel_print(hit)
        # skip near-duplicates of hits we already kept from this track
        if any(float(fp @ prev_fp) > 0.93 for _, prev_fp, *_ in picked[label]):
            continue
        picked[label].append((t, fp, hit, sub, mid, high, centroid))
        # keep collecting a few extra, trim to the strongest below
        if all(len(v) >= per_class * 2 for v in picked.values()):
            break
    out = []
    for label, hits in picked.items():
        # prefer the punchiest (highest early RMS) of the deduped survivors
        hits.sort(key=lambda h: -np.sqrt((h[2][: int(0.05 * SR)] ** 2).mean()))
        for t, _, hit, sub, mid, high, centroid in hits[:per_class]:
            peak = np.abs(hit).max() + 1e-9
            hit = hit / peak * 0.9
            # tail suppression: fade where the hit's own envelope dies, so
            # low-level music bleed doesn't ride the whole 0.6 s window
            win = int(0.02 * SR)
            env = np.sqrt(np.convolve(hit**2, np.ones(win) / win, mode="same"))
            floor = 0.15 * env.max()
            past = np.where(env[int(0.08 * SR):] < floor)[0]
            if len(past):
                cut = int(0.08 * SR) + past[0]
                fade_n = min(int(0.03 * SR), len(hit) - cut)
                hit[cut : cut + fade_n] *= np.linspace(1, 0, fade_n)
                hit[cut + fade_n :] = 0.0
            fade = int(0.003 * SR)
            hit[:fade] *= np.linspace(0, 1, fade)
            hit[-fade:] *= np.linspace(1, 0, fade)
            out.append((label, t, hit, sub, mid, high, centroid))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--library", required=True)
    ap.add_argument("--out", default="drums/oneshots")
    ap.add_argument("--per-class", type=int, default=3, help="max hits per class per track")
    ap.add_argument("--limit", type=int, default=0, help="only the first N tracks (0 = all)")
    a = ap.parse_args()

    exts = (".wav", ".mp3")
    tracks = sorted(
        os.path.join(a.library, f) for f in os.listdir(a.library) if f.lower().endswith(exts)
    )
    if a.limit:
        tracks = tracks[: a.limit]
    os.makedirs(a.out, exist_ok=True)

    counts = {"kick": 0, "snare": 0, "hat": 0}
    manifest = open(os.path.join(a.out, "manifest.jsonl"), "w", encoding="utf-8")
    for ti, path in enumerate(tracks):
        got = process_track(path, a.per_class)
        for label, t, hit, sub, mid, high, centroid in got:
            fname = f"{label}_{counts[label]:04d}.wav"
            sf.write(os.path.join(a.out, fname), hit, SR)
            manifest.write(
                json.dumps(
                    dict(
                        file=fname, label=label, track=os.path.basename(path),
                        t=round(float(t), 2), sub=round(sub, 3), mid=round(mid, 3),
                        high=round(high, 3), centroid=round(centroid, 1),
                    )
                )
                + "\n"
            )
            counts[label] += 1
        if (ti + 1) % 10 == 0:
            manifest.flush()
            print(f"[{ti + 1}/{len(tracks)}] kicks {counts['kick']}  snares {counts['snare']}  hats {counts['hat']}", flush=True)
    manifest.close()
    print(f"done: {counts}  -> {a.out}/", flush=True)


if __name__ == "__main__":
    main()
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0)
