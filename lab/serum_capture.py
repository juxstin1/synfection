"""Headless Serum capture -> Synfection match -> local audition gallery.

Run with .venv-serum/Scripts/python.exe lab/serum_capture.py.
Each native plugin render runs in a bounded, hidden child process.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 1


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, data):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')
    temp.replace(path)


def run_child(command, log, timeout, done=None):
    """Run a hidden child. `done` is a marker file the child writes last; when
    it exists, a nonzero exit (Serum's teardown crash) is tolerated."""
    with Path(log).open('w', encoding='utf-8') as output:
        result = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT,
                                timeout=timeout, cwd=ROOT,
                                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    if result.returncode and not (done and Path(done).is_file()):
        raise RuntimeError(f'Child exited {result.returncode}; see {log}')
    return result.returncode


def audio_stats(audio):
    import numpy as np
    if audio.size == 0 or not np.isfinite(audio).all():
        raise ValueError('Empty or non-finite audio')
    peak = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    if peak < 3.16e-5:
        raise ValueError('Silent audio: peak below -90 dBFS')
    return {'peak': peak, 'rms_dbfs': 20 * float(np.log10(max(rms, 1e-12))),
            'samples_at_or_above_full_scale': int(np.count_nonzero(np.abs(audio) >= 1))}


def preview(audio):
    """Loudness-match to -20 dBFS RMS (peak capped at -2 dBFS, boost capped at
    +24 dB) so Serum and Synfection A/B at equal level. Raw audio kept separately."""
    import numpy as np
    stats = audio_stats(audio)
    rms = 10 ** (stats['rms_dbfs'] / 20)
    gain = min(15.85, 0.79 / stats['peak'], 0.1 / max(rms, 1e-12))
    result = audio * gain
    fade = min(220, len(result) // 2)
    if fade:
        ramp = np.linspace(0, 1, fade)
        if result.ndim == 2:
            ramp = ramp[:, None]
        result[:fade] *= ramp
        result[-fade:] *= ramp[::-1]
    return result


def render_worker(job_path):
    # Native host must load before other audio libraries.
    print('worker: importing renderer', flush=True)
    from serum_render import RenderConfig, Renderer
    job = json.loads(Path(job_path).read_text(encoding='utf-8'))
    config = RenderConfig(serum2_plugin_path=job['plugin'], sample_rate=44100,
                          note=job['note'], velocity=job['velocity'],
                          duration=job['duration'], tail=job['tail'])
    print('worker: loading plugin and warming up', flush=True)
    # Retain native objects until process exit; calling __exit__/destructors
    # triggers a native access violation on the verified Windows combination.
    renderer = Renderer(config)
    renderer.__enter__()
    print('worker: loading preset and rendering', flush=True)
    audio = renderer.render(job['preset']).T.copy()
    print('worker: audio rendered', flush=True)
    import soundfile as sf
    stats = audio_stats(audio)
    folder = Path(job_path).parent
    sf.write(folder / 'serum.wav', audio, 44100, subtype='FLOAT')
    sf.write(folder / 'serum-preview.wav', preview(audio), 44100, subtype='PCM_16')
    atomic_json(folder / 'render.json', stats)
    print('worker: files written', flush=True)
    hard_exit()


def hard_exit():
    # os._exit still runs ExitProcess -> DLL_PROCESS_DETACH, where Serum 2.1.5
    # access-violates. TerminateProcess skips DLL teardown entirely.
    sys.stdout.flush()
    sys.stderr.flush()
    if os.name == 'nt':
        import ctypes
        kernel32 = ctypes.windll.kernel32
        kernel32.TerminateProcess(kernel32.GetCurrentProcess(), 0)
    os._exit(0)


def choose_presets(root, categories, limit):
    """Round-robin categories; hash order avoids selecting only alphabetic 808s."""
    groups = []
    for category in categories:
        files = list((root / category).rglob('*.SerumPreset'))
        files.sort(key=lambda p: hashlib.sha256(p.relative_to(root).as_posix().encode()).hexdigest())
        groups.append(files)
    selected = []
    while any(groups) and len(selected) < limit:
        for group in groups:
            if group and len(selected) < limit:
                selected.append(group.pop(0))
    return selected


def cached(record, folder, fingerprint):
    return (record.get('status') == 'ok' and record.get('fingerprint') == fingerprint
            and bool(record.get('artifacts'))
            and all((folder / name).is_file() and digest(folder / name) == sha
                    for name, sha in record['artifacts'].items()))


def gallery(out, records):
    cards = []
    for row in records:
        title = html.escape(row['name'])
        ident = row['id']
        if row['status'] != 'ok':
            cards.append(f'<article><h2>{title}</h2><p>Failed: {html.escape(row.get("error", ""))}</p></article>')
            continue
        cards.append(f'''<article data-id="{ident}"><p class="tag">{html.escape(row['category'])} · MIDI {row['note']}</p>
<h2>{title}</h2><label>Serum original<audio controls preload="none" src="{ident}/serum-preview.wav"></audio></label>
<label>Synfection recreation<audio controls preload="none" src="{ident}/match-preview.wav"></audio></label>
<p><a href="{ident}/match.wav.genome.txt" download>Save playable patch</a></p>
<label>Keep this direction? <select><option value="">Unrated</option><option value="1">Skip</option><option value="3">Interesting</option><option value="5">More like this</option></select></label>
<textarea placeholder="Optional: more bite, darker, less buzzy…" aria-label="Listening notes"></textarea></article>''')
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Synfection — Serum listening room</title><style>
body{background:#111714;color:#e7f1e9;font:17px system-ui;margin:0;padding:5vw;max-width:1200px}h1{font-size:clamp(32px,5vw,64px);margin:.2em 0}p{line-height:1.6;color:#b4c8bb}.tag{color:#a9d787;font-size:13px;text-transform:uppercase}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(290px,1fr));gap:24px}article{border:1px solid #405548;padding:24px;border-radius:12px}h2{font-size:22px}audio{display:block;width:100%;margin:10px 0 22px}select,textarea,button{background:#26392d;color:#edf4ef;border:1px solid #5b7864;padding:10px;border-radius:5px}textarea{display:block;box-sizing:border-box;width:100%;margin-top:14px}a{color:#b0e88f}button{cursor:pointer;margin:0 0 28px}
</style><p class="tag">Synfection / Serum sessions</p><h1>Find the sounds worth growing.</h1>
<p>Original Serum audio and a recreation using Synfection’s smaller synth. Previews are loudness-matched to each other; start with your volume low. Ratings stay in this browser. Nothing plays automatically.</p>
<button id="export">Export listening notes</button><main>''' + ''.join(cards) + '''</main><script>
const key='synfection-serum-ratings-v1';let ratings={};try{ratings=JSON.parse(localStorage.getItem(key)||'{}')}catch{}
document.querySelectorAll('article[data-id]').forEach(card=>{const id=card.dataset.id,s=card.querySelector('select'),t=card.querySelector('textarea');s.value=ratings[id]?.rating||'';t.value=ratings[id]?.notes||'';const save=()=>{ratings[id]={rating:s.value,notes:t.value,name:card.querySelector('h2').textContent};try{localStorage.setItem(key,JSON.stringify(ratings))}catch{}};s.onchange=save;t.oninput=save});
document.addEventListener('play',e=>{document.querySelectorAll('audio').forEach(a=>{if(a!==e.target)a.pause()})},true);
document.querySelector('#export').onclick=()=>{const u=URL.createObjectURL(new Blob([JSON.stringify(ratings,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=u;a.download='serum-listening-notes.json';a.click();setTimeout(()=>URL.revokeObjectURL(u),1000)};
</script></html>'''
    (out / 'index.html').write_text(page, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--presets', type=Path, default=Path.home() / 'Documents/Xfer/Serum 2 Presets/Presets/Factory')
    parser.add_argument('--plugin', type=Path, default=Path('C:/Program Files/Common Files/VST3/Serum2.vst3'))
    parser.add_argument('--out', type=Path, default=ROOT / 'lab/serum-output')
    parser.add_argument('--binary', type=Path, default=ROOT / 'target/release/synfection.exe')
    parser.add_argument('--categories', nargs='+', default=['Bass', 'Pluck', 'Organ', 'Synth', 'Lead'])
    parser.add_argument('--limit', type=int, default=10)
    parser.add_argument('--iters', type=int, default=30)
    parser.add_argument('--timeout', type=int, default=180)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--worker', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        render_worker(args.worker)
        # Serum 2.1.5/DawDreamer 0.9.0 on this Windows host crashes during
        # interpreter shutdown, after all output is closed. This disposable
        # process owns no pending work; let the OS reclaim native resources.
        hard_exit()
    if args.limit < 1 or args.iters < 0 or args.timeout < 1:
        parser.error('limit/timeout must be positive; iters must be nonnegative')
    for category in args.categories:
        if Path(category).name != category or category in ('.', '..'):
            parser.error('categories must be direct subdirectory names')
    presets = choose_presets(args.presets.resolve(), args.categories, args.limit)
    if not presets:
        parser.error('No .SerumPreset files found in the selected categories')
    if args.dry_run:
        print(json.dumps([str(p) for p in presets], indent=2))
        return 0
    if not args.binary.is_file() or not args.plugin.exists():
        parser.error('Need the Serum plugin and built Synfection binary before capture')
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    versions = {p: importlib.metadata.version(p) for p in ['serum-render', 'dawdreamer', 'serum2-preset-loader', 'numpy', 'soundfile']}
    plugin_files = [args.plugin] if args.plugin.is_file() else sorted(args.plugin.rglob('*.vst3'))
    plugin_hashes = {str(p): digest(p) for p in plugin_files if p.is_file()}
    if not plugin_hashes:
        parser.error('No plugin binary found')
    provenance = {'schema': SCHEMA, 'versions': versions, 'plugin_hashes': plugin_hashes,
                  'binary_sha256': digest(args.binary), 'script_sha256': digest(__file__), 'iters': args.iters}
    records = []
    for preset in presets:
        category = preset.relative_to(args.presets.resolve()).parts[0]
        job = dict(provenance, preset=str(preset), preset_sha256=digest(preset),
                   plugin=str(args.plugin.resolve()), note=36 if category == 'Bass' else 48,
                   velocity=100, duration=0.85, tail=0.35)
        fingerprint = hashlib.sha256(json.dumps(job, sort_keys=True).encode()).hexdigest()
        ident = fingerprint[:16]
        folder = args.out / ident
        folder.mkdir(exist_ok=True)
        record_path = folder / 'record.json'
        row = {'id': ident, 'name': preset.stem, 'category': category,
               'note': job['note'], 'fingerprint': fingerprint, 'status': 'pending', 'job': job}
        old = json.loads(record_path.read_text(encoding='utf-8')) if record_path.exists() else {}
        if cached(old, folder, fingerprint):
            row = old
            print(f'SKIP {preset.stem}', flush=True)
        else:
            start = time.monotonic()
            atomic_json(folder / 'job.json', job)
            try:
                print(f'RENDER {preset.stem}', flush=True)
                for stale in ['render.json', 'serum.wav', 'serum-preview.wav']:
                    (folder / stale).unlink(missing_ok=True)
                row['render_exit_code'] = run_child(
                    [sys.executable, '-X', 'faulthandler', '-u', str(Path(__file__).resolve()), '--worker', str(folder / 'job.json')],
                    folder / 'render.log', args.timeout, done=folder / 'render.json')
                run_child([str(args.binary.resolve()), 'match', str(folder / 'serum.wav'),
                           '--note', str(job['note']), '--iters', str(args.iters), '--seed', '0',
                           '--out', str(folder / 'match.wav')], folder / 'match.log', args.timeout)
                import soundfile as sf
                matched, sr = sf.read(folder / 'match.wav', dtype='float32')
                row['match_stats'] = audio_stats(matched)
                row['serum_stats'] = json.loads((folder / 'render.json').read_text())
                sf.write(folder / 'match-preview.wav', preview(matched), sr, subtype='PCM_16')
                values = [float(v) for v in (folder / 'match.wav.genome.txt').read_text().splitlines()
                          if v.strip() and not v.startswith('#')]
                if len(values) != 20 or not all(0 <= v <= 1 for v in values):
                    raise ValueError('Invalid Synfection genome')
                row['status'] = 'ok'
                row['artifacts'] = {n: digest(folder / n) for n in ['serum.wav', 'serum-preview.wav',
                    'match.wav', 'match-preview.wav', 'match.wav.genome.txt']}
            except (Exception, subprocess.TimeoutExpired) as exc:
                row.update(status='failed', error=str(exc))
                print(f'FAILED {preset.stem}: {exc}', flush=True)
            row['elapsed_seconds'] = round(time.monotonic() - start, 2)
            atomic_json(record_path, row)
        records.append(row)
        atomic_json(args.out / 'manifest.json', records)
        gallery(args.out, records)
    good = sum(r['status'] == 'ok' for r in records)
    print(f'{good}/{len(records)} complete. Gallery: {args.out / "index.html"}', flush=True)
    return 0 if good == len(records) else 1


if __name__ == '__main__':
    raise SystemExit(main())
