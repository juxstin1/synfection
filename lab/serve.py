"""
serve.py — local desktop rating shim for synfection patches.

Spins up a tiny localhost server, opens your browser to a clean rating UI, and
writes every star/discard straight to <dir>/ratings.csv the instant you click it
(nothing to export). Hit "Done — send to Cue" (or just close the tab) and the
server shuts down, which is the signal back to me that ratings are ready.

  python serve.py --dir rlhf/round01

Stdlib only — no Flask, no pip. Reads patches.jsonl from <dir>, resumes any
ratings already in <dir>/ratings.csv so you can stop and come back.
"""

import argparse
import csv
import json
import os
import threading
import webbrowser
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

PAGE = """<!doctype html><html><head><meta charset=utf-8><title>synfection rater</title>
<style>
 *{{box-sizing:border-box}}
 body{{font:15px/1.4 system-ui,sans-serif;background:#0f0f10;color:#eee;margin:0;
   padding:0 24px 80px}}
 header{{position:sticky;top:0;background:#0f0f10;padding:18px 0 12px;z-index:9;
   border-bottom:1px solid #222}}
 h1{{font-size:18px;margin:0}} .sub{{color:#888;margin:6px 0 0;font-size:13px}}
 #bar{{display:flex;gap:14px;align-items:center;margin-top:12px}}
 .grow{{flex:1}} #prog{{color:#9c9;font-weight:600}}
 button{{background:#2d7;color:#012;border:0;border-radius:7px;padding:9px 16px;
   font-weight:700;cursor:pointer;font-size:14px}}
 button.ghost{{background:#222;color:#bbb}}
 label.tog{{color:#9bd;font-size:13px;cursor:pointer;user-select:none}}
 .card{{background:#1a1a1c;border:1px solid #28282c;border-radius:11px;padding:13px 16px;
   margin:11px 0;display:flex;align-items:center;gap:16px;outline:none}}
 .card:focus{{border-color:#39f}} .card.done{{border-color:#2a6}}
 .card.discard{{border-color:#722;opacity:.6}}
 .meta{{flex:1;min-width:0}} .name{{font-weight:600}}
 .tag{{color:#7bd;font-size:13px}} audio{{height:36px}}
 .stars{{font-size:27px;cursor:pointer;user-select:none;white-space:nowrap}}
 .stars span{{color:#3a3a3a;padding:0 2px}} .stars span.on{{color:#fc4}}
 .x{{color:#955;cursor:pointer;font-size:14px;padding-left:8px}} .x.on{{color:#f55;font-weight:700}}
 textarea{{background:#141416;color:#ccc;border:1px solid #303034;border-radius:6px;
   width:150px;height:36px;resize:none;font:13px system-ui;padding:6px}}
 kbd{{background:#222;border:1px solid #333;border-radius:4px;padding:1px 6px;font-size:12px}}
 #toast{{position:fixed;bottom:18px;left:50%;transform:translateX(-50%);background:#1c3;
   color:#012;padding:8px 16px;border-radius:8px;font-weight:700;opacity:0;
   transition:opacity .25s;pointer-events:none}}
</style></head><body>
<header>
 <h1>synfection — patch quality rater 🎛️</h1>
 <p class=sub>Click a card, press <kbd>1</kbd>–<kbd>5</kbd> to star · <kbd>0</kbd> discard ·
  <kbd>Space</kbd> play/pause. Saves live — close anytime.</p>
 <div id=bar>
  <span id=prog></span><span class=grow></span>
  <label class=tog><input type=checkbox id=autoplay checked> auto-play next</label>
  <button class=ghost onclick=skipUnrated()>↧ next unrated</button>
  <button onclick=finish()>✓ Done — send to Cue</button>
 </div>
</header>
<div id=list></div>
<div id=toast></div>
<script>
const DATA = {data};
const R = {ratings};            // id -> {{rating, notes}}  (prefilled from disk)
const list = document.getElementById('list');
const toastEl = document.getElementById('toast');
let tmo;
function toast(m){{toastEl.textContent=m;toastEl.style.opacity=1;
  clearTimeout(tmo);tmo=setTimeout(()=>toastEl.style.opacity=0,900);}}
function save(p){{
  const r=R[p.id]||{{rating:'',notes:''}};
  fetch('/save',{{method:'POST',headers:{{'Content-Type':'application/json'}},
    body:JSON.stringify({{id:p.id,rating:r.rating,notes:r.notes}})}}).catch(()=>{{}});
}}
function draw(){{
 list.innerHTML='';
 DATA.forEach((p,i)=>{{
  const r=R[p.id]||{{rating:'',notes:''}};
  const card=document.createElement('div');
  card.className='card'+(r.rating===0?' discard':(r.rating!==''?' done':''));
  card.tabIndex=0; card.dataset.i=i; card.id='c'+i;
  let stars='';
  for(let s=1;s<=5;s++) stars+=`<span class="${{r.rating>=s&&r.rating?'on':''}}" onclick="rate(${{i}},${{s}})">★</span>`;
  card.innerHTML=`<div class=meta><div class=name>${{p.file}}</div>
    <div class=tag>${{p.archetype}} · MIDI ${{p.note}}</div></div>
    <audio controls preload=none src="${{p.file}}"></audio>
    <div class=stars>${{stars}}<span class="x ${{r.rating===0?'on':''}}" onclick="rate(${{i}},0)">✕</span></div>
    <textarea placeholder=notes oninput="note(${{i}},this.value)">${{r.notes||''}}</textarea>`;
  list.appendChild(card);
 }});
 const done=Object.values(R).filter(x=>x.rating!=='').length;
 document.getElementById('prog').textContent=`${{done}} / ${{DATA.length}} rated`;
}}
function rate(i,s){{
  const p=DATA[i]; R[p.id]={{rating:s,notes:(R[p.id]||{{}}).notes||''}};
  draw(); save(p); toast(s===0?'discarded':'★'.repeat(s));
  const next=[...document.querySelectorAll('.card')][i+1];
  if(next){{next.focus();next.scrollIntoView({{block:'center',behavior:'smooth'}});
    if(document.getElementById('autoplay').checked){{const a=next.querySelector('audio');a.play().catch(()=>{{}});}}}}
}}
function note(i,v){{const p=DATA[i];R[p.id]={{rating:(R[p.id]||{{rating:''}}).rating,notes:v}};save(p);}}
function skipUnrated(){{
  for(let i=0;i<DATA.length;i++){{const r=R[DATA[i].id];if(!r||r.rating===''){{
    const c=document.getElementById('c'+i);c.focus();c.scrollIntoView({{block:'center'}});
    if(document.getElementById('autoplay').checked)c.querySelector('audio').play().catch(()=>{{}});return;}}}}
  toast('all rated 🎉');
}}
document.addEventListener('keydown',e=>{{
 const c=document.activeElement.closest?document.activeElement.closest('.card'):null;
 if(!c||e.target.tagName==='TEXTAREA')return; const i=+c.dataset.i;
 if(e.key>='0'&&e.key<='5'){{rate(i,+e.key);e.preventDefault();}}
 if(e.key===' '){{const a=c.querySelector('audio');a.paused?a.play():a.pause();e.preventDefault();}}
}});
function finish(){{
  fetch('/done',{{method:'POST'}}).catch(()=>{{}});
  document.body.innerHTML='<header><h1>Sent to Cue ✓</h1>'+
    '<p class=sub>Ratings saved. You can close this tab.</p></header>';
}}
window.addEventListener('beforeunload',()=>{{
  navigator.sendBeacon('/done');   // closing the tab = sent to Cue
}});
draw();
</script></body></html>"""


class Rater:
    def __init__(self, directory):
        self.dir = directory
        self.patches = []
        with open(os.path.join(directory, "patches.jsonl")) as f:
            for line in f:
                line = line.strip()
                if line:
                    self.patches.append(json.loads(line))
        self.meta = {p["id"]: p for p in self.patches}
        self.ratings = {}                       # id -> {"rating":..,"notes":..}
        self._load_existing()
        self.lock = threading.Lock()

    def _load_existing(self):
        path = os.path.join(self.dir, "ratings.csv")
        if not os.path.exists(path):
            return
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                rt = (row.get("rating") or "").strip()
                if rt != "":
                    self.ratings[int(row["id"])] = {
                        "rating": int(rt), "notes": row.get("notes", "")}

    def page(self):
        light = [dict(id=p["id"], file=p["file"], archetype=p["archetype"],
                      note=p["note"]) for p in self.patches]
        return PAGE.format(data=json.dumps(light), ratings=json.dumps(self.ratings))

    def update(self, id_, rating, notes):
        with self.lock:
            if rating == "" or rating is None:
                self.ratings.pop(id_, None)
            else:
                self.ratings[id_] = {"rating": int(rating), "notes": notes or ""}
            self._flush()

    def _flush(self):
        path = os.path.join(self.dir, "ratings.csv")
        tmp = path + ".tmp"
        with open(tmp, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "file", "archetype", "note", "rating", "notes"])
            for p in self.patches:
                r = self.ratings.get(p["id"], {})
                w.writerow([p["id"], p["file"], p["archetype"], p["note"],
                            r.get("rating", ""), r.get("notes", "")])
        os.replace(tmp, path)


def make_handler(rater, shutdown):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):            # quiet
            pass

        def _send(self, code, body, ctype="text/html; charset=utf-8"):
            data = body.encode() if isinstance(body, str) else body
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            path = self.path.split("?")[0]
            if path in ("/", "/index.html"):
                self._send(200, rater.page())
                return
            fname = os.path.basename(path)
            full = os.path.join(rater.dir, fname)
            if fname.endswith(".wav") and os.path.exists(full):
                with open(full, "rb") as f:
                    self._send(200, f.read(), "audio/wav")
                return
            self._send(404, "not found", "text/plain")

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(n) if n else b""
            if self.path == "/save":
                try:
                    d = json.loads(raw or b"{}")
                    rater.update(int(d["id"]), d.get("rating", ""), d.get("notes", ""))
                    self._send(200, "ok", "text/plain")
                except Exception as e:
                    self._send(400, str(e), "text/plain")
                return
            if self.path == "/done":
                self._send(200, "bye", "text/plain")
                threading.Thread(target=shutdown, daemon=True).start()
                return
            self._send(404, "no", "text/plain")
    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="rlhf/round01")
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()

    rater = Rater(a.dir)
    httpd = ThreadingHTTPServer(("127.0.0.1", a.port), None)
    httpd.RequestHandlerClass = make_handler(rater, httpd.shutdown)
    url = f"http://127.0.0.1:{a.port}/"
    print(f"rating {len(rater.patches)} patches from {a.dir}/  ->  {url}", flush=True)
    print("close the tab or hit 'Done' when finished; ratings save live to "
          f"{os.path.join(a.dir, 'ratings.csv')}", flush=True)
    threading.Thread(target=lambda: webbrowser.open(url), daemon=True).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    rater._flush()
    done = sum(1 for v in rater.ratings.values() if v.get("rating") != "")
    print(f"server closed — {done}/{len(rater.patches)} rated, saved to "
          f"{os.path.join(a.dir, 'ratings.csv')}", flush=True)


if __name__ == "__main__":
    main()
