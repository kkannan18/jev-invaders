#!/usr/bin/env python3
"""Live split-screen race: JEV vs the LLM baseline, same seed, same start time,
each playing at real Atari speed inside its own Tenki sandbox and serving its
own spectator view through a Tenki preview URL.

  export TENKI_API_KEY=tk_... TYPESAFE_API_KEY=... ANTHROPIC_API_KEY=...
  python live_race.py --seed 3            # race in two Tenki sandboxes
  python live_race.py --seed 3 --local    # same race on this machine (fallback)

Opens race.html (split screen) in your browser, then records both games to
results.json and pushes, like every other game.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path

from invaders.results import record
from tenki_arena import EGRESS, KEYS, SHIP, meta

ROOT = Path(__file__).resolve().parent
PLAYERS = [("jev", "JEV 1.13", "TypeSafe decision model"), ("baseline", "Claude Haiku 4.5", "LLM via System One adapter")]

RACE_HTML = """<!doctype html><html lang=en><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>JEV vs Claude, live</title>
<link rel=preconnect href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Silkscreen:wght@400;700&family=IBM+Plex+Mono:wght@400;600&display=swap" rel=stylesheet>
<style>
:root{--bg:#07080c;--panel:#0f121a;--line:#232838;--fg:#e9ecf4;--dim:#8a93a8;--jev:#5cf2b3;--llm:#ff8a5c;--warn:#ffd35c}
*{box-sizing:border-box}html,body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.45 "IBM Plex Mono",ui-monospace,Menlo,monospace}
main{max-width:1100px;margin:0 auto;padding:20px 16px 28px}
h1{font:700 clamp(22px,4vw,34px)/1.1 Silkscreen,monospace;margin:0 0 4px;letter-spacing:.02em}
.sub{color:var(--dim);margin:0 0 18px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}
@media (max-width:760px){.grid{grid-template-columns:1fr}}
.p{background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:14px;min-width:0}
.p h2{font:700 18px Silkscreen,monospace;margin:0;display:flex;justify-content:space-between;gap:8px;align-items:baseline}
.p.jev h2{color:var(--jev)}.p.llm h2{color:var(--llm)}
.tag{font:12px "IBM Plex Mono",monospace;color:var(--dim)}
.screen{margin:10px 0;background:#000;border:1px solid var(--line);aspect-ratio:160/210;width:100%;max-width:100%}
.screen img{width:100%;height:100%;image-rendering:pixelated;display:block}
.stats{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}
.s{border-top:1px solid var(--line);padding-top:6px}.s b{display:block;font:700 22px Silkscreen,monospace;font-variant-numeric:tabular-nums}
.s span{font-size:11px;letter-spacing:.06em;text-transform:uppercase;color:var(--dim)}
.bar{height:8px;background:var(--line);border-radius:4px;margin-top:10px;overflow:hidden}.bar i{display:block;height:100%;width:0}
.jev .bar i{background:var(--jev)}.llm .bar i{background:var(--llm)}
.status{margin-top:8px;color:var(--warn);min-height:1.4em}
.foot{color:var(--dim);margin-top:16px;font-size:13px}
</style>
<main>
<h1>JEV vs Claude &mdash; live</h1>
<p class=sub>Same game, same seed, same start. Real Atari speed: 15 moves a second. Until a model answers, its ship repeats its last move.</p>
<div class=grid>
__PANELS__
</div>
<p class=foot>__FOOT__</p>
</main>
<script>
const P=__PLAYERS__;
function fmt(n){return n==null?'–':n}
async function poll(p){
  const root=document.getElementById(p.id);
  try{
    const s=await (await fetch(p.url+'state.json',{cache:'no-store'})).json();
    root.querySelector('.score').textContent=fmt(s.score);
    root.querySelector('.dec').textContent=fmt(s.decisions);
    const fresh=s.step?Math.round(100*(1-s.stale/s.step)):0;
    root.querySelector('.fresh').textContent=s.step?fresh+'%':'–';
    root.querySelector('.bar i').style.width=fresh+'%';
    root.querySelector('.status').textContent=s.status+(s.last_latency_ms?` · last answer ${s.last_latency_ms} ms`:'')+` · lives ${s.lives}`;
    root.querySelector('img').src=p.url+'frame.png?'+Date.now();
  }catch(e){root.querySelector('.status').textContent='connecting…'}
  setTimeout(()=>poll(p),110);
}
P.forEach(poll);
</script></html>"""

PANEL = """<section class="p __CLS__" id="__ID__"><h2>__NAME__ <span class=tag>__TAG__</span></h2>
<div class=screen><img alt="__NAME__ game screen"></div>
<div class=stats><div class=s><b class=score>–</b><span>score</span></div>
<div class=s><b class=dec>–</b><span>decisions</span></div>
<div class=s><b class=fresh>–</b><span>fresh moves</span></div></div>
<div class=bar><i></i></div><div class=status>connecting…</div></section>"""


def write_page(urls: dict, foot: str) -> Path:
    panels, players = [], []
    for kind, name, tag in PLAYERS:
        cls = "jev" if kind == "jev" else "llm"
        panels.append(PANEL.replace("__CLS__", cls).replace("__ID__", kind).replace("__NAME__", name).replace("__TAG__", tag))
        players.append({"id": kind, "url": urls[kind]})
    html = RACE_HTML.replace("__PANELS__", "\n".join(panels)).replace("__PLAYERS__", json.dumps(players)).replace("__FOOT__", foot)
    out = ROOT / "race.html"
    out.write_text(html)
    return out


def ship_files(sb):
    sb.exec("mkdir", "-p", "invaders", timeout=30)
    for f in SHIP + ["live_play.py"]:
        sb.fs.write_text(f, (ROOT / f).read_text())


def race_tenki(a):
    from tenki import Sandbox

    boxes, errors = {}, []

    def setup(kind):
        for attempt in range(4):
            try:
                sb = Sandbox.create(name=f"jev-live-{kind}-s{a.seed}", cpu_cores=2, memory_mb=2048,
                                    tags=["jev-bakeoff", "live-race", kind])
                ship_files(sb)
                sb.exec("bash", "-lc", "python3 -m pip install -q -r requirements.txt 2>&1 | tail -2",
                        timeout=900, check=True)
                boxes[kind] = sb
                print(f"  {kind}: sandbox {sb.id} ready", flush=True)
                return
            except Exception as exc:  # noqa: BLE001
                if attempt == 3 or not any(w in str(exc) for w in ("capacity", "concurrent", "placement")):
                    errors.append(f"{kind}: {exc}")
                    return
                print(f"  {kind}: Tenki busy, retrying in {20 * (attempt + 1)}s", flush=True)
                time.sleep(20 * (attempt + 1))

    print("creating two Tenki sandboxes and installing (1-2 min)...", flush=True)
    ts = [threading.Thread(target=setup, args=(k,)) for k, *_ in PLAYERS]
    [t.start() for t in ts]
    [t.join() for t in ts]
    if errors:
        for b in boxes.values():
            b.close()
        sys.exit("Tenki setup failed: " + "; ".join(errors) + "\nTry again, or run: python live_race.py --local")

    start_at = time.time() + a.countdown
    urls = {}
    for kind, sb in boxes.items():
        env = {k: os.environ[k] for k in KEYS[kind] if os.environ.get(k)}
        sb.exec("bash", "-lc",
                f"setsid nohup python3 live_play.py --decider {kind} --seed {a.seed} --port 8080 "
                f"--start-at {start_at} > live.log 2>&1 < /dev/null &", env=env, timeout=30)
        url = sb.expose_port(8080, ttl=3600).url
        urls[kind] = url.rstrip("/") + "/"
        print(f"  {kind} live view: {urls[kind]}", flush=True)
    page = write_page(urls, f"Each player runs in its own Tenki sandbox and serves its own view: "
                            f"JEV {boxes['jev'].id}, Claude {boxes['baseline'].id}. Seed {a.seed}.")
    webbrowser.open(page.as_uri())
    print(f"race page: {page}  (starts in {a.countdown}s)", flush=True)

    done = {}
    while len(done) < 2:
        time.sleep(4)
        for kind, sb in boxes.items():
            if kind in done:
                continue
            out = sb.exec("bash", "-lc", "cat out.jsonl 2>/dev/null", timeout=30).stdout_text.strip()
            if out:
                run = json.loads(out.splitlines()[-1])
                run["tenki_sandbox_id"] = str(sb.id)
                done[kind] = run
                print(f"  {kind} finished: score {run['score']}, {run['model_calls']} decisions", flush=True)
            elif time.time() > start_at + 900:
                log = sb.exec("bash", "-lc", "tail -15 live.log", timeout=30).stdout_text
                sys.exit(f"{kind} did not finish:\n{log}")
    finish(a, done, "live race, each player in its own Tenki sandbox")
    print(f"keeping the live views up for {a.hold}s so you can film the final screens (Ctrl+C to end)...")
    try:
        time.sleep(a.hold)
    except KeyboardInterrupt:
        pass
    for b in boxes.values():
        b.close()


def race_local(a):
    start_at = time.time() + a.countdown
    procs, urls, outs = {}, {}, {}
    for i, (kind, *_rest) in enumerate(PLAYERS):
        port = 8081 + i
        outs[kind] = ROOT / f"live_{kind}.jsonl"
        outs[kind].unlink(missing_ok=True)
        procs[kind] = subprocess.Popen([sys.executable, "live_play.py", "--decider", kind, "--seed", str(a.seed),
                                        "--port", str(port), "--start-at", str(start_at), "--out", str(outs[kind]),
                                        "--linger", str(a.hold)], cwd=ROOT)
        urls[kind] = f"http://localhost:{port}/"
    page = write_page(urls, f"Both players on this machine (local fallback). Seed {a.seed}.")
    time.sleep(2)
    webbrowser.open(page.as_uri())
    print(f"race page: {page}  (starts in {a.countdown}s)", flush=True)
    done = {}
    try:
        while len(done) < 2:
            time.sleep(2)
            for kind, f in outs.items():
                if kind not in done and f.exists() and f.read_text().strip():
                    done[kind] = json.loads(f.read_text().strip().splitlines()[-1])
                    print(f"  {kind} finished: score {done[kind]['score']}", flush=True)
        finish(a, done, "live race, local")
        print(f"keeping the live views up for {a.hold}s (Ctrl+C to end)...")
        time.sleep(a.hold)
    except KeyboardInterrupt:
        pass
    finally:
        for p in procs.values():
            p.terminate()


def finish(a, done, where):
    j, b = done["jev"], done["baseline"]
    note = (f"{where}; seed {a.seed}; JEV {j['score']} pts / {j['model_calls']} decisions vs "
            f"Claude {b['score']} pts / {b['model_calls']} decisions")
    for kind in ("jev", "baseline"):
        if not a.no_record:
            record(done[kind], meta(kind), note, push=not a.no_push)
    print("\nRESULT:", note, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--local", action="store_true", help="run both players on this machine instead of Tenki")
    ap.add_argument("--countdown", type=int, default=12, help="seconds between opening the page and the start")
    ap.add_argument("--hold", type=int, default=120, help="seconds to keep the views up after the race")
    ap.add_argument("--no-push", action="store_true")
    ap.add_argument("--no-record", action="store_true")
    a = ap.parse_args()
    race_local(a) if a.local else race_tenki(a)


if __name__ == "__main__":
    main()
