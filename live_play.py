#!/usr/bin/env python3
"""One live player: plays a real-time game and serves a spectator view over HTTP.

Runs inside a Tenki sandbox (started by live_race.py) or locally with --local.

  GET /            single-player spectator page
  GET /frame.png   the latest game frame (3x, nearest-neighbour)
  GET /state.json  score, lives, decisions made, stale moves, last latency, status
All responses allow cross-origin reads so the split-screen page can poll both players.
"""
from __future__ import annotations

import argparse
import io
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from PIL import Image

from invaders.deciders import make_decider
from invaders.env import ACTIONS
from invaders.game import play

STATE = {"status": "waiting", "decider": None, "model": None, "seed": None, "step": 0, "score": 0,
         "lives": 3, "decisions": 0, "stale": 0, "last_latency_ms": None, "action": None,
         "start_at": None, "result": None}
FRAME = {"png": b""}
LOCK = threading.Lock()

PAGE = """<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>Live player</title>
<style>body{margin:0;background:#0b0d12;color:#e8eaf0;font:15px/1.4 ui-monospace,Menlo,monospace;display:grid;place-items:center;min-height:100vh}
img{width:min(480px,92vw);image-rendering:pixelated;border:1px solid #2a2f3a}#s{margin-top:10px}</style>
<div><img id=f src="frame.png"><div id=s>waiting…</div></div>
<script>
async function tick(){try{const s=await (await fetch('state.json',{cache:'no-store'})).json();
document.getElementById('s').textContent=`${s.decider} · ${s.status} · score ${s.score} · lives ${s.lives} · decisions ${s.decisions} · stale moves ${s.stale}`;
document.getElementById('f').src='frame.png?'+Date.now();}catch(e){} setTimeout(tick,120)}tick();
</script>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, body: bytes, ctype: str):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?")[0]
        with LOCK:
            if path == "/frame.png":
                return self._send(FRAME["png"], "image/png")
            if path == "/state.json":
                return self._send(json.dumps(STATE).encode(), "application/json")
        return self._send(PAGE.encode(), "text/html; charset=utf-8")


def encode(frame) -> bytes:
    img = Image.fromarray(frame).resize((frame.shape[1] * 3, frame.shape[0] * 3), Image.NEAREST)
    buf = io.BytesIO()
    img.save(buf, format="PNG", compress_level=1)
    return buf.getvalue()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decider", required=True, choices=["jev", "baseline", "scripted", "random"])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--start-at", type=float, default=0.0, help="unix time to start, so two players start together")
    ap.add_argument("--out", default="out.jsonl")
    ap.add_argument("--linger", type=int, default=900, help="seconds to keep serving after the game ends")
    a = ap.parse_args()

    server = ThreadingHTTPServer(("0.0.0.0", a.port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    decider = make_decider(a.decider, seed=a.seed)
    with LOCK:
        STATE.update(decider=a.decider, model=decider.requested_model, seed=a.seed, start_at=a.start_at)
    # warm the first frame so the page is not blank while waiting
    from invaders.env import make_env
    e = make_env()
    e.reset(seed=a.seed)
    FRAME["png"] = encode(e.unwrapped.ale.getScreenRGB())
    e.close()

    while time.time() < a.start_at:
        with LOCK:
            STATE["status"] = f"starting in {int(a.start_at - time.time()) + 1}s"
        time.sleep(0.2)
    with LOCK:
        STATE["status"] = "playing"

    def on_step(steps, score, info, action, frame, live):
        png = encode(frame)
        with LOCK:
            FRAME["png"] = png
            STATE.update(step=steps, score=int(score), lives=int(info.get("lives", 0)),
                         decisions=live["calls"], stale=live["stale"], action=ACTIONS[action],
                         last_latency_ms=round(live["lat"]) if live["lat"] else None)

    run = play(decider, a.seed, realtime=True, on_step=on_step)
    run["live_race"] = True
    with open(a.out, "a") as f:
        f.write(json.dumps(run) + "\n")
    with LOCK:
        STATE.update(status="game over", score=run["score"], decisions=run["model_calls"],
                     stale=run.get("stale_steps", 0), result=run)
    print("DONE", json.dumps({k: run[k] for k in ("score", "steps", "model_calls", "latency_ms_p50")}), flush=True)
    time.sleep(a.linger)


if __name__ == "__main__":
    main()
