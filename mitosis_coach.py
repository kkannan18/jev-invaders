#!/usr/bin/env python3
"""Mitosis coach: JEV picks the strategy for the next game from what Cortex remembers.

Loop, one game per round:
  1. Recall: ask Cortex (Mitosis Labs memory) which strategies worked, with citations.
  2. Decide: JEV answers a typed `choice` question, "which strategy should play the
     next game?", with the recalled evidence as its state. It returns a pick and a
     calibrated confidence. JEV makes the call; no LLM is involved.
  3. Play: the chosen strategy plays the next seed in real time inside a Tenki sandbox.
  4. Remember: the result, and why that strategy was chosen, go back into Cortex.

Without Cortex the coach has no memory across runs; without Tenki there are no games.

  export MI_API_KEY=mi_... MI_OFFICE=<office id> TYPESAFE_API_KEY=... TENKI_API_KEY=tk_...
  python mitosis_coach.py --seed-memory             # load every past game from results.json into Cortex
  python mitosis_coach.py --rounds 5 --first-seed 11
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
ENDPOINT = os.environ.get("MI_ENDPOINT", "https://m.mitosislabs.ai")
FEED = "ext_agent.integration_feed"

# The strategies the coach can choose between (all JEV; they differ in how its answer is used).
STRATEGIES = {
    "six": {"variant": "six", "policy": "argmax", "conf": 0.0,
            "about": "One typed choice over the six moves; play the top pick."},
    "six:axes": {"variant": "six", "policy": "axes", "conf": 0.0,
                 "about": "One six-way choice; sum its probabilities into move and fire axes."},
    "six_lean": {"variant": "six_lean", "policy": "argmax", "conf": 0.0,
                 "about": "Six-way choice on a smaller, derived-features-only state."},
    "split": {"variant": "split", "policy": "argmax", "conf": 0.0,
              "about": "Two typed answers per frame: a steer choice (left/right/stay) and a yes/no fire value."},
    "split_lean": {"variant": "split_lean", "policy": "argmax", "conf": 0.0,
                   "about": "Steer choice plus fire yes/no, on the smaller state."},
}


def _post(path, body):
    key, office = os.environ.get("MI_API_KEY"), os.environ.get("MI_OFFICE")
    if not key or not office:
        sys.exit("Set MI_API_KEY and MI_OFFICE (your Mitosis API key and office id).")
    req = urllib.request.Request(f"{ENDPOINT}/api/v1/offices/{office}/cortex/v1/{path}",
                                 data=json.dumps(body).encode(), method="POST",
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json", "User-Agent": "curl/8.7.1"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read() or b"{}")


def strategy_of(run) -> str:
    v = run.get("variant") or "six"
    return f"{v}:axes" if run.get("policy") == "axes" else v


def memory_line(run, mode) -> str:
    return (f"Space Invaders {mode} game, seed {run['seed']}: strategy {strategy_of(run)} with "
            f"{run.get('served_model')} scored {run['score']} points, lost {run.get('lives_lost')} lives, "
            f"made {run.get('model_calls')} decisions over {run['steps']} steps.")


def ingest(runs_with_mode):
    rows = []
    for run, mode in runs_with_mode:
        uid = f"{mode}:{run.get('served_model')}:{strategy_of(run)}:seed{run['seed']}:{run['score']}"
        text = memory_line(run, mode)
        rows.append({"external_id": uid, "title": text, "content": text,
                     "metadata": {"seed": run["seed"], "score": run["score"], "strategy": strategy_of(run),
                                  "mode": mode, "model": run.get("served_model")}})
    for row in rows:
        _post("remember", {"agent": "jev-coach", "kind": "outcome", "text": row["content"]})
    return len(rows)


def seed_memory():
    d = json.loads((ROOT / "results.json").read_text())
    runs = [(r, "turn-based") for r in d["runs"] if str(r.get("served_model", "")).startswith("jev")]
    runs += [(r, "real-time") for r in d.get("realtime_showdown", {}).get("jev", [])]
    n = ingest(runs)
    print(f"Cortex now remembers {n} JEV games from results.json")


def recall():
    q = "Which JEV design scored the most points in Space Invaders games, and the Tenki tuning leaderboard?"
    res = _post("answer", {"query": q, "limit": 12})
    ev = []
    for r in res.get("results", []):
        ev.append({"memory": r.get("title") or r.get("content"), "cortex_id": r.get("universal_id"),
                   "relevance": round(r.get("score", 0), 3) if isinstance(r.get("score"), (int, float)) else None})
    return ev


def jev_pick(evidence, seed):
    from typesafe_sdk import TypeSafeClient

    client = TypeSafeClient(model="jev-latest")
    state = {"next_game": {"seed": seed, "mode": "real-time, 15 moves per second"},
             "strategies": {k: v["about"] for k, v in STRATEGIES.items()},
             "cortex_memories": evidence}
    q = {"next_strategy": {
        "type": "choice",
        "instructions": ("You are coaching a Space Invaders pilot. Using the remembered game results and "
                         "tuning leaderboards, pick the design most likely to score the most points in the next "
                         "real-time game. Prefer designs with higher remembered scores; when a design has few "
                         "remembered games, trying it can be worth it."),
        "criteria": {k: v["about"] for k, v in STRATEGIES.items()}}}
    t0 = time.perf_counter()
    r = client.system_one(state, q)
    a = r.answers["next_strategy"]
    return a.choice, a.confidence, dict(a.probabilities), round((time.perf_counter() - t0) * 1000), r.model


def play_round(name, seed):
    from invaders.results import record
    from tenki_arena import meta, run_job

    s = STRATEGIES[name]
    args = SimpleNamespace(max_steps=None, realtime=True, model=None, baseline_model=None, provider="anthropic",
                           policy=s["policy"], latency_tax_ms=0, lockdown=False, verbose=False, variant=s["variant"])
    run = run_job("jev", seed, s["conf"], args)
    return run, record, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-memory", action="store_true", help="load past JEV games from results.json into Cortex")
    ap.add_argument("--rounds", type=int, default=0)
    ap.add_argument("--first-seed", type=int, default=11)
    ap.add_argument("--no-push", action="store_true")
    a = ap.parse_args()
    if a.seed_memory:
        seed_memory()
    for i in range(a.rounds):
        seed = a.first_seed + i
        evidence = recall()
        choice, conf, probs, ms, model = jev_pick(evidence, seed)
        cited = [e["cortex_id"] for e in evidence[:5] if e.get("cortex_id")]
        print(f"\nround {i + 1}: Cortex recalled {len(evidence)} memories; JEV ({model}, {ms} ms) picks "
              f"{choice} with confidence {conf:.2f}  {json.dumps({k: round(v, 2) for k, v in probs.items()})}",
              flush=True)
        run, record, meta = play_round(choice, seed)
        run["coach"] = {"strategy": choice, "confidence": round(conf, 3), "probabilities": probs,
                        "cortex_evidence_ids": cited, "decision_ms": ms}
        m = meta("jev")
        run["phase"] = "coach"
        n = record(run, m, f"Mitosis coach round {i + 1}: JEV chose {choice} (conf {conf:.2f}) from "
                           f"{len(evidence)} Cortex memories; played in Tenki sandbox {run['tenki_sandbox_id']}",
                   push=not a.no_push)
        ingest([(run, "real-time")])
        _post("remember", {"agent": "jev-coach", "kind": "decision",
                           "text": f"For seed {seed} JEV chose {choice} with confidence {conf:.2f}; "
                                   f"it scored {run['score']} points."})
        print(f"  run {n}: seed {seed} {choice} scored {run['score']} ({run['model_calls']} decisions); "
              f"remembered in Cortex", flush=True)


if __name__ == "__main__":
    main()
