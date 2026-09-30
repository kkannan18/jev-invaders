#!/usr/bin/env python3
"""Tune the pilot in Tenki: race question designs on held-out seeds, crown a champion,
then test the champion on the seeds the LLM baseline played.

  python tune.py                                   # 5 designs x seeds 101-105, 5 sandboxes at a time
  python tune.py --test-seeds 1 2 3 4 5            # ...then test the champion turn-based and real-time
  python tune.py --skip-tuning --test-seeds 6 7 8  # test the existing champion on more seeds

Every game runs in its own Tenki sandbox and is pushed to results.json as it lands
(phase "tune" or "test"). The ranking is written to champion.json and pushed. When
MI_API_KEY and MI_OFFICE are set, every result is also remembered in Mitosis Cortex,
which is what mitosis_coach.py reads when JEV picks the next game's strategy.

Tuning seeds and test seeds never overlap, so the champion is not picked on the
games it is judged on.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

from invaders.results import record
from invaders.variants import CHAMPION_FILE, VARIANTS
from tenki_arena import meta, run_job

ROOT = Path(__file__).resolve().parent
DEFAULT_CANDIDATES = ["six", "six:axes", "six_lean", "split", "split_lean"]


def job_args(variant, policy, realtime):
    return SimpleNamespace(max_steps=None, realtime=realtime, model=None, baseline_model=None, provider="anthropic",
                           policy=policy, latency_tax_ms=0, lockdown=False, verbose=False, variant=variant)


def remember(text):
    if not (os.environ.get("MI_API_KEY") and os.environ.get("MI_OFFICE")):
        return
    try:
        from mitosis_coach import _post
        _post("remember", {"agent": "jev-tuner", "kind": "outcome", "text": text})
    except Exception as exc:  # noqa: BLE001 - memory is best effort; the game is already recorded
        print(f"  (Cortex remember failed: {exc})", flush=True)


def run_batch(jobs, phase, parallel, no_push):
    """jobs: list of (label, variant, policy, seed, realtime). Returns {label: [runs]}."""
    results = {}
    with ThreadPoolExecutor(parallel) as pool:
        futs = {pool.submit(run_job, "jev", seed, 0.0, job_args(v, p, rt)): (label, v, p, seed, rt)
                for label, v, p, seed, rt in jobs}
        for fut in as_completed(futs):
            label, v, p, seed, rt = futs[fut]
            try:
                run = fut.result()
            except Exception as exc:  # noqa: BLE001
                print(f"!! {label} seed {seed} failed: {exc}", flush=True)
                continue
            run["phase"] = phase
            m = meta("jev")
            n = record(run, m, f"{phase}: design {label}, {'real-time' if rt else 'turn-based'}, "
                               f"Tenki sandbox {run['tenki_sandbox_id']}", push=not no_push)
            results.setdefault(label, []).append(run)
            print(f"run {n}: [{phase}] {label:12s} seed {seed:>3} {'RT' if rt else 'TB'} score {run['score']:>4} "
                  f"p50 {run['latency_ms_p50']} ms", flush=True)
            remember(f"Space Invaders {'real-time' if rt else 'turn-based'} game, seed {seed}: JEV design {label} "
                     f"scored {run['score']} points with {run['model_calls']} decisions ({phase} phase).")
    return results


def git_push(paths, msg):
    subprocess.run(["git", "add", *paths], cwd=ROOT)
    subprocess.run(["git", "commit", "-m", msg], cwd=ROOT, capture_output=True)
    p = subprocess.run(["git", "push"], cwd=ROOT, capture_output=True, text=True)
    if p.returncode:
        subprocess.run(["git", "pull", "--rebase"], cwd=ROOT, capture_output=True)
        subprocess.run(["git", "push"], cwd=ROOT, capture_output=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", nargs="+", default=DEFAULT_CANDIDATES,
                    help="design[:policy], designs from " + ", ".join(VARIANTS))
    ap.add_argument("--tune-seeds", type=int, nargs="+", default=[101, 102, 103, 104, 105])
    ap.add_argument("--test-seeds", type=int, nargs="*", default=[])
    ap.add_argument("--skip-tuning", action="store_true")
    ap.add_argument("--no-realtime-test", action="store_true")
    ap.add_argument("--parallel", type=int, default=5)
    ap.add_argument("--no-push", action="store_true")
    a = ap.parse_args()
    overlap = set(a.tune_seeds) & set(a.test_seeds)
    if overlap and not a.skip_tuning:
        raise SystemExit(f"tuning and test seeds overlap: {sorted(overlap)}")

    if not a.skip_tuning:
        jobs = []
        for c in a.candidates:
            v, _, p = c.partition(":")
            jobs += [(c, v, p or "argmax", s, False) for s in a.tune_seeds]
        t0 = time.time()
        print(f"TUNING: {len(a.candidates)} designs x {len(a.tune_seeds)} seeds = {len(jobs)} Tenki sandboxes, "
              f"{a.parallel} at a time", flush=True)
        res = run_batch(jobs, "tune", a.parallel, a.no_push)
        board = sorted(({"design": k, "games": len(v), "mean_score": round(st.mean(r["score"] for r in v), 1),
                         "p50_ms": st.median(r["latency_ms_p50"] for r in v),
                         "mean_confidence": round(st.mean(r.get("mean_confidence", 0) for r in v), 3)}
                        for k, v in res.items()), key=lambda x: -x["mean_score"])
        if not board:
            raise SystemExit("no tuning games finished")
        print("\nLEADERBOARD")
        for i, b in enumerate(board, 1):
            print(f"  {i}. {b['design']:12s} mean {b['mean_score']:>6}  over {b['games']} games  "
                  f"p50 {b['p50_ms']} ms  conf {b['mean_confidence']}")
        best = board[0]
        v, _, p = best["design"].partition(":")
        champ = {"variant": v, "policy": p or "argmax", "design": best["design"], "tuned_on_seeds": a.tune_seeds,
                 "leaderboard": board, "tuning_wall_clock_s": round(time.time() - t0),
                 "chosen_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
        CHAMPION_FILE.write_text(json.dumps(champ, indent=2) + "\n")
        remember(f"Tenki tuning leaderboard for JEV Space Invaders designs: "
                 + "; ".join(f"{b['design']} mean {b['mean_score']}" for b in board)
                 + f". Champion: {best['design']}.")
        if not a.no_push:
            git_push(["champion.json"], f"champion: {best['design']} (mean {best['mean_score']} on seeds {a.tune_seeds})")
        print(f"\nCHAMPION: {best['design']}  -> champion.json", flush=True)

    if a.test_seeds:
        champ = json.loads(CHAMPION_FILE.read_text())
        v, p = champ["variant"], champ["policy"]
        jobs = [(champ["design"], v, p, s, False) for s in a.test_seeds]
        if not a.no_realtime_test:
            jobs += [(champ["design"], v, p, s, True) for s in a.test_seeds]
        print(f"\nTEST: champion {champ['design']} on seeds {a.test_seeds} "
              f"({'turn-based + real-time' if not a.no_realtime_test else 'turn-based'})", flush=True)
        run_batch(jobs, "test", a.parallel, a.no_push)
        print("\nRun `python summarize.py` for the champion vs LLM tables.")


if __name__ == "__main__":
    main()
