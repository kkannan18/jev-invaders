#!/usr/bin/env python3
"""Tenki arena: every game runs in its own disposable Tenki sandbox, in parallel.

This is how the numbers in results.json are produced. Each job (decider x seed
x confidence threshold) gets a fresh microVM that can only reach PyPI and the
one model API it needs; the local collector appends each finished game to
results.json and pushes it immediately.

  export TENKI_API_KEY=tk_... TYPESAFE_API_KEY=... ANTHROPIC_API_KEY=...
  python tenki_arena.py --deciders jev baseline --seeds 1 2 3 4 5
  python tenki_arena.py --deciders jev --seeds 1 2 3 --conf-thresholds 0 0.4 0.6 0.8   # tune the gate
  python tenki_arena.py --deciders jev baseline --seeds 1 2 3 --realtime               # latency showdown
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace

from invaders.results import record

ROOT = Path(__file__).resolve().parent
SHIP = ["play.py", "requirements.txt", "invaders/__init__.py", "invaders/env.py", "invaders/deciders.py",
        "invaders/game.py", "invaders/results.py"]
PYPI = ["pypi.org", "*.pypi.org", "files.pythonhosted.org", "*.pythonhosted.org"]
EGRESS = {"jev": ["api.typesafe.ai"], "baseline": ["api.anthropic.com", "api.openai.com",
                                                    "generativelanguage.googleapis.com"],
          "scripted": [], "random": []}
KEYS = {"jev": ["TYPESAFE_API_KEY", "JEV_MODEL"],
        "baseline": ["ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY"], "scripted": [], "random": []}


def meta(kind, model=None, provider="anthropic"):
    """What results.json needs to know about a decider, without opening an API client locally."""
    if kind == "jev":
        return SimpleNamespace(name="jev", provider="typesafe", requested_model=model or "jev-latest",
                               sdk_package="typesafe-sdk", sdk_version=version("typesafe-sdk"))
    if kind == "baseline":
        return SimpleNamespace(name="baseline", provider=provider, requested_model=model or "claude-haiku-4-5",
                               sdk_package="system-one-adapter", sdk_version=version("system-one-adapter"))
    return SimpleNamespace(name=kind, provider="none", requested_model=f"{kind}-policy",
                           sdk_package=None, sdk_version=None)


def run_job(kind, seed, thr, args):
    from tenki import Sandbox

    env = {k: os.environ[k] for k in KEYS[kind] if os.environ.get(k)}
    flags = f"--decider {kind} --seeds {seed} --conf-threshold {thr} --no-record --json-out out.jsonl --log-every 100"
    if args.max_steps:
        flags += f" --max-steps {args.max_steps}"
    if args.realtime:
        flags += " --realtime"
    if args.model and kind == "jev":
        flags += f" --model {args.model}"
    if args.baseline_model and kind == "baseline":
        flags += f" --model {args.baseline_model} --provider {args.provider}"
    t0 = time.time()
    with Sandbox.create(name=f"jev-invaders-{kind}-s{seed}-t{thr}", cpu_cores=2, memory_mb=2048,
                        allow_domains=PYPI + EGRESS[kind], max_duration=3 * 3600,
                        tags=["jev-bakeoff", kind]) as sb:
        for f in SHIP:
            sb.fs.write_text(f, (ROOT / f).read_text())
        sb.exec("bash", "-lc", "python3 -m pip install -q -r requirements.txt 2>&1 | tail -3",
                timeout=900, check=True)
        # run detached and poll, so a long LLM game is not tied to one stream
        sb.exec("bash", "-lc",
                f"setsid nohup bash -c 'python3 play.py {flags} > game.log 2>&1; echo $? > done' "
                "> /dev/null 2>&1 < /dev/null &", env=env, timeout=30)
        while True:
            time.sleep(15)
            r = sb.exec("bash", "-lc", "cat done 2>/dev/null; tail -1 game.log", timeout=30)
            lines = r.stdout_text.strip().splitlines()
            if lines and lines[0].strip().isdigit():
                break
            if lines and args.verbose:
                print(f"  [{kind} s{seed} t{thr}] {lines[-1]}", flush=True)
        out = sb.exec("bash", "-lc", "cat out.jsonl 2>/dev/null", timeout=60).stdout_text.strip()
        if not out:
            log = sb.exec("bash", "-lc", "tail -20 game.log", timeout=30).stdout_text
            raise RuntimeError(f"{kind} seed {seed}: game produced no result\n{log}")
        run = json.loads(out.splitlines()[-1])
        run["tenki_sandbox_id"] = str(sb.id)
        run["sandbox_wall_clock_s"] = round(time.time() - t0, 1)
        return run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deciders", nargs="+", default=["jev", "baseline"],
                    choices=["jev", "baseline", "scripted", "random"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--conf-thresholds", type=float, nargs="+", default=[0.0])
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--realtime", action="store_true")
    ap.add_argument("--model", default=None, help="JEV model id")
    ap.add_argument("--baseline-model", default=None)
    ap.add_argument("--provider", default="anthropic")
    ap.add_argument("--parallel", type=int, default=16)
    ap.add_argument("--notes", default="")
    ap.add_argument("--no-push", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    jobs = []
    for kind, seed in itertools.product(args.deciders, args.seeds):
        thrs = args.conf_thresholds if kind == "jev" else [0.0]
        jobs += [(kind, seed, t) for t in thrs]
    print(f"launching {len(jobs)} Tenki sandboxes ({args.parallel} at a time)", flush=True)
    by_thr: dict[float, list[int]] = {}
    with ThreadPoolExecutor(args.parallel) as pool:
        futs = {pool.submit(run_job, k, s, t, args): (k, s, t) for k, s, t in jobs}
        for fut in as_completed(futs):
            kind, seed, thr = futs[fut]
            try:
                run = fut.result()
            except Exception as exc:  # noqa: BLE001
                print(f"!! {kind} seed {seed} thr {thr} failed: {exc}", flush=True)
                continue
            note = args.notes or f"Tenki sandbox {run['tenki_sandbox_id']}; conf gate {thr}"
            n = record(run, meta(kind, args.model if kind == "jev" else args.baseline_model, args.provider),
                       note, args.max_steps, push=not args.no_push)
            if kind == "jev":
                by_thr.setdefault(thr, []).append(run["score"])
            print(f"run {n}: {kind} seed {seed} thr {thr} score {run['score']} p50 {run['latency_ms_p50']}ms",
                  flush=True)
    if len(by_thr) > 1:
        print("\nconfidence-gate sweep (mean JEV score):")
        for t, s in sorted(by_thr.items()):
            print(f"  threshold {t:.2f}: {sum(s) / len(s):7.1f}  over {len(s)} games")


if __name__ == "__main__":
    main()
