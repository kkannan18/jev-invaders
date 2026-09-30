#!/usr/bin/env python3
"""Play Space Invaders with a decider and log every game to results.json.

  python play.py --decider jev --seeds 1 2 3 4 5              # JEV pilot, pushes after each game
  python play.py --decider baseline --seeds 1 2 3 4 5         # Claude Haiku 4.5 via System One adapter
  python play.py --decider jev --seeds 1 --realtime           # 15 steps/s wall clock: latency costs lives
  python play.py --decider scripted --seeds 1 --no-push       # keyless harness check
  python play.py --decider jev --seeds 7 --json-out run.json --no-record   # used by tenki_arena.py
"""
import argparse
import json
import sys

from invaders.deciders import make_decider
from invaders.game import play
from invaders.results import record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decider", choices=["jev", "baseline", "random", "scripted"], required=True)
    ap.add_argument("--model", default=None, help="override model id (default jev-latest / claude-haiku-4-5)")
    ap.add_argument("--provider", default="anthropic", help="baseline provider: anthropic | openai | gemini")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--max-steps", type=int, default=None, help="cap each game (recorded as config.max_steps)")
    ap.add_argument("--conf-threshold", type=float, default=0.0, help="hold last action when confidence is below this")
    ap.add_argument("--realtime", action="store_true", help="run the game at real Atari speed (15 decisions/s max)")
    ap.add_argument("--policy", choices=["argmax", "axes"], default=None,
                    help="axes: sum the returned probabilities into move and fire axes")
    ap.add_argument("--latency-tax-ms", type=int, default=0,
                    help="control experiment: add this delay to every JEV answer (e.g. 1050 = Claude's median)")
    ap.add_argument("--variant", default=None,
                    help="question design: six | six_lean | split | split_lean | champion (see invaders/variants.py)")
    ap.add_argument("--notes", default="")
    ap.add_argument("--no-push", action="store_true", help="write results.json but do not commit/push")
    ap.add_argument("--no-record", action="store_true", help="do not touch results.json")
    ap.add_argument("--json-out", default=None, help="also write the run(s) as JSON lines to this file")
    ap.add_argument("--log-every", type=int, default=250)
    a = ap.parse_args()

    for seed in a.seeds:
        decider = make_decider(a.decider, model=a.model, provider=a.provider, seed=seed, policy=a.policy,
                               latency_tax_ms=a.latency_tax_ms, variant=a.variant)
        print(f"== {a.decider} seed {seed} ({'realtime' if a.realtime else 'turn-based'})", flush=True)
        run = play(decider, seed, a.max_steps, a.conf_threshold, a.realtime, a.log_every)
        print(json.dumps({k: run[k] for k in ("seed", "score", "steps", "lives_lost", "model_calls",
                                             "latency_ms_p50", "latency_ms_p95", "served_model")}), flush=True)
        if a.json_out:
            with open(a.json_out, "a") as f:
                f.write(json.dumps(run) + "\n")
        if not a.no_record:
            record(run, decider, a.notes, a.max_steps, push=not a.no_push)


if __name__ == "__main__":
    sys.exit(main())
