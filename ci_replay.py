#!/usr/bin/env python3
"""CI replay: on every push, replay games inside fresh Tenki sandboxes and post the scores.

1. Reproducibility check. The keyless scripted policy is deterministic for a seed, so its
   scores must match exactly; any change to the environment, encoder or game loop that
   alters play fails the check.
2. JEV score for this commit. When the TYPESAFE_API_KEY secret is set, the current
   champion design (or the original design if there is no champion.json) plays one
   real game in a Tenki sandbox and its score is posted on the commit.

Secrets: TENKI_API_KEY (required), TYPESAFE_API_KEY (optional).
"""
import json
import os
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

REFERENCE = {1: 710, 2: 330}  # scripted policy, full games (Mac, Linux container and Tenki all agree)


def args(**kw):
    base = dict(max_steps=None, realtime=False, model=None, baseline_model=None, provider="anthropic",
                policy=None, latency_tax_ms=0, lockdown=False, verbose=False, variant=None)
    base.update(kw)
    return SimpleNamespace(**base)


def main():
    lines, ok = [], True
    if not os.environ.get("TENKI_API_KEY"):
        lines.append("**TENKI_API_KEY secret is not set** (repo Settings > Secrets and variables > Actions).")
        ok = False
    else:
        from tenki_arena import run_job
        lines += ["| check | seed | expected | got | Tenki sandbox | result |", "|---|---|---|---|---|---|"]
        for seed, want in REFERENCE.items():
            try:
                run = run_job("scripted", seed, 0.0, args())
                got = run["score"]
                ok &= got == want
                lines.append(f"| replay (scripted) | {seed} | {want} | {got} | `{run['tenki_sandbox_id']}` | "
                             f"{'match' if got == want else 'MISMATCH'} |")
            except Exception as exc:  # noqa: BLE001
                ok = False
                traceback.print_exc()
                lines.append(f"| replay (scripted) | {seed} | {want} | error | | {str(exc)[:80]} |")
        if os.environ.get("TYPESAFE_API_KEY"):
            variant = "champion" if Path("champion.json").exists() else None
            label = json.loads(Path("champion.json").read_text())["design"] if variant else "six"
            try:
                run = run_job("jev", 7, 0.0, args(variant=variant))
                lines.append(f"| JEV score, design {label} | 7 | - | **{run['score']}** "
                             f"({run['model_calls']} decisions, p50 {run['latency_ms_p50']} ms) | "
                             f"`{run['tenki_sandbox_id']}` | posted |")
            except Exception as exc:  # noqa: BLE001
                traceback.print_exc()
                lines.append(f"| JEV score | 7 | - | error | | {str(exc)[:80]} |")
    report = "### Tenki replay\n\n" + "\n".join(lines) + "\n"
    print(report)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
            f.write(report)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
