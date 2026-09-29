#!/usr/bin/env python3
"""CI replay: on every push, replay reference games inside fresh Tenki sandboxes and
check that the scores reproduce exactly. Runs on GitHub Actions (see
.github/workflows/tenki-replay.yml) with the TENKI_API_KEY repository secret.

The reference games use the keyless scripted policy, which is deterministic for a
given seed, so any change to the environment, encoder or game loop that alters
play shows up as a failed check on the commit.
"""
import os
import sys
from types import SimpleNamespace

from tenki_arena import run_job

REFERENCE = {1: 710, 2: 330}  # scripted policy, full games (Mac, Linux container and Tenki all agree)

args = SimpleNamespace(max_steps=None, realtime=False, model=None, baseline_model=None, provider="anthropic",
                       policy="argmax", latency_tax_ms=0, lockdown=False, verbose=False)
lines, ok = ["| seed | expected | replayed in Tenki | sandbox | result |", "|---|---|---|---|---|"], True
for seed, want in REFERENCE.items():
    run = run_job("scripted", seed, 0.0, args)
    got = run["score"]
    ok &= got == want
    lines.append(f"| {seed} | {want} | {got} | `{run['tenki_sandbox_id']}` | {'match' if got == want else 'MISMATCH'} |")
report = "### Tenki replay\n\n" + "\n".join(lines) + "\n"
print(report)
if os.environ.get("GITHUB_STEP_SUMMARY"):
    with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
        f.write(report)
sys.exit(0 if ok else 1)
