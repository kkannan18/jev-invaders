"""results.json: append one game, sanity-check it, commit and push right away."""
from __future__ import annotations

import json
import os
import subprocess
import threading
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from .env import ACTION_REPEAT, ENV_ID, MAX_FRAMES, REPEAT_ACTION_PROBABILITY

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results.json"
_lock = threading.Lock()


def _v(pkg):
    try:
        return version(pkg)
    except PackageNotFoundError:
        return None


def config(max_steps=None):
    c = {
        "env_id": ENV_ID,
        "frameskip": ACTION_REPEAT,
        "repeat_action_probability": REPEAT_ACTION_PROBABILITY,
        "full_action_space": False,
        "max_num_frames_per_episode": MAX_FRAMES,
        "obs_type": "ram",
        "wrappers": ["harness action repeat x4 (env frameskip=1) with the 4 screens unioned to beat sprite flicker"],
        "decision_interval_steps": 1,
        "state_encoding": "Screen pixels of the 4 repeated frames decoded to ship x, alien positions, bombs with "
                          "direction and steps-to-impact, own missile, shields remaining, lives, score, plus derived "
                          "danger/safe-side/nearest-column features, as JSON (same JSON for every decider)",
        "ale_py_version": _v("ale-py"),
        "gymnasium_version": _v("gymnasium"),
    }
    if max_steps:
        c["max_steps"] = max_steps
    return c


def load():
    if RESULTS.exists():
        return json.loads(RESULTS.read_text())
    return {"schema_version": 2, "models": [], "config": config(), "runs": [], "baseline": {"model": None, "runs": []}}


def check(run):
    problems = []
    p50, p95 = run.get("latency_ms_p50"), run.get("latency_ms_p95")
    if p50 is not None and p95 is not None and p95 < p50:
        problems.append("p95 < p50")
    if run.get("model_calls", 0) > run.get("steps", 0):
        problems.append("model_calls > steps")
    if run.get("frames", 0) > run.get("steps", 0) * ACTION_REPEAT:
        problems.append("frames > steps*4")
    if problems:
        raise ValueError(f"inconsistent run, not recording: {problems}")


def _upsert_model(data, decider, role, served):
    entry = {"role": role, "provider": decider.provider, "requested_model": decider.requested_model,
             "served_model": served, "sdk_package": decider.sdk_package, "sdk_version": decider.sdk_version}
    entry = {k: v for k, v in entry.items() if v is not None}
    for i, m in enumerate(data["models"]):
        if m.get("role") == role and m.get("requested_model") == decider.requested_model:
            data["models"][i] = entry
            return
    data["models"].append(entry)


def total_runs(data):
    n = len(data["runs"]) + len(data["baseline"]["runs"])
    rt = data.get("realtime_showdown", {})
    return n + sum(len(v) for v in rt.values())


def record(run, decider, notes: str = "", max_steps=None, push: bool = True):
    """Append the run where it belongs, write results.json, then git add/commit/push."""
    check(run)
    with _lock:
        data = load()
        data["config"] = config(max_steps or data["config"].get("max_steps"))
        role = "baseline" if decider.name == "baseline" else "decider"
        _upsert_model(data, decider, role, run.get("served_model"))
        if notes:
            run["notes"] = notes
        if run.get("mode") == "realtime_15hz":
            data.setdefault("realtime_showdown", {}).setdefault(decider.name, []).append(run)
        elif role == "baseline":
            data["baseline"]["model"] = decider.requested_model
            data["baseline"]["runs"].append(run)
        else:
            data["runs"].append(run)
        n = total_runs(data)
        RESULTS.write_text(json.dumps(data, indent=2) + "\n")
        if push and os.environ.get("NO_PUSH") != "1":
            git_push(f"results: run {n}, score {run['score']} ({run['served_model']}, seed {run['seed']})")
        return n


def git_push(msg):
    def git(*a):
        return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True)

    git("add", "results.json")
    c = git("commit", "-m", msg)
    if c.returncode != 0 and "nothing to commit" not in c.stdout:
        print("git commit failed:", c.stderr.strip())
        return False
    p = git("push")
    if p.returncode != 0:
        # someone (another machine / CI) pushed first: rebase onto it and retry once
        if git("pull", "--rebase").returncode != 0:
            git("rebase", "--abort")
        p = git("push")
    print("pushed:" if p.returncode == 0 else "PUSH FAILED:", msg, p.stderr.strip()[-200:] if p.returncode else "")
    return p.returncode == 0
