#!/usr/bin/env python3
"""Print the JEV vs LLM scoreboard straight from results.json.

  python summarize.py            # tables for the terminal
  python summarize.py --json     # the same numbers as JSON (used by the scoreboard page)

Every number here is recomputed from the per-game records the harness wrote;
nothing is typed in by hand.
"""
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

HUMAN, RANDOM = 1668.7, 148.0  # human-normalized score anchors used by the judges


def norm(score):
    return (score - RANDOM) / (HUMAN - RANDOM)


def agg(runs):
    if not runs:
        return None
    calls = sum(r.get("model_calls", 0) for r in runs)
    cost = sum(r.get("cost_usd", 0) for r in runs)
    steps = sum(r["steps"] for r in runs)
    out = {
        "games": len(runs),
        "seeds": sorted(r["seed"] for r in runs),
        "mean_score": round(st.mean(r["score"] for r in runs), 1),
        "median_score": st.median(r["score"] for r in runs),
        "human_normalized": round(norm(st.mean(r["score"] for r in runs)), 3),
        "decisions": calls,
        "latency_ms_p50": st.median(r["latency_ms_p50"] for r in runs),
        "latency_ms_p95": st.median(r["latency_ms_p95"] for r in runs),
        "cost_usd_total": round(cost, 4),
        "cost_usd_per_decision": cost / calls if calls else None,
        "errors": sum(sum(r.get("errors_by_status", {}).values()) for r in runs),
        "fallbacks": sum(r.get("fallback_actions", 0) for r in runs),
    }
    if all("mean_confidence" in r for r in runs):
        out["mean_confidence"] = round(st.mean(r["mean_confidence"] for r in runs), 3)
    if all("stale_steps" in r for r in runs):
        stale = sum(r["stale_steps"] for r in runs)
        out["fresh_decision_rate"] = round(1 - stale / steps, 3)
        out["decisions_per_game"] = [r["model_calls"] for r in sorted(runs, key=lambda r: r["seed"])]
    return out


def head_to_head(a, b):
    """Per-seed wins/ties/losses of a over b, on seeds both played."""
    sa, sb = {r["seed"]: r["score"] for r in a}, {r["seed"]: r["score"] for r in b}
    w = t = l = 0
    for s in sorted(set(sa) & set(sb)):
        w += sa[s] > sb[s]
        t += sa[s] == sb[s]
        l += sa[s] < sb[s]
    return {"wins": w, "ties": t, "losses": l,
            "per_seed": {s: [sa[s], sb[s]] for s in sorted(set(sa) & set(sb))}}


def build(path="results.json"):
    d = json.loads(Path(path).read_text())
    jev_all = [r for r in d["runs"] if str(r.get("served_model", "")).startswith("jev")]
    jev = [r for r in jev_all if not r.get("conf_threshold") and r.get("policy", "argmax") == "argmax"]
    axes = [r for r in jev_all if r.get("policy") == "axes"]
    base = d["baseline"]["runs"]
    gate = defaultdict(list)
    for r in jev_all:
        if r.get("policy", "argmax") == "argmax":
            gate[r.get("conf_threshold") or 0.0].append(r)
    rt = d.get("realtime_showdown", {})
    rj_all, rb = rt.get("jev", []), rt.get("baseline", [])
    rj = [r for r in rj_all if r.get("policy", "argmax") == "argmax" and not r.get("latency_tax_ms")]
    rj_tax = [r for r in rj_all if r.get("latency_tax_ms")]
    rj_axes = [r for r in rj_all if r.get("policy") == "axes"]
    J, B = agg(jev), agg(base)
    out = {
        "models": {m["role"] + ":" + m.get("requested_model", ""): m.get("served_model") for m in d["models"]},
        "turn_based": {"jev": J, "baseline": B, "head_to_head": head_to_head(jev, base)},
        "realtime": {"jev": agg(rj), "baseline": agg(rb), "head_to_head": head_to_head(rj, rb)},
        "confidence_gate": {str(t): agg(v) for t, v in sorted(gate.items())},
        "latency_tax": agg(rj_tax),
        "axes_policy": {"turn_based": agg(axes), "realtime": agg(rj_axes)},
    }
    if J and B:
        out["ratios"] = {
            "speedup_p50": round(B["latency_ms_p50"] / J["latency_ms_p50"], 1),
            "cost_per_decision_ratio": round(B["cost_usd_per_decision"] / J["cost_usd_per_decision"], 1),
            "score_ratio": round(J["mean_score"] / B["mean_score"], 2),
        }
    if out["realtime"]["jev"] and out["realtime"]["baseline"]:
        out["ratios"]["realtime_score_ratio"] = round(
            out["realtime"]["jev"]["mean_score"] / out["realtime"]["baseline"]["mean_score"], 2)
    return out


def table(title, rows, cols):
    print(f"\n{title}")
    w = [max(len(str(c)), *(len(str(r[i])) for r in rows)) for i, c in enumerate(cols)]
    print("  " + "  ".join(str(c).ljust(w[i]) for i, c in enumerate(cols)))
    for r in rows:
        print("  " + "  ".join(str(v).ljust(w[i]) for i, v in enumerate(r)))


def main():
    s = build(sys.argv[2] if len(sys.argv) > 2 else "results.json")
    if "--json" in sys.argv:
        print(json.dumps(s, indent=2))
        return
    tb, rt, ra = s["turn_based"], s["realtime"], s.get("ratios", {})
    J, B = tb["jev"], tb["baseline"]
    usd = lambda x: f"${x:.6f}" if x is not None else "-"
    table("TURN-BASED (the game waits for each decision)",
          [["JEV", J["games"], J["mean_score"], J["human_normalized"], J["latency_ms_p50"], J["latency_ms_p95"],
            usd(J["cost_usd_per_decision"]), f"${J['cost_usd_total']:.2f}", J["errors"]],
           ["LLM baseline", B["games"], B["mean_score"], B["human_normalized"], B["latency_ms_p50"],
            B["latency_ms_p95"], usd(B["cost_usd_per_decision"]), f"${B['cost_usd_total']:.2f}", B["errors"]]],
          ["decider", "games", "mean score", "human-norm", "p50 ms", "p95 ms", "$/decision", "$ total", "errors"])
    h = tb["head_to_head"]
    print(f"  head to head by seed: JEV {h['wins']} wins, {h['ties']} ties, {h['losses']} losses")
    if rt["jev"] and rt["baseline"]:
        RJ, RB = rt["jev"], rt["baseline"]
        table("REAL-TIME (15 steps/s; the ship repeats its last move until the next answer lands)",
              [["JEV", RJ["games"], RJ["mean_score"], RJ["fresh_decision_rate"], RJ["decisions_per_game"]],
               ["LLM baseline", RB["games"], RB["mean_score"], RB["fresh_decision_rate"], RB["decisions_per_game"]]],
              ["decider", "games", "mean score", "fresh-decision rate", "decisions per game"])
        h = rt["head_to_head"]
        print(f"  head to head by seed: JEV {h['wins']} wins, {h['ties']} ties, {h['losses']} losses")
    g = s["confidence_gate"]
    if len(g) > 1:
        table("CONFIDENCE GATE (hold the last move when JEV's confidence is below the threshold)",
              [[t, v["games"], v["mean_score"], v.get("mean_confidence")] for t, v in g.items()],
              ["threshold", "games", "mean score", "mean confidence"])
    if s["latency_tax"]:
        T, RJ = s["latency_tax"], s["realtime"]["jev"]
        table("LATENCY-TAX CONTROL (real-time JEV with a delay added to match the LLM's speed)",
              [["JEV", RJ["games"], RJ["mean_score"], RJ["latency_ms_p50"]],
               ["JEV + delay", T["games"], T["mean_score"], T["latency_ms_p50"]]],
              ["decider", "games", "mean score", "p50 ms"])
    ax = s["axes_policy"]
    if ax["turn_based"] or ax["realtime"]:
        rows = [[m, v["games"], v["mean_score"]] for m, v in (("turn-based", ax["turn_based"]), ("real-time", ax["realtime"])) if v]
        table("AXES POLICY (JEV's probabilities summed into move and fire axes)", rows, ["mode", "games", "mean score"])
    print(f"\nJEV is {ra.get('speedup_p50')}x faster per decision (median) and "
          f"{ra.get('cost_per_decision_ratio')}x cheaper per decision; "
          f"mean score {ra.get('score_ratio')}x turn-based, {ra.get('realtime_score_ratio')}x real-time.")


if __name__ == "__main__":
    main()
