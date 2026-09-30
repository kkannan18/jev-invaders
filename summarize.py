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


def _by_seed(runs):
    out = defaultdict(list)
    for r in runs:
        out[r["seed"]].append(r["score"])
    return {k: st.mean(v) for k, v in out.items()}


def head_to_head(a, b):
    """Per-seed wins/ties/losses of a over b (mean score per seed), on seeds both played."""
    sa, sb = _by_seed(a), _by_seed(b)
    w = t = l = 0
    for s in sorted(set(sa) & set(sb)):
        w += sa[s] > sb[s]
        t += sa[s] == sb[s]
        l += sa[s] < sb[s]
    return {"wins": w, "ties": t, "losses": l,
            "per_seed": {s: [sa[s], sb[s]] for s in sorted(set(sa) & set(sb))}}


def shared(a, b):
    seeds = {r["seed"] for r in a} & {r["seed"] for r in b}
    return [r for r in a if r["seed"] in seeds], [r for r in b if r["seed"] in seeds]


def variant_of(r):
    return r.get("variant") or "six"


def is_plain(r):
    return (not r.get("conf_threshold") and r.get("policy", "argmax") == "argmax" and not r.get("latency_tax_ms")
            and r.get("phase") != "tune")


def build(path="results.json"):
    d = json.loads(Path(path).read_text())
    champ = None
    cf = Path(path).resolve().parent / "champion.json"
    if cf.exists():
        champ = json.loads(cf.read_text())
    jev_all = [r for r in d["runs"] if str(r.get("served_model", "")).startswith("jev")]
    base = [r for r in d["baseline"]["runs"] if variant_of(r) == "six"]
    rt = d.get("realtime_showdown", {})
    rj_all, rb = rt.get("jev", []), [r for r in rt.get("baseline", []) if variant_of(r) == "six"]

    def pick(runs, variant):
        return [r for r in runs if is_plain(r) and variant_of(r) == variant]

    jev, rj = pick(jev_all, "six"), pick(rj_all, "six")
    jev_p, base_p = shared(jev, base)
    rj_p, rb_p = shared(rj, rb)
    gate = defaultdict(list)
    for r in jev_all:
        if r.get("policy", "argmax") == "argmax" and variant_of(r) == "six" and r.get("phase") != "tune":
            gate[r.get("conf_threshold") or 0.0].append(r)
    tune = defaultdict(list)
    for r in jev_all:
        if r.get("phase") == "tune":
            tune[f"{variant_of(r)}/{r.get('policy', 'argmax')}"].append(r)
    J, B = agg(jev_p), agg(base_p)
    out = {
        "models": {m["role"] + ":" + m.get("requested_model", ""): m.get("served_model") for m in d["models"]},
        "turn_based": {"jev": J, "baseline": B, "head_to_head": head_to_head(jev_p, base_p)},
        "realtime": {"jev": agg(rj_p), "baseline": agg(rb_p), "head_to_head": head_to_head(rj_p, rb_p)},
        "confidence_gate": {str(t): agg(v) for t, v in sorted(gate.items())},
        "latency_tax": agg([r for r in rj_all if r.get("latency_tax_ms")]),
        "tuning": {k: agg(v) for k, v in sorted(tune.items(), key=lambda kv: -st.mean(r["score"] for r in kv[1]))},
        "champion": champ,
    }
    if champ:
        cv = champ["variant"]
        cp = champ.get("policy", "argmax")

        def is_champ(r):
            return (variant_of(r) == cv and r.get("policy", "argmax") == cp and r.get("phase") not in ("tune", "coach")
                    and not r.get("latency_tax_ms") and not r.get("conf_threshold") and not r.get("coach"))
        cj = [r for r in jev_all if is_champ(r)]
        crt = [r for r in rj_all if is_champ(r)]
        cj_p, cb_p = shared(cj, base)
        crt_p, crb_p = shared(crt, rb)
        out["champion_test"] = {
            "turn_based": {"jev": agg(cj_p), "baseline": agg(cb_p), "head_to_head": head_to_head(cj_p, cb_p)},
            "realtime": {"jev": agg(crt_p), "baseline": agg(crb_p), "head_to_head": head_to_head(crt_p, crb_p)},
        }
    if J and B:
        out["ratios"] = {
            "speedup_p50": round(B["latency_ms_p50"] / J["latency_ms_p50"], 1),
            "cost_per_decision_ratio": round(B["cost_usd_per_decision"] / J["cost_usd_per_decision"], 1),
            "score_ratio": round(J["mean_score"] / B["mean_score"], 2),
        }
    if out["realtime"]["jev"] and out["realtime"]["baseline"]:
        out.setdefault("ratios", {})["realtime_score_ratio"] = round(
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
    table("TURN-BASED, original design, on seeds both played (the game waits for each decision)",
          [["JEV", J["games"], J["mean_score"], J["human_normalized"], J["latency_ms_p50"], J["latency_ms_p95"],
            usd(J["cost_usd_per_decision"]), f"${J['cost_usd_total']:.2f}", J["errors"]],
           ["LLM baseline", B["games"], B["mean_score"], B["human_normalized"], B["latency_ms_p50"],
            B["latency_ms_p95"], usd(B["cost_usd_per_decision"]), f"${B['cost_usd_total']:.2f}", B["errors"]]],
          ["decider", "games", "mean score", "human-norm", "p50 ms", "p95 ms", "$/decision", "$ total", "errors"])
    h = tb["head_to_head"]
    print(f"  head to head by seed: JEV {h['wins']} wins, {h['ties']} ties, {h['losses']} losses")
    if rt["jev"] and rt["baseline"]:
        RJ, RB = rt["jev"], rt["baseline"]
        table("REAL-TIME, original design, on seeds both played (15 steps/s; stale moves repeat)",
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
    if s["latency_tax"] and s["realtime"]["jev"]:
        T, RJ = s["latency_tax"], s["realtime"]["jev"]
        table("LATENCY-TAX CONTROL (real-time JEV with a delay added to match the LLM's speed)",
              [["JEV", RJ["games"], RJ["mean_score"], RJ["latency_ms_p50"]],
               ["JEV + delay", T["games"], T["mean_score"], T["latency_ms_p50"]]],
              ["decider", "games", "mean score", "p50 ms"])
    if s["tuning"]:
        table("TENKI TUNING (question designs raced on held-out seeds, one sandbox per game)",
              [[k, v["games"], v["seeds"], v["mean_score"], v["latency_ms_p50"], v.get("mean_confidence")]
               for k, v in s["tuning"].items()],
              ["design/policy", "games", "seeds", "mean score", "p50 ms", "mean confidence"])
        if s["champion"]:
            print(f"  champion: {s['champion']['variant']} (chosen on seeds {s['champion']['tuned_on_seeds']})")
    ct = s.get("champion_test")
    if ct:
        for mode, title in (("turn_based", "CHAMPION vs LLM, turn-based, on the baseline's seeds"),
                            ("realtime", "CHAMPION vs LLM, real-time, on the baseline's seeds")):
            c = ct[mode]
            if c["jev"] and c["baseline"]:
                table(title, [["JEV champion", c["jev"]["games"], len(set(c["jev"]["seeds"])), c["jev"]["mean_score"],
                               c["jev"].get("decisions_per_game", "")],
                              ["LLM baseline", c["baseline"]["games"], len(set(c["baseline"]["seeds"])),
                               c["baseline"]["mean_score"], c["baseline"].get("decisions_per_game", "")]],
                      ["decider", "games", "seeds", "mean score (all games)", "decisions per game"])
                h = c["head_to_head"]
                pj = st.mean(v[0] for v in h["per_seed"].values())
                pb = st.mean(v[1] for v in h["per_seed"].values())
                print(f"  per-seed means: JEV {pj:.1f} vs LLM {pb:.1f} ({pj / pb:.2f}x); "
                      f"head to head by seed: JEV {h['wins']} wins, {h['ties']} ties, {h['losses']} losses")
    print(f"\nJEV is {ra.get('speedup_p50')}x faster per decision (median) and "
          f"{ra.get('cost_per_decision_ratio')}x cheaper per decision; "
          f"mean score {ra.get('score_ratio')}x turn-based, {ra.get('realtime_score_ratio')}x real-time.")


if __name__ == "__main__":
    main()
