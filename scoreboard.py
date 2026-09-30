#!/usr/bin/env python3
"""Build scoreboard.html from results.json (and champion.json), so every number on the
page is the number the harness logged. live_race.py calls this when a race ends and
opens the page with that race at the top.

  python scoreboard.py            # writes scoreboard.html
  python scoreboard.py --open     # ...and opens it
"""
from __future__ import annotations

import json
import statistics as st
import sys
import webbrowser
from collections import defaultdict
from pathlib import Path

from summarize import build, is_plain, shared, variant_of

ROOT = Path(__file__).resolve().parent


def latest_live_race(d):
    rt = d.get("realtime_showdown", {})
    js = [r for r in rt.get("jev", []) if r.get("live_race")]
    bs = [r for r in rt.get("baseline", []) if r.get("live_race")]
    for j in reversed(js):
        for b in reversed(bs):
            if b["seed"] == j["seed"]:
                return {"seed": j["seed"], "jev": j, "claude": b, "count": min(len(js), len(bs))}
    return None


def per_seed(runs, key):
    g = defaultdict(list)
    for r in runs:
        g[r["seed"]].append(r[key])
    return {s: round(st.mean(v)) for s, v in g.items()}


def payload():
    d = json.loads((ROOT / "results.json").read_text())
    s = build(str(ROOT / "results.json"))
    rt = d.get("realtime_showdown", {})
    champ = s.get("champion")
    if champ and (s.get("champion_test") or {}).get("realtime", {}).get("jev"):
        cv, cp = champ["variant"], champ.get("policy", "argmax")
        rj = [r for r in rt.get("jev", []) if variant_of(r) == cv and r.get("policy", "argmax") == cp
              and r.get("phase") not in ("tune", "coach") and not r.get("coach")
              and not r.get("latency_tax_ms") and not r.get("conf_threshold")]
        real_src = s["champion_test"]["realtime"]
        design = champ["design"]
    else:
        rj = [r for r in rt.get("jev", []) if is_plain(r) and variant_of(r) == "six"]
        real_src, design = None, "six"
    rb = [r for r in rt.get("baseline", []) if variant_of(r) == "six"]
    rj, rb = shared(rj, rb)
    seeds = sorted({r["seed"] for r in rj})
    sj, sb = per_seed(rj, "score"), per_seed(rb, "score")
    dj, db = per_seed(rj, "model_calls"), per_seed(rb, "model_calls")
    live = latest_live_race(d)

    def slim(r):
        return {k: r.get(k) for k in ("score", "model_calls", "steps", "stale_steps", "latency_ms_p50",
                                      "tenki_sandbox_id", "served_model", "variant", "lives_lost")}

    coach = [r for r in rt.get("jev", []) if r.get("coach")]
    tb, rtm = s["turn_based"], s["realtime"]
    return {
        "live": None if not live else {"seed": live["seed"], "races": live["count"],
                                       "jev": slim(live["jev"]), "claude": slim(live["claude"])},
        "turn": {"jev": tb["jev"], "claude": tb["baseline"], "h2h": tb["head_to_head"]},
        "real": {"jev": (real_src or rtm)["jev"], "claude": (real_src or rtm)["baseline"],
                 "h2h": (real_src or rtm)["head_to_head"], "design": design,
                 "seed_mean_jev": round(st.mean(sj.values()), 1), "seed_mean_claude": round(st.mean(sb.values()), 1),
                 "seeds": seeds, "score": [[x, sj[x], sb[x]] for x in seeds],
                 "decisions": [[x, dj[x], db[x]] for x in seeds]},
        "tax": s.get("latency_tax"),
        "ctrl": {"jev": rtm["jev"], "claude": rtm["baseline"]},  # the control ran on the original design, seeds 1-5
        "ratios": s.get("ratios", {}),
        "rt_ratio": round(st.mean(sj.values()) / st.mean(sb.values()), 2) if sb else None,
        "coach": [{"seed": r["seed"], "pick": r["coach"]["strategy"], "conf": r["coach"]["confidence"],
                   "ms": r["coach"].get("decision_ms"), "score": r["score"],
                   "memories": len(r["coach"].get("cortex_evidence_ids", []))} for r in coach],
        "tuning": s.get("tuning") or {},
        "champion": s.get("champion"),
        "champion_test": s.get("champion_test"),
        "games_logged": len(d["runs"]) + len(d["baseline"]["runs"]) + sum(len(v) for v in rt.values()),
    }


TEMPLATE = r"""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>JEV Invaders Scoreboard</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@500;700&display=swap" rel="stylesheet">
<style>
/* Layout: arcade-cabinet dark scoreboard. Latest live race first, then speed, then proof. */
:root{
  --bg:#0b0e14; --panel:#121722; --panel2:#171d2b; --line:#262e40; --grid:#1e2534;
  --ink:#f2f5fb; --ink2:#c3cad8; --ink3:#8f99ad;
  --jev:#23a87c; --jev-soft:rgba(35,168,124,.16); --cl:#e0703c; --cl-soft:rgba(224,112,60,.16);
  --gold:#f3c969;
  --display:"Space Grotesk","Inter",system-ui,sans-serif; --body:"Inter",system-ui,sans-serif;
  --mono:"JetBrains Mono",ui-monospace,Menlo,monospace;
  color-scheme:dark;
}
*{box-sizing:border-box}
html,body{margin:0;background:var(--bg);color:var(--ink);font:18px/1.55 var(--body);-webkit-font-smoothing:antialiased}
main{max-width:1040px;margin:0 auto;padding-inline:20px;padding-block:32px 64px;display:grid;grid-template-columns:minmax(0,1fr);gap:44px}
main>*,.chart,.live,.vs>*{min-width:0}
.eyebrow{font:600 13px var(--mono);letter-spacing:.14em;text-transform:uppercase;color:var(--ink3);margin:0}
h1{font:700 clamp(30px,4.6vw,52px)/1.08 var(--display);margin:10px 0 14px;letter-spacing:-.02em;text-wrap:balance}
h2{font:700 26px/1.2 var(--display);margin:0 0 6px;letter-spacing:-.01em}
.lede,.note{color:var(--ink2);margin:0;max-width:64ch}
.note{font-size:16.5px;margin-bottom:16px}
.j{color:var(--jev)} .c{color:var(--cl)}
.jt{color:#5fd3a8} .ct{color:#f29a6d} /* text-contrast variants of the series hues */
.dot{display:inline-block;width:12px;height:12px;border-radius:3px;margin-right:8px;vertical-align:-1px}
.dot.j{background:var(--jev)} .dot.c{background:var(--cl)}
/* live race */
.live{background:linear-gradient(180deg,var(--panel2),var(--panel));border:1px solid var(--line);border-radius:14px;padding:22px;display:grid;gap:18px}
.live-head{display:flex;flex-wrap:wrap;justify-content:space-between;gap:8px;align-items:baseline}
.pill{font:600 12px var(--mono);letter-spacing:.1em;text-transform:uppercase;padding:5px 10px;border-radius:999px;background:var(--jev-soft);color:#7fe0bb;border:1px solid rgba(35,168,124,.45)}
.vs{display:grid;grid-template-columns:1fr auto 1fr;gap:16px;align-items:stretch}
@media (max-width:720px){.vs{grid-template-columns:1fr}.vs .mid{display:none}}
.side{border-radius:10px;padding:18px;border:1px solid var(--line);background:var(--bg);display:grid;gap:10px;min-width:0}
.side.win{border-color:rgba(35,168,124,.7);box-shadow:0 0 0 1px rgba(35,168,124,.25) inset}
.side h3{margin:0;font:700 20px var(--display);display:flex;align-items:center;justify-content:space-between;gap:8px}
.crown{font:600 12px var(--mono);letter-spacing:.08em;text-transform:uppercase;color:var(--gold)}
.big{font:700 64px/1 var(--display);font-variant-numeric:tabular-nums;letter-spacing:-.02em}
.kv{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}
.kv div{border-top:1px solid var(--line);padding-top:8px;min-width:0}
.kv b{display:block;font:700 22px var(--mono);font-variant-numeric:tabular-nums}
.kv span{font:500 12.5px var(--mono);letter-spacing:.06em;text-transform:uppercase;color:var(--ink3)}
.mid{display:grid;place-items:center;font:700 22px var(--display);color:var(--ink3)}
.sbx{font:13px var(--mono);color:var(--ink3);word-break:break-all}
/* tiles */
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:14px}
.tile{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px 18px}
.tile b{display:block;font:700 44px/1.05 var(--display);color:#5fd3a8;font-variant-numeric:tabular-nums}
.tile span{font:500 14px var(--body);color:var(--ink2)}
/* charts */
.chart{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:20px;display:grid;gap:14px}
.legend{display:flex;flex-wrap:wrap;gap:20px;font:500 15px var(--body);color:var(--ink2)}
.row{display:grid;grid-template-columns:130px 1fr;gap:14px;align-items:center}
.row .k{font:600 14px var(--mono);color:var(--ink2)}
@media (max-width:560px){.row{grid-template-columns:1fr;gap:6px}}
.bars{display:grid;gap:6px;min-width:0}
.bar{display:flex;align-items:center;gap:10px;min-width:0}
.bar i{display:block;height:24px;border-radius:0 4px 4px 0;flex:0 0 auto;min-width:6px}
.bar i.j{background:var(--jev)} .bar i.c{background:var(--cl)}
.bar em{font:600 15px var(--mono);font-style:normal;color:var(--ink);white-space:nowrap;font-variant-numeric:tabular-nums}
.bar:hover em{color:var(--gold)}
/* one-second timeline */
.tl{display:grid;gap:18px}
.lane{display:grid;grid-template-columns:170px 1fr;gap:14px;align-items:center}
@media (max-width:560px){.lane{grid-template-columns:1fr}}
.track{position:relative;height:44px;border-radius:8px;background:var(--bg);border:1px solid var(--line);overflow:hidden}
.tick{position:absolute;top:6px;bottom:6px;width:6px;margin-left:-3px;border-radius:3px}
.tick.j{background:var(--jev)} .tick.c{background:var(--cl)}
.lane .cnt{font:700 15px var(--mono);color:var(--ink);margin-top:2px}
.track .lbl{position:absolute;left:50%;top:50%;transform:translate(-50%,-50%);white-space:nowrap;font:600 14px var(--mono);color:var(--ink);background:rgba(11,14,20,.85);padding:2px 8px;border-radius:6px}
.axis{display:flex;justify-content:space-between;font:12.5px var(--mono);color:var(--ink3);margin-left:184px}
@media (max-width:560px){.axis{margin-left:0}}
/* tables */
.tbl{overflow-x:auto;border:1px solid var(--line);border-radius:12px;background:var(--panel)}
table{border-collapse:collapse;width:100%;font:15.5px var(--mono);font-variant-numeric:tabular-nums}
th,td{padding:11px 14px;text-align:right;border-bottom:1px solid var(--grid);white-space:nowrap}
th:first-child,td:first-child{text-align:left}
tr:last-child td{border-bottom:0}
thead th{color:var(--ink3);font:600 12.5px var(--mono);letter-spacing:.07em;text-transform:uppercase}
td.win{color:#5fd3a8;font-weight:700}
.callout{border-left:3px solid var(--gold);padding:6px 0 6px 16px;color:var(--ink2);max-width:66ch;margin:14px 0 0}
pre{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px;overflow-x:auto;font:15px/1.65 var(--mono);margin:0;color:var(--ink2)}
a{color:#5fd3a8} a:focus-visible{outline:2px solid var(--gold);outline-offset:2px}
footer{color:var(--ink3);font-size:15px}
[hidden]{display:none!important}
@media (prefers-reduced-motion:no-preference){.tick{animation:pop .35s ease-out both}}
@keyframes pop{from{transform:scaleY(.2);opacity:.2}to{transform:none;opacity:1}}
</style>
<main>
  <header>
    <p class="eyebrow">UFA JEV Bake-Off · Pilot track · The Kan Man Can</p>
    <h1 id="headline"></h1>
    <p class="lede" id="lede"></p>
  </header>

  <section id="live-sec" hidden>
    <div class="live">
      <div class="live-head">
        <h2>Latest live race</h2><span class="pill" id="live-pill"></span>
      </div>
      <div class="vs">
        <div class="side" id="side-j"></div>
        <div class="mid">vs</div>
        <div class="side" id="side-c"></div>
      </div>
    </div>
  </section>

  <section class="tiles" id="tiles" aria-label="Headline numbers"></section>

  <section>
    <h2>One second of Space Invaders</h2>
    <p class="note">Each mark is an answer arriving, spaced by each model's median response time. The game moves 15 times in that second either way.</p>
    <div class="chart"><div class="tl" id="timeline"></div><div class="axis"><span>0 ms</span><span>250</span><span>500</span><span>750</span><span>1,000 ms</span></div></div>
  </section>

  <section>
    <h2>Time per decision</h2>
    <p class="note">Median and 95th-percentile time from sending the game state to getting the move back, turn-based games.</p>
    <div class="chart" id="c-lat"></div>
  </section>

  <section>
    <h2>Decisions per game, at real speed</h2>
    <p class="note">How many times each model got to decide in a full real-time game, same seeds.</p>
    <div class="chart" id="c-dec"></div>
  </section>

  <section>
    <h2>Real-time score by seed</h2>
    <p class="note" id="score-note"></p>
    <div class="chart" id="c-score"></div>
  </section>

  <section id="control-sec" hidden>
    <h2>The control: slow JEV down</h2>
    <p class="note">Same JEV (original design, seeds 1–5) with a delay added to every answer so it is exactly as slow as Claude.</p>
    <div class="tbl"><table><thead><tr><th>Real time</th><th>Median time per answer</th><th>Games</th><th>Mean score</th></tr></thead><tbody id="control"></tbody></table></div>
    <p class="callout" id="control-note"></p>
  </section>

  <section id="tune-sec" hidden>
    <h2>Tuned in Tenki</h2>
    <p class="note" id="tune-note"></p>
    <div class="tbl"><table><thead><tr><th>Question design</th><th>Games</th><th>Mean score</th><th>Median ms</th><th>Mean confidence</th></tr></thead><tbody id="tune"></tbody></table></div>
    <div class="tbl" style="margin-top:14px" id="champ-wrap" hidden><table><thead><tr><th>Champion vs Claude, seeds Claude played</th><th>Mode</th><th>JEV</th><th>Claude</th><th>Seeds won</th></tr></thead><tbody id="champ"></tbody></table></div>
  </section>

  <section id="coach-sec" hidden>
    <h2>JEV coaches itself, with Mitosis memory</h2>
    <p class="note">Every game is remembered in Mitosis Cortex. Before each new game JEV reads those memories and picks the design, with a typed answer and a confidence.</p>
    <div class="tbl"><table><thead><tr><th>Game seed</th><th>JEV's pick</th><th>Confidence</th><th>Decision time</th><th>Score</th></tr></thead><tbody id="coach"></tbody></table></div>
  </section>

  <section>
    <h2>Turn-based benchmark</h2>
    <p class="note">The emulator waits for every answer, so speed cannot cost points here.</p>
    <div class="tbl"><table><thead><tr><th></th><th>Games</th><th>Mean score</th><th>p50</th><th>p95</th><th>$ / decision</th><th>$ total</th><th>Errors</th></tr></thead><tbody id="turn"></tbody></table></div>
  </section>

  <section>
    <h2>Reproduce</h2>
    <pre>git clone https://github.com/kkannan18/jev-invaders && cd jev-invaders
pip install -r requirements.txt
python live_race.py --seed 3        # live race, two Tenki sandboxes, then this page
python tune.py --test-seeds 1 2 3 4 5
python scoreboard.py --open         # rebuild this page from results.json</pre>
  </section>

  <footer id="foot"></footer>
</main>
<script>
const D = __DATA__;
const $ = id => document.getElementById(id);
const fmt = n => n == null ? "–" : Number(n).toLocaleString();
const J = D.real.jev, C = D.real.claude, R = D.ratios;

$("headline").innerHTML = `At real Atari speed, <span class="ct">Claude</span> gets ${Math.min(...D.real.decisions.map(r=>r[2]))}–${Math.max(...D.real.decisions.map(r=>r[2]))} decisions a game. <span class="jt">JEV</span> gets up to ${fmt(Math.max(...D.real.decisions.map(r=>r[1])))}.`;
$("lede").textContent = `Same game, same seeds, same JSON state, same typed question. JEV answers in ${fmt(D.turn.jev.latency_ms_p50)} ms, Claude Haiku 4.5 in ${fmt(D.turn.claude.latency_ms_p50)} ms. At 15 moves a second, the slow answer is played blind.`;

// latest live race
if (D.live) {
  $("live-sec").hidden = false;
  const L = D.live, jw = L.jev.score > L.claude.score, cw = L.claude.score > L.jev.score;
  $("live-pill").textContent = `seed ${L.seed} · real time · ${L.races} live race${L.races>1?"s":""} logged`;
  const side = (el, name, cls, r, win) => {
    const fresh = r.steps ? Math.round(100 * (1 - (r.stale_steps||0) / r.steps)) : null;
    el.className = "side" + (win ? " win" : "");
    el.innerHTML = `<h3><span><span class="dot ${cls}"></span>${name}</span>${win ? '<span class="crown">winner</span>' : ''}</h3>
      <div class="big">${fmt(r.score)}</div>
      <div class="kv"><div><b>${fmt(r.model_calls)}</b><span>decisions</span></div>
      <div><b>${fresh==null?"–":fresh+"%"}</b><span>fresh moves</span></div>
      <div><b>${fmt(r.latency_ms_p50)} ms</b><span>median answer</span></div></div>
      <div class="sbx">Tenki sandbox ${r.tenki_sandbox_id || "local run"}</div>`;
  };
  side($("side-j"), "JEV 1.13", "j", L.jev, jw);
  side($("side-c"), "Claude Haiku 4.5", "c", L.claude, cw);
}

// tiles
const tiles = [
  [`${D.rt_ratio}×`, `real-time score vs Claude, ${D.real.seeds.length} seeds`],
  [`${R.speedup_p50}×`, "faster per decision (median)"],
  [`${R.cost_per_decision_ratio}×`, "cheaper per decision"],
  [`${D.real.h2h.wins} / ${D.real.h2h.wins + D.real.h2h.ties + D.real.h2h.losses}`, "real-time seeds won"],
];
$("tiles").innerHTML = tiles.map(([b, s]) => `<div class="tile"><b>${b}</b><span>${s}</span></div>`).join("");

// one second timeline
const lane = (name, cls, ms) => {
  const n = Math.floor(1000 / ms);
  let ticks = "";
  for (let i = 1; i <= n; i++) ticks += `<i class="tick ${cls}" style="left:${(i*ms/1000*100).toFixed(2)}%;animation-delay:${i*40}ms"></i>`;
  const label = `${n} answer${n === 1 ? "" : "s"}`;
  const wait = n ? "" : `<span class="lbl">still thinking · first answer at ${fmt(ms)} ms</span>`;
  return `<div class="lane"><div class="k"><div style="font:600 16px var(--mono)"><span class="dot ${cls}"></span>${name}</div>
    <div class="cnt">${label}</div></div>
    <div class="track" role="img" aria-label="${name}: ${label} in one second">${ticks}${wait}</div></div>`;
};
$("timeline").innerHTML = lane("JEV", "j", D.turn.jev.latency_ms_p50) + lane("Claude", "c", D.turn.claude.latency_ms_p50);

// bar charts: labels sit outside the bar so short bars stay readable
function chart(el, rows, unit) {
  const max = Math.max(...rows.flatMap(r => [r[1], r[2]]));
  const w = v => `calc(${(v / max * 100).toFixed(2)}% - ${(v/max)*90}px)`;
  el.innerHTML = `<div class="legend"><span><span class="dot j"></span>JEV 1.13</span><span><span class="dot c"></span>Claude Haiku 4.5</span></div>` +
    rows.map(([k, j, c]) => `<div class="row"><div class="k">${k}</div><div class="bars">
      <div class="bar" title="JEV · ${k}: ${fmt(j)}${unit}"><i class="j" style="width:${w(j)}"></i><em>${fmt(j)}${unit}</em></div>
      <div class="bar" title="Claude · ${k}: ${fmt(c)}${unit}"><i class="c" style="width:${w(c)}"></i><em>${fmt(c)}${unit}</em></div></div></div>`).join("");
}
chart($("c-lat"), [["median", D.turn.jev.latency_ms_p50, D.turn.claude.latency_ms_p50], ["p95", D.turn.jev.latency_ms_p95, D.turn.claude.latency_ms_p95]], " ms");
chart($("c-dec"), D.real.decisions.map(([s, j, c]) => ["seed " + s, j, c]), "");
chart($("c-score"), D.real.score.map(([s, j, c]) => ["seed " + s, j, c]), "");
$("score-note").textContent = `Mean score per seed over every real-time game on that seed (JEV flying the tuned ${D.real.design} design). Average across seeds: JEV ${D.real.seed_mean_jev}, Claude ${D.real.seed_mean_claude}. JEV won ${D.real.h2h.wins} of ${D.real.h2h.wins + D.real.h2h.ties + D.real.h2h.losses} seeds.`;

// control
if (D.tax) {
  $("control-sec").hidden = false;
  const J = D.ctrl.jev, C = D.ctrl.claude;
  $("control").innerHTML = `<tr><td class="win">JEV</td><td>${fmt(J.latency_ms_p50)} ms</td><td>${J.games}</td><td class="win">${J.mean_score}</td></tr>
    <tr><td>JEV slowed to Claude's speed</td><td>${fmt(D.tax.latency_ms_p50)} ms</td><td>${D.tax.games}</td><td>${D.tax.mean_score}</td></tr>
    <tr><td>Claude Haiku 4.5</td><td>${fmt(C.latency_ms_p50)} ms</td><td>${C.games}</td><td>${C.mean_score}</td></tr>`;
  const drop = Math.round(100 * (1 - D.tax.mean_score / J.mean_score));
  $("control-note").textContent = D.tax.mean_score > C.mean_score
    ? `Slowing JEV cost it ${drop}% of its score, so speed matters. Slowed down, it still beat Claude, so its decisions are better too.`
    : `Slowing JEV to Claude's speed cost it ${drop}% of its score: speed is what wins.`;
}

// tuning
const tk = Object.entries(D.tuning || {});
if (tk.length) {
  $("tune-sec").hidden = false;
  $("tune-note").textContent = `Question designs raced on held-out seeds ${D.champion ? D.champion.tuned_on_seeds.join(", ") : ""}, one Tenki sandbox per game. The winner flies the ship.`;
  $("tune").innerHTML = tk.map(([k, v], i) => `<tr><td class="${i===0?'win':''}">${k.replace("/argmax","")}${i===0?' · champion':''}</td><td>${v.games}</td><td class="${i===0?'win':''}">${v.mean_score}</td><td>${fmt(v.latency_ms_p50)}</td><td>${v.mean_confidence ?? "–"}</td></tr>`).join("");
  const ct = D.champion_test;
  if (ct) {
    const rows = [["turn_based", "turn-based"], ["realtime", "real time"]].filter(([m]) => ct[m].jev && ct[m].baseline);
    if (rows.length) {
      $("champ-wrap").hidden = false;
      $("champ").innerHTML = rows.map(([m, label]) => { const c = ct[m], h = c.head_to_head;
        return `<tr><td>${D.champion.design}</td><td>${label}</td><td class="${c.jev.mean_score > c.baseline.mean_score ? 'win' : ''}">${c.jev.mean_score}</td><td>${c.baseline.mean_score}</td><td>${h.wins} of ${h.wins + h.ties + h.losses}</td></tr>`; }).join("");
    }
  }
}

// coach
if (D.coach.length) {
  $("coach-sec").hidden = false;
  $("coach").innerHTML = D.coach.map(c => `<tr><td>${c.seed}</td><td>${c.pick}</td><td class="win">${c.conf.toFixed(2)}</td><td>${fmt(c.ms)} ms</td><td>${c.score}</td></tr>`).join("");
}

// turn-based table
const T = D.turn, usd = x => x == null ? "–" : "$" + x.toFixed(6);
$("turn").innerHTML = `<tr><td class="win">JEV 1.13</td><td>${T.jev.games}</td><td class="win">${T.jev.mean_score}</td><td class="win">${fmt(T.jev.latency_ms_p50)} ms</td><td>${fmt(T.jev.latency_ms_p95)} ms</td><td class="win">${usd(T.jev.cost_usd_per_decision)}</td><td>$${T.jev.cost_usd_total.toFixed(2)}</td><td>${T.jev.errors}</td></tr>
  <tr><td>Claude Haiku 4.5</td><td>${T.claude.games}</td><td>${T.claude.mean_score}</td><td>${fmt(T.claude.latency_ms_p50)} ms</td><td>${fmt(T.claude.latency_ms_p95)} ms</td><td>${usd(T.claude.cost_usd_per_decision)}</td><td>$${T.claude.cost_usd_total.toFixed(2)}</td><td>${T.claude.errors}</td></tr>`;

$("foot").innerHTML = `Built from results.json: ${D.games_logged} logged games · <a href="https://github.com/kkannan18/jev-invaders">github.com/kkannan18/jev-invaders</a> · jev-1.13.0 vs claude-haiku-4-5 · ALE/SpaceInvaders-v5`;
</script></html>"""


def render(out=ROOT / "scoreboard.html"):
    html = TEMPLATE.replace("__DATA__", json.dumps(payload()))
    out.write_text(html)
    return out


if __name__ == "__main__":
    p = render()
    print(f"wrote {p}")
    if "--open" in sys.argv:
        webbrowser.open(p.as_uri())
