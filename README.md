# JEV Invaders — The Kan Man Can

**At real Atari speed, slow thinking costs lives.** JEV (TypeSafe AI's decision
model) pilots Space Invaders one typed, confidence-scored choice per frame.
Claude Haiku 4.5 gets the identical state and the identical question through the
[System One adapter](https://github.com/typesafe-ai/system-one-adapter-python).
Every game runs in its own [Tenki](https://tenki.cloud) sandbox, and the live
race streams both players side by side from their sandboxes.

UFA JEV Bake-Off entry, **Pilot track**: JEV flies the ship and makes every move.

**The loop:** Tenki tunes the pilot, Cortex remembers what worked, and JEV decides.
Question designs race each other in parallel Tenki sandboxes on held-out seeds
(`tune.py`), every result is remembered in Mitosis Cortex, JEV picks each next
game's design from those memories (`mitosis_coach.py`), and the champion is
judged against Claude on the seeds Claude played. CI replays every commit in
Tenki and posts the score.

## Results

All numbers below are recomputed from `results.json` by `python summarize.py`.
Comparisons use only the seeds both deciders played.
Seeds 1–5, `ALE/SpaceInvaders-v5`, sticky actions 0.25, full games (no step cap).

### Turn-based: the emulator waits for every decision

| | **JEV 1.13** | Claude Haiku 4.5 | |
|---|---|---|---|
| Mean score, 5 games | **321** | 267 | 1.2× |
| Median latency per decision | **74 ms** | 1,055 ms | **14.3× faster** |
| p95 latency | **118 ms** | 1,673 ms | |
| Cost per decision | **$0.000048** | $0.001587 | **33× cheaper** |
| Cost, all 5 games | **$0.15** | $6.04 | |
| Errors / fallbacks | 0 / 0 | 0 / 0 | |

Per seed, JEV won 2, tied 1 and lost 2. Given unlimited thinking time, the LLM is
competitive. The difference shows up once time is real.

### Real-time: 15 moves a second, like a human player

In `--realtime` mode the game does not wait. The ship repeats its last move until
the decider's next answer lands, so a slow decider plays on stale information.

| | **JEV 1.13** | Claude Haiku 4.5 |
|---|---|---|
| Mean score, 5 games | **208** | 106 |
| Head to head by seed | **4 wins**, 1 loss | |
| Decisions per game | **232–713** | 20–48 |
| Moves based on a fresh decision | **70%** | 5% |

**JEV scores about 2× Claude in real time.** Claude gets to decide 20–48 times in
a whole game; the other 95% of its moves are repeats.

### Confidence gate: a finding that went the other way

JEV returns a calibrated `confidence` with every choice. We tried holding the
last move whenever confidence fell below a threshold. It hurt: at 0.5 the mean
fell from 321 to 169 (5 games), and at 0.65 to 238 (2 games). JEV's mean
confidence is about 0.62. Our reading: several moves are often about equally
good (firing while drifting left vs. just drifting left), so a low-confidence
answer is usually still a reasonable one, and freezing on it costs more. The shipped pilot trusts every answer. Those runs are in
`results.json` with `conf_threshold` and a note on each.

## Tuning the pilot in Tenki (`tune.py`)

JEV answers typed questions, so how you ask is a design choice. Four designs,
each one request per step (same latency):

| Design | What JEV is asked |
|---|---|
| `six` | one `choice` over the six ALE actions (the original) |
| `six_lean` | the same, on a smaller state (derived features only) |
| `split` | two typed answers in one call: a `choice` of where to steer (LEFT / RIGHT / STAY) and a `noul` (0 to 1) for "fire now" |
| `split_lean` | `split`, on the smaller state |

plus `six:axes` (sum the six-way probabilities into move and fire axes).

```bash
python tune.py --test-seeds 1 2 3 4 5
```

1. **Tune** on held-out seeds 101–105: 5 designs × 5 seeds = 25 games, each in
   its own Tenki sandbox, 5 at a time. Leaderboard → `champion.json`, pushed.
2. **Test** the champion on seeds 1–5, the seeds the LLM baseline played, turn-based
   and real-time. Tuning and test seeds never overlap, so the champion is not
   picked on the games it is judged on.
3. Every result is also remembered in Mitosis Cortex (when `MI_API_KEY` and
   `MI_OFFICE` are set).

`python summarize.py` prints the leaderboard and the champion vs LLM tables.
`--variant champion` plays the champion anywhere (`play.py`, `tenki_arena.py`,
`live_race.py`).

## Using the typed answer, not just the top pick

- **Live confidence.** The race page shows JEV's chosen move and its confidence
  on every frame, next to the game.
- **Axes steering** (`--policy axes`). JEV returns a probability for each of the
  six moves. Axes steering sums them along two axes, where to move and whether
  to fire, so a split vote between LEFT and LEFTFIRE still moves left.
- **Latency-tax control** (`--latency-tax-ms 1050`). The same JEV, with a delay
  added to every answer so it is as slow as Claude's median. If its real-time
  score falls toward Claude's, speed is what wins the game.

```bash
python tenki_arena.py --deciders jev --seeds 1 2 3 4 5 --realtime --latency-tax-ms 1050
python tenki_arena.py --deciders jev --seeds 1 2 3 4 5 --realtime --policy axes
python summarize.py   # adds LATENCY-TAX CONTROL and AXES POLICY tables once those runs exist
```

## Watch it: the live race

```bash
python live_race.py --seed 3
```

This starts two Tenki sandboxes, one for JEV and one for Claude. Each plays the
same seed at real speed, starting at the same second, and serves its own live
view (frame, score, decisions, fresh-move rate) through a Tenki preview URL.
`race.html` opens a split screen that polls both sandboxes. When the race ends,
both games are appended to `results.json` and pushed. `--local` runs the same
race on your own machine.

## How Tenki is used

- **Isolation and fairness.** Each game gets its own disposable microVM, so JEV
  and the LLM never share CPU or network with each other or with the laptop
  running the show.
- **The live race.** The spectator views are served from inside the sandboxes
  (`expose_port`). Without Tenki there is nothing to watch.
- **Parallel benchmark.** `tenki_arena.py` fans out decider × seed ×
  confidence-threshold jobs (5 at a time on the free workspace), retries when
  Tenki is out of capacity, and pushes each game the moment it finishes. The
  sandbox id is stored on each run as `tenki_sandbox_id`.
- **Tuning.** `tune.py` races question designs in parallel sandboxes and crowns
  the champion the pilot flies with.
- **CI replay on every push.** `.github/workflows/tenki-replay.yml` runs
  `ci_replay.py`: reference games replayed in fresh Tenki sandboxes must match
  exactly, and the current champion plays one JEV game whose score is posted on
  the commit. Secrets: `TENKI_API_KEY`, and `TYPESAFE_API_KEY` for the JEV score.
- The turn-based results, and the first real-time seeds, ran in Tenki. When
  Tenki had no capacity, later games ran locally with the same code; each run
  records where it ran.

## How Mitosis Labs is used: a coach with memory

`mitosis_coach.py` puts JEV in charge of its own training plan, with Cortex as
its memory. Each round:

1. **Recall.** Cortex returns the remembered games and Tenki tuning leaderboards
   most relevant to "which JEV design scores the most?", each with its Cortex id.
2. **Decide.** JEV answers a typed `choice` question, which of the five designs
   should play the next game, with those memories as its state. It returns a pick, a
   probability for every strategy, and a confidence.
3. **Play.** The chosen strategy plays the next seed in real time inside a
   Tenki sandbox.
4. **Remember.** The result, and the decision behind it, are written back to
   Cortex. The run in `results.json` carries a `coach` block with the pick, the
   probabilities and the Cortex ids it relied on.

```bash
export MI_API_KEY=mi_... MI_OFFICE=<your office id>
python mitosis_coach.py --seed-memory                 # load every past JEV game into Cortex
python mitosis_coach.py --rounds 5 --first-seed 11    # five coached games
```

Without Cortex the coach forgets everything between runs; without Tenki it has
no games to play.

## Reproduce

Python 3.10+.

```bash
pip install -r requirements.txt
export TYPESAFE_API_KEY=...     # JEV
export ANTHROPIC_API_KEY=...    # baseline (your own key)
export TENKI_API_KEY=tk_...     # sandboxes

# Turn-based benchmark: 5 JEV + 5 baseline games, fixed seeds, in Tenki sandboxes
python tenki_arena.py --deciders jev baseline --seeds 1 2 3 4 5

# Real-time showdown
python tenki_arena.py --deciders jev baseline --seeds 1 2 3 4 5 --realtime

# Confidence-gate sweep
python tenki_arena.py --deciders jev --seeds 1 2 3 4 5 --conf-thresholds 0.5 0.65

# Without Tenki, same games on this machine
python play.py --decider jev --seeds 1 2 3 4 5
python play.py --decider baseline --seeds 1 2 3 4 5

# The scoreboard
python summarize.py
```

No keys? `python play.py --decider scripted --seeds 1 --no-push` runs the whole
pipeline with a rule-based bot (logged as `provider: "none"`, never counted as
JEV). It is deterministic: seed 1 scores 710 and seed 2 scores 330 on a Mac, in
a Tenki sandbox, and in a Linux container.

To regenerate `results.json` from scratch, delete it and run the commands above.

## What gets measured

After every game the harness appends the run to `results.json`, checks it
(p95 ≥ p50, model calls ≤ steps, frames ≤ steps × 4), then commits and pushes
before the next game is recorded.

- **Score**: sum of env rewards. **Frames**: `info["episode_frame_number"]`.
- **Latency**: `time.perf_counter()` around each decision, retries included.
- **Tokens**: each response's `usage`. **Model id**: what the API returns
  (`jev-1.13.0`), not the alias requested.
- **Cost**: tokens × list price (JEV 1.13: $0.042 per million input tokens,
  output free; Claude Haiku 4.5: $1 / $5 per million input / output tokens).
- SDK auto-retries are off and retried by the harness, so every error (by HTTP
  status), retry and fallback is counted.
- Extra fields: `action_counts`, `mode`, `conf_threshold`, `stale_steps`
  (real-time moves with no fresh decision), `tenki_sandbox_id`, and the
  `realtime_showdown` block.

## How it works

1. **State.** The harness steps the emulator 4 frames per decision (the v5
   frameskip) and keeps all 4 screens. Atari Space Invaders draws bombs and the
   ship on alternating frames, so it unions the 4 screens and uses the first vs
   last frame to get each bomb's direction and steps to impact. The result is
   compact JSON: ship x, aliens, bombs with `dx_from_ship` and
   `steps_to_ship_row`, own missile, shields, lives, and derived
   `danger_now` / `safe_to_move_left` / `safe_to_move_right` / nearest column.
2. **Decision.** One `choice` question per step over the six ALE actions, with a
   short description of when each applies. JEV and the LLM get exactly the same
   JSON and question.
3. **Measurement and logging** as above.

| File | What it does |
|---|---|
| `invaders/env.py` | ALE env, 4-frame action repeat, screen → JSON encoder |
| `invaders/deciders.py` | JEV, LLM baseline, random and scripted policies; the shared question |
| `invaders/game.py` | One measured game, turn-based or real-time |
| `invaders/results.py` | Append to `results.json`, consistency checks, commit and push |
| `play.py` | Run games on this machine |
| `tenki_arena.py` | Parallel games in Tenki sandboxes, with capacity retries |
| `live_play.py` | One live player: real-time game plus spectator view over HTTP |
| `live_race.py` | Two live players in Tenki sandboxes, split-screen page, results pushed |
| `invaders/variants.py` | The question designs: state shaping, typed questions, decoding answers |
| `tune.py` | Tenki tuning sweep on held-out seeds, champion.json, champion test |
| `summarize.py` | The scoreboard, recomputed from `results.json` |
| `mitosis_coach.py` | JEV chooses the next strategy from Cortex memory; games run in Tenki |
| `ci_replay.py` | CI check: replay reference games in Tenki and compare scores |

## Rules compliance

JEV chooses every move in JEV runs; no other model is in that loop. The LLM is
only the comparison baseline. Keyless runs are labelled `provider: "none"`.
