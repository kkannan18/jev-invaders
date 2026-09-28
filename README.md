# JEV Invaders — The Kan Man Can

JEV (TypeSafe AI's decision model) pilots Atari Space Invaders: every step, the
game frame becomes compact JSON and JEV answers one typed `choice` question over
the six ALE actions, with a calibrated `confidence`. The identical loop, the
identical state JSON and the identical question run against Claude Haiku 4.5
through the [System One adapter](https://github.com/typesafe-ai/system-one-adapter-python),
so the comparison is apples to apples. Every game runs in its own disposable
[Tenki](https://tenki.cloud) sandbox, in parallel, and is pushed to
`results.json` the moment it ends.

Entry for the UFA JEV Bake-Off, **pilot** track.

## What makes it different

- **Confidence as a control signal.** When JEV's confidence is below a threshold
  the ship holds its last action instead of twitching on a coin flip
  (`--conf-threshold`). The threshold is tuned by a sweep that runs one sandbox
  per (seed, threshold) pair on Tenki.
- **Real-time mode.** Turn-based benchmarks hide latency: the emulator politely
  waits for the model. `--realtime` runs the game at real Atari speed (15
  decisions/s); the ship keeps doing its last action until the next answer
  lands. A 150 ms decider acts on fresh frames; a 1.5 s decider flies blind for
  ~20 steps at a time. `stale_steps` in each run counts exactly that.
- **Flicker-proof state.** Atari Space Invaders draws bombs and the ship on
  alternate frames. The harness steps the emulator 4 frames per decision
  (standard v5 frameskip), unions the 4 screens, and uses the first vs last
  frame to get each bomb's direction and steps-to-impact.

## Run it

Python 3.10+.

```bash
pip install -r requirements.txt
export TYPESAFE_API_KEY=...        # JEV
export ANTHROPIC_API_KEY=...       # baseline (your own key)
export TENKI_API_KEY=tk_...        # sandboxes

# The whole benchmark: 5 JEV games + 5 baseline games, fixed seeds, in parallel Tenki sandboxes.
python tenki_arena.py --deciders jev baseline --seeds 1 2 3 4 5

# Tune the confidence gate (one sandbox per seed x threshold).
python tenki_arena.py --deciders jev --seeds 1 2 3 --conf-thresholds 0 0.4 0.6 0.8

# Real-time latency showdown.
python tenki_arena.py --deciders jev baseline --seeds 1 2 3 --realtime

# Or run locally without Tenki:
python play.py --decider jev --seeds 1 2 3 4 5
python play.py --decider baseline --seeds 1 2 3 4 5

# No keys? Check the harness with a keyless policy (recorded honestly as non-JEV).
python play.py --decider scripted --seeds 1 --no-push
```

## results.json

Regenerate it by deleting `results.json` and rerunning the commands above.
After every game the harness appends the run, checks it for consistency
(p95 ≥ p50, model calls ≤ steps, frames ≤ steps × 4), then runs
`git add results.json && git commit && git push` before the next game is
recorded. Pass `--no-push` (or `NO_PUSH=1`) to write without pushing.

Everything is measured in code: score is the sum of env rewards; frames come
from `info["episode_frame_number"]`; latency is `time.perf_counter()` around each
decision (including retries); tokens come from each response's `usage`; the
model id is what the API returns in `model`. SDK auto-retries are turned off and
retried by the harness so that every error (by HTTP status), retry and fallback
action is counted. Cost is tokens × published list price (JEV 1.13: $0.042/Mtok
input, output free; Claude Haiku 4.5: $1/$5 per Mtok).

Extra fields: `action_counts`, `mode`, `conf_threshold`, `tenki_sandbox_id`,
and a `realtime_showdown` block for `--realtime` games.

## Layout

| File | What it does |
|---|---|
| `invaders/env.py` | ALE env, 4-frame action repeat, screen → JSON encoder |
| `invaders/deciders.py` | JEV, LLM baseline (System One adapter), random and scripted policies; the shared question |
| `invaders/game.py` | One measured game, turn-based or real-time |
| `invaders/results.py` | Append to `results.json`, sanity checks, commit + push |
| `play.py` | Run games locally |
| `tenki_arena.py` | Fan games out to parallel Tenki sandboxes with locked-down egress, collect and push |

## Rules compliance

JEV chooses every move in JEV runs; no other model is in that loop. The
baseline is only used for comparison. Keyless runs are labelled
`provider: "none"` and never count as JEV.
