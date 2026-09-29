"""One game (episode) of Space Invaders with a given decider, fully measured."""
from __future__ import annotations

import threading
import time

import numpy as np

from .env import ACTIONS, ACTION_REPEAT, Encoder, make_env, step_repeat

STEP_HZ = 60 / ACTION_REPEAT  # real Atari speed: 15 decisions per second


def _pct(xs, q):
    return int(round(float(np.percentile(xs, q)))) if xs else None


def play(decider, seed: int, max_steps: int | None = None, conf_threshold: float = 0.0,
         realtime: bool = False, log_every: int = 0, on_step=None) -> dict:
    """Play one game. In turn-based mode the emulator waits for each decision
    (decision_interval_steps=1). In realtime mode it runs at 15 steps/s and the
    ship keeps doing its last action until the next answer arrives, so slow
    deciders act on stale frames, like a real player would."""
    env = make_env()
    env.reset(seed=seed)
    # first observation: step one NOOP so the encoder has screens
    reward, term, trunc, info, frames = step_repeat(env, 0)
    enc = Encoder()
    score, steps, lives0 = reward, 1, info.get("lives", 3)
    last_action = 0
    lat, confs, low_conf = [], [], 0
    calls = in_tok = out_tok = retries = fallbacks = stale = 0
    errors: dict[str, int] = {}
    served = None
    live_conf = {"conf": None, "probs": None}
    t_start = time.perf_counter()
    actions_hist = [0] * len(ACTIONS)

    def absorb(d):
        nonlocal calls, in_tok, out_tok, retries, fallbacks, low_conf, served, last_action
        lat.append(d.latency_ms)
        calls += int(d.called_model)
        in_tok += d.input_tokens
        out_tok += d.output_tokens
        retries += d.retries
        for k, v in d.errors.items():
            errors[k] = errors.get(k, 0) + v
        if d.served_model:
            served = d.served_model
        if d.failed:
            fallbacks += 1  # keep doing the last action
            if calls >= 10 and fallbacks == calls:
                raise RuntimeError(f"decider unreachable: first {calls} calls all failed {errors}")
            return
        if d.confidence is not None:
            confs.append(d.confidence)
            live_conf["conf"] = d.confidence
            if d.confidence < conf_threshold:
                low_conf += 1  # not sure: hold the last action instead
                return
        last_action = d.action

    if not realtime:
        while not (term or trunc) and (max_steps is None or steps < max_steps):
            state = enc.encode(frames, info["lives"], score, steps)
            absorb(decider.decide(state))
            actions_hist[last_action] += 1
            r, term, trunc, info, frames = step_repeat(env, last_action)
            score += r
            steps += 1
            if on_step:
                on_step(steps, score, info, last_action, frames[-1],
                        {"calls": calls, "stale": stale, "lat": lat[-1] if lat else None, "conf": live_conf["conf"]})
            if log_every and steps % log_every == 0:
                print(f"  step {steps} score {int(score)} lives {info['lives']} p50 {_pct(lat, 50)}ms", flush=True)
    else:
        box = {"state": enc.encode(frames, info["lives"], score, steps), "fresh": None, "stop": False}
        lock = threading.Lock()

        def worker():
            while not box["stop"]:
                with lock:
                    st = box["state"]
                d = decider.decide(st)
                with lock:
                    box["fresh"] = d

        th = threading.Thread(target=worker, daemon=True)
        th.start()
        next_t = time.perf_counter()
        while not (term or trunc) and (max_steps is None or steps < max_steps):
            with lock:
                d, box["fresh"] = box["fresh"], None
            if d is None:
                stale += 1
            else:
                absorb(d)
            actions_hist[last_action] += 1
            r, term, trunc, info, frames = step_repeat(env, last_action)
            score += r
            steps += 1
            with lock:
                box["state"] = enc.encode(frames, info["lives"], score, steps)
            if on_step:
                on_step(steps, score, info, last_action, frames[-1],
                        {"calls": calls, "stale": stale, "lat": lat[-1] if lat else None, "conf": live_conf["conf"]})
            next_t += 1 / STEP_HZ
            time.sleep(max(0.0, next_t - time.perf_counter()))
        box["stop"] = True
        th.join(timeout=15)

    env.close()
    run = {
        "seed": seed,
        "score": int(score),
        "steps": steps,
        "frames": int(info.get("episode_frame_number", steps * ACTION_REPEAT)),
        "lives_lost": int(lives0 - info.get("lives", 0)),
        "terminated": bool(term),
        "truncated": bool(trunc) or (max_steps is not None and steps >= max_steps and not term),
        "model_calls": calls,
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "latency_ms_p50": _pct(lat, 50),
        "latency_ms_p95": _pct(lat, 95),
        "latency_ms_total": int(round(sum(lat))),
        "errors_by_status": errors,
        "retries": retries,
        "fallback_actions": fallbacks,
        "wall_clock_s": round(time.perf_counter() - t_start, 1),
        "served_model": served or decider.requested_model,
        "action_counts": dict(zip(ACTIONS, actions_hist)),
        "mode": "realtime_15hz" if realtime else "turn_based",
        "conf_threshold": conf_threshold,
        "policy": getattr(decider, "policy", "argmax"),
        "latency_tax_ms": getattr(decider, "latency_tax_ms", 0),
    }
    if confs:
        run["mean_confidence"] = round(float(np.mean(confs)), 3)
        run["low_conf_rate"] = round(low_conf / max(1, len(confs)), 3)
    if realtime:
        run["stale_steps"] = stale
    cost = getattr(decider, "cost", None)
    if cost and calls:
        c = cost(in_tok, out_tok)
        if c is not None:
            run["cost_usd"] = round(c, 4)
    return run
