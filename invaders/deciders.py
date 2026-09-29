"""Deciders: JEV (the pilot), an LLM baseline through the System One adapter,
and two keyless policies (random, scripted) for testing the harness.

Every decider answers the SAME typed question about the SAME state JSON, so the
JEV vs LLM comparison is apples to apples. SDK-level retries are disabled and
retried here instead, so every error and retry is counted, not hidden.
"""
from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass, field

from .env import ACTIONS

INSTRUCTIONS = (
    "You are piloting the laser cannon in Atari Space Invaders. The state is the current frame "
    "as JSON (x grows to the right; y grows downward; the ship sits at y=185). "
    "Pick the single best action for the next 4 frames. Priorities, in order: "
    "1) Survive: if danger_now is true or a bomb has small steps_to_ship_row and |dx_from_ship| <= 8, "
    "move away from it toward the side marked safe (safe_to_move_left / safe_to_move_right), or stay under a shield. "
    "2) Score: when my_missile_in_flight is false and nearest_alien_column is within 4 px, fire. "
    "3) Otherwise move toward nearest_alien_column (negative dx_from_ship means it is to the left), "
    "firing while moving when no missile is in flight. Only one missile can be in flight, so FIRE does "
    "nothing while my_missile_in_flight is true."
)

CRITERIA = {
    "NOOP": "Stay still and do not fire (hold position, e.g. sheltering under a shield while a missile is in flight).",
    "FIRE": "Stay still and fire straight up (an alien is directly above and no missile is in flight).",
    "RIGHT": "Move right without firing (dodge a bomb on the left, or line up with an alien to the right).",
    "LEFT": "Move left without firing (dodge a bomb on the right, or line up with an alien to the left).",
    "RIGHTFIRE": "Move right and fire (an alien is slightly right and no missile is in flight).",
    "LEFTFIRE": "Move left and fire (an alien is slightly left and no missile is in flight).",
}

RETRY_STATUSES = {408, 429, 500, 502, 503, 504, 529}
MAX_RETRIES = 2

# Published list prices, USD per million tokens. Only used where a provider
# reports token usage; unknown prices leave cost_usd out rather than guess.
PRICES = {
    "jev-latest": (0.042, 0.0),  # docs.typesafe.ai/models: $0.042 / Mtok input, output free
    "jev-1.13.0": (0.042, 0.0),
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-sonnet-4-5": (3.00, 15.00),
}


@dataclass
class Decision:
    action: int
    confidence: float | None
    latency_ms: float
    called_model: bool
    input_tokens: int = 0
    output_tokens: int = 0
    served_model: str | None = None
    errors: dict = field(default_factory=dict)
    retries: int = 0
    failed: bool = False  # no answer came back; the harness falls back


def axes_action(probs: dict) -> str:
    """Decompose the six-way distribution into two axes: where to move and whether to fire.

    JEV's probability mass is often split between twin actions (LEFT vs LEFTFIRE).
    Summing along each axis keeps both votes instead of throwing one away.
    """
    p = {a: float(probs.get(a, 0.0)) for a in ACTIONS}
    move = {"LEFT": p["LEFT"] + p["LEFTFIRE"], "RIGHT": p["RIGHT"] + p["RIGHTFIRE"], "STAY": p["NOOP"] + p["FIRE"]}
    fire = p["FIRE"] + p["LEFTFIRE"] + p["RIGHTFIRE"]
    m = max(move, key=move.get)
    if m == "STAY":
        return "FIRE" if fire > 0.5 else "NOOP"
    return m + "FIRE" if fire > 0.5 else m


def question():
    return {"move": {"type": "choice", "instructions": INSTRUCTIONS, "criteria": CRITERIA}}


def _status_of(exc) -> str:
    s = getattr(exc, "status", None) or getattr(exc, "status_code", None)
    return str(s) if s else type(exc).__name__


class _Remote:
    provider = "?"
    requested_model = "?"
    policy = "argmax"  # or "axes": see axes_action
    latency_tax_ms = 0  # control experiment: add a fixed delay to every answer

    def cost(self, itok, otok):
        p = PRICES.get(self.requested_model)
        return None if p is None else (itok * p[0] + otok * p[1]) / 1e6

    def _call(self, state):  # -> (choice, confidence, in_tok, out_tok, served_model)
        raise NotImplementedError

    def decide(self, state: dict) -> Decision:
        errors, retries = {}, 0
        t0 = time.perf_counter()
        for attempt in range(MAX_RETRIES + 1):
            try:
                choice, conf, itok, otok, served, probs = self._call(state)
                if self.latency_tax_ms:
                    time.sleep(self.latency_tax_ms / 1000)
                if self.policy == "axes" and probs:
                    choice = axes_action(probs)
                return Decision(ACTIONS.index(choice), conf, (time.perf_counter() - t0) * 1000, True,
                                itok or 0, otok or 0, served, errors, retries)
            except Exception as exc:  # noqa: BLE001 - counted, then retried or fallen back
                code = _status_of(exc)
                errors[code] = errors.get(code, 0) + 1
                retryable = code.isdigit() and int(code) in RETRY_STATUSES or "Timeout" in code or "Connection" in code
                if attempt < MAX_RETRIES and retryable:
                    retries += 1
                    time.sleep(0.25 * 2 ** attempt)
                    continue
                if code in {"401", "403"}:
                    raise
                break
        return Decision(0, None, (time.perf_counter() - t0) * 1000, True, errors=errors, retries=retries, failed=True)


class JEVDecider(_Remote):
    provider = "typesafe"
    name = "jev"

    def __init__(self, model: str = "jev-latest"):
        from typesafe_sdk import RetryPolicy, TypeSafeClient
        import typesafe_sdk

        self.requested_model = model
        self.sdk_package, self.sdk_version = "typesafe-sdk", typesafe_sdk.__version__
        self.client = TypeSafeClient(model=model, retry=RetryPolicy(max_retries=0), timeout=10.0)

    def _call(self, state):
        r = self.client.system_one(state, question())
        a = r.answers["move"]
        return (a.choice, a.confidence, r.usage.input_tokens, r.usage.output_tokens, r.model,
                getattr(a, "probabilities", None))


class LLMDecider(_Remote):
    name = "baseline"

    def __init__(self, provider: str = "anthropic", model: str = "claude-haiku-4-5"):
        from importlib.metadata import version
        from system_one_adapter import SystemOneAdapterClient

        self.provider, self.requested_model = provider, model
        self.sdk_package, self.sdk_version = "system-one-adapter", version("system-one-adapter")
        self.client = SystemOneAdapterClient(structured_outputs=True, llm_answer_mode="probabilities",
                                             normalize_probabilities=True, provider=provider, model=model)

    def _call(self, state):
        r = self.client.system_one(state, question())
        a = r.answers["move"]
        u = r.usage
        itok = getattr(u, "input_tokens_total", None) or u.input_tokens
        otok = getattr(u, "output_tokens_total", None) or u.output_tokens
        served = None
        for att in (r.debug or {}).get("llm_attempts", [])[::-1]:
            served = (att.get("debug_info") or {}).get("model") or served
            if served:
                break
        return (a.choice, a.confidence, itok, otok, served or self.requested_model,
                getattr(a, "probabilities", None))



class RandomDecider:
    provider, requested_model, name = "none", "random-policy", "random"
    sdk_package = sdk_version = None

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def decide(self, state):
        t0 = time.perf_counter()
        a = self.rng.randrange(len(ACTIONS))
        return Decision(a, None, (time.perf_counter() - t0) * 1000, False, served_model="random-policy")


class ScriptedDecider:
    """Hand-written rules over the same state JSON. No model, used to sanity check the encoder."""

    provider, requested_model, name = "none", "scripted-policy", "scripted"
    sdk_package = sdk_version = None

    def decide(self, s):
        t0 = time.perf_counter()
        fire = not s["my_missile_in_flight"]
        if s["danger_now"]:
            left_ok, right_ok = s["safe_to_move_left"], s["safe_to_move_right"]
            a = "LEFT" if left_ok and (not right_ok or s["ship_x"] > 78) else "RIGHT"
        else:
            t = s["nearest_alien_column"]
            dx = t["dx_from_ship"] if t else 0
            if abs(dx) <= 3:
                a = "FIRE" if fire else "NOOP"
            elif dx < 0:
                a = "LEFTFIRE" if fire else "LEFT"
            else:
                a = "RIGHTFIRE" if fire else "RIGHT"
        return Decision(ACTIONS.index(a), None, (time.perf_counter() - t0) * 1000, False, served_model="scripted-policy")


def make_decider(kind: str, **kw):
    if kind == "jev":
        d = JEVDecider(kw.get("model") or os.environ.get("JEV_MODEL", "jev-latest"))
        d.policy = kw.get("policy") or "argmax"
        d.latency_tax_ms = kw.get("latency_tax_ms") or 0
        return d
    if kind == "baseline":
        d = LLMDecider(kw.get("provider") or "anthropic", kw.get("model") or "claude-haiku-4-5")
        d.policy = kw.get("policy") or "argmax"
        return d
    if kind == "random":
        return RandomDecider(kw.get("seed", 0))
    if kind == "scripted":
        return ScriptedDecider()
    raise ValueError(kind)
