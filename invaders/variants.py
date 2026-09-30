"""Question designs: how the game state and the typed question are put to the decider.

Every variant is one request per step, so latency stays the same. What changes is
how much of JEV's typed output we use.

  six         one `choice` over the six ALE actions (the original design)
  six_lean    the same question on a smaller state (derived features only)
  split       two typed answers in one call: a `choice` of where to steer
              (LEFT / RIGHT / STAY) and a `noul` (0-1 truth value) for "fire now"
  split_lean  split, on the smaller state

`tune.py` races these against each other in Tenki sandboxes on held-out seeds and
writes the winner to champion.json; `--variant champion` plays with it.
"""
from __future__ import annotations

import json
from pathlib import Path

VARIANTS = ["six", "six_lean", "split", "split_lean"]
CHAMPION_FILE = Path(__file__).resolve().parent.parent / "champion.json"

SIX_INSTRUCTIONS = (
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
SIX_CRITERIA = {
    "NOOP": "Stay still and do not fire (hold position, e.g. sheltering under a shield while a missile is in flight).",
    "FIRE": "Stay still and fire straight up (an alien is directly above and no missile is in flight).",
    "RIGHT": "Move right without firing (dodge a bomb on the left, or line up with an alien to the right).",
    "LEFT": "Move left without firing (dodge a bomb on the right, or line up with an alien to the left).",
    "RIGHTFIRE": "Move right and fire (an alien is slightly right and no missile is in flight).",
    "LEFTFIRE": "Move left and fire (an alien is slightly left and no missile is in flight).",
}

STEER_INSTRUCTIONS = (
    "You steer the laser cannon in Atari Space Invaders for the next 4 frames. The state is the current "
    "frame as JSON (x grows to the right). First priority is survival: if danger_now is true, or a bomb "
    "has |dx_from_ship| <= 8 and few steps_to_ship_row, move toward a side marked safe "
    "(safe_to_move_left / safe_to_move_right). Otherwise move toward nearest_alien_column "
    "(negative dx_from_ship means it is to the left), and stay put when it is within 4 px."
)
STEER_CRITERIA = {
    "LEFT": "Move left: dodge a bomb coming down on the right, or an alien column is to the left.",
    "RIGHT": "Move right: dodge a bomb coming down on the left, or an alien column is to the right.",
    "STAY": "Hold position: an alien is directly above, or staying under a shield is safest.",
}
FIRE_INSTRUCTIONS = (
    "Firing a missile this step is worthwhile: my_missile_in_flight is false and an alien is roughly "
    "above the ship (nearest_alien_column.dx_from_ship between -6 and 6)."
)

LEAN_KEYS = ["lives", "ship_x", "ship_limits", "under_shield", "aliens_left", "lowest_alien_y",
             "nearest_alien_column", "my_missile_in_flight", "danger_now", "safe_to_move_left",
             "safe_to_move_right"]


def resolve(variant: str | None) -> str:
    if not variant:
        return "six"
    if variant == "champion":
        try:
            return json.loads(CHAMPION_FILE.read_text())["variant"]
        except (OSError, KeyError, ValueError):
            raise SystemExit("No champion.json yet. Run `python tune.py` first, or pass --variant six.")
    if variant not in VARIANTS:
        raise SystemExit(f"unknown variant {variant}; choose from {VARIANTS} or champion")
    return variant


def champion_policy() -> str | None:
    try:
        return json.loads(CHAMPION_FILE.read_text()).get("policy")
    except (OSError, ValueError):
        return None


def prepare(state: dict, variant: str) -> dict:
    if variant.endswith("_lean"):
        s = {k: state[k] for k in LEAN_KEYS if k in state}
        s["bombs"] = state.get("bombs", [])[:3]
        return s
    return state


def questions(variant: str) -> dict:
    if variant.startswith("split"):
        return {"steer": {"type": "choice", "instructions": STEER_INSTRUCTIONS, "criteria": STEER_CRITERIA},
                "fire": {"type": "noul", "instructions": FIRE_INSTRUCTIONS}}
    return {"move": {"type": "choice", "instructions": SIX_INSTRUCTIONS, "criteria": SIX_CRITERIA}}


def decode(answers: dict, variant: str):
    """-> (action name, confidence, probabilities over the six actions or None)."""
    if variant.startswith("split"):
        s, f = answers["steer"], answers["fire"]
        fire = float(f.noul) > 0.5
        steer = s.choice
        if steer == "STAY":
            action = "FIRE" if fire else "NOOP"
        else:
            action = steer + ("FIRE" if fire else "")
        return action, s.confidence, None
    a = answers["move"]
    return a.choice, a.confidence, getattr(a, "probabilities", None)
