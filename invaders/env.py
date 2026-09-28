"""Space Invaders environment + frame-to-JSON state encoder.

The ALE env runs at frameskip=1; the harness repeats each decision for 4 frames
(the standard v5 frameskip) and keeps all 4 screens. Atari Space Invaders
flickers (bombs and the ship are drawn on alternate frames), so reading a
single frame misses objects; the union of the 4 frames does not, and the first
vs last frame gives each bullet's direction within the step.
"""
from __future__ import annotations

import gymnasium as gym
import ale_py
import numpy as np
from scipy import ndimage

gym.register_envs(ale_py)

ENV_ID = "ALE/SpaceInvaders-v5"
ACTION_REPEAT = 4
REPEAT_ACTION_PROBABILITY = 0.25
MAX_FRAMES = 108000
ACTIONS = ["NOOP", "FIRE", "RIGHT", "LEFT", "RIGHTFIRE", "LEFTFIRE"]

ALIEN = (134, 134, 29)
BULLET = (142, 142, 142)
SHIP = (50, 132, 50)
SHIELD = (181, 83, 40)
PLAY_TOP, SHIP_TOP, GROUND = 25, 180, 195
SHIELD_X = [(42, 49), (74, 81), (106, 113)]  # three bunkers (screen px)


def make_env():
    return gym.make(
        ENV_ID,
        obs_type="ram",
        frameskip=1,
        repeat_action_probability=REPEAT_ACTION_PROBABILITY,
        full_action_space=False,
        max_num_frames_per_episode=MAX_FRAMES,
    )


def step_repeat(env, action: int):
    """Apply `action` for ACTION_REPEAT frames. Returns reward, done flags, info, screens."""
    ale = env.unwrapped.ale
    total, frames = 0.0, []
    terminated = truncated = False
    info = {}
    for _ in range(ACTION_REPEAT):
        _, r, terminated, truncated, info = env.step(action)
        total += r
        frames.append(ale.getScreenRGB().copy())
        if terminated or truncated:
            break
    return total, terminated, truncated, info, frames


def _mask(frames, color):
    return np.stack([(f == color).all(-1) for f in frames])


class Encoder:
    """Turns the last step's screens into compact JSON the deciders read."""

    def __init__(self):
        self.ship_x = 80
        self.shield_full = None

    def encode(self, frames, lives: int, score: float, step: int) -> dict:
        # --- ship: last frame where it is drawn
        ship = _mask(frames, SHIP)[:, SHIP_TOP:GROUND]
        for m in ship[::-1]:
            xs = np.where(m.any(0))[0]
            if len(xs):
                self.ship_x = int((xs[0] + xs[-1]) // 2)
                break
        sx = self.ship_x

        # --- aliens: cluster dilated pixels into one blob per invader
        a = _mask(frames[-1:], ALIEN)[0]
        a[:PLAY_TOP] = False
        lab, n = ndimage.label(ndimage.binary_dilation(a, iterations=2))
        aliens = []
        for sl in ndimage.find_objects(lab):
            y = (sl[0].start + sl[0].stop) // 2
            x = (sl[1].start + sl[1].stop) // 2
            aliens.append((int(x), int(y)))
        aliens.sort(key=lambda p: (-p[1], abs(p[0] - sx)))

        # --- bullets: direction from first vs last frame they appear in
        bm = _mask(frames, BULLET)
        bm[:, :PLAY_TOP] = False
        bm[:, GROUND:] = False
        lab, n = ndimage.label(bm.any(0))
        bombs, missile = [], None
        for i, sl in enumerate(ndimage.find_objects(lab), start=1):
            comp = lab == i
            ys = [np.where((m & comp).any(1))[0] for m in bm]
            ys = [y.mean() for y in ys if len(y)]
            x = int((sl[1].start + sl[1].stop) // 2)
            y = int(sl[0].stop)
            dy = (ys[-1] - ys[0]) / max(1, len(ys) - 1) if len(ys) > 1 else 0.0
            going_up = dy < 0 or (dy == 0 and y > 150 and abs(x - sx) <= 4)
            if going_up:
                missile = {"x": x, "y": int(sl[0].start)}
            else:
                v = max(dy * ACTION_REPEAT, 4.0)  # px per step, bombs fall ~4-8px/step
                steps_to_ship = max(0.0, (SHIP_TOP - y) / v)
                bombs.append({"x": x, "y": y, "dx_from_ship": x - sx,
                              "steps_to_ship_row": round(float(steps_to_ship), 1)})
        bombs.sort(key=lambda b: b["steps_to_ship_row"])

        # --- shields: fraction of bunker pixels remaining
        sh = _mask(frames[-1:], SHIELD)[0][150:180]
        counts = [int(sh[:, x0 - 4:x1 + 4].sum()) for x0, x1 in SHIELD_X]
        if self.shield_full is None:
            self.shield_full = [max(c, 1) for c in counts]
        shields = [{"x": (x0 + x1) // 2, "left": round(c / f, 2)}
                   for (x0, x1), c, f in zip(SHIELD_X, counts, self.shield_full)]

        # --- derived tactical features (same for every decider)
        threats = [b for b in bombs if abs(b["dx_from_ship"]) <= 8 and b["steps_to_ship_row"] <= 8]
        cover = next((s for s in shields if abs(s["x"] - sx) <= 7 and s["left"] > 0.3), None)
        target = None
        if aliens:
            # lowest alien in the column closest to the ship
            tx, ty = min(aliens, key=lambda p: (abs(p[0] - sx), -p[1]))
            target = {"x": tx, "y": ty, "dx_from_ship": tx - sx}
        safe_left = not any(-16 <= b["dx_from_ship"] <= 2 and b["steps_to_ship_row"] <= 8 for b in bombs)
        safe_right = not any(-2 <= b["dx_from_ship"] <= 16 and b["steps_to_ship_row"] <= 8 for b in bombs)

        return {
            "step": step,
            "lives": int(lives),
            "score": int(score),
            "ship_x": sx,
            "ship_limits": [37, 119],
            "under_shield": bool(cover),
            "aliens_left": len(aliens),
            "lowest_alien_y": max((y for _, y in aliens), default=None),
            "nearest_alien_column": target,
            "my_missile_in_flight": missile is not None,
            "bombs": bombs[:6],
            "danger_now": len(threats) > 0,
            "safe_to_move_left": safe_left,
            "safe_to_move_right": safe_right,
            "shields": shields,
            "aliens": [[x, y] for x, y in aliens[:24]],
        }
