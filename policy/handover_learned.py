"""Handover motion learned from the teleop demos (replaces the hand-set offsets in handover.py).

In the demos the operator grasps, brings the box up to their hand, holds still and presses Save; the
release itself isn't recorded. So each demo's grasp -> settle segment is the handover path and its
last frame is the handover pose. `fit` time-normalizes those segments and takes the per-joint median,
giving an absolute-joint template. At run time the template is shifted to start exactly at the
robot's grasp pose, with the shift fading out (smoothstep) so it ends on the demos' handover pose.

    python -m policy.handover_learned datasets/real_pick_ep0 handover_template.json
"""

import json
import sys
from pathlib import Path

import numpy as np

from .data import FPS
from .handover import GRIPPER, GraspDetector

N_POINTS = 60


def segments(episodes: list[dict], settle_deg=4.0, max_cmd_drift=5.0) -> list[np.ndarray]:
    """Grasp -> settled-at-end state segments of the demos with a clean grasp (the commanded gripper
    no longer closing at detection, so the grasp time is when the box was actually held)."""
    out = []
    for e in episodes:
        a, s = e["action"], e["state"]
        det, g = GraspDetector(), None
        for t in range(len(a)):
            if det.update(float(a[t, GRIPPER]), float(s[t, GRIPPER])):
                g = t
                break
        if g is None or abs(a[g, GRIPPER] - a[-1, GRIPPER]) > max_cmd_drift:
            continue
        near = np.abs(s[g:, :5] - s[-1, :5]).max(1) < settle_deg
        settle = g + int(np.argmax(near))
        if settle - g >= 10:
            out.append(s[g: settle + 1])
    return out


def fit(episodes: list[dict]) -> dict:
    segs = segments(episodes)
    if not segs:
        raise ValueError("no demo with a clean grasp followed by a handover")
    u = np.linspace(0, 1, N_POINTS)
    res = [np.stack([np.interp(u, np.linspace(0, 1, len(sg)), sg[:, j]) for j in range(sg.shape[1])], 1) for sg in segs]
    return {"template": np.median(np.stack(res), 0).round(3).tolist(),
            "duration_s": float(np.median([(len(sg) - 1) / FPS for sg in segs])),
            "n_demos": len(segs)}


def trajectory(grasp_q: np.ndarray, grip_cmd: float, fitted: dict, speed=1.0) -> np.ndarray:
    """(T, 6) joint targets from the robot's grasp pose to the demos' handover pose, gripper held at grip_cmd."""
    tpl = np.asarray(fitted["template"], np.float32)
    n = max(2, round(fitted["duration_s"] / speed * FPS))
    u = np.linspace(0, 1, n)
    path = np.stack([np.interp(u, np.linspace(0, 1, len(tpl)), tpl[:, j]) for j in range(tpl.shape[1])], 1)
    fade = 1 - u * u * (3 - 2 * u)
    path = path + (np.asarray(grasp_q, np.float32) - path[0])[None] * fade[:, None]
    path[:, GRIPPER] = grip_cmd
    return path.astype(np.float32)


if __name__ == "__main__":
    from .dataset_io import load_lerobot_episodes

    fitted = fit(load_lerobot_episodes(Path(sys.argv[1]), trim=False))
    Path(sys.argv[2]).write_text(json.dumps(fitted, indent=1))
    end = np.asarray(fitted["template"])[-1]
    print(f"fitted from {fitted['n_demos']} demos, {fitted['duration_s']:.1f}s, handover pose {np.round(end[:5], 1)}")
