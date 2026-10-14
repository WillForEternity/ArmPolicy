"""Scripted handover that runs after the learned pick.

    pick (policy) --grasp detected--> lift up-and-back (scripted) --person tugs / Enter--> open --> home

Grasp detection: the commanded gripper keeps closing but the follower's jaws stall on the box, so
the gripper tracking error stays large while the command is steady. That needs no extra sensor and
uses the same signal in recorded demos (leader vs. follower) and at run time (policy vs. follower).

The handover motion is relative to the grasp pose: shoulder_lift tilts the upper arm ~45 deg back
(up and toward the base), the elbow folds by the same amount so the box rises rather than swinging
out, and the wrist flexes to keep the box roughly level. Pan, roll and the gripper command stay as
they were at grasp. Signs and sizes live in HandoverConfig and MUST be checked on the real arm at low
speed first; they depend on the calibration's joint directions.

Pure functions only: nothing here talks to the servos.
"""

from dataclasses import dataclass, field

import numpy as np

from .data import FPS, JOINTS

GRIPPER = JOINTS.index("gripper")


@dataclass
class HandoverConfig:
    # Joint offsets from the grasp pose, LeRobot degrees (gripper excluded).
    offsets: dict[str, float] = field(
        default_factory=lambda: {"shoulder_lift": -45.0, "elbow_flex": -45.0, "wrist_flex": 30.0}
    )
    move_s: float = 2.5  # duration of the lift, min-jerk
    open_value: float = 60.0  # gripper command to release (RANGE_0_100)
    home: tuple[float, ...] = (0, -95, 95, 70, 0, 5)  # replace with the real rest pose from a teleop session
    home_s: float = 3.0
    tug_deg: float = 6.0  # arm joint pulled this far from its goal at the handover pose = person is taking it


class GraspDetector:
    """True once the gripper is commanded closed but held open by the box for `hold_s` seconds."""

    def __init__(self, min_error=8.0, max_cmd_change=1.5, hold_s=0.3):
        self.min_error, self.max_cmd_change = min_error, max_cmd_change
        self.need = round(hold_s * FPS)
        self.reset()

    def reset(self) -> None:
        self.count, self.prev_cmd = 0, None

    def update(self, cmd_gripper: float, measured_gripper: float) -> bool:
        steady = self.prev_cmd is not None and abs(cmd_gripper - self.prev_cmd) < self.max_cmd_change
        blocked = abs(measured_gripper - cmd_gripper) > self.min_error
        self.count = self.count + 1 if steady and blocked else 0
        self.prev_cmd = cmd_gripper
        return self.count >= self.need


def min_jerk(start: np.ndarray, end: np.ndarray, seconds: float) -> np.ndarray:
    """(N, 6) joint targets from start to end with zero velocity and acceleration at both ends."""
    s = np.linspace(0, 1, max(2, round(seconds * FPS)))[:, None]
    return start + (end - start) * (10 * s**3 - 15 * s**4 + 6 * s**5)


def handover_pose(grasp_q: np.ndarray, grasp_cmd_gripper: float, cfg: HandoverConfig) -> np.ndarray:
    q = np.array(grasp_q, np.float32)
    for name, delta in cfg.offsets.items():
        q[JOINTS.index(name)] += delta
    q[GRIPPER] = grasp_cmd_gripper  # keep squeezing with the command that holds the box
    return q


def lift_trajectory(grasp_q: np.ndarray, grasp_cmd_gripper: float, cfg: HandoverConfig) -> np.ndarray:
    start = np.array(grasp_q, np.float32)
    start[GRIPPER] = grasp_cmd_gripper
    return min_jerk(start, handover_pose(grasp_q, grasp_cmd_gripper, cfg), cfg.move_s)


def person_tugging(goal_q: np.ndarray, measured_q: np.ndarray, cfg: HandoverConfig) -> bool:
    arm = [i for i in range(len(JOINTS)) if i != GRIPPER]
    return float(np.max(np.abs(np.asarray(measured_q)[arm] - np.asarray(goal_q)[arm]))) > cfg.tug_deg


def release_and_home(hold_q: np.ndarray, cfg: HandoverConfig) -> np.ndarray:
    opened = np.array(hold_q, np.float32)
    opened[GRIPPER] = cfg.open_value
    pause = np.repeat(opened[None], round(0.7 * FPS), 0)  # let the person take the weight before moving away
    return np.vstack([min_jerk(hold_q, opened, 0.4), pause, min_jerk(opened, np.float32(cfg.home), cfg.home_s)])


def trim_at_grasp(episode: dict, after_s=0.5) -> dict:
    """Cut a recorded demo shortly after the grasp is detected, so the policy only learns the pick."""
    det = GraspDetector()
    for t, (a, s) in enumerate(zip(episode["action"][:, GRIPPER], episode["state"][:, GRIPPER])):
        if det.update(float(a), float(s)):
            end = min(len(episode["action"]), t + round(after_s * FPS))
            return {k: v[:end] for k, v in episode.items()}
    return episode
