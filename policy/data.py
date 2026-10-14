"""Episodes -> (state, box, action-chunk) training samples.

An episode is a dict of float32 arrays sampled at 30 fps:
    state  (T, 6)  follower joint positions (LeRobot units, joint order below)
    action (T, 6)  leader joint positions sent to the follower that step
    box    (T, 6)  box features from perception.TableFrame.features (held pose)
"""

import math

import numpy as np
import torch

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
FPS = 30


def fake_episodes(n: int, seed=0) -> list[dict]:
    """Synthetic pick demos (home -> grasp -> short lift) whose grasp depends nonlinearly on the box pose.

    The learned policy only picks; the handover afterwards is scripted (handover.py), so demos end
    at the lift. Only for testing the pipeline end to end before real demos exist; the kinematics
    are made up. The follower's gripper stalls on the box while the leader keeps closing, which is
    the signal handover.GraspDetector looks for.
    """
    rng = np.random.default_rng(seed)
    home = np.float32([0, -95, 95, 70, 0, 5])
    eps = []
    for _ in range(n):
        x, y = rng.uniform(-0.6, 0.6, 2)
        a = rng.uniform(-math.pi / 2, math.pi / 2)
        reach = math.hypot(x, y + 1.2)
        pan = math.degrees(math.atan2(x, y + 1.2))
        pre = np.float32([pan, -40 + 25 * reach, 40 - 20 * reach, 60, math.degrees(a) - pan, 60])
        grasp = pre + np.float32([0, 15, -10, 15, 0, 0])
        closed = grasp.copy()
        closed[5] = 2
        lift = closed + np.float32([0, -25, 10, -10, 0, 0])
        T = 150
        keys = [(0, home), (55, pre), (85, grasp), (105, closed), (T - 1, lift)]
        t = np.arange(T)
        action = np.stack([np.interp(t, [k for k, _ in keys], [p[j] for _, p in keys]) for j in range(6)], 1)
        action += rng.normal(0, 0.8, action.shape).cumsum(0) * 0.05  # teleop wobble
        state = np.vstack([home, action[:-1]]) + rng.normal(0, 0.3, action.shape)  # follower lags one tick
        state[:, 5] = np.maximum(state[:, 5], 20)  # box blocks the jaws at 20
        box = np.tile([x, y, math.cos(2 * a), math.sin(2 * a), 0.9, 0.57], (T, 1))
        eps.append({"state": state.astype(np.float32), "action": action.astype(np.float32), "box": box.astype(np.float32)})
    return eps


class Normalizer:
    """Per-dimension mean/std, saved with the model so the board applies the exact same scaling."""

    def __init__(self, stats: dict[str, dict[str, list[float]]]):
        self.stats = {k: {s: np.float32(v) for s, v in d.items()} for k, d in stats.items()}

    @classmethod
    def fit(cls, episodes: list[dict]) -> "Normalizer":
        stats = {}
        for key in ("state", "action", "box"):
            x = np.concatenate([e[key] for e in episodes])
            stats[key] = {"mean": x.mean(0).tolist(), "std": np.maximum(x.std(0), 1e-2).tolist()}
        return cls(stats)

    def norm(self, key: str, x):
        return (x - self.stats[key]["mean"]) / self.stats[key]["std"]

    def denorm(self, key: str, x):
        return x * self.stats[key]["std"] + self.stats[key]["mean"]

    def to_json(self) -> dict:
        return {k: {s: v.tolist() for s, v in d.items()} for k, d in self.stats.items()}


class ChunkDataset(torch.utils.data.Dataset):
    """Every frame of every episode -> (state_t, box_t, action[t : t+chunk], pad mask).

    Chunks running past the episode end repeat the last action and are masked out of the loss (ACT's is_pad).
    """

    def __init__(self, episodes: list[dict], normalizer: Normalizer, chunk=30, state_noise=0.0, box_noise=0.0):
        self.chunk, self.state_noise, self.box_noise = chunk, state_noise, box_noise
        self.eps = [{k: normalizer.norm(k, e[k]).astype(np.float32) for k in ("state", "action", "box")} for e in episodes]
        self.index = [(i, t) for i, e in enumerate(self.eps) for t in range(len(e["state"]))]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        i, t = self.index[idx]
        e = self.eps[i]
        T = len(e["action"])
        steps = np.minimum(np.arange(t, t + self.chunk), T - 1)
        state = e["state"][t] + np.random.normal(0, self.state_noise, 6).astype(np.float32)
        box = e["box"][t] + np.random.normal(0, self.box_noise, e["box"].shape[1]).astype(np.float32)
        pad = np.arange(t, t + self.chunk) >= T
        return state, box, e["action"][steps], pad


def split_episodes(episodes: list[dict], val_frac=0.15, seed=0) -> tuple[list[dict], list[dict]]:
    """Hold out whole episodes (not frames) so validation measures generalization to new box poses."""
    order = np.random.default_rng(seed).permutation(len(episodes))
    n_val = max(1, round(val_frac * len(episodes)))
    return [episodes[i] for i in order[n_val:]], [episodes[i] for i in order[:n_val]]
