"""Isaac Sim demo chunks -> contract episodes (CONTRACT.md §3-5).

    python -m sim_bridge.convert_sim sim_data/raw sim_data/sim_episodes.npz

Per successful episode: joints rad -> LeRobot units via joint_map.json; box held at its first
settled pose with per-episode perception noise (the real BoxTracker latches the arm-clear pose);
cut 0.5 s after the grasp with the same detector the real demos use.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from policy.handover import trim_at_grasp  # noqa: E402
from sim_bridge.joint_map import JointMap  # noqa: E402

SETTLE_STEP = 3  # box spawns 1 mm above the cloth; pose is stable after a few ticks
XY_NOISE_M, YAW_NOISE_DEG = 0.005, 2.0


def box_features(ob: np.ndarray) -> np.ndarray:
    """One frame of obs_box -> [x, y, cos 2θ, sin 2θ]. Accepts (x,y,z,yaw) or (x,y,z,cos2θ,sin2θ)."""
    if ob.shape[-1] == 4:
        return np.float32([ob[0], ob[1], np.cos(2 * ob[3]), np.sin(2 * ob[3])])
    return np.float32([ob[0], ob[1], ob[3], ob[4]])


def convert(raw_dir: Path, jmap: JointMap, seed=0) -> list[dict]:
    rng = np.random.default_rng(seed)
    episodes, n_total = [], 0
    for f in sorted(raw_dir.glob("*.npz")):
        d = np.load(f)
        keep = d["success"].astype(bool)
        if "gp_flip" in d.files:
            # Flipped-wrist grasps put the roll 180 deg from the unflipped ones for the same box pose;
            # an L1 policy would average the two modes into a roll that fits neither.
            keep &= ~d["gp_flip"].astype(bool)
        for i in np.flatnonzero(keep):
            n_total += 1
            T = int(d["length"][i])
            box = box_features(d["obs_box"][i, min(SETTLE_STEP, T - 1)])
            x, y, c, s = box
            yaw2 = np.arctan2(s, c) + 2 * np.radians(rng.normal(0, YAW_NOISE_DEG))
            box = np.float32([x + rng.normal(0, XY_NOISE_M), y + rng.normal(0, XY_NOISE_M), np.cos(yaw2), np.sin(yaw2)])
            ep = {
                "state": jmap.to_real(d["obs_joint_pos"][i, :T]),
                "action": jmap.to_real(d["action"][i, :T]),
                "box": np.repeat(box[None], T, 0),
            }
            trimmed = trim_at_grasp(ep)
            if len(trimmed["action"]) < T:  # grasp detected: keep; otherwise the stall signal never showed
                episodes.append(trimmed)
    print(f"{len(episodes)} of {n_total} successful sim episodes had a detectable grasp")
    return episodes


def save(episodes: list[dict], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **{f"ep{i}_{k}": v for i, e in enumerate(episodes) for k, v in e.items()})


def load(path: Path) -> list[dict]:
    d = np.load(path)
    n = len({k.split("_")[0] for k in d.files})
    return [{k: d[f"ep{i}_{k}"] for k in ("state", "action", "box")} for i in range(n)]


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    eps = convert(Path(sys.argv[1]), JointMap.load())
    save(eps, Path(sys.argv[2]))
    lens = [len(e["action"]) for e in eps]
    if lens:
        print(f"saved {len(eps)} episodes, {sum(lens)} frames, length {min(lens)}-{max(lens)} -> {sys.argv[2]}")
