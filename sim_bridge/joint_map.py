"""Sim joint radians <-> real LeRobot units (degrees for joints 1-5, 0-100 gripper). See CONTRACT.md §3, §6.

Fit from poses the user holds the real follower in (torque off) and the URDF angles of the same poses:
    P1 stretched, P2 "L"  -> offsets (and signs for lift/elbow/wrist_flex)
    P3 = P2 with pan turned to the robot's left, wrist bent down, and wrist roll turned clockwise
         (seen from behind) -> signs for the joints that don't move between P1 and P2

    python -m sim_bridge.joint_map fit poses.json       writes sim_bridge/joint_map.json
poses.json: {"sim_rad": {"P1": [6], "P2": [6], "P3": [6]}, "real": {"P1": [6], "P2": [6], "P3": [6]},
             "gripper": {"sim_closed_rad": f, "sim_open_rad": f, "real_closed": f, "real_open": f}}
"""

import json
import math
import sys
from pathlib import Path

import numpy as np

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
MAP_FILE = Path(__file__).resolve().parent / "joint_map.json"


class JointMap:
    def __init__(self, cfg: dict):
        self.sign = np.float32([cfg["joints"][j]["sign"] for j in JOINTS[:5]])
        self.offset = np.float32([cfg["joints"][j]["offset_deg"] for j in JOINTS[:5]])
        g = cfg["gripper"]
        self.g = (g["sim_closed_rad"], g["sim_open_rad"], g["real_closed"], g["real_open"])

    @classmethod
    def load(cls, path: Path = MAP_FILE) -> "JointMap":
        return cls(json.loads(path.read_text()))

    def to_real(self, q_rad: np.ndarray) -> np.ndarray:
        q_rad = np.asarray(q_rad, np.float32)
        out = np.empty_like(q_rad)
        out[..., :5] = self.sign * np.degrees(q_rad[..., :5]) + self.offset
        s0, s1, r0, r1 = self.g
        out[..., 5] = r0 + (q_rad[..., 5] - s0) / (s1 - s0) * (r1 - r0)
        return out

    def to_sim(self, q_real: np.ndarray) -> np.ndarray:
        q_real = np.asarray(q_real, np.float32)
        out = np.empty_like(q_real)
        out[..., :5] = np.radians((q_real[..., :5] - self.offset) / self.sign)
        s0, s1, r0, r1 = self.g
        out[..., 5] = s0 + (q_real[..., 5] - r0) / (r1 - r0) * (s1 - s0)
        return out


def fit(poses: dict) -> dict:
    sim = {k: np.degrees(np.float32(v)) for k, v in poses["sim_rad"].items()}
    real = {k: np.float32(v) for k, v in poses["real"].items()}
    joints, report = {}, []
    for j, name in enumerate(JOINTS[:5]):
        if abs(sim["P2"][j] - sim["P1"][j]) > 20:  # this joint moves between P1 and P2: sign and scale from those
            ratio = (real["P2"][j] - real["P1"][j]) / (sim["P2"][j] - sim["P1"][j])
            sign = math.copysign(1, ratio)
            report.append(f"{name}: real/sim slope {ratio:+.2f} (expect ±1)")
        else:  # joints that P1/P2 don't move: direction from P3
            ds, dr = sim["P3"][j] - sim["P2"][j], real["P3"][j] - real["P2"][j]
            if abs(ds) < 15 or abs(dr) < 15:
                raise ValueError(f"{name}: P3 must move this joint >15 deg from P2 (sim {ds:+.0f}, real {dr:+.0f})")
            sign = math.copysign(1, ds * dr)
            report.append(f"{name}: sign from P3 (sim {ds:+.0f}, real {dr:+.0f})")
        offset = float(np.mean([real[p][j] - sign * sim[p][j] for p in ("P1", "P2")]))
        spread = abs((real["P1"][j] - sign * sim["P1"][j]) - (real["P2"][j] - sign * sim["P2"][j]))
        report[-1] += f", offset {offset:+.1f} deg (P1/P2 disagree by {spread:.1f})"
        joints[name] = {"sign": sign, "offset_deg": offset}
    print("\n".join(report))
    return {"joints": joints, "gripper": poses["gripper"]}


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "fit":
        sys.exit(__doc__)
    MAP_FILE.write_text(json.dumps(fit(json.loads(Path(sys.argv[2]).read_text())), indent=1))
    print(f"wrote {MAP_FILE}")
