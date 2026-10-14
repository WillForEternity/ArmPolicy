"""Sim-to-real calibration from one hand-guided recording (motors off), no tape measure:

    rest pose, an "L" pose, and the closed jaw tips touched to 4 crosses visible in one phone frame.

1. Joint map: signs +1 (LeIsaac's real-leader <-> SO-101 USD convention, confirmed by the L pose).
   Pan offset from the rest/L heading; lift/elbow/wrist_flex offsets fitted so all 4 touched tips lie
   on the table plane, starting from the L pose.
2. Cross positions in the contract frame (pan axis, z up, forward -y) = FK of the touches.
3. Homography image px -> contract meters (table_calib.json). Which touch is which image cross is
   found by trying all orderings; the check is that the box in the calibration frame comes out
   88.5 x 65.9 mm.

    python -m sim_bridge.calibrate
"""

import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
URDF = HERE / "so101_new_calib.urdf"
CHAIN = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
TIP_LOCAL = np.array([-0.0088, 0.0, -0.104])  # closed jaw tips in gripper_link (Isaac probe_gripper.py)
BOX_MM = (88.5, 65.9)
PLATE_MM = 3.0  # crosses are drawn on the plate floor, a few mm above the cloth


def _rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def _axis_angle(axis, a):
    return cv2.Rodrigues(np.asarray(axis, float) * a)[0]


class FK:
    def __init__(self, urdf=URDF):
        joints = {j.get("name"): j for j in ET.parse(urdf).getroot().findall("joint")}
        self.links = []
        for name in CHAIN:
            j = joints[name]
            o = j.find("origin")
            T = np.eye(4)
            T[:3, :3] = _rpy(*map(float, o.get("rpy").split()))
            T[:3, 3] = list(map(float, o.get("xyz").split()))
            self.links.append((T, np.array(list(map(float, j.find("axis").get("xyz").split())))))
        # Contract frame: origin on the pan axis at base_link z=0 (table), forward (reach at pan=0) = -y.
        pan_T = self.links[0][0]
        self.origin = np.array([pan_T[0, 3], pan_T[1, 3], 0.0])
        reach = self._tip_urdf(np.radians([0, 74.1, -75.5, 0, 0])) - self.origin  # stretched forward (sim P1)
        yaw = math.atan2(reach[1], reach[0])
        self.R = _axis_angle([0, 0, 1], -math.pi / 2 - yaw)  # rotate reach direction onto -y

    def _tip_urdf(self, q_rad):
        T = np.eye(4)
        for (Tj, axis), a in zip(self.links, q_rad):
            Tq = np.eye(4)
            Tq[:3, :3] = _axis_angle(axis, a)
            T = T @ Tj @ Tq
        return (T @ np.append(TIP_LOCAL, 1))[:3]

    def tip(self, q_rad) -> np.ndarray:
        """Closed-jaw tip in the contract frame (m)."""
        return self.R @ (self._tip_urdf(q_rad) - self.origin)


def least_squares(f, x0, iters=50):
    """Gauss-Newton with a numeric Jacobian (small problems only)."""
    x = np.asarray(x0, float).copy()
    for _ in range(iters):
        r = f(x)
        J = np.stack([(f(x + e) - r) / 1e-4 for e in np.eye(len(x)) * 1e-4], 1)
        step = np.linalg.lstsq(J, -r, rcond=None)[0]
        x += step
        if np.abs(step).max() < 1e-6:
            break
    return type("Fit", (), {"x": x})


def stills(csv: Path, window=30, tol=1.0):
    """Mean pose of every stretch where no arm joint moves more than tol deg over window samples."""
    d = np.loadtxt(csv, delimiter=",", skiprows=1)
    q = d[:, 1:]
    still = np.array([np.ptp(q[i:i + window, :5], 0).max() < tol for i in range(len(q) - window)] + [False] * window)
    out, i = [], 0
    while i < len(still):
        if still[i]:
            j = i
            while still[j]:
                j += 1
            out.append((d[i, 0], q[i:j + window].mean(0)))
            i = j
        else:
            i += 1
    return out


PERM = (0, 1, 3, 2)
# Touch order A,B,C,D -> image crosses 1,2,4,3. The 4 crosses are nearly point-symmetric, so two
# labelings fit equally; only this one puts the pan axis top-right of the frame, where the arm is.


def main():
    segs = stills(HERE / "touches.csv")
    for t, q in segs:
        print(f"  still @{t:6.1f}s  " + " ".join(f"{v:7.2f}" for v in q))
    # Recording order: rest, cross A, cross B (held twice), cross C, cross D, L, wrist-bent, rest.
    rest, A, B1, B2, C, D, L = (segs[i][1] for i in range(7))
    touches = np.stack([A, (B1 + B2) / 2, C, D])
    fk = FK()

    sim_L = np.degrees([0, -0.2775, 0.2531, 0, 0])  # Isaac FK: upper arm vertical, forearm horizontal
    off0 = L[:5] - sim_L
    off0[0] = (rest[0] + L[0]) / 2  # sim rest and L both point straight ahead (pan 0)
    off0[4] = 0.0  # roll offset is invisible to the tip position; LeRobot centers roll at mid-travel

    # Crosses as seen AFTER the touches (the jaws nudged the plates 7-11 mm).
    px = np.array(json.loads((HERE / "cross_px_after.json").read_text()))
    mu = px.mean(0)
    ip = (px[list(PERM)] - mu) / 1000.0

    def tips(o3, qs=touches):
        off = off0.copy()
        off[1:4] = o3
        return np.stack([fk.tip(np.radians(q[:5] - off)) for q in qs])

    def img_to_table(x, p_px):
        s, th = x[3], x[4]
        R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
        return ((np.asarray(p_px) - mu) / 1000.0) @ R.T * s + x[5:7]

    def resid(x):
        P = tips(x[:3])
        m = img_to_table(x, px[list(PERM)])
        # xy agreement with the camera (mm), tips on the plate floor (mm), weak pull toward the L-pose offsets
        return np.concatenate([((m - P[:, :2]) * 1000).ravel(), (P[:, 2] - PLATE_MM / 1000) * 1000, (x[:3] - off0[1:4]) * 0.3])

    best = None
    for th0 in np.linspace(-math.pi, math.pi, 12, endpoint=False):
        fit = least_squares(resid, np.concatenate([off0[1:4], [0.22, th0, 0.0, -0.22]]), iters=100)
        cost = float((resid(fit.x) ** 2).sum())
        if best is None or cost < best[0]:
            best = (cost, fit.x)
    _, x = best
    r = resid(x)
    off = off0.copy()
    off[1:4] = x[:3]
    print(f"offsets (deg) {off.round(2)}  (L pose said {off0.round(2)})")
    print(f"scale {abs(x[3]):.3f} mm/px; camera vs arm xy error per cross (mm) {np.abs(r[:8]).reshape(4, 2).max(1).round(1)};"
          f" tip height error (mm) {r[8:12].round(1)}")

    import sys

    sys.path.insert(0, str(HERE.parent))
    from policy.perception import detect

    corners = np.load(HERE / "box_corners_px.npy") if (HERE / "box_corners_px.npy").exists() else None
    if corners is not None:
        c = img_to_table(x, corners)
        sides = sorted([np.linalg.norm(c[1] - c[0]), np.linalg.norm(c[2] - c[1])], reverse=True)
        print(f"box outline through the map: {sides[0] * 1000:.1f} x {sides[1] * 1000:.1f} mm (lid is {BOX_MM[0]} x {BOX_MM[1]};"
              " the outline includes a visible side face and the lid sits 25 mm nearer the camera)")

    rest_sim_deg = rest[:5] - off
    joint_map = {
        "joints": {n: {"sign": 1.0, "offset_deg": float(o)} for n, o in zip(CHAIN, off)},
        "gripper": {"sim_closed_rad": -0.1745, "sim_open_rad": 1.745, "real_closed": 0.0, "real_open": 100.0},
        "source": "sim_bridge/calibrate.py from touches.csv",
        "real_rest_deg": [float(v) for v in rest],
        "sim_rest_deg_for_isaac": [float(v) for v in rest_sim_deg] + [-10.0],
    }
    (HERE / "joint_map.json").write_text(json.dumps(joint_map, indent=1))
    table = img_to_table(x, px)  # 4 image points and their table positions under the fitted similarity
    calib = {"image_px": px.tolist(), "table_cm": table.tolist(), "units": "meters, contract frame (key name kept)",
             "scale_mm_per_px": float(abs(x[3]))}
    (HERE.parent / "table_calib.json").write_text(json.dumps(calib, indent=1))
    print("wrote sim_bridge/joint_map.json and table_calib.json; sim REST_DEG for Isaac:", np.round(rest_sim_deg, 1).tolist())


if __name__ == "__main__":
    main()
