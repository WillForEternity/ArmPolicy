# Data and runtime contract: real SO-101 + Isaac Sim + UNO Q policy

Three sessions build against this file: arm/teleop, policy (`policy/`), and Isaac Sim (`sim_bridge/` on the Mac, generator on the pod).
Change it only by agreement; bump the version line.

**Version 2 (2026-10-13)**: frame origin, reach, box size, gripper confirmed by the Isaac session (FK-checked)

## 1. Task
Pick up the UNO Q box from the cloth with the learned policy, then hand it over with the scripted motion in `policy/handover.py`.
The policy learns only **rest → grasp → short lift**. Every episode, real or sim, is cut 0.5 s after the grasp is detected, using `handover.trim_at_grasp`.

## 2. Timing
30 Hz control. One row per control tick.

## 3. Joints (`state` = measured follower, `action` = commanded target)
- **Order:** `shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper`.
- **Units are LeRobot 0.6.1 `so101_follower` with `use_degrees=True`:**
  - Joints 1–5 are degrees: `(raw − (range_min+range_max)/2) · 360/4095`. Calibration is `~/.cache/huggingface/lerobot/calibration/robots/so_follower/follower.json`.
  - The gripper is 0–100 across its calibrated range.
- **Sim exports in these units, not radians.** It converts through `sim_bridge/joint_map.json`:
  - Joints 1–5: `deg_real = sign_j · degrees(q_sim_j) + offset_j`.
  - Gripper: `real = lerp(sim_closed_rad → real_closed, sim_open_rad → real_open)`.
  - How to fill the map is in §6.

## 4. Box observation (`box`, 4 floats)
`[x, y, cos 2θ, sin 2θ]`
- **Frame:** the robot base frame, in meters. Origin is on the shoulder_pan axis at table height, z is up, and forward (the arm's reach at pan = 0) is **−y**. Looking along forward, **+x is the robot's left**. These axes are the sim's USD frame.
  - The sim's pan axis sits at USD (+0.0208, −0.0231). The sim re-centers on it (`frame="pan_axis"` in its npz).
- **Workspace (where the box goes):** r = 0.12–0.28 m from the pan axis, bearing within ±34° of forward, any yaw.
  - Sim top-down grasp success is 94% at r 0.12–0.16 m, 80–85% at 0.16–0.28 m, and under 50% beyond 0.28 m.
  - Real demos must keep the box inside this region.
- **Box:** 89 × 66 × 25 mm, measured from the rectified iPhone frame.
- **θ:** the direction of the box's long side, measured counterclockwise about +z from +x. Doubled, because the box is 180° symmetric.
- **No size features.** There is one known box, and measured sizes would only add perception noise.
- **The value is held from the arm-clear frames,** exactly as `perception.BoxTracker` does: the pose seen before the gripper occludes the box. Sim must export the box pose from the first settled frame, repeated for every row, **not** the live pose. Add perception-like noise per episode (σ ≈ 5 mm in x/y, ≈ 2° in θ).
- **Real pipeline:** the iPhone overhead frame goes through `perception.detect` to get box corners in pixels. A homography from `table_calib.json` maps those to base-frame meters. The features are computed from the mapped corners.

## 5. Episode files
- **Real:** a LeRobot v3 dataset recorded with `lerobot-record`, overhead camera = iPhone. `policy/dataset_io.py` derives the box features.
- **Sim:** `sim_bridge/convert_sim.py` turns pod chunks into `sim_episodes.npz`.
  - Keys are `ep{i}_state`, `ep{i}_action`, `ep{i}_box`, already in contract units and already trimmed.
  - Only successful episodes are kept.
- **Start pose:** sim starts from the real rest pose (± noise), converted through the joint map. It must match the real rest pose read from the follower.

## 6. Calibration steps (one time; torque off, nothing moves by itself)
- **Table homography (`table_calib.json`).** Tape 4 small crosses at these base-frame points, measured with a ruler from the pan axis:

  | Cross | x (m) | y (m) | Where (looking along the arm's reach) |
  |---|---|---|---|
  | A | +0.10 | −0.15 | left, near |
  | B | −0.10 | −0.15 | right, near |
  | C | +0.10 | −0.30 | left, far |
  | D | −0.10 | −0.30 | right, far |

  Click or detect them in one iPhone frame. Store `{"image_px": [[u,v]×4], "table_cm": [[x,y]×4]}` with values **in meters**; the key name is kept for compatibility.
- **Joint map (`sim_bridge/joint_map.json`).** With the follower's torque off, hold it by hand in two poses and read `observation.state`:
  - P1 "stretched": the arm fully extended horizontally forward, wrist straight, roll neutral, gripper closed. This is the URDF old-calib zero pose.
  - P2 "L": the upper arm vertical, the forearm horizontal forward, the wrist straight.
  - P3: P2, but with the pan turned clearly to the robot's left, the wrist bent down, and the roll turned clockwise as seen from behind. This pose is used for signs only.

  The far crosses (r ≈ 0.32 m) are calibration marks only; they sit outside the grasp workspace.

  The sim session's URDF angles are in `sim_bridge/poses.json`. Fill in the real readings there. Run `python -m sim_bridge.joint_map fit poses.json` to get `sign_j` and `offset_j`. The gripper map comes from the real readings fully closed and fully open.
- **Gripper:**
  - LeRobot 0–100 maps to sim [−0.1745, 1.745] rad, with 0 = closed. The real closed/open readings go in `poses.json`.
  - Sim grasps best at 0.30 N·m (about 3.5 N). At 0.6 N·m and above, the box gets wedged out.
  - LeRobot's real default is Max_Torque_Limit 500, roughly 1.0–1.5 N·m. The Isaac session recommends about 150 for the paper box. This is the user's decision, to be checked by hand.
  - LeRobot's `configure()` rewrites the limit to 500 on every connect, so any lower value must be applied after connecting.
- **Phone:** fixed mount, fixed zoom, Center Stage and other video effects off. If the phone moves, redo the homography.

## 7. Runtime (demo)
```
iPhone ──Continuity──► Mac: perception → box[4] ─┐
Mac: follower bus (Feetech, USB) → state[6] ─────┼─► UNO Q (USB link, adb-forwarded TCP)
                                                 │     looped transformer (ONNX) → action chunk[30×6]
Mac: write goal positions ◄──────────────────────┘
Mac: GraspDetector → handover.py (scripted) → release on tug
```
Both arms and the UNO Q plug into the Mac. The board only runs the policy.
