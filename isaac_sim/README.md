# Isaac Sim: SO-101 UNO Q box pick

Scripted demo generation for pretraining the pick policy. Physics only (PhysX, headless). The data format and frames are defined in `../CONTRACT.md`.

## Pod (RunPod, RTX PRO 6000 Blackwell, Ubuntu 24.04)

```
ssh root@216.81.151.48 -p 11266 -i ~/.ssh/runpod_ed25519
/root/isaac/run.sh <script.py> [args]      # venv + EULA + LEISAAC_ASSETS_ROOT, cwd = /root/isaac/unoq_sim
```

Stack: Isaac Sim 5.1.0 (pip), Isaac Lab v2.3.2, LeIsaac 0.4.0 (SO-101 USD in LeRobot joint conventions), torch 2.7.0+cu128, Python 3.11.

Install is `pod/setup.sh` and then `pod/setup2.sh`, plus these fixes I applied by hand:

- `isaaclab.sh` calls `tabs`, so run it with `TERM=xterm`.
- The core `isaaclab` package fails to build because `flatdict` needs `pkg_resources`. Fix:
  `pip install "setuptools<81" && pip install --no-build-isolation flatdict==4.0.1 && pip install -e IsaacLab/source/isaaclab`
- `lerobot` upgrades numpy and gymnasium. Re-pin with:
  `pip install numpy==1.26.0 gymnasium==1.2.1 packaging==24.2 psutil==5.9.8`
- SO-101 USD: `assets/robots/so101_follower.usd` from the LeIsaac v0.1.0 GitHub release.

**Known limitation: there are no camera renders on this pod.** The host driver is 595.91 (new feature branch). Isaac Sim 5.1's RTX renderer segfaults in `librtx.scenedb` on 595.x, and NVIDIA validates 5.1 on driver 580.x. Physics-only headless works. For rendered cameras (videos, vision data), rent a pod on a host with a 580 production driver.

## Files

| file | what |
|---|---|
| `unoq_task.py` | Isaac Lab env. SO-101 at the origin on a dark table, measured box 89×66×25 mm, 50–120 g, friction 0.35–0.6, spawn r 0.12–0.28 m from the pan axis, bearing ±34°, any yaw. 30 Hz control, gripper torque cap 0.30 N·m. |
| `gen_demos.py` | Parallel scripted demos: approach, descend, close, lift, then the scripted hand-back (shoulder_lift −45°) and hold. Uses DLS IK with integral action, and DART noise on executed actions (clean actions are recorded). Writes npz chunks. |
| `probe.py`, `probe_gripper.py`, `fk_poses.py` | Kinematics and gripper geometry probes, and FK checks of the calibration poses P1–P3. |
| `measure/` | Box size measurement from the iPhone frame: perspective-rectified lid, scaled by the 1:1 printed UNO footprint. |

Example: `/root/isaac/run.sh gen_demos.py --num_envs 1024 --rounds 12 --quiet --seed 1 --out /root/isaac/data/sim_demos_v1` takes about 6 s per round of 1024 episodes.

## Grasp search (8,192 random grasps, no noise, `gen_demos.py --search --p_length_axis 0.25 --r_max 0.33`)

- Closing across the 89 mm side succeeded 0 times in 2,057 tries; the SO-101 jaws are too small. Only grasps across the 66 mm side work.
- Approach tilt: straight down is best (42% overall). ±9–15° drops to 16–17%, and tilt does not extend reach.
- Grasp height ≤ 9 mm is fine (76–79% in the good region); 9–11 mm drops to 49%.
- Opening 54–61° and clearance 5–10 mm are best. Offset (±30% along the box), hover/lift height and speed don't matter.
- Default sampling uses the validated envelope: width axis, tilt ±4°, z 4–9 mm, open 52–63°, clearance 4–10 mm, 30% wrist flip. `--fixed_recipe` reproduces v1.

Datasets in `../sim_data/raw/`: `sim_demos_v1` uses one recipe (7,941 successful episodes); `sim_demos_v2` varies the grasp (5,988 successful). Both have DART noise.

Box size was checked on a second iPhone frame (`measure/iphone_frame2/`): 65.9 × 88.5 mm, against 65.8 × 88.9 mm from frame 1.

## Tuned numbers

- **Grasp:** top-down across the 66 mm side. TCP at 6.5 mm (fixed-jaw tip just above the table), open 55°, the fixed jaw placed 6 mm outside the box edge.
- **Gripper cap:** swept 0.2–1.0 N·m. 0.30 N·m (≈3.5 N squeeze) gives 85% success. Higher caps wedge the box out (0–1%).
- **Success by reach:** 94% at r < 0.16 m, 80–85% at 0.16–0.28 m, 49% at 0.28–0.31 m, 31% beyond 0.31 m.
