"""Run the exported policy on the real follower from the demo UI (first real-arm test).

    "Grab me the Arduino Q" / Space   lock the box pose seen by the overhead camera, run the policy
    grasp detected                    0.5 s more of policy (its short lift), then hold still
    Enter                             release: open the gripper, pause, go to the rest pose
    Stop / Esc                        same release-and-home, from any phase
The pick times out after --pick-s and releases/homes the same way. The scripted handover lift in
policy/handover.py is NOT used: its offsets haven't been checked on the real arm yet.

Inference here is ONNX Runtime FP32 on the Mac (the UNO Q runtime is the next step).

    .venv/bin/python scripts/run_policy.py --run runs/finetune_real --loops 8
    open http://localhost:8766/#demo
"""

import argparse
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from lerobot.cameras.opencv import OpenCVCameraConfig  # noqa: E402
from lerobot.robots.so_follower import SO101FollowerConfig, SOFollower  # noqa: E402

from demo_ui.bridge import DemoBus, calib_overlay  # noqa: E402
from policy.data import FPS, JOINTS  # noqa: E402
from policy.handover import GraspDetector, HandoverConfig, min_jerk, release_and_home  # noqa: E402
from policy.handover_learned import trajectory as handover_trajectory  # noqa: E402
from policy.perception import BoxTracker, TableFrame, annotate  # noqa: E402
from record_ui import port, rest_pose  # noqa: E402

CAM_KEY = "overhead"
CALIB = ROOT / "table_calib.json"


class BoxWatcher(threading.Thread):
    """Box pose from the overhead frames + annotated frames for the UI.

    While `free` (arm idle, nothing near the box) every frame is a fresh detection, so moving or
    rotating the box between runs is picked up immediately and a removed box clears the pose.
    Only while not free (a pick in progress) does BoxTracker hold the last arm-clear pose."""

    def __init__(self, bus: DemoBus, hz=10.0):
        super().__init__(daemon=True)
        self.bus, self.period = bus, 1.0 / hz
        self.latest, self.table, self.tracker, self.features = None, None, BoxTracker(), None
        self.free = True

    def run(self):
        while True:
            t0, rgb = time.perf_counter(), self.latest
            if rgb is not None:
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                h, w = bgr.shape[:2]
                self.table = self.table or TableFrame((w, h), CALIB)
                if self.free:
                    self.tracker.reset()  # no hold between runs: always the box where it is now
                pose = self.tracker.update(bgr)
                if pose is None and self.free:
                    self.features = None
                    self.bus.publish(box={"xy": None, "theta_deg": None, "corners_px": None})
                if pose is not None:
                    f = self.table.features(pose)[:4].astype(np.float32)
                    self.features = f
                    self.bus.publish(box={"xy": [float(f[0]), float(f[1])],
                                          "theta_deg": float(np.degrees(np.arctan2(f[3], f[2])) / 2),
                                          "corners_px": (np.asarray(pose.corners()) / [w, h]).round(4).tolist()})
                ok, jpg = cv2.imencode(".jpg", cv2.resize(annotate(bgr, pose), (w // 2, h // 2)),
                                       [cv2.IMWRITE_JPEG_QUALITY, 70])
                if ok:
                    self.bus.publish_frame(jpg.tobytes())
            time.sleep(max(0.0, self.period - (time.perf_counter() - t0)))


def set_grip_torque(robot: SOFollower, bus: DemoBus, value: int) -> None:
    """Change the gripper's live torque limit without ever taking the runner down: a servo that has
    tripped its overload protection answers writes with an error, and a release must still happen."""
    try:
        robot.bus.write("Torque_Limit", "gripper", value)
    except (RuntimeError, ConnectionError) as e:
        bus.log(f"gripper torque limit not set ({e}); continuing", "warn")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="runs/finetune_real")
    p.add_argument("--loops", type=int, default=8)
    p.add_argument("--rest-from", default="datasets/real_pick_ep0", help="dataset whose median episode start pose is home")
    p.add_argument("--port", type=int, default=8766)
    p.add_argument("--cam", type=int, default=1)
    p.add_argument("--replan", type=int, default=10, help="execute this many chunk steps, then re-infer")
    p.add_argument("--pick-s", type=float, default=20.0)
    p.add_argument("--after-grasp-s", type=float, default=0.5)
    p.add_argument("--max-step", type=float, default=8.0)
    p.add_argument("--gripper-torque", type=int, default=300)
    p.add_argument("--handover", default="handover_template.json", help="from python -m policy.handover_learned")
    p.add_argument("--handover-speed", type=float, default=1.0, help=">1 is faster than the demos")
    p.add_argument("--squeeze", type=float, default=15.0, help="gripper command this far below the jaws' stop on the box")
    p.add_argument("--hold-torque", type=int, default=300, help="gripper Torque_Limit from grasp until release (above ~300 the servo trips overload while squeezing)")
    p.add_argument("--tug-deg", type=float, default=6.0, help="extra joint deflection that counts as a tug")
    args = p.parse_args()

    sess = ort.InferenceSession(str(ROOT / args.run / "onnx" / f"policy_k{args.loops}.onnx"),
                                providers=["CPUExecutionProvider"])
    in_state, in_env = (i.name for i in sess.get_inputs())
    rest = rest_pose(ROOT / args.rest_from)
    home = np.float32([rest[j] for j in JOINTS])
    cfg = HandoverConfig(home=tuple(home.tolist()))
    handover_fit = __import__("json").loads((ROOT / args.handover).read_text())
    tug_base, tug_n, grip_hold = [], 0, 0.0

    bus = DemoBus(args.port).start()
    bus.publish(mode="live", task="pick", phase="idle", calib=calib_overlay(CALIB, (1920, 1080)),
                links={"camera": False, "follower": False, "board": False})

    robot = SOFollower(SO101FollowerConfig(
        port=port("follower"), id="follower", max_relative_target=args.max_step,
        cameras={CAM_KEY: OpenCVCameraConfig(index_or_path=args.cam, width=1920, height=1080, fps=FPS)}))
    subprocess.run([sys.executable, str(ROOT / "scripts/hold_pose.py")], check=True)
    robot.connect()
    robot.bus.write("Torque_Limit", "gripper", args.gripper_torque)
    bus.publish(links={"camera": True, "follower": True})
    bus.log(f"policy {args.run} K={args.loops} on this Mac (ONNX FP32); press Grab to run one pick", "event")
    stale = bus.poll_commands()  # anything a browser tab sent while we were starting up is not a fresh press
    if stale:
        bus.log(f"ignored {len(stale)} command(s) sent before the runner was ready", "warn")

    watch = BoxWatcher(bus)
    watch.start()
    keys = [f"{j}.pos" for j in JOINTS]
    phase, t_phase, box, chunk, ci = "idle", time.time(), None, None, 0
    det, hold_q, traj, ms_hist = GraspDetector(), None, None, deque(maxlen=90)
    period = 1.0 / FPS
    try:
        while True:
            t0 = time.perf_counter()
            try:
                obs = robot.get_observation()
            except TimeoutError as e:
                bus.log(f"camera stalled, skipping a tick ({e})", "warn")
                time.sleep(period)
                continue
            watch.latest = obs[CAM_KEY]
            q = np.float32([obs[k] for k in keys])

            for cmd in bus.poll_commands():
                c = cmd.get("cmd")
                if c == "grab" and phase == "idle":
                    if watch.features is None:
                        bus.log("no box seen yet: put the box in front of the arm", "warn")
                        continue
                    box = watch.features.copy()
                    det.reset()
                    off = float(np.abs(q[:5] - home[:5]).max())
                    if off > 3.0:  # start every pick from the pose the teleop demos start from
                        traj, ci, phase, t_phase = min_jerk(q, home, max(1.0, off / 30.0)), 0, "prehome", time.time()
                        bus.log(f"moving to the teleop start pose first ({off:.0f} deg away)", "event")
                    else:
                        phase, t_phase, chunk, ci = "pick", time.time(), None, 0
                    bus.log(f"pick: box at x={box[0]*100:.1f} cm, y={box[1]*100:.1f} cm", "event")
                elif c in ("release", "estop") and phase == "prehome":
                    phase, traj = "idle", None
                    bus.log("cancelled", "warn")
                elif c in ("release", "estop") and phase in ("pick", "lift", "handover", "hold"):
                    hold_q = q.copy() if hold_q is None else hold_q
                    traj, ci, phase, t_phase = release_and_home(hold_q, cfg), 0, "release", time.time()
                    set_grip_torque(robot, bus, args.gripper_torque)
                    bus.log("release and home", "event")

            target = None
            if phase == "prehome":
                target = traj[min(ci, len(traj) - 1)]
                ci += 1
                if ci >= len(traj) + FPS // 3:  # settle, then start the policy from the start pose
                    phase, t_phase, chunk, ci, traj = "pick", time.time(), None, 0, None
            elif phase in ("pick", "lift"):
                if chunk is None or ci >= args.replan:
                    t_inf = time.perf_counter()
                    chunk = sess.run(None, {in_state: q[None], in_env: box[None]})[0][0]
                    ms_hist.append((time.perf_counter() - t_inf) * 1000)
                    ci = 0
                target = chunk[ci].copy()
                ci += 1
                if phase == "lift":
                    target[-1] = grip_hold
                if phase == "pick" and det.update(float(target[-1]), float(q[-1])):
                    phase, t_phase = "lift", time.time()
                    # From here on the gripper is not the policy's: squeeze args.squeeze below where the
                    # jaws stopped on the box, so it stays clamped through lift, handover and hold.
                    grip_hold = max(0.0, min(float(target[-1]), float(q[-1]) - args.squeeze))
                    set_grip_torque(robot, bus, args.hold_torque)
                    bus.log(f"grasp detected: jaws at {q[-1]:.0f}, clamping at {grip_hold:.0f}", "event")
                elif phase == "pick" and time.time() - t_phase > args.pick_s:
                    hold_q = q.copy()
                    traj, ci, phase, t_phase = release_and_home(hold_q, cfg), 0, "release", time.time()
                    set_grip_torque(robot, bus, args.gripper_torque)
                    bus.log(f"no grasp within {args.pick_s:.0f}s: releasing and going home", "warn")
                    target = None
                elif phase == "lift" and time.time() - t_phase > args.after_grasp_s:
                    traj = handover_trajectory(target.copy(), grip_hold, handover_fit, args.handover_speed)
                    phase, t_phase, ci = "handover", time.time(), 0
                    bus.log(f"handing over ({len(traj) / FPS:.1f}s, learned from the demos)", "event")
            elif phase == "handover":
                target = traj[min(ci, len(traj) - 1)]
                ci += 1
                if ci >= len(traj):
                    hold_q, phase, t_phase, tug_base, tug_n = target.copy(), "hold", time.time(), [], 0
                    bus.log("holding the box out: take it (a tug releases it) or press Enter", "event")
            elif phase == "hold":
                target = hold_q
                dev = float(np.abs(q[:5] - hold_q[:5]).max())  # arm joints pushed off their goal
                if time.time() - t_phase < 0.7:
                    tug_base.append(dev)  # sag under the box's weight while settling
                else:
                    tug_n = tug_n + 1 if dev > max(tug_base, default=0.0) + args.tug_deg else 0
                    if tug_n >= 3:
                        traj, ci, phase, t_phase = release_and_home(hold_q, cfg), 0, "release", time.time()
                        set_grip_torque(robot, bus, args.gripper_torque)
                        bus.log(f"tug felt ({dev:.0f} deg): releasing", "event")
            elif phase == "release":
                target = traj[min(ci, len(traj) - 1)]
                ci += 1
                if ci >= len(traj) + FPS // 2:
                    phase, hold_q, traj = "idle", None, None
                    bus.log("home: ready for another pick", "event")

            watch.free = phase == "idle"
            if target is not None:
                robot.send_action({k: float(v) for k, v in zip(keys, target)})
            bus.publish(phase={"prehome": "pick", "handover": "lift"}.get(phase, phase),
                        phase_t=time.time() - t_phase, state=q.tolist(),
                        action=(target.tolist() if target is not None else q.tolist()),
                        chunk=(chunk.tolist() if chunk is not None and phase in ("pick", "lift") else None),
                        infer={"host": "mac", "ms": ms_hist[-1] if ms_hist else None, "ms_hist": list(ms_hist),
                               "loops": args.loops, "model": f"finetune_real_k{args.loops}_fp32"},
                        grasped=phase in ("lift", "handover", "hold"))
            time.sleep(max(0.0, period - (time.perf_counter() - t0)))
    except KeyboardInterrupt:
        pass
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
