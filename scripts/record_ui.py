"""Teleop data collection driven from the demo UI (same LeRobot v3 format as lerobot-record).

Leader -> follower teleop runs the whole time; episodes are recorded only between UI commands
(the UI shows its record screen because this publishes task="record"):
    "Start recording" / Space   start an episode                      (cmd grab)
    "Save" / Enter              save it, then reset both arms to rest   (cmd release)
    "Discard" / Esc             drop it, then reset both arms to rest   (cmd estop)
The reset powers the LEADER's motors for a few seconds (hands off), then turns them off again. If a
leader joint lags the ramp (stalled, or a hand on it), the leader's torque is cut and teleop resumes.
The rest pose is the first frame of episode 0. An episode auto-saves after --max-s seconds.
Ctrl+C quits (follower torque off, dataset finalized).

    .venv/bin/python scripts/record_ui.py [--root datasets/real_pick_ep0] [--port 8766] [--cam 1]
    open http://localhost:8766/#demo     (Console tab: key D)
"""

import argparse
import subprocess
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lerobot.cameras.opencv import OpenCVCameraConfig  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from lerobot.robots.so_follower import SO101FollowerConfig, SOFollower  # noqa: E402
from lerobot.teleoperators.so_leader import SO101LeaderConfig, SOLeader  # noqa: E402
from lerobot.utils.feature_utils import build_dataset_frame  # noqa: E402

from demo_ui.bridge import DemoBus, calib_overlay  # noqa: E402
from policy.perception import TableFrame, annotate, detect  # noqa: E402

TASK = "Pick up the UNO Q box and hand it to the person"
CAM_KEY = "overhead"
CALIB = ROOT / "table_calib.json"


def port(role: str) -> str:
    return subprocess.check_output([sys.executable, str(ROOT / "scripts/arms.py"), "port", role], text=True).strip()


def rest_pose(root: Path) -> dict[str, float]:
    """Median follower state at the first frame of every episode, keyed by motor name: the rest pose
    demos actually start from (episode 0 alone started ~7-9 deg off on shoulder_lift / wrist_flex)."""
    import pandas as pd

    names = __import__("json").loads((root / "meta/info.json").read_text())["features"]["observation.state"]["names"]
    data = pd.concat(pd.read_parquet(p) for p in sorted((root / "data").glob("*/*.parquet")))
    firsts = np.stack(data[data["frame_index"] == 0]["observation.state"].to_list())
    return {n.removesuffix(".pos"): float(v) for n, v in zip(names, np.median(firsts, 0))}


def reset_to_rest(robot: SOFollower, leader: SOLeader, rest: dict[str, float], bus: DemoBus,
                  fps=30, max_lag=25.0, gripper_open=60.0) -> bool:
    """Release the box, then drive the leader (under its own power) and the follower together to the
    rest pose, then release the leader's torque so it can be moved by hand again.

    Segments: open the gripper where the arm is and wait for the box to drop -> move the joints to
    rest with the gripper open -> close the gripper to its rest value. Returns False if aborted because
    a leader joint fell more than max_lag degrees behind (5 V servos stalling, or a hand on it)."""
    lb = leader.bus
    start = lb.sync_read("Present_Position")
    lb.sync_write("Goal_Position", start)  # torque-on holds the current pose instead of jumping
    lb.enable_torque()

    def ramp(frm: dict, to: dict, seconds: float, hold: float, label: str) -> bool:
        n = max(1, int(seconds * fps))
        for i in range(1, n + int(hold * fps) + 1):
            s = min(1.0, i / n)
            s = s * s * (3 - 2 * s)  # smoothstep: gentle start and stop
            goal = {m: frm[m] + (to[m] - frm[m]) * s for m in frm}
            lb.sync_write("Goal_Position", goal)
            robot.send_action({f"{m}.pos": v for m, v in goal.items()})
            bus.publish(phase="home", action=list(goal.values()))
            time.sleep(1.0 / fps)
            now = lb.sync_read("Present_Position")
            lag = {m: abs(now[m] - goal[m]) for m in goal}
            worst = max(lag, key=lag.get)
            if lag[worst] > max_lag:
                bus.log(f"leader {worst} {lag[worst]:.0f} deg behind while {label}: stopping leader motors, "
                        "move it to rest by hand", "warn")
                return False
        return True

    try:
        opened = {**start, "gripper": max(start["gripper"], gripper_open)}
        at_rest_open = {**rest, "gripper": opened["gripper"]}
        return (ramp(start, opened, 0.7, 0.8, "opening the gripper")
                and ramp(opened, at_rest_open, 3.0, 0.2, "moving to rest")
                and ramp(at_rest_open, rest, 0.7, 0.3, "closing the gripper"))
    finally:
        lb.disable_torque()


class UIFrames(threading.Thread):
    """Box detection + annotated JPEGs for the UI, off the 30 Hz control thread."""

    def __init__(self, bus: DemoBus, hz=10.0):
        super().__init__(daemon=True)
        self.bus, self.period, self.latest, self.table = bus, 1.0 / hz, None, None

    def run(self):
        while True:
            t0, rgb = time.perf_counter(), self.latest
            if rgb is not None:
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                h, w = bgr.shape[:2]
                self.table = self.table or TableFrame((w, h), CALIB)
                pose = detect(bgr)
                if pose is not None:
                    x, y, c2, s2 = self.table.features(pose)[:4]
                    self.bus.publish(box={"xy": [float(x), float(y)],
                                          "theta_deg": float(np.degrees(np.arctan2(s2, c2)) / 2),
                                          "corners_px": (np.asarray(pose.corners()) / [w, h]).round(4).tolist(),
                                          "latched": False})
                small = cv2.resize(annotate(bgr, pose), (w // 2, h // 2))
                ok, jpg = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 70])
                if ok:
                    self.bus.publish_frame(jpg.tobytes())
            time.sleep(max(0.0, self.period - (time.perf_counter() - t0)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="datasets/real_pick_ep0")
    p.add_argument("--repo-id", default="local/real_pick_ep0")
    p.add_argument("--port", type=int, default=8766)
    p.add_argument("--cam", type=int, default=1)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--max-s", type=float, default=30.0)
    p.add_argument("--max-step", type=float, default=8.0)
    p.add_argument("--no-auto-reset", dest="auto_reset", action="store_false",
                   help="don't power the arms back to rest after Save/Discard")
    p.add_argument("--gripper-torque", type=int, default=300, help="follower gripper Torque_Limit (0-1000)")
    args = p.parse_args()

    bus = DemoBus(args.port).start()
    bus.publish(mode="live", phase="idle", calib=calib_overlay(CALIB, (1920, 1080)),
                links={"camera": False, "follower": False, "board": False})

    rest = rest_pose(ROOT / args.root)
    dataset = LeRobotDataset.resume(args.repo_id, root=ROOT / args.root, image_writer_threads=4)
    bus.log(f"dataset {args.root}: {dataset.num_episodes} episodes already saved")
    bus.log("rest pose (episode 0, frame 0): " + ", ".join(f"{k} {v:.0f}" for k, v in rest.items()))

    leader = SOLeader(SO101LeaderConfig(port=port("leader"), id="leader"))
    robot = SOFollower(SO101FollowerConfig(
        port=port("follower"), id="follower", max_relative_target=args.max_step,
        cameras={CAM_KEY: OpenCVCameraConfig(index_or_path=args.cam, width=1920, height=1080, fps=args.fps)}))
    subprocess.run([sys.executable, str(ROOT / "scripts/hold_pose.py")], check=True)
    leader.connect()
    robot.connect()
    # Holding the box through a handover at LeRobot's default gripper torque (500) trips the servo's
    # overload protection and drops the box. Torque_Limit is the live RAM limit; configure() resets it.
    robot.bus.write("Torque_Limit", "gripper", args.gripper_torque)
    bus.publish(links={"camera": True, "follower": True})
    n_start = dataset.num_episodes
    bus.publish(task="record", saved=0)
    bus.log("teleop live: Start recording, then Save or Discard"
            + (" (both reset the arms to rest)" if args.auto_reset else "; reset the arms by hand"), "event")

    ui = UIFrames(bus)
    ui.start()
    recording, saving, t_start, n_frames = False, None, 0.0, 0
    period = 1.0 / args.fps
    try:
        while True:
            t0 = time.perf_counter()
            for cmd in bus.poll_commands():
                c = cmd.get("cmd")
                if c == "grab" and not recording:
                    if saving is not None and saving.is_alive():
                        bus.log("still saving the last episode, try again in a moment", "warn")
                    else:
                        recording, t_start, n_frames = True, time.time(), 0
                        bus.log(f"recording episode {dataset.num_episodes}", "event")
                elif c in ("release", "estop"):
                    if not recording:
                        pass  # nothing to save; still reset the arms below
                    elif c == "release" and n_frames > 0:
                        saving = threading.Thread(target=dataset.save_episode, daemon=True)
                        saving.start()
                        bus.log(f"saving episode ({n_frames} frames, {n_frames / args.fps:.1f}s)", "event")
                    else:
                        dataset.clear_episode_buffer()
                        bus.log("episode discarded", "warn")
                    recording = False
                    if args.auto_reset:
                        bus.log("resetting both arms to rest: hands off the leader", "event")
                        if reset_to_rest(robot, leader, rest, bus):
                            bus.log("at rest: teleop live again", "event")

            try:
                obs = robot.get_observation()
            except TimeoutError as e:  # camera hiccup: skip this tick instead of dropping the arm
                bus.log(f"camera stalled, skipping a frame ({e})", "warn")
                time.sleep(period)
                continue
            act = leader.get_action()
            robot.send_action(act)
            ui.latest = obs[CAM_KEY]

            if recording:
                frame = {**build_dataset_frame(dataset.features, obs, prefix="observation"),
                         **build_dataset_frame(dataset.features, act, prefix="action"), "task": TASK}
                dataset.add_frame(frame)
                n_frames += 1
                if time.time() - t_start >= args.max_s:
                    recording = False
                    saving = threading.Thread(target=dataset.save_episode, daemon=True)
                    saving.start()
                    bus.log(f"max length reached, saving episode ({n_frames} frames); press Stop to reset arms", "event")

            joints = list(act.keys())
            busy = saving is not None and saving.is_alive()
            bus.publish(phase="pick" if recording else ("home" if busy else "idle"),
                        saved=dataset.num_episodes - n_start,
                        phase_t=(time.time() - t_start) if recording else 0.0,
                        state=[float(obs[k]) for k in joints], action=[float(act[k]) for k in joints])
            time.sleep(max(0.0, period - (time.perf_counter() - t0)))
    except KeyboardInterrupt:
        pass
    finally:
        if recording:
            dataset.clear_episode_buffer()
        if saving is not None:
            saving.join()
        dataset.finalize()
        robot.disconnect()
        leader.disconnect()
        print(f"done: {dataset.num_episodes} episodes in {args.root}")


if __name__ == "__main__":
    main()
