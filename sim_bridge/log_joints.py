"""Log the follower's joint positions (LeRobot degrees, gripper 0-100) while the user moves it by hand.

Read-only: never enables torque. Refuses to run if any motor already has torque on, since moving a
torqued arm by hand fights the servos. Output: CSV of t, 6 joints.

    python -m sim_bridge.log_joints OUT.csv [seconds]
"""

import json
import subprocess
import sys
import time
from pathlib import Path

from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus

ROOT = Path(__file__).resolve().parent.parent
CALIB = Path.home() / ".cache/huggingface/lerobot/calibration/robots/so_follower/follower.json"
JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")


def main():
    out, seconds = Path(sys.argv[1]), float(sys.argv[2]) if len(sys.argv) > 2 else 360
    port = subprocess.check_output([sys.executable, str(ROOT / "scripts/arms.py"), "port", "follower"], text=True).strip()
    calib = {k: MotorCalibration(**v) for k, v in json.loads(CALIB.read_text()).items()}
    motors = {n: Motor(i + 1, "sts3215", MotorNormMode.DEGREES) for i, n in enumerate(JOINTS[:5])}
    motors["gripper"] = Motor(6, "sts3215", MotorNormMode.RANGE_0_100)
    bus = FeetechMotorsBus(port=port, motors=motors, calibration=calib)
    bus.connect()
    torque = bus.sync_read("Torque_Enable", normalize=False)
    if any(torque.values()):
        bus.port_handler.closePort()
        sys.exit(f"torque is ON for {[k for k, v in torque.items() if v]}; not touching it. Stop whatever holds the arm first.")
    print(f"reading {port}, torque off on all motors, logging {seconds:.0f}s to {out}", flush=True)
    t0 = time.time()
    with out.open("w") as f:
        f.write("t," + ",".join(JOINTS) + "\n")
        while time.time() - t0 < seconds:
            q = bus.sync_read("Present_Position")
            f.write(f"{time.time() - t0:.3f}," + ",".join(f"{q[j]:.2f}" for j in JOINTS) + "\n")
            f.flush()
            time.sleep(0.05)
    bus.port_handler.closePort()  # plain close: torque is already off, nothing to write


if __name__ == "__main__":
    main()
