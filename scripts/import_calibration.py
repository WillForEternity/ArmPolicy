"""Write a LeRobot calibration file from the calibration already stored in an arm's servos.

An arm calibrated before keeps its homing offsets and position limits in servo EEPROM. This reads
them (read-only: no torque, no writes) and saves the JSON that lerobot-calibrate would have written,
so the arm can be used without redoing the hand-guided range sweep.

    import_calibration.py follower|leader
"""

import subprocess
import sys
from pathlib import Path

from lerobot.robots.so_follower import SO101FollowerConfig, SOFollower
from lerobot.teleoperators.so_leader import SO101LeaderConfig, SOLeader

HERE = Path(__file__).resolve().parent


def main() -> None:
    role = sys.argv[1] if len(sys.argv) > 1 else ""
    if role not in ("follower", "leader"):
        sys.exit(__doc__)
    port = subprocess.run([sys.executable, HERE / "arms.py", "assign", role],
                          capture_output=True, text=True)
    if port.returncode:
        sys.exit(port.stderr.strip())
    port = port.stdout.strip()

    if role == "follower":
        dev = SOFollower(SO101FollowerConfig(port=port, id="follower"))
    else:
        dev = SOLeader(SO101LeaderConfig(port=port, id="leader"))

    dev.bus.connect()  # opens the port and pings all 6 motors; does not touch torque
    try:
        calib = dev.bus.read_calibration()
    finally:
        dev.bus.disconnect(disable_torque=False)

    for name, c in calib.items():
        print(f"{name:14s} id={c.id} offset={c.homing_offset:6d} range=[{c.range_min}, {c.range_max}]")
        if c.range_min >= c.range_max:
            sys.exit(f"error: {name} has an empty range; this arm needs a real calibration (calibrate.sh {role})")
    dev.calibration = calib
    dev._save_calibration()
    print(f"saved {dev.calibration_fpath}")


if __name__ == "__main__":
    main()
