"""Print leader vs follower joint readings in LeRobot units (read-only, no torque).
Joints are degrees; gripper is 0-100. Pose both arms the same way first: rows should agree."""
import subprocess, sys
from pathlib import Path
from lerobot.robots.so_follower import SO101FollowerConfig, SOFollower
from lerobot.teleoperators.so_leader import SO101LeaderConfig, SOLeader

here = Path(__file__).resolve().parent
port = lambda r: subprocess.check_output([sys.executable, here / "arms.py", "port", r], text=True).strip()
f = SOFollower(SO101FollowerConfig(port=port("follower"), id="follower"))
l = SOLeader(SO101LeaderConfig(port=port("leader"), id="leader"))
for d in (f, l):
    d.bus.connect()
try:
    fp, lp = f.bus.sync_read("Present_Position"), l.bus.sync_read("Present_Position")
finally:
    for d in (f, l):
        d.bus.disconnect(disable_torque=False)
print(f"{'joint':14s} {'leader':>8s} {'follower':>9s} {'diff':>7s}")
for m in fp:
    print(f"{m:14s} {lp[m]:8.1f} {fp[m]:9.1f} {fp[m] - lp[m]:7.1f}")
