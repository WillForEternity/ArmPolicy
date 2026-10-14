"""Set every follower servo's Goal_Position to its Present_Position, with torque off.

STS3215 servos drive toward Goal_Position the moment torque is enabled. After calibration the goal
register can be 0, so enabling torque swings each joint to the end of its range and the current spike
can brown out the bus. Running this first makes torque-enable a no-op hold of the current pose.
"""

import subprocess
import sys
from pathlib import Path

from scservo_sdk import COMM_SUCCESS, PacketHandler, PortHandler

ADDR_TORQUE_ENABLE, ADDR_GOAL_POSITION, ADDR_PRESENT_POSITION = 40, 42, 56

port = subprocess.check_output(
    [sys.executable, Path(__file__).resolve().parent / "arms.py", "port", "follower"], text=True
).strip()
ph, pk = PortHandler(port), PacketHandler(0)
if not (ph.openPort() and ph.setBaudRate(1_000_000)):
    sys.exit(f"error: cannot open {port}")
try:
    for motor_id in range(1, 7):
        torque, res, _ = pk.read1ByteTxRx(ph, motor_id, ADDR_TORQUE_ENABLE)
        if res != COMM_SUCCESS:
            sys.exit(f"error: servo {motor_id} not responding")
        if torque:
            sys.exit(f"error: servo {motor_id} already has torque on; not touching it")
        pos, res, _ = pk.read2ByteTxRx(ph, motor_id, ADDR_PRESENT_POSITION)
        res2, _ = pk.write2ByteTxRx(ph, motor_id, ADDR_GOAL_POSITION, pos)
        goal, res3, _ = pk.read2ByteTxRx(ph, motor_id, ADDR_GOAL_POSITION)
        if COMM_SUCCESS not in (res,) or res2 != COMM_SUCCESS or goal != pos:
            sys.exit(f"error: could not set servo {motor_id} goal (pos={pos}, goal={goal})")
        print(f"servo {motor_id}: goal = present = {pos}")
finally:
    ph.closePort()
