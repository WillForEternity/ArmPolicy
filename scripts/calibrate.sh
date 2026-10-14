#!/usr/bin/env bash
# Calibrate one SO-101 arm with LeRobot. Usage: scripts/calibrate.sh follower|leader
# The first run for a role binds that role to whichever single unassigned arm adapter is plugged in.
set -euo pipefail
cd "$(dirname "$0")/.."
ROLE="${1:?usage: scripts/calibrate.sh follower|leader}"
PY=.venv/bin/python

PORT=$($PY scripts/arms.py assign "$ROLE")
echo "Calibrating $ROLE on $PORT"
echo "  1. Move the arm to the MIDDLE of every joint's range, press ENTER."
echo "  2. Move EVERY joint (incl. wrist roll and gripper) through its FULL range, press ENTER."

if [ "$ROLE" = follower ]; then
  exec .venv/bin/lerobot-calibrate --robot.type=so101_follower --robot.port="$PORT" --robot.id=follower
else
  exec .venv/bin/lerobot-calibrate --teleop.type=so101_leader --teleop.port="$PORT" --teleop.id=leader
fi
