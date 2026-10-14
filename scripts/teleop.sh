#!/usr/bin/env bash
# Leader -> follower teleoperation. Usage: scripts/teleop.sh [extra lerobot-teleoperate flags]
# MAX_STEP caps how far any follower joint may move per control tick (degrees), so the follower
# eases toward the leader's pose at startup instead of snapping to it. Ctrl+C stops and drops torque.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
FOLLOWER=$($PY scripts/arms.py port follower)
LEADER=$($PY scripts/arms.py port leader)
MAX_STEP="${MAX_STEP:-8}"
echo "follower=$FOLLOWER leader=$LEADER max_step=${MAX_STEP}deg/tick"
$PY scripts/hold_pose.py  # so enabling torque holds the current pose instead of jumping to a stale goal

exec .venv/bin/lerobot-teleoperate \
  --robot.type=so101_follower --robot.port="$FOLLOWER" --robot.id=follower \
  --robot.max_relative_target="$MAX_STEP" \
  --teleop.type=so101_leader --teleop.port="$LEADER" --teleop.id=leader \
  --fps=30 "$@"
