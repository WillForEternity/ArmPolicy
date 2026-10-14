#!/usr/bin/env bash
# Record leader->follower teleop demos of "pick up the box and hand it over" (CONTRACT.md §5).
# Usage: scripts/record.sh [num_episodes]   (re-running appends to the same dataset)
# Episodes are fixed-length; LeRobot speaks "Recording episode N" / "Reset the environment".
# Env: CAM_INDEX (iPhone overhead, default 1), EPISODE_S (default 20), RESET_S (default 12), MAX_STEP.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
NUM="${1:-10}"
FOLLOWER=$($PY scripts/arms.py port follower)
LEADER=$($PY scripts/arms.py port leader)
ROOT=datasets/real_pick_handover
RESUME=false; [ -d "$ROOT" ] && RESUME=true
echo "follower=$FOLLOWER leader=$LEADER cam=${CAM_INDEX:-1} episodes=$NUM resume=$RESUME"
$PY scripts/hold_pose.py  # torque-on holds the current pose instead of jumping to a stale goal

exec .venv/bin/lerobot-record \
  --robot.type=so101_follower --robot.port="$FOLLOWER" --robot.id=follower \
  --robot.max_relative_target="${MAX_STEP:-8}" \
  --robot.cameras="{overhead: {type: opencv, index_or_path: ${CAM_INDEX:-1}, width: 1920, height: 1080, fps: 30}}" \
  --teleop.type=so101_leader --teleop.port="$LEADER" --teleop.id=leader \
  --dataset.repo_id=local/real_pick_handover --dataset.root="$ROOT" \
  --dataset.single_task="Pick up the UNO Q box and hand it to the person" \
  --dataset.fps=30 --dataset.num_episodes="$NUM" \
  --dataset.episode_time_s="${EPISODE_S:-20}" --dataset.reset_time_s="${RESET_S:-12}" \
  --dataset.push_to_hub=false --resume="$RESUME"
