#!/usr/bin/env bash
# One command: sim chunks -> real-unit episodes -> GPU training (looped MLP) -> ONNX -> UNO Q.
#
#   experiments/build_policy.sh [run_name] [real_episodes.npz ...]
#
# Needs sim_bridge/joint_map.json (real joint units) for the result to transfer to the real arm.
# Optional real teleop episodes (contract npz: ep{i}_state/action/box) are co-trained at 50% of each batch.
# ~4 min end to end. Writes runs/<run_name>/ (policy.pt, normalizer.json, onnx/) and pushes onnx to the board's
# ~/policy/models/<run_name>/ over adb. Never commands the arm.
set -euo pipefail
cd "$(dirname "$0")/.."
RUN="${1:-main}"; shift || true
REAL=("$@")
POD="root@216.81.151.48"; PORT=11266; KEY=~/.ssh/runpod_ed25519
SSH=(ssh -o IdentitiesOnly=yes -p $PORT -i $KEY $POD)
SCP=(scp -q -o IdentitiesOnly=yes -P $PORT -i $KEY)
PY=.venv/bin/python

[ -f sim_bridge/joint_map.json ] || { echo "error: sim_bridge/joint_map.json missing - policy would not transfer to the real arm"; exit 1; }

echo "== 1/5 convert sim chunks to real joint units"
for v in v1 v2; do $PY -m sim_bridge.convert_sim sim_data/raw/sim_demos_$v sim_data/sim_episodes_$v.npz; done

echo "== 2/5 upload"
"${SSH[@]}" "mkdir -p /root/policy_exp/data/$RUN"
"${SCP[@]}" experiments/arch_sweep.py experiments/train_main.py policy/model.py $POD:/root/policy_exp/
"${SCP[@]}" sim_data/sim_episodes_v1.npz sim_data/sim_episodes_v2.npz $POD:/root/policy_exp/data/$RUN/
REAL_ARGS=""
if [ ${#REAL[@]} -gt 0 ]; then
  "${SCP[@]}" "${REAL[@]}" $POD:/root/policy_exp/data/$RUN/
  REAL_ARGS="--real $(for f in "${REAL[@]}"; do printf 'data/%s/%s ' "$RUN" "$(basename "$f")"; done)"
fi

echo "== 3/5 train looped MLP on GPU"
"${SSH[@]}" "cd /root/policy_exp && /root/isaac/env/bin/python train_main.py --data data/$RUN/sim_episodes_v1.npz data/$RUN/sim_episodes_v2.npz $REAL_ARGS --out runs/$RUN 2>&1 | grep -E 'params|step|val err|per-joint|TRAIN_DONE|Error'"

echo "== 4/5 export ONNX"
mkdir -p runs/$RUN
"${SCP[@]}" "$POD:/root/policy_exp/runs/$RUN/*" runs/$RUN/
rm -rf runs/$RUN/onnx
$PY -m policy.export runs/$RUN --loops 4 6 8 2>&1 | grep -E "^K=" | cut -c1-120

echo "== 5/5 push to UNO Q"
adb shell "mkdir -p ~/policy/models/$RUN"
for f in runs/$RUN/onnx/policy_k*.onnx* runs/$RUN/onnx/normalizer.json runs/$RUN/onnx/export_report.json; do adb push "$f" "/home/arduino/policy/models/$RUN/" >/dev/null; done
adb shell "ls ~/policy/models/$RUN"
echo "done: runs/$RUN  (board: ~/policy/models/$RUN)"
