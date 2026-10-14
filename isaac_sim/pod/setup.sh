#!/usr/bin/env bash
# Isaac Sim 5.1.0 + Isaac Lab 2.3.2 headless install for RunPod (Ubuntu 24.04, Blackwell)
set -euxo pipefail
export DEBIAN_FRONTEND=noninteractive
export TMPDIR=/dev/shm/tmp UV_CACHE_DIR=/dev/shm/uvcache UV_LINK_MODE=copy PIP_NO_CACHE_DIR=1
mkdir -p $TMPDIR /root/isaac
apt-get update -qq
apt-get install -y -qq tmux git-lfs build-essential cmake libglu1-mesa libxt6 libegl1 libgl1 libxrandr2 libxinerama1 libxcursor1 libxi6 ffmpeg >/dev/null
cd /root/isaac
[ -d env ] || uv venv --seed --python 3.11 env
source env/bin/activate
uv pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
uv pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
[ -d IsaacLab ] || git clone --depth 1 --branch v2.3.2 https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y
./isaaclab.sh --install rsl_rl
rm -rf /dev/shm/uvcache
df -h /
echo SETUP_DONE
