set -ex
export TERM=xterm TMPDIR=/dev/shm/tmp UV_CACHE_DIR=/dev/shm/uvcache PIP_NO_CACHE_DIR=1 OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y
source /root/isaac/env/bin/activate
cd /root/isaac/IsaacLab
./isaaclab.sh --install rsl_rl
cd /root/isaac
[ -d leisaac ] || git clone --depth 1 https://github.com/LightwheelAI/leisaac.git
pip install -e "leisaac/source/leisaac[remote]"
pip install "lerobot==0.4.2" || echo LEROBOT_FAILED
python -c "import torch;print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
df -h /
echo SETUP2_DONE
