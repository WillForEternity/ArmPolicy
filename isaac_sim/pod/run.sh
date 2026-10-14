#!/usr/bin/env bash
# usage: run.sh script.py [args]   (runs inside the Isaac venv from /root/isaac/unoq_sim)
source /root/isaac/env/bin/activate
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y TERM=xterm LEISAAC_ASSETS_ROOT=/root/isaac/assets PYTHONUNBUFFERED=1
cd /root/isaac/unoq_sim
exec python "$@"
