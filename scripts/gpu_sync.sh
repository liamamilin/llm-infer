#!/usr/bin/env bash
# 把仓库同步到 GPU 机器并（可选）执行命令。
#   GPU_SSH="-p 38798 root@host" GPU_DIR=/root/autodl-tmp/llm-infer scripts/gpu_sync.sh [远程命令...]
# 要求先用 ssh-copy-id 装好密钥；本脚本不处理密码。
set -euo pipefail
: "${GPU_SSH:?设置 GPU_SSH，例如 \"-p 38798 root@host\"}"
GPU_DIR="${GPU_DIR:-/root/autodl-tmp/llm-infer}"
cd "$(dirname "$0")/.."
HOSTPART="${GPU_SSH##* }"; PORTOPT="${GPU_SSH% *}"
rsync -az --delete -e "ssh $PORTOPT" \
  --exclude .git --exclude .venv --exclude node_modules --exclude models --exclude 'docs/.vitepress' \
  --exclude evidence/runs --exclude __pycache__ ./ "$HOSTPART:$GPU_DIR/"
if [ "$#" -gt 0 ]; then
  ssh $GPU_SSH "cd $GPU_DIR && source /root/autodl-tmp/venv/bin/activate && export HF_ENDPOINT=https://hf-mirror.com && $*"
fi
