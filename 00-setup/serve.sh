#!/usr/bin/env bash
# Serve a model with vLLM: an OpenAI-compatible API on http://localhost:8000
# Run inside WSL2 (vLLM doesn't run on native Windows) or on Linux.
#
#   bash 00-setup/serve.sh                                   # Qwen2.5-7B-Instruct-AWQ
#   MODEL=Qwen/Qwen2.5-14B-Instruct-AWQ bash 00-setup/serve.sh
#
# One-time install (in a WSL/Linux shell):
#   uv venv ~/vllm-env && source ~/vllm-env/bin/activate && uv pip install "vllm==0.25.0"
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct-AWQ}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.75}"   # share of VRAM vLLM may use (weights + KV cache)
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}" # longest prompt + answer, in tokens

# Keep everything local: no usage statistics sent anywhere.
export VLLM_NO_USAGE_STATS=1 VLLM_DO_NOT_TRACK=1 DO_NOT_TRACK=1

# WSL2 only: vLLM needs pinned memory, which it disables under WSL by default.
if grep -qi microsoft /proc/version 2>/dev/null; then
  export VLLM_WSL2_ENABLE_PIN_MEMORY=1
fi

# Settings used for the measurements in the post:
# - eager mode (no CUDA graphs): avoids graph-capture problems under WSL2 and keeps startup fast.
# - vLLM's built-in sampler: FlashInfer's sampler is compiled on first use and needs the full CUDA toolkit.
export VLLM_USE_FLASHINFER_SAMPLER=0

echo "Serving $MODEL on :8000 (gpu-memory-utilization=$GPU_MEM_UTIL, max-model-len=$MAX_MODEL_LEN)"
echo "Watch the startup log for 'GPU KV cache size: N tokens' -- compare it with kv_budget.py."
exec vllm serve "$MODEL" --port 8000 \
  --gpu-memory-utilization "$GPU_MEM_UTIL" --max-model-len "$MAX_MODEL_LEN" --enforce-eager
