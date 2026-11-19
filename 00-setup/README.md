# 00 · Setup: the full LLM stack on one 16GB GPU

Post: *The full LLM stack on one 16GB GPU: the plan and the setup* (link added when published).

## What you'll build

A working local setup: GPU check, a model served with vLLM behind an OpenAI-compatible API, a load test, and a quick way to estimate what fits in your VRAM.

## Run it

```bash
# 1. Check the GPU (Windows or Linux)
uv run 00-setup/check_gpu.py

# 2. What fits? KV cache budget for a model (needs internet for config.json only)
uv run 00-setup/kv_budget.py Qwen/Qwen2.5-7B-Instruct-AWQ --weights-gib 5.2 --util 0.75

# 3. Serve a model (WSL2 or Linux), then 4. benchmark it from a second shell
bash 00-setup/serve.sh
bash 00-setup/bench.sh
```

## Expected output (my machine)

`check_gpu.py`:
```
GPU: NVIDIA GeForce RTX 4070 Ti SUPER | VRAM: 16376 MiB
```

`kv_budget.py` for the 7B and 14B (AWQ, 4-bit weights):
```
Qwen2.5-7B-Instruct-AWQ   KV cache per token: 56 KiB   Tokens that fit: ~127,000
Qwen2.5-14B-Instruct-AWQ  KV cache per token: 192 KiB  Tokens that fit: ~14,200
```
vLLM itself reported 119,232 and 12,576 tokens. The estimate ignores activation memory, so it lands slightly high.

## What I measured (RTX 4070 Ti Super 16GB, vLLM 0.25.0)

Qwen2.5-7B-Instruct-AWQ, `--gpu-memory-utilization 0.75`, 8K context. aiperf 0.11.0, 512 input / 256 output tokens.

| Concurrent users | Speed per user | Time to first token (p50) | Total output |
| --- | --- | --- | --- |
| 1 | 58 tok/s | 103 ms | 56 tok/s |
| 8 | 56 tok/s | 611 ms | 394 tok/s |
| 16 | 50 tok/s | 873 ms | 683 tok/s |
| 32 | 42 tok/s | 1,119 ms | 1,070 tok/s |

Cross-checked with guidellm 0.7.1 at 16 users: 662 tok/s, 876 ms TTFT (within ~3%).

## Try next

- Run `MODEL=Qwen/Qwen2.5-14B-Instruct-AWQ` in both scripts: per-user speed drops to ~34 tok/s, and at 16 users throughput stalls around 360 tok/s because the KV cache fills.
- Lower `GPU_MEM_UTIL` and watch the "GPU KV cache size" line in the vLLM log shrink.
