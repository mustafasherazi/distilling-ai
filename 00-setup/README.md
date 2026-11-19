# 00 · Setup

Checks that your machine can run the posts in this repo.

```bash
uv run 00-setup/check_gpu.py                 # GPU and VRAM via nvidia-smi
uv sync --extra torch
uv run --extra torch 00-setup/check_gpu.py   # + PyTorch CUDA and a test matmul
```

Expected output (my machine):

```
GPU: NVIDIA GeForce RTX 4070 Ti SUPER | VRAM: 16376 MiB
```

Post: *The full LLM stack on one 16GB GPU: the plan and the setup* (coming soon).
