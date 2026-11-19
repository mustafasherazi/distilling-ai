"""Check that this machine is ready for the posts: GPU, VRAM, and (optionally) PyTorch + CUDA.

Run:  uv run 00-setup/check_gpu.py            (no PyTorch needed)
      uv run --extra torch 00-setup/check_gpu.py
"""

import platform
import shutil
import subprocess
import sys


def nvidia_smi() -> None:
    if not shutil.which("nvidia-smi"):
        print("nvidia-smi: not found (NVIDIA driver missing, or no NVIDIA GPU)")
        return
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    for line in out.splitlines():
        name, mem, driver = (part.strip() for part in line.split(","))
        print(f"GPU: {name} | VRAM: {mem} | driver {driver}")


def torch_check() -> None:
    try:
        import torch
    except ImportError:
        print("PyTorch: not installed (add it with: uv sync --extra torch)")
        return
    print(f"PyTorch {torch.__version__} | CUDA build {torch.version.cuda} | CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        x = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
        torch.cuda.synchronize()
        y = x @ x
        torch.cuda.synchronize()
        print(f"Matmul on {torch.cuda.get_device_name(0)}: OK ({tuple(y.shape)})")


if __name__ == "__main__":
    print(f"Python {sys.version.split()[0]} on {platform.system()} {platform.release()}")
    nvidia_smi()
    torch_check()
