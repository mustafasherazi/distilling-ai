"""What fits in your GPU? Estimate how many tokens of KV cache a model leaves room for.

Reads a model's config.json from Hugging Face (no weights downloaded) and prints:
  - KV cache size per token
  - how many tokens fit in the memory left after the weights

Run:
  uv run 00-setup/kv_budget.py Qwen/Qwen2.5-7B-Instruct-AWQ --weights-gib 5.2 --gpu-gib 16 --util 0.75
  uv run 00-setup/kv_budget.py Qwen/Qwen2.5-14B-Instruct-AWQ --weights-gib 9.4 --gpu-gib 16 --util 0.75

--weights-gib is the size of the model files (roughly what vLLM reports as "model weights").
--util is vLLM's --gpu-memory-utilization (the share of VRAM vLLM is allowed to use).
The estimate ignores activation memory and CUDA graphs, so vLLM's own number will be a bit lower.
"""

import argparse
import json
import ssl
import urllib.request


def ssl_context() -> ssl.SSLContext:
    # Behind HTTPS-inspecting antivirus or a corporate proxy, the OS certificate store
    # trusts the proxy; Python's default bundle may not. truststore uses the OS store.
    try:
        import truststore

        return truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    except ImportError:
        return ssl.create_default_context()


def load_config(model: str) -> dict:
    url = f"https://huggingface.co/{model}/raw/main/config.json"
    with urllib.request.urlopen(url, context=ssl_context(), timeout=30) as r:
        return json.load(r)


def kv_bytes_per_token(cfg: dict, kv_bytes: int) -> tuple[int, dict]:
    layers = cfg["num_hidden_layers"]
    heads = cfg["num_attention_heads"]
    kv_heads = cfg.get("num_key_value_heads", heads)  # fewer KV heads = grouped-query attention
    head_dim = cfg.get("head_dim") or cfg["hidden_size"] // heads
    # For every layer we store one Key vector and one Value vector per KV head.
    per_token = layers * 2 * kv_heads * head_dim * kv_bytes
    return per_token, {"layers": layers, "attention heads": heads, "KV heads": kv_heads, "head size": head_dim}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("model", help="Hugging Face model id, e.g. Qwen/Qwen2.5-7B-Instruct-AWQ")
    ap.add_argument("--weights-gib", type=float, required=True, help="model weights size in GiB")
    ap.add_argument("--gpu-gib", type=float, default=16.0, help="total GPU memory in GiB (default 16)")
    ap.add_argument("--util", type=float, default=0.9, help="vLLM --gpu-memory-utilization (default 0.9)")
    ap.add_argument("--kv-bytes", type=int, default=2, help="bytes per KV value: 2 = FP16/BF16, 1 = FP8")
    args = ap.parse_args()

    cfg = load_config(args.model)
    per_token, shape = kv_bytes_per_token(cfg, args.kv_bytes)
    budget_gib = args.gpu_gib * args.util - args.weights_gib
    tokens = int(budget_gib * 1024**3 // per_token) if budget_gib > 0 else 0

    print(f"Model: {args.model}")
    print("Shape: " + ", ".join(f"{k} {v}" for k, v in shape.items()))
    print(f"KV cache per token: {per_token / 1024:.0f} KiB")
    print(f"Memory for KV cache: {args.gpu_gib:g} GiB x {args.util:g} - {args.weights_gib:g} GiB weights = {budget_gib:.1f} GiB")
    print(f"Tokens that fit: ~{tokens:,}")
    if tokens:
        for ctx in (2048, 8192, 32768):
            print(f"  = {tokens // ctx:,} conversations of {ctx:,} tokens at once")


if __name__ == "__main__":
    main()
