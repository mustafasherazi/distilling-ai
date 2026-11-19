# distilling-ai

Code for **[mustafasherazi.com](https://mustafasherazi.com)**: *Distilling AI, one layer at a time.*

The goal: build the whole LLM stack on **one 16GB GPU**, in public, one runnable step per post.

| Folder | Series | What you build |
| --- | --- | --- |
| [`00-setup/`](00-setup/) | Kickoff | Environment check, baseline benchmark |
| [`a-llm-from-scratch/`](a-llm-from-scratch/) | Build a tiny LLM | Tokenizer → attention → GPT → training → sampling |
| [`b-finetune-distill/`](b-finetune-distill/) | Teach it new tricks | LoRA, instruction tuning, QLoRA, distillation, GRPO |
| [`c-serve-fast/`](c-serve-fast/) | Serve it fast | KV cache, benchmarking, batching, quantization, speculative decoding |
| [`d-agents/`](d-agents/) | Agents on top | Agent loop, tools, memory, RAG, evals, security |

Every post folder has a README with **what you'll build · how to run it · expected output · what I measured**.

## Hardware used for every measurement

- NVIDIA RTX 4070 Ti Super, 16GB VRAM (Ada)

Your numbers will differ on other hardware. The shape of the results (what gets faster, what runs out of memory) should hold.

## Quick start

```bash
git clone https://github.com/mustafasherazi/distilling-ai
cd distilling-ai
uv sync                      # base tools (numpy, matplotlib)
uv sync --extra torch        # + PyTorch with CUDA 12.8
uv run 00-setup/check_gpu.py
```

Requires [uv](https://docs.astral.sh/uv/) and an NVIDIA driver. Each post's README lists any extra dependencies.

**Behind a corporate proxy or antivirus that inspects HTTPS?** If `uv sync` fails with `invalid peer certificate: UnknownIssuer`, point `SSL_CERT_FILE` at the proxy's root certificate and run it again.

## License

MIT. Explanations live on the blog; sources (courses, books, papers) are credited in each post.
