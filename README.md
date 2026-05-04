# vLLM Inference Server

Standalone vLLM server with Cloudflare Tunnel for ResaleAI LLM inference.

## Quick Start

1. Copy `.env.example` to `.env` and set `VLLM_API_KEY`:
   ```bash
   cp .env.example .env
   # Edit .env — set VLLM_API_KEY to a strong random string
   ```

2. Start the server:
   ```bash
   docker compose up -d
   ```

3. Watch cloudflared logs for the tunnel URL:
   ```bash
   docker compose logs -f cloudflared
   ```
   Look for a line like: `Your quick Tunnel has been created! Visit it at https://xxx-yyy.trycloudflare.com`

4. Set the tunnel URL as `VLLM_BASE_URL` in your ResaleAI `.env`.

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_IMAGE` | `vllm/vllm-openai:latest` | Docker image to run. Use `latest` with the RTX 3090 for new models, or `ghcr.io/sasha0552/vllm:latest` with the Tesla P40 legacy path. |
| `VLLM_MODEL` | `Qwen/Qwen3-4B` | HuggingFace model to load |
| `VLLM_DTYPE` | `auto` | vLLM dtype. Use `auto` on newer GPUs and `half` on the Tesla P40. |
| `VLLM_API_KEY` | _(empty)_ | API key — all requests must include `Authorization: Bearer <key>` |
| `HF_TOKEN` | _(empty)_ | HuggingFace token for gated models |
| `VLLM_GPU_DEVICES` | `0` | GPU device ID to expose to vLLM. On this host, `0` is the RTX 3090 and `1` is the Tesla P40. |
| `CLOUDFLARE_TUNNEL_TOKEN` | _(empty)_ | For named tunnels (persistent URL) |

## GPU Modes

For the RTX 3090 with the newest vLLM and newer models:

```bash
VLLM_IMAGE=vllm/vllm-openai:latest
VLLM_GPU_DEVICES=0
VLLM_MODEL=Qwen/Qwen3-4B
VLLM_DTYPE=auto
```

For the Tesla P40, use the Pascal-compatible image and avoid FP8/new architectures that require newer compute capabilities:

```bash
VLLM_IMAGE=ghcr.io/sasha0552/vllm:latest
VLLM_GPU_DEVICES=1
VLLM_MODEL=Qwen/Qwen2.5-7B-Instruct
VLLM_DTYPE=half
```

## Quick Tunnel vs Named Tunnel

- **Quick tunnel** (default): Auto-generates a `*.trycloudflare.com` URL on each start. Good for development.
- **Named tunnel**: Set `CLOUDFLARE_TUNNEL_TOKEN` and change cloudflared command to `tunnel run` for a persistent subdomain.

## Benchmarking

Run the Django-free LLM benchmark against the Cloudflare endpoint:

```bash
python3 scripts/benchmark_llm.py \
  --url https://vllm.leounghome.loan \
  --batch-sizes 5,10,20 \
  --concurrency-levels 10,20,40,80 \
  --concurrency-batch-size 80
```

For a lower-level raw concurrency benchmark:

```bash
python3 scripts/benchmark_concurrency.py \
  --base-url https://vllm.leounghome.loan \
  --requests 32 \
  --concurrency 4 \
  --max-tokens 128
```

## Requirements

- NVIDIA GPU with Docker GPU support (`nvidia-container-toolkit`)
- ~4GB VRAM for the 4B model
- Docker Compose v2+
