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
| `VLLM_MODEL` | `Qwen/Qwen3-4B` | HuggingFace model to load |
| `VLLM_API_KEY` | _(empty)_ | API key — all requests must include `Authorization: Bearer <key>` |
| `HF_TOKEN` | _(empty)_ | HuggingFace token for gated models |
| `CLOUDFLARE_TUNNEL_TOKEN` | _(empty)_ | For named tunnels (persistent URL) |

## Quick Tunnel vs Named Tunnel

- **Quick tunnel** (default): Auto-generates a `*.trycloudflare.com` URL on each start. Good for development.
- **Named tunnel**: Set `CLOUDFLARE_TUNNEL_TOKEN` and change cloudflared command to `tunnel run` for a persistent subdomain.

## Requirements

- NVIDIA GPU with Docker GPU support (`nvidia-container-toolkit`)
- ~4GB VRAM for the 4B model
- Docker Compose v2+
