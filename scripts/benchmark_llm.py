#!/usr/bin/env python3
"""Benchmark an OpenAI-compatible LLM endpoint without Django.

Runs the same style of checks as the ResaleAI benchmark:
- warmup
- single inference latency
- batch throughput
- concurrency comparison
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import statistics
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_BASE_URL = "https://vllm.leounghome.loan"

SCHEMA = {
    "type": "object",
    "properties": {
        "brand": {"type": "string"},
        "model": {"type": "string"},
        "category": {"type": "string"},
    },
    "required": ["brand", "model", "category"],
}

SYSTEM_PROMPT = (
    "Extract the brand, product model, and category from luxury resale listing "
    "text. Return compact JSON with keys: brand, model, category."
)

PROMPTS = [
    "Extract brand, model, and category from: 'Louis Vuitton Speedy 25 Monogram Canvas Handbag in excellent condition with dust bag and receipt'",
    "Extract brand, model, and category from: 'Gucci GG Marmont Small Matelasse Shoulder Bag Black Leather with gold hardware'",
    "Extract brand, model, and category from: 'Hermes Birkin 30 Togo Leather Gold with Palladium Hardware, stamp Y 2020'",
    "Extract brand, model, and category from: 'Chanel Classic Double Flap Medium Caviar Leather Black with Silver Hardware'",
    "Extract brand, model, and category from: 'Prada Re-Nylon Re-Edition 2005 Shoulder Bag Black with silver triangle logo'",
    "Extract brand, model, and category from: 'Balenciaga City Classic Metallic Edge Small Arena Leather Grey'",
    "Extract brand, model, and category from: 'Dior Lady Dior Medium Cannage Lambskin Black with gold charms ABCDIOR'",
    "Extract brand, model, and category from: 'Saint Laurent Loulou Puffer Medium Quilted Leather Black with gold YSL logo'",
    "Extract brand, model, and category from: 'Bottega Veneta Cassette Bag Intrecciato Padded Leather Thunder Grey'",
    "Extract brand, model, and category from: 'Celine Luggage Nano Drummed Calfskin Tricolor Black Taupe Burgundy'",
    "Extract brand, model, and category from: 'Fendi Baguette Medium FF Jacquard Canvas Brown with gold F clasp'",
    "Extract brand, model, and category from: 'Valentino Garavani Rockstud Spike Medium Quilted Leather Red Crossbody'",
    "Extract brand, model, and category from: 'Loewe Puzzle Small Bag Classic Calfskin Tan Brown with adjustable strap'",
    "Extract brand, model, and category from: 'Givenchy Antigona Medium Goatskin Black with silver hardware and lock'",
    "Extract brand, model, and category from: 'Miu Miu Matelasse Nappa Leather Shoulder Bag Orchid Pink with crystal buckle'",
    "Extract brand, model, and category from: 'Burberry TB Bag Small Quilted Monogram Lambskin Beige with Thomas Burberry clasp'",
    "Extract brand, model, and category from: 'Alexander McQueen Four Ring Clutch Studded Leather Black with skull ring detail'",
    "Extract brand, model, and category from: 'Versace La Medusa Medium Handbag Calfskin Black with gold Medusa head plaque'",
    "Extract brand, model, and category from: 'Jacquemus Le Chiquito Long Top Handle Bag Smooth Leather Light Brown mini size'",
    "Extract brand, model, and category from: 'Goyard Saint Louis PM Tote Goyardine Canvas Black Tan with pouch and dog hook'",
]


@dataclass
class CompletionResult:
    ok: bool
    latency_s: float
    content: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    error: str = ""


def load_dotenv(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def repeat_prompts(count: int) -> list[str]:
    return (PROMPTS * ((count // len(PROMPTS)) + 1))[:count]


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]

    ordered = sorted(values)
    index = (len(ordered) - 1) * pct
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def parse_json_object(content: str) -> dict[str, Any] | None:
    content = content.strip()
    try:
        parsed = json.loads(content)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass

    start = content.find("{")
    end = content.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        parsed = json.loads(content[start : end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


class LLMClient:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        max_concurrent: int,
        max_tokens: int,
        temperature: float,
        timeout_s: float,
        structured: bool,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.max_concurrent = max_concurrent
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout_s = timeout_s
        self.structured = structured

    def infer(self, prompt: str) -> CompletionResult:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stream": False,
        }
        if self.structured:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "product_extraction",
                    "schema": SCHEMA,
                    "strict": True,
                },
            }

        request = urllib.request.Request(
            self.base_url + "/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "vllm-llm-benchmark/1.0",
            },
        )

        started = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                body = response.read()
        except urllib.error.HTTPError as exc:
            latency = time.monotonic() - started
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            return CompletionResult(False, latency, error=f"HTTP {exc.code}: {detail}")
        except Exception as exc:  # noqa: BLE001 - benchmark should capture all request failures.
            latency = time.monotonic() - started
            return CompletionResult(False, latency, error=f"{type(exc).__name__}: {exc}")

        latency = time.monotonic() - started
        try:
            parsed: dict[str, Any] = json.loads(body)
            content = parsed["choices"][0]["message"]["content"]
        except Exception as exc:  # noqa: BLE001
            return CompletionResult(False, latency, error=f"Invalid response: {exc}")

        usage = parsed.get("usage") or {}
        return CompletionResult(
            ok=True,
            latency_s=latency,
            content=content,
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            total_tokens=int(usage.get("total_tokens") or 0),
        )

    def batch_infer(self, prompts: list[str]) -> list[CompletionResult]:
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_concurrent) as executor:
            futures = [executor.submit(self.infer, prompt) for prompt in prompts]
            return [future.result() for future in futures]


def summarize(results: list[CompletionResult], elapsed_s: float) -> dict[str, float | int]:
    successes = [result for result in results if result.ok]
    failures = [result for result in results if not result.ok]
    latencies = [result.latency_s for result in successes]
    completion_tokens = sum(result.completion_tokens for result in successes)
    total_tokens = sum(result.total_tokens for result in successes)

    return {
        "count": len(results),
        "success": len(successes),
        "failed": len(failures),
        "elapsed_s": elapsed_s,
        "req_per_s": len(successes) / elapsed_s if elapsed_s else 0.0,
        "completion_tok_per_s": completion_tokens / elapsed_s if elapsed_s else 0.0,
        "total_tok_per_s": total_tokens / elapsed_s if elapsed_s else 0.0,
        "lat_avg_s": statistics.mean(latencies) if latencies else 0.0,
        "lat_p50_s": percentile(latencies, 0.50),
        "lat_p95_s": percentile(latencies, 0.95),
        "lat_p99_s": percentile(latencies, 0.99),
    }


def print_summary(summary: dict[str, float | int]) -> None:
    print(f"Total time:       {summary['elapsed_s']:.2f}s")
    print(f"Throughput:       {summary['req_per_s']:.2f} req/s")
    print(f"Completion TPS:   {summary['completion_tok_per_s']:.2f} tok/s")
    print(f"Total TPS:        {summary['total_tok_per_s']:.2f} tok/s")
    print(f"Latency avg/p50:  {summary['lat_avg_s']:.2f}s / {summary['lat_p50_s']:.2f}s")
    print(f"Latency p95/p99:  {summary['lat_p95_s']:.2f}s / {summary['lat_p99_s']:.2f}s")
    print(f"Success rate:     {summary['success']}/{summary['count']}")


def benchmark_single(client: LLMClient) -> None:
    start = time.monotonic()
    result = client.batch_infer([PROMPTS[0]])[0]
    elapsed = time.monotonic() - start

    print("\n--- Single Inference ---")
    print(f"Latency: {elapsed:.2f}s")
    if result.ok:
        parsed = parse_json_object(result.content)
        print(f"Result:  {parsed if parsed is not None else result.content[:200]}")
    else:
        print(f"Error:   {result.error}")


def benchmark_batch(client: LLMClient, batch_size: int) -> None:
    prompts = repeat_prompts(batch_size)
    start = time.monotonic()
    results = client.batch_infer(prompts)
    elapsed = time.monotonic() - start

    print(f"\n--- Batch of {batch_size} (max_concurrent={client.max_concurrent}) ---")
    print_summary(summarize(results, elapsed))
    first_error = next((result.error for result in results if not result.ok), "")
    if first_error:
        print(f"First error:      {first_error}")


def benchmark_concurrency(client: LLMClient, batch_size: int, levels: list[int]) -> None:
    print(f"\n--- Concurrency Comparison (batch={batch_size}) ---")
    prompts = repeat_prompts(batch_size)

    for concurrency in levels:
        client.max_concurrent = concurrency
        start = time.monotonic()
        results = client.batch_infer(prompts)
        elapsed = time.monotonic() - start
        summary = summarize(results, elapsed)
        print(
            f"  concurrent={concurrency:>3}: "
            f"{summary['elapsed_s']:.2f}s  "
            f"{summary['req_per_s']:.2f} req/s  "
            f"{summary['completion_tok_per_s']:.1f} out tok/s  "
            f"{summary['success']}/{summary['count']} ok  "
            f"p95={summary['lat_p95_s']:.2f}s"
        )


def parse_int_list(value: str) -> list[int]:
    try:
        parsed = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a comma-separated integer list") from exc
    if not parsed or any(item < 1 for item in parsed):
        raise argparse.ArgumentTypeError("all values must be >= 1")
    return parsed


def build_client(args: argparse.Namespace) -> LLMClient | None:
    dotenv = load_dotenv(Path(args.env_file))
    api_key = args.api_key or os.getenv("VLLM_API_KEY") or dotenv.get("VLLM_API_KEY", "")
    model = args.model or os.getenv("VLLM_MODEL") or dotenv.get("VLLM_MODEL", "")

    if not api_key:
        print("VLLM_API_KEY not configured. Set it in .env or pass --api-key.")
        return None
    if not model:
        print("VLLM_MODEL not configured. Set it in .env or pass --model.")
        return None

    return LLMClient(
        base_url=args.url,
        api_key=api_key,
        model=model,
        max_concurrent=args.max_concurrent,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        timeout_s=args.timeout,
        structured=args.structured,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark an OpenAI-compatible LLM server")
    parser.add_argument("--url", default=DEFAULT_BASE_URL, help="Base URL, e.g. https://vllm.example.com")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--model", default="")
    parser.add_argument("--max-concurrent", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--batch-sizes", type=parse_int_list, default=[1, 5, 10, 20])
    parser.add_argument("--concurrency-levels", type=parse_int_list, default=[50,100,150,200])
    parser.add_argument("--concurrency-batch-size", type=int, default=800)
    parser.add_argument(
        "--structured",
        action="store_true",
        help="Send OpenAI response_format=json_schema. Leave off if the backend rejects it.",
    )
    parser.add_argument("--skip-warmup", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    client = build_client(args)
    if client is None:
        return 1

    print(f"Target: {client.base_url}")
    print(f"Model:  {client.model}")
    print(f"Structured response_format: {client.structured}")

    if not args.skip_warmup:
        print("\nWarming up...")
        warmup = client.batch_infer(["Say hello in JSON: {\"brand\":\"test\",\"model\":\"hello\",\"category\":\"test\"}."])
        if warmup and not warmup[0].ok:
            print(f"Warmup failed: {warmup[0].error}")

    benchmark_single(client)

    for batch_size in args.batch_sizes:
        benchmark_batch(client, batch_size)

    benchmark_concurrency(client, args.concurrency_batch_size, args.concurrency_levels)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
