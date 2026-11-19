"""
High-Throughput vLLM Serving & Concurrent Throughput Benchmark.
Supports launching an OpenAI-compatible vLLM server or running a concurrent
batch benchmark measuring latency, total tokens/sec, and p50/p95 timings.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from statistics import quantiles
from typing import Any

from vllm import LLM, SamplingParams


BENCHMARK_PROMPTS = [
    "Derive the key mathematical difference between PPO critic value estimation and GRPO group-relative advantage normalization.",
    "Explain how PagedAttention eliminates KV-cache fragmentation during high-concurrency LLM serving.",
    "Compare LoRA, rsLoRA, and DoRA weight updates on linear projection layers in transformers.",
    "How does speculative decoding accelerate autoregressive token generation without changing output distribution?",
    "Design a two-stage SFT -> GRPO training curriculum for multi-step arithmetic reasoning.",
    "What are attention sinks in streaming LLMs, and how do they stabilize long-context perplexity?",
    "Explain why 4-bit NF4 quantization with double quantization preserves reasoning quality better than naive uniform INT4.",
    "How should reward functions be shaped to prevent reward hacking on XML-formatted math answers?",
]


def launch_openai_server(args: argparse.Namespace) -> None:
    cmd = [
        "python",
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        args.model,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--gpu-memory-utilization",
        str(args.gpu_mem_util),
        "--max-model-len",
        str(args.max_model_len),
        "--tensor-parallel-size",
        str(args.tp_size),
        "--enable-prefix-caching",
    ]
    if args.quantization:
        cmd.extend(["--quantization", args.quantization])
    if args.speculative_model:
        cmd.extend(["--speculative-model", args.speculative_model, "--num-speculative-tokens", str(args.num_speculative_tokens)])

    print("Launching vLLM OpenAI server:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    llm_kwargs: dict[str, Any] = {
        "model": args.model,
        "gpu_memory_utilization": args.gpu_mem_util,
        "max_model_len": args.max_model_len,
        "tensor_parallel_size": args.tp_size,
        "enable_prefix_caching": True,
        "trust_remote_code": True,
    }
    if args.quantization:
        llm_kwargs["quantization"] = args.quantization
    if args.speculative_model:
        llm_kwargs["speculative_model"] = args.speculative_model
        llm_kwargs["num_speculative_tokens"] = args.num_speculative_tokens

    llm = LLM(**llm_kwargs)
    params = SamplingParams(temperature=args.temperature, top_p=0.95, max_tokens=args.max_tokens)

    _ = llm.generate(BENCHMARK_PROMPTS[:2], SamplingParams(temperature=0.0, max_tokens=32))

    prompts = (BENCHMARK_PROMPTS * ((args.num_requests // len(BENCHMARK_PROMPTS)) + 1))[: args.num_requests]
    per_req_latencies: list[float] = []
    total_tokens = 0

    t_start = time.perf_counter()
    outputs = llm.generate(prompts, params)
    total_elapsed = max(1e-6, time.perf_counter() - t_start)

    for out in outputs:
        gen_len = len(out.outputs[0].token_ids)
        total_tokens += gen_len
        metrics = getattr(out, "metrics", None)
        if metrics and metrics.first_token_time and metrics.finished_time:
            per_req_latencies.append(max(0.0, metrics.finished_time - metrics.first_token_time))

    lat_stats: dict[str, float] = {}
    if per_req_latencies:
        qs = quantiles(per_req_latencies, n=20)
        lat_stats = {
            "mean_decode_s": round(sum(per_req_latencies) / len(per_req_latencies), 4),
            "p50_decode_s": round(qs[9], 4),
            "p95_decode_s": round(qs[18], 4),
        }

    return {
        "model": args.model,
        "requests": len(prompts),
        "total_output_tokens": total_tokens,
        "elapsed_seconds": round(total_elapsed, 3),
        "output_tokens_per_sec": round(total_tokens / total_elapsed, 2),
        "requests_per_sec": round(len(prompts) / total_elapsed, 2),
        "latency": lat_stats,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Advanced vLLM Serving & Benchmark Runner")
    parser.add_argument("--mode", choices=["benchmark", "serve"], default="benchmark")
    parser.add_argument("--model", default="unsloth/llama-3-8b-Instruct-bnb-4bit")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--gpu-mem-util", type=float, default=0.88)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--quantization", default=None, help="e.g. fp8, awq, gptq")
    parser.add_argument("--speculative-model", default=None)
    parser.add_argument("--num-speculative-tokens", type=int, default=5)
    parser.add_argument("--num-requests", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "serve":
        launch_openai_server(args)
    else:
        print(json.dumps(run_benchmark(args), indent=2))


if __name__ == "__main__":
    main()
