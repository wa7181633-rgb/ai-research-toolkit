#!/bin/bash
# Example high-throughput vLLM serve command (2026 best practices)

MODEL="meta-llama/Llama-3.1-8B-Instruct"   # change as needed

vllm serve $MODEL \
  --dtype auto \
  --kv-cache-dtype fp8 \
  --gpu-memory-utilization 0.92 \
  --max-model-len 16384 \
  --max-num-seqs 256 \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --tensor-parallel-size 1 \
  --port 8000
