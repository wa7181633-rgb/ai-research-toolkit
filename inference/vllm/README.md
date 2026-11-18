# vLLM + PagedAttention Best Practices (2026)

PagedAttention treats the KV cache like virtual memory (fixed-size blocks).  
This dramatically reduces fragmentation and increases concurrent capacity.

### Key Settings

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --dtype auto \
  --kv-cache-dtype fp8 \
  --gpu-memory-utilization 0.92 \
  --max-model-len 32768 \
  --enable-prefix-caching \
  --block-size 16
```

### Important Knobs

| Parameter | Recommendation | Why |
|-----------|----------------|-----|
| `kv-cache-dtype` | `fp8` on H100+ | Halves KV memory with almost no quality loss |
| `gpu-memory-utilization` | 0.90 – 0.95 | Leaves headroom for activations |
| `max-model-len` | Set to *actual* needed length | Bigger max = fewer concurrent sequences |
| `enable-prefix-caching` | On for chat / agents | Free speedup on shared system prompts |
| `block-size` | 16 (default) or 32 for very long context | Trade-off bookkeeping vs waste |

### Memory Reality Check

KV cache (not FLOPs) is usually the concurrency bottleneck.  
Always calculate how many concurrent long-context requests your GPU can actually hold before launching production traffic.
