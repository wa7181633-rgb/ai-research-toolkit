# Inference Module (Advanced)

## Key File

- `vllm_server.py` — High-performance vLLM configuration with:
  - FP8 KV cache
  - Prefix caching
  - Controlled memory utilization
  - Ready for tensor parallel

## Best Practices Applied

- Prefer `kv_cache_dtype="fp8"` on Hopper and newer
- Set `max_model_len` to the actual needed context (not the model maximum)
- Keep `gpu_memory_utilization` around 0.90–0.93
- Enable prefix caching for chat / agent workloads
