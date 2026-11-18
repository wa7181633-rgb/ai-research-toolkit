# FlashAttention-3 & Attention Backends

FlashAttention-3 is the current high-performance attention kernel for Hopper (H100) and newer GPUs.

### When it matters

- Training throughput
- Prefill speed in serving
- Long context performance

Most modern stacks (Unsloth, vLLM, PyTorch 2.x+) will automatically select the best available kernel.  
You usually only need to force it when debugging or when using older containers.

### Tips

- Prefer official Docker images from Unsloth / vLLM that ship recent FA3 builds
- On Blackwell, check for FA4 / FlashInfer paths
- FP8 KV cache + FA3 is a very strong combination on H100
