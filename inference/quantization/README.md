# Quantization Strategies

### Weights
- 4-bit (bitsandbytes / AWQ / GPTQ) → great for fine-tuning & local inference
- FP8 → excellent quality/speed trade-off on Hopper+

### KV Cache
- **FP8 KV** is one of the highest-ROI optimizations in 2026
- Roughly doubles concurrent capacity vs BF16 KV with minimal quality drop

### Practical Order of Preference (serving)

1. FP8 weights + FP8 KV (when supported)
2. BF16 weights + FP8 KV
3. 4-bit weights + FP8/BF16 KV (for memory-constrained machines)
