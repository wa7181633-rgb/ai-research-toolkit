# Unsloth Recipes

Unsloth is currently one of the best ways to do memory-efficient fine-tuning and GRPO on limited VRAM.

### Why Unsloth?

- 2–5× faster training
- Significantly lower VRAM usage
- Excellent integration with TRL (including GRPO)
- Supports 4-bit, LoRA, and fast inference backends

### Recommended Patterns

1. **SFT first** → then GRPO (especially important for format following)
2. Use `use_gradient_checkpointing = "unsloth"`
3. Prefer `r = 16–64` for most reasoning tasks
4. Enable `fast_inference=True` when doing GRPO generation

### Useful Links

- Official Unsloth docs: https://docs.unsloth.ai
- GRPO notebooks: search for “Unsloth GRPO” on Colab
