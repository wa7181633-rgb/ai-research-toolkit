# LoRA / DoRA / Rank-Stabilized Adapters

Modern parameter-efficient fine-tuning techniques.

- **LoRA** — classic low-rank adaptation
- **DoRA** — Weight-Decomposed Low-Rank Adaptation (usually better than LoRA)
- **rsLoRA / rank-stabilized** — helps with higher ranks

### Practical Advice (2026)

- Start with DoRA when available in your trainer
- Rank 16–64 is sweet spot for most 7B–14B models
- Alpha = 2 × rank is a strong default with Unsloth
- Always target attention + MLP projections
