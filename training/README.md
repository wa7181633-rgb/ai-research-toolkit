# Training & Alignment

Modern post-training techniques focused on **reasoning** and **efficiency**.

## Modules

- **[grpo/](./grpo)** — Group Relative Policy Optimization (DeepSeek R1 style)
- **[unsloth/](./unsloth)** — Memory-efficient fine-tuning recipes
- **[lora-dora/](./lora-dora)** — LoRA, DoRA, and rank-stabilized adapters

## Recommended Stack (2026)

```bash
pip install unsloth trl peft transformers datasets accelerate
```

Unsloth + TRL is currently the most practical way to run GRPO on consumer GPUs (even 1.5B–8B models on 8–16 GB VRAM).
