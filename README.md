# Advanced AI Research & Engineering Toolkit

Professional-grade toolkit for AI researchers and engineers.  
Focused on **modern post-training**, **high-throughput inference**, and **mechanistic interpretability**.

Inspired by and complementary to the excellent [ai-research-skills](https://github.com/Orchestra-Research/AI-Research-SKILLs) library.

---

## What's Inside

| Module | Path | Focus |
|--------|------|-------|
| **Training & Alignment** | [`/training`](./training) | GRPO, Unsloth, LoRA/DoRA, memory-efficient pipelines |
| **Inference & Optimization** | [`/inference`](./inference) | vLLM + PagedAttention, FlashAttention-3, FP8, speculative decoding |
| **Mechanistic Interpretability** | [`/interpretability`](./interpretability) | Transformer Lens, residual stream probing, attention analysis |
| **Prompt Templates** | [`/prompts`](./prompts) | Research ideation, experiment design, paper writing, reward modeling |
| **Benchmarks & Recipes** | [`/benchmarks`](./benchmarks) | Ready-to-run evaluation scripts and recipes |

---

## Quick Start

```bash
git clone https://github.com/wa7181633-rgb/ai-research-toolkit.git
cd ai-research-toolkit
```

Recommended environment (2026 stack):

```bash
pip install unsloth vllm transformers datasets accelerate peft trl
pip install transformer-lens  # for interpretability
```

---

## Core Philosophy

1. **Practical over theoretical** — every folder has runnable examples or clear configs.
2. **Memory-efficient first** — Unsloth + GRPO + FP8 so you can train and serve on consumer / mid-range GPUs.
3. **Agent-ready** — prompts and structures designed so coding agents (Claude Code, Cursor, etc.) can use this toolkit autonomously.
4. **Up-to-date** — tracks 2025–2026 techniques: GRPO / Dr.GRPO / DAPO, FlashAttention-3, PagedAttention best practices, FP8 KV cache.

---

## Recommended Learning Path

1. Start with [`prompts/`](./prompts) → get good research & experiment prompts.
2. Go to [`training/grpo/`](./training/grpo) → train your first reasoning model with Unsloth + GRPO.
3. Move to [`inference/`](./inference) → serve it efficiently with vLLM.
4. Use [`interpretability/`](./interpretability) to understand *why* the model behaves a certain way.

---

## Maintainer

**Farmoz** ([@wa7181633-rgb](https://github.com/wa7181633-rgb))  
Websites & AI automations for small businesses.

---

## License

MIT — free to use, modify, and build upon.
