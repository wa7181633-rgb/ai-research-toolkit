# Advanced AI Research & Engineering Toolkit

**Research-grade** toolkit for modern LLM post-training, high-throughput inference, and mechanistic interpretability.

Focused on practical, high-performance workflows used in 2025–2026 research:
- Group Relative Policy Optimization (GRPO)
- Memory-efficient training with Unsloth
- High-throughput serving with vLLM + FP8 + PagedAttention
- Strong evaluation & synthetic data pipelines

---

## Core Modules

| Module | Path | Level |
|--------|------|-------|
| **GRPO Training** | [`training/`](./training) | Full pipeline with strong rewards |
| **Evaluation** | [`evaluation/`](./evaluation) | Real GSM8K-style scoring |
| **Inference** | [`inference/`](./inference) | Production-oriented vLLM configs |
| **Synthetic Data** | [`pipeline/`](./pipeline) | Reasoning data generation |
| **Interpretability** | [`interpretability/`](./interpretability) | Transformer Lens recipes |
| **Prompts** | [`prompts/`](./prompts) | Research-grade templates |

---

## Recommended Stack (2026)

```bash
pip install unsloth trl peft transformers datasets accelerate vllm
pip install transformer-lens  # optional for interpretability
```

---

## Quick Start (Advanced Path)

1. **Train a reasoning model**
   ```bash
   python training/grpo_math_train.py
   ```

2. **Evaluate**
   ```bash
   python evaluation/eval_gsm8k.py
   ```

3. **Serve efficiently**
   ```bash
   python inference/vllm_server.py
   ```

---

## Design Principles

- **No toy code** — every script is meant to be extended into real experiments
- **Memory-efficient first** — Unsloth + 4-bit + GRPO
- **Verifiable rewards** preferred over learned reward models
- **Agent-ready** structure for Claude / Cursor / coding agents

---

## Maintainer

**Farmoz** ([@wa7181633-rgb](https://github.com/wa7181633-rgb))

MIT License
