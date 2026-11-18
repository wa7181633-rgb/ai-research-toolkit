# Training Module (Advanced)

## Main Entry Points

| File | Purpose |
|------|--------|
| `grpo_math_train.py` | Full GRPO pipeline for math reasoning |
| `rewards.py` | Reusable strong reward functions |
| `train.py` | Basic SFT starter (kept for reference) |

## Recommended Workflow

1. Generate or load a high-quality dataset (`../pipeline/synthetic_data.py` or GSM8K/MATH)
2. (Optional) Run a short SFT stage for format learning
3. Run GRPO with the multi-component rewards in `rewards.py`
4. Evaluate with `../evaluation/eval_gsm8k.py`

## Key Design Choices

- **Unsloth** for memory efficiency and speed
- **Multiple reward functions** instead of a single weak signal
- **Explicit `<thought>` + `<answer>` format** for easier process supervision and evaluation
- **Group size (`num_generations`)** is one of the most important hyperparameters in GRPO
