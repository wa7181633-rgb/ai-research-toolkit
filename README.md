# Advanced AI Research & Engineering Toolkit

Production-structured research scripts for **SFT -> GRPO reasoning alignment**, **verifiable math rewards**, **GSM8K evaluation**, **vLLM serving/benchmarking**, **AST-verified synthetic data generation**, and **Transformer Lens residual probing**, aligned with the `ai-research-skills` framework.

## Repository Layout

- `training/grpo_math_train.py`: Two-stage `SFT -> GRPO` pipeline (`--stage sft|grpo|both`) using Unsloth (`4-bit` + `rsLoRA` + Unsloth gradient checkpointing) and TRL `GRPOTrainer` with numeric equivalence accuracy, XML `<thought>`/`<answer>` formatting checks, reasoning token length shaping, and JSONL experiment logging.
- `evaluation/eval_gsm8k.py`: vLLM-backed evaluation runner computing numeric accuracy, XML compliance rate, output token throughput, and saving full JSON reports.
- `inference/vllm_server.py`: Supports both OpenAI-compatible serving (`--mode serve`, prefix caching, quantization, speculative decoding) and concurrent throughput/latency benchmarking (`--mode benchmark`).
- `pipeline/synthetic_data.py`: Generates deduplicated multi-step word problems verified via safe AST evaluation into `.jsonl` datasets.
- `interpretability/residual_probe.py`: Transformer Lens residual stream probe (`hook_resid_post` + `apply_ln_to_stack` + unembedding) tracking candidate token probabilities across layers.
- `requirements.txt`: Dependencies for training, GRPO/SFT, vLLM, and interpretability.

## Quick Execution Flow

```bash
# 1. Generate verified synthetic training data
python pipeline/synthetic_data.py --count 250 --output pipeline/synthetic_math_train.jsonl

# 2. Run SFT warmup followed by GRPO alignment
python training/grpo_math_train.py --stage both --dataset openai/gsm8k --max-samples 300 --sft-steps 40 --grpo-steps 40

# 3. Evaluate on GSM8K via vLLM
python evaluation/eval_gsm8k.py --model artifacts/math_pipeline/grpo_policy --max-samples 100

# 4. Benchmark or serve via vLLM
python inference/vllm_server.py --mode benchmark --model unsloth/llama-3-8b-Instruct-bnb-4bit --num-requests 16
```

---
*Maintained by Farmoz (`wa7181633-rgb`)*
