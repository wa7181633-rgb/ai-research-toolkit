# Evaluation Module

`eval_gsm8k.py` provides a solid foundation for math reasoning evaluation:

- Robust answer extraction (`<answer>` tags, `####`, last number)
- Exact + soft numeric matching
- Easy to extend to larger benchmarks

### Next upgrades (recommended)

- Full GSM8K / MATH dataset loading
- Pass@k support
- Format compliance rate reporting
- Comparison table: base model vs GRPO model
