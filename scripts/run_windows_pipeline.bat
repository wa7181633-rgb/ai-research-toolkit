@echo off
REM Windows helper script for ai-research-toolkit workflow:
REM Data -> Train -> Evaluate
REM Note: Use Python 3.10 or 3.11 virtual environment for vLLM / FlashAttention / Unsloth.

if not exist "outputs" mkdir outputs

echo [1/3] Generating synthetic math JSONL dataset...
python pipeline\synthetic_data.py --num-samples 50 --output outputs\synthetic_math.jsonl

echo [2/3] Running GRPO training (with LoRA/DoRA)...
python training\grpo_math_train.py --dataset outputs\synthetic_math.jsonl --max-samples 50 --output-dir outputs\grpo-math-run

echo [3/3] Evaluating checkpoint on GSM8K / JSONL...
python evaluation\eval_gsm8k.py --model outputs\grpo-math-run --dataset outputs\synthetic_math.jsonl --max-samples 50 --save-results outputs\eval_results.jsonl

echo Done! Check outputs\ folder for checkpoints and JSONL logs.
