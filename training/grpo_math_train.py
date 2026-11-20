"""
Advanced GRPO (Group Relative Policy Optimization) Math Reasoning Trainer
Follows workflow: Data -> Train -> Evaluate -> Serve -> Analyze
"""

import argparse
import os
import re
from typing import Dict, List, Optional

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer


SYSTEM_PROMPT = (
    "You are a precise mathematical reasoning assistant. "
    "Solve step by step inside <reasoning>...</reasoning> tags, "
    "then provide the final numeric answer inside <answer>...</answer> tags."
)


def extract_xml_answer(text: str) -> str:
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    fallback = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return fallback[-1] if fallback else ""


def normalize_number(val: str) -> str:
    cleaned = val.strip().replace(",", "").rstrip(".")
    try:
        num = float(cleaned)
        if num.is_integer():
            return str(int(num))
        return f"{num:.6f}".rstrip("0").rstrip(".")
    except ValueError:
        return cleaned


def format_reward_func(completions: List[str], **kwargs) -> List[float]:
    """Rewards clean <reasoning> and <answer> XML structure."""
    scores: List[float] = []
    for text in completions:
        has_reasoning = bool(re.search(r"<reasoning>.*?</reasoning>", text, flags=re.DOTALL | re.IGNORECASE))
        has_answer = bool(re.search(r"<answer>.*?</answer>", text, flags=re.DOTALL | re.IGNORECASE))
        score = 0.0
        if has_reasoning:
            score += 0.25
        if has_answer:
            score += 0.25
        scores.append(score)
    return scores


def correctness_reward_func(prompts: List[str], completions: List[str], answer: List[str], **kwargs) -> List[float]:
    """Rewards exact normalized numeric match against ground truth."""
    rewards: List[float] = []
    for completion, gt in zip(completions, answer):
        pred_raw = extract_xml_answer(completion)
        gt_clean = gt.split("####")[-1].strip() if "####" in gt else gt.strip()
        pred_norm = normalize_number(pred_raw)
        gt_norm = normalize_number(gt_clean)
        rewards.append(2.0 if (pred_norm and pred_norm == gt_norm) else 0.0)
    return rewards


def build_math_dataset(dataset_name: str, split: str = "train", max_samples: Optional[int] = None) -> Dataset:
    if os.path.isfile(dataset_name) and dataset_name.endswith(".jsonl"):
        ds = load_dataset("json", data_files=dataset_name, split="train")
    else:
        ds = load_dataset(dataset_name, "main", split=split)

    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))

    def _map_row(row: Dict) -> Dict:
        question = row.get("question") or row.get("prompt") or ""
        gt_answer = row.get("answer") or row.get("ground_truth") or ""
        prompt_text = f"{SYSTEM_PROMPT}\n\nQuestion: {question}\n"
        return {
            "prompt": prompt_text,
            "answer": str(gt_answer),
        }

    return ds.map(_map_row, remove_columns=[c for c in ds.column_names if c not in {"prompt", "answer"}])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GRPO math training with LoRA/DoRA support.")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--dataset", type=str, default="gsm8k")
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument("--output-dir", type=str, default="outputs\\grpo-math-run")
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--max-completion-length", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--use-dora", action="store_true", help="Enable Weight-Decomposed LoRA (DoRA)")
    parser.add_argument("--report-to", type=str, default="none", choices=["none", "wandb"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    cuda_available = torch.cuda.is_available()
    bf16_supported = cuda_available and torch.cuda.is_bf16_supported()

    print(f"CUDA available: {cuda_available} | bf16 supported: {bf16_supported}")
    if not cuda_available:
        print("Warning: Running without CUDA GPU. GRPO RL rollouts are much faster on an NVIDIA CUDA GPU.")

    train_dataset = build_math_dataset(args.dataset, split="train", max_samples=args.max_samples)

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_dora=args.use_dora,
    )

    training_args = GRPOConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        bf16=bf16_supported,
        fp16=cuda_available and not bf16_supported,
        logging_steps=10,
        save_steps=100,
        report_to=args.report_to,
    )

    trainer = GRPOTrainer(
        model=args.model_name,
        reward_funcs=[format_reward_func, correctness_reward_func],
        args=training_args,
        train_dataset=train_dataset,
        peft_config=peft_config,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"Training completed. Model checkpoint saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
