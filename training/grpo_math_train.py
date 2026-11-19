"""
SFT Warmup -> GRPO Math Reasoning Training Pipeline
Uses Unsloth (4-bit + rsLoRA/LoRA + Unsloth gradient checkpointing) and TRL GRPOTrainer
with multi-stage verifiable rewards (numerical equivalence, XML reasoning structure,
and length/efficiency shaping) plus JSONL experiment logging.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import GRPOConfig, GRPOTrainer, SFTTrainer
from unsloth import FastLanguageModel


SYSTEM_PROMPT = (
    "You are a rigorous mathematical reasoning engine. Solve problems step by step.\n"
    "Always structure your response strictly using XML tags:\n"
    "<thought>\n"
    "Show clear equations, intermediate quantities, and unit checks.\n"
    "</thought>\n"
    "<answer>\n"
    "Provide ONLY the final numeric value (no currency symbols or extra words).\n"
    "</answer>"
)

_NUM_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?")


def normalize_number(raw: str | None) -> float | None:
    """Parse integers, decimals, comma-formatted numbers, or simple fractions into float."""
    if not raw:
        return None
    cleaned = raw.strip().replace(",", "").replace("$", "")
    if "/" in cleaned:
        parts = cleaned.split("/", 1)
        try:
            num = float(parts[0])
            den = float(parts[1])
            return num / den if den != 0 else None
        except ValueError:
            return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def extract_answer_value(text: str) -> float | None:
    """Extract numeric answer from <answer> tag, GSM8K #### marker, or fallback tail."""
    ans_match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.DOTALL | re.IGNORECASE)
    if ans_match:
        nums = _NUM_RE.findall(ans_match.group(1))
        if nums:
            return normalize_number(nums[-1])

    gsm_match = re.search(r"####\s*([^\n]+)", text)
    if gsm_match:
        nums = _NUM_RE.findall(gsm_match.group(1))
        if nums:
            return normalize_number(nums[-1])

    nums = _NUM_RE.findall(text)
    if nums:
        return normalize_number(nums[-1])
    return None


def _expand_field(val: Any, n: int) -> list[Any]:
    if isinstance(val, list):
        return val
    return [val] * n


def make_accuracy_reward(tol: float = 1e-4):
    """Verifiable numeric accuracy reward with tolerance for floating-point/fraction equivalence."""

    def accuracy_reward(completions: list[str], answer: Any = None, solution: Any = None, **kwargs: Any) -> list[float]:
        targets = answer if answer is not None else solution
        if targets is None:
            return [0.0] * len(completions)
        target_list = _expand_field(targets, len(completions))
        scores: list[float] = []
        for comp, tgt in zip(completions, target_list):
            pred_val = extract_answer_value(comp)
            tgt_val = extract_answer_value(str(tgt))
            if pred_val is None or tgt_val is None:
                scores.append(0.0)
                continue
            if abs(pred_val - tgt_val) <= max(tol, tol * abs(tgt_val)):
                scores.append(2.0)
            else:
                scores.append(-0.5)
        return scores

    accuracy_reward.__name__ = "accuracy_reward"
    return accuracy_reward


def format_reward(completions: list[str], **kwargs: Any) -> list[float]:
    """Reward strict XML reasoning structure (<thought>...</thought><answer>...</answer>)."""
    rewards: list[float] = []
    for comp in completions:
        thought_match = re.search(r"<thought>\s*(.+?)\s*</thought>", comp, flags=re.DOTALL | re.IGNORECASE)
        answer_match = re.search(r"<answer>\s*([-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?)\s*</answer>", comp, flags=re.DOTALL | re.IGNORECASE)
        if thought_match and answer_match:
            thought_len = len(thought_match.group(1).strip())
            rewards.append(1.0 if thought_len >= 20 else 0.5)
        elif thought_match or answer_match:
            rewards.append(0.2)
        else:
            rewards.append(-0.25)
    return rewards


def reasoning_efficiency_reward(completions: list[str], max_tokens: int = 512, **kwargs: Any) -> list[float]:
    """Penalize runaway verbosity while rewarding concise step-by-step derivations."""
    rewards: list[float] = []
    for comp in completions:
        approx_tokens = max(1, len(comp.split()) * 4 // 3)
        if approx_tokens > max_tokens:
            overflow_ratio = min(1.0, (approx_tokens - max_tokens) / max_tokens)
            rewards.append(-0.5 * overflow_ratio)
        elif 30 <= approx_tokens <= int(max_tokens * 0.75):
            rewards.append(0.25)
        else:
            rewards.append(0.0)
    return rewards


def prepare_math_dataset(dataset_name: str, split: str, max_samples: int | None = None) -> Dataset:
    """Load GSM8K or custom JSON/JSONL dataset into normalized prompt + answer rows."""
    local_path = Path(dataset_name)
    if local_path.exists():
        raw_ds = load_dataset("json", data_files=str(local_path), split="train")
    else:
        raw_ds = load_dataset(dataset_name, "main", split=split) if dataset_name == "openai/gsm8k" else load_dataset(dataset_name, split=split)

    if max_samples is not None and max_samples > 0:
        raw_ds = raw_ds.select(range(min(max_samples, len(raw_ds))))

    def _to_row(item: dict[str, Any]) -> dict[str, str]:
        question = item.get("question") or item.get("instruction") or item.get("prompt") or ""
        raw_ans = item.get("answer") or item.get("solution") or item.get("output") or ""
        thought = item.get("thought") or ""
        numeric_ans = extract_answer_value(str(raw_ans))
        ans_str = f"{numeric_ans:g}" if numeric_ans is not None else str(raw_ans).strip()

        if not thought and "####" in str(raw_ans):
            thought = str(raw_ans).split("####", 1)[0].strip()

        sft_target = (
            f"<thought>\n{thought if thought else 'Break down step-by-step arithmetic carefully.'}\n</thought>\n"
            f"<answer>\n{ans_str}\n</answer>"
        )
        prompt_text = f"{SYSTEM_PROMPT}\n\n### Problem:\n{question.strip()}\n"
        return {
            "prompt": prompt_text,
            "answer": ans_str,
            "sft_text": f"{prompt_text}\n### Solution:\n{sft_target}",
        }

    return raw_ds.map(_to_row, remove_columns=raw_ds.column_names)


def log_run_metadata(output_dir: str, payload: dict[str, Any]) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    record = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), **payload}
    with (out / "experiment_metrics.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


def run_sft_warmup(
    model_name: str,
    dataset: Dataset,
    output_dir: str,
    max_seq_length: int,
    max_steps: int,
    lr: float,
) -> str:
    """Stage 1: SFT warmup using Unsloth FastLanguageModel + LoRA."""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=32,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        use_rslora=True,
        random_state=3407,
    )

    bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="sft_text",
        max_seq_length=max_seq_length,
        args=TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=10,
            max_steps=max_steps,
            learning_rate=lr,
            bf16=bf16_ok,
            fp16=not bf16_ok,
            optim="adamw_8bit",
            lr_scheduler_type="cosine",
            logging_steps=5,
            save_strategy="steps",
            save_steps=max(10, max_steps // 2),
            seed=3407,
        ),
    )
    stats = trainer.train()
    sft_ckpt_dir = str(Path(output_dir) / "sft_lora_adapter")
    model.save_pretrained(sft_ckpt_dir)
    tokenizer.save_pretrained(sft_ckpt_dir)
    log_run_metadata(output_dir, {"stage": "sft", "model_name": model_name, "metrics": stats.metrics})
    return sft_ckpt_dir


def run_grpo_stage(
    model_path: str,
    dataset: Dataset,
    output_dir: str,
    max_prompt_length: int,
    max_completion_length: int,
    num_generations: int,
    max_steps: int,
    lr: float,
) -> None:
    """Stage 2: GRPO reinforcement learning using multi-objective verifiable rewards."""
    dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    grpo_args = GRPOConfig(
        output_dir=output_dir,
        learning_rate=lr,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_generations=num_generations,
        max_prompt_length=max_prompt_length,
        max_completion_length=max_completion_length,
        max_steps=max_steps,
        logging_steps=1,
        save_steps=max(10, max_steps // 2),
        bf16=(dtype == torch.bfloat16),
        fp16=(dtype == torch.float16),
        beta=0.04,
        seed=3407,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            make_accuracy_reward(),
            format_reward,
            lambda completions, **kw: reasoning_efficiency_reward(completions, max_tokens=max_completion_length, **kw),
        ],
        args=grpo_args,
        train_dataset=dataset,
    )
    stats = trainer.train()
    final_dir = str(Path(output_dir) / "grpo_policy")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    log_run_metadata(output_dir, {"stage": "grpo", "model_path": model_path, "metrics": stats.metrics})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SFT -> GRPO Math Reasoning Pipeline")
    parser.add_argument("--stage", choices=["sft", "grpo", "both"], default="both")
    parser.add_argument("--model-name", default="unsloth/llama-3-8b-Instruct-bnb-4bit")
    parser.add_argument("--grpo-base-model", default=None, help="Override model for GRPO when --stage grpo")
    parser.add_argument("--dataset", default="openai/gsm8k")
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument("--max-seq-length", type=int, default=1536)
    parser.add_argument("--max-prompt-length", type=int, default=512)
    parser.add_argument("--max-completion-length", type=int, default=512)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--sft-steps", type=int, default=60)
    parser.add_argument("--grpo-steps", type=int, default=60)
    parser.add_argument("--sft-lr", type=float, default=2e-4)
    parser.add_argument("--grpo-lr", type=float, default=5e-6)
    parser.add_argument("--output-dir", default="artifacts/math_pipeline")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    dataset = prepare_math_dataset(args.dataset, args.split, max_samples=args.max_samples)

    grpo_source = args.grpo_base_model or args.model_name
    if args.stage in ("sft", "both"):
        grpo_source = run_sft_warmup(
            model_name=args.model_name,
            dataset=dataset,
            output_dir=args.output_dir,
            max_seq_length=args.max_seq_length,
            max_steps=args.sft_steps,
            lr=args.sft_lr,
        )

    if args.stage in ("grpo", "both"):
        run_grpo_stage(
            model_path=grpo_source,
            dataset=dataset,
            output_dir=args.output_dir,
            max_prompt_length=args.max_prompt_length,
            max_completion_length=args.max_completion_length,
            num_generations=args.num_generations,
            max_steps=args.grpo_steps,
            lr=args.grpo_lr,
        )


if __name__ == "__main__":
    main()
