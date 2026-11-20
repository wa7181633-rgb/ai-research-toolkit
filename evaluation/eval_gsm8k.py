"""
GSM8K & Custom JSONL Benchmark Evaluator
Supports HuggingFace model IDs, local checkpoints, or PEFT adapter directories.
"""

import argparse
import json
import os
import re
from typing import Dict, List, Optional, Tuple

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def extract_ground_truth(answer_field: str) -> str:
    if "####" in answer_field:
        return answer_field.split("####")[-1].strip()
    return answer_field.strip()


def extract_prediction(response: str) -> str:
    xml_match = re.search(r"<answer>\s*(.*?)\s*</answer>", response, flags=re.DOTALL | re.IGNORECASE)
    if xml_match:
        return xml_match.group(1).strip()

    hash_match = re.search(r"####\s*([^\n]+)", response)
    if hash_match:
        return hash_match.group(1).strip()

    numbers = re.findall(r"-?\d+(?:\.\d+)?", response.replace(",", ""))
    return numbers[-1] if numbers else ""


def normalize_val(val: str) -> str:
    cleaned = val.strip().replace(",", "").rstrip(".")
    try:
        num = float(cleaned)
        if num.is_integer():
            return str(int(num))
        return f"{num:.6f}".rstrip("0").rstrip(".")
    except ValueError:
        return cleaned


def load_model_and_tokenizer(model_path: str, base_model: Optional[str] = None) -> Tuple:
    tokenizer = AutoTokenizer.from_pretrained(base_model or model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
    device_map = "auto" if torch.cuda.is_available() else None

    adapter_cfg_path = os.path.join(model_path, "adapter_config.json")
    if os.path.isdir(model_path) and os.path.isfile(adapter_cfg_path):
        with open(adapter_cfg_path, "r", encoding="utf-8") as f:
            adapter_cfg = json.load(f)
        resolved_base = base_model or adapter_cfg.get("base_model_name_or_path")
        if not resolved_base:
            raise ValueError("Adapter checkpoint detected, but no --base-model could be resolved.")
        base = AutoModelForCausalLM.from_pretrained(resolved_base, dtype=dtype, device_map=device_map, trust_remote_code=True)
        model = PeftModel.from_pretrained(base, model_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype, device_map=device_map, trust_remote_code=True)

    model.eval()
    return model, tokenizer


def load_eval_samples(dataset_path: Optional[str], split: str, max_samples: Optional[int]) -> List[Dict]:
    if dataset_path and os.path.isfile(dataset_path):
        rows: List[Dict] = []
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    else:
        ds = load_dataset("gsm8k", "main", split=split)
        rows = [dict(x) for x in ds]

    if max_samples is not None:
        rows = rows[:max_samples]
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate model or LoRA/DoRA adapter on GSM8K or JSONL.")
    parser.add_argument("--model", type=str, required=True, help="Model ID, checkpoint dir, or PEFT adapter dir")
    parser.add_argument("--base-model", type=str, default=None, help="Optional base model override for PEFT adapters")
    parser.add_argument("--dataset", type=str, default=None, help="Optional path to custom .jsonl file")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--save-results", type=str, default="outputs\\eval_gsm8k_results.jsonl")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(os.path.dirname(args.save_results) or ".", exist_ok=True)

    model, tokenizer = load_model_and_tokenizer(args.model, args.base_model)
    samples = load_eval_samples(args.dataset, args.split, args.max_samples)

    correct = 0
    total = len(samples)
    results = []

    for idx, item in enumerate(samples):
        question = item.get("question") or item.get("prompt") or ""
        gt_raw = extract_ground_truth(str(item.get("answer") or item.get("ground_truth") or ""))
        gt_norm = normalize_val(gt_raw)

        prompt = (
            "Solve step by step inside <reasoning>...</reasoning> tags, "
            "then write the final numeric answer inside <answer>...</answer> tags.\n\n"
            f"Question: {question}\n"
        )
        inputs = tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        gen_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(gen_tokens, skip_special_tokens=True)

        pred_raw = extract_prediction(response)
        pred_norm = normalize_val(pred_raw)
        is_correct = bool(pred_norm and pred_norm == gt_norm)
        if is_correct:
            correct += 1

        results.append(
            {
                "index": idx,
                "question": question,
                "ground_truth": gt_norm,
                "prediction": pred_norm,
                "correct": is_correct,
                "response": response,
            }
        )

    accuracy = (correct / total * 100.0) if total > 0 else 0.0
    print(f"Evaluated {total} samples | Correct: {correct} | Accuracy: {accuracy:.2f}%")

    with open(args.save_results, "w", encoding="utf-8") as out_f:
        for r in results:
            out_f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Detailed evaluation log saved to: {args.save_results}")


if __name__ == "__main__":
    main()
