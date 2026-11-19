"""
GSM8K & Math Reasoning Evaluation Harness using vLLM.
Computes exact numeric accuracy, XML formatting compliance, average latency,
and saves detailed per-example audit logs + summary JSON.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

from datasets import load_dataset
from vllm import LLM, SamplingParams


_NUM_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?")


def parse_numeric(raw: str | None) -> float | None:
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


def extract_prediction(text: str) -> tuple[float | None, bool]:
    """Return (parsed_number, strict_xml_valid)."""
    xml_match = re.search(
        r"<thought>\s*(.+?)\s*</thought>\s*<answer>\s*([-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?)\s*</answer>",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if xml_match:
        return parse_numeric(xml_match.group(2)), True

    tag_only = re.search(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.DOTALL | re.IGNORECASE)
    if tag_only:
        nums = _NUM_RE.findall(tag_only.group(1))
        if nums:
            return parse_numeric(nums[-1]), False

    nums = _NUM_RE.findall(text)
    if nums:
        return parse_numeric(nums[-1]), False
    return None, False


def extract_ground_truth(raw_answer: str) -> float | None:
    if "####" in raw_answer:
        tail = raw_answer.split("####", 1)[1]
        nums = _NUM_RE.findall(tail)
        if nums:
            return parse_numeric(nums[-1])
    nums = _NUM_RE.findall(raw_answer)
    return parse_numeric(nums[-1]) if nums else None


def build_prompt(question: str) -> str:
    return (
        "Solve the math problem step by step inside <thought>...</thought>, "
        "then output ONLY the final number inside <answer>...</answer>.\n\n"
        f"Question: {question.strip()}\n"
    )


def evaluate(
    model_path: str,
    dataset_name: str,
    split: str,
    max_samples: int,
    max_tokens: int,
    temperature: float,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    report_path: str,
) -> dict[str, Any]:
    local_path = Path(dataset_name)
    if local_path.exists():
        ds = load_dataset("json", data_files=str(local_path), split="train")
    elif dataset_name == "openai/gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split=split)
    else:
        ds = load_dataset(dataset_name, split=split)

    if max_samples > 0:
        ds = ds.select(range(min(max_samples, len(ds))))

    prompts = [build_prompt(row.get("question") or row.get("instruction") or "") for row in ds]
    ground_truths = [extract_ground_truth(str(row.get("answer") or row.get("solution") or "")) for row in ds]

    llm = LLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=True,
    )
    sampling_params = SamplingParams(temperature=temperature, top_p=0.95 if temperature > 0 else 1.0, max_tokens=max_tokens)

    t0 = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params)
    elapsed_s = max(1e-6, time.perf_counter() - t0)

    correct = 0
    xml_valid_count = 0
    total_completion_tokens = 0
    records: list[dict[str, Any]] = []

    for idx, (out, gt) in enumerate(zip(outputs, ground_truths)):
        gen_text = out.outputs[0].text
        comp_tokens = len(out.outputs[0].token_ids)
        total_completion_tokens += comp_tokens
        pred, xml_ok = extract_prediction(gen_text)
        if xml_ok:
            xml_valid_count += 1

        is_correct = False
        if pred is not None and gt is not None:
            is_correct = abs(pred - gt) <= max(1e-4, 1e-4 * abs(gt))
        if is_correct:
            correct += 1

        records.append(
            {
                "index": idx,
                "ground_truth": gt,
                "prediction": pred,
                "correct": is_correct,
                "xml_valid": xml_ok,
                "completion_tokens": comp_tokens,
                "response": gen_text,
            }
        )

    total = len(records)
    summary = {
        "model": model_path,
        "dataset": dataset_name,
        "split": split,
        "samples": total,
        "accuracy": round(correct / max(1, total), 4),
        "xml_format_compliance": round(xml_valid_count / max(1, total), 4),
        "elapsed_seconds": round(elapsed_s, 3),
        "throughput_tokens_per_sec": round(total_completion_tokens / elapsed_s, 2),
    }

    out_file = Path(report_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps({"summary": summary, "records": records}, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GSM8K Evaluation Harness via vLLM")
    parser.add_argument("--model", default="unsloth/llama-3-8b-Instruct-bnb-4bit")
    parser.add_argument("--dataset", default="openai/gsm8k")
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--gpu-mem-util", type=float, default=0.85)
    parser.add_argument("--report", default="benchmarks/gsm8k_eval_report.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = evaluate(
        model_path=args.model,
        dataset_name=args.dataset,
        split=args.split,
        max_samples=args.max_samples,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        tensor_parallel_size=args.tp_size,
        gpu_memory_utilization=args.gpu_mem_util,
        report_path=args.report,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
