"""
Advanced GSM8K-style Evaluation
===============================
- Proper answer extraction
- Exact match + soft match scoring
- Support for format-aware evaluation
- Easy to extend to MATH, AIME, etc.
"""

import re
import json
from typing import List, Dict
from vllm import LLM, SamplingParams

# ======================
# Configuration
# ======================

MODEL_PATH = "unsloth/Qwen2.5-1.5B-Instruct"   # Change to your fine-tuned model
TENSOR_PARALLEL_SIZE = 1
MAX_TOKENS = 1024
TEMPERATURE = 0.0          # Greedy for evaluation

# ======================
# Answer Extraction
# ======================

def extract_answer(text: str) -> str:
    """Robust answer extraction for math problems."""
    # 1. Prefer explicit <answer> tags
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL | re.IGNORECASE)
    if match:
        return clean_number(match.group(1))

    # 2. Look for "####" style (GSM8K original)
    if "####" in text:
        return clean_number(text.split("####")[-1])

    # 3. Last number in the response
    numbers = re.findall(r"[-+]?(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d+)?", text)
    if numbers:
        return clean_number(numbers[-1])

    return text.strip()[-30:]


def clean_number(s: str) -> str:
    s = s.strip().replace(",", "").replace("$", "").replace("%", "")
    # Remove trailing punctuation
    s = re.sub(r"[^0-9.\-]+", "", s)
    return s


def is_correct(prediction: str, ground_truth: str) -> bool:
    pred = clean_number(prediction)
    gold = clean_number(str(ground_truth))
    if pred == gold:
        return True
    try:
        return abs(float(pred) - float(gold)) < 1e-6
    except Exception:
        return False

# ======================
# Evaluation Logic
# ======================

def evaluate(
    questions: List[str],
    ground_truths: List[str],
    model_path: str = MODEL_PATH,
) -> Dict:
    llm = LLM(
        model=model_path,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        trust_remote_code=True,
        max_model_len=4096,
    )

    sampling_params = SamplingParams(
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        stop=["</answer>"],
    )

    # Add a light instruction for structured output
    prompts = [
        f"Solve the following problem step by step. Put the final answer inside <answer> </answer> tags.\n\nProblem: {q}\n\nSolution:"
        for q in questions
    ]

    outputs = llm.generate(prompts, sampling_params)

    results = []
    correct = 0

    for i, output in enumerate(outputs):
        generated = output.outputs[0].text
        pred = extract_answer(generated)
        gold = ground_truths[i]
        match = is_correct(pred, gold)

        if match:
            correct += 1

        results.append({
            "question": questions[i],
            "generated": generated,
            "prediction": pred,
            "ground_truth": gold,
            "correct": match,
        })

    accuracy = correct / len(questions) if questions else 0.0

    summary = {
        "accuracy": accuracy,
        "correct": correct,
        "total": len(questions),
        "results": results,
    }

    return summary


if __name__ == "__main__":
    # Example questions (replace with real GSM8K / MATH samples)
    sample_questions = [
        "Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
        "A robe takes 2 bolts of blue fabric and half as much white fabric. How many bolts of fabric does it take in total?",
    ]
    sample_answers = ["18", "3"]

    print("Running advanced evaluation...")
    summary = evaluate(sample_questions, sample_answers)

    print(f"\nAccuracy: {summary['accuracy']*100:.1f}% ({summary['correct']}/{summary['total']})")
    print("\nDetailed results saved conceptually in summary['results']")

    # Optionally save
    with open("eval_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("Results written to eval_results.json")
