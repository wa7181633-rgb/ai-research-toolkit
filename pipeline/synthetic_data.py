"""
Advanced Synthetic Reasoning Data Pipeline
==========================================
Generates high-quality math / reasoning examples with:
- Structured thought process
- Explicit final answer
- Filtering hooks for quality control

In real research you would call a strong model (GPT-4o, Claude, DeepSeek-R1, etc.)
instead of the simple template used here.
"""

import json
import random
from typing import List, Dict
from pathlib import Path

# ======================
# Templates & Generators
# ======================

TEMPLATES = [
    {
        "type": "linear_equation",
        "prompt": "Solve for x: {a}x + {b} = {c}",
        "thought": "Subtract {b} from both sides: {a}x = {c_minus_b}. Then divide both sides by {a}: x = {answer}.",
        "answer_fn": lambda a, b, c: (c - b) // a if (c - b) % a == 0 else None,
    },
    {
        "type": "percentage",
        "prompt": "A product costs ${price}. It is discounted by {pct}%. What is the final price?",
        "thought": "{pct}% of {price} is {discount}. Subtracting the discount: {price} - {discount} = {answer}.",
        "answer_fn": lambda price, pct: price - (price * pct // 100),
    },
]

def generate_one() -> Dict | None:
    template = random.choice(TEMPLATES)

    if template["type"] == "linear_equation":
        a = random.randint(2, 9)
        b = random.randint(1, 20)
        answer = random.randint(1, 15)
        c = a * answer + b
        thought = template["thought"].format(
            a=a, b=b, c=c, c_minus_b=c-b, answer=answer
        )
        prompt = template["prompt"].format(a=a, b=b, c=c)
        return {
            "prompt": prompt,
            "thought": thought,
            "answer": str(answer),
            "type": template["type"],
        }

    elif template["type"] == "percentage":
        price = random.choice([40, 50, 80, 100, 120, 150, 200])
        pct = random.choice([10, 15, 20, 25, 30, 40, 50])
        discount = price * pct // 100
        answer = price - discount
        thought = template["thought"].format(
            price=price, pct=pct, discount=discount, answer=answer
        )
        prompt = template["prompt"].format(price=price, pct=pct)
        return {
            "prompt": prompt,
            "thought": thought,
            "answer": str(answer),
            "type": template["type"],
        }

    return None


def generate_dataset(n: int = 200, seed: int = 42) -> List[Dict]:
    random.seed(seed)
    dataset = []
    seen = set()

    while len(dataset) < n:
        item = generate_one()
        if item is None:
            continue
        key = item["prompt"]
        if key in seen:
            continue
        seen.add(key)

        # Final structured format (ready for GRPO / SFT)
        formatted = {
            "prompt": item["prompt"],
            "completion": f"<thought>\n{item['thought']}\n</thought>\n<answer>\n{item['answer']}\n</answer>",
            "solution": item["answer"],
            "type": item["type"],
        }
        dataset.append(formatted)

    return dataset


def save_dataset(dataset: List[Dict], path: str = "synthetic_reasoning.jsonl"):
    path = Path(path)
    with path.open("w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Saved {len(dataset)} examples → {path}")


if __name__ == "__main__":
    print("Generating advanced synthetic reasoning dataset...")
    data = generate_dataset(n=300)
    save_dataset(data)

    # Show a sample
    print("\nSample example:")
    print(json.dumps(data[0], indent=2))
