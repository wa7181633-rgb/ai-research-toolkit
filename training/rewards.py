"""
Reusable Advanced Reward Functions for GRPO
===========================================
Import these into any GRPO training script.
"""

import re
from typing import List

def extract_final_answer(text: str) -> str:
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", text.replace(",", ""))
    if numbers:
        return numbers[-1]
    return text.strip()[-40:]


def accuracy_reward(completions: List[str], solutions: List[str], **kwargs) -> List[float]:
    rewards = []
    for completion, solution in zip(completions, solutions):
        pred = extract_final_answer(completion).lower().replace(",", "").strip()
        gold = str(solution).lower().replace(",", "").strip()

        if pred == gold:
            rewards.append(1.0)
        elif gold in pred or pred in gold:
            rewards.append(0.6)
        else:
            try:
                if abs(float(pred) - float(gold)) < 1e-4:
                    rewards.append(1.0)
                else:
                    rewards.append(0.0)
            except Exception:
                rewards.append(0.0)
    return rewards


def format_reward(completions: List[str], **kwargs) -> List[float]:
    rewards = []
    for text in completions:
        score = 0.0
        if "<thought>" in text and "</thought>" in text:
            score += 0.45
        if "<answer>" in text and "</answer>" in text:
            score += 0.55
        rewards.append(score)
    return rewards


def process_reward(completions: List[str], **kwargs) -> List[float]:
    """
    Simple process supervision proxy:
    Reward presence of intermediate reasoning steps / equations.
    """
    rewards = []
    for text in completions:
        score = 0.0
        # Presence of math operators or step indicators
        if any(op in text for op in ["=", "+", "-", "*", "/"]):
            score += 0.3
        if any(word in text.lower() for word in ["therefore", "so", "thus", "step"]):
            score += 0.2
        if text.count("\n") >= 2:   # multi-line reasoning
            score += 0.2
        rewards.append(min(score, 1.0))
    return rewards


def length_reward(completions: List[str], **kwargs) -> List[float]:
    rewards = []
    for text in completions:
        length = len(text)
        if 80 <= length <= 1200:
            rewards.append(0.15)
        elif length < 40:
            rewards.append(-0.4)
        elif length > 2000:
            rewards.append(-0.25)
        else:
            rewards.append(0.0)
    return rewards
