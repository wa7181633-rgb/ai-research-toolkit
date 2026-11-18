"""
Advanced GRPO Training Pipeline for Math Reasoning
==================================================
Research-grade skeleton using Unsloth + TRL GRPO.

Features:
- Unsloth memory-efficient loading + LoRA
- Multi-component reward functions (accuracy + format + process)
- Proper GRPOConfig for reasoning models
- Easy to plug real datasets (GSM8K / MATH / custom)

Replace the placeholder dataset section with your actual data.
"""

import re
import torch
from datasets import load_dataset, Dataset
from unsloth import FastLanguageModel
from trl import GRPOConfig, GRPOTrainer

# =======================
# 1. Model Configuration
# ======================

MODEL_NAME = "unsloth/Qwen2.5-1.5B-Instruct"   # Change to 3B / 7B / 8B as needed
MAX_SEQ_LENGTH = 2048
LORA_RANK = 32
LOAD_IN_4BIT = True

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=LOAD_IN_4BIT,
    fast_inference=True,          # Important for GRPO generation speed
)

model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_RANK,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=LORA_RANK * 2,
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

# ======================
# 2. Reward Functions
# ======================

def extract_final_answer(text: str) -> str:
    """Extract the final numerical or short answer from model output."""
    # Prefer <answer>...</answer> format
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # Fallback: last number in the text
    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", text)
    if numbers:
        return numbers[-1]
    return text.strip()[-50:]  # last resort


def accuracy_reward(completions, solutions, **kwargs):
    """
    Primary reward: exact or near-exact match with ground truth.
    Returns list of floats.
    """
    rewards = []
    for completion, solution in zip(completions, solutions):
        pred = extract_final_answer(completion).lower().replace(",", "").strip()
        gold = str(solution).lower().replace(",", "").strip()

        if pred == gold:
            rewards.append(1.0)
        elif gold in pred or pred in gold:
            rewards.append(0.5)   # partial credit
        else:
            rewards.append(0.0)
    return rewards


def format_reward(completions, **kwargs):
    """
    Encourages structured reasoning:
    <thought> ... </thought>
    <answer> ... </answer>
    """
    rewards = []
    for completion in completions:
        score = 0.0
        if "<thought>" in completion and "</thought>" in completion:
            score += 0.4
        if "<answer>" in completion and "</answer>" in completion:
            score += 0.6
        rewards.append(score)
    return rewards


def length_penalty(completions, **kwargs):
    """Mild penalty for extremely short or extremely long answers."""
    rewards = []
    for completion in completions:
        length = len(completion)
        if length < 30:
            rewards.append(-0.3)
        elif length > 1800:
            rewards.append(-0.2)
        else:
            rewards.append(0.1)
    return rewards


# ======================
# 3. Dataset Preparation
# ======================

def build_dataset():
    """
    Replace this with a real dataset.

    Expected format for GRPO:
    - prompt: the question
    - solution: ground truth answer (used by accuracy_reward)
    """
    # Example using a small public dataset (uncomment and adapt)
    # raw = load_dataset("openai/gsm8k", "main", split="train[:2000]")
    #
    # def format_example(example):
    #     return {
    #         "prompt": example["question"],
    #         "solution": example["answer"].split("####")[-1].strip(),
    #     }
    #
    # return raw.map(format_example)

    # Placeholder for demonstration
    data = [
        {
            "prompt": "What is 15 + 27?",
            "solution": "42",
        },
        {
            "prompt": "A store sells pens for $2 each. If you buy 8 pens, how much do you pay?",
            "solution": "16",
        },
    ]
    return Dataset.from_list(data)


train_dataset = build_dataset()

# ======================
# 4. GRPO Configuration
# ======================

training_args = GRPOConfig(
    output_dir="./outputs/grpo-math",
    learning_rate=5e-6,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_generations=8,                  # Group size (critical for GRPO)
    max_prompt_length=512,
    max_completion_length=1024,
    max_steps=500,                      # Increase for real runs
    logging_steps=5,
    save_steps=50,
    bf16=True,
    report_to="none",
    # Advanced options (uncomment when needed):
    # loss_type="grpo",                # or "dr_grpo", "dapo"
    # mask_truncated_completions=True,
)

# ======================
# 5. Trainer
# ======================

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        accuracy_reward,
        format_reward,
        length_penalty,
    ],
    args=training_args,
    train_dataset=train_dataset,
)

if __name__ == "__main__":
    print("Starting advanced GRPO training...")
    print(f"Model: {MODEL_NAME}")
    print(f"LoRA rank: {LORA_RANK}")
    print(f"Dataset size: {len(train_dataset)}")
    print("Reward functions: accuracy + format + length")
    # Uncomment to actually train:
    # trainer.train()
    print("\nPipeline ready. Uncomment trainer.train() when dataset is prepared.")
