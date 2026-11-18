"""
Minimal GRPO training skeleton using Unsloth + TRL style.
This is a template — adapt model, dataset and reward function to your task.
"""

from unsloth import FastLanguageModel
from trl import GRPOConfig, GRPOTrainer
from datasets import load_dataset

# 1. Load model with Unsloth
max_seq_length = 2048
lora_rank = 32

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-1.5B-Instruct",   # change as needed
    max_seq_length=max_seq_length,
    load_in_4bit=True,
    fast_inference=True,          # enable vLLM for generation
)

model = FastLanguageModel.get_peft_model(
    model,
    r=lora_rank,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=lora_rank * 2,
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

# 2. Simple reward function example (replace with your logic)
def reward_func(completions, **kwargs):
    """Very basic example: reward length + presence of thinking tags."""
    rewards = []
    for completion in completions:
        score = 0.0
        if "<think>" in completion and "</think>" in completion:
            score += 1.0
        # Prefer reasonable length
        length = len(completion)
        if 100 < length < 1500:
            score += 0.5
        rewards.append(score)
    return rewards

# 3. Config
config = GRPOConfig(
    output_dir="./grpo-qwen-reasoning",
    learning_rate=5e-6,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_generations=8,
    max_prompt_length=512,
    max_completion_length=1024,
    max_steps=300,                # start small, watch reward curves
    logging_steps=5,
    save_steps=50,
    report_to="none",
)

# 4. Dataset (replace with your own)
# dataset = load_dataset("your-dataset", split="train")

print("Template ready. Add your dataset + improve the reward function.")
