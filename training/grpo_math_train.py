import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

# 1. Load Base Model & Tokenizer for GRPO (Group Relative Policy Optimization)
model_id = "unsloth/llama-3-8b-Instruct-bnb-4bit"
model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.bfloat16, device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# 2. Define Math Reasoning Reward Functions (GSM8K Style)
def accuracy_reward(completions, solutions, **kwargs):
    """Reward function checking if the generated answer matches the ground truth solution."""
    rewards = []
    for completion, solution in zip(completions, solutions):
        # Extract numerical answer and compare
        rewards.append(1.0 if solution.strip() in completion else 0.0)
    return rewards

def format_reward(completions, **kwargs):
    """Reward function enforcing step-by-step reasoning structure (<thought> ... <answer>)."""
    rewards = []
    for completion in completions:
        has_thought = "<thought>" in completion and "</thought>" in completion
        has_answer = "<answer>" in completion and "</answer>" in completion
        rewards.append(1.0 if (has_thought and has_answer) else 0.2)
    return rewards

# 3. Configure GRPO Training Arguments
training_args = GRPOConfig(
    output_dir="grpo_math_output",
    learning_rate=5.0e-6,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_generations=4, # Group size for relative policy scoring
    max_prompt_length=512,
    max_completion_length=512,
    logging_steps=1,
    save_steps=10,
    bf16=True,
)

# 4. Initialize GRPOTrainer (No separate value model required)
trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[accuracy_reward, format_reward],
    args=training_args,
    train_dataset=None, # Provide your GSM8K-style math dataset here
)

if __name__ == "__main__":
    print("GRPO Math Reasoning pipeline initialized successfully using ai-research-skills.")
