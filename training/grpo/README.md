# GRPO — Group Relative Policy Optimization

GRPO is the reinforcement learning algorithm used by DeepSeek to train R1-style reasoning models.  
It removes the need for a separate value/critic model by using group statistics of rewards.

### Why GRPO?

- Much lower memory than classic PPO
- Works extremely well with verifiable rewards (math, code, structured output)
- Excellent when combined with Unsloth

### Key Ideas

1. For each prompt, generate a **group** of responses (e.g. 4–16).
2. Score every response with a reward function.
3. Normalize scores relative to the group mean / std.
4. Update the policy to prefer higher-scoring responses in the group.

### Quick Start with Unsloth

See the example notebook and script in this folder, or follow the official Unsloth GRPO tutorial:

- [Unsloth GRPO Docs](https://docs.unsloth.ai/get-started/reinforcement-learning-rl-guide/tutorial-train-your-own-reasoning-model-with-grpo)

### Minimal Config Example (TRL + Unsloth style)

```python
from trl import GRPOConfig, GRPOTrainer

config = GRPOConfig(
    output_dir="./grpo-output",
    learning_rate=5e-6,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_generations=8,          # group size
    max_prompt_length=512,
    max_completion_length=1024,
    num_train_epochs=1,
    logging_steps=10,
    save_steps=100,
    use_vllm=True,              # fast generation
)
```

### Reward Function Tips

- Math: exact match + partial credit for intermediate steps
- Code: unit tests + execution success
- Format: reward for correct `<think>...\n</think>` + final answer structure

Start simple. A clean, verifiable reward beats a fancy learned reward model for most research use cases.
