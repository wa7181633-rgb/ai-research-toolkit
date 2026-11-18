# Reward Function Design Prompt

```
Design a set of reward functions for GRPO / RLVR training on the following task:

Task description:
{{TASK}}

Requirements:
- Prefer verifiable / programmatic rewards over learned reward models when possible
- Cover correctness, format, intermediate reasoning quality, and safety if relevant
- Make the rewards dense enough to provide learning signal early in training
- Suggest how to normalize or combine multiple reward components
- Warn about potential reward hacking vectors

Output concrete Python-style reward functions + explanation.
```
