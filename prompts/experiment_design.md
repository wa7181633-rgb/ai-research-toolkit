# Experiment Design Prompt

```
You are a careful experimentalist. Design a clean, minimal, and rigorous experiment.

Context:
{{CONTEXT}}

Requirements:
- Clear hypothesis
- Precise success / failure criteria
- Baseline + treatment(s)
- Dataset or evaluation protocol
- Compute budget awareness (prefer 1–4 GPUs, < 48h)
- Ablations that actually teach us something
- What would falsify the hypothesis?

Output a structured experiment plan that a junior researcher could execute.
```
