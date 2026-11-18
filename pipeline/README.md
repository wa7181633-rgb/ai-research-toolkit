# Pipeline Module

## Synthetic Data

`synthetic_data.py` generates structured reasoning examples in the format:

```json
{
  "prompt": "...",
  "completion": "<thought>\n...\n</thought>\n<answer>\n...\n</answer>",
  "solution": "..."
}
```

This format works well for both SFT and GRPO.

In serious research, replace the simple template generator with calls to a strong model + filtering (correctness, diversity, difficulty).
