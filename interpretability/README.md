# Mechanistic Interpretability

Tools and recipes for understanding *what* language models are actually doing internally.

## Main Library

**Transformer Lens** remains the most popular and practical library for residual stream analysis, attention head inspection, and circuit discovery.

```bash
pip install transformer-lens
```

## Common Workflows

1. Load a model into Transformer Lens
2. Run forward passes with caching
3. Inspect residual stream norms / directions
4. Ablate or patch specific heads / layers
5. Look for induction heads, previous-token heads, etc.

See `transformer_lens_starter.py` for a minimal example.
