"""
Minimal Transformer Lens starter.
Load a small model and inspect residual stream / attention patterns.
"""

import torch
from transformer_lens import HookedTransformer

# Load a small model (change as needed)
model = HookedTransformer.from_pretrained(
    "gpt2-small",
    device="cuda" if torch.cuda.is_available() else "cpu"
)

prompt = "The capital of France is"
tokens = model.to_tokens(prompt)

# Run with cache
logits, cache = model.run_with_cache(tokens)

print("Available cache keys (first 10):")
print(list(cache.keys())[:10])

# Example: residual stream at final layer
resid = cache["resid_post", -1]          # (batch, seq, d_model)
print("\nResidual stream shape (final layer):", resid.shape)

# Example: attention pattern of layer 0, head 0
attn = cache["pattern", 0][:, 0]         # (batch, dest, src)
print("Attention pattern shape (L0H0):", attn.shape)

print("\nStarter ready. Explore cache.keys() and model.generate for more.")
