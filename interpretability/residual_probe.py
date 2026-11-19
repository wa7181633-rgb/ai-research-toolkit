"""
Mechanistic Interpretability Probe using Transformer Lens.
Captures residual stream activations across layers and computes logit attribution
for candidate continuation tokens so researchers can inspect where reasoning shifts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from transformer_lens import HookedTransformer


def run_residual_probe(
    model_name: str,
    prompt: str,
    target_tokens: list[str],
    output_file: str,
) -> dict[str, Any]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    model = HookedTransformer.from_pretrained(model_name, device=device, dtype=dtype)
    tokens = model.to_tokens(prompt, prepend_bos=True)

    _, cache = model.run_with_cache(tokens)
    target_ids: dict[str, int] = {}
    for tok_str in target_tokens:
        ids = model.to_single_token(tok_str) if len(model.to_tokens(tok_str, prepend_bos=False)[0]) == 1 else model.to_tokens(tok_str, prepend_bos=False)[0, -1].item()
        target_ids[tok_str] = int(ids)

    layer_profiles: list[dict[str, Any]] = []
    n_layers = model.cfg.n_layers
    for layer_idx in range(n_layers):
        resid_post = cache[f"blocks.{layer_idx}.hook_resid_post"][0, -1, :]
        scaled_resid = cache.apply_ln_to_stack(resid_post.unsqueeze(0), layer=layer_idx, pos_slice=-1)
        logits = model.unembed(scaled_resid)[0]
        probs = torch.softmax(logits, dim=-1)

        token_stats = {
            label: {
                "logit": round(float(logits[tid].item()), 4),
                "prob": round(float(probs[tid].item()), 6),
            }
            for label, tid in target_ids.items()
        }
        layer_profiles.append({"layer": layer_idx, "targets": token_stats})

    final_logits = model(tokens)[0, -1]
    top_k = torch.topk(torch.softmax(final_logits, dim=-1), k=5)
    top_predictions = [
        {"token": model.to_single_str_token(int(idx)), "prob": round(float(p), 5)}
        for p, idx in zip(top_k.values, top_k.indices)
    ]

    report = {
        "model": model_name,
        "device": device,
        "prompt": prompt,
        "layers_probed": n_layers,
        "top_next_tokens": top_predictions,
        "layer_trajectory": layer_profiles,
    }

    out = Path(output_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return {
        "model": model_name,
        "device": device,
        "layers_probed": n_layers,
        "top_next_tokens": top_predictions,
        "report_path": str(out),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transformer Lens Residual Stream Probe")
    parser.add_argument("--model", default="gpt2-small")
    parser.add_argument("--prompt", default="Step 1: 12 * 4 = 48. Step 2: 48 - 9 =")
    parser.add_argument("--targets", nargs="+", default=[" 39", " 38", " 57", " 48"])
    parser.add_argument("--output", default="benchmarks/residual_probe.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_residual_probe(
        model_name=args.model,
        prompt=args.prompt,
        target_tokens=args.targets,
        output_file=args.output,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
