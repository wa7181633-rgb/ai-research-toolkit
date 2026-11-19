"""
Automated Synthetic Math Reasoning Dataset Generator with Verification & Deduplication.
Generates step-by-step arithmetic/algebra problems, verifies answers via Python AST
safe arithmetic evaluation, deduplicates prompts, and exports JSONL ready for SFT/GRPO.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import operator
import random
from pathlib import Path
from typing import Any


_SAFE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def safe_eval_expr(expr: str) -> float:
    """Safely evaluate pure arithmetic expressions using AST nodes."""
    node = ast.parse(expr.strip(), mode="eval").body

    def _eval(n: ast.AST) -> float:
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return float(n.value)
        if isinstance(n, ast.BinOp) and type(n.op) in _SAFE_OPS:
            left = _eval(n.left)
            right = _eval(n.right)
            if isinstance(n.op, (ast.Div, ast.FloorDiv, ast.Mod)) and right == 0:
                raise ZeroDivisionError("Division by zero")
            if isinstance(n.op, ast.Pow) and abs(right) > 12:
                raise ValueError("Exponent too large")
            return float(_SAFE_OPS[type(n.op)](left, right))
        if isinstance(n, ast.UnaryOp) and type(n.op) in _SAFE_OPS:
            return float(_SAFE_OPS[type(n.op)](_eval(n.operand)))
        raise ValueError(f"Unsupported AST construct: {type(n).__name__}")

    return _eval(node)


def generate_word_problem(rng: random.Random) -> dict[str, Any]:
    """Generate a verified multi-step word problem with explicit reasoning trace."""
    kind = rng.choice(["inventory_batch", "discount_margin", "rate_allocation"])
    if kind == "inventory_batch":
        boxes = rng.randint(6, 30)
        units_per_box = rng.randint(12, 48)
        defective = rng.randint(1, max(2, units_per_box // 3))
        price_per_unit = rng.choice([15, 25, 40, 50, 75, 120])

        total_units = boxes * units_per_box
        usable_units = total_units - defective
        revenue = usable_units * price_per_unit
        assert safe_eval_expr(f"({boxes} * {units_per_box} - {defective}) * {price_per_unit}") == float(revenue)

        question = (
            f"A warehouse receives {boxes} boxes with {units_per_box} units each. "
            f"{defective} units are damaged and discarded. Each usable unit sells for {price_per_unit}. "
            f"What is the total revenue from usable units?"
        )
        thought = (
            f"1. Total units = {boxes} * {units_per_box} = {total_units}.\n"
            f"2. Usable units after discarding {defective} damaged units = {total_units} - {defective} = {usable_units}.\n"
            f"3. Total revenue = {usable_units} * {price_per_unit} = {revenue}."
        )
        return {"question": question, "thought": thought, "answer": str(revenue), "category": kind}

    if kind == "discount_margin":
        base_price = rng.choice([200, 450, 800, 1200, 2500])
        discount_pct = rng.choice([10, 15, 20, 25])
        qty = rng.randint(3, 15)

        discounted_unit = base_price * (100 - discount_pct) / 100.0
        total_cost = discounted_unit * qty
        assert abs(safe_eval_expr(f"{base_price} * (100 - {discount_pct}) / 100 * {qty}") - total_cost) < 1e-6

        question = (
            f"An item costs {base_price} and has a {discount_pct}% discount. "
            f"If a buyer purchases {qty} discounted items, what is the total amount payable?"
        )
        thought = (
            f"1. Discount amount per item = {base_price} * {discount_pct} / 100 = {base_price * discount_pct / 100.0:g}.\n"
            f"2. Discounted price per item = {base_price} - {base_price * discount_pct / 100.0:g} = {discounted_unit:g}.\n"
            f"3. Total payable for {qty} items = {discounted_unit:g} * {qty} = {total_cost:g}."
        )
        return {"question": question, "thought": thought, "answer": f"{total_cost:g}", "category": kind}

    total_hours = rng.randint(12, 60)
    shift_hours = rng.choice([4, 6, 8, 12])
    hourly_rate = rng.choice([150, 250, 350, 500])
    full_shifts = total_hours // shift_hours
    remainder_hours = total_hours % shift_hours
    total_payout = total_hours * hourly_rate
    assert safe_eval_expr(f"{total_hours} * {hourly_rate}") == float(total_payout)

    question = (
        f"A process runs for {total_hours} hours across shifts of {shift_hours} hours each, "
        f"paid at {hourly_rate} per hour. How many full shifts occur, and what is the total payout?"
    )
    thought = (
        f"1. Full shifts = {total_hours} // {shift_hours} = {full_shifts} (remainder {remainder_hours}h).\n"
        f"2. Total payout across {total_hours} hours at {hourly_rate}/hr = {total_payout}."
    )
    return {
        "question": question,
        "thought": thought,
        "answer": str(total_payout),
        "full_shifts": full_shifts,
        "category": kind,
    }


def build_dataset(count: int, seed: int, output_file: str) -> dict[str, Any]:
    rng = random.Random(seed)
    seen_hashes: set[str] = set()
    rows: list[dict[str, Any]] = []
    attempts = 0
    max_attempts = count * 10

    while len(rows) < count and attempts < max_attempts:
        attempts += 1
        item = generate_word_problem(rng)
        digest = hashlib.sha256(item["question"].encode("utf-8")).hexdigest()
        if digest in seen_hashes:
            continue
        seen_hashes.add(digest)
        item["id"] = f"syn_{len(rows) + 1:05d}"
        item["sha256"] = digest[:16]
        rows.append(item)

    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")

    return {
        "output_file": str(out_path),
        "generated_count": len(rows),
        "attempts": attempts,
        "unique_questions": len(seen_hashes),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verified Synthetic Math Dataset Pipeline")
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--output", default="pipeline/synthetic_math_train.jsonl")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_dataset(count=args.count, seed=args.seed, output_file=args.output)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
