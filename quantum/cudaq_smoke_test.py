#!/usr/bin/env python3
"""Minimal CUDA-Q runtime smoke test using a two-qubit Bell state.

This test intentionally uses CUDA-Q's default target. Select/configure a target
with the CUDA-Q CLI or environment appropriate to your installed release.
It does not require a physical QPU.
"""

from __future__ import annotations

import argparse
import sys


def build_bell_kernel(cudaq):
    @cudaq.kernel
    def bell_state():
        qubits = cudaq.qvector(2)
        h(qubits[0])
        x.ctrl(qubits[0], qubits[1])
        mz(qubits)

    return bell_state


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shots", type=int, default=1000)
    parser.add_argument(
        "--min-correlated-fraction",
        type=float,
        default=0.90,
        help="Minimum fraction of shots measured as 00 or 11.",
    )
    args = parser.parse_args()

    if args.shots < 100:
        parser.error("--shots must be at least 100 for a useful smoke test")
    if not 0.5 < args.min_correlated_fraction <= 1.0:
        parser.error("--min-correlated-fraction must be in (0.5, 1.0]")

    try:
        import cudaq
    except ImportError:
        print(
            "CUDA-Q is not installed in this Python environment. "
            "Install the CUDA-Q SDK using NVIDIA's instructions, then retry.",
            file=sys.stderr,
        )
        return 2

    try:
        kernel = build_bell_kernel(cudaq)
        result = cudaq.sample(kernel, shots_count=args.shots)
        outcomes = dict(result.items())
    except Exception as exc:  # Surface SDK/target/runtime incompatibilities.
        print(f"CUDA-Q execution failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 3

    total = sum(int(count) for count in outcomes.values())
    if total <= 0:
        print("CUDA-Q returned no measurement shots.", file=sys.stderr)
        return 4

    correlated = sum(
        int(count)
        for bitstring, count in outcomes.items()
        if str(bitstring).replace(" ", "") in {"00", "11"}
    )
    fraction = correlated / total

    print(f"Shots returned: {total}")
    print(f"Measurement counts: {outcomes}")
    print(f"Bell-state correlation fraction: {fraction:.4f}")

    if total != args.shots:
        print(
            f"Expected {args.shots} shots but received {total}.",
            file=sys.stderr,
        )
        return 5
    if fraction < args.min_correlated_fraction:
        print(
            "Bell-state correlation is below the configured threshold.",
            file=sys.stderr,
        )
        return 6

    print("CUDA-Q smoke test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
