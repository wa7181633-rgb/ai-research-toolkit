# NVIDIA Skills Alignment Audit

**Repository:** `wa7181633-rgb/ai-research-toolkit`  
**Reviewed:** 2026-11-19  
**Base revision:** `e0855e7fe86b51304407c38b674b114565561873`  
**Scope:** Static review of the default-branch source and comparison with the live NVIDIA Skills catalog. No GPU runtime, model download, or training job was executed.

## Executive assessment

The toolkit is a useful single-node LLM research starter: it covers synthetic math data, SFT/GRPO, numeric evaluation, vLLM serving/benchmarking, and a small TransformerLens probe. It is **not yet production-ready or NVIDIA-stack integrated**. Its NVIDIA alignment is currently indirect: it uses PyTorch, Unsloth, TRL, vLLM, and TransformerLens, but has no CUDA-Q, NeMo, TensorRT-LLM, Triton, or NVIDIA GPU profiling workflow.

## What is already strong

- End-to-end research path from generated data through SFT/GRPO to evaluation.
- Numeric reward checks and a separate evaluation script with per-example records.
- Synthetic-data generation uses an AST allowlist rather than unrestricted `eval`.
- Serving and benchmark concerns are at least separated into a dedicated module.
- Interpretability work is represented instead of treating model accuracy as the only metric.

## Findings and priorities

### P0 — Resolve before trusting training results

1. **SFT-to-GRPO adapter handoff is suspect.** The SFT stage saves a PEFT adapter directory, then the GRPO stage passes that directory to `AutoModelForCausalLM.from_pretrained`. A PEFT adapter directory is not generally a standalone base-model checkpoint. Load the original base model and attach the adapter with PEFT/Unsloth's supported adapter-loading path, then verify the exact installed TRL/Unsloth versions.
2. **Dependency constraints are not reproducible.** `requirements.txt` uses broad lower bounds for Torch, Transformers, TRL, vLLM, and an Unsloth Git URL. These packages have tightly coupled compatibility requirements and can resolve to an incompatible environment. Add tested CUDA-specific lock/constraints files and a CPU-only lint/test environment.
3. **The advertised quick-start is not a verified run.** The README's end-to-end command chain assumes the training output is directly loadable by vLLM and that the installed package APIs match the code. Validate every transition with a tiny smoke run before calling it a working pipeline.

### P1 — Reliability and evaluation quality

4. **Reward extraction can mis-score answers.** Numeric extraction takes the last number in free-form text; malformed or multi-number answers can be rewarded accidentally. Prefer strict answer-tag parsing, explicit invalid-format handling, and a Decimal/Fraction-based comparator for exact numeric tasks.
5. **Reward implementations are duplicated.** `training/rewards.py` and `training/grpo_math_train.py` contain separate reward logic. Consolidate them and add unit tests for commas, negatives, fractions, zero denominators, missing tags, multiple numbers, and mismatched batch lengths.
6. **Evaluation is narrow.** GSM8K accuracy and XML compliance are useful but insufficient. Add multiple seeds, confidence intervals, pass@k where applicable, contamination checks, a held-out benchmark suite (e.g. MATH/AIME-style tasks), and regression thresholds.
7. **Benchmark latency labels need precision.** Report TTFT, end-to-end latency, and inter-token/decode latency separately. Current decode timing should not be presented as full request latency. Add warm-up/repeated trials and report percentiles with sample counts.

### P1 — Serving and security

8. **Server exposure defaults to all interfaces.** `0.0.0.0` is convenient in containers but unsafe on an untrusted network without authentication, TLS, firewalling, and explicit deployment guidance. Default to loopback for local use.
9. **Remote model code is trusted implicitly** in evaluation/benchmark paths. Make `trust_remote_code` opt-in and document the supply-chain risk.
10. Add health/readiness checks, structured logs, request limits, graceful shutdown, and a small load-test profile.

### P2 — NVIDIA-specific opportunities

The live NVIDIA catalog has directly relevant lanes and skills, including:
- **Training AI:** `launch-nemo-rl`, `nemo-rl-docs`, `nemo-automodel-distributed-training`, `nemo-mbridge-perf-memory-tuning`, `nemo-mbridge-perf-parallelism-strategies`.
- **Inference AI:** `fastgen-inference`.
- **GPU Development:** `compute-sanitizer-diagnose-cuda-bugs`.
- **Quantum Computing:** `cudaq-guide`, `cudaq-importing`.
- **Agentic AI / retrieval:** `rag-blueprint`, `rag-eval`, `rag-perf`.

Recommended evolution:
1. Add an optional NVIDIA NeMo RL backend alongside the current TRL/Unsloth backend; do not replace the lightweight path until benchmarked.
2. Add a CUDA-Q research module with reproducible examples and classical baselines if quantum ML is an intended project direction.
3. Add TensorRT-LLM/Triton or NVIDIA's current inference stack as an optional serving backend, with apples-to-apples throughput, TTFT, latency, and quality comparisons.
4. Add Nsight Systems/Compute profiling recipes and GPU memory/throughput telemetry.
5. Add a retrieval/RAG evaluation lane only if research scope includes knowledge-grounded systems.

## Validation status

| Check | Status |
|---|---|
| Repository metadata and source inspection | Completed |
| NVIDIA live catalog lookup | Completed |
| Python unit tests | Not run in this review environment |
| CUDA/GPU training smoke test | Not run |
| vLLM server launch/load test | Not run |
| End-to-end SFT → GRPO → vLLM test | Not run |

## Suggested next implementation sprint

- Fix and test adapter handoff.
- Pin a known-good CUDA/PyTorch/TRL/Unsloth/vLLM matrix.
- Add unit tests and CI (lint, import checks, CPU-safe tests).
- Add an NVIDIA NeMo RL comparison recipe and CUDA-Q starter module.
- Add a secure local-serving profile and accurate latency instrumentation.

This audit is a static engineering review, not a claim that the toolkit has passed runtime or GPU validation.
