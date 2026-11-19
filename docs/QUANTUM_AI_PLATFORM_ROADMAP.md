# Quantum-AI Platform Roadmap

**Status:** Architecture proposal  
**Target:** Evolve this repository into a reproducible Quantum-AI research platform.  
**Principle:** Classical baselines first; quantum components must demonstrate measurable value against those baselines.

## 1. Product direction

Build a modular research workbench that lets researchers define an experiment, select classical or hybrid quantum-classical methods, execute on local hardware/simulators or supported NVIDIA GPU infrastructure, and compare results with reproducible artifacts.

The platform is not a claim of quantum advantage. Every quantum experiment must include a classical baseline, resource accounting, and an honest statement of simulator or hardware limitations.

## 2. Target architecture

| Layer | Responsibility | Initial implementation |
|---|---|---|
| Experiment interface | Config, run metadata, reproducibility | YAML/JSON experiment specs |
| Dataset and task layer | Versioned datasets, task definitions, splits | Existing synthetic-data and GSM8K utilities |
| Classical ML runtime | SFT, GRPO, evaluation, inference | Existing training/evaluation modules |
| Quantum runtime | Circuits, simulators, hardware adapters | Optional NVIDIA CUDA-Q integration |
| Hybrid orchestration | Classical optimizer + quantum circuit execution | Python interfaces; no hard CUDA-Q dependency |
| Evaluation | Accuracy, quality, cost, latency, variance | Shared benchmark schema |
| Observability | Logs, metrics, artifacts, environment manifest | JSONL initially |
| Serving/API | Run submission and result retrieval | Later phase; not required for first milestone |

Suggested package boundaries:

```text
ai_research_toolkit/
  core/                 # experiment specs, run context, interfaces
  classical/            # adapters around current SFT/GRPO/evaluation
  quantum/
    backends/            # simulator and hardware adapters
    circuits/            # reusable circuit templates
    algorithms/          # QNN, VQE/QAOA experiments
  hybrid/               # parameter updates and orchestration
  benchmarks/            # shared metrics and baseline comparisons
  artifacts/             # manifests, logs, result schemas
  tests/
docs/
```

This is a target layout, not a request to move existing files immediately. Preserve the working scripts while introducing interfaces incrementally.

## 3. NVIDIA technology alignment

Use NVIDIA skills and SDKs where they solve a concrete engineering problem:

- **CUDA-Q:** primary candidate for quantum circuit definition, simulation, and supported QPU execution. Start with simulator-backed examples and keep backend selection configurable.
- **NeMo RL / AutoModel:** evaluate for scalable LLM post-training after the current TRL/Unsloth pipeline is stable. Do not migrate solely for branding.
- **TensorRT-LLM / vLLM:** compare serving stacks with measured throughput, time-to-first-token, inter-token latency, and memory use before selecting a production path.
- **Nsight Systems / Compute Sanitizer:** use for profiling and CUDA correctness once custom GPU kernels or CUDA-Q integration are introduced.

The NVIDIA Skills catalog is a workflow reference, not a substitute for SDK compatibility checks or runtime tests. Pin SDK versions and record GPU, driver, CUDA, and backend details per run.

## 4. First quantum research tracks

### Track A — Variational quantum circuits / QNN

- Define small, controlled classification or regression tasks.
- Compare a parameterized quantum circuit against a parameter-matched classical model.
- Record qubit count, circuit depth, shots, simulator/backend, optimizer, seeds, and wall-clock cost.
- Test noise sensitivity and scaling limits before making performance claims.

### Track B — QAOA for optimization

- Start with small Max-Cut instances with known exact solutions.
- Compare QAOA against exact classical solvers and strong heuristic baselines.
- Report approximation ratio, solution quality, runtime, shot count, and total compute cost.
- Increase graph size only when the simulator or hardware can support a meaningful comparison.

### Track C — Hybrid quantum-classical experimentation

- Keep the classical optimizer independent from the quantum backend.
- Define a stable interface for circuit execution and expectation-value results.
- Support simulator-first testing; hardware credentials and provider-specific dependencies remain optional.

## 5. Experiment contract

Every run should persist a machine-readable manifest containing:

- experiment ID, source revision, config hash, timestamp
- task, dataset identifier/version, split, and preprocessing
- model/circuit architecture and all hyperparameters
- random seeds and deterministic settings
- runtime versions: Python, PyTorch, CUDA, CUDA-Q, driver, backend
- hardware details: GPU, QPU/provider, qubits, shots, simulator mode
- metrics, resource use, duration, warnings, and artifact locations

Do not compare runs when task definitions, data splits, or measurement conditions differ without explicitly documenting the difference.

## 6. Delivery sequence

### Phase 0 — Stabilize the existing classical toolkit
1. Fix and test the SFT adapter-to-GRPO model-loading handoff.
2. Pin compatible dependency sets and add CI smoke tests.
3. Make reward parsing strict and consolidate duplicate reward logic.
4. Harden inference defaults and make remote-code trust opt-in.
5. Standardize evaluation and benchmark metrics.

**Exit criteria:** clean environment setup, unit tests, and a documented small end-to-end run. GPU training is separately validated on actual supported hardware.

### Phase 1 — Add the quantum foundation
1. Add an optional CUDA-Q dependency extra; classical installation must work without CUDA-Q.
2. Add a backend interface and a simulator smoke test.
3. Implement one small QNN experiment and one small QAOA/Max-Cut experiment.
4. Add classical baselines and shared experiment manifests.

**Exit criteria:** reproducible simulator runs with tests and comparable classical baselines. No quantum-advantage claim.

### Phase 2 — Hybrid orchestration
1. Add reusable optimizer/circuit execution loops.
2. Add noise and shot-budget sweeps.
3. Add result comparison and experiment tracking.
4. Validate on supported NVIDIA GPU/QPU configurations.

**Exit criteria:** repeatable hybrid runs with transparent cost and variance reporting.

### Phase 3 — Research platform
1. Add experiment registry and artifact browser.
2. Add API/CLI orchestration and queued execution.
3. Add team-level access controls, secrets handling, and deployment profiles.
4. Add scalable classical LLM training/inference integrations only where benchmark evidence supports them.

## 7. Non-negotiable quality gates

- Classical-only installation and tests must remain functional.
- Quantum SDK imports must be lazy/optional.
- Unit tests must not require a physical QPU.
- Simulator results must be labeled as simulator results.
- Quantum methods must be compared with credible classical baselines.
- Report variance across seeds/shots and include resource costs.
- Never describe a result as quantum advantage without a statistically and methodologically defensible comparison.
- No secrets, API keys, or provider credentials in source control.

## 8. Immediate implementation backlog

| Priority | Work item | Reason |
|---|---|---|
| P0 | Repair SFT adapter loading into GRPO and test the handoff | Existing pipeline correctness |
| P0 | Add pinned/validated dependency groups and CI smoke tests | Reproducibility |
| P1 | Shared experiment manifest and result schema | Enables comparable classical/quantum runs |
| P1 | Optional CUDA-Q backend interface + simulator smoke test | Establishes quantum foundation safely |
| P1 | QAOA Max-Cut baseline experiment | Small, verifiable first quantum benchmark |
| P2 | QNN task and noise/shot sweeps | Broader quantum ML research |
| P2 | Experiment registry/API/UI | Platformization after runtime stability |

## 9. Scope and validation note

This roadmap is an architecture proposal based on a static repository review. It does not mean CUDA-Q, GPU training, QPU access, or end-to-end execution has been tested. Before selecting exact CUDA-Q APIs or dependency pins, validate the target CUDA-Q release, operating system, Python version, NVIDIA driver, CUDA compatibility, and available simulator/QPU backend.
