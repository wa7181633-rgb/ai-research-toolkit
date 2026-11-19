# CUDA-Q smoke test

The repository includes an optional Bell-state smoke test at
`quantum/cudaq_smoke_test.py`. It checks that the CUDA-Q Python package imports,
a two-qubit kernel compiles and executes on the configured default target, and
measurement results show the expected Bell-state correlation.

## Install

CUDA-Q is an optional SDK and is deliberately not added to the base
`requirements.txt`. Install it using the current NVIDIA CUDA-Q installation
instructions for your operating system, Python version, CUDA/driver setup, and
desired target. Do not assume that a wheel or target supported on one platform
is supported on another.

Official resources:

- CUDA-Q documentation: https://nvidia.github.io/cuda-quantum/latest/
- CUDA-Q GitHub repository: https://github.com/NVIDIA/cuda-quantum
- NVIDIA Skills catalog: https://github.com/NVIDIA/skills

## Run

Activate the Python environment in which CUDA-Q is installed, then run:

```bash
python quantum/cudaq_smoke_test.py --shots 1000
```

The script uses the CUDA-Q default target. It does not request a physical QPU.
If the selected default target is a CPU simulator, this validates the SDK and
simulator path only; it does not validate NVIDIA GPU acceleration or QPU access.

Exit codes:

- `0`: test passed
- `2`: CUDA-Q package is not importable
- `3`: kernel compilation or execution failed
- `4`: no measurement results returned
- `5`: returned shot count differs from requested shot count
- `6`: Bell-state correlation is below threshold

## What this test proves — and does not prove

A pass demonstrates a small Bell-state circuit ran on the selected CUDA-Q
target and returned correlated measurements. It is not a performance benchmark,
a quantum-advantage result, a hardware qualification, or a test of the wider
training pipeline.

For a complete environment validation, record the CUDA-Q version, selected
target, Python version, operating system, NVIDIA driver, CUDA version, and GPU
model (when applicable) alongside the test output. Repeat on each intended
target. GPU simulator and QPU tests require their respective supported runtime
and access; a CPU simulator pass is not a substitute.
