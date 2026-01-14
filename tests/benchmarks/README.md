<!---
Copyright 2026 The EGen Team. All rights reserved.

Licensed under the MIT License.
-->
# `tests/benchmarks/`

Benchmark scripts for THL.

## Why this folder exists

These scripts are intended for manual profiling/measurement (not strict unit tests). They help validate boundedness claims and routing behavior under longer runs.

## Key scripts

- `benchmark_memory.py`
  - Measures peak memory usage vs sequence length (particularly for layered inference).
- `benchmark_routing.py`
  - Collects routing diagnostics over a run (entropy, slot usage, etc.).
