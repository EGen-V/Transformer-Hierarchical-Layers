<!---
Copyright 2026 The EGen Team. All rights reserved.

Licensed under the MIT License.
--->
# 🧪 `tests/` Suite

This directory contains the automated test suite for the **THL** project. We prioritize high test coverage to ensure the stability of the hierarchical clocks, memory bank invariants, and routing policies.

## 🗂️ Test Categories

### 🔬 Unit Tests
- **[`memory/`](./memory/)**: Validates memory bank persistence, decay logic, and sparse router selection accuracy.
- **[`tiers/`](./tiers/)**: Ensures complex hierarchical clocks update correctly according to the specified timescales ($\tau_k = 2^k$).
- **[`training/`](./training/)**: Tests specific components like Gumbel-Softmax routing, loss functions, and straight-through estimators.
- **[`utils/`](./utils/)**: Verifies shared numerical helpers and profiling diagnostics.

### 🔗 Integration & System Tests
- **[`integration/`](./integration/)**: End-to-end forward pass tests across various configurations and task heads.
- **[`inference/`](./inference/)**: Validates the **Layered Inference Engine** and ensure the `InferenceState` remains bounded and correct over long sequences.

### 📊 Performance & Diagnostics
- **[`benchmarks/`](./benchmarks/)**: Scripts to measure memory consumption, throughput, and routing entropy.
- **[`graphs/`](./graphs/)**: Utilities to visualize model behavior and benchmark results.

## 🚀 Running Tests

We use `pytest` as our primary testing framework.

```bash
# Run the full suite
./scripts/run_tests.sh

# Or run with pytest directly
pytest tests/
```

## 🛠️ Configuration
Common test fixtures and small-scale `THLConfig` instances are defined in **[`conftest.py`](./conftest.py)** to keep unit tests fast and predictable.
