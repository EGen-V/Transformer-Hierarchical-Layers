<!---
Copyright 2026 The EGen Team. All rights reserved.

Licensed under the MIT License.
--->
# 🛠️ `scripts/`

Utility scripts for repository maintenance, development workflows, and automated testing.

## 📜 Key Scripts

- **[`run_tests.sh`](./run_tests.sh)**: The primary entry point for the test suite. It handles environment setup, runs `pytest`, and ensures all critical subsystems (Memory, Tiers, Inference) are functional.

## 🚀 Usage

Most scripts are intended to be run from the repository root:

```bash
./scripts/run_tests.sh
```

---
*For contributing new scripts, please ensure they are documented here and follow the project's formatting guidelines.*
