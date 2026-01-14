<!---
Copyright 2026 The EGen Team. All rights reserved.

Licensed under the MIT License.
-->
# `tests/utils/`

Unit tests for shared utilities.

## Why this folder exists

Numerical helper functions are used across routing/writing and need stable behavior.

## What’s tested

- `test_numerical.py`
  - Softmax helpers, masking correctness, and norm clipping invariants.
