<!---
Copyright 2026 The EGen Team. All rights reserved.

Licensed under the MIT License.
-->
# `tests/training/`

Unit tests for training utilities.

## Why this folder exists

Training has additional concerns beyond inference correctness (loss composition, auxiliary terms, STE correctness). These tests keep training behavior stable.

## What’s tested

- `test_losses.py`
  - `THLLoss` base loss and optional auxiliary terms.
- `test_diagnostics.py`
  - Training diagnostics logging.
- `test_straight_through.py`
  - Straight-through Top-K forward/backward behavior.
