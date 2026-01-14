<!---
Copyright 2026 The EGen Team. All rights reserved.

Licensed under the MIT License.
-->
# `tests/memory/`

Unit tests for the external memory system.

## Why this folder exists

Memory behavior is central to THL correctness and stability. These tests validate the invariants of read routing, metadata updates, and write/update policies.

## What’s tested

- `test_bank.py`
  - Memory decay and sparse write application.
- `test_router.py`
  - Multi-head Top-K routing shapes and normalization.
- `test_writer.py`
  - Novelty-based write vs update behavior and deterministic slot selection paths.
