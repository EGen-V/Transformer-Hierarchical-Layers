<!---
Copyright 2026 The EGen Team. All rights reserved.

Licensed under the MIT License.
-->
# `tests/`

This folder contains the automated test suite for THL.

## Why this folder exists

THL has several interacting subsystems (routing, memory metadata, tier clocks, streaming inference). Tests ensure:

- Correct tensor shapes and invariants
- Bounded state behavior over time
- Deterministic behavior for critical policies (e.g. write/update)
- Integration of subsystems in end-to-end forward passes

## Structure

- `conftest.py`
  - Shared pytest fixtures (e.g. small `THLConfig` for fast tests).
- `memory/`
  - Unit tests for memory bank/router/writer.
- `tiers/`
  - Unit tests for tier cells/clock behavior.
- `training/`
  - Unit tests for losses, diagnostics, and straight-through routing helpers.
- `inference/`
  - Tests for bounded inference state and layered inference engine.
- `integration/`
  - End-to-end tests for forward pass and task heads.
- `benchmarks/`
  - Optional benchmark scripts (not strict unit tests).
- `graphs/`
  - Helpers for plotting benchmark outputs.
