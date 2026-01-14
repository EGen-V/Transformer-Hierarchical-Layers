<!---
Copyright 2026 The EGen Team. All rights reserved.

Licensed under the MIT License.
-->
# `tests/tiers/`

Unit tests for tier computation.

## Why this folder exists

Tier update math and gating are core to THL’s recurrent dynamics. These tests validate that tier cells update when active and preserve state when clocked off.

## What’s tested

- `test_cell.py`
  - `TierCell` update behavior and shape invariants.
