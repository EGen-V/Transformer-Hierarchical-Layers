<!---
Copyright 2026 The EGen Team. All rights reserved.

Licensed under the MIT License.
-->
# `thl/tiers/`

This folder implements the hierarchical recurrent computation graph (“tiers”).

## Why this folder exists

Instead of stacking many Transformer blocks, THL uses a small number of recurrent tiers operating at different timescales. This isolates tier update logic and scheduling from the rest of the model.

## Key files

- `cell.py`
  - `TierCell`: GRU-like recurrent update that consumes lower-tier input and a memory read.
- `clock.py`
  - `TierClock`: determines which tiers update at each timestep.
- `stack.py`
  - `HierarchicalTierStack`: orchestrates tier updates and passes signals bottom-up.
