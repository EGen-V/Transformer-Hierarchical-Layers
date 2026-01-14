<!---
Copyright 2026 The EGen Team. All rights reserved.

Licensed under the MIT License.
-->
# `thl/training/`

This folder contains training-time utilities for THL.

## Why this folder exists

Some components (notably Top-K routing) require special handling for training vs inference. Training also combines multiple loss terms (e.g. optional routing regularizers).

## Key files

- `straight_through.py`
  - Helpers for Top-K routing.
  - Includes `DifferentiableRouter` logic using Gumbel-Softmax for end-to-end training.
- `losses.py`
  - `THLLoss`: task loss plus optional auxiliary terms.
- `diagnostics.py`
  - Training diagnostics helpers (loss logging on top of routing diagnostics).
