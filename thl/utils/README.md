<!---
Copyright 2026 The EGen Team. All rights reserved.

Licensed under the MIT License.
-->
# `thl/utils/`

This folder contains small shared utilities used across the THL codebase.

## Why this folder exists

Keeps common math helpers and lightweight utilities out of model logic.

## Key files

- `numerical.py`
  - Numerically stable softmax variants and vector norm clipping.
- `embeddings.py`
  - Rotary positional embedding helpers (generic utilities).
- `profiling.py`
  - Routing diagnostics and lightweight profiling helpers.
