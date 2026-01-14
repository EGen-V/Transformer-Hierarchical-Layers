<!---
Copyright 2026 The EGen Team. All rights reserved.

Licensed under the MIT License.
-->
# `thl/`

This is the core Python package implementing the Transformer Hierarchical Layers (THL) model.

## Why this folder exists

All model code lives under `thl/` so it can be imported as a standard library module (e.g. `from thl.model import THLModel`).

## Key files

- `config.py`
  - Central configuration dataclass (`THLConfig`) used across the model.
- `model.py`
  - Main model implementation (`THLModel`) and task heads.
- `tokenizer.py`
  - Minimal tokenizer stub used by tests/examples.

## Subpackages

- `memory/`
  - External memory bank + router + writer + metadata.
- `tiers/`
  - Hierarchical recurrent tier stack (clocked updates).
- `inference/`
  - Bounded inference state and layered inference engine (module streaming).
- `training/`
  - Training utilities (losses, STE for Top-K routing).
- `utils/`
  - Shared numerical / embedding / profiling helpers.
