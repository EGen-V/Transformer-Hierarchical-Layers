<!---
Copyright 2026 The EGen Team. All rights reserved.

Licensed under the MIT License.
-->
# `tests/integration/`

End-to-end integration tests.

## Why this folder exists

Integration tests ensure subsystems compose correctly: embedding → router → tiers → writer → memory bank → heads.

## What’s tested

- `test_forward_pass.py`
  - Full sequence forward and single-step forward, including bounded inference state fields.
- `test_model_heads.py`
  - Task heads (sequence classification, multiple choice, token classification) on top of `THLModel`.
