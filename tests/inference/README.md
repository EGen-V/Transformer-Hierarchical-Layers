<!---
Copyright 2026 The EGen Team. All rights reserved.

Licensed under the MIT License.
-->
# `tests/inference/`

Tests for inference-time utilities.

## Why this folder exists

Inference is where THL’s bounded-state guarantees matter most. These tests focus on boundedness, state evolution, and layered inference execution.

## What’s tested

- `test_state.py`
  - `InferenceState` initialization and bounded fields (including Tier-0 local attention buffer).
- `test_layered.py`
  - `LayeredInferenceEngine.step` produces correct logits/state updates.
- `test_streaming.py`
  - `ModuleStreamer` module move/restore behavior.
