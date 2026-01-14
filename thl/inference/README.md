<!---
Copyright 2026 The EGen Team. All rights reserved.

Licensed under the MIT License.
-->
# `thl/inference/`

This folder contains inference-time utilities focused on bounded state and low-VRAM execution.

## Why this folder exists

THL is designed for streaming / long-context inference with a strictly bounded recurrent state. The code here makes that explicit and provides an inference engine that can swap weights to fit small GPUs.

## Key files

- `state.py`
  - `InferenceState`: the bounded recurrent state (tiers + memory + metadata + Tier-0 local attention buffer).
- `layered.py`
  - `LayeredInferenceEngine`: runs a step by streaming modules onto the device one at a time.
- `streaming.py`
  - `ModuleStreamer`: small helper to move modules between CPU/GPU during layered inference.
