<!---
Copyright 2026 The EGen Team. All rights reserved.

Licensed under the MIT License.
-->
# `thl/memory/`

This folder contains the THL external memory system.

## Why this folder exists

THL replaces Transformer KV-cache growth with a fixed-size, slot-indexed memory bank. This module isolates all memory logic (read routing, write/update policy, metadata) from the recurrent tiers.

## Key files

- `bank.py`
  - `MemoryBank`: maintains the slot tensor and applies decay + sparse writes.
- `router.py`
  - `SparseRouter`: multi-head Top-K routed reads from the memory bank.
  - Supports optional per-step slot capacity and optional load-balancing auxiliary statistics.
- `writer.py`
  - `MemoryWriter`: decides whether to write new information vs update existing slots using novelty-based logic.
- `metadata.py`
  - `BatchedMemoryMetadata`: tracks read/write timestamps and EMAs used for load penalty and LRU-style policies.
