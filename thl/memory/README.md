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
  - `SparseRouter`: multi-head routed reads from the memory bank.
  - Supports **Gumbel-Softmax** for differentiable training and hard Top-K for inference.
  - Includes load-balancing penalties and entropy regularization terms.
- `writer.py`
  - `MemoryWriter`: handles memory consolidation.
  - Uses **Max-Affinity Novelty** gating to filter redundant information.
  - Implements LRU and utility-based slot replacement policies.
- `metadata.py`
  - `BatchedMemoryMetadata`: tracks read/write timestamps and EMAs used for load penalty and LRU-style policies.
