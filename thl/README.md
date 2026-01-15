<!---
Copyright 2026 The EGen Team. All rights reserved.

Licensed under the MIT License.
--->
# 📦 `thl/` Core Package

This is the primary Python package for **Transformer Hierarchical Layers (THL)**. It implements a strictly non-Transformer, hierarchical recurrent architecture designed for high-performance sequence modeling with constant memory complexity.

## 🚀 Package Overview

The `thl/` directory is structured to separate different aspects of the hierarchical recurrent graph:

### 🧠 Core Model
- **[`model.py`](./model.py)**: The main `THLModel` implementation. It integrates the embedding layer, memory system, tier stack, and output heads. Also includes task-specific wrappers like `THLForSequenceClassification`.
- **[`config.py`](./config.py)**: `THLConfig` dataclass for centralizing all model hyperparameters (dimensions, number of tiers, timescales, memory slots).

### 📖 Subsystems
- **[`memory/`](./memory/)**: The sequence-length independent memory system.
  - `bank.py`: Storage management for memory slots.
  - `router.py`: Sparse multi-head attention-like routing for memory access.
  - `writer.py`: Novelty-gated memory update logic.
- **[`tiers/`](./tiers/)**: The hierarchical computation engine.
  - `stack.py`: Manages the stack of recurrent cells.
  - `cell.py`: Specialized recurrent units (Clocked-GRUs).
  - `clock.py`: Logic for hierarchical timescale updates.

### 🛠️ Infrastructure
- **[`inference/`](./inference/)**: Tools for running models under extreme constraints. Includes the **Layered Inference Engine** for module-by-module streaming.
- **[`training/`](./training/)**: Utilities for end-to-end training, including Straight-Through Estimators (STE) for discrete routing.
- **[`utils/`](./utils/)**: Shared numerical utilities, positional embeddings, and routing diagnostics.
- **[`tokenizer.py`](./tokenizer.py)**: A robust, fallback-safe byte-level tokenizer.

## 📥 Installation

```bash
pip install thl
```

## 🧪 Quick Usage

```python
from thl import THLModel, THLConfig

config = THLConfig(num_tiers=3, memory_slots=1024)
model = THLModel(config)
```
