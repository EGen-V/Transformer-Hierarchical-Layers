<!---
Copyright 2026 EGen Team. All rights reserved.

Licensed under the MIT License.
-->

<div align="center">
    <img src="docs/assets/banner.png" alt="THL Banner" width="100%"/>
</div>
<br>

<p align="center">
    <img src="https://img.shields.io/badge/python-3.8+-blue.svg" alt="Python Version">
    <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License">
    <img src="https://img.shields.io/badge/vram-4GB-orange.svg" alt="VRAM Optimized">
    <a href="https://github.com/EGen-V/Transformer-Hierarchical-Layers/actions"><img src="https://github.com/EGen-V/Transformer-Hierarchical-Layers/workflows/Tests/badge.svg" alt="Tests"></a>
</p>

<h1 align="center">🤗 THL: Transformer Hierarchical Layers</h1>

<p align="center">
    <a href="docs/lang/README_AR.md">العربية</a> |
    <a href="README.md">English</a> |
    <a href="docs/lang/README_ES.md">Español</a> |
    <a href="docs/lang/README_FR.md">Français</a> |
    <a href="docs/lang/README_zh-hans.md">简体中文</a>
</p>

<h3 align="center">
    State-of-the-art Hierarchical Recurrent Models for Low-Resource Hardware
</h3>

<p align="center">
    THL is a strictly non-Transformer, hierarchical recurrent computation graph designed to run large language models on <b>4GB VRAM</b> and mobile devices.
</p>

---

**THL** solves the specific problem of **KV cache memory explosion** in Transformers by using **Sequence-Length Independent Memory** (O(1) memory per layer). It achieves Transformer-competitive performance (approx.) while enabling inference on consumer hardware.

## ⚡ Why Use THL?

1.  **Bounded Memory (O(1))**: Forget about O(T) KV cache. THL uses fixed-slot memory (`J=1024`), allowing infinite context length generation without crashing your GPU.
2.  **Hierarchical Recurrence**: Multi-timescale GRU tiers process information at different frequencies ($\tau_k$), capturing both local syntax and global semantics efficiently.
3.  **Low VRAM Inference**: Built-in **Layered Inference Engine** allows running 7B+ parameter models on <4GB VRAM.
4.  **Sparse Routing**: Multi-head Top-K routing ensures relevant memories are accessed without processing the entire history.

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/EGen-V/Transformer-Hierarchical-Layers.git
cd Core

# Install dependencies
pip install -r requirements.txt
pip install .
```

## 🚀 Quick Tour

### 1. Basic Language Modeling

Easily instantiate a model and run a forward pass:

```python
import torch
from thl.config import THLConfig
from thl.model import THLModel

# Configure for 4GB VRAM
config = THLConfig(
    num_tiers=3,
    memory_slots=1024,
    dim=768
)

model = THLModel(config)
input_ids = torch.randint(0, 50257, (1, 32))
logits, state = model(input_ids)
```

### 2. Low-VRAM Generation (Streaming)

Run larger models by streaming layers to the GPU one by one:

```python
from thl.inference.layered import LayeredInferenceEngine
from thl.inference.state import InferenceState

engine = LayeredInferenceEngine(model, device="cuda")
state = InferenceState.init(1, config, model.tiers, model.memory_bank)

# Single token generation step
token = torch.tensor([123])
logit, state = engine.step(token, state)
```

## 🏗️ Architecture

| Component | Symbol | Description |
|-----------|--------|-------------|
| **Memory Bank** | $M_t$ | Fixed-size matrix ($J \times d$) holding long-term context. |
| **Sparse Router** | $r_t$ | Top-K routing mechanism to read relevant slots. |
| **Hierarchical Tiers** | $s_t^{(k)}$ | Stack of recurrent cells updating at exponential intervals $\tau=2^k$. |
| **Novelty Writer** | $w_t$ | Gated mechanism to write only novel information to memory. |

## 🧪 Verified Performance

We test THL rigorously. Run the suite yourself:
```bash
./scripts/run_tests.sh
```

## 📜 License

This project is licensed under the MIT License.
