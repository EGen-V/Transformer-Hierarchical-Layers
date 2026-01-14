<!---
Copyright 2026 EGen Team. All rights reserved.

Licensed under the MIT License.
-->

<div align="center">
    <img src="../../docs/assets/banner.png" alt="THL Banner" width="100%"/>
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
    <a href="README_AR.md">العربية</a> |
    <a href="../../README.md">English</a> |
    <a href="README_ES.md">Español</a> |
    <a href="README_FR.md">Français</a> |
    <a href="README_zh-hans.md">简体中文</a>
</p>

<h3 align="center">
    面向低资源硬件的最先进分层循环模型
</h3>

<p align="center">
    THL 是一个严格非 Transformer 的分层循环计算图，专为在 <b>4GB 显存</b> 和移动设备上运行大型语言模型而设计。
</p>

---

**THL** 通过使用 **序列长度无关内存**（每层 O(1) 内存）解决了 Transformers 中 **KV 缓存显存爆炸** 的具体问题。它在实现与 Transformer 相当的性能的同时，使得在消费级硬件上进行推理成为可能。

## ⚡ 为什么选择 THL？

1.  **受限内存 (O(1))**：忘掉 O(T) 的 KV 缓存吧。THL 使用固定槽位内存 (`J=1024`)，允许无限上下文长度生成而不会导致 GPU 崩溃。
2.  **分层循环**：多时间尺度 GRU 层级以不同频率 ($\tau_k$) 处理信息，有效捕捉局部语法和全局语义。
3.  **低显存推理**：内置的 **分层推理引擎** 允许在小于 4GB 显存上运行 7B+ 参数的模型。
4.  **稀疏路由**：多头 Top-K 路由确保仅访问相关的记忆，而无需处理整个历史记录。

## 🛠️ 安装

```bash
# 克隆仓库
git clone https://github.com/EGen-V/Transformer-Hierarchical-Layers.git
cd Core

# 安装依赖
pip install -r requirements.txt
pip install .
```

## 🚀 快速入门

### 1. 基础语言建模

轻松实例化模型并运行前向传播：

```python
import torch
from thl.config import THLConfig
from thl.model import THLModel

# 配置为 4GB 显存
config = THLConfig(
    num_tiers=3,
    memory_slots=1024,
    dim=768
)

model = THLModel(config)
input_ids = torch.randint(0, 50257, (1, 32))
logits, state = model(input_ids)
```

### 2. 低显存生成 (流式)

通过逐层流式传输到 GPU 来运行更大的模型：

```python
from thl.inference.layered import LayeredInferenceEngine
from thl.inference.state import InferenceState

engine = LayeredInferenceEngine(model, device="cuda")
state = InferenceState.init(1, config, model.tiers, model.memory_bank)

# 单个 token 生成步骤
token = torch.tensor([123])
logit, state = engine.step(token, state)
```

## 🏗️ 架构

| 组件 | 符号 | 描述 |
|-----------|---|-------------|
| **记忆库** | $M_t$ | 固定大小矩阵 ($J \times d$) 保存长期上下文。 |
| **稀疏路由器** | $r_t$ | 用于读取相关槽位的 Top-K 路由机制。 |
| **分层层级** | $s_t^{(k)}$ | 按指数间隔 $\tau=2^k$ 更新的循环单元堆栈。 |
| **新奇写入器** | $w_t$ | 仅将新信息写入内存的门控机制。 |

## 🧪 验证性能

我们严格测试 THL。您可以自己运行测试套件：
```bash
./scripts/run_tests.sh
```

## 📜 许可证

本项目采用 MIT 许可证。
