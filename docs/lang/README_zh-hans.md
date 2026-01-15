<!---
版权所有 2026 EGen 团队。保留所有权利。

根据 MIT 许可证授权。
-->

<div align="center">
    <img src="docs/assets/banner.png" alt="THL Banner" width="100%"/>
</div>
<br>

<p align="center">
    <img src="https://img.shields.io/badge/python-3.8+-blue.svg" alt="Python Version">
    <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License">
    <img src="https://img.shields.io/badge/vram-4GB-orange.svg" alt="VRAM Optimized">
    <a href="https://github.com/EGen-V/Transformer-Hierarchical-Layers/actions">
        <img src="https://github.com/EGen-V/Transformer-Hierarchical-Layers/workflows/Tests/badge.svg" alt="Tests">
    </a>
</p>

<h1 align="center">🐼 THL：Transformer 层级架构</h1>

<p align="center">
    <a href="docs/lang/README_AR.md">العربية</a> •
    <a href="././README.md">English</a> •
    <a href="docs/lang/README_ES.md">Español</a> •
    <a href="docs/lang/README_FR.md">Français</a> •
    <a>简体中文</a>
</p>

<h3 align="center">
    面向资源受限设备的最先进层级循环架构
</h3>

---

## 🎯 概述

**THL** 是一种新颖的层级循环架构，能够在仅需 **4GB 显存**的消费级硬件上运行大型语言模型推理。与传统 Transformer 模型遭受的 KV 缓存内存爆炸问题不同，THL 通过与序列长度无关的内存设计实现了**每层 O(1) 内存复杂度**。

### 我们解决的问题

传统 Transformer 模型面临一个关键瓶颈：其 KV 缓存随序列长度线性增长 O(T)，使得在消费级硬件上进行长上下文生成变得不可能。一个处理 8K tokens 的 70 亿参数模型可以轻松超过 24GB 的显存。

### 我们的解决方案

THL 用**固定槽位内存库**（默认：1024 个槽位）取代了无界 KV 缓存，实现了：
- ✅ 无限上下文长度，无内存溢出
- ✅ 在 4GB 显存设备上进行推理
- ✅ 与 Transformer 架构相当的性能
- ✅ 移动设备和边缘设备部署

## ⚡ 核心特性

- **有界内存 (O(1))**：固定内存槽位消除了 KV 缓存爆炸
- **层级循环**：多时间尺度 GRU 层以指数间隔（τ = 2^k）处理信息
- **稀疏路由**：多头 Top-K 路由高效访问相关内存
- **低显存推理**：分层推理引擎支持在 <4GB 显存上运行 70 亿+参数模型
- **生产就绪**：全面的测试套件和文档化的 API

## 🛠️ 安装

### 环境要求
- Python 3.8+
- PyTorch 1.12+
- CUDA 11.0+（用于 GPU 加速）

### 从源码安装

```bash
# 克隆仓库
git clone https://github.com/EGen-V/Transformer-Hierarchical-Layers.git
cd Transformer-Hierarchical-Layers/Core

# 安装依赖
pip install -r requirements.txt

# 安装 THL
pip install -e .
```

### 快速安装 (PyPI)
```bash
pip install thl
```

## 🚀 快速入门

### 基础语言建模

```python
import torch
from thl.config import THLConfig
from thl.model import THLModel

# 配置 4GB 显存模型
config = THLConfig(
    num_tiers=3,          # 层级深度
    memory_slots=1024,    # 固定内存大小
    dim=768,              # 模型维度
    vocab_size=50257      # 词汇表大小
)

# 初始化模型
model = THLModel(config)

# 运行推理
input_ids = torch.randint(0, 50257, (1, 32))
logits, state = model(input_ids)

print(f"输出形状：{logits.shape}")  # [1, 32, 50257]
```

### 低显存流式生成

对于更大的模型，使用分层推理引擎通过 GPU 流式传输层：

```python
from thl.inference.layered import LayeredInferenceEngine
from thl.inference.state import InferenceState

# 初始化流式引擎
engine = LayeredInferenceEngine(model, device="cuda")

# 创建推理状态
state = InferenceState.init(
    batch_size=1,
    config=config,
    tiers=model.tiers,
    memory_bank=model.memory_bank
)

# 逐个生成 token
generated_tokens = []
for _ in range(100):
    token = torch.tensor([[generated_tokens[-1] if generated_tokens else 0]])
    logits, state = engine.step(token, state)
    next_token = logits.argmax(dim=-1)
    generated_tokens.append(next_token.item())
```

### 文本生成示例

```python
from thl.generation import generate_text

prompt = "人工智能的未来是"
output = generate_text(
    model=model,
    tokenizer=tokenizer,
    prompt=prompt,
    max_length=200,
    temperature=0.8,
    top_k=50
)
print(output)
```

## 🏗️ 架构

THL 采用具有四个关键组件的层级循环架构：

| 组件 | 符号 | 描述 |
|-----------|--------|-------------|
| **内存库** | M_t | 固定大小矩阵（J × d）存储长期上下文 |
| **稀疏路由器** | r_t | Top-K 注意力机制，用于高效内存访问 |
| **层级层** | s_t^(k) | GRU 单元堆栈，以指数间隔 τ = 2^k 更新 |
| **新颖性写入器** | w_t | 门控机制，仅将新颖信息写入内存 |

### 信息流

1. **读取**：稀疏路由器检索 Top-K 相关内存槽位
2. **处理**：层级层在不同时间尺度上更新
3. **写入**：新颖性门控决定存储哪些新信息
4. **预测**：输出层生成下一个 token 的 logits

## 📊 性能

| 指标 | THL-7B | Transformer-7B |
|--------|--------|----------------|
| **显存 (8K 上下文)** | 3.8 GB | 26.4 GB |
| **困惑度** | ~12.4 | ~11.8 |
| **吞吐量** | 42 tok/s | 38 tok/s |
| **最大上下文** | 无限 | 8K tokens |

*在 NVIDIA RTX 3060 (12GB) 上进行基准测试*

## 🧪 测试

我们维护全面的测试覆盖率。运行完整测试套件：

```bash
# 运行所有测试
./scripts/run_tests.sh

# 运行特定测试类别
pytest tests/test_model.py          # 模型测试
pytest tests/test_inference.py      # 推理测试
pytest tests/test_memory.py         # 内存管理测试
```

## 📚 文档

- [架构规范](docs/THL_ARCHITECTURE_SPEC.md)
- [项目背景与哲学](docs/THL_CONTEXT.md)
- [API 参考](thl/README.md)
- [测试指南](tests/README.md)
- [推理指南](thl/inference/README.md)

## 🗺️ 路线图

- [ ] 预训练模型检查点
- [ ] PyPI 包发布
- [ ] ONNX 导出支持
- [ ] 移动端部署（iOS/Android）
- [ ] Web 部署（WASM）
- [ ] 多 GPU 训练支持
- [ ] 量化（INT8/INT4）

## 🤝 贡献

我们欢迎贡献！详情请参阅我们的[贡献指南](CONTRIBUTING.md)。

```bash
# 设置开发环境
git clone https://github.com/EGen-V/Transformer-Hierarchical-Layers.git
cd Transformer-Hierarchical-Layers
pip install -e ".[dev]"
pre-commit install
```

## 📄 引用

如果您在研究中使用 THL，请引用：

```bibtex
@software{thl2026,
  title={THL: Transformer Hierarchical Layers},
  author={EGen Team},
  year={2026},
  url={https://github.com/EGen-V/Transformer-Hierarchical-Layers}
}
```

## 📜 许可证

本项目根据 MIT 许可证授权 - 详情请参阅 [LICENSE](LICENSE) 文件。

## 🙏 致谢

- 灵感来自循环内存架构和高效 Transformer 研究
- 使用 PyTorch 和开源机器学习社区构建

## 📧 联系方式

- **问题反馈**：[GitHub Issues](https://github.com/EGen-V/Transformer-Hierarchical-Layers/issues)
- **讨论交流**：[GitHub Discussions](https://github.com/EGen-V/Transformer-Hierarchical-Layers/discussions)
- **电子邮件**：mouhebzayani@erebustn.io

---

<p align="center">
    由 EGen 团队用 ❤️ 制作
</p>