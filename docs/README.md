<!---
Copyright 2026 The EGen Team. All rights reserved.

Licensed under the MIT License.
--->
# 📑 THL Documentation

This directory contains the technical documentation, architecture specifications, and design philosophy for the **Transformer Hierarchical Layers (THL)** project.

## 🗂️ Contents

### 🏗️ Architecture & Philosophy
*   **[`THL_ARCHITECTURE_SPEC.md`](./THL_ARCHITECTURE_SPEC.md)**: Detailed technical specification of the model components (Memory Bank, Sparse Router, Hierarchical Tiers).
*   **[`THL_CONTEXT.md`](./THL_CONTEXT.md)**: The "Why" behind THL. Explains the transition from $O(T)$ attention to $O(1)$ memory and our goal of democratizing large-scale intelligence.

### 🌐 Multilingual Readmes
*   **[`lang/`](./lang/)**: Contains localized versions of the main project README for a global audience.
    *   [العربية (Arabic)](./lang/README_AR.md)
    *   [Español (Spanish)](./lang/README_ES.md)
    *   [Français (French)](./lang/README_FR.md)
    *   [简体中文 (Chinese)](./lang/README_zh-hans.md)

---
*For development-specific documentation, please refer to the README files in the respective code subdirectories (`thl/`, `tests/`, etc.).*
