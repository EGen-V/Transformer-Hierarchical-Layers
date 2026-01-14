# THL Architecture Specification

**Goal**: Reproduce and exceed Transformer capabilities using a strictly non-Transformer, hierarchical recurrent architecture (THL) with O(1) memory per layer.

## 1. Core Philosophy
To compete with Transformers, THL must solve:
1.  **Expressive Interactions**: Approximate global attention via multi-head, sparse routing.
2.  **Stable Optimization**: Deep residuals, pre-LN, and gradient hygiene.
3.  **Emergent Hierarchy**: Multi-timescale tiers capturing local-to-global abstractions.
4.  **Transferability**: Unsupervised pretraining (next-token) -> downstream fine-tuning.

## 2. Component Specifications

### 2.1. Memory Bank ($M_t$)
-   **Structure**: Fixed set of $J$ slots (e.g., $J=1024$) of dimension $d$.
-   **Lifecycle**: Slots persist across the entire sequence (and potentially across documents via streaming).
-   **Aging**: Each slot tracks `last_read`, `last_write`, `read_ema`, and `write_ema` to guide replacement.

### 2.2. Sparse Router ($r_t$)
-   **Function**: Selects $K$ slots to read from $J$.
-   **Mechanism**:
    -   **Inference**: Hard Top-K selection using affinity scores $q \cdot k$.
    -   **Training**: Gumbel-Softmax or Differentiable Sparsemax to allow gradient flow through slot selection.
-   **Multi-Head**: Multiple heads route to different subsets of slots to capture diverse context.
-   **Load Balancing**: Auxiliary loss penalizes slot over/under-usage: $\mathcal{L}_{bal} = \lambda \sum (u_j - \bar{u})^2$.

### 2.3. Novelty Writer ($w_t$)
-   **Goal**: Write only "novel" information to memory to conserve capacity.
-   **Novelty Score**: $1 - \max(\text{affinity}(q, M_t))$. usage of reconstruction error.
-   **Policy**:
    -   Update existing slots if affinity > threshold (refinement).
    -   Overwrite least-useful slots (stalest / lowest read-utility) if novelty is high.
-   **Gating**: Information flow is gated: $m_{new} = m_{old} + \eta \cdot g_t \cdot w_t$.

### 2.4. Hierarchical Tiers ($s_t^{(k)}$)
-   **Structure**: Stack of recurrent cells (e.g., GRU/LSTM variants) running at timescales $\tau_k = 2^k$.
-   **Connectivity**:
    -   Bottom-up: Fast tiers feed slow tiers.
    -   Top-down: Slow tiers condition fast tiers.
    -   **Local Attention**: Tier 0 (token level) uses a small sliding window attention ($W \approx 64$) to capture immediate syntax cheaply.
-   **Normalization**: Pre-LayerNorm is standard. Residual connections $x + F(x)$ everywhere.

## 3. Training Recipe

### 3.1. Objectives
1.  **Causal Language Modeling (CLM)**: Minimize NLL of next token.
2.  **Load Balancing**: Router auxiliary loss.
3.  **Slot Prediction (Optional)**: Mask a slot and reconstruct it to stabilize memory semantics.

### 3.2. Optimization
-   **Optimizer**: AdamW.
-   **Schedule**: Linear warmup + Cosine decay.
-   **Precision**: Mixed precision (BF16/FP16).

### 3.3. Curriculum
-   Start with short sequences to train local tiers.
-   Gradually increase sequence length to train router and long-term memory.

## 4. Evaluation Strategy
-   **Perplexity**: Standard LM benchmark (WikiText-103, C4).
-   **Long-Context**: Passkey retrieval, long-document QA.
-   **Ablations**:
    -   Hard vs Soft Routing.
    -   Slot count $J$ sensitivity.
    -   Local Attention impact.
