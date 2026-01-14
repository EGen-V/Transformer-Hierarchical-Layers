# Transformer Hierarchical Layers (THL): Context & Philosophy

## 1. Philosophy: Democratizing Intelligence
The central philosophy of THL is **"Computation over Retrieval"**. 
Current Large Language Models (LLMs) based on Transformers suffer from the **KV Cache Bottleneck**: memory usage grows linearly with sequence length ($O(T)$). This restricts long-context reasoning to expensive datacenter-grade hardware (H100s), effectively centralizing AI capabilities.

THL is designed to break this dependency. By enforcing **bounded memory** ($O(1)$ w.r.t $T$), THL enables ostensibly "infinite" context processing on consumer hardware (4GB VRAM GPUs, Laptops, Mobile). We believe that true AI democratization requires architectures that are efficient *by design*, not just quantized after the fact.

## 2. Strategy: The Hierarchical Recurrent Graph
To achieve high performance without the massive memory footprint of Attention, THL employs a novel strategy:

### A. Sparse Routed Memory (vs. Dense Attention)
Instead of attending to *all* previous tokens (Transformer), THL routes information to a fixed set of **Memory Slots** ($J=1024$). 
- **Transformer**: Query $q_t$ attends to keys $k_{1...t-1}$. Cost: $O(t)$.
- **THL**: Query $q_t$ selects Top-K relevant slots from $M_{t-1}$. Cost: $O(K)$ (constant).
- **Result**: Inference speed and memory usage remain constant regardless of whether the context is 1k or 100k tokens.

THL also supports a **Tier-0 hybrid local attention** path: Tier 0 attends over a sliding window of the most recent `W` token embeddings and combines that local context with the routed memory read.

### B. Hierarchical Tiers (vs. Deep Layers)
THL organizes computation into **Tiers** that operate at different *timescales* ($\tau_k$).
- **Tier 0**: Fast, processes every token (syntax, local form).
- **Tier K**: Slow, updates every $\tau_K$ steps (semantics, long-term reasoning).
This mimics biological memory function—short-term sensory buffers feeding into long-term consolidation—allowing the model to reason over vast timescales efficiently.

### C. Layered Inference (vs. Model Parallelism)
Since the state is bounded, we can execute the model **layer-by-layer** even on a single GPU. We stream weights from RAM to VRAM, compute one layer, and swap. This allows running 7B+ parameter models on <4GB VRAM with minimal performance penalty compared to swap-heavy Transformers.

## 3. Project Structure
The codebase is organized to support this modular, inference-first approach.

### `thl/` - Core Package
- **`config.py`**: The "DNA" of the model. Defines dimensions ($d_{model}$, $J$ slots), tiered timescales ($\tau$), and routing hyperparameters.
- **`model.py`**: The central nervous system. Connects the Embedding, Router, Tiers, and Writer into a coherent forward pass.

### `thl/memory/` - The Memory System
- **`bank.py`**: The storage medium ($M_t$). Implements non-leaky decay and sparse updates.
- **`router.py`**: The retrieval mechanism. Uses dot-product affinity + load penalties to select the "right" memories.
- **`writer.py`**: The consolidation mechanism. Uses a **Novelty Gate** to decide *what* is worth remembering, preventing memory saturation.

### `thl/tiers/` - The Computation Engine
- **`stack.py`**: Manages the hierarchy. Passes signals up (bottom-up inference) and down (top-down bias?).
- **`cell.py`**: The actual processing unit (Modified GRU). Updates state only when its clock $\tau_k$ allows.
- **`clock.py`**: The scheduler. Determines which tiers are active at step $t$.

### `thl/inference/` - Deployment
- **`layered.py`**: The "AirLLM-style" engine. Manages weight splitting and streaming for extreme memory constraints.
- **`state.py`**: Defines the *exact* shape of the recurrent state, ensuring it never grows beyond the defined bounds (including the bounded Tier-0 local attention buffer).

### `thl/training/` - Learning
- **`straight_through.py`**: Implements the Straight-Through Estimator (STE) to allow backpropagation through the discrete Top-K routing decisions.
