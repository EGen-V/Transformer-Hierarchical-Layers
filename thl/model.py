import torch
import torch.nn as nn
import math
from typing import Tuple, Optional

from thl.config import THLConfig
from thl.memory.bank import MemoryBank
from thl.memory.router import SparseRouter
from thl.memory.writer import MemoryWriter
from thl.tiers.stack import HierarchicalTierStack
from thl.inference.state import InferenceState
from thl.utils.profiling import RoutingDiagnostics


class LocalSlidingWindowAttention(nn.Module):
    def __init__(self, config: THLConfig):
        super().__init__()
        self.window = int(getattr(config, "local_window", 0))
        self.q = nn.Linear(config.embedding_dim, config.query_dim, bias=False)
        self.k = nn.Linear(config.embedding_dim, config.query_dim, bias=False)
        self.v = nn.Linear(config.embedding_dim, config.value_dim, bias=False)

    def forward(self, e_t: torch.Tensor, buffer: torch.Tensor, valid: int) -> torch.Tensor:
        W = buffer.size(1)
        if W == 0 or valid <= 0:
            return torch.zeros(e_t.size(0), self.v.out_features, device=e_t.device, dtype=e_t.dtype)

        v = min(int(valid), int(W))
        buf = buffer[:, :v, :].to(device=e_t.device, dtype=e_t.dtype)

        q = self.q(e_t)
        k = self.k(buf)
        scores = torch.einsum("btd,bd->bt", k, q) / math.sqrt(k.size(-1))
        attn = torch.softmax(scores, dim=-1)

        values = self.v(buf)
        return torch.bmm(attn.unsqueeze(1), values).squeeze(1)

    def update_buffer(self, buffer: torch.Tensor, valid: int, e_t: torch.Tensor) -> Tuple[torch.Tensor, int]:
        W = buffer.size(1)
        if W == 0:
            return buffer, 0

        e_t = e_t.to(device=buffer.device, dtype=buffer.dtype)
        if valid < W:
            new_buffer = buffer.clone()
            new_buffer[:, valid, :] = e_t
            return new_buffer, valid + 1

        return torch.cat([buffer[:, 1:, :], e_t.unsqueeze(1)], dim=1), valid

class THLModel(nn.Module):
    """
    Transformer Hierarchical Layers (THL) Model.
    Strict non-Transformer recurrent architecture.
    """
    def __init__(self, config: THLConfig):
        super().__init__()
        self.config = config
        
        # 1. Embedding
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim)

        self.local_attn = LocalSlidingWindowAttention(config)
        
        # 2. Memory System
        self.memory_bank = MemoryBank(config)
        self.router = SparseRouter(config)
        self.writer = MemoryWriter(config)
        
        # 3. Hierarchical Tiers
        self.tiers = HierarchicalTierStack(config)
        
        # 4. Output Head
        # Concatenate all tier states for prediction? 
        # Spec says: "h_t = G([s_t^(0); ...; s_t^(K-1)])"
        total_tier_dim = sum(config.tier_dims)
        self.output_head = nn.Linear(total_tier_dim, config.output_dim)
        
        # Layer Normalization for stability
        self.layer_norm = nn.LayerNorm(total_tier_dim)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """
        Custom initialization for better convergence.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        
    def forward_features_step(self, 
                              token_id: torch.Tensor, 
                              state: InferenceState,
                              diagnostics: Optional[RoutingDiagnostics] = None) -> Tuple[torch.Tensor, InferenceState]:
        """
        Computes the hidden states/features for a single step, WITHOUT the output head.
        Returns:
            normalized_states: [B, total_tier_dim]
            new_state: Updated InferenceState
        """
        # 1. Embed
        e_t = self.embedding(token_id) # [B, d_e]

        local_read = self.local_attn(e_t, state.local_buffer, state.local_valid)
        state.local_buffer, state.local_valid = self.local_attn.update_buffer(state.local_buffer, state.local_valid, e_t)
        
        # 2. Router Read
        # u_t = [e_t; s_{t-1}^(0); s_{t-1}^(K-1)]
        s_prev_0 = state.tier_states[0]
        s_prev_K = state.tier_states[-1]
        u_t = torch.cat([e_t, s_prev_0, s_prev_K], dim=-1)
        
        r_t, alpha, read_indices = self.router(
            u_t, 
            state.memory_state, 
            state.memory_metadata.read_ema
        )
        
        if diagnostics:
            diagnostics.update(alpha, read_indices)
            
        # Update Read Metadata
        state.memory_metadata.update_read(state.timestep, alpha, read_indices)
        
        # 3. Update Tiers
        new_tier_states = self.tiers(
            e_t, 
            r_t, 
            state.tier_states, 
            state.timestep,
            tier0_memory_read=r_t + local_read
        )
        
        # 4. Memory Write
        # w_t, g_t, write_indices
        s_new_0 = new_tier_states[0]
        s_new_K = new_tier_states[-1]
        
        staleness = state.memory_metadata.get_staleness(state.timestep)
        underuse = state.memory_metadata.get_underuse()
        
        w_t, g_t, write_indices = self.writer(
            s_new_0, s_new_K, r_t, staleness, underuse, state.memory_state
        )
        
        # Update Memory Bank
        new_memory_state = self.memory_bank(
            state.memory_state, 
            write_indices, 
            w_t, 
            g_t
        )
        
        # Update Write Metadata
        state.memory_metadata.update_write(state.timestep, write_indices)
        
        # 5. Feature Output
        concat_states = torch.cat(new_tier_states, dim=-1)
        normalized_states = self.layer_norm(concat_states)
        
        # Update State Object
        state.tier_states = new_tier_states
        state.memory_state = new_memory_state
        state.timestep += 1
        
        return normalized_states, state

    def forward_step(self, 
                     token_id: torch.Tensor, 
                     state: InferenceState,
                     diagnostics: Optional[RoutingDiagnostics] = None) -> Tuple[torch.Tensor, InferenceState]:
        """
        Single step forward pass (end-to-end with head).
        """
        features, state = self.forward_features_step(token_id, state, diagnostics)
        logits = self.output_head(features)
        return logits, state

    @classmethod
    def from_pretrained(cls, path_or_hub_id: str, device: str = "cpu", cache_dir: Optional[str] = None) -> 'THLModel':
        import json
        import os
        
        # If path is local directory
        if os.path.exists(path_or_hub_id):
            base_path = path_or_hub_id
        else:
            if cache_dir:
                os.makedirs(cache_dir, exist_ok=True)
                base_path = os.path.join(cache_dir, path_or_hub_id)
            else:
                 base_path = path_or_hub_id
                 
            if not os.path.exists(base_path):
                 from thl.errors import THLRuntimeError
                 raise THLRuntimeError(f"Model path {base_path} does not exist.", hint="Check the path or use a valid Hub ID.")

        # Load config
        with open(os.path.join(base_path, "config.json"), "r") as f:
            config_dict = json.load(f)
        
        config = THLConfig(**config_dict)
        config.device = device
        
        model = cls(config)
        
        # Load weights
        state_dict = torch.load(os.path.join(base_path, "pytorch_model.bin"), map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        
        return model

    def save_pretrained(self, path: str):
        import json
        import os
        os.makedirs(path, exist_ok=True)
        
        # Save config
        from dataclasses import asdict
        config_dict = asdict(self.config)
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(config_dict, f, indent=2)
            
        # Save weights
        torch.save(self.state_dict(), os.path.join(path, "pytorch_model.bin"))

    def forward(self, input_ids: torch.Tensor, state: Optional[InferenceState] = None):
        """
        Sequence processing (mainly for training or batch inference).
        For T steps, we loop T times.
        input_ids: [B, T]
        """
        batch_size, seq_len = input_ids.shape
        if state is None:
            state = InferenceState.init(batch_size, self.config, self.tiers, self.memory_bank)
            
        logits_list = []
        
        for t in range(seq_len):
            token_ids = input_ids[:, t]
            logits, state = self.forward_step(token_ids, state)
            logits_list.append(logits.unsqueeze(1))
            
        return torch.cat(logits_list, dim=1), state


class THLForSequenceClassification(nn.Module):
    def __init__(self, config: THLConfig, num_labels: int = 2):
        super().__init__()
        self.num_labels = num_labels
        self.thl = THLModel(config)
        # Remove language modeling head if possible or ignore it?
        # Usually we use the final state.
        # THLModel returns logits. We need internal state? 
        # THLModel.forward returns (logits, state).
        # We can us state.tier_states[-1] as the sequence representation.
        
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(config.tier_dims[-1], num_labels)
        
    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None):
        outputs, state = self.thl(input_ids)
        # Use final state of top tier as pooling
        pooled_output = state.tier_states[-1] 
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            
        return logits, loss

class THLForMultipleChoice(nn.Module):
    def __init__(self, config: THLConfig):
        super().__init__()
        self.thl = THLModel(config)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(config.tier_dims[-1], 1)
        
    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None):
        # input_ids: [B, NumChoices, SeqLen]
        batch_size, num_choices, seq_len = input_ids.shape
        flat_input_ids = input_ids.view(-1, seq_len)
        
        outputs, state = self.thl(flat_input_ids)
        pooled_output = state.tier_states[-1]
        
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output) # [B*N, 1]
        reshaped_logits = logits.view(batch_size, num_choices)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
            
        return reshaped_logits, loss

class THLForTokenClassification(nn.Module):
    def __init__(self, config: THLConfig, num_labels: int = 2):
        super().__init__()
        self.num_labels = num_labels
        self.thl = THLModel(config)
        self.dropout = nn.Dropout(0.1)
        # Replaces standard output head with specific classifier
        # Note: THLModel already has an output head. 
        # Ideally refactor THLModel to have optional head.
        # For now, we project from the hidden stats used by the head (concat of tiers)
        
        self.classifier = nn.Linear(sum(config.tier_dims), num_labels)
        
    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None):
        # We need the per-token hidden states.
        # THLModel returns logits. Modifications needed to return hidden states.
        # Currently THLModel.forward_step returns logits, state.
        # The logits are from thl.output_head(concat_states).
        # We can hack this by replacing output_head or accessing internals.
        # Better: Refactor THLModel to return hidden states if requested.
        # For this snippet, I will modify THLModel to allow returning hidden states.
        
        # Assume THLModel modified to return hidden_states
        # Or, we just use the logic here.
        
        # To avoid refactoring base class heavily in replacement chunk, 
        # I will iterate step by step here using forward_step.
        
        batch_size, seq_len = input_ids.shape
        state = InferenceState.init(batch_size, self.thl.config, self.thl.tiers, self.thl.memory_bank)
        
        logits_list = []
        
        for t in range(seq_len):
            token_ids = input_ids[:, t]
            
            # Use shared feature extraction logic
            features, state = self.thl.forward_features_step(token_ids, state)
            
            # Project to labels
            token_logits = self.classifier(self.dropout(features))
            logits_list.append(token_logits.unsqueeze(1))
            
        logits = torch.cat(logits_list, dim=1) # [B, T, num_labels]
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            # Only keep active part of loss
            if self.thl.config.pad_token_id is not None:
                active_loss = labels.view(-1) != -100 # Standard ignore index
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                
        return logits, loss
