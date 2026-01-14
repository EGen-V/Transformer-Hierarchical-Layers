import torch
import torch.nn as nn
from typing import Tuple, Optional
from thl.model import THLModel
from thl.inference.state import InferenceState
from thl.inference.streaming import ModuleStreamer
from thl.utils.profiling import RoutingDiagnostics

class LayeredInferenceEngine:
    """
    Executes THLModel inference by streaming modules one by one.
    Reduces peak VRAM usage to max(single_module) + state.
    """
    def __init__(self, model: THLModel, device: str = "cuda"):
        self.model = model
        self.streamer = ModuleStreamer(device)
        self.device = device
        
        # Ensure model is initially on CPU
        self.model.to("cpu")
        
    def step(self, 
             token_id: torch.Tensor, 
             state: InferenceState,
             diagnostics: Optional[RoutingDiagnostics] = None) -> Tuple[torch.Tensor, InferenceState]:
        """
        Executes one forward step with layered execution.
        """
        # Move token to device
        token_id = token_id.to(self.device)
        
        # 1. Embedding
        with self.streamer.use_module(self.model.embedding):
            with torch.autocast(device_type=self.device, enabled=True): # Enable AMP
                e_t = self.model.embedding(token_id)

        with self.streamer.use_module(self.model.local_attn):
            local_buffer_gpu = state.local_buffer.to(self.device)
            with torch.autocast(device_type=self.device, enabled=True):
                local_read = self.model.local_attn(e_t, local_buffer_gpu, state.local_valid)
            local_buffer_gpu, local_valid = self.model.local_attn.update_buffer(local_buffer_gpu, state.local_valid, e_t)

        state.local_buffer = local_buffer_gpu
        state.local_valid = local_valid
        
        # 2. Router Read
        s_prev_0 = state.tier_states[0].to(self.device)
        s_prev_K = state.tier_states[-1].to(self.device)
        u_t = torch.cat([e_t, s_prev_0, s_prev_K], dim=-1)
        
        with self.streamer.use_module(self.model.router):
            # Move necessary state parts to device
            mem_state_gpu = state.memory_state.to(self.device)
            # read_ema is on CPU likely if state is CPU.
            read_ema_gpu = state.memory_metadata.read_ema.to(self.device)
            
            with torch.autocast(device_type=self.device, enabled=True):
                r_t, alpha, read_indices = self.model.router(
                    u_t, 
                    mem_state_gpu, 
                    read_ema_gpu
                )
            
        if diagnostics:
            diagnostics.update(alpha, read_indices)
            
        # Update Read Metadata (CPU update to save GPU mem? Or GPU?)
        state.memory_metadata.update_read(state.timestep, alpha, read_indices)
        
        # 3. Update Tiers
        new_tier_states = []
        current_lower_input = e_t
        active_flags = self.model.tiers.clock.get_active_tiers(state.timestep)
        
        for k in range(self.model.config.num_tiers):
            cell = self.model.tiers.cells[k]
            s_prev = state.tier_states[k].to(self.device)
            r_k = (r_t + local_read) if k == 0 else r_t
            
            with self.streamer.use_module(cell):
                with torch.autocast(device_type=self.device, enabled=True):
                    s_new = cell(s_prev, current_lower_input, r_k, active=active_flags[k])
                
            new_tier_states.append(s_new)
            current_lower_input = s_new
            
        # 4. Memory Write
        s_new_0 = new_tier_states[0]
        s_new_K = new_tier_states[-1]
        
        staleness = state.memory_metadata.get_staleness(state.timestep).to(self.device)
        underuse = state.memory_metadata.get_underuse().to(self.device)
        
        with self.streamer.use_module(self.model.writer):
            with torch.autocast(device_type=self.device, enabled=True):
                w_t, g_t, write_indices = self.model.writer(
                    s_new_0, s_new_K, r_t, staleness, underuse, mem_state_gpu
                )
        
        # Update Memory Bank (No weights here, just operation)
        # But we need to use 'self.model.memory_bank' if it had parameters (it doesn't, usually).
        # Actually MemoryBank is nn.Module but has no weights in our impl (just decay/config).
        # But for correctness:
        new_memory_state = self.model.memory_bank(
            mem_state_gpu, 
            write_indices, 
            w_t, 
            g_t
        )
        
        state.memory_metadata.update_write(state.timestep, write_indices)
        
        # 5. Output Head
        concat_states = torch.cat(new_tier_states, dim=-1)
        with self.streamer.use_module(self.model.output_head):
            with torch.autocast(device_type=self.device, enabled=True):
                logits = self.model.output_head(concat_states)
            
        # Update State
        # Move state tensors back to CPU to save GPU memory for next step's weights?
        # If we are extremely constrained: yes.
        # For now, let's keep them on device if they fit, or move to cpu if enforced.
        # Generally, "Layered Inference" means weights are swapped, activations/state stay on GPU (if they fit).
        # We update the state object with GPU tensors.
        
        state.tier_states = new_tier_states
        state.memory_state = new_memory_state
        state.timestep += 1
        
        return logits, state
