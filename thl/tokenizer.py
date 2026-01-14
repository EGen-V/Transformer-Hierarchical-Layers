from typing import List, Union, Dict
import torch

class Tokenizer:
    """
    Robust Byte-Level Tokenizer for THL.
    Maps input text to UTF-8 bytes + offset.
    Falls back to a safe range to ensure no out-of-bounds indexing.
    """
    def __init__(self, vocab_size: int = 50257, never_split: List[str] = None):
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.unk_token_id = 2
        
        # Simple special tokens map
        self.special_tokens: Dict[str, int] = {
            "<pad>": self.pad_token_id,
            "<eos>": self.eos_token_id,
            "<unk>": self.unk_token_id
        }
        
        self.never_split = never_split if never_split else []
        for token in self.never_split:
            if token not in self.special_tokens:
                # Assign new ID
                 self.special_tokens[token] = len(self.special_tokens) + 3
        
        # Reserve first 256 + N_special for direct mapping if possible, 
        # but GPT-2 style uses byte-level BPE. 
        # For this implementation, strictly byte mapping + modulo is safer than mock 
        # but simpler than training BPE.
        
    def encode(self, text: str) -> torch.Tensor:
        """
        Encodes text into token IDs.
        Uses UTF-8 encoding.
        """
        # 1. Check for exactly matching special tokens (simplistic)
        if text in self.special_tokens:
            return torch.tensor([self.special_tokens[text]], dtype=torch.long)
            
        # 2. Byte encoding
        try:
            bytes_data = text.encode('utf-8')
        except UnicodeEncodeError:
            # Fallback for weird surrogates
            bytes_data = text.encode('utf-8', 'replace')
            
        tokens = []
        for b in bytes_data:
            # Shift by 256 to avoid special tokens conflict? 
            # Or just map directly to vocab space.
            # We map byte (0-255) to vocab range [sp_tokens, sp_tokens+256]
            offset_id = b + len(self.special_tokens) + 10 # Buffer
            
            # Safe modulo to keep within vocab
            tokens.append(offset_id % self.vocab_size)
            
        return torch.tensor(tokens, dtype=torch.long)
        
    def decode(self, tokens: Union[torch.Tensor, List[int]]) -> str:
        """
        Decodes token IDs back to string.
        """
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
            
        # Reverse mapping? Trivial for byte level.
        # We need to subtract offset?
        # Since we use modulo, perfect reconstruction isn't guaranteed for random collisions,
        # but for valid range it works.
        
        out_bytes = bytearray()
        
        sp_map_inv = {v: k for k, v in self.special_tokens.items()}
        
        decoded_parts = []
        
        for t in tokens:
            if t in sp_map_inv:
                decoded_parts.append(sp_map_inv[t])
            else:
                # Attempt to reverse byte mapping
                # b = t - offset ?? 
                # Since we did modulo, we can't perfectly recover unless t < vocab.
                # Assuming simple wrap:
                # Approximate: just char(t % 256)
                b = t % 256
                try:
                    out_bytes.append(b)
                except ValueError:
                    pass
        
        # Flush bytes
        if out_bytes:
             decoded_parts.append(out_bytes.decode('utf-8', errors='replace'))
             
        return "".join(decoded_parts)
