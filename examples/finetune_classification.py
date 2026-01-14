import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from thl.config import THLConfig
from thl.model import THLForSequenceClassification
from thl.tokenizer import Tokenizer
import torch.optim as optim

class DummyDataset(Dataset):
    def __init__(self, size=100, seq_len=32):
        self.tokenizer = Tokenizer()
        self.data = [torch.randint(0, 50257, (seq_len,)) for _ in range(size)]
        self.labels = [torch.tensor(i % 2) for i in range(size)]
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 1. Setup Model & Config
    config = THLConfig(num_tiers=3, device=device)
    model = THLForSequenceClassification(config, num_labels=2)
    model.to(device)
    
    # 2. Setup Data
    train_dataset = DummyDataset(size=50)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    
    # 3. Setup Training
    optimizer = optim.AdamW(model.parameters(), lr=5e-5)
    scaler = torch.cuda.amp.GradScaler() if device == "cuda" else None
    
    model.train()
    print("Starting training...")
    
    for epoch in range(2):
        total_loss = 0
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward with AMP
            with torch.autocast(device_type=device, enabled=(scaler is not None)):
                logits, loss = model(inputs, labels=labels)
                
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
                
            total_loss += loss.item()
            
            if batch_idx % 5 == 0:
                print(f"Epoch {epoch} | Batch {batch_idx} | Loss: {loss.item():.4f}")
                
        print(f"Epoch {epoch} finished. Avg Loss: {total_loss / len(train_loader):.4f}")
    
    # 4. Save Fine-Tuned Model
    output_dir = "finetuned_thl"
    print(f"Saving model to {output_dir}...")
    # Unwrap 'thl' from SequenceClassification for saving if we want to save base
    # But usually we save the whole thing.
    # THLFor... doesn't implement save_pretrained, its wrapped 'thl' does.
    # We should implement save_pretrained on wrapper or use torch.save logic manually.
    # For now, manually save state dict.
    
    import os
    os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
    model.thl.config.save(output_dir) # Config helper if available, else manual
    # Manual config save
    import json
    from dataclasses import asdict
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(asdict(model.thl.config), f)
        
    print("Done!")

if __name__ == "__main__":
    main()
