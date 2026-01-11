import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import clip
from sklearn.metrics import f1_score, accuracy_score, average_precision_score
from sklearn.manifold import TSNE
from tqdm import tqdm
import matplotlib.pyplot as plt

# Import the MM-IMDb dataset
from mmimdb_loader import MMIMDbDatasetCLIP, analyze_dataset, collate_fn


class FrozenCLIPImageEncoder(nn.Module):
    """Frozen CLIP image encoder"""
    def __init__(self, device):
        super().__init__()
        self.model, self.preprocess = clip.load("ViT-B/32", device=device)
        # Freeze parameters
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()
        self.device = device
    
    def forward(self, images):
        """
        images: [batch, 3, 224, 224] tensor (already preprocessed)
        """
        with torch.no_grad():
            features = self.model.encode_image(images)
            return features.float()  # [batch, 512]


class FrozenCLIPTextEncoder(nn.Module):
    """Frozen CLIP text encoder"""
    def __init__(self, device):
        super().__init__()
        self.model, _ = clip.load("ViT-B/32", device=device)
        # Freeze parameters
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()
        self.device = device
    
    def forward(self, texts):
        """
        texts: list of strings
        """
        with torch.no_grad():
            # Tokenize texts using CLIP tokenizer
            tokens = clip.tokenize(texts, truncate=True).to(self.device)
            features = self.model.encode_text(tokens)
            return features.float()  # [batch, 512]


class FusionModule(nn.Module):
    """Multi-Head Self-Attention fusion for two modalities"""
    def __init__(self, input_dims, output_dim=256, num_heads=4, num_layers=2):
        super().__init__()
        self.input_dims = input_dims
        self.output_dim = output_dim
        self.num_modalities = len(input_dims)
        
        # Project each modality to the same dimension for attention
        self.modality_projections = nn.ModuleList([
            nn.Linear(dim, output_dim) for dim in input_dims
        ])
        
        # Multi-head attention layers
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=output_dim, num_heads=num_heads, batch_first=True)
            for _ in range(num_layers)
        ])
        
        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(output_dim) for _ in range(num_layers)
        ])
        
        # Final projection
        self.output_proj = nn.Linear(output_dim, output_dim)
        self.relu = nn.ReLU()
    
    def forward(self, *features):
        batch_size = features[0].shape[0]
        
        # Project each modality to common dimension
        projected = []
        for i, feat in enumerate(features):
            proj = self.modality_projections[i](feat)
            projected.append(proj)
        
        # Stack modalities as sequence
        modality_sequence = torch.stack(projected, dim=1)
        
        # Apply multi-head attention layers
        attn_output = modality_sequence
        for attn_layer, layer_norm in zip(self.attention_layers, self.layer_norms):
            attn_out, _ = attn_layer(attn_output, attn_output, attn_output)
            attn_output = layer_norm(attn_output + attn_out)
        
        # Pool across modalities
        fused = attn_output.mean(dim=1)
        
        # Final projection
        output = self.relu(self.output_proj(fused))
        
        return output


class Simulator(nn.Module):
    """Three-layer MLP simulator"""
    def __init__(self, input_dim, output_dim, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        return self.fc3(self.relu2(self.fc2(self.relu1(self.fc1(x)))))


class CrossModalSimulationModel(nn.Module):
    """Complete model with cross-modal simulation for 2 modalities"""
    def __init__(self, num_classes, device):
        super().__init__()
        
        # Frozen CLIP encoders
        self.image_encoder = FrozenCLIPImageEncoder(device)
        self.text_encoder = FrozenCLIPTextEncoder(device)
        
        # Feature dimensions (CLIP ViT-B/32 outputs 512-dim)
        self.image_dim = 512
        self.text_dim = 512
        self.fusion_dim = 256
        
        # Pairwise fusion module
        self.fuse_it = FusionModule([self.image_dim, self.text_dim], 
                                     output_dim=self.fusion_dim, 
                                     num_heads=4, num_layers=2)
        
        # Pairwise simulators
        self.sim_t_i = Simulator(self.text_dim, self.image_dim, hidden_dim=256)
        self.sim_i_t = Simulator(self.image_dim, self.text_dim, hidden_dim=256)
        
        # Final fusion and classifier
        self.final_fusion = FusionModule([self.image_dim, self.text_dim], 
                                         output_dim=self.fusion_dim,
                                         num_heads=4, num_layers=2)
        
        self.classifier = nn.Sequential(
            nn.Linear(self.fusion_dim, 128),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, images, texts, has_image, has_text, 
                return_features=False, skip_verification=False):
        batch_size = images.shape[0]
        device = next(self.parameters()).device
        
        # Extract features for available modalities
        image_feat = torch.zeros(batch_size, self.image_dim).to(device)
        text_feat = torch.zeros(batch_size, self.text_dim).to(device)
        
        # Encode available modalities
        image_available_indices = [i for i in range(batch_size) if has_image[i]]
        text_available_indices = [i for i in range(batch_size) if has_text[i]]
        
        if image_available_indices:
            image_batch = images[image_available_indices]
            image_feat[image_available_indices] = self.image_encoder(image_batch)
        
        if text_available_indices:
            text_batch = [texts[i] for i in text_available_indices]
            text_feat[text_available_indices] = self.text_encoder(text_batch)
        
        # Store simulation losses
        sim_losses = []
        
        # Final features (will be filled with real or simulated)
        final_image = image_feat.clone()
        final_text = text_feat.clone()
        
        # Process each sample individually for simulation
        for i in range(batch_size):
            i_avail = has_image[i]
            t_avail = has_text[i]
            
            # Case: Both available
            if i_avail and t_avail:
                # Bidirectional simulation
                sim_losses.append(F.mse_loss(
                    self.sim_t_i(text_feat[i:i+1]), 
                    image_feat[i:i+1]
                ))
                sim_losses.append(F.mse_loss(
                    self.sim_i_t(image_feat[i:i+1]), 
                    text_feat[i:i+1]
                ))
            
            # Case: Only image available
            elif i_avail and not t_avail:
                final_text[i:i+1] = self.sim_i_t(image_feat[i:i+1])
            
            # Case: Only text available
            elif t_avail and not i_avail:
                final_image[i:i+1] = self.sim_t_i(text_feat[i:i+1])
        
        # Verification
        if not skip_verification:
            for i in range(batch_size):
                if torch.all(final_image[i] == 0):
                    raise RuntimeError(f"Sample {i}: Image features are all zeros!")
                if torch.all(final_text[i] == 0):
                    raise RuntimeError(f"Sample {i}: Text features are all zeros!")
        
        # Final fusion and classification
        fused = self.final_fusion(final_image, final_text)
        logits = self.classifier(fused)
        
        # Average simulation loss
        avg_sim_loss = torch.mean(torch.stack(sim_losses)) if sim_losses else torch.tensor(0.0).to(device)
        
        if return_features:
            return logits, avg_sim_loss, fused
        return logits, avg_sim_loss


def train_epoch(model, dataloader, optimizer, alpha=0.5, beta=1.0, device='cuda'):
    model.train()
    model.image_encoder.eval()
    model.text_encoder.eval()
    
    total_loss = 0
    total_sim_loss = 0
    total_cls_loss = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        images = batch['images'].to(device)
        texts = batch['texts']
        labels = batch['labels'].to(device)
        has_image = batch['has_image']
        has_text = batch['has_text']
        
        optimizer.zero_grad()
        
        logits, sim_loss = model(images, texts, has_image, has_text)
        cls_loss = F.binary_cross_entropy_with_logits(logits, labels)
        
        loss = alpha * sim_loss + beta * cls_loss
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_sim_loss += sim_loss.item()
        total_cls_loss += cls_loss.item()
    
    return total_loss / len(dataloader), total_sim_loss / len(dataloader), total_cls_loss / len(dataloader)


def evaluate(model, dataloader, device='cuda', threshold=0.5):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            images = batch['images'].to(device)
            texts = batch['texts']
            labels = batch['labels'].to(device)
            has_image = batch['has_image']
            has_text = batch['has_text']
            
            logits, _ = model(images, texts, has_image, has_text)
            probs = torch.sigmoid(logits)
            preds = (probs > threshold).float()
            
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
    
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    all_probs = np.vstack(all_probs)
    
    # Compute metrics
    f1_micro = f1_score(all_labels.flatten(), all_preds.flatten(), average='micro', zero_division=0)
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    map_score = average_precision_score(all_labels, all_probs, average='macro')
    
    return f1_micro, f1_macro, map_score


def main():
    # Configuration
    dataset_path = "/home/office/Downloads/_Dataset/mmimdb/"
    split_file = os.path.join(dataset_path, "split.json")
    
    #missing_config = "100_image_20_text"
    #missing_config = "complex_20_40_40"
    missing_config = "20_image_100_text"

    batch_size = 32
    num_epochs = 20
    lr = 1e-4
    alpha = 0.5
    beta = 1.0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\nUsing device: {device}")
    
    # Analyze dataset
    splits, genre_list, genre_to_idx, sample_info = analyze_dataset(
        dataset_path, split_file
    )
    
    num_classes = len(genre_list)
    print(f"\nNumber of genres (classes): {num_classes}")
    
    # Create datasets
    train_dataset = MMIMDbDatasetCLIP(
        root_dir=dataset_path,
        split_file=split_file,
        split='train',
        genre_list=genre_list,
        genre_to_idx=genre_to_idx,
        missing_config=missing_config,
        seed=42
    )
    
    dev_dataset = MMIMDbDatasetCLIP(
        root_dir=dataset_path,
        split_file=split_file,
        split='dev',
        genre_list=genre_list,
        genre_to_idx=genre_to_idx,
        missing_config=missing_config,
        seed=42
    )
    
    test_dataset = MMIMDbDatasetCLIP(
        root_dir=dataset_path,
        split_file=split_file,
        split='test',
        genre_list=genre_list,
        genre_to_idx=genre_to_idx,
        missing_config=missing_config,
        seed=42
    )
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                             shuffle=True, collate_fn=collate_fn, num_workers=2)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, 
                           shuffle=False, collate_fn=collate_fn, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                            shuffle=False, collate_fn=collate_fn, num_workers=2)
    
    # Create model
    model = CrossModalSimulationModel(num_classes=num_classes, device=device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    best_dev_f1 = 0
    best_epoch = 0
    
    print("\n" + "="*80)
    print("TRAINING CROSS-MODAL SIMULATION MODEL FOR MM-IMDB")
    print(f"Configuration: alpha={alpha}, beta={beta}, epochs={num_epochs}, lr={lr}")
    print(f"Missing config: {missing_config}")
    print("="*80 + "\n")
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 80)
        
        # Train
        train_loss, train_sim_loss, train_cls_loss = train_epoch(
            model, train_loader, optimizer, alpha, beta, device
        )
        print(f"Train - Loss: {train_loss:.4f} | Sim: {train_sim_loss:.4f} | Cls: {train_cls_loss:.4f}")
        
        # Validate
        dev_f1_micro, dev_f1_macro, dev_map = evaluate(model, dev_loader, device)
        print(f"Dev   - F1-Micro: {dev_f1_micro:.4f} | F1-Macro: {dev_f1_macro:.4f} | mAP: {dev_map:.4f}")
        
        # Save best model
        if dev_f1_macro > best_dev_f1:
            best_dev_f1 = dev_f1_macro
            best_epoch = epoch + 1
            torch.save(model.state_dict(), missing_config + '-best_model-mmimdb.pt')
            print(f"★ New best model saved (F1-Macro: {best_dev_f1:.4f})")
    
    # Final evaluation
    print("\n" + "="*80)
    print(f"FINAL EVALUATION (Best model from epoch {best_epoch})")
    print("="*80 + "\n")
    
    model.load_state_dict(torch.load(missing_config +'-best_model-mmimdb.pt'))
    test_f1_micro, test_f1_macro, test_map = evaluate(model, test_loader, device)
    
    print(f"Test Results:")
    print(f"  F1-Micro:  {test_f1_micro:.4f}")
    print(f"  F1-Macro:  {test_f1_macro:.4f}")
    print(f"  mAP:       {test_map:.4f}")
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
