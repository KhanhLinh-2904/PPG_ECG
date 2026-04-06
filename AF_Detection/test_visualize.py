import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, TensorDataset
from ecg_transform import ECGTransformerModel

def test_visualize_tsne(backbone: nn.Module, dataloader: DataLoader, device: str = 'cuda'):
    backbone.eval()
    backbone.to(device)
    
    all_trans_reps = [] # To store Transformer features
    all_cnn_reps = []   # To store CNN features
    all_labels = []
    
    print("⏳ Extracting features from model...")
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            
            # 1. Run through the entire model to get Transformer Reps & Padding mask
            outputs = backbone(inputs)
            local_reps = outputs["local_reps"]
            
            # 2. 🟢 MANUALLY RUN THROUGH CNN TO GET CNN REPS
            # CNN output shape: [Batch, Channels, Time]
            cnn_features = backbone.feature_extractor(inputs)
            # Transpose to [Batch, Time, Channels] to align with padding_mask format
            cnn_features = cnn_features.transpose(1, 2)
            
            # Global Average Pooling across the time dimension to get representative vectors
            global_trans = local_reps.mean(dim=1)
            global_cnn = cnn_features.mean(dim=1)
                
            all_trans_reps.append(global_trans.cpu().numpy())
            all_cnn_reps.append(global_cnn.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            
    X_trans = np.concatenate(all_trans_reps, axis=0) 
    X_cnn = np.concatenate(all_cnn_reps, axis=0) 
    y = np.concatenate(all_labels, axis=0)    
    
    # --- RUN t-SNE ---
    print("⚙️ Calculating t-SNE for CNN (this may take a few minutes)...")
    tsne_cnn = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42)
    X_tsne_cnn = tsne_cnn.fit_transform(X_cnn)

    print("⚙️ Calculating t-SNE for Transformer...")
    tsne_trans = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42)
    X_tsne_trans = tsne_trans.fit_transform(X_trans)
    
    # --- PLOTTING (SIDE-BY-SIDE) ---
    print("🎨 Plotting results...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    idx_non_af = (y == 0)
    idx_af = (y == 1)
    
    # Plot 1: CNN
    axes[0].scatter(X_tsne_cnn[idx_non_af, 0], X_tsne_cnn[idx_non_af, 1], 
                c='lightgray', label='non-AF', alpha=0.6, edgecolors='none', s=50)
    axes[0].scatter(X_tsne_cnn[idx_af, 0], X_tsne_cnn[idx_af, 1], 
                c='red', label='AF', alpha=0.6, edgecolors='w', s=50)
    axes[0].set_title('t-SNE: CNN Feature Extractor', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Dimension 1', fontsize=12)
    axes[0].set_ylabel('Dimension 2', fontsize=12)
    axes[0].legend(fontsize=12)
    axes[0].grid(True, linestyle='--', alpha=0.5)

    # Plot 2: Transformer
    axes[1].scatter(X_tsne_trans[idx_non_af, 0], X_tsne_trans[idx_non_af, 1], 
                c='lightgray', label='non-AF', alpha=0.6, edgecolors='none', s=50)
    axes[1].scatter(X_tsne_trans[idx_af, 0], X_tsne_trans[idx_af, 1], 
                c='red', label='AF', alpha=0.6, edgecolors='w', s=50)
    axes[1].set_title('t-SNE: Transformer Encoder', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Dimension 1', fontsize=12)
    axes[1].legend(fontsize=12)
    axes[1].grid(True, linestyle='--', alpha=0.5)
    
    plt.suptitle('Comparison of Feature Representations', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
    print("✅ Completed!")


if __name__ == "__main__":
    EMBED_DIM = 256
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    try:
        test_data = np.load('datasets/total_z.npz')

        X_test = torch.tensor(test_data["ecgs"], dtype=torch.float32)
        y_test = torch.tensor(test_data["labels"], dtype=torch.long)
    except FileNotFoundError:
        print("Error: Could not find the file path!")
        exit()

  
    if X_test.dim() == 2:
        X_test = X_test.unsqueeze(1) # [Batch, 1, 2400]

    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    transformer_encoder = nn.TransformerEncoder(
        nn.TransformerEncoderLayer(d_model=EMBED_DIM, nhead=8, dim_feedforward=1024, dropout=0.1, batch_first=True), 
        num_layers=4
    )

    backbone = ECGTransformerModel(
        in_channels=1, 
        embed_dim=EMBED_DIM,
        conv_layers=[(256, 10, 5), (256, 3, 2), (256, 3, 2)], 
        vq_dim=EMBED_DIM,
        transformer_encoder=transformer_encoder
    )

    pretrained_path = "AF_Detection/checkpoints/pretrained_backbone.pth"
    if os.path.exists(pretrained_path):
        print(f"Loading weights from file: {pretrained_path}")
        backbone.load_state_dict(torch.load(pretrained_path, map_location=DEVICE))
        print("Weight files loaded successfully!")
    else:
        print(f"Warning: Could not find '{pretrained_path}'. Using random initialization.")

    test_visualize_tsne(backbone, test_loader, device=DEVICE)