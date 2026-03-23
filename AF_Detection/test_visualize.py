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
    
    all_global_reps = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            
            outputs = backbone(inputs)
            local_reps = outputs["local_reps"]
            padding_mask = outputs["padding_mask"]
            
            if padding_mask is not None and padding_mask.any():
                local_reps[padding_mask] = 0.0
                valid_lengths = (~padding_mask).sum(dim=1, keepdim=True).clamp(min=1)
                global_reps = local_reps.sum(dim=1) / valid_lengths
            else:
                global_reps = local_reps.mean(dim=1)
                
            all_global_reps.append(global_reps.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            
    X = np.concatenate(all_global_reps, axis=0) 
    y = np.concatenate(all_labels, axis=0)    
    
    tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42)
    X_tsne = tsne.fit_transform(X)
    
    plt.figure(figsize=(10, 8))
    
    idx_non_af = (y == 0)
    idx_af = (y == 1)
    
    plt.scatter(X_tsne[idx_non_af, 0], X_tsne[idx_non_af, 1], 
                c='blue', label='non-AF', alpha=0.6, edgecolors='w', s=50)
    plt.scatter(X_tsne[idx_af, 0], X_tsne[idx_af, 1], 
                c='red', label='AF', alpha=0.6, edgecolors='w', s=50)
    
    plt.title('t-SNE: Global Representations after Pre-training', fontsize=14, fontweight='bold')
    plt.xlabel('t-SNE Dimension 1', fontsize=12)
    plt.ylabel('t-SNE Dimension 2', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.show()
    print("Complete!")


if __name__ == "__main__":
    EMBED_DIM = 256
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    try:
        test_data = np.load('AF_Detection/deepbeat_ecg_reconstructions.npz')
        X_test = torch.tensor(test_data["ecgs"], dtype=torch.float32)
        y_test = torch.tensor(test_data["labels"], dtype=torch.long)
    except FileNotFoundError:
        print("Error: Can not find the link!")
        exit()

  
    if X_test.dim() == 2:
        X_test = X_test.unsqueeze(1) #  [Batch, 1, 2400]

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
        print("Complete loading weight files!")
    else:
        print(f"Can not find the file '{pretrained_path}'. Random installization")

    test_visualize_tsne(backbone, test_loader, device=DEVICE)