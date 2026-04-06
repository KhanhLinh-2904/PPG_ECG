import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import random
from ecg_transform import ECGTransformerModel, VQContrastiveLoss, ECGClusterAndClassifier, ECGClusterClassifier
import os
import torch
import numpy as np
import pickle
from sklearn.preprocessing import normalize
from coral_utils import get_coral_stats
from scipy.linalg import fractional_matrix_power
from sklearn.neighbors import KNeighborsClassifier
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    # ==========================================
    DO_PRETRAIN = False 
    DO_FINETUNE = True   
    # ==========================================

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"running on: {device}")

    CHECKPOINT_DIR = "AF_Detection/checkpoints"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    pretrained_path = os.path.join(CHECKPOINT_DIR, "pretrained_backbone.pth")
    print(f"folder saved: {CHECKPOINT_DIR}")

    print("⏳ loading Train set...")
    try:
        train_data = np.load('processed_data/MIT_BIH_train_segments.npz')
        X_train = torch.tensor(train_data["ecgs"], dtype=torch.float32)
        y_train = torch.tensor(train_data["labels"], dtype=torch.long)
    except FileNotFoundError:
        print("can not find Train set, using (Dummy Data).")
        X_train = torch.randn(100, 1, 2400)
        y_train = torch.randint(0, 2, (100,))

    if X_train.dim() == 2:
        X_train = X_train.unsqueeze(1)

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    print("⏳ loading Test sets...")
    test_loaders = {}
    
    test_files = {
        "MIT-BIH": 'processed_data/MIT_BIH_test_segments.npz',
        "Total z (MIMIC AF)": 'datasets/total_z.npz',
        "(MIMIC AF) z-Recon": 'AF_Detection/total_ecg_reconstructions.npz',
        "Deepbeat Recon": 'AF_Detection/deepbeat_ecg_reconstructions.npz'
    }

    for name, path in test_files.items():
        try:
            t_data = np.load(path)
            X_t = torch.tensor(t_data["ecgs"], dtype=torch.float32)
            y_t = torch.tensor(t_data["labels"], dtype=torch.long)
            
            if X_t.dim() == 2:
                X_t = X_t.unsqueeze(1)
                
            t_dataset = TensorDataset(X_t, y_t)
            test_loaders[name] = DataLoader(t_dataset, batch_size=32, shuffle=False)
            print(f"  loading Test set: {name} (shape: {X_t.shape[0]} samples)")
        except FileNotFoundError:
            print(f"  error: not found file Test '{path}'. skip.")
        except KeyError as e:
            print(f" error: can not find {e} in file '{path}'")

    if not test_loaders:
        print("can not load any Test set, using (Dummy Data).")

    # --- (BACKBONE) ---
    print("initializing Backbone...")
    EMBED_DIM = 256
    transformer_encoder = nn.TransformerEncoder(
        nn.TransformerEncoderLayer(d_model=EMBED_DIM, nhead=8, dim_feedforward=1024, dropout=0.1, batch_first=True), 
        num_layers=4
    )
    
    backbone = ECGTransformerModel(
        in_channels=X_train.shape[1],
        embed_dim=EMBED_DIM,
        conv_layers=[(256, 10, 5), (256, 3, 2), (256, 3, 2)], 
        vq_dim=EMBED_DIM,
        feature_grad_mult = 1.0,
        transformer_encoder=transformer_encoder
    ).to(device)


    # ==========================================================================
    # stage 1: PRE-TRAINING
    # ==========================================================================
    if DO_PRETRAIN:
        print("\n" + "="*50)
        print("stage 1: PRE-TRAINING BACKBONE")
        print("="*50)
        
        pretrain_epochs = 15
        pretrain_criterion = VQContrastiveLoss().to(device)
        pretrain_optimizer = torch.optim.AdamW(backbone.parameters(), lr=5e-4)

        for epoch in range(pretrain_epochs):
            backbone.train()
            running_loss = 0.0
            
            
            for batch_idx, (inputs, labels) in enumerate(train_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)
                pretrain_optimizer.zero_grad()
                outputs = backbone(inputs)
              
                
                loss, c_loss = pretrain_criterion(
                    local_reps = outputs["local_reps"], 
                    q_targets = outputs["q_targets"], 
                    mask_indices = outputs["mask_indices"], 
                    vq_loss = outputs["vq_loss"],
                    vq_perplexity=outputs["perplexity"]
                )
                
                if loss.requires_grad:
                    loss.backward()
                    pretrain_optimizer.step()
                    
                running_loss += loss.item()
                
            print(f"Pre-train Epoch [{epoch+1}/{pretrain_epochs}] | Total Loss: {running_loss/len(train_loader):.4f}")

        torch.save(backbone.state_dict(), pretrained_path)
        print(f"save Backbone successfully: {pretrained_path}")
    else:
        print("\n" + "="*50)
        print("skip PRE-TRAINING. loading FILE...")
        print("="*50)
        if os.path.exists(pretrained_path):
            backbone.load_state_dict(torch.load(pretrained_path, map_location=device))
            print(f"save weights Backbone from: {pretrained_path}")
        else:
            print(f"error: not found '{pretrained_path}'. Backbone will use random initialization!")


    def get_coral_stats(features):
        mu = np.mean(features, axis=0)
        cov = np.cov(features, rowvar=False) + np.eye(features.shape[1]) * 1e-5
        return mu, cov
    # ==========================================================================
    # stage 2: FEATURE EXTRACTION, CORAL COMPUTATION & KNN TRAINING
    # ==========================================================================
    if DO_FINETUNE:
        print("\n" + "="*50)
        print("🎯 stage 2: CORAL ALIGNMENT & k-NN CLASSIFICATION")
        print("="*50)

        # 1. initialize Backbone and Load Pretrained Weights
        model_transformer = ECGTransformerModel(
            in_channels=X_train.shape[1],
            embed_dim=EMBED_DIM,
            conv_layers=[(256, 10, 5), (256, 3, 2), (256, 3, 2)], 
            vq_dim=EMBED_DIM,
            feature_grad_mult = 1.0,
            transformer_encoder=transformer_encoder
        ).to(device)
        
        # Load wieghts pretrain from stage 1
        model_transformer.load_state_dict(torch.load(pretrained_path, map_location=device))
        model_transformer.eval() 

        # ---------------------------------------------------------
        # A: ectract features from TRAIN
        # ---------------------------------------------------------
        print("extracting features from Train...")
        all_train_features = []
        all_train_labels = []

        with torch.no_grad():
            for inputs, labels in train_loader:
                inputs = inputs.to(device)
                
                outputs = model_transformer(inputs) 
                local_reps = outputs['local_reps'] # Shape: [Batch, Seq_len, Embed_dim]
                
                ecg_features = local_reps.mean(dim=1) 
                
                all_train_features.append(ecg_features.cpu().numpy())
                all_train_labels.append(labels.cpu().numpy())

        all_train_features = np.concatenate(all_train_features, axis=0)
        all_train_labels = np.concatenate(all_train_labels, axis=0)

        # ---------------------------------------------------------
        # B: calculate CORAL STATS & train KNN
        # ---------------------------------------------------------
        source_mu, source_cov = get_coral_stats(all_train_features)
        print(f"  - Mean Vector Shape: {source_mu}")
        print(f"  - Covariance Matrix Shape: {source_cov}")

        # k-NN K=5 and metric='cosine' 
        classifier = KNeighborsClassifier(n_neighbors=9, metric='cosine', weights='distance')
        classifier.fit(all_train_features, all_train_labels)

        # ---------------------------------------------------------
        # C: save CLASSIFIER and CORAL STATS
        # ---------------------------------------------------------
        checkpoint_data = {
            'classifier': classifier,
            'source_mu': source_mu,
            'source_cov': source_cov
        }

        model_save_path = os.path.join(CHECKPOINT_DIR, "coral_knn_checkpoint_13.pkl")
        with open(model_save_path, 'wb') as f:
            pickle.dump(checkpoint_data, f)
            
        print("\n✅ completed stage 2!")
        print(f"  - Model & CORAL Stats saved at: {model_save_path}")
        print(f"  - Vector Mean Shape: {source_mu.shape}")
        print(f"  - Covariance Matrix Shape: {source_cov.shape}")
        
