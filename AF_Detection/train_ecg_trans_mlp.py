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
from sklearn.neural_network import MLPClassifier

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    # ==========================================
    # ⚙️ RUN CONFIGURATION (TOGGLE PHASES)
    # ==========================================
    DO_PRETRAIN = False 
    DO_FINETUNE = True   
    # ==========================================

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Running on device: {device}")

    # Create directory to save weights
    CHECKPOINT_DIR = "AF_Detection/checkpoints"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    pretrained_path = os.path.join(CHECKPOINT_DIR, "pretrained_backbone.pth")
    print(f"📁 Checkpoint directory: {CHECKPOINT_DIR}")

    # --- PREPARE TRAIN DATA ---
    print("⏳ Loading Train data...")
    try:
        train_data = np.load('processed_data/MIT_BIH_train_segments.npz')
        X_train = torch.tensor(train_data["ecgs"], dtype=torch.float32)
        y_train = torch.tensor(train_data["labels"], dtype=torch.long)
    except FileNotFoundError:
        print("⚠️ Train data file not found, using Dummy Data.")
        X_train = torch.randn(100, 1, 2400)
        y_train = torch.randint(0, 2, (100,))

    if X_train.dim() == 2:
        X_train = X_train.unsqueeze(1)

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # --- PREPARE 3 TEST DATASETS ---
    print("⏳ Loading Test datasets...")
    test_loaders = {} # Use dictionary to store dataset names and corresponding loaders
    
    test_files = {
        "MIT-BIH": 'processed_data/MIT_BIH_test_segments.npz',
        "Total z (MIMIC AF)": 'datasets/total_z.npz',
        "(MIMIC AF) z-Recon": 'AF_Detection/total_ecg_reconstructions.npz'
    }

    for name, path in test_files.items():
        try:
            t_data = np.load(path)
            # Note: Need to check if the keys in the new .npz files are the same as MIT-BIH.
            # Assuming they all use 'ecgs' and 'labels' keys.
            X_t = torch.tensor(t_data["ecgs"], dtype=torch.float32)
            y_t = torch.tensor(t_data["labels"], dtype=torch.long)
            
            if X_t.dim() == 2:
                X_t = X_t.unsqueeze(1)
                
            t_dataset = TensorDataset(X_t, y_t)
            test_loaders[name] = DataLoader(t_dataset, batch_size=32, shuffle=False)
            print(f"  ✅ Successfully loaded Test set: {name} (Size: {X_t.shape[0]} samples)")
        except FileNotFoundError:
            print(f"  ⚠️ Error: Test file '{path}' not found. Skipping this set.")
        except KeyError as e:
            print(f"  ⚠️ Error: Key {e} not found in file '{path}'. Please check the .npz file structure.")

    if not test_loaders:
        print("⚠️ No Test sets loaded!")

    # --- (BACKBONE) ---
    print("🧠 Initializing Backbone model...")
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
    # PHASE 1: PRE-TRAINING
    # ==========================================================================
    if DO_PRETRAIN:
        print("\n" + "="*50)
        print("🌟 PHASE 1: PRE-TRAINING BACKBONE")
        print("="*50)
        
        pretrain_epochs = 15
        pretrain_criterion = VQContrastiveLoss().to(device)
        pretrain_optimizer = torch.optim.AdamW(backbone.parameters(), lr=5e-4)

        for epoch in range(pretrain_epochs):
            backbone.train()
            running_loss = 0.0
            
            # ❌ DELETED: backbone.quantizer.set_num_updates(epoch)
            
            for batch_idx, (inputs, labels) in enumerate(train_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)
                pretrain_optimizer.zero_grad()
                outputs = backbone(inputs)
              
                # Use mask_indices for the "masked prediction" task
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
        print(f"✅ Backbone saved successfully at: {pretrained_path}")
    else:
        print("\n" + "="*50)
        print("⏩ SKIPPING PRE-TRAINING. LOADING WEIGHTS FROM FILE...")
        print("="*50)
        if os.path.exists(pretrained_path):
            backbone.load_state_dict(torch.load(pretrained_path, map_location=device))
            print(f"✅ Backbone weights loaded from: {pretrained_path}")
        else:
            print(f"⚠️ WARNING: '{pretrained_path}' not found. Backbone will use random initial weights!")


    def get_coral_stats(features):
        mu = np.mean(features, axis=0)
        # Add 1e-5 to the diagonal to ensure the matrix is invertible
        cov = np.cov(features, rowvar=False) + np.eye(features.shape[1]) * 1e-5
        return mu, cov

    # ==========================================================================
    # PHASE 2: FEATURE EXTRACTION, CORAL CALCULATION, AND MLP TRAINING
    # ==========================================================================
    if DO_FINETUNE:
        print("\n" + "="*50)
        print("🎯 PHASE 2: CORAL ALIGNMENT & MLP CLASSIFICATION")
        print("="*50)

        # 1. Initialize Backbone and Load Pretrained Weights
        model_transformer = ECGTransformerModel(
            in_channels=X_train.shape[1],
            embed_dim=EMBED_DIM,
            conv_layers=[(256, 10, 5), (256, 3, 2), (256, 3, 2)], 
            vq_dim=EMBED_DIM,
            feature_grad_mult = 1.0,
            transformer_encoder=transformer_encoder
        ).to(device)
        
        # Load weights pretrained from Phase 1
        model_transformer.load_state_dict(torch.load(pretrained_path, map_location=device))
        model_transformer.eval() 

        # ---------------------------------------------------------
        # STEP A: EXTRACT ALL FEATURES FROM TRAIN SET
        # ---------------------------------------------------------
        print("🔄 Extracting features from Train set...")
        all_train_features = []
        all_train_labels = []

        with torch.no_grad():
            for inputs, labels in train_loader:
                inputs = inputs.to(device)
                
                # Forward pass through backbone
                outputs = model_transformer(inputs) 
                local_reps = outputs['local_reps'] # Shape: [Batch, Seq_len, Embed_dim]
                
                # Global Average Pooling across time axis to get representative vectors
                ecg_features = local_reps.mean(dim=1) 
                
                all_train_features.append(ecg_features.cpu().numpy())
                all_train_labels.append(labels.cpu().numpy())

        # Concatenate into 2D Numpy arrays
        all_train_features = np.concatenate(all_train_features, axis=0)
        all_train_labels = np.concatenate(all_train_labels, axis=0)

        print("🧠 Calculating CORAL stats for Train set (Source)...")
        source_mu, source_cov = get_coral_stats(all_train_features)

        print("🧠 Training MLP (Multi-Layer Perceptron) classifier...")
        # Initialize MLP with 2 hidden layers of size 128 and 64.
        # ReLU activation helps warp the space to separate AF/non-AF classes.
        mlp_classifier = MLPClassifier(
            hidden_layer_sizes=(128, 64), 
            activation='relu', 
            solver='adam', 
            max_iter=500,        # Max iterations for training
            random_state=42, 
            early_stopping=True  # Stop early if performance plateaus to avoid Overfitting
        )

        # MLP starts "studying hard" (finding non-linear boundaries)
        mlp_classifier.fit(all_train_features, all_train_labels)

        # ---------------------------------------------------------
        # STORE CLASSIFIER AND CORAL STATS IN PICKLE
        # ---------------------------------------------------------
        checkpoint_data = {
            'classifier': mlp_classifier, # Now using MLP instead of k-NN
            'source_mu': source_mu,
            'source_cov': source_cov
        }

        model_save_path = os.path.join(CHECKPOINT_DIR, "coral_mlp_checkpoint.pkl")
        with open(model_save_path, 'wb') as f:
            pickle.dump(checkpoint_data, f)
            
        print("✅ MLP training complete and checkpoint saved!")