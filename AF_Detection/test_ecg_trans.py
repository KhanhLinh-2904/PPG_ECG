import os
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pickle
from scipy.linalg import fractional_matrix_power
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

# Import ECGTransformerModel
from ecg_transform import ECGTransformerModel

# ==============================================================================
# HELPER FUNCTIONS FOR CORAL ALIGNMENT
# ==============================================================================
def get_coral_stats(features):
    """Calculates Mean and Covariance of the Test set (Used for alignment)"""
    mu = np.mean(features, axis=0)
    cov = np.cov(features, rowvar=False) + np.eye(features.shape[1]) * 1e-5
    return mu, cov

def apply_coral(source_mu, source_cov, target_features):
    """
    Aligns the Test data (Target) distribution to match the Train (Source) distribution.
    - source_mu, source_cov: Extracted from pickle file (MIT-BIH training stats)
    - target_features: Raw features extracted from the Backbone of the current Test set
    """
    target_mu, target_cov = get_coral_stats(target_features)
    
    # Calculate the spatial transformation matrices
    cov_s_half = fractional_matrix_power(source_cov, 0.5)
    cov_t_inv_half = fractional_matrix_power(target_cov, -0.5)
    
    # Apply rotation, scaling, and translation
    target_aligned = np.dot(target_features - target_mu, cov_t_inv_half)
    target_aligned = np.dot(target_aligned, cov_s_half) + source_mu
    
    return target_aligned.real

# ==============================================================================
# PATH CONFIGURATION
# ==============================================================================
pretrained_path = '/home/linhhima/PPG_ECG/AF_Detection/checkpoints/pretrained_backbone.pth'
# Load file containing CORAL stats and the k-NN model
coral_model_path = '/home/linhhima/PPG_ECG/AF_Detection/checkpoints/coral_knn_checkpoint.pkl'

test_files = {
    "MIT-BIH": '/home/linhhima/PPG_ECG/datasets/z_score_norm/MIT_BIH_test_segments.npz',
    "Total z (MIMIC AF)": '/home/linhhima/PPG_ECG/datasets/z_score_norm/total_mimic_af.npz',
    "total_mimic_af_flow": '/home/linhhima/PPG_ECG/AF_Detection/total_mimic_af_flow_segment.npz',
    "total_mimic_af_flow_subject": '/home/linhhima/PPG_ECG/AF_Detection/total_mimic_af_flow_subject.npz',
    "total_mimic_af_ppg2ecg": '/home/linhhima/PPG_ECG/AF_Detection/total_mimic_af_ppg2ecg_segment.npz',
    "total_mimic_af_ppg2ecg_subject": '/home/linhhima/PPG_ECG/AF_Detection/total_mimic_af_ppg2ecg_subject.npz',


    # "Deepbeat Recon": '/home/linhhima/PPG_ECG/AF_Detection/deepbeat_ecg_reconstructions.npz'
}

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Starting Evaluation on device: {device}")

    # ==============================================================================
    # 1. LOAD k-NN CLASSIFIER & CORAL STATS FROM PHASE 2
    # ==============================================================================
    print("📚 Loading k-NN Model and CORAL parameters...")
    try:
        with open(coral_model_path, 'rb') as f:
            checkpoint_data = pickle.load(f)
            
        classifier = checkpoint_data['classifier']
        source_mu = checkpoint_data['source_mu']
        source_cov = checkpoint_data['source_cov']
        
        print("✅ Successfully loaded Classifier and Source Feature Map (MIT-BIH).")
    except FileNotFoundError:
        print(f"❌ Error: File not found at {coral_model_path}. Please run the Phase 2 training script.")
        exit()
    except KeyError as e:
        print(f"❌ Error: Pickle file is missing key {e}. This file might be an outdated version.")
        exit()

    # ==============================================================================
    # 2. INITIALIZE BACKBONE (MUST MATCH TRAINING ARCHITECTURE)
    # ==============================================================================
    print("🧠 Initializing Transformer Backbone architecture...")
    EMBED_DIM = 256
    
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
    ).to(device)

    try:
        backbone.load_state_dict(torch.load(pretrained_path, map_location=device))
        print("✅ Backbone weights loaded successfully.")
    except FileNotFoundError:
        print(f"❌ Error: Backbone weights not found at {pretrained_path}")
        exit()

    backbone.eval() 

    # ==============================================================================
    # 3. RUN PREDICTIONS ON EACH DATASET
    # ==============================================================================
    print("\n" + "="*60)
    print("🔍 PERFORMING EVALUATION (CORAL ALIGNMENT + k-NN)...")
    print("="*60)

    for test_name, file_path in test_files.items():
        print(f"\n📂 Dataset: [{test_name}]")
        
        try:
            test_data = np.load(file_path)
            X_test_raw = test_data["ecgs"]
            y_test_raw = test_data["labels"]
            
            X_test_tensor = torch.tensor(X_test_raw, dtype=torch.float32)
            y_test_tensor = torch.tensor(y_test_raw, dtype=torch.long)
        except Exception as e:
            print(f"    ⚠️ Error loading file {file_path}: {e}. Skipping.")
            continue
            
        if X_test_tensor.dim() == 2:
            X_test_tensor = X_test_tensor.unsqueeze(1)

        test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=64, shuffle=False)

        all_test_features = []
        all_true_labels = []

        # 3.1: Extract raw features from Backbone
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                outputs = backbone(inputs)
                
                # Global Average Pooling across the time dimension
                features = outputs['local_reps'].mean(dim=1).cpu().numpy()
                all_test_features.append(features)
                all_true_labels.extend(labels.numpy())

        # Concatenate into (N_samples, 256) matrix
        all_test_features = np.concatenate(all_test_features, axis=0)
        all_true_labels = np.array(all_true_labels)
      
        # ==============================================================================
        # 3.2: APPLY CORAL ALIGNMENT TO TRANSFORM COORDINATES
        # ==============================================================================
        print(f"    🔄 Aligning feature distribution using CORAL Alignment...")
        # Force the test set feature cloud (MIMIC/MIT-BIH) to match the training set (MIT-BIH)
        all_test_features_aligned = apply_coral(source_mu, source_cov, all_test_features)

        # ==============================================================================
        # 3.3: CLASSIFY USING k-NN ON ALIGNED DATA
        # ==============================================================================
        print(f"    🧠 Predicting with k-NN...")
        all_preds = classifier.predict(all_test_features_aligned)

        # ==============================================================================
        # 4. CALCULATE AND PRINT RESULTS
        # ==============================================================================
        acc = accuracy_score(all_true_labels, all_preds)
        prec = precision_score(all_true_labels, all_preds, zero_division=0)
        rec = recall_score(all_true_labels, all_preds, zero_division=0)
        cm = confusion_matrix(all_true_labels, all_preds)
        
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        else:
            # Fallback for single-class batches or edge cases
            tn = cm[0,0] if all_true_labels[0] == 0 else 0
            tp = cm[0,0] if all_true_labels[0] == 1 else 0
            fp = fn = 0
            spec = 1.0 if tn > 0 else 0.0

        print("   " + "-" * 35)
        print(f"      • Accuracy    : {acc * 100:.2f}%")
        print(f"      • Precision   : {prec * 100:.2f}%")
        print(f"      • Recall (Sen): {rec * 100:.2f}%")
        print(f"      • Specificity : {spec * 100:.2f}%")
        print("   " + "-" * 35)
        
        print(f"    📉 Confusion Matrix: [TN: {tn}, FP: {fp}] / [FN: {fn}, TP: {tp}]")
        print("="*60)