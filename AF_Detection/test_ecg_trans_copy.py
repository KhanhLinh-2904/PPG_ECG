import os
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pickle
from scipy.linalg import fractional_matrix_power

from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

from ecg_transform import ECGTransformerModel

# ==============================================================================
# CORAL SUPPORT FUNCTIONS
# ==============================================================================
def get_coral_stats(features):
    """Calculates Mean and Covariance of the Test set (Used for spatial alignment)"""
    mu = np.mean(features, axis=0)
    cov = np.cov(features, rowvar=False) + np.eye(features.shape[1]) * 1e-5
    return mu, cov

def apply_coral(source_mu, source_cov, target_features):
    """
    Aligns the Test (Target) data cloud to match the 'distribution footprint' of the Train (Source) set.
    - source_mu, source_cov: Extracted from the pickle file (MIT-BIH)
    - target_features: Raw features extracted from the Backbone of the current Test set
    """
    target_mu, target_cov = get_coral_stats(target_features)
    
    # Calculate spatial transformation matrices
    cov_s_half = fractional_matrix_power(source_cov, 0.5)
    cov_t_inv_half = fractional_matrix_power(target_cov, -0.5)
    
    # Apply rotation, scaling, and translation
    target_aligned = np.dot(target_features - target_mu, cov_t_inv_half)
    target_aligned = np.dot(target_aligned, cov_s_half) + source_mu
    
    return target_aligned.real

# ==============================================================================
# FEW-SHOT CALIBRATION SUPPORT FUNCTIONS
# ==============================================================================
def few_shot_predict(features, labels, n_shots=10):
    """
    Extracts n_shots samples per class to serve as anchors (Calibration),
    then predicts the remaining samples based on Cosine Distance.
    """
    idx_0 = np.where(labels == 0)[0]
    idx_1 = np.where(labels == 1)[0]
    
    # Safety check for sample counts
    actual_shots = min(len(idx_0), len(idx_1), n_shots)
    if actual_shots < n_shots:
        print(f"    ⚠️ Not enough samples for {n_shots} shots/class. Reducing calibration samples to {actual_shots}.")
        
    # Randomly select samples for the "Support Set"
    np.random.seed(42) # Fixed seed for reproducibility
    support_idx_0 = np.random.choice(idx_0, actual_shots, replace=False)
    support_idx_1 = np.random.choice(idx_1, actual_shots, replace=False)
    
    # Extract support set features
    support_features_0 = features[support_idx_0]
    support_features_1 = features[support_idx_1]
    
    # Calculate Centroids (Prototypes) for the new space
    centroid_0 = np.mean(support_features_0, axis=0, keepdims=True)
    centroid_1 = np.mean(support_features_1, axis=0, keepdims=True)
    
    # Separate the data to be predicted (Query Set - excludes support samples)
    all_support_idx = np.concatenate([support_idx_0, support_idx_1])
    query_mask = np.ones(len(features), dtype=bool)
    query_mask[all_support_idx] = False
    
    query_features = features[query_mask]
    query_labels = labels[query_mask] 
    
    # Classify using Cosine Distance
    dist_to_0 = cosine_distances(query_features, centroid_0).flatten()
    dist_to_1 = cosine_distances(query_features, centroid_1).flatten()
    
    # Assign 0 if closer to Centroid 0, else 1
    preds = (dist_to_1 < dist_to_0).astype(int)
    
    print(f"    ✅ Calibration complete with {actual_shots * 2} samples. Predicting {len(query_labels)} remaining samples...")
    return query_labels, preds


# ==============================================================================
# PATH CONFIGURATIONS
# ==============================================================================
pretrained_path = 'AF_Detection/checkpoints/pretrained_backbone.pth'
coral_model_path = 'AF_Detection/checkpoints/coral_knn_checkpoint.pkl'

test_files = {
    "MIT-BIH": 'processed_data/MIT_BIH_test_segments.npz',
    "Total z (MIMIC AF)": 'datasets/total_z.npz',
    "(MIMIC AF) z-Recon": 'AF_Detection/total_ecg_reconstructions.npz'
}

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Starting Test process on device: {device}")

    # ==============================================================================
    # 1. LOAD MODEL AND CORAL PARAMETERS
    # ==============================================================================
    print("📚 Loading k-NN Model and CORAL parameters...")
    try:
        with open(coral_model_path, 'rb') as f:
            checkpoint_data = pickle.load(f)
            
        classifier = checkpoint_data['classifier']
        source_mu = checkpoint_data['source_mu']
        source_cov = checkpoint_data['source_cov']
        print("✅ Model and parameters loaded successfully.")
    except Exception as e:
        print(f"❌ Error: {e}")
        exit()

    # ==============================================================================
    # 2. INITIALIZE BACKBONE
    # ==============================================================================
    EMBED_DIM = 256
    transformer_encoder = nn.TransformerEncoder(
        nn.TransformerEncoderLayer(d_model=EMBED_DIM, nhead=8, dim_feedforward=1024, dropout=0.1, batch_first=True), 
        num_layers=4
    )
    
    backbone = ECGTransformerModel(
        in_channels=1, embed_dim=EMBED_DIM,
        conv_layers=[(256, 10, 5), (256, 3, 2), (256, 3, 2)], 
        vq_dim=EMBED_DIM, transformer_encoder=transformer_encoder
    ).to(device)

    backbone.load_state_dict(torch.load(pretrained_path, map_location=device))
    backbone.eval() 

    # ==============================================================================
    # 3. RUN EVALUATION
    # ==============================================================================
    print("\n" + "="*60)
    print("🔍 PERFORMING EVALUATION (FEW-SHOT CALIBRATION)...")
    print("="*60)

    for test_name, file_path in test_files.items():
        print(f"\n📂 Dataset: [{test_name}]")
        
        try:
            test_data = np.load(file_path)
            X_test_tensor = torch.tensor(test_data["ecgs"], dtype=torch.float32)
            y_test_tensor = torch.tensor(test_data["labels"], dtype=torch.long)
        except Exception as e:
            print(f"    ⚠️ Error loading file: {e}. Skipping.")
            continue
            
        if X_test_tensor.dim() == 2:
            X_test_tensor = X_test_tensor.unsqueeze(1)

        test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=64, shuffle=False)

        all_test_features = []
        all_true_labels = []

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                features = backbone(inputs)['local_reps'].mean(dim=1).cpu().numpy()
                all_test_features.append(features)
                all_true_labels.extend(labels.numpy())

        all_test_features = np.concatenate(all_test_features, axis=0)
        all_true_labels = np.array(all_true_labels)

        # ==============================================================================
        # 3.2: DOMAIN-ADAPTIVE CLASSIFICATION
        # ==============================================================================
        if "MIT-BIH" in test_name:
            print(f"    🔄 Applying CORAL Alignment for the native domain...")
            all_test_features_aligned = apply_coral(source_mu, source_cov, all_test_features)
            
            print(f"    🧠 Predicting with k-NN (Calibrated Data)...")
            all_preds = classifier.predict(all_test_features_aligned)
            eval_labels = all_true_labels # For MIT-BIH, we test the entire set
        else:
            print(f"    🎯 Performing Few-Shot Calibration (10 shots/class) for {test_name}...")
            # Skip CORAL to avoid distorting natural cluster structures, pass raw features directly
            eval_labels, all_preds = few_shot_predict(all_test_features, all_true_labels, n_shots=10)

        # ==============================================================================
        # 4. RESULTS OUTPUT
        # ==============================================================================
        # NOTE: Use eval_labels (which excludes support set samples) instead of all_true_labels
        acc = accuracy_score(eval_labels, all_preds)
        prec = precision_score(eval_labels, all_preds, zero_division=0)
        rec = recall_score(eval_labels, all_preds, zero_division=0)
        cm = confusion_matrix(eval_labels, all_preds)
        
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        else:
            tn, tp = (cm[0,0], 0) if eval_labels[0] == 0 else (0, cm[0,0])
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