import os
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

# Libraries for calculating advanced evaluation metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from ecg_transform import ECGTransformerModel, ECGAFClassifier

if __name__ == "__main__":
    # 1. Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Starting testing process on device: {device}")

    # ==============================================================================
    # 2. LOAD TEST DATA
    # ==============================================================================
    print("⏳ Loading Test Data...")
    try:
        # Replace with your actual test data file
        test_data = np.load('AF_Detection/deepbeat_ecg_reconstructions.npz')
        X_test = torch.tensor(test_data["ecgs"], dtype=torch.float32)
        y_test = torch.tensor(test_data["labels"], dtype=torch.long)
    except FileNotFoundError:
        print("⚠️ Data file not found. Using Dummy Data for testing purposes.")
        X_test = torch.randn(50, 1, 2400) # 50 test samples
        y_test = torch.randint(0, 2, (50,))

    # Ensure shape format is [Batch, Channels, Length]
    if X_test.dim() == 2:
        X_test = X_test.unsqueeze(1)

    print(f"Test Set Size: {X_test.shape}")
    
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False) # No need to shuffle during testing

    # ==============================================================================
    # 3. RE-INITIALIZE MODEL ARCHITECTURE (Must exactly match Training)
    # ==============================================================================
    print("🧠 Initializing model architecture...")
    EMBED_DIM = 256
    
    transformer_encoder = nn.TransformerEncoder(
        nn.TransformerEncoderLayer(d_model=EMBED_DIM, nhead=8, dim_feedforward=1024, dropout=0.1, batch_first=True), 
        num_layers=4
    )
    
    backbone = ECGTransformerModel(
        in_channels=X_test.shape[1],
        embed_dim=EMBED_DIM,
        conv_layers=[(256, 10, 5), (256, 3, 2), (256, 3, 2)], 
        vq_dim=EMBED_DIM,
        transformer_encoder=transformer_encoder
    )

    model = ECGAFClassifier(
        backbone=backbone,
        embed_dim=EMBED_DIM,
        hidden_dim=int(EMBED_DIM/2), # Added based on your previous architectural updates
        num_classes=2,
        dropout_prob=0.0, # Automatically disabled during eval, setting to 0 to be safe
        freeze_backbone=True 
    ).to(device)

    # ==============================================================================
    # 4. LOAD TRAINED WEIGHTS (CHECKPOINT)
    # ==============================================================================
    MODEL_PATH = "AF_Detection/checkpoints/final_af_classifier_unfreezed.pth" # Updated to load the best model
    
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print(f"✅ Successfully loaded weights from: {MODEL_PATH}")
    else:
        print(f"❌ ERROR: Weight file not found at {MODEL_PATH}. Please check the path!")
        exit()

    # ==============================================================================
    # 5. INFERENCE LOOP
    # ==============================================================================
    print("\n" + "="*50)
    print("🔍 PERFORMING PREDICTIONS...")
    print("="*50)
    
    # Switch model to Evaluation mode (CRITICAL)
    # Disables Dropout, turns off Batch/Layer Norm learning
    model.eval() 
    
    all_preds = []
    all_labels = []

    # Enable no_grad context to disable gradient calculation, boosting speed and saving RAM
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            logits = model(inputs)
            
            # Get the class with the highest probability (0 or 1)
            _, predictions = torch.max(logits, dim=1)
            
            # Move results from GPU back to CPU for sklearn calculations
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # ==============================================================================
    # 6. CALCULATE AND PRINT EVALUATION METRICS
    # ==============================================================================
    
    # Calculate basic metrics
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    
    # Generate Confusion Matrix to extract TP, TN, FP, FN
    cm = confusion_matrix(all_labels, all_preds)
    
    # Unpack Confusion Matrix (Assuming binary classification: 0=Negative, 1=Positive)
    # cm format: [[TN, FP],
    #             [FN, TP]]
    tn, fp, fn, tp = cm.ravel()
    
    # Calculate Specificity (True Negative Rate)
    # Formula: TN / (TN + FP)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    total_samples = len(all_labels)

    print("\n📊 TEST REPORT:")
    print("-" * 40)
    print(f"  • Accuracy        : {acc * 100:.2f}%")
    print(f"  • Precision       : {prec * 100:.2f}%")
    print(f"  • Recall (Sensitivity): {rec * 100:.2f}%")
    print(f"  • Specificity     : {specificity * 100:.2f}%")
    print("-" * 40)
    
    print("\n🔢 DETAILED COUNTS:")
    print("-" * 40)
    print(f"  • True Positives (TP) : {tp}")
    print(f"  • True Negatives (TN) : {tn}")
    print(f"  • False Positives (FP): {fp}")
    print(f"  • False Negatives (FN): {fn}")
    print(f"  • Total Samples       : {total_samples}")
    print("-" * 40)
    
    print("\n📉 CONFUSION MATRIX:")
    print("                  Predicted Non-AF (0) | Predicted AF (1)")
    print(f"Actual Non-AF (0) |        {tn:<17} |      {fp:<12}")
    print(f"Actual AF (1)     |        {fn:<17} |      {tp:<12}")