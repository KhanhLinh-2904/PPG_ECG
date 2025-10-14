# test.py
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from load_data_ppg import LoadData
from model import  Res34SimSiamNoise
AE_CHECKPOINT = "checkpoint_epoch_192.pth"
DIM1, DIM2 = 512, 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Remove 'module.' prefix if present
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in checkpoint.items():
        new_key = k.replace("module.", "")  # strip "module."
        new_state_dict[new_key] = v

    model.load_state_dict(new_state_dict, strict=False)
    return model

def evaluate(model, dataloader, single = True, ecg_only = True):
    model.eval()
    all_preds_ECG, all_preds_PPG, all_targets = [], [], []

    with torch.no_grad():
        if single:
            for  data in dataloader:
                ecg = data[0]
                target =  data[1]
                # print("target: ", target)
                ecg, target = ecg.unsqueeze(1).to(device), target.to(device)

                # _, _ , class_pred = model(ecg)  # forward pass
                if ecg_only:
                    class_pred = model(ecg, PPG=None)  # forward pass
                else:
                    class_pred = model(ECG=None, PPG=ecg)  
                # print("class_pred: ", class_pred)
                # preds_ECG = torch.argmax(class_pred, dim=1)
                # print("class_pred: ", class_pred)

                preds_ECG = (class_pred > 0.5).int()

                all_preds_ECG.extend(preds_ECG.cpu().numpy())
                all_targets.extend(target.cpu().numpy())

            acc = accuracy_score(all_targets, all_preds_ECG)
            prec = precision_score(all_targets, all_preds_ECG, zero_division=0)
            rec = recall_score(all_targets, all_preds_ECG, zero_division=0)
            f1 = f1_score(all_targets, all_preds_ECG, zero_division=0)

            return acc, prec, rec, f1, all_preds_ECG, all_targets
        else:
            for  data in dataloader:
                ppg = data[0]
                ecg = data[1]
                target =  data[2]
                # print("target: ", target)
                ecg, ppg, target = ecg.unsqueeze(1).to(device), ppg.unsqueeze(1).to(device), target.to(device)

                _, _ ,_ , _, class_pred1, class_pred2 = model(ecg, ppg)  # forward pass
              
                preds_ECG = (class_pred1 > 0.5).int()
                preds_PPG = (class_pred2 > 0.5).int()

                # print("preds_ECG: ", preds_ECG)

                all_preds_ECG.extend(preds_ECG.cpu().numpy())
                all_preds_PPG.extend(preds_PPG.cpu().numpy())
                all_targets.extend(target.cpu().numpy())

            # acc = accuracy_score(all_targets, all_preds_ECG)
            # prec = precision_score(all_targets, all_preds_ECG, zero_division=0)
            # rec = recall_score(all_targets, all_preds_ECG, zero_division=0)
            # f1 = f1_score(all_targets, all_preds_ECG, zero_division=0)

            acc = accuracy_score(all_targets, all_preds_PPG)
            prec = precision_score(all_targets, all_preds_PPG, zero_division=0)
            rec = recall_score(all_targets, all_preds_PPG, zero_division=0)
            f1 = f1_score(all_targets, all_preds_PPG, zero_division=0)

            return acc, prec, rec, f1, all_preds_PPG, all_targets

if __name__ == "__main__":
    # === Config ===
    test_data_path = "datasets/deepbeat_data_extract/test_cleaned_signals.npz"
    checkpoint_path = "saved_models/model_21.pt"   # adjust if needed
    batch_size = 64  
    # print("checkpoint_path: ", checkpoint_path)
    # === Load Dataset ===
    test_dataset = LoadData(test_data_path)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # === Load Model ===
    model = Res34SimSiamNoise(DIM1, DIM2, single_source_mode=True)         # wrap encoder with classifier if used
    model = load_checkpoint(model, checkpoint_path)
    model = model.to(device)


    # === Evaluate ===
    acc, prec, rec, f1, preds, targets = evaluate(model, test_loader, single=True, ecg_only=False)
    print("\n📊 Test Results:")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")

    # (Optional) Save predictions
    np.savez("test_predictions.npz", preds=preds, targets=targets)
    print("💾 Predictions saved to test_predictions.npz")
