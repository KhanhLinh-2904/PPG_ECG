import os
import random
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Thay đổi tên module import cho phù hợp với cấu trúc thư mục của bạn
from load_data import LoadData
from ppg2ecg import PPGtoECGDualBranchReconstructionNet

# =========================================================
# CONFIGURATION
# =========================================================
CONFIG = {
    "data_path": "/home/linhhima/PPG_ECG/datasets/min_max_norm/total_mimic_af_min_max.npz",
    "ckpt_path": "/home/linhhima/PPG_ECG/Disentanglement/saved_models_ppg_to_ecg_dual_branch_min_max/best_model.pth",
    "save_path": "/home/linhhima/PPG_ECG/Disentanglement/results_ppg_to_ecg_dual_branch_test/total_mimic_af_min_max_recon.npz", # Đường dẫn file .npz
    "batch_size": 64,
    "dims": (48, 96, 128, 256),
    "input_len": 2400,
    "num_blocks_per_stage": 2,
    "num_heads": 8,
    "attn_depth_qrs": 2,
    "attn_depth_non_qrs": 2,
    "drop_path": 0.0, # Khi inference nên để drop_path = 0.0
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "num_workers": 4,
    "seed": 42,
}

# =========================================================
# UTILS
# =========================================================
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_checkpoint_state(model: nn.Module, ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    return ckpt

# =========================================================
# MAIN GENERATION & SAVE FUNCTION
# =========================================================
def save_ecg_reconstruction(output_path=CONFIG["save_path"]):
    seed_everything(CONFIG["seed"])
    device = CONFIG["device"]

    print(f"[*] Loading data from: {CONFIG['data_path']}")
    try:
        dataset = LoadData(CONFIG["data_path"])
        # Lưu ý: shuffle=False để giữ nguyên thứ tự dữ liệu (dễ đối chiếu sau này)
        loader = DataLoader(
            dataset, batch_size=CONFIG["batch_size"], 
            shuffle=False, num_workers=CONFIG["num_workers"],
            pin_memory=(device.type == "cuda")
        )
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    print(f"[*] Loading model from: {CONFIG['ckpt_path']}")
    model = PPGtoECGDualBranchReconstructionNet(
        input_len=CONFIG["input_len"],
        dims=CONFIG["dims"],
        num_blocks_per_stage=CONFIG["num_blocks_per_stage"],
        num_heads=CONFIG["num_heads"],
        attn_depth_qrs=CONFIG["attn_depth_qrs"],
        attn_depth_non_qrs=CONFIG["attn_depth_non_qrs"],
        drop_path=CONFIG["drop_path"],
    ).to(device)

    load_checkpoint_state(model, CONFIG["ckpt_path"], device)
    model.eval()

    all_predicted_ecgs = []
    all_ground_truth_ecgs = []
    all_original_ppgs = []
    all_labels = []
    all_record_names = []

    print("[*] Running inference and gathering predictions...")
    with torch.no_grad():
        for batch in tqdm(loader, desc="Reconstructing"):
            # Linh hoạt bóc tách dữ liệu tuỳ thuộc vào DataLoader của bạn
            # Giả định dataset trả về tuple: (ecg, ppg, record_names, label) hoặc dict
            if isinstance(batch, (list, tuple)) and len(batch) >= 4:
                ecg, ppg, record_names, label = batch[0], batch[1], batch[2], batch[3]
            elif isinstance(batch, (list, tuple)):
                ppg, ecg = batch[0], batch[1]
                record_names, label = None, None
            else: # Dict format
                ppg, ecg = batch["ppg"], batch["ecg"]
                record_names = batch.get("record_names", None)
                label = batch.get("label", None)

            # Xử lý shape: đảm bảo có channel dimension [B, 1, L]
            if ppg.ndim == 2: ppg = ppg.unsqueeze(1)
            
            ppg_input = ppg.to(device).float()

            # Forward pass vào Dual Branch Model
            outputs = model(ppg_input, return_features=False)
            predicted_ecg = outputs["ecg_final_pred"] # Lấy output chính thức
            
            # Lưu lại vào list (xoá bỏ dimension Channel (1) để mảng numpy gọn hơn)
            all_predicted_ecgs.append(predicted_ecg.squeeze(1).cpu().numpy())
            all_ground_truth_ecgs.append(ecg.squeeze().cpu().numpy())
            all_original_ppgs.append(ppg.squeeze(1).cpu().numpy())
            
            if label is not None:
                all_labels.append(label.cpu().numpy())
            if record_names is not None:
                all_record_names.extend(record_names) # Dạng list các string

    # ==================================
    # Đóng gói và lưu File
    # ==================================
    save_dict = {
        "ecgs": np.concatenate(all_predicted_ecgs, axis=0),
        "ppgs": np.concatenate(all_original_ppgs, axis=0),
    }
    
    if len(all_labels) > 0:
        save_dict["labels"] = np.concatenate(all_labels, axis=0)
    if len(all_record_names) > 0:
        save_dict["records"] = np.array(all_record_names, dtype=object)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez_compressed(output_path, **save_dict)
    
    print(f"\n[*] DONE! Saved reconstructions to: {output_path}")
    print(f"    - Pred ECG shape: {save_dict['ecgs'].shape}")
    print(f"    - True PPG shape: {save_dict['ppgs'].shape}")

if __name__ == "__main__":
    save_ecg_reconstruction()