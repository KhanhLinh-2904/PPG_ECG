import os
import random
from typing import Dict, Tuple

import numpy as np
import neurokit2 as nk
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from load_data import LoadData
from ppg2ecg import PPGtoECGDualBranchReconstructionNet


CONFIG = {
    "data_path": "/home/linhhima/PPG_ECG/version3/processed_data/mimic3_signal_split_train.npz",
    "save_dir": "saved_models_ppg_to_ecg_dual_branch",
    "batch_size": 64,
    "epochs": 200,
    "lr": 3e-4,
    "weight_decay": 1e-4,
    "sampling_rate": 125,
    "input_len": 2400,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "num_workers": 4,
    "use_amp": True,
    "grad_clip": 1.0,
    "save_every": 10,
    "seed": 42,

    # QRS mask
    "qrs_dilate": 21,

    # model
    "dims": (48, 96, 128, 256),
    "num_blocks_per_stage": 2,
    "num_heads": 8,
    "attn_depth_qrs": 2,
    "attn_depth_non_qrs": 2,
    "drop_path": 0.05,

    # augmentation
    "use_aug": True,
    "gain_min": 0.95,
    "gain_max": 1.05,
    "noise_std_max": 0.005,

    # loss weights
    "lambda_qrs": 1.0,
    "lambda_non": 0.7,
    "lambda_final": 1.2,
    "lambda_slope": 0.2,
    "lambda_gate": 0.1,
    "lambda_ortho": 0.02,
    "lambda_leak": 0.2,
}


# =========================================================
# 1. UTILS
# =========================================================
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def extract_ppg_ecg_from_batch(batch):
    """
    Hỗ trợ:
    - (ppg, ecg, ...)
    - {"ppg": ..., "ecg": ...}
    """
    if isinstance(batch, dict):
        ppg = batch["ppg"]
        ecg = batch["ecg"]
    elif isinstance(batch, (list, tuple)):
        if len(batch) < 2:
            raise ValueError("Batch phải chứa ít nhất (ppg, ecg).")
        ppg, ecg = batch[0], batch[1]
    else:
        raise ValueError(f"Batch type không hỗ trợ: {type(batch)}")

    if ppg.ndim == 2:
        ppg = ppg.unsqueeze(1)
    if ecg.ndim == 2:
        ecg = ecg.unsqueeze(1)

    return ppg, ecg


def generate_qrs_mask_neurokit(
    ecg_batch: torch.Tensor,
    sampling_rate: int = 125,
    dilate_size: int = 21,
) -> torch.Tensor:
    """
    ecg_batch: [B, 1, L]
    return: [B, 1, L] binary mask 0/1
    """
    assert ecg_batch.ndim == 3 and ecg_batch.shape[1] == 1, \
        f"Expected ecg_batch [B,1,L], got {ecg_batch.shape}"

    batch_size, _, length = ecg_batch.shape
    mask = torch.zeros_like(ecg_batch, dtype=torch.float32)

    ecg_np = ecg_batch.squeeze(1).detach().cpu().numpy()
    pad = dilate_size // 2

    for i in range(batch_size):
        signal = ecg_np[i]
        try:
            _, info = nk.ecg_peaks(
                signal,
                sampling_rate=sampling_rate,
                method="pantompkins1985",
            )
            rpeaks = info.get("ECG_R_Peaks", [])
            for r in rpeaks:
                start = max(0, int(r) - pad)
                end = min(length, int(r) + pad + 1)
                mask[i, 0, start:end] = 1.0
        except Exception:
            continue

    return mask


def align_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(pred, target) + 0.5 * F.mse_loss(pred, target)


def masked_align_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Tính L1 + 0.5*MSE chỉ trên vùng mask
    """
    diff = pred - target
    abs_term = torch.abs(diff) * mask
    sq_term = (diff ** 2) * mask

    l1 = abs_term.sum() / (mask.sum() + eps)
    mse = sq_term.sum() / (mask.sum() + eps)

    return l1 + 0.5 * mse


def first_derivative(x: torch.Tensor) -> torch.Tensor:
    return x[:, :, 1:] - x[:, :, :-1]


# =========================================================
# 2. LOSS
# =========================================================
class DualBranchReconstructionLoss(nn.Module):
    """
    Loss cho model:
    - ecg_qrs_pred
    - ecg_non_qrs_pred
    - ecg_final_pred
    """
    def __init__(
        self,
        lambda_qrs: float = 1.0,
        lambda_non: float = 0.7,
        lambda_final: float = 1.2,
        lambda_slope: float = 0.2,
        lambda_gate: float = 0.1,
        lambda_ortho: float = 0.02,
        lambda_leak: float = 0.2,
    ):
        super().__init__()
        self.lambda_qrs = lambda_qrs
        self.lambda_non = lambda_non
        self.lambda_final = lambda_final
        self.lambda_slope = lambda_slope
        self.lambda_gate = lambda_gate
        self.lambda_ortho = lambda_ortho
        self.lambda_leak = lambda_leak

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        ecg_raw: torch.Tensor,
        qrs_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        ecg_qrs_pred = outputs["ecg_qrs_pred"]
        ecg_non_qrs_pred = outputs["ecg_non_qrs_pred"]
        ecg_final_pred = outputs["ecg_final_pred"]
        gate = outputs["gate"]

        z_qrs = outputs["z_qrs"]
        z_non_qrs = outputs["z_non_qrs"]

        non_qrs_mask = 1.0 - qrs_mask
        zero = torch.zeros_like(ecg_raw)

        # QRS branch: tái tạo đúng vùng QRS + suppress ngoài vùng QRS
        loss_qrs_main = masked_align_loss(ecg_qrs_pred, ecg_raw, qrs_mask)
        loss_qrs_leak = masked_align_loss(ecg_qrs_pred, zero, non_qrs_mask)
        loss_qrs = loss_qrs_main + self.lambda_leak * loss_qrs_leak

        # non-QRS branch: tái tạo đúng vùng non-QRS + suppress trong vùng QRS
        loss_non_main = masked_align_loss(ecg_non_qrs_pred, ecg_raw, non_qrs_mask)
        loss_non_leak = masked_align_loss(ecg_non_qrs_pred, zero, qrs_mask)
        loss_non = loss_non_main + self.lambda_leak * loss_non_leak

        # Final ECG reconstruction
        loss_final = align_loss(ecg_final_pred, ecg_raw)

        # Slope / morphology loss
        d_pred = first_derivative(ecg_final_pred)
        d_real = first_derivative(ecg_raw)
        loss_slope = F.l1_loss(d_pred, d_real)

        # Gate regularization: vùng QRS ưu tiên branch QRS
        loss_gate = F.mse_loss(gate, qrs_mask)

        # Orthogonality nhẹ giữa 2 latent
        zq = F.normalize(z_qrs, p=2, dim=1)
        zn = F.normalize(z_non_qrs, p=2, dim=1)
        loss_ortho = torch.mean(torch.abs(torch.sum(zq * zn, dim=1)))

        total = (
            self.lambda_qrs * loss_qrs
            + self.lambda_non * loss_non
            + self.lambda_final * loss_final
            + self.lambda_slope * loss_slope
            + self.lambda_gate * loss_gate
            + self.lambda_ortho * loss_ortho
        )

        metrics = {
            "loss_total": total.detach(),
            "loss_qrs": loss_qrs.detach(),
            "loss_non": loss_non.detach(),
            "loss_final": loss_final.detach(),
            "loss_slope": loss_slope.detach(),
            "loss_gate": loss_gate.detach(),
            "loss_ortho": loss_ortho.detach(),
            "loss_qrs_main": loss_qrs_main.detach(),
            "loss_qrs_leak": loss_qrs_leak.detach(),
            "loss_non_main": loss_non_main.detach(),
            "loss_non_leak": loss_non_leak.detach(),
        }
        return total, metrics


# =========================================================
# 3. TRAIN
# =========================================================
def train():
    seed_everything(CONFIG["seed"])
    os.makedirs(CONFIG["save_dir"], exist_ok=True)

    device = CONFIG["device"]
    print(f"[*] Device: {device}")

    # -------------------------
    # Data
    # -------------------------
    dataset = LoadData(CONFIG["data_path"])
    loader = DataLoader(
        dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=CONFIG["num_workers"],
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    # -------------------------
    # Model
    # -------------------------
    model = PPGtoECGDualBranchReconstructionNet(
        input_len=CONFIG["input_len"],
        dims=CONFIG["dims"],
        num_blocks_per_stage=CONFIG["num_blocks_per_stage"],
        num_heads=CONFIG["num_heads"],
        attn_depth_qrs=CONFIG["attn_depth_qrs"],
        attn_depth_non_qrs=CONFIG["attn_depth_non_qrs"],
        drop_path=CONFIG["drop_path"],
    ).to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=CONFIG["lr"],
        weight_decay=CONFIG["weight_decay"],
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=CONFIG["epochs"],
    )

    criterion = DualBranchReconstructionLoss(
        lambda_qrs=CONFIG["lambda_qrs"],
        lambda_non=CONFIG["lambda_non"],
        lambda_final=CONFIG["lambda_final"],
        lambda_slope=CONFIG["lambda_slope"],
        lambda_gate=CONFIG["lambda_gate"],
        lambda_ortho=CONFIG["lambda_ortho"],
        lambda_leak=CONFIG["lambda_leak"],
    ).to(device)

    scaler = torch.cuda.amp.GradScaler(
        enabled=(CONFIG["use_amp"] and device.type == "cuda")
    )

    best_loss = float("inf")
    print("[*] Start training dual-branch PPG->ECG model ...")

    for epoch in range(CONFIG["epochs"]):
        model.train()

        running = {
            "loss_total": 0.0,
            "loss_qrs": 0.0,
            "loss_non": 0.0,
            "loss_final": 0.0,
            "loss_slope": 0.0,
            "loss_gate": 0.0,
            "loss_ortho": 0.0,
        }

        pbar = tqdm(loader, total=len(loader), desc=f"Epoch {epoch + 1}/{CONFIG['epochs']}")

        for step, batch in enumerate(pbar):
            ppg_raw, ecg_raw = extract_ppg_ecg_from_batch(batch)
            ppg_raw = ppg_raw.to(device=device, dtype=torch.float32)
            ecg_raw = ecg_raw.to(device=device, dtype=torch.float32)

            # augmentation nhẹ trên PPG
            ppg_input = ppg_raw
            if CONFIG["use_aug"]:
                gain = random.uniform(CONFIG["gain_min"], CONFIG["gain_max"])
                noise_std = random.uniform(0.0, CONFIG["noise_std_max"])
                ppg_input = ppg_input * gain + torch.randn_like(ppg_input) * noise_std

            # QRS mask từ ECG ground truth
            qrs_mask = generate_qrs_mask_neurokit(
                ecg_raw.detach().cpu(),
                sampling_rate=CONFIG["sampling_rate"],
                dilate_size=CONFIG["qrs_dilate"],
            ).to(device)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(CONFIG["use_amp"] and device.type == "cuda")):
                outputs = model(ppg_input, return_features=True)
                total_loss, metrics = criterion(outputs, ecg_raw, qrs_mask)

            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip"])
            scaler.step(optimizer)
            scaler.update()

            for k in running:
                running[k] += metrics[k].item()

            avg = {k: running[k] / (step + 1) for k in running}
            pbar.set_postfix({
                "Tot": f"{avg['loss_total']:.4f}",
                "QRS": f"{avg['loss_qrs']:.4f}",
                "NON": f"{avg['loss_non']:.4f}",
                "Final": f"{avg['loss_final']:.4f}",
            })

        scheduler.step()
        epoch_loss = running["loss_total"] / len(loader)
        lr_now = optimizer.param_groups[0]["lr"]

        print(
            f"[Epoch {epoch + 1:03d}] "
            f"loss={epoch_loss:.6f} | "
            f"qrs={running['loss_qrs']/len(loader):.6f} | "
            f"non={running['loss_non']/len(loader):.6f} | "
            f"final={running['loss_final']/len(loader):.6f} | "
            f"slope={running['loss_slope']/len(loader):.6f} | "
            f"gate={running['loss_gate']/len(loader):.6f} | "
            f"ortho={running['loss_ortho']/len(loader):.6f} | "
            f"lr={lr_now:.8f}"
        )

        # save best
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_path = os.path.join(CONFIG["save_dir"], "best_model.pth")
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_loss": best_loss,
                "config": CONFIG,
            }, best_path)
            print(f"✅ Saved best model -> {best_path}")

        # periodic save
        if (epoch + 1) % CONFIG["save_every"] == 0 or (epoch + 1) == CONFIG["epochs"]:
            ckpt_path = os.path.join(CONFIG["save_dir"], f"epoch_{epoch + 1}.pth")
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "loss": epoch_loss,
                "config": CONFIG,
            }, ckpt_path)
            print(f"💾 Saved checkpoint -> {ckpt_path}")

    print("[*] Done training.")


if __name__ == "__main__":
    train()