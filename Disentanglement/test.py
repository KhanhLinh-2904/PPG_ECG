import os
import random
from typing import Dict, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")   # nếu chạy local có GUI thì có thể comment dòng này
import matplotlib.pyplot as plt

from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from load_data import LoadData
from ppg2ecg import PPGtoECGDualBranchReconstructionNet


CONFIG = {
    "data_path": "/home/linhhima/PPG_ECG/datasets/min_max_norm/total_mimic_af_min_max.npz",
    "ckpt_path": "/home/linhhima/PPG_ECG/Disentanglement/saved_models_ppg_to_ecg_dual_branch_min_max/best_model.pth",
    "save_dir": "results_ppg_to_ecg_dual_branch_test",
    "batch_size": 64,
    "sampling_rate": 125,
    "input_len": 2400,
    "dims": (48, 96, 128, 256),
    "num_blocks_per_stage": 2,
    "num_heads": 8,
    "attn_depth_qrs": 2,
    "attn_depth_non_qrs": 2,
    "drop_path": 0.05,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "num_workers": 4,
    "seed": 44,
    "num_plot_samples": 8,

    # NEW
    "max_shift_samples": 40,   # ví dụ ±40 samples ~ ±320 ms ở fs=125
}


# =========================================================
# 1. UTILS
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


def compute_pearson_and_rmse(pred: np.ndarray, target: np.ndarray):
    pred = pred.reshape(-1)
    target = target.reshape(-1)

    rmse = np.sqrt(np.mean((pred - target) ** 2))

    pred_std = np.std(pred)
    target_std = np.std(target)

    if pred_std < 1e-8 or target_std < 1e-8:
        pearson = 0.0
    else:
        pearson = np.corrcoef(pred, target)[0, 1]

    return float(pearson), float(rmse)


def compute_average_metrics(all_preds, all_targets):
    pearsons = []
    rmses = []

    for pred, target in zip(all_preds, all_targets):
        pearson, rmse = compute_pearson_and_rmse(pred, target)
        pearsons.append(pearson)
        rmses.append(rmse)

    return {
        "pearson_mean": float(np.mean(pearsons)),
        "pearson_std": float(np.std(pearsons)),
        "rmse_mean": float(np.mean(rmses)),
        "rmse_std": float(np.std(rmses)),
        "all_pearson": pearsons,
        "all_rmse": rmses,
    }


def compute_global_metrics(all_preds, all_targets):
    pred_concat = np.concatenate([p.reshape(-1) for p in all_preds], axis=0)
    target_concat = np.concatenate([t.reshape(-1) for t in all_targets], axis=0)

    pearson, rmse = compute_pearson_and_rmse(pred_concat, target_concat)
    return {
        "pearson_global": float(pearson),
        "rmse_global": float(rmse),
    }


# =========================================================
# 2. SHIFT ALIGNMENT
# =========================================================
def crop_by_shift(pred: np.ndarray, target: np.ndarray, shift: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    shift > 0:
        pred bị trễ hơn target -> bỏ bớt phần đầu của pred
    shift < 0:
        pred sớm hơn target -> bỏ bớt phần đầu của target
    """
    pred = pred.reshape(-1)
    target = target.reshape(-1)

    if shift > 0:
        pred_crop = pred[shift:]
        target_crop = target[:len(pred_crop)]
    elif shift < 0:
        s = -shift
        target_crop = target[s:]
        pred_crop = pred[:len(target_crop)]
    else:
        L = min(len(pred), len(target))
        pred_crop = pred[:L]
        target_crop = target[:L]

    L = min(len(pred_crop), len(target_crop))
    pred_crop = pred_crop[:L]
    target_crop = target_crop[:L]

    return pred_crop, target_crop


def find_best_shift_by_pearson(
    pred: np.ndarray,
    target: np.ndarray,
    max_shift: int = 40,
) -> Dict[str, np.ndarray]:
    """
    Tìm shift tốt nhất sao cho Pearson lớn nhất.
    """
    best_shift = 0
    best_pearson = -np.inf
    best_rmse = np.inf
    best_pred = None
    best_target = None

    for shift in range(-max_shift, max_shift + 1):
        pred_crop, target_crop = crop_by_shift(pred, target, shift)

        if len(pred_crop) < 10:
            continue

        pearson, rmse = compute_pearson_and_rmse(pred_crop, target_crop)

        if pearson > best_pearson:
            best_pearson = pearson
            best_rmse = rmse
            best_shift = shift
            best_pred = pred_crop.copy()
            best_target = target_crop.copy()

    return {
        "best_shift": best_shift,
        "pred_aligned": best_pred,
        "target_aligned": best_target,
        "pearson_aligned": float(best_pearson),
        "rmse_aligned": float(best_rmse),
    }


# =========================================================
# 3. PLOTTING
# =========================================================
def plot_ppg_ecg_comparison(
    ppg: np.ndarray,
    ecg_recon: np.ndarray,
    ecg_gt: np.ndarray,
    fs: int = 125,
    title: str = "",
    save_path: str = None,
    show: bool = False,
):
    ppg = ppg.reshape(-1)
    ecg_recon = ecg_recon.reshape(-1)
    ecg_gt = ecg_gt.reshape(-1)

    L = min(len(ppg), len(ecg_recon), len(ecg_gt))
    t = np.arange(L) / fs

    pearson, rmse = compute_pearson_and_rmse(ecg_recon[:L], ecg_gt[:L])

    plt.figure(figsize=(15, 8))

    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(t, ppg[:L], linewidth=1.2)
    ax1.set_title("PPG Input", loc="left")
    ax1.set_ylabel("Amplitude")
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2 = plt.subplot(3, 1, 2)
    ax2.plot(t, ecg_recon[:L], linewidth=1.2)
    ax2.set_title("Reconstructed ECG", loc="left")
    ax2.set_ylabel("Amplitude")
    ax2.grid(True, linestyle="--", alpha=0.5)

    ax3 = plt.subplot(3, 1, 3)
    ax3.plot(t, ecg_gt[:L], linewidth=1.2)
    ax3.set_title(f"Ground Truth ECG | Pearson={pearson:.4f}, RMSE={rmse:.4f}", loc="left")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Amplitude")
    ax3.grid(True, linestyle="--", alpha=0.5)

    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


def plot_overlay_before_after_shift(
    ecg_recon: np.ndarray,
    ecg_gt: np.ndarray,
    ecg_recon_aligned: np.ndarray,
    ecg_gt_aligned: np.ndarray,
    best_shift: int,
    fs: int = 125,
    title: str = "",
    save_path: str = None,
    show: bool = False,
):
    pearson_before, rmse_before = compute_pearson_and_rmse(ecg_recon, ecg_gt)
    pearson_after, rmse_after = compute_pearson_and_rmse(ecg_recon_aligned, ecg_gt_aligned)

    L1 = min(len(ecg_recon), len(ecg_gt))
    t1 = np.arange(L1) / fs

    L2 = min(len(ecg_recon_aligned), len(ecg_gt_aligned))
    t2 = np.arange(L2) / fs

    plt.figure(figsize=(14, 8))

    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(t1, ecg_gt[:L1], label="Ground Truth ECG", linewidth=1.2)
    ax1.plot(t1, ecg_recon[:L1], label="Reconstructed ECG", linewidth=1.2, alpha=0.85)
    ax1.set_title(f"Before shift | Pearson={pearson_before:.4f}, RMSE={rmse_before:.4f}", loc="left")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2 = plt.subplot(2, 1, 2)
    ax2.plot(t2, ecg_gt_aligned[:L2], label="Ground Truth ECG (aligned)", linewidth=1.2)
    ax2.plot(t2, ecg_recon_aligned[:L2], label="Reconstructed ECG (aligned)", linewidth=1.2, alpha=0.85)
    ax2.set_title(f"After shift={best_shift} samples | Pearson={pearson_after:.4f}, RMSE={rmse_after:.4f}", loc="left")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Amplitude")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


# =========================================================
# 4. TEST
# =========================================================
def test(
    data_path: str = None,
    ckpt_path: str = None,
    batch_size: int = None,
    num_workers: int = None,
    num_plot_samples: int = None,
    show_plots: bool = False,
):
    seed_everything(CONFIG["seed"])

    device = CONFIG["device"]
    fs = CONFIG["sampling_rate"]

    data_path = CONFIG["data_path"] if data_path is None else data_path
    ckpt_path = CONFIG["ckpt_path"] if ckpt_path is None else ckpt_path
    batch_size = CONFIG["batch_size"] if batch_size is None else batch_size
    num_workers = CONFIG["num_workers"] if num_workers is None else num_workers
    num_plot_samples = CONFIG["num_plot_samples"] if num_plot_samples is None else num_plot_samples

    os.makedirs(CONFIG["save_dir"], exist_ok=True)
    plot_dir = os.path.join(CONFIG["save_dir"], "plots")
    os.makedirs(plot_dir, exist_ok=True)

    print(f"[*] Device    : {device}")
    print(f"[*] Data path : {data_path}")
    print(f"[*] Ckpt path : {ckpt_path}")

    # -------------------------
    # Data
    # -------------------------
    dataset = LoadData(data_path)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
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

    load_checkpoint_state(model, ckpt_path, device)
    model.eval()

    # BEFORE SHIFT
    all_recon_before = []
    all_gt_before = []

    # AFTER SHIFT
    all_recon_after = []
    all_gt_after = []
    all_best_shifts = []

    plotted = 0

    with torch.no_grad():
        pbar = tqdm(loader, total=len(loader), desc="Testing")

        for batch_idx, batch in enumerate(pbar):
            ppg_raw, ecg_raw = extract_ppg_ecg_from_batch(batch)
            ppg_raw = ppg_raw.to(device=device, dtype=torch.float32)
            ecg_raw = ecg_raw.to(device=device, dtype=torch.float32)

            outputs = model(ppg_raw, return_features=False)
            ecg_recon = outputs["ecg_final_pred"]

            ppg_np = ppg_raw.detach().cpu().numpy()
            ecg_gt_np = ecg_raw.detach().cpu().numpy()
            ecg_recon_np = ecg_recon.detach().cpu().numpy()

            batch_pearson_before = []
            batch_rmse_before = []
            batch_pearson_after = []
            batch_rmse_after = []

            for i in range(ecg_gt_np.shape[0]):
                ppg_i = ppg_np[i, 0]
                gt_i = ecg_gt_np[i, 0]
                recon_i = ecg_recon_np[i, 0]

                # BEFORE SHIFT
                pearson_before, rmse_before = compute_pearson_and_rmse(recon_i, gt_i)
                batch_pearson_before.append(pearson_before)
                batch_rmse_before.append(rmse_before)

                all_recon_before.append(recon_i.copy())
                all_gt_before.append(gt_i.copy())

                # AFTER SHIFT
                align_result = find_best_shift_by_pearson(
                    pred=recon_i,
                    target=gt_i,
                    max_shift=CONFIG["max_shift_samples"],
                )

                recon_aligned = align_result["pred_aligned"]
                gt_aligned = align_result["target_aligned"]
                best_shift = align_result["best_shift"]
                pearson_after = align_result["pearson_aligned"]
                rmse_after = align_result["rmse_aligned"]

                batch_pearson_after.append(pearson_after)
                batch_rmse_after.append(rmse_after)

                all_recon_after.append(recon_aligned.copy())
                all_gt_after.append(gt_aligned.copy())
                all_best_shifts.append(best_shift)

                if plotted < num_plot_samples:
                    sample_id = plotted + 1

                    save_triple = os.path.join(plot_dir, f"sample_{sample_id}_ppg_ecg.png")
                    plot_ppg_ecg_comparison(
                        ppg=ppg_i,
                        ecg_recon=recon_i,
                        ecg_gt=gt_i,
                        fs=fs,
                        title=f"Sample {sample_id}",
                        save_path=save_triple,
                        show=show_plots,
                    )

                    save_shift = os.path.join(plot_dir, f"sample_{sample_id}_before_after_shift.png")
                    plot_overlay_before_after_shift(
                        ecg_recon=recon_i,
                        ecg_gt=gt_i,
                        ecg_recon_aligned=recon_aligned,
                        ecg_gt_aligned=gt_aligned,
                        best_shift=best_shift,
                        fs=fs,
                        title=f"Sample {sample_id}",
                        save_path=save_shift,
                        show=show_plots,
                    )

                    plotted += 1

            pbar.set_postfix({
                "P_before": f"{np.mean(batch_pearson_before):.4f}",
                "R_before": f"{np.mean(batch_rmse_before):.4f}",
                "P_after": f"{np.mean(batch_pearson_after):.4f}",
                "R_after": f"{np.mean(batch_rmse_after):.4f}",
            })

    # -------------------------
    # FINAL METRICS
    # -------------------------
    avg_before = compute_average_metrics(all_recon_before, all_gt_before)
    global_before = compute_global_metrics(all_recon_before, all_gt_before)

    avg_after = compute_average_metrics(all_recon_after, all_gt_after)
    global_after = compute_global_metrics(all_recon_after, all_gt_after)

    print("\n===== TEST RESULT: BEFORE SHIFT =====")
    print("[1] Average per-sample metrics")
    print(f"Mean Pearson Correlation : {avg_before['pearson_mean']:.6f} ± {avg_before['pearson_std']:.6f}")
    print(f"Mean RMSE                : {avg_before['rmse_mean']:.6f} ± {avg_before['rmse_std']:.6f}")

    print("\n[2] Global concatenated metrics")
    print(f"Global Pearson Correlation : {global_before['pearson_global']:.6f}")
    print(f"Global RMSE                : {global_before['rmse_global']:.6f}")

    print("\n===== TEST RESULT: AFTER SHIFT =====")
    print("[1] Average per-sample metrics")
    print(f"Mean Pearson Correlation : {avg_after['pearson_mean']:.6f} ± {avg_after['pearson_std']:.6f}")
    print(f"Mean RMSE                : {avg_after['rmse_mean']:.6f} ± {avg_after['rmse_std']:.6f}")

    print("\n[2] Global concatenated metrics")
    print(f"Global Pearson Correlation : {global_after['pearson_global']:.6f}")
    print(f"Global RMSE                : {global_after['rmse_global']:.6f}")

    print(f"\nMean best shift (samples): {np.mean(all_best_shifts):.4f}")
    print(f"Saved plots in: {plot_dir}")

    return {
        "before_shift": {
            **avg_before,
            **global_before,
        },
        "after_shift": {
            **avg_after,
            **global_after,
        },
        "best_shifts": all_best_shifts,
    }


if __name__ == "__main__":
    result = test(
        data_path=CONFIG["data_path"],
        ckpt_path=CONFIG["ckpt_path"],
        batch_size=CONFIG["batch_size"],
        num_workers=CONFIG["num_workers"],
        num_plot_samples=CONFIG["num_plot_samples"],
        show_plots=False,
    )