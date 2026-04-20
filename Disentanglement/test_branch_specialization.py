import os
import json
import random
from typing import Dict, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")   # nếu chạy local có GUI thì có thể comment dòng này
import matplotlib.pyplot as plt

import neurokit2 as nk
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from load_data import LoadData
from ppg2ecg import PPGtoECGDualBranchReconstructionNet


CONFIG = {
    "data_path": "/home/linhhima/PPG_ECG/version3/processed_data/mimic3_signal_split_test.npz",
    "ckpt_path": "saved_models_ppg_to_ecg_dual_branch/best_model.pth",
    "save_dir": "results_branch_specialization_test",
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
    "seed": 42,
    "num_plot_samples": 8,
    "eps": 1e-8,
}


# =========================================================
# 1. BASIC UTILS
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


def masked_rmse(pred: np.ndarray, target: np.ndarray, mask: np.ndarray, eps: float = 1e-8):
    pred = pred.reshape(-1)
    target = target.reshape(-1)
    mask = mask.reshape(-1)

    se = ((pred - target) ** 2) * mask
    denom = np.sum(mask) + eps
    return float(np.sqrt(np.sum(se) / denom))


def masked_pearson(pred: np.ndarray, target: np.ndarray, mask: np.ndarray):
    pred = pred.reshape(-1)
    target = target.reshape(-1)
    mask = mask.reshape(-1) > 0.5

    if np.sum(mask) < 10:
        return 0.0

    pred_m = pred[mask]
    target_m = target[mask]
    return compute_pearson_and_rmse(pred_m, target_m)[0]


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
    }


# =========================================================
# 2. QRS MASK
# =========================================================
def generate_qrs_mask_neurokit(
    ecg_batch: torch.Tensor,
    sampling_rate: int = 125,
    dilate_size: int = 21,
) -> torch.Tensor:
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


# =========================================================
# 3. BRANCH SPECIALIZATION METRICS
# =========================================================
def cosine_similarity_flat(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8):
    a = a.flatten(1)
    b = b.flatten(1)
    a = a / (a.norm(dim=1, keepdim=True) + eps)
    b = b / (b.norm(dim=1, keepdim=True) + eps)
    return (a * b).sum(dim=1)


def component_energy(x: np.ndarray, mask: np.ndarray, eps: float = 1e-8):
    x = x.reshape(-1)
    mask = mask.reshape(-1)
    return float(np.sum((x ** 2) * mask) / (np.sum(mask) + eps))


def specialization_metrics_per_sample(
    qrs_pred: np.ndarray,
    non_pred: np.ndarray,
    ecg_gt: np.ndarray,
    qrs_mask: np.ndarray,
    latent_qrs: np.ndarray,
    latent_non: np.ndarray,
    eps: float = 1e-8,
):
    non_mask = 1.0 - qrs_mask
    zero = np.zeros_like(ecg_gt)

    # Branch quality on correct regions
    qrs_fit_rmse = masked_rmse(qrs_pred, ecg_gt, qrs_mask)
    non_fit_rmse = masked_rmse(non_pred, ecg_gt, non_mask)

    qrs_fit_pearson = masked_pearson(qrs_pred, ecg_gt, qrs_mask)
    non_fit_pearson = masked_pearson(non_pred, ecg_gt, non_mask)

    # Leakage
    qrs_leak_rmse = masked_rmse(qrs_pred, zero, non_mask)
    non_leak_rmse = masked_rmse(non_pred, zero, qrs_mask)

    # Energy purity
    qrs_energy_total = component_energy(qrs_pred, np.ones_like(qrs_mask), eps)
    qrs_energy_on_qrs = component_energy(qrs_pred, qrs_mask, eps)
    qrs_energy_on_non = component_energy(qrs_pred, non_mask, eps)

    non_energy_total = component_energy(non_pred, np.ones_like(qrs_mask), eps)
    non_energy_on_qrs = component_energy(non_pred, qrs_mask, eps)
    non_energy_on_non = component_energy(non_pred, non_mask, eps)

    qrs_purity = qrs_energy_on_qrs / (qrs_energy_total + eps)
    non_purity = non_energy_on_non / (non_energy_total + eps)

    qrs_leak_ratio = qrs_energy_on_non / (qrs_energy_total + eps)
    non_leak_ratio = non_energy_on_qrs / (non_energy_total + eps)

    # Collapse check: branch energy compared to target energy on intended region
    gt_qrs_energy = component_energy(ecg_gt, qrs_mask, eps)
    gt_non_energy = component_energy(ecg_gt, non_mask, eps)

    qrs_energy_ratio = qrs_energy_on_qrs / (gt_qrs_energy + eps)
    non_energy_ratio = non_energy_on_non / (gt_non_energy + eps)

    # Redundancy
    qrs_pred_flat = qrs_pred.reshape(-1)
    non_pred_flat = non_pred.reshape(-1)
    comp_cos = np.dot(qrs_pred_flat, non_pred_flat) / (
        (np.linalg.norm(qrs_pred_flat) + eps) * (np.linalg.norm(non_pred_flat) + eps)
    )

    latent_qrs_flat = latent_qrs.reshape(-1)
    latent_non_flat = latent_non.reshape(-1)
    latent_cos = np.dot(latent_qrs_flat, latent_non_flat) / (
        (np.linalg.norm(latent_qrs_flat) + eps) * (np.linalg.norm(latent_non_flat) + eps)
    )

    return {
        "qrs_fit_rmse": float(qrs_fit_rmse),
        "qrs_fit_pearson": float(qrs_fit_pearson),
        "non_fit_rmse": float(non_fit_rmse),
        "non_fit_pearson": float(non_fit_pearson),
        "qrs_leak_rmse": float(qrs_leak_rmse),
        "non_leak_rmse": float(non_leak_rmse),
        "qrs_purity": float(qrs_purity),
        "non_purity": float(non_purity),
        "qrs_leak_ratio": float(qrs_leak_ratio),
        "non_leak_ratio": float(non_leak_ratio),
        "qrs_energy_ratio": float(qrs_energy_ratio),
        "non_energy_ratio": float(non_energy_ratio),
        "component_cosine": float(comp_cos),
        "latent_cosine": float(latent_cos),
    }


def summarize_specialization(all_specs):
    keys = all_specs[0].keys()
    summary = {}
    for k in keys:
        vals = [x[k] for x in all_specs]
        summary[k] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
        }

    # Heuristic flags
    flags = {}

    qrs_collapse = summary["qrs_energy_ratio"]["mean"] < 0.10
    non_collapse = summary["non_energy_ratio"]["mean"] < 0.10

    severe_redundancy = (
        summary["latent_cosine"]["mean"] > 0.90 and
        summary["component_cosine"]["mean"] > 0.90
    )

    flags["qrs_branch_collapsed"] = bool(qrs_collapse)
    flags["non_qrs_branch_collapsed"] = bool(non_collapse)
    flags["branches_highly_redundant"] = bool(severe_redundancy)

    summary["flags"] = flags
    return summary


# =========================================================
# 4. RECONSTRUCTION MODES
# =========================================================
def reconstruct_modes(model, outputs):
    """
    full       : dùng cả 2 branch như bình thường
    qrs_only   : chỉ giữ ecg_qrs_pred, non_qrs = 0
    non_only   : chỉ giữ ecg_non_qrs_pred, qrs = 0
    """
    qrs = outputs["ecg_qrs_pred"]
    non = outputs["ecg_non_qrs_pred"]

    zero_qrs = torch.zeros_like(qrs)
    zero_non = torch.zeros_like(non)

    full = outputs["ecg_final_pred"]
    qrs_only = model.fusion_head(qrs, zero_non)["ecg_final_pred"]
    non_only = model.fusion_head(zero_qrs, non)["ecg_final_pred"]

    return {
        "full": full,
        "qrs_only": qrs_only,
        "non_only": non_only,
    }


# =========================================================
# 5. PLOTTING
# =========================================================
def plot_branch_ablation(
    ppg: np.ndarray,
    ecg_gt: np.ndarray,
    ecg_full: np.ndarray,
    ecg_qrs_only: np.ndarray,
    ecg_non_only: np.ndarray,
    fs: int = 125,
    title: str = "",
    save_path: str = None,
    show: bool = False,
):
    ppg = ppg.reshape(-1)
    ecg_gt = ecg_gt.reshape(-1)
    ecg_full = ecg_full.reshape(-1)
    ecg_qrs_only = ecg_qrs_only.reshape(-1)
    ecg_non_only = ecg_non_only.reshape(-1)

    L = min(len(ppg), len(ecg_gt), len(ecg_full), len(ecg_qrs_only), len(ecg_non_only))
    t = np.arange(L) / fs

    p_full, r_full = compute_pearson_and_rmse(ecg_full[:L], ecg_gt[:L])
    p_qrs, r_qrs = compute_pearson_and_rmse(ecg_qrs_only[:L], ecg_gt[:L])
    p_non, r_non = compute_pearson_and_rmse(ecg_non_only[:L], ecg_gt[:L])

    plt.figure(figsize=(15, 12))

    ax1 = plt.subplot(4, 1, 1)
    ax1.plot(t, ppg[:L], linewidth=1.1)
    ax1.set_title("PPG Input", loc="left")
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2 = plt.subplot(4, 1, 2)
    ax2.plot(t, ecg_gt[:L], label="GT ECG", linewidth=1.2)
    ax2.plot(t, ecg_full[:L], label="Full Recon", linewidth=1.1, alpha=0.85)
    ax2.set_title(f"Full reconstruction | Pearson={p_full:.4f}, RMSE={r_full:.4f}", loc="left")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.5)

    ax3 = plt.subplot(4, 1, 3)
    ax3.plot(t, ecg_gt[:L], label="GT ECG", linewidth=1.2)
    ax3.plot(t, ecg_qrs_only[:L], label="QRS-only Recon", linewidth=1.1, alpha=0.85)
    ax3.set_title(f"QRS-only reconstruction | Pearson={p_qrs:.4f}, RMSE={r_qrs:.4f}", loc="left")
    ax3.legend()
    ax3.grid(True, linestyle="--", alpha=0.5)

    ax4 = plt.subplot(4, 1, 4)
    ax4.plot(t, ecg_gt[:L], label="GT ECG", linewidth=1.2)
    ax4.plot(t, ecg_non_only[:L], label="Non-QRS-only Recon", linewidth=1.1, alpha=0.85)
    ax4.set_title(f"Non-QRS-only reconstruction | Pearson={p_non:.4f}, RMSE={r_non:.4f}", loc="left")
    ax4.legend()
    ax4.grid(True, linestyle="--", alpha=0.5)

    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.98])

    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


# =========================================================
# 6. TEST
# =========================================================
def test_branch_specialization(
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
    plot_dir = os.path.join(CONFIG["save_dir"], "plots_branch_specialization")
    os.makedirs(plot_dir, exist_ok=True)

    print(f"[*] Device    : {device}")
    print(f"[*] Data path : {data_path}")
    print(f"[*] Ckpt path : {ckpt_path}")

    dataset = LoadData(data_path)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

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

    # reconstruction metrics
    all_full = []
    all_qrs_only = []
    all_non_only = []
    all_gt = []

    # specialization metrics
    all_specs = []

    plotted = 0

    with torch.no_grad():
        pbar = tqdm(loader, total=len(loader), desc="Testing specialization")

        for batch_idx, batch in enumerate(pbar):
            ppg_raw, ecg_raw = extract_ppg_ecg_from_batch(batch)
            ppg_raw = ppg_raw.to(device=device, dtype=torch.float32)
            ecg_raw = ecg_raw.to(device=device, dtype=torch.float32)

            qrs_mask = generate_qrs_mask_neurokit(
                ecg_raw.detach().cpu(),
                sampling_rate=CONFIG["sampling_rate"],
                dilate_size=21,
            ).to(device)

            outputs = model(ppg_raw, return_features=True)
            modes = reconstruct_modes(model, outputs)

            ppg_np = ppg_raw.detach().cpu().numpy()
            gt_np = ecg_raw.detach().cpu().numpy()
            qrs_mask_np = qrs_mask.detach().cpu().numpy()

            full_np = modes["full"].detach().cpu().numpy()
            qrs_only_np = modes["qrs_only"].detach().cpu().numpy()
            non_only_np = modes["non_only"].detach().cpu().numpy()

            qrs_pred_np = outputs["ecg_qrs_pred"].detach().cpu().numpy()
            non_pred_np = outputs["ecg_non_qrs_pred"].detach().cpu().numpy()

            z_qrs_np = outputs["z_qrs"].detach().cpu().numpy()
            z_non_np = outputs["z_non_qrs"].detach().cpu().numpy()

            for i in range(gt_np.shape[0]):
                ppg_i = ppg_np[i, 0]
                gt_i = gt_np[i, 0]
                mask_i = qrs_mask_np[i, 0]

                full_i = full_np[i, 0]
                qrs_only_i = qrs_only_np[i, 0]
                non_only_i = non_only_np[i, 0]

                qrs_pred_i = qrs_pred_np[i, 0]
                non_pred_i = non_pred_np[i, 0]

                z_qrs_i = z_qrs_np[i]
                z_non_i = z_non_np[i]

                all_full.append(full_i.copy())
                all_qrs_only.append(qrs_only_i.copy())
                all_non_only.append(non_only_i.copy())
                all_gt.append(gt_i.copy())

                spec = specialization_metrics_per_sample(
                    qrs_pred=qrs_pred_i,
                    non_pred=non_pred_i,
                    ecg_gt=gt_i,
                    qrs_mask=mask_i,
                    latent_qrs=z_qrs_i,
                    latent_non=z_non_i,
                    eps=CONFIG["eps"],
                )
                all_specs.append(spec)

                if plotted < num_plot_samples:
                    sample_id = plotted + 1
                    save_path = os.path.join(plot_dir, f"sample_{sample_id}_branch_ablation.png")
                    plot_branch_ablation(
                        ppg=ppg_i,
                        ecg_gt=gt_i,
                        ecg_full=full_i,
                        ecg_qrs_only=qrs_only_i,
                        ecg_non_only=non_only_i,
                        fs=fs,
                        title=f"Sample {sample_id}",
                        save_path=save_path,
                        show=show_plots,
                    )
                    plotted += 1

            pbar.set_postfix({
                "FullP": f"{compute_average_metrics(all_full[-len(gt_np):], all_gt[-len(gt_np):])['pearson_mean']:.3f}",
                "QrsP": f"{compute_average_metrics(all_qrs_only[-len(gt_np):], all_gt[-len(gt_np):])['pearson_mean']:.3f}",
                "NonP": f"{compute_average_metrics(all_non_only[-len(gt_np):], all_gt[-len(gt_np):])['pearson_mean']:.3f}",
            })

    full_metrics = compute_average_metrics(all_full, all_gt)
    qrs_only_metrics = compute_average_metrics(all_qrs_only, all_gt)
    non_only_metrics = compute_average_metrics(all_non_only, all_gt)
    spec_summary = summarize_specialization(all_specs)

    result = {
        "full_reconstruction": full_metrics,
        "qrs_only_reconstruction": qrs_only_metrics,
        "non_only_reconstruction": non_only_metrics,
        "specialization_summary": spec_summary,
    }

    print("\n===== RECONSTRUCTION METRICS =====")
    print("[Full]")
    print(f"Pearson: {full_metrics['pearson_mean']:.6f} ± {full_metrics['pearson_std']:.6f}")
    print(f"RMSE   : {full_metrics['rmse_mean']:.6f} ± {full_metrics['rmse_std']:.6f}")

    print("\n[QRS-only]")
    print(f"Pearson: {qrs_only_metrics['pearson_mean']:.6f} ± {qrs_only_metrics['pearson_std']:.6f}")
    print(f"RMSE   : {qrs_only_metrics['rmse_mean']:.6f} ± {qrs_only_metrics['rmse_std']:.6f}")

    print("\n[Non-QRS-only]")
    print(f"Pearson: {non_only_metrics['pearson_mean']:.6f} ± {non_only_metrics['pearson_std']:.6f}")
    print(f"RMSE   : {non_only_metrics['rmse_mean']:.6f} ± {non_only_metrics['rmse_std']:.6f}")

    print("\n===== SPECIALIZATION / COLLAPSE / REDUNDANCY =====")
    for k, v in spec_summary.items():
        if k == "flags":
            continue
        print(f"{k:20s}: mean={v['mean']:.6f}, std={v['std']:.6f}")

    print("\nFlags:")
    for k, v in spec_summary["flags"].items():
        print(f"  {k}: {v}")

    save_json = os.path.join(CONFIG["save_dir"], "branch_specialization_summary.json")
    with open(save_json, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nSaved plots in : {plot_dir}")
    print(f"Saved summary  : {save_json}")

    return result


if __name__ == "__main__":
    result = test_branch_specialization(
        data_path=CONFIG["data_path"],
        ckpt_path=CONFIG["ckpt_path"],
        batch_size=CONFIG["batch_size"],
        num_workers=CONFIG["num_workers"],
        num_plot_samples=CONFIG["num_plot_samples"],
        show_plots=False,
    )